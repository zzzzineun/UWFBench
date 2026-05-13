import os
import re
import json

from openai import OpenAI
import google.generativeai as genai
import pandas as pd
from tqdm import tqdm

from rubric_prompt import (
    clinical_validity_prompt,
    clinical_alignment_and_coverage_prompt,
    hallucination_prompt,
    safety_prompt,
    diagnosis_prompt,
)

OPENAI_API_KEY = "sk-xxxxx"
GEMINI_API_KEY = "xxxxxxx"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are an expert ophthalmologist evaluating the quality of a candidate fundus report.
Assess the <CANDIDATE_REPORT> against the <TARGET_REPORT> according to the given rubric."""

ALL_RUBRICS = {
    "clinical_validity": clinical_validity_prompt,
    "clinical_alignment_and_coverage": clinical_alignment_and_coverage_prompt,
    "hallucination": hallucination_prompt,
    "safety": safety_prompt,
    "diagnosis": diagnosis_prompt,
}
DIAGNOSIS_RUBRICS = {"diagnosis"}

# ==========================================
# Config
# ==========================================
CSV_FILES = [
    "./inference_results/inference_xxxx.csv",
]
GPT_MODEL = "o3-mini"
GEMINI_MODEL = "gemini-3.1-flash"
MAX_COUNT = ""   

# test rubrics
RUBRICS_TO_RUN = [
    "clinical_validity",
    "clinical_alignment_and_coverage",
    "hallucination",
    "safety",
    "diagnosis",
]
# ==========================================


def parse_result(raw: str) -> dict:
    text = re.sub(r"```json\s*|\s*```", "", raw).strip()
    return json.loads(text)
def parse_score_rubric(parsed: dict) -> dict:
    parsed["score"] = int(parsed["score"])
    return parsed
def parse_diagnosis_rubric(parsed: dict) -> dict:
    required = {"primary_diagnosis_target", "primary_diagnosis_candidate", "diagnostic_match"}
    missing = required - parsed.keys()
    if missing:
        raise ValueError(f"missing keys: {missing}")
    parsed["diagnostic_match"] = int(parsed["diagnostic_match"])
    return parsed


def call_gpt(target_report: str, candidate_report: str, rubric_prompt: str) -> str:
    user_prompt = rubric_prompt.format(
        target_report=target_report,
        candidate_report=candidate_report,
    )
    response = client_openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_gemini(target_report: str, candidate_report: str, rubric_prompt: str) -> str:
    user_prompt = rubric_prompt.format(
        target_report=target_report,
        candidate_report=candidate_report,
    )
    gemini = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    response = gemini.generate_content(user_prompt)
    return response.text


def build_summary(rubric_name: str, entries: list, skipped: int) -> dict:
    if rubric_name in DIAGNOSIS_RUBRICS:
        matches = [e["diagnostic_match"] for e in entries]
        total = len(matches)
        matched = sum(matches)
        accuracy = round(matched / total, 4) if total > 0 else None
        return {
            "accuracy": accuracy,
            "matched_count": matched,
            "total_count": total,
            "skipped_count": skipped,
        }
    else:
        scores = [e["score"] for e in entries]
        average_score = round(sum(scores) / len(scores), 4) if scores else None
        return {
            "average_score": average_score,
            "skipped_count": skipped,
        }


def run_judge(csv_path: str):
    use_gpt = GPT_MODEL != ""
    use_gemini = GEMINI_MODEL != ""

    if not use_gpt and not use_gemini:
        print("NO JUDGE MODEL SPECIFIED. Set GPT_MODEL and/or GEMINI_MODEL to run the judge.")
        return

    df = pd.read_csv(csv_path)
    candidate_col = df.columns[-1]
    samples = df.copy()
    if MAX_COUNT != "":
        samples = samples.head(int(MAX_COUNT))
    df = samples.copy()

    json_results: dict[tuple, dict] = {}
    for rubric_name in RUBRICS_TO_RUN:
        if rubric_name not in ALL_RUBRICS:
            continue
        if use_gpt:
            json_results[("gpt", rubric_name)] = {"entries": [], "skipped": 0}
        if use_gemini:
            json_results[("gemini", rubric_name)] = {"entries": [], "skipped": 0}


    for rubric_name in RUBRICS_TO_RUN:
        if rubric_name not in ALL_RUBRICS:
            continue
        if use_gpt and f"judge_result_{rubric_name}_gpt" not in df.columns:
            df[f"judge_result_{rubric_name}_gpt"] = None
        if use_gemini and f"judge_result_{rubric_name}_gemini" not in df.columns:
            df[f"judge_result_{rubric_name}_gemini"] = None

    for idx, row in tqdm(list(samples.iterrows()), desc="Judging", unit="row"):
        target_report = str(row.get("report", ""))
        candidate_report = str(row[candidate_col])
        filename = str(row.get("filename", ""))

        for rubric_name in RUBRICS_TO_RUN:
            if rubric_name not in ALL_RUBRICS:
                tqdm.write(f"[WARNING] Unknown rubric '{rubric_name}', skipping.")
                continue

            prompt = ALL_RUBRICS[rubric_name]
            is_diagnosis = rubric_name in DIAGNOSIS_RUBRICS

            for model_tag, caller in [("gpt", call_gpt), ("gemini", call_gemini)]:
                if model_tag == "gpt" and not use_gpt:
                    continue
                if model_tag == "gemini" and not use_gemini:
                    continue

                col = f"judge_result_{rubric_name}_{model_tag}"
                try:
                    raw = caller(target_report, candidate_report, prompt)
                    parsed = parse_result(raw)
                    if is_diagnosis:
                        parsed = parse_diagnosis_rubric(parsed)
                        score_str = f"diagnostic_match={parsed['diagnostic_match']}"
                    else:
                        parsed = parse_score_rubric(parsed)
                        score_str = f"score={parsed['score']}"
                    tqdm.write(f"[{idx}] {model_tag.upper()} / {rubric_name}: {score_str}")
                    df.at[idx, col] = json.dumps(parsed, ensure_ascii=False)
                    json_results[(model_tag, rubric_name)]["entries"].append(
                        {"row_index": idx, "filename": filename, **parsed}
                    )
                except Exception as e:
                    tqdm.write(f"[{idx}] {model_tag.upper()} / {rubric_name} error: {e}")
                    df.at[idx, col] = f"Error: {e}"
                    json_results[(model_tag, rubric_name)]["skipped"] += 1

    # save results
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    subfolder = base_name.split("inference_", 1)[-1] if "inference_" in base_name else base_name
    out_dir = f"./score/{subfolder}"
    os.makedirs(out_dir, exist_ok=True)

    # save CSV with judge results
    csv_out = f"{out_dir}/{base_name}_eval.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\nCSV saved: {os.path.abspath(csv_out)}")

    # save JSON results for each rubric and model
    gpt_key = GPT_MODEL.replace("/", "-") if use_gpt else ""
    gemini_key = GEMINI_MODEL.replace("/", "-") if use_gemini else ""

    for (model_tag, rubric_name), data in json_results.items():
        model_key = gpt_key if model_tag == "gpt" else gemini_key
        entries = data["entries"]
        summary = build_summary(rubric_name, entries, data["skipped"])
        output = {**summary, "results": entries}
        json_path = f"{out_dir}/{model_key}_{rubric_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f"JSON saved: {os.path.abspath(json_path)}")


for csv_file in CSV_FILES:
    print(f"\n{'='*50}")
    print(f"Processing: {csv_file}")
    print(f"{'='*50}")
    run_judge(csv_file)
