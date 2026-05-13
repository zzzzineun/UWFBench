import os
import base64
import traceback

from PIL import Image
from io import BytesIO
from openai import OpenAI
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm

from prompts import SYSTEM_PROMPT, USER_PROMPTS

# ==========================================
# Config
# ==========================================
OPENAI_API_KEY = "sk-xxxxxx"
GEMINI_API_KEY = "xxxxxxxxx"

client_openai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Model List for inference
MODELS_TO_RUN = [
    "o3",
    "gpt-4o",
    "gpt-4.1",
    "gemini-2.5-pro",
    "gemini-2.5-flash"
]

# Data paths
CSV_FILE   = "./data_backxxxxxx.csv" # GT file with "relative_path" and "is_completed" columns
IMAGE_BASE = "./image/Unified_UWF_Dataset" # Base dir for images, to be joined with "relative_path" from CSV
MAX_COUNT  = ""   # "" = all rows


# ==========================================
# Image preprocessing
# ==========================================
def process_image(image_path, max_size=2048):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    width, height = img.size
    if max(width, height) > max_size:
        if width > height:
            new_width, new_height = max_size, int(max_size * height / width)
        else:
            new_height, new_width = max_size, int(max_size * width / height)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return img, buffered.getvalue()


# ==========================================
# API call helpers
# ==========================================
def _call_openai(model_id, base64_img, system_prompt, user_prompt):
    res = client_openai.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            ]},
        ],
    )
    return res.choices[0].message.content


def _call_gemini(model_id, pil_img, system_prompt, user_prompt):
    gemini = genai.GenerativeModel(model_id)
    res = gemini.generate_content([system_prompt, user_prompt, pil_img])
    return res.text


# ==========================================
# Per-model inference + save
# ==========================================
def run_model(samples, base_img_dir, output_prefix, model_id, system_prompt, user_prompt, prompt_key):
    print(f"\n{'='*60}\n[{model_id}] inference start\n{'='*60}")
    is_gemini = model_id.startswith("gemini")

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc=model_id, unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            pil_img, img_bytes = process_image(full_path)
            if is_gemini:
                result = _call_gemini(model_id, pil_img, system_prompt, user_prompt)
            else:
                base64_img = base64.b64encode(img_bytes).decode("utf-8")
                result = _call_openai(model_id, base64_img, system_prompt, user_prompt)
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_prefix)


# ==========================================
# CSV save helper
# ==========================================
def _save_csv(df, col, output_prefix):
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    safe_model = col.replace("/", "-")
    out_path = f"{output_prefix}_{safe_model}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {os.path.abspath(out_path)}")


# ==========================================
# Main orchestration
# ==========================================
def run_all(csv_path, base_img_dir, max_count="", models=None, prompts=None):
    df = pd.read_csv(csv_path)
    samples = df[df["is_completed"].astype(str).str.lower() == "true"].copy()
    if max_count != "":
        samples = samples.head(int(max_count))

    target_models = models or MODELS_TO_RUN
    target_prompts = prompts or USER_PROMPTS  # {prompt_key: {"output_prefix": str, "text": str}}

    total = len(target_models) * len(target_prompts)
    print(f"total : {len(target_models)} models x {len(target_prompts)} prompts = {total} times\n")

    for prompt_key, prompt_cfg in target_prompts.items():
        output_prefix = prompt_cfg["output_prefix"]
        user_prompt   = prompt_cfg["text"]
        print(f"\n{'#'*60}\n[PROMPT] {prompt_key}  →  {output_prefix}_{{model}}.csv\n{'#'*60}")
        for model_id in target_models:
            try:
                run_model(samples, base_img_dir, output_prefix, model_id, SYSTEM_PROMPT, user_prompt, prompt_key)
            except Exception as e:
                print(f"[ERROR] {model_id} / {prompt_key} 전체 실패: {e}")
                traceback.print_exc()


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    run_all(
        csv_path=CSV_FILE,
        base_img_dir=IMAGE_BASE,
        max_count=MAX_COUNT,
        # models=["o3"],                                     
        # prompts={"report_v1": USER_PROMPTS["report_v1"]}, 
    )
