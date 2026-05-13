#!/usr/bin/env python3
"""
Evaluate inference results: recall, BLEU, BERTScore, ROUGE-L, MedCon
"""

# ===================== Configuration =====================
CSV_FILE   = "~"    # path to CSV file with "report" and prediction columns.
OUTPUT_FILE = None  # if json output: "eval/results_gpt-4.1.json"
# ==========================================================

import json
import warnings
from pathlib import Path

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
import sacrebleu
import spacy

warnings.filterwarnings("ignore")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


def load_csv(path: str) -> tuple[list[str], list[str], str]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["report"])
    last_col = df.columns[-1]
    df = df.dropna(subset=[last_col])
    references = df["report"].tolist()
    hypotheses = df[last_col].tolist()
    return references, hypotheses, last_col


def compute_token_recall(references: list[str], hypotheses: list[str]) -> float:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = set(word_tokenize(ref.lower()))
        hyp_tokens = set(word_tokenize(hyp.lower()))
        if not ref_tokens:
            continue
        overlap = ref_tokens & hyp_tokens
        scores.append(len(overlap) / len(ref_tokens))
    return sum(scores) / len(scores) if scores else 0.0


def compute_bleu(references: list[str], hypotheses: list[str]) -> float:
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score / 100.0


def compute_bertscore(references: list[str], hypotheses: list[str]) -> dict:
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }


def compute_rouge_l(references: list[str], hypotheses: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_meteor(references: list[str], hypotheses: list[str]) -> float:
    scores = [
        meteor_score([word_tokenize(ref.lower())], word_tokenize(hyp.lower()))
        for ref, hyp in zip(references, hypotheses)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def compute_medcon(references: list[str], hypotheses: list[str]) -> dict:
    """Medical concept F1 using scispacy en_core_sci_sm."""
    nlp = spacy.load("en_core_sci_sm")

    precision_scores, recall_scores, f1_scores = [], [], []
    for ref, hyp in zip(references, hypotheses):
        ref_concepts = {e.text.lower() for e in nlp(ref).ents}
        hyp_concepts = {e.text.lower() for e in nlp(hyp).ents}

        if not ref_concepts and not hyp_concepts:
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            f1_scores.append(1.0)
            continue

        overlap = ref_concepts & hyp_concepts
        prec = len(overlap) / len(hyp_concepts) if hyp_concepts else 0.0
        rec = len(overlap) / len(ref_concepts) if ref_concepts else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        precision_scores.append(prec)
        recall_scores.append(rec)
        f1_scores.append(f1)

    return {
        "precision": sum(precision_scores) / len(precision_scores),
        "recall": sum(recall_scores) / len(recall_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }


def evaluate(csv_path: str, output_path: str | None = None) -> dict:
    print(f"Loading: {csv_path}")
    references, hypotheses, pred_col = load_csv(csv_path)
    print(f"  reference column : 'report'")
    print(f"  prediction column: '{pred_col}'")
    print(f"  samples          : {len(references)}\n")

    print("Computing Token Recall...")
    recall = compute_token_recall(references, hypotheses)

    print("Computing BLEU...")
    bleu = compute_bleu(references, hypotheses)

    print("Computing BERTScore...")
    bertscore = compute_bertscore(references, hypotheses)

    print("Computing ROUGE-L...")
    rouge_l = compute_rouge_l(references, hypotheses)

    print("Computing METEOR...")
    meteor = compute_meteor(references, hypotheses)

    print("Computing MedCon...")
    medcon = compute_medcon(references, hypotheses)

    results = {
        "file": csv_path,
        "prediction_column": pred_col,
        "n_samples": len(references),
        "recall": round(recall, 4),
        "bleu": round(bleu, 4),
        "bertscore": {k: round(v, 4) for k, v in bertscore.items()},
        "rouge_l": round(rouge_l, 4),
        "meteor": round(meteor, 4),
        "medcon": {k: round(v, 4) for k, v in medcon.items()},
    }

    print("\n===== Results =====")
    print(f"Token Recall : {results['recall']:.4f}")
    print(f"BLEU         : {results['bleu']:.4f}")
    print(f"BERTScore P/R/F1: {bertscore['precision']:.4f} / {bertscore['recall']:.4f} / {bertscore['f1']:.4f}")
    print(f"ROUGE-L      : {results['rouge_l']:.4f}")
    print(f"METEOR       : {results['meteor']:.4f}")
    print(f"MedCon P/R/F1: {medcon['precision']:.4f} / {medcon['recall']:.4f} / {medcon['f1']:.4f}")

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved to: {output_path}")

    return results


if __name__ == "__main__":
    evaluate(CSV_FILE, OUTPUT_FILE)
