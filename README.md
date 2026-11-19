# UWFBench

Code associated with the study **"Benchmarking Multimodal Large Language Models for Automated Clinical Reporting in Ultra-Widefield Fundus Photography: A Comparative Evaluation Study"** (BMJ Open Ophthalmology, under review; also presented at APOIS 2026).

This project evaluates multimodal large language models for generating clinical reports from ultra-widefield (UWF) fundus images.

## Dataset

**[Fundus Report Dataset (Hugging Face)](https://huggingface.co/datasets/zzzzineun/fundus-report-dataset)**

A curated benchmark of **422 UWF fundus cases** from **DeepDRiD (203)** and **OUWFD (219)**. Each record includes an image filename/path, clinical keyword labels, source dataset, and a corresponding free-text clinical report for evaluating automated report generation.

**Note:** The Hugging Face dataset provides report annotations and image references, **not the original images**. Obtain images from the cited source datasets and follow their respective licenses.

## Repository contents

- **`inference/api_all.py`** — Report generation with API-based models.
- **`inference/hf_inference_all.py`** — Report generation with locally run vision-language models.
- **`inference/prompts.py`** — Ophthalmology-specific reporting prompts.
- **`eval/evaluate_score.py`** — Text-based evaluation metrics, including BLEU, ROUGE-L, BERTScore, METEOR, and MedCon.
- **`eval/llm_judge.py`** and **`eval/rubric_prompt.py`** — Rubric-based assessment of clinical validity, report coverage, hallucinations, safety, and diagnoses.

## Notes

These are research scripts rather than a packaged application. Configure the model credentials, image/data paths, and output locations in the relevant scripts before running them. Model credentials and original images are not included in this repository.
