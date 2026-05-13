import os
import gc
import base64
import warnings
import traceback

import torch
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# ==========================================
# Config
# ==========================================
HF_TOKEN = "hf_xxxxxxx"  # Hugging Face API token, required for some models

# Model List for inference
MODELS_TO_RUN = [
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "google/medgemma-27b-it",
    "deepseek-ai/deepseek-vl2",
    "OpenGVLab/InternVL3-78B-Instruct",
]

# Data paths
CSV_FILE   = "./data_backxxxxxx.csv" # GT file with "relative_path" and "is_completed" columns
IMAGE_BASE = "./image/Unified_UWF_Dataset" # Base dir for images, to be joined with "relative_path" from CSV
MAX_COUNT  = ""   # "" = all rows

# ==========================================
# Prompts — edit in prompts.py
# ==========================================
from prompts import SYSTEM_PROMPT, USER_PROMPTS


# ==========================================
# Image preprocessing (same as api.py)
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
# InternVL3 image preprocessing helpers
# ==========================================
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    aspect_ratio = orig_w / orig_h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    tar_ar = _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_w, orig_h, image_size)
    target_w, target_h = image_size * tar_ar[0], image_size * tar_ar[1]
    blocks = tar_ar[0] * tar_ar[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    cols = target_w // image_size
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def _internvl_load_pixel_values(image_path, input_size=448, max_num=12):
    image = Image.open(image_path).convert("RGB")
    transform = _internvl_build_transform(input_size)
    tiles = _internvl_dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


# ==========================================
# GPU memory cleanup helper
# ==========================================
def _free_gpu(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# Model-specific inference functions
# Each function:
#   1. Loads the model
#   2. Runs all samples
#   3. Saves CSV
#   4. Unloads the model
# ==========================================

def run_qwen25_vl(samples, base_img_dir, output_csv_prefix, system_prompt, user_prompt, prompt_key):
    model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"\n{'='*60}\n[Qwen2.5-VL] Loading {model_id}\n{'='*60}")
    from transformers import AutoModelForImageTextToText, AutoProcessor
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError("Install qwen-vl-utils: pip install qwen-vl-utils")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    model.eval()

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc="Qwen2.5-VL", unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            pil_img, _ = process_image(full_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            result = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_csv_prefix, prompt_key)
    _free_gpu(model, processor)
    return df


def run_llama32_vision(samples, base_img_dir, output_csv_prefix, system_prompt, user_prompt, prompt_key):
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    print(f"\n{'='*60}\n[Llama-3.2-Vision] Loading {model_id}\n{'='*60}")

    from transformers import MllamaForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc="Llama-3.2-Vision", unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            pil_img, _ = process_image(full_path)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(pil_img, input_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=512)
            result = processor.decode(
                output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_csv_prefix, prompt_key)
    _free_gpu(model, processor)
    return df


def _internvl_split_model(num_layers=80):
    import math
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA GPUs found"
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers_per_gpu = [num_layers_per_gpu] * world_size
    layers_per_gpu[0] = math.ceil(num_layers_per_gpu * 0.5)

    device_map = {}
    layer_cnt = 0
    for gpu_id, n in enumerate(layers_per_gpu):
        for _ in range(n):
            if layer_cnt >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_cnt}"] = gpu_id
            layer_cnt += 1

    for key in [
        "vision_model", "mlp1",
        "language_model.model.tok_embeddings",
        "language_model.model.embed_tokens",
        "language_model.model.norm",
        "language_model.output",
        "language_model.lm_head",
        f"language_model.model.layers.{num_layers - 1}",
    ]:
        device_map[key] = 0
    return device_map


def run_internvl3(samples, base_img_dir, output_csv_prefix, system_prompt, user_prompt, prompt_key):
    model_id = "OpenGVLab/InternVL3-78B-Instruct"
    print(f"\n{'='*60}\n[InternVL3] Loading {model_id}\n{'='*60}")
    import transformers.modeling_utils as _mu
    _orig_get_total_byte_count = _mu.get_total_byte_count
    def _patched_get_total_byte_count(model, *args, **kwargs):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig_get_total_byte_count(model, *args, **kwargs)
    _mu.get_total_byte_count = _patched_get_total_byte_count

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=False
    )
    device_map = _internvl_split_model(num_layers=80)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    generation_config = dict(max_new_tokens=512, do_sample=False)
    question = f"<image>\n{user_prompt}"

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc="InternVL3", unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            pixel_values = _internvl_load_pixel_values(full_path, max_num=12)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            result = model.chat(tokenizer, pixel_values, question, generation_config)
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_csv_prefix, prompt_key)
    _free_gpu(model, tokenizer)
    return df


def run_medgemma(samples, base_img_dir, output_csv_prefix, system_prompt, user_prompt, prompt_key):
    """google/medgemma-27b-it — image-text-to-text (Gemma 3 27B + SigLIP vision encoder)"""
    model_id = "google/medgemma-27b-it"
    print(f"\n{'='*60}\n[MedGemma-27B] Loading {model_id}\n{'='*60}")

    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    model.eval()

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc="MedGemma-27B", unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            pil_img, _ = process_image(full_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_prompt},
                ]},
            ]
            input_text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=input_text, images=pil_img, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            result = processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_csv_prefix, prompt_key)
    _free_gpu(model, processor)
    return df


def run_deepseek_vl2(samples, base_img_dir, output_csv_prefix, system_prompt, user_prompt, prompt_key):
    """
    Must run with deepseek_vl2 package installed, which contains the custom model code for DeepSeek-VL2.
     pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
    """
    model_id = "deepseek-ai/deepseek-vl2"
    print(f"\n{'='*60}\n[DeepSeek-VL2] Loading {model_id}\n{'='*60}")

    try:
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        from deepseek_vl2.utils.io import load_pil_images
    except ImportError:
        raise ImportError(
            "deepseek_vl2 package needed:\n"
            "pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git"
        )

    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = DeepseekVLV2ForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    col = model_id
    df = samples.copy()
    df[col] = None

    for idx, row in tqdm(list(df.iterrows()), desc="DeepSeek-VL2", unit="img"):
        full_path = os.path.join(base_img_dir, row["relative_path"])
        if not os.path.exists(full_path):
            df.at[idx, col] = "Error: file not found"
            continue
        try:
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{user_prompt}",
                    "images": [full_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=system_prompt,
            ).to(model.device)

            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            lm = (model.language_model if hasattr(model, "language_model")
                  else model.model if hasattr(model, "model")
                  else model)
            with torch.no_grad():
                outputs = lm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True,
                )
            result = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            df.at[idx, col] = result
            tqdm.write(f"  [{idx}] {result[:80]}...")
        except Exception as e:
            df.at[idx, col] = f"Error: {e}"
            tqdm.write(f"  [{idx}] Error: {e}")

    _save_csv(df, col, output_csv_prefix, prompt_key)
    _free_gpu(model, processor)
    return df


# ==========================================
# CSV save helper
# ==========================================
def _save_csv(df, col, output_csv_prefix, prompt_key):
    os.makedirs(os.path.dirname(output_csv_prefix) or ".", exist_ok=True)
    safe_model = col.replace("/", "-")
    out_path = f"{output_csv_prefix}_{safe_model}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  saved: {os.path.abspath(out_path)}")


# ==========================================
# Main orchestration
# ==========================================
MODEL_RUNNERS = {
    "Qwen/Qwen2.5-VL-72B-Instruct":              run_qwen25_vl,
    "meta-llama/Llama-3.2-90B-Vision-Instruct":  run_llama32_vision,
    "OpenGVLab/InternVL3-78B-Instruct":           run_internvl3,
    "google/medgemma-27b-it":                      run_medgemma,
    "deepseek-ai/deepseek-vl2":                   run_deepseek_vl2,
}


def run_all(csv_path, base_img_dir, max_count="", models=None, prompts=None):
    df = pd.read_csv(csv_path)
    samples = df[df["is_completed"].astype(str).str.lower() == "true"].copy()
    if max_count != "":
        samples = samples.head(int(max_count))

    target_models = models or MODELS_TO_RUN   # list of model_id
    target_prompts = prompts or USER_PROMPTS  # dict {prompt_key: {"output_prefix": str, "text": str}}

    total = len(target_models) * len(target_prompts)
    print(f"total   : {len(target_models)} models x {len(target_prompts)} prompts = {total} times\n")

    for prompt_key, prompt_cfg in target_prompts.items():
        output_prefix = prompt_cfg["output_prefix"]
        user_prompt   = prompt_cfg["text"]
        print(f"\n{'#'*60}\n[PROMPT] {prompt_key}  →  {output_prefix}_{{model}}.csv\n{'#'*60}")
        for model_id in target_models:
            runner = MODEL_RUNNERS.get(model_id)
            if runner is None:
                print(f"[SKIP] not found: {model_id}")
                continue
            try:
                runner(samples, base_img_dir, output_prefix, SYSTEM_PROMPT, user_prompt, prompt_key)
            except Exception as e:
                print(f"[ERROR] {model_id} / {prompt_key} all failed: {e}")
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


# ==========================================
# Entry point
# ==========================================
if __name__ == "__main__":
    run_all(
        csv_path=CSV_FILE,
        base_img_dir=IMAGE_BASE,
        max_count=MAX_COUNT,
        # models=["Qwen/Qwen2.5-VL-72B-Instruct"],          
        # prompts={"report_v1": USER_PROMPTS["report_v1"]},
    )
