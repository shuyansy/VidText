# benchmark.py — Unified evaluator for 8 video–LLM tasks with Qwen2.5-VL 7B
# -----------------------------------------------------------------------------
# Usage example:
#     python benchmark.py \
#         --model_path /data/yangzhifei/project/Qwen2.5-VL-7B-Instruct \
#         --gpu 0 \
#         --output_dir /data/yangzhifei/project/Result/qwen25_7b \
#         --tasks all               # or comma‑sep subset: OCR,HR,TR,TempG,…
# -----------------------------------------------------------------------------

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# --------------------------- 1. GLOBAL HELPERS --------------------------------

def load_model(model_path: str, gpu: int):
    """Load Qwen2.5‑VL 7B once and share across tasks."""
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor, device


def uniform_sample_indices(total_frames: int, fps: float, max_frames: int):
    duration = total_frames / fps if fps > 0 else 0
    if duration > max_frames:
        return np.linspace(0, total_frames - 1, num=max_frames, dtype=int)
    step = max(1, int(round(fps)))
    return np.arange(0, total_frames, step, dtype=int)


def read_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    cap.release()
    return fps, frames, frames / fps


def build_messages(video_path: str, prompt: str, fps: float):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360 * 420,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_text(model, processor, device, messages, max_new_tokens=128):
    from qwen_vl_utils import process_vision_info  # local helper

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trim_ids = [o[len(inp):] for inp, o in zip(inputs.input_ids, out_ids)]
    return processor.batch_decode(trim_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

# --------------------------- 2. TASK HELPERS ----------------------------------


def mc_prompt(question: str, options: List[Dict[str, str]], tail: str = "Only output the correct option.") -> str:
    body = "\n".join([f"Option {o['option_id']}: {o['text']}" for o in options])
    return (
        "Watch the video and answer the following multiple-choice question based on its content.\n\n"
        f"Question: {question}\n\nOptions:\n{body}\n\n{tail}"
    )


def parse_single_option(text: str) -> str:
    text = (
        text.replace("Answer", "")
        .replace("Correct Answer", "")
        .replace("Option", "")
        .replace(":", "")
        .strip()
    )
    return text[0] if text and text[0] in "ABCD" else "INVALID"


def parse_multi_options(text: str) -> List[str]:
    # expects like "Correct Answers: A, B, C"
    letters = re.findall(r"[A-D]", text.upper())
    return sorted(set(letters))

# --------------------------- 3. RUNNERS ---------------------------------------


def run_holistic_ocr(cfg, mp):
    ann = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    out_list = []
    prompt = (
        "Recognize all visual texts in the video. If the text is not in English, do not provide an English translation. "
        "Do not include any descriptions, narrative, or context. Output only the extracted text lines, each on a new line."
    )
    for item in ann:
        vid = item["video_id"]
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        msg = build_messages(vfile, prompt, eff_fps)
        answer = generate_text(*mp, messages=msg, max_new_tokens=128)
        out_list.append({**item, "model_answer": answer})
    out_path = f"{cfg['out_dir']}/hOCR_results.json"
    json.dump(out_list, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path


def run_holistic_reasoning(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res, correct = [], 0
    for d in data:
        vid = d["video_id"]
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        # options dict → list
        opts_dict = d["option"]
        options = [{"option_id": k, "text": v} for k, v in opts_dict.items()]
        prompt = (
            "Watch the video carefully and select the correct three answers.\n\n"
            f"Question: {d['question']}\n\nOptions:\n"
            + "\n".join([f"{k}: {v}" for k, v in sorted(opts_dict.items())])
            + "\n\nPlease output your answer in the format: `Correct Answers: A, B, C`\nDo not provide any additional explanations."
        )
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        pred_ids = parse_multi_options(ans_text)
        gt_ids = parse_multi_options(d.get("answer", ""))
        is_ok = set(pred_ids) == set(gt_ids)
        correct += is_ok
        res.append({**d, "model_answer": ans_text, "parsed_answer_ids": pred_ids, "is_correct": is_ok})
    acc = correct / len(res)
    out_path = f"{cfg['out_dir']}/holistic_reasoning_merged.json"
    json.dump({"summary": {"accuracy": acc}, "details": res}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path


def run_Local_reasoning(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res, correct = [], 0
    for item in data:
        vid = item.get("video_id") or item.get("video_name")
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        raw_opts = item["options"]
        options = [{"option_id": k, "text": v} for k, v in raw_opts.items()] if isinstance(raw_opts, dict) else raw_opts
        prompt = mc_prompt(item["question"], options)
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        pred = parse_single_option(ans_text)
        gt = item["answer"]["option_id"] if isinstance(item["answer"], dict) else item["answer"]
        ok = pred == gt
        correct += ok
        res.append({**item, "model_answer": ans_text, "parsed_answer_id": pred, "is_correct": ok})
    acc = correct / len(res)
    out_path = f"{cfg['out_dir']}/LocalReasoning_results.json"
    json.dump({"summary": {"accuracy": acc}, "details": res}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path

# -------- Temporal‑Causal Reasoning --------------------------------------------

def run_temporal_causal_reasoning(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res, correct = [], 0
    for item in data:
        vid = item.get("video_id") or item.get("video_name")
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        raw_opts = item["options"]
        options = (
            [{"option_id": k, "text": v} for k, v in raw_opts.items()]
            if isinstance(raw_opts, dict)
            else raw_opts
        )
        prompt = mc_prompt(item["question"], options)
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        pred = parse_single_option(ans_text)
        gt = item["answer"]["option_id"] if isinstance(item["answer"], dict) else item["answer"]
        ok = pred == gt
        correct += ok
        res.append({**item, "model_answer": ans_text, "parsed_answer_id": pred, "is_correct": ok})
    acc = correct / len(res)
    out_path = f"{cfg['out_dir']}/TCR_results_merged.json"
    json.dump({"summary": {"accuracy": acc}, "details": res}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path

# -------- Spatial Reasoning ----------------------------------------------------

def run_spatial_reasoning(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res, correct = [], 0
    for item in data:
        vid = item.get("video_id") or item.get("video_name")
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        raw_opts = item["options"]
        options = (
            [{"option_id": k, "text": v} for k, v in raw_opts.items()]
            if isinstance(raw_opts, dict)
            else raw_opts
        )
        prompt = mc_prompt(item["question"], options)
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        pred = parse_single_option(ans_text)
        gt = item["answer"]["option_id"] if isinstance(item["answer"], dict) else item["answer"]
        ok = pred == gt
        correct += ok
        res.append({**item, "model_answer": ans_text, "parsed_answer_id": pred, "is_correct": ok})
    acc = correct / len(res) if res else 0.0
    out_path = f"{cfg['out_dir']}/SR_results_merged.json"
    json.dump({"summary": {"accuracy": acc}, "details": res}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path

# TODO: implement the remaining 3 run_<task>() similarly (TemporalGrounding, TextGrounding, spatialGrounding).
# -------- Text Grounding ------------------------------------------------------

def run_TextLocalization(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res = []
    for item in data:
        vid = item["video_id"]
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 196 else 196 / dur
        prompt = (
            "Watch the video and answer the following question based on its content."
            "Please provide the time interval (in seconds, precise to 0.1s) during which the text appears in the video. "
            "Output your answer in JSON format with keys 'start' and 'end'. For example: {\"start\": 0.0, \"end\": 30.0}. "
            "Do not include any extra commentary."
        )
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        res.append({**item, "model_answer": ans_text})
    out_path = f"{cfg['out_dir']}/TextLocalization_results.json"
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path


# -------- Temporal Grounding --------------------------------------------------

def run_LocalOCR(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res = []
    for item in data:
        vid = item["video_id"]
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        if not os.path.exists(vfile):
            print(f"[TempG] missing video {vfile}")
            continue
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        prompt = (
            "Watch the video and answer the following question based on its content."
            f"Question: {item['question']}"
            "Please output only the texts that appear in the specified time interval as a JSON array of strings, "
            "with each element representing one piece of text. Do not include any additional description or translation."
        )
        msg = build_messages(vfile, prompt, eff_fps)
        ans = generate_text(*mp, messages=msg, max_new_tokens=256)
        res.append({**item, "model_answer": ans})
    out_path = f"{cfg['out_dir']}/LocalOCR_results.json"
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path


# -------- Spatial Grounding ---------------------------------------------------

def run_TextTracking(cfg, mp):
    data = json.load(open(cfg["ann"], "r", encoding="utf-8"))
    res = []
    for item in data:
        vid = item["video_id"]
        vfile = f"{cfg['video_dir']}/{vid}.mp4"
        if not os.path.exists(vfile):
            print(f"[SG] missing video {vfile}")
            continue
        fps, _, dur = read_video_info(vfile)
        eff_fps = 1.0 if dur <= 768 else 768 / dur
        choices = item.get("choices", [])
        question = item["question"]
        # build prompt
        prompt_lines = [f"{question} We have four possible bounding boxes:"]
        for c in choices:
            prompt_lines.append(
                f"Option {c['label']}: start_points={c['start_points']}, end_points={c['end_points']}"
            )
        prompt_lines.append("Which option (A/B/C/D) is correct?")
        prompt = "".join(prompt_lines)
        msg = build_messages(vfile, prompt, eff_fps)
        ans_text = generate_text(*mp, messages=msg)
        res.append({**item, "model_answer": ans_text})
    out_path = f"{cfg['out_dir']}/TextTracking_results.json"
    json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    return out_path


# --------------------------- 4. TASK REGISTRY ---------------------------------



TASKS = {
    "OCR":  run_holistic_ocr,          # 不变
    "HR":   run_holistic_reasoning,    # 不变
    "LocalReasoning": run_Local_reasoning,          # 旧 TR → 新 LocalReasoning
    "TCR":  run_temporal_causal_reasoning,         # 不变
    "SR":   run_spatial_reasoning,                 # 不变
    "LocalOCR":       run_LocalOCR,      #  LocalOCR
    "TextLocalization": run_TextLocalization,        #  TextLocalization
    "TextTracking":   run_TextTracking,       # TextTracking
}


# --------------------------- 5. CLI -------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Unified benchmark runner")
    p.add_argument("--model_path", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tasks", default="all", help="Comma‑sep subset or 'all'")
    p.add_argument("--config_dir", default="configs", help="Folder containing per‑task JSON configs")
    return p.parse_args()


def load_task_cfg(config_dir: str, task_key: str) -> Dict[str, str]:
    cfg_file = Path(config_dir) / f"{task_key}.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Missing config for task {task_key}: {cfg_file}")
    return json.load(open(cfg_file, "r", encoding="utf-8"))


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, processor, device = load_model(args.model_path, args.gpu)
    mp = (model, processor, device)

    selected = list(TASKS.keys()) if args.tasks == "all" else [k.strip() for k in args.tasks.split(",")]
    summary = {}
    for key in selected:
        if key not in TASKS:
            print(f"[WARN] Unknown task key: {key}, skipped.")
            continue
        cfg = load_task_cfg(args.config_dir, key)
        cfg["out_dir"] = args.output_dir
        start = time.time()
        out_file = TASKS[key](cfg, mp)
        summary[key] = {"output": out_file, "seconds": round(time.time() - start, 1)}
        torch.cuda.empty_cache()
    json.dump(summary, open(f"{args.output_dir}/summary.json", "w", encoding="utf-8"), indent=4)
    print("\nAll done! See summary.json for details.")


if __name__ == "__main__":
    main()


