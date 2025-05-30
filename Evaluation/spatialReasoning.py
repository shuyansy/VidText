# tasks/SR_qw25.py
import os, json, re, torch, numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def build_prompt(q, opts):
    p = ("Watch the video and answer the multiple-choice question.\n\n"
         f"Question: {q}\n\nOptions:\n")
    for o in opts:
        p += f"Option {o['option_id']}: {o['text']}\n"
    return p + "Which option (A/B/C/D) is correct?"

def fps_eff(vpath):
    vr  = VideoReader(vpath, ctx=cpu(0))
    dur = len(vr) / vr.get_avg_fps()
    return 1.0 if dur <= 768 else 768 / dur

def infer(model, proc, vpath, prompt, fps, device="cuda:0"):
    msgs = [{"role":"user","content":[
        {"type":"text","text":prompt},
        {"video": f"file://{vpath}", "fps": fps},
    ]}]
    txt  = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inp = proc(text=[txt], images=img_in, videos=vid_in,
               padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=256)
    gen = ids[0, inp.input_ids.shape[-1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()

def parse_choice(t):
    t = t.replace("Answer:", "").replace("Correct Answer:", "").replace("Option", "").strip()
    return t[0] if t and t[0] in "ABCD" else "INVALID"

# ------------------- public entry ------------------
def run(cfg: dict):
    """
    cfg 必含: model_path, video_dir, annotation_path
    """
    model_path   = cfg["model_path"]
    video_dir    = cfg["video_dir"]
    anno_path    = cfg["annotation_path"]
    device       = cfg.get("device", "cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto")
    proc  = AutoProcessor.from_pretrained(model_path)

    tasks = json.load(open(anno_path, encoding="utf-8"))
    results = []

    for t in tqdm(tasks, desc="SpatialReasoning"):
        vid = t["video_id"]
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            print("Missing video:", vpath); continue

        # options 统一
        opts_raw = t["options"]
        opts = ( [{"option_id":k,"text":v} for k,v in opts_raw.items()]
                 if isinstance(opts_raw, dict) else opts_raw )
        gt_id = t["answer"] if isinstance(t["answer"], str) else t["answer"]["option_id"]

        prompt = build_prompt(t["question"], opts)
        try:
            pred_raw = infer(model, proc, vpath, prompt, fps_eff(vpath), device)
        except Exception as e:
            pred_raw = f"ERROR: {e}"

        pred_id = parse_choice(pred_raw)

        results.append({
            "question_id": t["question_id"],
            "video_id":    vid,
            "question":    t["question"],
            "options":     opts,
            "ground_truth":{"option_id": gt_id},
            "model_answer": pred_raw,
            "parsed_answer_id": pred_id,
            "is_correct":  pred_id == gt_id
        })
    return results            # 交给 task_runner 保存
