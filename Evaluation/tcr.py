# tasks/TCR_qw25.py
import os, json, re, torch, numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------- prompt ----------
def build_prompt(q, opts):
    p = "Watch the video and answer the multiple-choice question.\n\n"
    p += f"Question: {q}\n\nOptions:\n"
    for o in opts:
        p += f"Option {o['option_id']}: {o['text']}\n"
    return p + "\nPlease select the correct option."

# ---------- fps 抽帧 ----------
def get_fps_eff(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    dur = len(vr) / vr.get_avg_fps()
    return 1.0 if dur <= 768 else 768 / dur

# ---------- model infer ----------
def infer(model, proc, vpath, prompt, fps_eff, device):
    msgs = [{"role": "user", "content": [
        {"type": "text",  "text": prompt},
        {"video": f"file://{vpath}", "fps": fps_eff},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inp = proc(text=[text], images=img_in, videos=vid_in,
               padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256)
    gen = out[0, inp.input_ids.shape[-1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()

def parse_pred(text):
    text = text.replace("Answer:", "").replace("Correct Answer:", "").replace("Option", "").strip()
    return text[0] if text and text[0] in "ABCD" else "INVALID"

# ---------- run(cfg) ----------
def run(cfg: dict):
    """
    cfg 必填字段：model_path, annotation_path, video_dir
    """
    model_path   = cfg["model_path"]
    anno_path    = cfg["annotation_path"]
    video_dir    = cfg["video_dir"]
    device       = cfg.get("device", "cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    proc = AutoProcessor.from_pretrained(model_path)

    tasks = json.load(open(anno_path, encoding="utf-8"))
    results = []

    for t in tqdm(tasks, desc="TempCausalReasoning"):
        qid, vid = t["question_id"], t["video_id"]
        # options 统一为 list[dict]
        opts_raw = t["options"]
        opts = ( [{"option_id":k,"text":v} for k,v in opts_raw.items()]
                 if isinstance(opts_raw, dict) else opts_raw )
        gt_id = t["answer"] if isinstance(t["answer"], str) else t["answer"]["option_id"]

        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            print("Missing video:", vpath); continue

        prompt = build_prompt(t["question"], opts)
        fps_eff = get_fps_eff(vpath)

        try:
            out_raw = infer(model, proc, vpath, prompt, fps_eff, device)
        except Exception as e:
            out_raw = f"ERROR: {e}"

        pred_id = parse_pred(out_raw)
        results.append({
            "question_id": qid,
            "video_id":    vid,
            "question":    t["question"],
            "options":     opts,
            "ground_truth":{"option_id": gt_id},
            "model_answer": out_raw,
            "parsed_answer_id": pred_id,
            "is_correct":  pred_id == gt_id
        })

    return results           # ← 方案 A：只返回逐样本列表
    # 若要方案 B，改成：
    # acc = sum(r["is_correct"] for r in results)/len(results)
    # return {"summary":{"total":len(results),"correct":sum(r['is_correct'] for r in results),"accuracy":acc}, "details":results}
