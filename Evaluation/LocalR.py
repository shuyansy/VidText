# tasks/TextR_qw25.py
import os, json, torch, re, hashlib, numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------- prompt ----------------
def build_prompt(q, opts):
    p = "Watch the video and answer the multiple-choice question.\n\n"
    p += f"Question: {q}\n\nOptions:\n"
    for opt in opts:
        p += f"Option {opt['option_id']}: {opt['text']}\n"
    return p + "\nPlease select the correct option."

# ---------------- util ------------------
def parse_option_id(out: str):
    out = out.strip()
    try:
        j = json.loads(out);            # case: {"option_id":"B", ...}
        if isinstance(j, dict) and "option_id" in j:
            return j["option_id"]
    except Exception:
        pass
    m = re.search(r"(Option|Answer)?\s*([A-D])\b", out, re.I)
    return m.group(2).upper() if m else "INVALID"

def quick_vpath(vdir, vid):            # 用于拼接/检查视频文件
    vp = os.path.join(vdir, f"{vid}.mp4")
    if not os.path.exists(vp):
        raise FileNotFoundError(vp)
    return vp

# ---------------- model infer -----------
def infer(model, proc, vpath, prompt, device="cuda:0"):
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"video": vpath, "fps": 1.0},
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    gen = out[0, inputs.input_ids.shape[-1]:]
    return proc.decode(gen, skip_spec_
