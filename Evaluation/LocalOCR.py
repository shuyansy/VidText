# tasks/temporal_grounding.py
import os, json, torch, cv2, hashlib, requests, numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import decord
from decord import VideoReader, cpu

# ---------------- prompt builder -----------------
def build_prompt(question_text: str) -> str:
    return (
        "Watch the video and answer the following question based on its content.\n\n"
        f"Question: {question_text}\n\n"
        "Please output only the texts that appear in the specified time interval as a JSON array of strings, "
        "without extra description or translation."
    )

# ---------------- video helper -------------------
def download_video(url: str, dest: str):
    resp = requests.get(url, stream=True); resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def get_video_frames(video_path: str, cache_dir=".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    vhash = hashlib.md5(video_path.encode()).hexdigest()
    local_path = video_path
    # 如果是网络视频先下载
    if video_path.startswith(("http://", "https://")):
        local_path = os.path.join(cache_dir, f"{vhash}.mp4")
        if not os.path.exists(local_path):
            download_video(video_path, local_path)

    vr = VideoReader(local_path, ctx=cpu(0))
    total, fps = len(vr), vr.get_avg_fps()
    duration = total / fps
    if duration > 768:                 # 均匀采样 768 帧
        idx = np.linspace(0, total - 1, 768, dtype=int)
    else:                              # 每秒 1 帧
        step = max(1, round(fps))
        idx  = np.arange(0, total, step)
    # 仅返回本地视频路径，frames/timestamps 若你后续需要再加
    return local_path

# ---------------- model inference ----------------
def infer(model, processor, video_path: str, prompt: str, device="cuda:0"):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"video": f"file://{video_path}", "fps": 1.0},
        ],
    }]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_inps, vid_inps = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_inps, videos=vid_inps,
                       padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256)
    gen = out[0, inputs.input_ids.shape[-1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

# ---------------- unified run(cfg) ----------------
def run(cfg: dict):
    """
    cfg 应包含：
      model_path, annotation_path, video_dir
    """
    model_path      = cfg["model_path"]
    annot_path      = cfg["annotation_path"]
    video_dir       = cfg["video_dir"]
    device          = cfg.get("device", "cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    with open(annot_path, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    for t in tqdm(tasks, desc="TemporalGrounding"):
        qid, vid = t["question_id"], t["video_id"]
        prompt   = build_prompt(t["question"])
        vfile    = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vfile):
            print("Missing video:", vfile); continue
        vpath = get_video_frames(vfile)

        try:
            answer = infer(model, processor, vpath, prompt, device)
        except Exception as e:
            answer = f"ERROR: {e}"

        results.append({
            "question_id": qid,
            "video_id":    vid,
            "question":    t["question"],
            "prompt":      prompt,
            "model_answer": answer,
            "ground_truth": t.get("answer"),
        })
    return results
