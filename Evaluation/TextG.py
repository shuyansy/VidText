# tasks/TextGrounding.py
import os, json, re, cv2, torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROMPT_TEMPLATE = (
    "Watch the video and answer the following question based on its content.\n\n"
    "Please provide the time interval (in seconds, precise to 0.1s) during which the text appears in the video. "
    "Output your answer in JSON format with keys 'start' and 'end'. For example: {\"start\": 0.0, \"end\": 30.0}. "
    "Do not include any extra commentary."
)

def _infer_once(model, proc, video_path, question, fps_eff, device="cuda:0"):
    msgs = [{
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_path}",
             "max_pixels": 360*420, "fps": fps_eff},
            {"type": "text", "text": question},
        ],
    }]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128)
    gen = ids[0, inputs.input_ids.shape[-1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()

def _parse_json_span(ans: str):
    m = re.search(r'{"start":\s*([\d\.]+),\s*"end":\s*([\d\.]+)}', ans)
    if m:
        return {"start": float(m.group(1)), "end": float(m.group(2))}
    return {"start": None, "end": None}

# ------------- public entry -------------
def run(cfg: dict):
    """
    cfg 必填字段：model_path, video_dir, annotation_path
    """
    model_path  = cfg["model_path"]
    video_dir   = cfg["video_dir"]
    anno_path   = cfg["annotation_path"]
    device      = cfg.get("device", "cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    proc  = AutoProcessor.from_pretrained(model_path)

    annotations = json.load(open(anno_path, encoding="utf-8"))
    results = []

    for ann in tqdm(annotations, desc="TextGrounding"):
        vid = ann["video_id"]
        question = f"When does the text '{ann['question']}' exist in the video?"
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            print("Missing video:", vpath); continue

        # fps 估算
        cap = cv2.VideoCapture(vpath); fps = cap.get(cv2.CAP_PROP_FPS)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (fps or 1); cap.release()
        fps_eff = 1.0 if dur <= 196 else 196/dur

        try:
            answer_raw = _infer_once(model, proc, vpath, PROMPT_TEMPLATE, fps_eff, device)
        except Exception as e:
            answer_raw = f"ERROR: {e}"

        results.append({
            "video_id":     vid,
            "question_id":  ann.get("question_id"),
            "question":     question,
            "model_answer": _parse_json_span(answer_raw),
            "ground_truth": ann.get("answer"),
            "type":         ann.get("type"),
            "source":       ann.get("source"),
        })
    return results      # 交给 task_runner 保存
