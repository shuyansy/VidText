# tasks/spatialG_25.py
import os, json, cv2, torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def build_prompt(question, choices):
    prompt = ( "Watch the video and answer the grounding question.\n\n"
               f"{question}\n\nWe have four bounding-box options:\n" )
    for c in choices:
        prompt += (f"Option {c['label']}: "
                   f"start_points={c['start_points']}, end_points={c['end_points']}\n")
    return prompt + "Which option (A/B/C/D) is correct?"

def _effective_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    cap.release()
    return 1.0 if dur <= 768 else 768/dur

def _infer_once(model, proc, vpath, prompt, fps, device):
    msgs = [{"role":"user","content":[
        {"type":"video","video":f"file://{vpath}","fps":fps,"max_pixels":360*420},
        {"type":"text","text":prompt}
    ]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = proc(text=[text], images=img_in, videos=vid_in,
                  padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=128)
    ans = proc.decode(ids[0, inputs.input_ids.shape[-1]:],
                      skip_special_tokens=True).strip()
    return ans

# public 入口 ---------------------------------------------------
def run(cfg: dict):
    """
    cfg 需包含：
      model_path, video_dir, annotation_path
    """
    model_path   = cfg["model_path"]
    video_dir    = cfg["video_dir"]
    anno_path    = cfg["annotation_path"]
    device       = cfg.get("device", "cuda:0")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto")
    proc  = AutoProcessor.from_pretrained(model_path)

    annos = json.load(open(anno_path, encoding="utf-8"))
    results = []

    for a in tqdm(annos, desc="SpatialGrounding"):
        vid = a["video_id"]
        vpath = os.path.join(video_dir, f"{vid}.mp4")
        if not os.path.exists(vpath):
            print("Missing video:", vpath); continue

        prompt = build_prompt(a["question"], a["choices"])
        fps_eff = _effective_fps(vpath)

        try:
            answer = _infer_once(model, proc, vpath, prompt, fps_eff, device)
        except Exception as e:
            answer = f"ERROR: {e}"

        results.append({
            "video_id": vid,
            "question": a["question"],
            "prompt":   prompt,
            "model_answer": answer,
            "ground_truth": {"correct_label": a["correct_label"],
                             "choices": a["choices"]},
            "resolution": a.get("resolution",[1920,1080])
        })
    return results      # 与 FORMAT_REGISTRY 的 _pick_and_order 对应
