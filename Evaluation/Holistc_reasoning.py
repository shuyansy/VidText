# Holistc_reasoning.py
import os, json, torch, cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def run(cfg: dict):
    """
    推理 Holistic Reasoning 多选问题。
    cfg 应该包含：
    {
        'model_path':      str,
        'video_dir':       str,
        'annotation_path': str
    }
    """
    model_path      = cfg["model_path"]
    video_dir       = cfg["video_dir"]
    annotation_path = cfg["annotation_path"]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto",
        attn_implementation="flash_attention_2", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    results = []

    for ann in tqdm(annotations):
        video_id      = ann["video_id"]
        question_text = ann["question"]
        options_dict  = ann["option"]

        # 构造选项字符串
        option_lines = [f"{k}: {options_dict[k]}" for k in sorted(options_dict)]
        option_str   = "\n".join(option_lines)

        # 构造 Prompt
        prompt_text = f"""Watch the video carefully and select the correct three answers.

Question: {question_text}

Options:
{option_str}

Please output your answer in the format: `Correct Answers: A, B, C`
Do not provide any additional explanations."""

        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        # 视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        fps  = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0:
            continue
        duration = frames / fps
        effective_fps = 1.0 if duration <= 768 else 768 / duration

        try:
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 360 * 420,
                        "fps": effective_fps,
                    },
                    {"type": "text", "text": prompt_text},
                ]
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to("cuda:0")

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=128)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
            decoded = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

            results.append({
                "video_id": video_id,
                "question": question_text,
                "options": options_dict,
                "prompt": prompt_text,
                "model_answer": decoded,
                "ground_truth": ann.get("answer")
            })

            del inputs, out_ids, trimmed
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error: {video_id}: {e}")
            continue

    return results
