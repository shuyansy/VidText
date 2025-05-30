# holistic_OCR.py
import os, json, torch, cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

def run(cfg: dict):
    """
    批量推理 Holistic OCR 视频任务。
    参数 cfg 示例：
    {
        'model_path': '/path/to/Qwen2.5-VL',
        'video_dir': '/path/to/videos',
        'annotation_path': '/path/to/holisticOCRTest.json',
    }
    """
    model_path      = cfg["model_path"]
    video_dir       = cfg["video_dir"]
    annotation_path = cfg["annotation_path"]

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    prompt = (
        "Recognize all visual texts in the video. "
        "If the text is not in English, do not provide an English translation. "
        "Do not include any descriptions, narrative, or context. "
        "Output only the extracted text lines, each on a new line."
    )

    results = []

    for ann in tqdm(annotations):
        video_id = ann["video_id"]
        question = prompt
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Missing video: {video_path}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Cannot open video: {video_path}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps <= 0:
            continue
        duration = frames / fps
        effective_fps = 1.0 if duration <= 768 else 768 / duration

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "max_pixels": 360 * 420,
                            "fps": effective_fps,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda:0")

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128)
            trimmed_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()

            results.append({
                "video_id": video_id,
                "question_id": ann.get("question_id"),
                "prompt": question,
                "model_answer": output_text,
                "ground_truth": ann.get("answer"),
                "type": ann.get("type"),
                "source": ann.get("source"),
            })

            # 清理缓存
            del inputs, output_ids, trimmed_ids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error with video {video_id}: {e}")
            continue

    return results   # 返回交给 task_runner.py 保存
