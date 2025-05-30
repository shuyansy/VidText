#!/usr/bin/env python
# evaluate_tasks.py
# -------------------------------------------------------------
# 评估 8 个任务（OCR F1、Localization mIoU、Reasoning ACC…）
# -------------------------------------------------------------
import os, json, argparse, pathlib, re
from collections import Counter
from typing import Dict, List, Tuple

###############################################################################
# 1. 任务与文件模式  —— 请把 anno 路径改成你真实标注文件
###############################################################################
TASK_SPECS = {
    "holistic_ocr": {
        "pred":   "outputs/holistic_ocr_*.json",
        "anno":   "PATH/TO/holisticOCRTest.json",
        "metric": "f1",
    },
    "local_ocr": {
        "pred":   "outputs/local_ocr_*.json",
        "anno":   "PATH/TO/localOCR.json",
        "metric": "f1",
    },
    "holistic_reasoning": {
        "pred":   "outputs/holistic_reasoning_*.json",
        "anno":   "PATH/TO/new_hoReasoning.json",
        "metric": "acc_all3",
    },
    "temp_causal_reasoning": {
        "pred":   "outputs/tcr_*.json",
        "anno":   "PATH/TO/tempCausalReasoning_balanced.json",
        "metric": "acc_single",
    },
    "text_localization": {
        "pred":   "outputs/text_localization_*.json",
        "anno":   "PATH/TO/TextLocalizationTest.json",
        "metric": "miou",
    },
    "text_tracking": {
        "pred":   "outputs/text_tracking_*.json",
        "anno":   "PATH/TO/spatial_grounding_MCQ.json",
        "metric": "acc_single",
    },
    "spatial_reasoning": {
        "pred":   "outputs/spatial_reasoning_*.json",
        "anno":   "PATH/TO/spatialReasoning_balanced.json",
        "metric": "acc_single",
    },
    "local_reasoning": {
        "pred":   "outputs/local_reasoning_*.json",
        "anno":   "PATH/TO/TextReasoning.json",
        "metric": "acc_single",
    },
}

###############################################################################
# 2. 改进分词 + 单样本 F1
###############################################################################
TOKEN_RE = re.compile(
    r'[\u4e00-\u9fff]+'                # 连续中文
    r'|[A-Za-z0-9/]+'                  # 不拆分 17309/C
    r'|[^\u4e00-\u9fffA-Za-z0-9\s]',   # 其它符号
    re.U
)

def tokenize_text(text: str) -> List[str]:
    """大小写无关分词，保持 '17309/C' 完整"""
    return TOKEN_RE.findall(text.lower())

def f1_micro(pred: str, truth: str) -> float:
    p_tok, t_tok = tokenize_text(pred), tokenize_text(truth)
    common = Counter(p_tok) & Counter(t_tok)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tok) if p_tok else 0.0
    recall    = num_same / len(t_tok) if t_tok else 0.0
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0

###############################################################################
# 3. 各指标
###############################################################################
def eval_f1(pred: List[Dict], gt_map: Dict[str, str]) -> float:
    f1s = [f1_micro(it["model_answer"], gt_map[str(it["video_id"])])
           for it in pred if str(it["video_id"]) in gt_map]
    return sum(f1s)/len(f1s) if f1s else 0.0

def eval_acc_all3(pred: List[Dict], gt_map: Dict[str, List[str]]) -> float:
    hit = 0
    for it in pred:
        qid = str(it["question_id"])
        if qid not in gt_map: continue
        gt = set(gt_map[qid])
        pred_ids = re.findall(r"[A-D]", it["model_answer"].upper())
        if set(pred_ids) == gt and len(gt) == 3:
            hit += 1
    return hit / len(pred) if pred else 0.0

def _pred_choice(it):
    raw = it.get("parsed_answer_id") or it.get("model_answer", "")
    m = re.search(r"[A-D]", raw.upper())
    return m.group(0) if m else "?"

def eval_acc_single(pred: List[Dict], gt_map: Dict[str, str]) -> float:
    correct = sum(_pred_choice(it) == gt_map.get(str(it["question_id"]), "?") for it in pred)
    return correct / len(pred) if pred else 0.0

def iou(seg_p, seg_g):
    inter = max(0, min(seg_p["end"], seg_g["end"]) - max(seg_p["start"], seg_g["start"]))
    union = max(seg_p["end"], seg_g["end"]) - min(seg_p["start"], seg_g["start"])
    return inter / union if union > 0 else 0.0

def eval_miou(pred: List[Dict], gt_map: Dict[str, Dict]) -> float:
    ious = [iou(it["model_answer"], gt_map[str(it["question_id"])])
            for it in pred if str(it["question_id"]) in gt_map and it["model_answer"]["start"] is not None]
    return sum(ious)/len(ious) if ious else 0.0

METRICS = {
    "f1":         eval_f1,
    "acc_all3":   eval_acc_all3,
    "acc_single": eval_acc_single,
    "miou":       eval_miou,
}

###############################################################################
# 4. 工具
###############################################################################
def latest_file(pattern: str) -> str | None:
    files = list(pathlib.Path().glob(pattern))
    return str(max(files, key=lambda p: p.stat().st_mtime)) if files else None

###############################################################################
# 5. CLI
###############################################################################
def main():
    ap = argparse.ArgumentParser("Evaluate tasks")
    ap.add_argument("--override_pred", action="append", nargs=2,
                    metavar=("task", "pred_file"))
    ap.add_argument("--override_anno", action="append", nargs=2,
                    metavar=("task", "anno_file"))
    args = ap.parse_args()
    pred_override = dict(args.override_pred or [])
    anno_override = dict(args.override_anno or [])

    report = {}
    for task, spec in TASK_SPECS.items():
        pred_path = pred_override.get(task) or latest_file(spec["pred"])
        anno_path = anno_override.get(task) or spec["anno"]
        if not pred_path or not os.path.exists(pred_path):
            print(f"[{task}] ❌ 缺少预测文件"); continue
        if not os.path.exists(anno_path):
            print(f"[{task}] ❌ 缺少标注文件"); continue

        pred = json.load(open(pred_path, encoding="utf-8"))
        anno = json.load(open(anno_path, encoding="utf-8"))

        # 构建 ground-truth 映射
        metric = spec["metric"]
        if metric == "f1":
            gt_map = {str(a["video_id"]): a["answer"] for a in anno}
        elif metric == "miou":
            gt_map = {str(a["question_id"]): a["answer"] for a in anno}
        elif metric == "acc_all3":
            gt_map = {str(a["question_id"]): a["answer"] for a in anno}  # list[str]
        else:  # acc_single
            gt_map = {str(a["question_id"]): (a["answer"] if isinstance(a["answer"], str)
                     else a["answer"]["option_id"]) for a in anno}

        score = METRICS[metric](pred, gt_map)
        report[task] = round(score * 100, 2)
        print(f"[{task}] {metric} → {report[task]}%")

    if report:
        print("\n===== Summary =====")
        avg = sum(report.values()) / len(report)
        for k, v in report.items():
            print(f"{k:<22}: {v:6.2f}%")
        print(f"{'Average':<22}: {avg:6.2f}%")

if __name__ == "__main__":
    main()
