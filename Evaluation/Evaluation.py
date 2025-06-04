#!/usr/bin/env python
"""evaluation.py – Compute metrics for each task (single‑task or batch).

Usage examples
--------------
1) Evaluate **all** tasks that have result files in `results_dir`:

    python evaluation.py --results_dir ./results

2) Evaluate a subset (e.g. OCR and HR only):

    python evaluation.py --results_dir ./results --tasks OCR,HR

Outputs
-------
For every evaluated task the script creates a file next to the result file:
`results_dir/eval_<TASK>.json` containing

    {
        "summary": { metric_name: value, ... },
        "details": [   # sample‑level augmented records
            { ..original fields.., metric1: ..., metric2: ... },
            ...
        ]
    }

If more than one task is evaluated an aggregated `metrics_report.json` is
written (task -> summary).

Metric rules implemented according to the specification:
* OCR & TemporalGrounding   – Precision / Recall / **F1** and average
  Levenshtein edit distance.
* TextGrounding             – **mIoU** for 1‑D time intervals.
* HolisticReasoning         – Accuracy, counting **1 only when all 3
  options correct**.
* TR, TCR, SR, SG           – Standard single‑choice accuracy.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Levenshtein import distance as levenshtein_distance

# ---------------------------------------------------------------------------
# 1. Utility functions – tokenisation & F1 for OCR‑style text
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Case‑insensitive tokeniser tailored for OCR strings.

    Rules (ordered by priority):
    1. pattern 17309/C  (digits + '/' + alnum)
    2. consecutive Chinese characters
    3. letters / digits / '/'
    4. any other visible non‑whitespace symbol
    """
    text = text.lower()
    pattern = r"\d+/[a-z0-9]+|[\u4e00-\u9fff]+|[a-z0-9/]+|[^\u4e00-\u9fffa-z0-9\s]"
    return re.findall(pattern, text)


def prf1(pred: str, truth: str) -> Tuple[float, float, float, int]:
    """Return Precision, Recall, F1 and Levenshtein distance."""
    p_tok, t_tok = tokenize(pred), tokenize(truth)
    overlap = set(p_tok) & set(t_tok)
    n_same = sum(min(p_tok.count(t), t_tok.count(t)) for t in overlap)
    precision = n_same / len(p_tok) if p_tok else 0.0
    recall    = n_same / len(t_tok) if t_tok else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dist      = levenshtein_distance(pred.lower(), truth.lower())
    return precision, recall, f1, dist

# ---------------------------------------------------------------------------
# 2. IoU for 1‑D intervals (TextGrounding)
# ---------------------------------------------------------------------------

def interval_iou(pred: Dict, truth: Dict) -> float:
    """Intersection‑over‑Union for [start, end] intervals."""
    ps, pe = pred.get("start"), pred.get("end")
    ts, te = truth.get("start"), truth.get("end")
    if None in (ps, pe, ts, te) or pe < ps or te < ts:
        return 0.0
    inter = max(0.0, min(pe, te) - max(ps, ts))
    union = max(pe, te) - min(ps, ts)
    return inter / union if union else 0.0

# ---------------------------------------------------------------------------
# 3. Loader – unify list vs. {summary, details}
# ---------------------------------------------------------------------------

def load_items(path: str) -> List[Dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    return data["details"] if isinstance(data, dict) and "details" in data else data

# ---------------------------------------------------------------------------
# 4. Task‑specific evaluators
# ---------------------------------------------------------------------------

def eval_ocr(path: str) -> Dict:
    items, details = load_items(path), []
    for it in items:
        p, r, f, d = prf1(it.get("model_answer", ""), it.get("ground_truth", ""))
        details.append({**it, "precision": p, "recall": r, "f1": f, "edit_dist": d})
    if not details:
        return {}
    arr = np.array([[d["precision"], d["recall"], d["f1"], d["edit_dist"]] for d in details])
    summary = {
        "precision": arr[:, 0].mean().item(),
        "recall":    arr[:, 1].mean().item(),
        "f1":        arr[:, 2].mean().item(),
        "edit_dist": arr[:, 3].mean().item(),
        "samples":   len(details),
    }
    return {"summary": summary, "details": details}


def eva_localOCR(path: str) -> Dict:
    items, details = load_items(path), []
    for it in items:
        pred_raw = it.get("model_answer", "")
        # Temporal Grounding answer may be JSON array encoded as string
        if isinstance(pred_raw, str):
            try:
                pred_raw = " ".join(json.loads(pred_raw))
            except Exception:
                pass
        truth_raw = " ".join(it.get("answer", []))
        p, r, f, d = prf1(str(pred_raw), truth_raw)
        details.append({**it, "precision": p, "recall": r, "f1": f, "edit_dist": d})
    if not details:
        return {}
    arr = np.array([[d["precision"], d["recall"], d["f1"], d["edit_dist"]] for d in details])
    summary = {
        "precision": arr[:, 0].mean().item(),
        "recall":    arr[:, 1].mean().item(),
        "f1":        arr[:, 2].mean().item(),
        "edit_dist": arr[:, 3].mean().item(),
        "samples":   len(details),
    }
    return {"summary": summary, "details": details}


def eva_TextLocalization(path: str) -> Dict:
    items, details = load_items(path), []
    for it in items:
        pred = it.get("model_answer")
        if isinstance(pred, str):
            try:
                pred = json.loads(pred)
            except Exception:
                pred = {"start": None, "end": None}
        iou = interval_iou(pred or {}, it.get("ground_truth", {}))
        details.append({**it, "IoU": iou})
    miou = float(np.mean([d["IoU"] for d in details])) if details else 0.0
    return {"summary": {"mIoU": miou, "samples": len(details)}, "details": details}


def eval_accuracy(path: str) -> Dict:
    items = load_items(path)
    details = [{**it, "correct": bool(it.get("is_correct", False))} for it in items]
    acc = np.mean([d["correct"] for d in details]) if details else 0.0
    return {"summary": {"accuracy": acc, "samples": len(details)}, "details": details}

# ---------------------------------------------------------------------------
# 5. Registry mapping task‑key -> evaluator
# ---------------------------------------------------------------------------
EVAL = {
    "OCR":        eval_ocr,
    "LocalOCR":   eva_localOCR,          # 原 TempG
    "TextLocalization": eva_TextLocalization,  # 原 TextGround
    "HR":  eval_accuracy,
    "LocalReasoning": eval_accuracy,   # 原 TR
    "TCR": eval_accuracy,
    "SR":  eval_accuracy,
    "TextTracking": eval_accuracy,     # 原 SG
}

# ---------------------------------------------------------------------------
# 6. Command‑line interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compute metrics for benchmark results (single task or batch)")
    p.add_argument("--results_dir", required=True, help="Directory containing *_results*.json files")
    p.add_argument("--tasks", default="all", help="Comma‑sep list of task keys or 'all'")
    p.add_argument("--out", default="metrics_report.json", help="Aggregated report filename – generated only when ≥2 tasks")
    return p.parse_args()


def locate_result_file(res_dir: Path, key: str) -> str:
    """Find the first result file for a given task key (supporting several name patterns)."""
    patterns = [f"*{key}*result*.json", f"*{key}*Results*.json", f"*{key}*_merged.json"]
    for pat in patterns:
        matches = list(res_dir.glob(pat))
        if matches:
            return str(matches[0])
    return ""


def main():
    args = parse_args()
    res_dir = Path(args.results_dir)
    chosen = list(EVAL) if args.tasks.lower() == "all" else [t.strip() for t in args.tasks.split(",")]

    aggregated = {}
    for key in chosen:
        if key not in EVAL:
            print(f"[WARN] Unknown key '{key}', skipped")
            continue
        fpath = locate_result_file(res_dir, key)
        if not fpath:
            print(f"[WARN] No result file for {key} in {res_dir}")
            continue
        outcome = EVAL[key](fpath)
        if not outcome:
            print(f"[WARN] Empty evaluation for {key}")
            continue
        # write per‑task evaluation JSON
        eval_path = res_dir / f"eval_{key}.json"
        json.dump(outcome, open(eval_path, "w", encoding="utf-8"), indent=4)
        aggregated[key] = outcome["summary"]
        print(f"{key}: {outcome['summary']}  ->  saved to {eval_path.name}")

    if len(aggregated) > 1:
        json.dump(aggregated, open(args.out, "w", encoding="utf-8"), indent=4)
        print(f"\nAggregated report written to {args.out}")


if __name__ == "__main__":
    main()
