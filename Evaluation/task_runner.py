# -*- coding: utf-8 -*-
"""
Unified Task Runner for Video-LLM Benchmark (8 tasks)
====================================================
Runs a single task or all tasks and writes predictions to task-specific JSON
following each benchmark’s schema.
"""
from __future__ import annotations
import sys, argparse, importlib, json, pathlib
from datetime import datetime
from typing import Any, Callable, Dict, List

# 允许从当前目录 import 模块
sys.path.append(".")

###############################################################################
# 1️⃣  Task registry  – maps CLI name → python module exposing `run(cfg)`
###############################################################################
TASK_REGISTRY: Dict[str, str] = {
    "holistic_ocr":          "holistic_OCR",
    "holistic_reasoning":    "Holistc_reasoning",
    "text_localization":     "TextLocalization",   # rename of text_grounding
    "temp_causal_reasoning": "TCR_qw25",
    "local_ocr":             "LocalOCR",          # 如果有本地 OCR 任务
    "text_tracking":         "spatialG_25",       # rename of spatial_grounding
    "spatial_reasoning":     "SR_qw25",
    "local_reasoning":       "TextR",             # rename of text_reasoning
}

###############################################################################
# 2️⃣  Helper transforms
###############################################################################
Transform = Callable[[Any], Any]

def _identity(x: Any) -> Any:
    return x

def _pick_and_order(fields: List[str]) -> Transform:
    def _fn(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: item.get(k) for k in fields} for item in raw]
    return _fn

def _transform_text_localization(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wanted = [
        "video_id", "question_id", "question",
        "model_answer", "ground_truth", "type", "source"
    ]
    fixed = []
    for item in raw:
        new = {k: item.get(k) for k in wanted}
        ma = new.get("model_answer", {})
        if isinstance(ma, dict):
            for key in ("start", "end"):
                try:
                    ma[key] = None if ma.get(key) is None else float(ma[key])
                except Exception:
                    ma[key] = None
        new["model_answer"] = ma
        fixed.append(new)
    return fixed

###############################################################################
# 3️⃣  Format registry  – filename + transform
###############################################################################
FORMAT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "holistic_ocr": {
        "filename": lambda ts: f"holistic_ocr_{ts}.json",
        "transform": _pick_and_order([
            "video_id","question_id","prompt",
            "model_answer","ground_truth","type","source"
        ]),
    },
    "holistic_reasoning": {
        "filename": lambda ts: f"holistic_reasoning_{ts}.json",
        "transform": _identity,
    },
    "text_localization": {
        "filename": lambda ts: f"text_localization_{ts}.json",
        "transform": _transform_text_localization,
    },
    "temp_causal_reasoning": {
        "filename": lambda ts: f"tcr_{ts}.json",
        "transform": _identity,
    },
    "local_ocr": {
        "filename": lambda ts: f"local_ocr_{ts}.json",
        "transform": _identity,
    },
    "text_tracking": {
        "filename": lambda ts: f"text_tracking_{ts}.json",
        "transform": _pick_and_order([
            "video_id","question","prompt",
            "model_answer","ground_truth","resolution"
        ]),
    },
    "spatial_reasoning": {
        "filename": lambda ts: f"spatial_reasoning_{ts}.json",
        "transform": _identity,
    },
    "local_reasoning": {
        "filename": lambda ts: f"local_reasoning_{ts}.json",
        "transform": _identity,
    },
}

###############################################################################
# 4️⃣  Helpers – dynamic import & JSON save
###############################################################################
def _load_runner(task: str):
    if task not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{task}'. Registered: {list(TASK_REGISTRY)}")
    mod_name = TASK_REGISTRY[task]
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Cannot import module '{mod_name}' for task '{task}'.") from e
    if not hasattr(module, "run"):
        raise AttributeError(f"Module '{mod_name}' lacks required `run(cfg)`.")
    return module.run

def _save_json(data: Any, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

###############################################################################
# 5️⃣  CLI
###############################################################################
def main():
    ap = argparse.ArgumentParser("Unified Video-LLM task runner")
    ap.add_argument("--task", help="Run single task by name")
    ap.add_argument("--all", action="store_true", help="Run all tasks")
    ap.add_argument("--config", help="YAML config file per-task blocks")
    ap.add_argument("--output_dir", default="outputs", help="Where to save JSON")
    args = ap.parse_args()

    if args.all:
        tasks = list(TASK_REGISTRY.keys())
    elif args.task:
        tasks = [args.task]
    else:
        print("\nAvailable tasks:")
        for t in TASK_REGISTRY: print(" •", t)
        print("\nUse --task <name> or --all  (-h for help)")
        return

    cfg: Dict[str, Any] = {}
    if args.config:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding="utf-8")) or {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = pathlib.Path(args.output_dir)

    for task in tasks:
        print(f"\n🚀 Running: {task}")
        raw = _load_runner(task)(cfg.get(task, {}))
        fmt = FORMAT_REGISTRY[task]
        data = fmt["transform"](raw)
        fname = fmt["filename"](ts) if callable(fmt["filename"]) else fmt["filename"]
        _save_json(data, out_root / fname)
        print("✅ Saved →", out_root / fname)

if __name__ == "__main__":
    main()
