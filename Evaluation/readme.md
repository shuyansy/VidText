# VidText-Bench 🏁  
Comprehensive benchmark & evaluation code-base for **video-text understanding** with large multimodal models (LMMs)

<details>
<summary><strong>Table of contents</strong></summary>

1. [What’s inside?](#whats-inside)  
2. [Quick start](#quick-start)  
3. [Folder layout](#folder-layout)  
4. [Running inference](#running-inference)  
5. [Running evaluation](#running-evaluation)  
6. [Configuration files](#configuration-files)  
7. [Adding new models or tasks](#adding-new-models-or-tasks)  
8. [Citation](#citation)  
</details>

---

## What’s inside?
| Part | File(s) | Purpose |
|------|---------|---------|
| **Inference** | `benchmark.py` | Runs any subset of the 8 tasks and stores raw model outputs. |
| **Scoring**   | `evaluation.py` | Turns raw outputs into metrics (`accuracy`, `F1`, `mIoU`, …). |
| **Configs**   | `configs/*.json` | Points each task key (e.g. `HR`) to its annotation file & video folder. |
| **Docs**      | this `README.md` | How to set up & use the code-base. |

### Supported tasks 📋

| Key | Task name (old → new)      | Level       | Metric |
|-----|----------------------------|-------------|--------|
| `OCR`            | Holistic OCR                       | video   | F1 |
| `HR`             | Holistic Reasoning                | video   | 3-way accuracy |
| `LocalOCR`       | Local OCR (TempGrounding → LocalOCR) | clip    | F1 |
| `LocalReasoning` | Local Reasoning (TextReasoning)   | clip    | accuracy |
| `TextLocalization` | Text Localization (TextGrounding) | clip    | *m*IoU |
| `TCR`            | Temporal Causal Reasoning         | clip    | accuracy |
| `SR`             | Spatial Reasoning                 | instance| accuracy |
| `TextTracking`   | Text Tracking (SpatialGrounding)  | instance| accuracy |

---

## Quick start

```bash
# 1- Install requirements
Download Qwen2.5-VL 7B yourself
   → Visit the official Qwen GitHub or HuggingFace page and follow their instructions.

# 2- Prepare data & configs  (update paths inside configs/*.json)

# 3- Run inference on all tasks with a local Qwen2.5-VL checkpoint
python benchmark.py \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --gpu 0 \
    --output_dir ./results \
    --tasks all            # or e.g.:  --tasks OCR,LocalOCR

# 4- Evaluate
python evaluation.py --results_dir ./results            # all tasks found
# customised subset:
python evaluation.py --results_dir ./results --tasks OCR,SR
