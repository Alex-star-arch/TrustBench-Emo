# TrustBench-Emo

A **trustworthiness evaluation toolkit** for emotion-aware multimodal large language
models (MLLMs). It is model-agnostic: plug in any HuggingFace-style vision-language or
text model through a thin wrapper, and obtain trust metrics on your own samples.

## Trust dimensions

| Dim | Name | Metric |
|-----|------|--------|
| D1 | **Calibration** | Expected Calibration Error (ECE) + accuracy at 90% coverage |
| D2 | **Explanation faithfulness** | Rationale-flip rate under input perturbation |
| D3 | **Modality robustness** | Accuracy drop (pp) under input perturbation |
| D4 | **Fairness** | Max–min per-class accuracy gap (pp) |

A training-free mitigation (Platt scaling + selective prediction) is included for D1.

## Repository structure

```
TrustBench-Emo/
├── benchmark/                 # Reproducible evaluation code
│   ├── requirements.txt
│   ├── run_benchmark.py       # CLI entry point (one model×dataset cell)
│   └── src/
│       ├── prompts.py         # Emotion label set, prompt templates, parser
│       ├── models.py          # EmotionModel wrapper
│       ├── metrics.py         # D1–D4 metrics + Platt scaling
│       └── eval.py            # evaluate_cell() end-to-end pipeline
├── LICENSE                    # Apache-2.0
├── .gitignore
└── .gitattributes
```

The `paper/` and `data/` directories are intentionally **not** part of this public
repository; the toolkit runs on data you supply locally.

## Quick start

### 1. Install dependencies

```bash
cd benchmark
pip install -r requirements.txt   # torch, transformers, Pillow, numpy, scikit-learn, tqdm
```

### 2. Prepare a samples manifest (JSONL)

Each line: `{"id": 0, "label": "happy", "image": "path/or/null", "text": "..."}`
- **Image dataset** (e.g. FER2013): set `image`, leave `text` null.
- **Text dataset** (e.g. GoEmotions): set `text`, leave `image` null.

### 3. Run one cell

```bash
python run_benchmark.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --kind qwen2vl \
    --dataset FER2013 \
    --samples samples/fer2013_val.jsonl \
    --out results/qwen2vl_fer2013 \
    --dims D1 D2 D3 D4
```

Outputs `results/<cell>/records.jsonl` (per-sample conf/label/rationale) and
`results/<cell>/results.json` (aggregated D1–D4 metrics).

### 4. Use the mitigation

```python
from benchmark.src.metrics import fit_platt, apply_platt, expected_calibration_error
a, b = fit_platt(calib_confs, calib_correct)
calibrated = apply_platt(test_confs, a, b)
```

## License

Released under the [Apache License 2.0](LICENSE).
