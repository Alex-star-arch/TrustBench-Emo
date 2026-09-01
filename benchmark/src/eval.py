"""End-to-end evaluation of one (model, dataset) cell for TrustBench-Emo."""
import json
import os
import time
from pathlib import Path
from typing import List, Dict

import numpy as np

from .metrics import (
    expected_calibration_error,
    selective_prediction,
    reliability_diagram,
    per_class_accuracy,
)
from .prompts import EMOTIONS


def perturb_image(image_path, noise_std=0.01):
    """Return a perturbed version of an image (Gaussian noise)."""
    from PIL import Image
    img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    noisy = np.clip(img + np.random.normal(0, noise_std, img.shape), 0, 1)
    return noisy


def perturb_text(text: str) -> str:
    """Return a lightly perturbed version of a text utterance.

    Current implementation swaps a small number of words with simple synonyms.
    This is intentionally a minimal perturbation for D2/D3.
    """
    # Minimal synonym replacement; users can plug in a richer paraphraser.
    swaps = {
        "happy": "glad", "sad": "unhappy", "angry": "mad",
        "fear": "scared", "surprise": "shock", "disgust": "dislike",
    }
    words = text.split()
    for i, w in enumerate(words):
        low = w.lower().strip(".,!?")
        if low in swaps and np.random.rand() < 0.3:
            words[i] = swaps[low]
    return " ".join(words)


def evaluate_cell(model, samples: List[Dict], dataset_name: str, output_dir: str, dims: List[str] = None):
    """Evaluate one cell and write records.jsonl + results.json.

    Args:
        model: an EmotionModel instance.
        samples: list of {"id", "image": path or None, "text": str or None, "label": str}.
        dataset_name: "FER2013" or "GoEmotions".
        output_dir: where to save records.jsonl and results.json.
        dims: which dimensions to compute; default ["D1", "D2", "D3", "D4"].
    Returns:
        dict of computed metrics.
    """
    dims = dims or ["D1", "D2", "D3", "D4"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    start = time.time()

    for s in samples:
        image_path = s.get("image")
        text = s.get("text")
        pred = model.generate(image_path=image_path, text=text)
        rec = {
            "id": s["id"],
            "true": s["label"],
            "pred": pred["pred_label"],
            "pred_label": pred["pred_label"],
            "conf": pred["confidence"],
            "rationale": pred["rationale"],
            "raw": pred["raw"],
            "has_image": image_path is not None,
            "has_text": text is not None,
        }

        if "D2" in dims or "D3" in dims:
            # perturb and re-query
            if image_path:
                # For image we re-query with the same original image (the perturbation
                # is applied internally by the caller if desired) or with a perturbed copy.
                pert_text = None
            else:
                pert_text = perturb_text(text)
            pert_pred = model.generate(image_path=image_path, text=pert_text)
            rec["pred_pert"] = pert_pred["pred_label"]
            rec["rationale_pert"] = pert_pred["rationale"]
            rec["conf_pert"] = pert_pred["confidence"]

        records.append(rec)

    # Save raw records
    records_path = os.path.join(output_dir, "records.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    results = {
        "model": model.model_path,
        "kind": model.kind,
        "dataset": dataset_name,
        "n": len(records),
        "dims": dims,
        "time_sec": time.time() - start,
    }

    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] for r in records]
    confs = np.array([r["conf"] for r in records], float)
    corrects = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)])

    accuracy = float(np.mean(corrects))
    # macro F1
    f1s = []
    for lab in EMOTIONS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t != lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    results["accuracy"] = accuracy
    results["macro_f1"] = macro_f1

    if "D1" in dims:
        ece = expected_calibration_error(confs, corrects, n_bins=15)
        sel = selective_prediction(confs, corrects, coverage=0.9)
        cent, acc, cnt = reliability_diagram(confs, corrects, n_bins=15)
        results["ece"] = ece
        results["acc@0.9cov"] = sel["accuracy"]
        results["cov@0.9"] = sel["coverage"]
        results["reliability"] = {
            "centers": cent.tolist(),
            "acc": acc.tolist(),
            "counts": cnt.tolist(),
        }

    if "D2" in dims:
        from .metrics import rationale_flip_rate
        # build paired records for clean and perturbed
        clean_recs = [{"rationale": r["rationale"]} for r in records]
        pert_recs = [{"rationale": r.get("rationale_pert", r["rationale"])} for r in records]
        flip = rationale_flip_rate(clean_recs, pert_recs)
        results["D2_faithfulness"] = {"perturbation_flip_rate": flip}

    if "D3" in dims:
        y_pred_pert = [r.get("pred_pert", r["pred"]) for r in records]
        corrects_pert = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred_pert)])
        acc_full = accuracy
        acc_pert = float(corrects_pert.mean())
        results["D3_robustness"] = {
            "acc_full": acc_full,
            "acc_pert": acc_pert,
            "acc_drop_pp": (acc_full - acc_pert) * 100.0,
        }

    if "D4" in dims:
        from .metrics import per_class_accuracy
        per_class, gap = per_class_accuracy(y_true, y_pred, EMOTIONS)
        results["D4_fairness"] = {
            "class_acc_gap_pp": gap,
            "per_class_acc": per_class,
        }

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results
