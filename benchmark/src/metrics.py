"""TrustBench-Emo metrics: D1 calibration, D2 faithfulness, D3 robustness, D4 fairness, plus Platt scaling."""
import numpy as np
from typing import List, Dict, Tuple


def reliability_diagram(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 15):
    """Return bin centers, accuracies, and counts for a reliability diagram."""
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confs >= lo) & (confs <= hi) if i == n_bins - 1 else (confs >= lo) & (confs < hi)
        if mask.sum() > 0:
            centers.append(float(confs[mask].mean()))
            accs.append(float(corrects[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(counts)


def expected_calibration_error(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    """Binned ECE (15 equal-width bins by default)."""
    centers, accs, counts = reliability_diagram(confs, corrects, n_bins)
    if counts.sum() == 0:
        return float("nan")
    return float(np.sum(counts * np.abs(centers - accs)) / counts.sum())


def selective_prediction(confs: np.ndarray, corrects: np.ndarray, coverage: float = 0.9) -> Dict[str, float]:
    """Accuracy at a given coverage threshold (retain highest-confidence predictions)."""
    confs = np.asarray(confs, float)
    corrects = np.asarray(corrects, float)
    n = len(confs)
    k = max(1, int(round(coverage * n)))
    idx = np.argsort(-confs)[:k]
    return {
        "coverage": coverage,
        "n_retained": int(k),
        "accuracy": float(corrects[idx].mean()),
    }


def fit_platt(c_train: np.ndarray, y_train: np.ndarray, iters: int = 3000, lr: float = 0.03) -> Tuple[float, float]:
    """Fit a 2-parameter logistic Platt scaling on logit(conf)."""
    c = np.clip(c_train, 1e-5, 1 - 1e-5)
    t = np.log(c / (1 - c))
    y = np.asarray(y_train, float)
    a, b = 0.0, 0.0
    for _ in range(iters):
        z = a * t + b
        p = 1.0 / (1.0 + np.exp(-z))
        da = np.mean((p - y) * t)
        db = np.mean(p - y)
        a -= lr * da
        b -= lr * db
    return float(a), float(b)


def apply_platt(confs: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply a fitted Platt scaling."""
    c = np.clip(np.asarray(confs, float), 1e-5, 1 - 1e-5)
    t = np.log(c / (1 - c))
    return 1.0 / (1.0 + np.exp(-(a * t + b)))


def rationale_flip_rate(records_clean: List[Dict], records_pert: List[Dict]) -> float:
    """Fraction of cases where the model's stated rationale changes under perturbation.

    The prediction label is allowed to stay the same or change; we only compare
    the free-text rationale after normalisation.
    """
    n = min(len(records_clean), len(records_pert))
    flips = 0
    for rc, rp in zip(records_clean[:n], records_pert[:n]):
        r1 = _normalise_rationale(rc.get("rationale", ""))
        r2 = _normalise_rationale(rp.get("rationale", ""))
        if r1 != r2:
            flips += 1
    return flips / n if n > 0 else float("nan")


def _normalise_rationale(text: str) -> str:
    text = text.lower().strip(" .,;:-")
    # drop the label word at the start if present
    from .prompts import EMOTIONS
    for emo in EMOTIONS:
        text = text.replace(emo, "")
    text = " ".join(text.split())
    return text


def per_class_accuracy(y_true: List[str], y_pred: List[str], labels: List[str]) -> Tuple[Dict[str, float], float]:
    """Return per-class accuracy dict and max-min gap in percentage points."""
    accs = {}
    for lab in labels:
        n_lab = sum(1 for yt in y_true if yt == lab)
        if n_lab == 0:
            accs[lab] = 0.0
        else:
            accs[lab] = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab == yp) / n_lab
    gap = max(accs.values()) - min(accs.values())
    return accs, gap * 100.0
