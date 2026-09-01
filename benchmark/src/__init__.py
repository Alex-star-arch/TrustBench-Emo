"""TrustBench-Emo benchmark source package.

Public API:
    from benchmark.src import EmotionModel, evaluate_cell, EMOTIONS
    from benchmark.src.metrics import (expected_calibration_error,
                                       fit_platt, apply_platt,
                                       rationale_flip_rate, per_class_accuracy)

Note: `EmotionModel` requires PyTorch + transformers (see benchmark/requirements.txt).
The metric/parser utilities import without those dependencies.
"""
from .prompts import EMOTIONS, INSTRUCTION, image_prompt, text_prompt, parse_output
from .eval import evaluate_cell, perturb_image, perturb_text

# EmotionModel needs torch; defer so the rest of the package is importable
# on environments (e.g. CI figure generation) that only have numpy.
try:  # pragma: no cover - depends on optional heavy deps
    from .models import EmotionModel
except ImportError:  # torch/transformers missing
    EmotionModel = None  # type: ignore

__all__ = [
    "EMOTIONS",
    "INSTRUCTION",
    "image_prompt",
    "text_prompt",
    "parse_output",
    "evaluate_cell",
    "perturb_image",
    "perturb_text",
    "EmotionModel",
]
