#!/usr/bin/env python3
"""CLI entry point for TrustBench-Emo.

Runs one (model, dataset) cell end-to-end and writes records.jsonl + results.json
into the chosen output directory.

Example
-------
    python run_benchmark.py \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --kind qwen2vl \
        --dataset FER2013 \
        --samples samples/fer2013_val.jsonl \
        --out results/qwen2vl_fer2013 \
        --dims D1 D2 D3 D4

The samples manifest is a JSONL file where each line has at least an "id" and a
"label" (one of the 7 emotions) and optionally an "image" (path) or "text" field.
For FER2013 set "image"; for GoEmotions set "text".
"""
import argparse
import json
import sys
from pathlib import Path

# Allow running as a script: ensure the package root is importable.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from src import EmotionModel, evaluate_cell, EMOTIONS  # noqa: E402


def load_samples(path: str):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj:
                obj["id"] = len(samples)
            samples.append(obj)
    return samples


def parse_args():
    p = argparse.ArgumentParser(description="Run one TrustBench-Emo cell.")
    p.add_argument("--model", required=True, help="HuggingFace id or local path")
    p.add_argument("--kind", required=True,
                   choices=["qwen2vl", "qwen2_5_vl", "qwen_text",
                            "llama", "gemma", "causal", "llava"],
                   help="Model architecture kind")
    p.add_argument("--dataset", required=True, choices=["FER2013", "GoEmotions"],
                   help="Dataset name (affects perturbation + reporting)")
    p.add_argument("--samples", required=True, help="JSONL manifest of samples")
    p.add_argument("--out", required=True, help="Output directory for results")
    p.add_argument("--dims", nargs="+", default=["D1", "D2", "D3", "D4"],
                   choices=["D1", "D2", "D3", "D4"],
                   help="Which trust dimensions to compute")
    p.add_argument("--seed", type=int, default=42, help="Random seed for perturbations")
    p.add_argument("--device", default=None, help="Torch device (default auto)")
    return p.parse_args()


def main():
    args = parse_args()
    import numpy as np
    np.random.seed(args.seed)

    print(f"[TrustBench-Emo] loading model {args.model} ({args.kind}) ...")
    model = EmotionModel(args.model, args.kind, device=args.device)

    print(f"[TrustBench-Emo] loading samples from {args.samples} ...")
    samples = load_samples(args.samples)
    print(f"[TrustBench-Emo] {len(samples)} samples, dataset={args.dataset}, dims={args.dims}")

    results = evaluate_cell(
        model=model,
        samples=samples,
        dataset_name=args.dataset,
        output_dir=args.out,
        dims=args.dims,
    )

    acc = results.get("accuracy")
    ece = results.get("ece")
    print(f"[TrustBench-Emo] done. accuracy={acc:.4f}" + (f" ece={ece:.4f}" if ece is not None else ""))
    print(f"[TrustBench-Emo] results written to {args.out}/results.json")


if __name__ == "__main__":
    main()
