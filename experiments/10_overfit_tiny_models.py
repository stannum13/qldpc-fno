from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.training.overfit import overfit_fno, predict_fno


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    split = json.loads((args.tensors / "split.json").read_text())
    train_stop = int(split["train"]["stop"])
    if not 0 < args.shots <= train_stop:
        raise ValueError(f"shots must be between 1 and the training split size ({train_stop})")
    syndromes = np.load(args.tensors / "syndromes.npy", mmap_mode="r")[: args.shots]
    targets = np.load(args.tensors / "corrections.npy", mmap_mode="r")[: args.shots]
    model, metrics = overfit_fno(syndromes, targets, steps=args.steps, seed=args.seed)

    logits = predict_fno(model, syndromes)
    correction_bits = (logits >= 0).astype(np.uint8).reshape(args.shots, -1)
    syndrome_bits = np.asarray(syndromes, dtype=np.uint8).reshape(args.shots, -1)
    hx = sparse.load_npz(args.code / "hx.npz").tocsr()
    predicted_syndromes = np.asarray(hx @ correction_bits.T).T % 2
    syndrome_valid = np.all(predicted_syndromes == syndrome_bits, axis=1)
    metrics.update(
        {
            "gates": {
                "loss_decreased": metrics["final_weighted_bce"]
                < metrics["initial_weighted_bce"],
                "syndrome_valid_at_least_90_percent": float(syndrome_valid.mean()) >= 0.9,
                "teacher_bit_accuracy_above_99_percent": metrics["teacher_bit_accuracy"]
                > 0.99,
            },
            "syndrome_valid": int(syndrome_valid.sum()),
            "syndrome_valid_rate": float(syndrome_valid.mean()),
        }
    )

    model_path = args.out / "model.pt"
    torch.save(
        {"configuration": model.configuration(), "state_dict": model.state_dict()}, model_path
    )
    write_canonical_json(
        args.out / "model.json",
        {
            "configuration": model.configuration(),
            "format": "pytorch_state_dict",
            "sha256": sha256_file(model_path),
            "source_sha256": split["sha256"],
        },
    )
    write_canonical_json(args.out / "train_metrics.json", metrics)


if __name__ == "__main__":
    main()
