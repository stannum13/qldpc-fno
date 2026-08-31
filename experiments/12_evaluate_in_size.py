from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy import sparse

from qldpc_fno.artifacts import write_canonical_json
from qldpc_fno.metrics.decoding import evaluate_correction_logits
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.stim.b8 import read_b8
from qldpc_fno.training.overfit import predict_fno


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    split = json.loads((args.tensors / "split.json").read_text())
    test_start = int(split["test"]["start"])
    test_stop = int(split["test"]["stop"])
    syndrome_fields = np.load(args.tensors / "syndromes.npy", mmap_mode="r")[
        test_start:test_stop
    ]
    teacher_fields = np.load(args.tensors / "corrections.npy", mmap_mode="r")[
        test_start:test_stop
    ]
    shots = test_stop - test_start

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = RingFNO(**checkpoint["configuration"])
    model.load_state_dict(checkpoint["state_dict"])
    started = perf_counter()
    logits = predict_fno(model, syndrome_fields)
    inference_seconds = perf_counter() - started

    sample_metadata = json.loads((args.samples / "samples.json").read_text())
    all_observables = read_b8(
        args.samples / "obs_actual.b8",
        shots=int(sample_metadata["shots"]),
        bits_per_shot=int(sample_metadata["num_observables"]),
    )
    hx = sparse.load_npz(args.code / "hx.npz").tocsr()
    logical_x = sparse.load_npz(args.dem / "logical_x.npz").tocsr()
    metrics = evaluate_correction_logits(
        logits,
        hx=hx,
        syndromes=np.asarray(syndrome_fields, dtype=np.uint8).reshape(shots, -1),
        logical_x=logical_x,
        actual_observables=all_observables[test_start:test_stop],
    )
    predictions = logits >= 0
    metrics.update(
        {
            "inference_latency_seconds": {
                "amortized_per_shot": inference_seconds / shots,
                "batch_size": shots,
                "total": inference_seconds,
            },
            "split": {"start": test_start, "stop": test_stop},
            "teacher_bit_accuracy": float(np.mean(predictions == teacher_fields)),
        }
    )
    write_canonical_json(args.out / "metrics.json", metrics)


if __name__ == "__main__":
    main()
