from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import write_canonical_json
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.metrics.decoding import score_observable_predictions
from qldpc_fno.stim.b8 import read_b8, write_b8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--error-rate", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    sample_metadata = json.loads((args.samples / "samples.json").read_text())
    shots = int(sample_metadata["shots"])
    detector_count = int(sample_metadata["num_detectors"])
    observable_count = int(sample_metadata["num_observables"])
    syndromes = read_b8(args.samples / "dets.b8", shots=shots, bits_per_shot=detector_count)
    actual_observables = read_b8(
        args.samples / "obs_actual.b8",
        shots=shots,
        bits_per_shot=observable_count,
    )
    hx = sparse.load_npz(args.code / "hx.npz").tocsr()
    logical_x = sparse.load_npz(args.dem / "logical_x.npz").tocsr()
    result = decode_bplsd_batch(
        hx,
        syndromes,
        logical_x,
        error_rate=args.error_rate,
    )
    write_b8(args.out / "corrections.b8", result.corrections)
    write_b8(args.out / "obs_predicted.b8", result.predicted_observables)
    write_canonical_json(
        args.out / "decode.json",
        {
            "configuration": {
                "bp_method": "minimum_sum",
                "error_rate": args.error_rate,
                "lsd_method": "LSD_E",
                "lsd_order": 5,
                "max_iter": 100,
                "ms_scaling_factor": 0.0,
                "schedule": "serial",
            },
            "converged": int(result.converged.sum()),
            "latency_seconds": {
                "max": float(result.latency_seconds.max()),
                "mean": float(result.latency_seconds.mean()),
                "median": float(np.median(result.latency_seconds)),
                "total": float(result.latency_seconds.sum()),
            },
            "shots": shots,
            "syndrome_valid": int(result.syndrome_valid.sum()),
            "syndrome_valid_rate": float(result.syndrome_valid.mean()),
        },
    )
    write_canonical_json(
        args.out / "metrics.json",
        score_observable_predictions(actual_observables, result.predicted_observables),
    )


if __name__ == "__main__":
    main()
