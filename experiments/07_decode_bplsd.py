from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
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
    code_metadata = json.loads((args.code / "code.json").read_text())
    dem_metadata = json.loads((args.dem / "dem.json").read_text())
    samples_manifest_path = args.samples / "samples.json"
    code_manifest_path = args.code / "code.json"
    shots = int(sample_metadata["shots"])
    detector_count = int(sample_metadata["num_detectors"])
    observable_count = int(sample_metadata["num_observables"])
    detections_path = args.samples / "dets.b8"
    observables_path = args.samples / "obs_actual.b8"
    hx_path = args.code / "hx.npz"
    logical_path = args.dem / "logical_x.npz"
    verify_sha256(detections_path, sample_metadata["sha256"]["detections"], label="detections")
    verify_sha256(
        observables_path,
        sample_metadata["sha256"]["observables_actual"],
        label="actual observables",
    )
    verify_sha256(hx_path, code_metadata["hx_sha256"], label="Hx")
    verify_sha256(logical_path, dem_metadata["logical_x_sha256"], label="logical X")
    verify_sha256(
        code_manifest_path, dem_metadata["code_metadata_sha256"], label="DEM source code manifest"
    )
    if sample_metadata["source_dem_sha256"] != dem_metadata["model_sha256"]:
        raise ValueError("samples were not generated from the supplied DEM")
    if float(dem_metadata["error_rate"]) != args.error_rate:
        raise ValueError("decoder error rate does not match DEM manifest")
    syndromes = read_b8(detections_path, shots=shots, bits_per_shot=detector_count)
    actual_observables = read_b8(
        observables_path,
        shots=shots,
        bits_per_shot=observable_count,
    )
    hx = sparse.load_npz(hx_path).tocsr()
    logical_x = sparse.load_npz(logical_path).tocsr()
    result = decode_bplsd_batch(
        hx,
        syndromes,
        logical_x,
        error_rate=args.error_rate,
    )
    correction_path = args.out / "corrections.b8"
    predicted_path = args.out / "obs_predicted.b8"
    write_b8(correction_path, result.corrections)
    write_b8(predicted_path, result.predicted_observables)
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
            "source_sha256": {
                "code_manifest": sha256_file(code_manifest_path),
                "detections": sample_metadata["sha256"]["detections"],
                "logical_x": dem_metadata["logical_x_sha256"],
                "observables_actual": sample_metadata["sha256"]["observables_actual"],
                "samples_manifest": sha256_file(samples_manifest_path),
            },
            "sha256": {
                "corrections": sha256_file(correction_path),
                "observables_predicted": sha256_file(predicted_path),
            },
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
