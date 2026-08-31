from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
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
    model_metadata = json.loads(args.model.with_name("model.json").read_text())
    samples_manifest_path = args.samples / "samples.json"
    code_manifest_path = args.code / "code.json"
    sample_metadata = json.loads(samples_manifest_path.read_text())
    code_metadata = json.loads(code_manifest_path.read_text())
    dem_metadata = json.loads((args.dem / "dem.json").read_text())
    syndrome_path = args.tensors / "syndromes.npy"
    correction_path = args.tensors / "corrections.npy"
    observable_path = args.samples / "obs_actual.b8"
    hx_path = args.code / "hx.npz"
    logical_path = args.dem / "logical_x.npz"
    verify_sha256(syndrome_path, split["sha256"]["syndromes"], label="syndrome tensor")
    verify_sha256(correction_path, split["sha256"]["corrections"], label="correction tensor")
    verify_sha256(args.model, model_metadata["sha256"], label="FNO model")
    verify_sha256(
        observable_path,
        sample_metadata["sha256"]["observables_actual"],
        label="actual observables",
    )
    verify_sha256(hx_path, code_metadata["hx_sha256"], label="Hx")
    verify_sha256(logical_path, dem_metadata["logical_x_sha256"], label="logical X")
    verify_sha256(
        code_manifest_path, dem_metadata["code_metadata_sha256"], label="DEM source code manifest"
    )
    if sha256_file(samples_manifest_path) != split["source_sha256"]["samples_manifest"]:
        raise ValueError("samples manifest does not match tensor source")
    if sample_metadata["source_dem_sha256"] != dem_metadata["model_sha256"]:
        raise ValueError("samples were not generated from the supplied DEM")
    if model_metadata["source_sha256"] != split["sha256"]:
        raise ValueError("FNO model source hashes do not match evaluation tensors")
    test_start = int(split["test"]["start"])
    test_stop = int(split["test"]["stop"])
    syndrome_fields = np.load(syndrome_path, mmap_mode="r")[test_start:test_stop]
    teacher_fields = np.load(correction_path, mmap_mode="r")[test_start:test_stop]
    shots = test_stop - test_start

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = RingFNO(**checkpoint["configuration"])
    model.load_state_dict(checkpoint["state_dict"])
    started = perf_counter()
    logits = predict_fno(model, syndrome_fields)
    inference_seconds = perf_counter() - started

    all_observables = read_b8(
        observable_path,
        shots=int(sample_metadata["shots"]),
        bits_per_shot=int(sample_metadata["num_observables"]),
    )
    hx = sparse.load_npz(hx_path).tocsr()
    logical_x = sparse.load_npz(logical_path).tocsr()
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
