from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.stim.b8 import read_b8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--ell", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)

    samples_manifest_path = args.samples / "samples.json"
    decode_manifest_path = args.corrections.parent / "decode.json"
    metadata = json.loads(samples_manifest_path.read_text())
    decode_metadata = json.loads(decode_manifest_path.read_text())
    shots = int(metadata["shots"])
    detector_count = int(metadata["num_detectors"])
    error_count = int(metadata["num_errors"])
    if detector_count % args.ell or error_count % args.ell:
        raise ValueError("detector and correction widths must be divisible by ell")

    detections_path = args.samples / "dets.b8"
    verify_sha256(detections_path, metadata["sha256"]["detections"], label="detections")
    verify_sha256(
        args.corrections, decode_metadata["sha256"]["corrections"], label="teacher corrections"
    )
    samples_manifest_sha256 = sha256_file(samples_manifest_path)
    if decode_metadata["source_sha256"]["samples_manifest"] != samples_manifest_sha256:
        raise ValueError("teacher corrections do not belong to the supplied samples")
    syndrome_bits = read_b8(detections_path, shots=shots, bits_per_shot=detector_count)
    correction_bits = read_b8(args.corrections, shots=shots, bits_per_shot=error_count)
    syndromes = to_ring_field(
        syndrome_bits, channels=detector_count // args.ell, ell=args.ell
    )
    corrections = to_ring_field(
        correction_bits, channels=error_count // args.ell, ell=args.ell
    )
    syndrome_path = args.out / "syndromes.npy"
    correction_path = args.out / "corrections.npy"
    np.save(syndrome_path, syndromes, allow_pickle=False)
    np.save(correction_path, corrections, allow_pickle=False)

    train_shots = shots * 3 // 4
    write_canonical_json(
        args.out / "split.json",
        {
            "correction_shape": list(corrections.shape),
            "ell": args.ell,
            "sha256": {
                "corrections": sha256_file(correction_path),
                "syndromes": sha256_file(syndrome_path),
            },
            "source_sha256": {
                "decode_manifest": sha256_file(decode_manifest_path),
                "samples_manifest": samples_manifest_sha256,
            },
            "split_policy": "contiguous_first_75_percent",
            "syndrome_shape": list(syndromes.shape),
            "test": {"start": train_shots, "stop": shots},
            "train": {"start": 0, "stop": train_shots},
        },
    )


if __name__ == "__main__":
    main()
