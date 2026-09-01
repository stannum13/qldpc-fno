"""Reproduce the calibration candidate-throughput capacity measurement."""

from __future__ import annotations

import argparse
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file, verify_sha256
from qldpc_fno.campaign.shard_io import load_campaign_code
from qldpc_fno.data.conditional_fields import add_noise_channel
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.training.calibration import CALIBRATION_GRID

_EXPECTED_SHA256 = {
    "calibration_manifest": "8c931a791204d5c521a00d2814e6ded4cd88feb4c9564420d9c70d0074590534",
    "code_manifest": "83a3f6d0dd5229cb686ca9642f251e2ed424bc271ee8c3a2d48b36b4abe4d277",
    "config": "b1751f4373475192c58dd0226dcc9372eea9944da890c763ee5cc0617d38ba11",
    "model_checkpoint": "5417579795b3117fc2728e849e2736f1887e915ec626caf218c4325665750682",
    "model_manifest": "ffe8840fc18529a1a565e6a1beab1a6bacfe397a7a1c4e0f271229f8f142ebcf",
}


def _load_calibration_payloads(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    completion = json.loads((root / "manifest.json").read_text())
    if completion.get("complete") is not True or completion.get("role") != "calibration":
        raise ValueError("benchmark calibration publication is incomplete")
    declared = completion.get("shards")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("benchmark calibration shard table is malformed")
    arrays: dict[str, list[np.ndarray]] = {
        "dets.b8": [],
        "errors.b8": [],
        "obs_actual.b8": [],
    }
    rates: list[np.ndarray] = []
    for relative, digest in sorted(declared.items()):
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("benchmark calibration shard path is unsafe")
        manifest_path = root / relative
        verify_sha256(manifest_path, str(digest), label="benchmark shard manifest")
        manifest = json.loads(manifest_path.read_text())
        shots = manifest.get("shots")
        dimensions = manifest.get("dimensions")
        hashes = manifest.get("sha256")
        if (
            type(shots) is not int
            or shots <= 0
            or not isinstance(dimensions, dict)
            or not isinstance(hashes, dict)
        ):
            raise ValueError("benchmark shard manifest is malformed")
        for name, chunks in arrays.items():
            width = dimensions.get(name)
            if type(width) is not int or width <= 0:
                raise ValueError("benchmark shard dimension is malformed")
            payload_path = manifest_path.parent / name
            verify_sha256(payload_path, str(hashes[name]), label=f"benchmark {name}")
            packed = np.fromfile(payload_path, dtype=np.uint8)
            unpacked = np.unpackbits(packed, bitorder="little")[: shots * width]
            chunks.append(unpacked.reshape(shots, width).astype(np.uint8, copy=False))
        rates.append(np.full(shots, float(manifest["error_rate"]), dtype=np.float64))
    return (
        np.concatenate(arrays["dets.b8"]),
        np.concatenate(arrays["errors.b8"]),
        np.concatenate(arrays["obs_actual.b8"]),
        np.concatenate(rates),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or args.trials <= 0 or args.warmups < 0:
        raise ValueError("threads/trials must be positive and warmups non-negative")

    root = args.artifact.resolve()
    config_path = root / "inputs/config.json"
    code = root / "inputs/code"
    calibration = root / "calibration"
    model_dir = root / "training"
    config = json.loads(config_path.read_text())
    training_batch_size = config.get("training_batch_size")
    if type(training_batch_size) is not int or training_batch_size <= 0:
        raise ValueError("benchmark config training batch size is invalid")
    code_manifest = code / "code.json"
    _, hx, _, logical_x = load_campaign_code(code)
    syndromes, actual_errors, actual_observables, error_rates = (
        _load_calibration_payloads(calibration)
    )
    model_manifest_path = model_dir / "model.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    checkpoints = model_manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("benchmark artifact has no checkpoint candidates")
    checkpoint = checkpoints[0]
    if not isinstance(checkpoint, dict):
        raise TypeError("benchmark checkpoint identity is malformed")
    checkpoint_path = model_dir / str(checkpoint["path"])
    verify_sha256(checkpoint_path, str(checkpoint["sha256"]), label="benchmark checkpoint")
    artifact_sha256 = {
        "calibration_manifest": sha256_file(calibration / "manifest.json"),
        "code_manifest": sha256_file(code_manifest),
        "config": sha256_file(config_path),
        "model_checkpoint": sha256_file(checkpoint_path),
        "model_manifest": sha256_file(model_manifest_path),
    }
    if artifact_sha256 != _EXPECTED_SHA256:
        raise ValueError("benchmark artifact identity does not match the recorded measurement")
    if args.verify_only:
        print(json.dumps({"artifact_sha256": artifact_sha256}, indent=2, sort_keys=True))
        return
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    network = RingFNO(**model_manifest["configuration"])
    network.load_state_dict(payload["model_state_dict"])
    network.requires_grad_(False)
    network.eval()
    torch.set_num_threads(args.threads)

    shots = len(syndromes)
    model_inputs = np.empty((shots, 22, 45), dtype=np.float32)
    for offset in range(0, shots, training_batch_size):
        indices = np.arange(
            offset,
            min(offset + training_batch_size, shots),
            dtype=np.int64,
        )
        syndrome_batch = syndromes[indices]
        rate_batch = error_rates[indices]
        fields = to_ring_field(syndrome_batch, channels=21, ell=45)
        model_inputs[indices] = add_noise_channel(fields, rate_batch)
    with torch.no_grad():
        logits = network(torch.from_numpy(model_inputs)).numpy()

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "experiments/16_calibrate_hybrid_priors.py")
    )
    score_candidate = cast(Callable[..., object], namespace["_score_candidate"])

    def score() -> None:
        score_candidate(
            parameters=CALIBRATION_GRID[0],
            logits=logits,
            syndromes=syndromes,
            actual_errors=actual_errors,
            actual_observables=actual_observables,
            error_rates=error_rates,
            hx=hx,
            logical_x=logical_x,
            batch_size=training_batch_size,
        )

    for _ in range(args.warmups):
        score()
    trials: list[float] = []
    for _ in range(args.trials):
        started = perf_counter()
        score()
        trials.append(perf_counter() - started)
    result = {
        "artifact": str(root),
        "artifact_sha256": artifact_sha256,
        "candidate": {"alpha": 0.25, "beta": 0.0, "temperature": 0.5},
        "candidate_shots_per_second": shots / float(np.median(trials)),
        "median_seconds": float(np.median(trials)),
        "shots": shots,
        "threads": args.threads,
        "trials_seconds": trials,
        "warmups": args.warmups,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
