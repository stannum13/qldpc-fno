from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shard_io import load_campaign_code, load_verified_shards
from qldpc_fno.data.conditional_fields import add_noise_channel
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.decoders.hybrid import decode_residual_batch, decode_soft_prior_batch
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.training.calibration import (
    CALIBRATION_GRID,
    CalibrationParameters,
    CalibrationScore,
    calibrated_probabilities,
    select_calibration,
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _score_payload(score: CalibrationScore) -> dict[str, object]:
    return {
        "block_errors": score.block_errors,
        "invalid_count": score.invalid_count,
        "nll": score.nll,
    }


def _selected_payload(score: CalibrationScore) -> dict[str, object]:
    return {"parameters": asdict(score.parameters), **_score_payload(score)}


def _probability_nll(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(
        -np.mean(targets * np.log(probabilities) + (1.0 - targets) * np.log1p(-probabilities))
    )


def _score_candidate(
    *,
    parameters: CalibrationParameters,
    logits: np.ndarray,
    syndromes: np.ndarray,
    actual_errors: np.ndarray,
    actual_observables: np.ndarray,
    error_rates: np.ndarray,
    hx: object,
    logical_x: object,
    batch_size: int,
) -> tuple[CalibrationScore, CalibrationScore]:
    soft_invalid = 0
    soft_failures = 0
    residual_invalid = 0
    residual_failures = 0
    nll_sum = 0.0
    shots = logits.shape[0]
    for offset in range(0, shots, batch_size):
        batch = slice(offset, min(offset + batch_size, shots))
        probabilities = calibrated_probabilities(logits[batch], error_rates[batch], parameters)
        flat_probabilities = probabilities.reshape(probabilities.shape[0], -1)
        nll_sum += _probability_nll(flat_probabilities, actual_errors[batch]) * len(
            flat_probabilities
        )

        soft = decode_soft_prior_batch(hx, syndromes[batch], logical_x, flat_probabilities)
        soft_mismatch = np.any(soft.predicted_observables != actual_observables[batch], axis=1)
        soft_invalid += int(np.count_nonzero(~soft.syndrome_valid))
        soft_failures += int(np.count_nonzero((~soft.syndrome_valid) | soft_mismatch))

        residual = decode_residual_batch(hx, syndromes[batch], logical_x, flat_probabilities)
        residual_mismatch = np.any(
            residual.predicted_observables != actual_observables[batch], axis=1
        )
        residual_invalid += int(np.count_nonzero(~residual.syndrome_valid))
        residual_failures += int(np.count_nonzero((~residual.syndrome_valid) | residual_mismatch))
    nll = nll_sum / shots
    return (
        CalibrationScore(parameters, soft_invalid, soft_failures, nll),
        CalibrationScore(parameters, residual_invalid, residual_failures, nll),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--grid-limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    candidates = CALIBRATION_GRID
    if args.grid_limit is not None:
        if args.grid_limit <= 0 or args.grid_limit > len(CALIBRATION_GRID):
            raise ValueError(f"grid-limit must be between 1 and {len(CALIBRATION_GRID)}")
        candidates = candidates[: args.grid_limit]
    if args.out.resolve() != args.calibration.resolve():
        raise ValueError("calibration output must be the verified calibration role directory")
    selected_path = args.out / "selected.json"
    if selected_path.exists():
        raise FileExistsError(f"refusing to overwrite completed calibration {selected_path}")

    code_manifest_path = args.code / "code.json"
    _, hx, _, logical_x = load_campaign_code(args.code)
    shards = load_verified_shards(
        args.calibration,
        role="calibration",
        config_path=args.config,
        code_manifest_path=code_manifest_path,
    )
    model_manifest_path = args.model / "model.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    if model_manifest.get("complete") is not True or model_manifest.get("source_role") != "train":
        raise ValueError("model is not a completed train-role publication")
    if model_manifest.get("git_commit") != _git_commit():
        raise ValueError("model Git commit does not match the calibration executable")
    model_sources = model_manifest.get("source_sha256")
    if not isinstance(model_sources, dict):
        raise TypeError("model is missing source SHA-256 provenance")
    if model_sources.get("config") != sha256_file(args.config):
        raise ValueError("model configuration provenance does not match calibration")
    if model_sources.get("code_manifest") != sha256_file(code_manifest_path):
        raise ValueError("model code provenance does not match calibration")
    model_path = args.model / "model.pt"
    verify_sha256(model_path, str(model_manifest["sha256"]), label="model")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint.get("configuration") != model_manifest.get("configuration"):
        raise ValueError("model configuration disagrees with model.json")
    model = RingFNO(**checkpoint["configuration"])
    model.load_state_dict(checkpoint["state_dict"])
    model.requires_grad_(False)
    model.eval()

    shots = shards.shots
    logits = np.empty((shots, 58, 45), dtype=np.float32)
    syndromes = np.empty((shots, 945), dtype=np.uint8)
    actual_errors = np.empty((shots, 2610), dtype=np.uint8)
    actual_observables = np.empty((shots, 744), dtype=np.uint8)
    error_rates = np.empty(shots, dtype=np.float64)
    with torch.no_grad():
        for offset in range(0, shots, config.training_batch_size):
            indices = np.arange(
                offset, min(offset + config.training_batch_size, shots), dtype=np.int64
            )
            syndrome_batch = shards.read("dets.b8", indices)
            rate_batch = shards.error_rates(indices)
            fields = to_ring_field(syndrome_batch, channels=21, ell=45)
            inputs = add_noise_channel(fields, rate_batch)
            logits[indices] = model(torch.from_numpy(inputs)).numpy()
            syndromes[indices] = syndrome_batch
            actual_errors[indices] = shards.read("errors.b8", indices)
            actual_observables[indices] = shards.read("obs_actual.b8", indices)
            error_rates[indices] = rate_batch

    rows: list[dict[str, object]] = []
    soft_scores: list[CalibrationScore] = []
    residual_scores: list[CalibrationScore] = []
    for parameters in candidates:
        soft, residual = _score_candidate(
            parameters=parameters,
            logits=logits,
            syndromes=syndromes,
            actual_errors=actual_errors,
            actual_observables=actual_observables,
            error_rates=error_rates,
            hx=hx,
            logical_x=logical_x,
            batch_size=config.training_batch_size,
        )
        soft_scores.append(soft)
        residual_scores.append(residual)
        rows.append(
            {
                "parameters": asdict(parameters),
                "residual": _score_payload(residual),
                "soft_prior": _score_payload(soft),
            }
        )

    source_sha256: dict[str, object] = {
        "calibration_manifest": shards.manifest_sha256,
        "calibration_shard_manifests": shards.shard_manifest_sha256,
        "code_manifest": sha256_file(code_manifest_path),
        "config": sha256_file(args.config),
        "model_manifest": sha256_file(model_manifest_path),
    }
    grid_path = args.out / "grid.json"
    grid_temporary = args.out / ".grid.json.tmp"
    write_canonical_json(
        grid_temporary,
        {
            "candidates": rows,
            "grid_policy": "fixed_prefix_for_reduced_runs"
            if args.grid_limit
            else "fixed_full_grid",
            "logits_sha256": _array_sha256(logits),
            "shots": shots,
            "source_role": "calibration",
            "source_sha256": source_sha256,
        },
    )
    os.replace(grid_temporary, grid_path)
    selected_sources = {**source_sha256, "grid": sha256_file(grid_path)}
    selected_temporary = args.out / ".selected.json.tmp"
    write_canonical_json(
        selected_temporary,
        {
            "complete": True,
            "selection_rule": [
                "invalid_count",
                "block_errors",
                "nll",
                "alpha",
                "beta",
                "temperature",
            ],
            "selected": {
                "residual": _selected_payload(select_calibration(residual_scores)),
                "soft_prior": _selected_payload(select_calibration(soft_scores)),
            },
            "source_role": "calibration",
            "source_sha256": selected_sources,
        },
    )
    os.replace(selected_temporary, selected_path)


if __name__ == "__main__":
    main()
