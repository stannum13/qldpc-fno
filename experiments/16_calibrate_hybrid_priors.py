from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.local import resolve_git_commit
from qldpc_fno.campaign.shard_io import (
    load_campaign_code,
    load_verified_shards,
    load_verified_teacher_artifact,
)
from qldpc_fno.data.conditional_fields import add_noise_channel
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.decoders.hybrid import decode_residual_batch, decode_soft_prior_batch
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.training.calibration import (
    CALIBRATION_GRID,
    CalibrationParameters,
    CalibrationScore,
    calibrated_probabilities,
    validate_calibration_progress_rows,
)


def _git_commit() -> str:
    return resolve_git_commit(Path(__file__).resolve().parents[1])


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


def _selection_key(row: dict[str, object], method: str) -> tuple[object, ...]:
    score = row[method]
    parameters = row["parameters"]
    if not isinstance(score, dict) or not isinstance(parameters, dict):
        raise TypeError("calibration candidate row is malformed")
    return (
        int(int(score["invalid_count"]) != 0),
        int(score["block_errors"]),
        float(score["nll"]),
        float(row["inference_latency_seconds"]),
        int(row["model_epoch"]),
        float(parameters["alpha"]),
        float(parameters["beta"]),
        float(parameters["temperature"]),
    )


def _selected_payload(row: dict[str, object], method: str) -> dict[str, object]:
    score = row[method]
    if not isinstance(score, dict):
        raise TypeError("calibration candidate score is malformed")
    return {
        **score,
        "inference_latency_seconds": row["inference_latency_seconds"],
        "model_checkpoint": row["model_checkpoint"],
        "model_epoch": row["model_epoch"],
        "parameters": row["parameters"],
    }


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
    parser.add_argument("--max-work-units-this-run", type=int)
    parser.add_argument("--resume", action="store_true")
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
    if args.max_work_units_this_run is not None and args.max_work_units_this_run <= 0:
        raise ValueError("max-work-units-this-run must be positive")
    progress_path = args.out / "progress.json"
    if progress_path.exists() and not args.resume:
        raise FileExistsError(f"calibration progress requires --resume: {progress_path}")
    if args.resume and not progress_path.is_file():
        raise FileNotFoundError(f"calibration resume progress is missing: {progress_path}")

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
    teacher = model_manifest.get("teacher")
    if not isinstance(teacher, dict):
        raise TypeError("model manifest is missing teacher provenance")
    model_split = model_manifest.get("split")
    if not isinstance(model_split, dict):
        raise TypeError("model manifest is missing split provenance")
    train_shots = model_split.get("train_shots")
    validation_shots = model_split.get("validation_shots")
    if (
        type(train_shots) is not int
        or train_shots <= 0
        or type(validation_shots) is not int
        or validation_shots <= 0
    ):
        raise ValueError("model split shot counts must be positive integers")
    teacher_metadata = load_verified_teacher_artifact(
        args.model,
        expected_metadata_sha256=str(teacher["metadata_sha256"]),
        expected_shots=train_shots + validation_shots,
    )
    if teacher_metadata.get("sha256") != teacher.get("corrections_sha256"):
        raise ValueError("model teacher correction SHA-256 disagrees with teacher metadata")
    model_path = args.model / "model.pt"
    verify_sha256(model_path, str(model_manifest["sha256"]), label="model")
    final_model = torch.load(model_path, map_location="cpu", weights_only=True)
    if final_model.get("configuration") != model_manifest.get("configuration"):
        raise ValueError("model configuration disagrees with model.json")
    checkpoint_candidates = model_manifest.get("checkpoints")
    if not isinstance(checkpoint_candidates, list) or not checkpoint_candidates:
        raise ValueError("model manifest does not declare epoch checkpoint candidates")
    candidate_epochs = [int(candidate["epoch"]) for candidate in checkpoint_candidates]
    if len(set(candidate_epochs)) != len(candidate_epochs):
        raise ValueError("model manifest declares duplicate checkpoint epochs")
    expected_identity = {**model_sources, "git_commit": model_manifest["git_commit"]}

    source_sha256: dict[str, object] = {
        "calibration_manifest": shards.manifest_sha256,
        "calibration_shard_manifests": shards.shard_manifest_sha256,
        "code_manifest": sha256_file(code_manifest_path),
        "config": sha256_file(args.config),
        "model_manifest": sha256_file(model_manifest_path),
    }
    grid_policy = "fixed_prefix_for_reduced_runs" if args.grid_limit else "fixed_full_grid"
    total_work_units = len(checkpoint_candidates) * len(candidates)
    shots = shards.shots
    rows: list[dict[str, object]]
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if not isinstance(progress, dict):
            raise TypeError("calibration progress must be a JSON object")
        if (
            progress.get("source_sha256") != source_sha256
            or progress.get("grid_policy") != grid_policy
            or progress.get("total_work_units") != total_work_units
        ):
            raise ValueError("calibration progress provenance does not match this run")
        progress_rows = progress.get("candidates")
        if progress.get("completed_work_units") != (
            len(progress_rows) if isinstance(progress_rows, list) else None
        ):
            raise ValueError("calibration progress work-unit count is inconsistent")
        rows = validate_calibration_progress_rows(
            progress_rows,
            checkpoint_candidates=checkpoint_candidates,
            candidates=candidates,
            shots=shots,
        )
    else:
        rows = []

    model_inputs = np.empty((shots, 22, 45), dtype=np.float32)
    syndromes = np.empty((shots, 945), dtype=np.uint8)
    actual_errors = np.empty((shots, 2610), dtype=np.uint8)
    actual_observables = np.empty((shots, 744), dtype=np.uint8)
    error_rates = np.empty(shots, dtype=np.float64)
    for offset in range(0, shots, config.training_batch_size):
        indices = np.arange(offset, min(offset + config.training_batch_size, shots), dtype=np.int64)
        syndrome_batch = shards.read("dets.b8", indices)
        rate_batch = shards.error_rates(indices)
        fields = to_ring_field(syndrome_batch, channels=21, ell=45)
        model_inputs[indices] = add_noise_channel(fields, rate_batch)
        syndromes[indices] = syndrome_batch
        actual_errors[indices] = shards.read("errors.b8", indices)
        actual_observables[indices] = shards.read("obs_actual.b8", indices)
        error_rates[indices] = rate_batch

    completed_at_start = len(rows)
    completed_this_run = 0
    stop_requested = False
    for checkpoint_index, candidate in enumerate(checkpoint_candidates):
        if not isinstance(candidate, dict):
            raise TypeError("model checkpoint candidate must be an object")
        first_work_index = checkpoint_index * len(candidates)
        if first_work_index + len(candidates) <= completed_at_start:
            continue
        epoch = int(candidate["epoch"])
        checkpoint_path = args.model / str(candidate["path"])
        verify_sha256(
            checkpoint_path,
            str(candidate["sha256"]),
            label=f"epoch {epoch} checkpoint",
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if int(checkpoint.get("epoch", -1)) != epoch:
            raise ValueError("checkpoint epoch disagrees with model manifest")
        if (
            checkpoint.get("identity") != expected_identity
            or checkpoint.get("split") != model_manifest.get("split")
            or checkpoint.get("teacher_metadata_sha256") != teacher["metadata_sha256"]
        ):
            raise ValueError("checkpoint provenance disagrees with model manifest")
        model = RingFNO(**model_manifest["configuration"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.requires_grad_(False)
        model.eval()
        logits = np.empty((shots, 58, 45), dtype=np.float32)
        inference_started = perf_counter()
        with torch.no_grad():
            for offset in range(0, shots, config.training_batch_size):
                batch = slice(offset, min(offset + config.training_batch_size, shots))
                logits[batch] = model(torch.from_numpy(model_inputs[batch])).numpy()
        inference_latency = perf_counter() - inference_started
        logits_sha256 = _array_sha256(logits)
        previous_checkpoint_rows = rows[
            first_work_index : min(completed_at_start, first_work_index + len(candidates))
        ]
        if previous_checkpoint_rows:
            previous_logits = {
                row.get("logits_sha256")
                for row in previous_checkpoint_rows
                if isinstance(row, dict)
            }
            previous_latencies = {
                row.get("inference_latency_seconds")
                for row in previous_checkpoint_rows
                if isinstance(row, dict)
            }
            if previous_logits != {logits_sha256} or len(previous_latencies) != 1:
                raise ValueError("resumed checkpoint inference provenance is inconsistent")
            stored_latency = next(iter(previous_latencies))
            if type(stored_latency) not in (int, float) or stored_latency < 0:
                raise ValueError("resumed checkpoint inference latency is invalid")
            inference_latency = float(stored_latency)

        for parameter_index, parameters in enumerate(candidates):
            work_index = first_work_index + parameter_index
            if work_index < completed_at_start:
                continue
            if (
                args.max_work_units_this_run is not None
                and completed_this_run >= args.max_work_units_this_run
            ):
                stop_requested = True
                break
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
            rows.append(
                {
                    "inference_latency_seconds": inference_latency,
                    "logits_sha256": logits_sha256,
                    "model_checkpoint": {
                        "path": str(candidate["path"]),
                        "sha256": str(candidate["sha256"]),
                    },
                    "model_epoch": epoch,
                    "parameters": asdict(parameters),
                    "residual": _score_payload(residual),
                    "soft_prior": _score_payload(soft),
                }
            )
            completed_this_run += 1
            progress_temporary = args.out / ".progress.json.tmp"
            write_canonical_json(
                progress_temporary,
                {
                    "candidates": rows,
                    "completed_work_units": len(rows),
                    "grid_policy": grid_policy,
                    "source_role": "calibration",
                    "source_sha256": source_sha256,
                    "total_work_units": total_work_units,
                },
            )
            os.replace(progress_temporary, progress_path)
        if stop_requested:
            break

    if len(rows) < total_work_units:
        return
    grid_path = args.out / "grid.json"
    grid_temporary = args.out / ".grid.json.tmp"
    write_canonical_json(
        grid_temporary,
        {
            "candidates": rows,
            "grid_policy": grid_policy,
            "logit_evaluations": len(checkpoint_candidates),
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
                "has_any_invalid_correction",
                "block_errors",
                "nll",
                "inference_latency_seconds",
                "model_epoch",
                "alpha",
                "beta",
                "temperature",
            ],
            "selected": {
                "residual": _selected_payload(
                    min(rows, key=lambda row: _selection_key(row, "residual")),
                    "residual",
                ),
                "soft_prior": _selected_payload(
                    min(rows, key=lambda row: _selection_key(row, "soft_prior")),
                    "soft_prior",
                ),
            },
            "source_role": "calibration",
            "source_sha256": selected_sources,
        },
    )
    os.replace(selected_temporary, selected_path)


if __name__ == "__main__":
    main()
