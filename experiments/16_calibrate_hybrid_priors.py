from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
    calibration_shortlists,
    deterministic_calibration_subset,
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
        "screening_proxy": row["screening_proxy"],
        "work_index": row["work_index"],
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


def _screen_candidate(
    *,
    parameters: CalibrationParameters,
    logits: np.ndarray,
    syndromes: np.ndarray,
    actual_errors: np.ndarray,
    error_rates: np.ndarray,
    hx: object,
    batch_size: int,
) -> dict[str, object]:
    """Compute cheap calibration-only proxies without invoking either BP-LSD hybrid."""
    nll_sum = 0.0
    proposal_invalid_count = 0
    residual_weight_sum = 0
    shots = logits.shape[0]
    for offset in range(0, shots, batch_size):
        batch = slice(offset, min(offset + batch_size, shots))
        probabilities = calibrated_probabilities(logits[batch], error_rates[batch], parameters)
        flat_probabilities = probabilities.reshape(probabilities.shape[0], -1)
        nll_sum += _probability_nll(flat_probabilities, actual_errors[batch]) * len(
            flat_probabilities
        )
        proposal = (flat_probabilities >= 0.5).astype(np.uint8)
        proposal_syndromes = np.asarray((hx @ proposal.T).T, dtype=np.uint8) % 2
        residual = syndromes[batch] ^ proposal_syndromes
        residual_weights = np.count_nonzero(residual, axis=1)
        proposal_invalid_count += int(np.count_nonzero(residual_weights))
        residual_weight_sum += int(residual_weights.sum())
    return {
        "mean_residual_syndrome_weight": residual_weight_sum / shots,
        "nll": nll_sum / shots,
        "proposal_invalid_count": proposal_invalid_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--grid-limit", type=int)
    parser.add_argument(
        "--campaign-mode",
        choices=("canonical", "reduced_non_scientific"),
        default="canonical",
    )
    parser.add_argument("--max-work-units-this-run", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    if args.campaign_mode == "canonical" and args.grid_limit is not None:
        raise ValueError("canonical calibration cannot reduce the fixed full grid")
    if args.campaign_mode == "reduced_non_scientific" and args.grid_limit is None:
        raise ValueError("reduced calibration requires an explicit grid-limit")
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
    grid_policy = (
        "fixed_prefix_for_reduced_runs"
        if args.campaign_mode == "reduced_non_scientific"
        else "fixed_full_grid"
    )
    shots = shards.shots
    rates = sorted({shard.rate_index for shard in shards.shards})
    decode_indices, decode_subset = deterministic_calibration_subset(
        {rate_index: shards.indices_for_rate(rate_index) for rate_index in rates},
        max_shots=config.calibration_decode_shots_cap,
        seed=config.campaign_seed,
    )
    shortlist_size = min(config.calibration_shortlist_per_method, len(candidates))
    policy: dict[str, object] = {
        "decode_subset": decode_subset,
        "grid_policy": grid_policy,
        "screening_candidate_count": len(candidates),
        "screening_checkpoint_count": len(checkpoint_candidates),
        "screening_proxy": [
            "correction_nll",
            "threshold_proposal_invalid_count",
            "mean_residual_syndrome_weight",
        ],
        "screening_shots": shots,
        "shortlist_per_method": shortlist_size,
    }
    screening_rows: list[dict[str, object]] = []
    hybrid_rows: list[dict[str, object]] = []
    shortlists: dict[str, list[int]] | None = None
    decode_work_indices: list[int] | None = None
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if not isinstance(progress, dict):
            raise TypeError("calibration progress must be a JSON object")
        expected_fields = {
            "completed_work_units",
            "decode_work_indices",
            "hybrid_candidates",
            "policy",
            "screening_candidates",
            "shortlists",
            "source_role",
            "source_sha256",
            "total_work_units",
        }
        if set(progress) != expected_fields:
            raise ValueError("calibration progress schema does not match the two-stage policy")
        if (
            progress.get("source_role") != "calibration"
            or progress.get("source_sha256") != source_sha256
            or progress.get("policy") != policy
        ):
            raise ValueError("calibration progress provenance does not match this run")
        raw_screening = progress.get("screening_candidates")
        raw_hybrid = progress.get("hybrid_candidates")
        if not isinstance(raw_screening, list) or not isinstance(raw_hybrid, list):
            raise TypeError("calibration progress candidate tables must be lists")
        screening_rows = raw_screening
        hybrid_rows = raw_hybrid
        if len(screening_rows) % len(candidates) != 0:
            raise ValueError("calibration progress screening is not a checkpoint-aligned prefix")
        if len(screening_rows) > len(checkpoint_candidates) * len(candidates):
            raise ValueError("calibration progress screening exceeds the declared grid")
        screening_fields = {
            "inference_latency_seconds",
            "logits_sha256",
            "mean_residual_syndrome_weight",
            "model_checkpoint",
            "model_epoch",
            "nll",
            "parameters",
            "proposal_invalid_count",
            "work_index",
        }
        checkpoint_measurements: dict[int, tuple[str, float]] = {}
        for work_index, row in enumerate(screening_rows):
            if not isinstance(row, dict) or set(row) != screening_fields:
                raise ValueError("calibration progress screening row is malformed")
            checkpoint_index = work_index // len(candidates)
            checkpoint = checkpoint_candidates[checkpoint_index]
            parameters = candidates[work_index % len(candidates)]
            numeric = (
                row.get("inference_latency_seconds"),
                row.get("mean_residual_syndrome_weight"),
                row.get("nll"),
            )
            if (
                row.get("work_index") != work_index
                or row.get("model_epoch") != int(checkpoint["epoch"])
                or row.get("model_checkpoint")
                != {"path": str(checkpoint["path"]), "sha256": str(checkpoint["sha256"])}
                or row.get("parameters") != asdict(parameters)
                or not isinstance(row.get("logits_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", str(row["logits_sha256"])) is None
                or any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in numeric)
                or type(row.get("proposal_invalid_count")) is not int
                or not 0 <= int(row["proposal_invalid_count"]) <= shots
                or float(row["mean_residual_syndrome_weight"]) > hx.shape[0]
            ):
                raise ValueError("calibration progress screening order or measurement is invalid")
            measurement = (
                str(row["logits_sha256"]),
                float(row["inference_latency_seconds"]),
            )
            previous_measurement = checkpoint_measurements.setdefault(
                checkpoint_index, measurement
            )
            if previous_measurement != measurement:
                raise ValueError("calibration progress checkpoint measurements diverge")
        raw_shortlists = progress.get("shortlists")
        raw_decode_work = progress.get("decode_work_indices")
        if raw_shortlists is None or raw_decode_work is None:
            if raw_shortlists is not None or raw_decode_work is not None:
                raise ValueError("calibration progress shortlist transition is inconsistent")
            if hybrid_rows:
                raise ValueError("calibration progress decoded before screening completed")
        else:
            expected_shortlists = calibration_shortlists(
                screening_rows,
                per_method=shortlist_size,
            )
            expected_decode_work = sorted(
                set(expected_shortlists["soft_prior"]) | set(expected_shortlists["residual"])
            )
            if raw_shortlists != expected_shortlists or raw_decode_work != expected_decode_work:
                raise ValueError("calibration progress shortlist provenance is inconsistent")
            shortlists = expected_shortlists
            decode_work_indices = expected_decode_work
            if len(hybrid_rows) > len(decode_work_indices):
                raise ValueError("calibration progress hybrid table exceeds the shortlist union")
            for offset, row in enumerate(hybrid_rows):
                if not isinstance(row, dict) or row.get("work_index") != decode_work_indices[offset]:
                    raise ValueError("calibration progress hybrid rows are not an ordered prefix")
                if set(row) != {
                    "inference_latency_seconds",
                    "logits_sha256",
                    "model_checkpoint",
                    "model_epoch",
                    "parameters",
                    "residual",
                    "screening_proxy",
                    "soft_prior",
                    "work_index",
                }:
                    raise ValueError("calibration progress hybrid row is malformed")
                screening = screening_rows[decode_work_indices[offset]]
                if (
                    not isinstance(row.get("logits_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", str(row["logits_sha256"])) is None
                    or type(row.get("inference_latency_seconds")) not in (int, float)
                    or not np.isfinite(row["inference_latency_seconds"])
                    or float(row["inference_latency_seconds"]) < 0
                ):
                    raise ValueError("calibration progress hybrid measurement is invalid")
                for field in ("model_checkpoint", "model_epoch", "parameters"):
                    if row.get(field) != screening.get(field):
                        raise ValueError("calibration progress hybrid provenance diverges")
                if row.get("screening_proxy") != {
                    "mean_residual_syndrome_weight": screening["mean_residual_syndrome_weight"],
                    "nll": screening["nll"],
                    "proposal_invalid_count": screening["proposal_invalid_count"],
                }:
                    raise ValueError("calibration progress screening proxy diverges")
                score_nlls: list[float] = []
                for method in ("soft_prior", "residual"):
                    score = row.get(method)
                    if not isinstance(score, dict) or set(score) != {
                        "block_errors",
                        "invalid_count",
                        "nll",
                    }:
                        raise ValueError(
                            f"calibration progress hybrid {method} score is malformed"
                        )
                    invalid_count = score["invalid_count"]
                    block_errors = score["block_errors"]
                    nll = score["nll"]
                    if (
                        type(invalid_count) is not int
                        or type(block_errors) is not int
                        or not 0 <= invalid_count <= block_errors <= len(decode_indices)
                        or type(nll) not in (int, float)
                        or not np.isfinite(nll)
                        or float(nll) < 0
                    ):
                        raise ValueError(
                            f"calibration progress hybrid {method} score is invalid"
                        )
                    score_nlls.append(float(nll))
                if score_nlls[0] != score_nlls[1]:
                    raise ValueError("calibration progress hybrid method NLL values diverge")
        screened_checkpoints = len(screening_rows) // len(candidates)
        expected_total = len(checkpoint_candidates) + (
            len(decode_work_indices)
            if decode_work_indices is not None
            else min(2 * shortlist_size, len(checkpoint_candidates) * len(candidates))
        )
        if (
            progress.get("completed_work_units") != screened_checkpoints + len(hybrid_rows)
            or progress.get("total_work_units") != expected_total
        ):
            raise ValueError("calibration progress work-unit count is inconsistent")

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

    completed_this_run = 0

    def write_progress() -> None:
        screened_checkpoints = len(screening_rows) // len(candidates)
        total = len(checkpoint_candidates) + (
            len(decode_work_indices)
            if decode_work_indices is not None
            else min(2 * shortlist_size, len(checkpoint_candidates) * len(candidates))
        )
        progress_temporary = args.out / ".progress.json.tmp"
        write_canonical_json(
            progress_temporary,
            {
                "completed_work_units": screened_checkpoints + len(hybrid_rows),
                "decode_work_indices": decode_work_indices,
                "hybrid_candidates": hybrid_rows,
                "policy": policy,
                "screening_candidates": screening_rows,
                "shortlists": shortlists,
                "source_role": "calibration",
                "source_sha256": source_sha256,
                "total_work_units": total,
            },
        )
        os.replace(progress_temporary, progress_path)

    def limit_reached() -> bool:
        return (
            args.max_work_units_this_run is not None
            and completed_this_run >= args.max_work_units_this_run
        )

    def load_checkpoint_model(candidate: dict[str, object]) -> tuple[int, RingFNO]:
        epoch = int(candidate["epoch"])
        checkpoint_path = args.model / str(candidate["path"])
        verify_sha256(checkpoint_path, str(candidate["sha256"]), label=f"epoch {epoch} checkpoint")
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
        return epoch, model

    screened_checkpoints = len(screening_rows) // len(candidates)
    for checkpoint_index, candidate in enumerate(checkpoint_candidates):
        if not isinstance(candidate, dict):
            raise TypeError("model checkpoint candidate must be an object")
        if checkpoint_index < screened_checkpoints:
            continue
        if limit_reached():
            return
        first_work_index = checkpoint_index * len(candidates)
        epoch, model = load_checkpoint_model(candidate)
        logits = np.empty((shots, 58, 45), dtype=np.float32)
        inference_started = perf_counter()
        with torch.no_grad():
            for offset in range(0, shots, config.training_batch_size):
                batch = slice(offset, min(offset + config.training_batch_size, shots))
                logits[batch] = model(torch.from_numpy(model_inputs[batch])).numpy()
        inference_latency = perf_counter() - inference_started
        logits_sha256 = _array_sha256(logits)
        for parameter_index, parameters in enumerate(candidates):
            work_index = first_work_index + parameter_index
            proxy = _screen_candidate(
                parameters=parameters,
                logits=logits,
                syndromes=syndromes,
                actual_errors=actual_errors,
                error_rates=error_rates,
                hx=hx,
                batch_size=config.training_batch_size,
            )
            screening_rows.append(
                {
                    "inference_latency_seconds": inference_latency,
                    "logits_sha256": logits_sha256,
                    "model_checkpoint": {
                        "path": str(candidate["path"]),
                        "sha256": str(candidate["sha256"]),
                    },
                    "model_epoch": epoch,
                    "parameters": asdict(parameters),
                    **proxy,
                    "work_index": work_index,
                }
            )
        completed_this_run += 1
        write_progress()

    if shortlists is None:
        shortlists = calibration_shortlists(screening_rows, per_method=shortlist_size)
        decode_work_indices = sorted(set(shortlists["soft_prior"]) | set(shortlists["residual"]))
        write_progress()
    if decode_work_indices is None:
        raise RuntimeError("calibration shortlist union was not established")

    for work_index in decode_work_indices[len(hybrid_rows) :]:
        if limit_reached():
            return
        screening = screening_rows[work_index]
        candidate = checkpoint_candidates[work_index // len(candidates)]
        parameters = candidates[work_index % len(candidates)]
        epoch, model = load_checkpoint_model(candidate)
        subset_inputs = model_inputs[decode_indices]
        inference_started = perf_counter()
        with torch.no_grad():
            subset_logits = model(torch.from_numpy(subset_inputs)).numpy()
        inference_latency = perf_counter() - inference_started
        soft, residual = _score_candidate(
            parameters=parameters,
            logits=subset_logits,
            syndromes=syndromes[decode_indices],
            actual_errors=actual_errors[decode_indices],
            actual_observables=actual_observables[decode_indices],
            error_rates=error_rates[decode_indices],
            hx=hx,
            logical_x=logical_x,
            batch_size=config.training_batch_size,
        )
        hybrid_rows.append(
            {
                "inference_latency_seconds": inference_latency,
                "logits_sha256": _array_sha256(subset_logits),
                "model_checkpoint": screening["model_checkpoint"],
                "model_epoch": epoch,
                "parameters": asdict(parameters),
                "residual": _score_payload(residual),
                "screening_proxy": {
                    "mean_residual_syndrome_weight": screening[
                        "mean_residual_syndrome_weight"
                    ],
                    "nll": screening["nll"],
                    "proposal_invalid_count": screening["proposal_invalid_count"],
                },
                "soft_prior": _score_payload(soft),
                "work_index": work_index,
            }
        )
        completed_this_run += 1
        write_progress()

    grid_path = args.out / "grid.json"
    grid_temporary = args.out / ".grid.json.tmp"
    write_canonical_json(
        grid_temporary,
        {
            "hybrid_candidates": hybrid_rows,
            "logit_evaluations": len(checkpoint_candidates),
            "policy": policy,
            "screening_candidates": screening_rows,
            "shortlists": shortlists,
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
                "model_epoch",
                "alpha",
                "beta",
                "temperature",
            ],
            "selected": {
                "residual": _selected_payload(
                    min(
                        (
                            row
                            for row in hybrid_rows
                            if row["work_index"] in shortlists["residual"]
                        ),
                        key=lambda row: _selection_key(row, "residual"),
                    ),
                    "residual",
                ),
                "soft_prior": _selected_payload(
                    min(
                        (
                            row
                            for row in hybrid_rows
                            if row["work_index"] in shortlists["soft_prior"]
                        ),
                        key=lambda row: _selection_key(row, "soft_prior"),
                    ),
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
