from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import monotonic, perf_counter

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.local import resolve_git_commit
from qldpc_fno.campaign.shard_io import (
    VerifiedShardSet,
    load_campaign_code,
    load_verified_shards,
    load_verified_teacher_artifact,
)
from qldpc_fno.data.conditional_fields import add_noise_channel
from qldpc_fno.data.ring_fields import to_ring_field
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.decoders.hybrid import decode_residual_batch, decode_soft_prior_batch
from qldpc_fno.metrics.paired import (
    paired_comparison_status,
    paired_decoder_summary,
    test_stop_reason,
)
from qldpc_fno.models.fno1d import RingFNO
from qldpc_fno.training.calibration import (
    CALIBRATION_GRID,
    CalibrationParameters,
    calibrated_probabilities,
    deterministic_calibration_subset,
    validate_two_stage_calibration,
)

_DECODERS = ("baseline", "soft_prior", "residual")
_HYBRIDS = ("soft_prior", "residual")
_MODEL_CONFIGURATION = {
    "depth": 2,
    "in_channels": 22,
    "modes": 12,
    "out_channels": 58,
    "width": 32,
}
_SELECTION_RULE = [
    "has_any_invalid_correction",
    "block_errors",
    "nll",
    "model_epoch",
    "alpha",
    "beta",
    "temperature",
]
_BOOLEAN_OUTCOME_FIELDS = {
    f"{decoder}_{suffix}"
    for decoder in _DECODERS
    for suffix in ("converged", "failure", "logical_mismatch", "syndrome_valid")
}
_INTEGER_OUTCOME_FIELDS = {
    f"{decoder}_{suffix}" for decoder in _DECODERS for suffix in ("correction_weight", "iterations")
} | {
    "residual_delta_weight",
    "residual_proposal_weight",
    "residual_syndrome_weight",
    "shot_indices",
}
_FLOAT_OUTCOME_FIELDS = {f"{decoder}_latency_seconds" for decoder in _DECODERS} | {
    f"{method}_{suffix}_latency_seconds"
    for method in _HYBRIDS
    for suffix in ("bp", "end_to_end", "fno", "preprocessing")
}
_OUTCOME_FIELDS = _BOOLEAN_OUTCOME_FIELDS | _INTEGER_OUTCOME_FIELDS | _FLOAT_OUTCOME_FIELDS


@dataclass(frozen=True, slots=True)
class _SelectedModel:
    model: RingFNO
    parameters: CalibrationParameters
    checkpoint: dict[str, object]
    epoch: int


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    """Verified inputs and bounded controls for one adaptive evaluation run."""

    config: Path
    code: Path
    selection: Path
    test: Path
    model: Path
    calibration: Path
    out: Path
    campaign_mode: str = "canonical"
    bootstrap_samples: int = 10_000
    deadline_monotonic: float | None = None
    max_batches_this_run: int | None = None
    resume: bool = False


def _git_commit() -> str:
    return resolve_git_commit(Path(__file__).resolve().parents[3])


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    write_canonical_json(temporary, payload)
    os.replace(temporary, path)


def _verify_selection(
    selection_path: Path,
    *,
    config_path: Path,
    code_manifest_path: Path,
) -> tuple[float, ...]:
    pilot_manifest_path = selection_path.parent / "manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text())
    if pilot_manifest.get("complete") is not True or pilot_manifest.get("role") != "pilot":
        raise ValueError("selection does not belong to a completed pilot publication")
    verify_sha256(selection_path, str(pilot_manifest["selection_sha256"]), label="selection")
    selection = json.loads(selection_path.read_text())
    sources = selection.get("source_sha256")
    if not isinstance(sources, dict):
        raise TypeError("selection is missing source SHA-256 provenance")
    if sources.get("config") != sha256_file(config_path):
        raise ValueError("selection configuration provenance mismatch")
    if sources.get("code_manifest") != sha256_file(code_manifest_path):
        raise ValueError("selection code provenance mismatch")
    config = CampaignConfig.from_json(config_path)
    if selection.get("selection_mode") != config.selection_mode:
        raise ValueError("selection mode does not match campaign configuration")
    evidence_roles = {
        "fixed": "predeclared_selection_not_evidence",
        "pilot": "selection_only_not_held_out",
    }
    if selection.get("evidence_role") != evidence_roles[config.selection_mode]:
        raise ValueError("selection evidence role does not match campaign configuration")
    raw_rates = selection.get("selected_noise_points")
    if not isinstance(raw_rates, list) or not raw_rates:
        raise ValueError("selection must contain non-empty selected_noise_points")
    if config.selection_mode == "fixed" and tuple(raw_rates) != config.noise_grid:
        raise ValueError("fixed selection rates do not match configured noise_grid")
    if any(type(rate) not in (int, float) or not math.isfinite(rate) for rate in raw_rates):
        raise ValueError("selected noise points must be finite numbers")
    rates = tuple(float(rate) for rate in raw_rates)
    if any(not 0.0 < rate < 0.5 for rate in rates) or any(
        left >= right for left, right in pairwise(rates)
    ):
        raise ValueError("selected noise points must be strictly increasing probabilities")
    return rates


def _verify_shard_selection(
    shards: VerifiedShardSet,
    *,
    rates: tuple[float, ...],
    selection_path: Path,
) -> None:
    rate_map: dict[int, float] = {}
    expected_sources = {
        "code_manifest": None,
        "config": None,
        "dem": None,
        "pilot_manifest": sha256_file(selection_path.parent / "manifest.json"),
        "selection": sha256_file(selection_path),
    }
    for shard in shards.shards:
        previous = rate_map.setdefault(shard.rate_index, shard.error_rate)
        if previous != shard.error_rate:
            raise ValueError(f"{shards.role} shards have inconsistent rate coordinates")
        manifest = json.loads(shard.manifest_path.read_text())
        sources = manifest.get("source_sha256")
        if not isinstance(sources, dict) or set(sources) != set(expected_sources):
            raise ValueError(f"{shards.role} shard source provenance fields are incomplete")
        for key in ("pilot_manifest", "selection"):
            if sources.get(key) != expected_sources[key]:
                raise ValueError(f"{shards.role} shard {key} provenance mismatch")
    if tuple(rate_map[index] for index in sorted(rate_map)) != rates:
        raise ValueError(f"{shards.role} shard rates do not match the pilot selection")


def _model_sources(
    model_manifest: dict[str, object],
    *,
    config_path: Path,
    code_manifest_path: Path,
) -> dict[str, object]:
    if model_manifest.get("complete") is not True or model_manifest.get("source_role") != "train":
        raise ValueError("model is not a completed train-role publication")
    if model_manifest.get("git_commit") != _git_commit():
        raise ValueError("model Git commit does not match the evaluation executable")
    sources = model_manifest.get("source_sha256")
    expected_fields = {
        "code_manifest",
        "config",
        "train_manifest",
        "train_shard_manifests",
    }
    if not isinstance(sources, dict) or set(sources) != expected_fields:
        raise ValueError("model source provenance fields are incomplete")
    if sources.get("config") != sha256_file(config_path):
        raise ValueError("model configuration provenance mismatch")
    if sources.get("code_manifest") != sha256_file(code_manifest_path):
        raise ValueError("model code provenance mismatch")
    if not isinstance(sources.get("train_manifest"), str) or not isinstance(
        sources.get("train_shard_manifests"), dict
    ):
        raise TypeError("model train provenance must contain manifest SHA-256 values")
    if any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in sources["train_shard_manifests"].items()
    ):
        raise TypeError("model train shard provenance must map paths to SHA-256 values")
    return sources


def _selected_payload_from_grid(row: dict[str, object], method: str) -> dict[str, object]:
    score = row.get(method)
    if not isinstance(score, dict):
        raise TypeError("calibration grid candidate score is malformed")
    return {
        **score,
        "inference_latency_seconds": row.get("inference_latency_seconds"),
        "model_checkpoint": row.get("model_checkpoint"),
        "model_epoch": row.get("model_epoch"),
        "parameters": row.get("parameters"),
        "screening_proxy": row.get("screening_proxy"),
        "work_index": row.get("work_index"),
    }


def _load_selected_models(
    *,
    config_path: Path,
    code_manifest_path: Path,
    model_dir: Path,
    calibration_dir: Path,
    calibration_shards: VerifiedShardSet,
    campaign_mode: str,
    syndrome_checks: int,
) -> tuple[dict[str, _SelectedModel], dict[str, object]]:
    config = CampaignConfig.from_json(config_path)
    model_manifest_path = model_dir / "model.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    sources = _model_sources(
        model_manifest,
        config_path=config_path,
        code_manifest_path=code_manifest_path,
    )
    if model_manifest.get("configuration") != _MODEL_CONFIGURATION:
        raise ValueError("model configuration is not the pinned conditional FNO")
    split = model_manifest.get("split")
    teacher = model_manifest.get("teacher")
    if not isinstance(split, dict) or not isinstance(teacher, dict):
        raise TypeError("model is missing split or teacher provenance")
    train_shots = split.get("train_shots")
    validation_shots = split.get("validation_shots")
    if (
        type(train_shots) is not int
        or train_shots <= 0
        or type(validation_shots) is not int
        or validation_shots <= 0
    ):
        raise ValueError("model split shot counts must be positive integers")
    teacher_metadata = load_verified_teacher_artifact(
        model_dir,
        expected_metadata_sha256=str(teacher["metadata_sha256"]),
        expected_shots=train_shots + validation_shots,
    )
    if teacher_metadata.get("sha256") != teacher.get("corrections_sha256"):
        raise ValueError("model teacher correction provenance mismatch")
    model_path = model_dir / "model.pt"
    verify_sha256(model_path, str(model_manifest["sha256"]), label="model")
    final_model = torch.load(model_path, map_location="cpu", weights_only=True)
    if final_model.get("configuration") != _MODEL_CONFIGURATION:
        raise ValueError("model.pt configuration disagrees with model.json")
    final_state = final_model.get("state_dict")
    if not isinstance(final_state, dict):
        raise TypeError("model.pt is missing a state dictionary")
    verified_final = RingFNO(**_MODEL_CONFIGURATION)
    verified_final.load_state_dict(final_state)

    selected_path = calibration_dir / "selected.json"
    selected = json.loads(selected_path.read_text())
    if not isinstance(selected, dict) or set(selected) != {
        "complete",
        "selected",
        "selection_rule",
        "source_role",
        "source_sha256",
    }:
        raise ValueError("calibration selection publication schema is malformed")
    if selected.get("complete") is not True or selected.get("source_role") != "calibration":
        raise ValueError("calibration is not a completed calibration-role publication")
    if selected.get("selection_rule") != _SELECTION_RULE:
        raise ValueError("calibration selection rule does not match the declared policy")
    selected_sources = selected.get("source_sha256")
    expected_source_fields = {
        "calibration_manifest",
        "calibration_shard_manifests",
        "code_manifest",
        "config",
        "grid",
        "model_manifest",
    }
    if not isinstance(selected_sources, dict) or set(selected_sources) != expected_source_fields:
        raise ValueError("calibration source provenance fields are incomplete")
    if selected_sources.get("config") != sha256_file(config_path):
        raise ValueError("calibration configuration provenance mismatch")
    if selected_sources.get("code_manifest") != sha256_file(code_manifest_path):
        raise ValueError("calibration code provenance mismatch")
    if selected_sources.get("model_manifest") != sha256_file(model_manifest_path):
        raise ValueError("calibration model provenance mismatch")
    if selected_sources.get("calibration_manifest") != calibration_shards.manifest_sha256:
        raise ValueError("calibration role-manifest provenance mismatch")
    if (
        selected_sources.get("calibration_shard_manifests")
        != calibration_shards.shard_manifest_sha256
    ):
        raise ValueError("calibration shard-manifest provenance mismatch")

    grid_path = calibration_dir / "grid.json"
    verify_sha256(grid_path, str(selected_sources["grid"]), label="calibration grid")
    grid = json.loads(grid_path.read_text())
    if not isinstance(grid, dict) or set(grid) != {
        "hybrid_candidates",
        "logit_evaluations",
        "policy",
        "screening_candidates",
        "shortlists",
        "source_role",
        "source_sha256",
    }:
        raise ValueError("calibration grid publication schema is malformed")
    grid_sources = grid.get("source_sha256")
    if grid.get("source_role") != "calibration" or not isinstance(grid_sources, dict):
        raise ValueError("calibration grid provenance is malformed")
    if grid_sources != {key: value for key, value in selected_sources.items() if key != "grid"}:
        raise ValueError("calibration grid provenance disagrees with selected.json")
    candidates = grid.get("hybrid_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("calibration grid must contain shortlisted hybrid candidate results")
    screening = grid.get("screening_candidates")
    shortlists = grid.get("shortlists")
    policy = grid.get("policy")
    if (
        not isinstance(screening, list)
        or not screening
        or not isinstance(shortlists, dict)
        or set(shortlists) != set(_HYBRIDS)
        or not isinstance(policy, dict)
    ):
        raise ValueError("calibration grid two-stage policy is malformed")
    decode_subset = policy.get("decode_subset")
    if (
        not isinstance(decode_subset, dict)
        or type(decode_subset.get("shots")) is not int
        or decode_subset["shots"] <= 0
        or not isinstance(decode_subset.get("indices_sha256"), str)
    ):
        raise ValueError("calibration decode subset provenance is malformed")
    selected_methods = selected.get("selected")
    if not isinstance(selected_methods, dict) or set(selected_methods) != set(_HYBRIDS):
        raise ValueError("calibration must select soft_prior and residual independently")

    checkpoint_candidates = model_manifest.get("checkpoints")
    if not isinstance(checkpoint_candidates, list) or not checkpoint_candidates:
        raise ValueError("model manifest does not declare checkpoint candidates")
    if campaign_mode == "canonical":
        expected_candidates = CALIBRATION_GRID
        grid_policy = "fixed_full_grid"
    elif campaign_mode == "reduced_non_scientific":
        candidate_count = policy.get("screening_candidate_count")
        if (
            type(candidate_count) is not int
            or not 0 < candidate_count <= len(CALIBRATION_GRID)
        ):
            raise ValueError("reduced calibration grid candidate count is malformed")
        expected_candidates = CALIBRATION_GRID[:candidate_count]
        grid_policy = "fixed_prefix_for_reduced_runs"
    else:
        raise ValueError("evaluation campaign mode is invalid")
    rate_indices = sorted({shard.rate_index for shard in calibration_shards.shards})
    _, expected_decode_subset = deterministic_calibration_subset(
        {
            rate_index: calibration_shards.indices_for_rate(rate_index)
            for rate_index in rate_indices
        },
        max_shots=config.calibration_decode_shots_cap,
        seed=config.campaign_seed,
    )
    shortlist_size = min(config.calibration_shortlist_per_method, len(expected_candidates))
    expected_policy = {
        "decode_subset": expected_decode_subset,
        "grid_policy": grid_policy,
        "screening_candidate_count": len(expected_candidates),
        "screening_checkpoint_count": len(checkpoint_candidates),
        "screening_proxy": [
            "correction_nll",
            "threshold_proposal_invalid_count",
            "mean_residual_syndrome_weight",
        ],
        "screening_shots": calibration_shards.shots,
        "shortlist_per_method": shortlist_size,
    }
    if policy != expected_policy:
        raise ValueError("calibration two-stage policy does not match the declared campaign mode")
    if grid.get("logit_evaluations") != len(checkpoint_candidates):
        raise ValueError("calibration grid logit evaluation count is incomplete")
    validate_two_stage_calibration(
        screening,
        candidates,
        shortlists,
        selected_methods,
        checkpoint_candidates=checkpoint_candidates,
        candidates=expected_candidates,
        screening_shots=calibration_shards.shots,
        syndrome_checks=syndrome_checks,
        decode_shots=int(expected_decode_subset["shots"]),
        shortlist_per_method=shortlist_size,
    )
    declared_checkpoints = {
        (candidate.get("path"), candidate.get("sha256"), candidate.get("epoch"))
        for candidate in checkpoint_candidates
        if isinstance(candidate, dict)
    }
    expected_identity = {**sources, "git_commit": model_manifest["git_commit"]}
    loaded: dict[str, _SelectedModel] = {}
    for method in _HYBRIDS:
        method_selection = selected_methods[method]
        if not isinstance(method_selection, dict):
            raise TypeError(f"selected {method} calibration must be an object")
        if method_selection not in [
            _selected_payload_from_grid(row, method)
            for row in candidates
            if isinstance(row, dict) and row.get("work_index") in shortlists[method]
        ]:
            raise ValueError(f"selected {method} calibration is absent from the verified grid")
        checkpoint = method_selection.get("model_checkpoint")
        epoch = method_selection.get("model_epoch")
        if not isinstance(checkpoint, dict) or type(epoch) is not int or epoch <= 0:
            raise ValueError(f"selected {method} checkpoint provenance is malformed")
        path_value = checkpoint.get("path")
        digest = checkpoint.get("sha256")
        if (
            not isinstance(path_value, str)
            or Path(path_value).name != path_value
            or not isinstance(digest, str)
            or (path_value, digest, epoch) not in declared_checkpoints
        ):
            raise ValueError(f"selected {method} checkpoint is not declared by the model")
        checkpoint_path = model_dir / path_value
        verify_sha256(checkpoint_path, digest, label=f"selected {method} checkpoint")
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint_payload.get("epoch") != epoch
            or checkpoint_payload.get("identity") != expected_identity
            or checkpoint_payload.get("split") != split
            or checkpoint_payload.get("teacher_metadata_sha256") != teacher["metadata_sha256"]
            or checkpoint_payload.get("teacher_sha256") != teacher["corrections_sha256"]
        ):
            raise ValueError(f"selected {method} checkpoint internal provenance mismatch")
        raw_parameters = method_selection.get("parameters")
        if not isinstance(raw_parameters, dict) or set(raw_parameters) != {
            "alpha",
            "beta",
            "temperature",
        }:
            raise ValueError(f"selected {method} calibration parameters are malformed")
        parameters = CalibrationParameters(
            alpha=float(raw_parameters["alpha"]),
            beta=float(raw_parameters["beta"]),
            temperature=float(raw_parameters["temperature"]),
        )
        network = RingFNO(**_MODEL_CONFIGURATION)
        network.load_state_dict(checkpoint_payload["model_state_dict"])
        network.requires_grad_(False)
        network.eval()
        loaded[method] = _SelectedModel(network, parameters, dict(checkpoint), epoch)

    provenance: dict[str, object] = {
        "calibration_grid": sha256_file(grid_path),
        "calibration_manifest": calibration_shards.manifest_sha256,
        "calibration_selected": sha256_file(selected_path),
        "calibration_shard_manifests": calibration_shards.shard_manifest_sha256,
        "code_manifest": sha256_file(code_manifest_path),
        "config": sha256_file(config_path),
        "model_manifest": sha256_file(model_manifest_path),
    }
    return loaded, provenance


def _indices_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def _distribution(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values)
    if array.size == 0:
        return {"max": None, "mean": None, "min": None, "p50": None, "p95": None}
    return {
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _paired_table(baseline: np.ndarray, hybrid: np.ndarray) -> dict[str, int]:
    return {
        "baseline_only_failure": int(np.count_nonzero(baseline & ~hybrid)),
        "both_fail": int(np.count_nonzero(baseline & hybrid)),
        "both_succeed": int(np.count_nonzero(~baseline & ~hybrid)),
        "hybrid_only_failure": int(np.count_nonzero(~baseline & hybrid)),
    }


def _probability_reliability(
    probabilities: np.ndarray,
    actual_errors: np.ndarray,
) -> list[dict[str, object]]:
    flat_probabilities = probabilities.reshape(-1)
    flat_errors = actual_errors.reshape(-1)
    bin_indices = np.minimum((flat_probabilities * 10).astype(np.int64), 9)
    rows: list[dict[str, object]] = []
    for bin_index in range(10):
        selected = bin_indices == bin_index
        rows.append(
            {
                "bin_high": (bin_index + 1) / 10,
                "bin_low": bin_index / 10,
                "count": int(np.count_nonzero(selected)),
                "observed_errors": int(np.count_nonzero(flat_errors[selected])),
                "probability_sum": float(np.sum(flat_probabilities[selected], dtype=np.float64)),
            }
        )
    return rows


def _write_outcomes_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _decode_batch(
    *,
    indices: np.ndarray,
    rate: float,
    shards: VerifiedShardSet,
    hx: object,
    logical_x: object,
    selected_models: dict[str, _SelectedModel],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, list[dict[str, object]]],
]:
    syndromes = shards.read("dets.b8", indices)
    actual_observables = shards.read("obs_actual.b8", indices)
    actual_errors = shards.read("errors.b8", indices)
    error_rates = np.full(indices.size, rate, dtype=np.float64)
    inputs = add_noise_channel(to_ring_field(syndromes, channels=21, ell=45), error_rates)

    probabilities: dict[str, np.ndarray] = {}
    inference_latency: dict[str, float] = {}
    with torch.no_grad():
        for method in _HYBRIDS:
            started = perf_counter()
            logits = selected_models[method].model(torch.from_numpy(inputs)).numpy()
            inference_latency[method] = perf_counter() - started
            probabilities[method] = calibrated_probabilities(
                logits,
                error_rates,
                selected_models[method].parameters,
            ).reshape(indices.size, -1)

    baseline = decode_bplsd_batch(
        hx,
        syndromes,
        logical_x,
        error_rate=rate,
    )
    soft = decode_soft_prior_batch(hx, syndromes, logical_x, probabilities["soft_prior"])
    residual = decode_residual_batch(hx, syndromes, logical_x, probabilities["residual"])
    results = {"baseline": baseline, "soft_prior": soft, "residual": residual}
    arrays: dict[str, np.ndarray] = {"shot_indices": indices.astype(np.int64, copy=False)}
    for decoder, result in results.items():
        mismatch = np.any(result.predicted_observables != actual_observables, axis=1)
        failure = (~result.syndrome_valid) | mismatch
        arrays[f"{decoder}_failure"] = failure.astype(np.bool_)
        arrays[f"{decoder}_syndrome_valid"] = result.syndrome_valid.astype(np.bool_)
        arrays[f"{decoder}_logical_mismatch"] = mismatch.astype(np.bool_)
        arrays[f"{decoder}_converged"] = result.converged.astype(np.bool_)
        arrays[f"{decoder}_iterations"] = result.iterations.astype(np.int64)
        arrays[f"{decoder}_correction_weight"] = np.asarray(
            result.corrections.sum(axis=1), dtype=np.int64
        )
        arrays[f"{decoder}_latency_seconds"] = result.latency_seconds.astype(np.float64)
    for method, result in (("soft_prior", soft), ("residual", residual)):
        arrays[f"{method}_preprocessing_latency_seconds"] = (
            result.preprocessing_latency_seconds.astype(np.float64)
        )
        arrays[f"{method}_bp_latency_seconds"] = result.bp_latency_seconds.astype(np.float64)
        arrays[f"{method}_fno_latency_seconds"] = np.full(
            indices.size,
            inference_latency[method] / indices.size,
            dtype=np.float64,
        )
        arrays[f"{method}_end_to_end_latency_seconds"] = (
            arrays[f"{method}_latency_seconds"] + arrays[f"{method}_fno_latency_seconds"]
        )
    arrays["residual_proposal_weight"] = residual.proposal_weight.astype(np.int64)
    arrays["residual_syndrome_weight"] = residual.residual_syndrome_weight.astype(np.int64)
    arrays["residual_delta_weight"] = residual.delta_weight.astype(np.int64)
    reliability = {
        method: _probability_reliability(probabilities[method], actual_errors)
        for method in _HYBRIDS
    }
    return arrays, inference_latency, reliability


def _batch_manifest(
    *,
    rate_index: int,
    rate: float,
    batch_index: int,
    start: int,
    arrays: dict[str, np.ndarray],
    inference_latency: dict[str, float],
    probability_reliability: dict[str, list[dict[str, object]]],
    outcomes_path: Path,
    source_sha256: dict[str, object],
) -> dict[str, object]:
    shots = int(arrays["shot_indices"].size)
    failures = {name: int(np.count_nonzero(arrays[f"{name}_failure"])) for name in _DECODERS}
    return {
        "batch_index": batch_index,
        "complete": True,
        "error_rate": rate,
        "failures": failures,
        "fno_inference_latency_seconds": inference_latency,
        "invalid": {
            name: int(np.count_nonzero(~arrays[f"{name}_syndrome_valid"])) for name in _DECODERS
        },
        "logical_mismatch": {
            name: int(np.count_nonzero(arrays[f"{name}_logical_mismatch"])) for name in _DECODERS
        },
        "outcomes_path": outcomes_path.name,
        "outcomes_sha256": sha256_file(outcomes_path),
        "paired": {
            method: _paired_table(arrays["baseline_failure"], arrays[f"{method}_failure"])
            for method in _HYBRIDS
        },
        "probability_reliability": probability_reliability,
        "rate_index": rate_index,
        "shot_indices_sha256": _indices_sha256(arrays["shot_indices"]),
        "shots": shots,
        "source_sha256": source_sha256,
        "start": start,
        "stop": start + shots,
    }


def _verify_batch_manifest(
    path: Path,
    *,
    rate_index: int,
    rate: float,
    batch_index: int,
    expected_start: int,
    expected_indices: np.ndarray,
    source_sha256: dict[str, object],
) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError(f"evaluation batch is incomplete: {path}")
    expected_values = {
        "batch_index": batch_index,
        "error_rate": rate,
        "rate_index": rate_index,
        "start": expected_start,
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"evaluation batch {path} has mismatched {key}")
    shots = manifest.get("shots")
    stop = manifest.get("stop")
    if (
        type(shots) is not int
        or shots <= 0
        or type(stop) is not int
        or stop != expected_start + shots
    ):
        raise ValueError(f"evaluation batch {path} has invalid shot coordinates")
    if manifest.get("source_sha256") != source_sha256:
        raise ValueError(f"evaluation batch {path} source provenance mismatch")
    batch_expected_indices = expected_indices[expected_start : expected_start + shots]
    if batch_expected_indices.size != shots or manifest.get(
        "shot_indices_sha256"
    ) != _indices_sha256(batch_expected_indices):
        raise ValueError(f"evaluation batch {path} test-shot provenance mismatch")
    expected_name = "outcomes.npz"
    if manifest.get("outcomes_path") != expected_name:
        raise ValueError(f"evaluation batch {path} outcome path mismatch")
    outcomes_path = path.parent / expected_name
    verify_sha256(outcomes_path, str(manifest["outcomes_sha256"]), label="batch outcomes")
    for field in ("failures", "invalid", "logical_mismatch"):
        counts = manifest.get(field)
        if (
            not isinstance(counts, dict)
            or set(counts) != set(_DECODERS)
            or any(
                type(counts[name]) is not int or not 0 <= counts[name] <= shots
                for name in _DECODERS
            )
        ):
            raise ValueError(f"evaluation batch {path} has invalid {field} counts")
    paired = manifest.get("paired")
    if not isinstance(paired, dict) or set(paired) != set(_HYBRIDS):
        raise ValueError(f"evaluation batch {path} has invalid paired tables")
    for method in _HYBRIDS:
        table = paired[method]
        if (
            not isinstance(table, dict)
            or set(table)
            != {
                "baseline_only_failure",
                "both_fail",
                "both_succeed",
                "hybrid_only_failure",
            }
            or any(type(count) is not int or count < 0 for count in table.values())
            or sum(table.values()) != shots
        ):
            raise ValueError(f"evaluation batch {path} has malformed {method} paired table")
    reliability = manifest.get("probability_reliability")
    if not isinstance(reliability, dict) or set(reliability) != set(_HYBRIDS):
        raise ValueError(f"evaluation batch {path} has malformed reliability bins")
    for method in _HYBRIDS:
        rows = reliability[method]
        if (
            not isinstance(rows, list)
            or len(rows) != 10
            or any(not isinstance(row, dict) for row in rows)
            or sum(int(row.get("count", -1)) for row in rows) != shots * 2_610
        ):
            raise ValueError(f"evaluation batch {path} has malformed {method} reliability bins")
        for bin_index, row in enumerate(rows):
            expected_keys = {
                "bin_high",
                "bin_low",
                "count",
                "observed_errors",
                "probability_sum",
            }
            if set(row) != expected_keys:
                raise ValueError(
                    f"evaluation batch {path} has malformed {method} reliability fields"
                )
            count = row["count"]
            observed = row["observed_errors"]
            probability_sum = row["probability_sum"]
            bin_low = bin_index / 10
            bin_high = (bin_index + 1) / 10
            if (
                row["bin_low"] != bin_low
                or row["bin_high"] != bin_high
                or type(count) is not int
                or count < 0
                or type(observed) is not int
                or not 0 <= observed <= count
                or type(probability_sum) not in (int, float)
                or not math.isfinite(probability_sum)
                or not 0.0 <= probability_sum <= count
            ):
                raise ValueError(f"evaluation batch {path} has invalid {method} reliability values")
            tolerance = max(1.0, count) * 1e-12
            if not (count * bin_low - tolerance <= probability_sum <= count * bin_high + tolerance):
                raise ValueError(
                    f"evaluation batch {path} reliability probability sum falls outside "
                    f"the declared {method} bin"
                )
    _verify_outcome_archive(
        outcomes_path,
        manifest=manifest,
        expected_indices=batch_expected_indices,
    )
    return manifest


def _verify_outcome_archive(
    path: Path,
    *,
    manifest: dict[str, object],
    expected_indices: np.ndarray,
) -> None:
    shots = int(manifest["shots"])
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _OUTCOME_FIELDS:
            raise ValueError("batch outcome schema does not match the declared evaluation format")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
        if any(array.shape != (shots,) for array in arrays.values()):
            raise ValueError("batch outcomes must contain one value per test shot")
        if any(arrays[name].dtype != np.dtype(np.bool_) for name in _BOOLEAN_OUTCOME_FIELDS):
            raise ValueError("batch boolean outcomes have invalid dtypes")
        if any(arrays[name].dtype != np.dtype(np.int64) for name in _INTEGER_OUTCOME_FIELDS):
            raise ValueError("batch integer outcomes have invalid dtypes")
        if any(arrays[name].dtype != np.dtype(np.float64) for name in _FLOAT_OUTCOME_FIELDS):
            raise ValueError("batch latency outcomes have invalid dtypes")
        if not np.array_equal(arrays["shot_indices"], expected_indices):
            raise ValueError("batch outcomes do not contain the declared deterministic test shots")
        if any(
            np.any(~np.isfinite(arrays[name])) or np.any(arrays[name] < 0.0)
            for name in _FLOAT_OUTCOME_FIELDS
        ):
            raise ValueError("batch latency outcomes must be finite and non-negative")
        for decoder in _DECODERS:
            expected_failure = (
                ~arrays[f"{decoder}_syndrome_valid"] | arrays[f"{decoder}_logical_mismatch"]
            )
            if not np.array_equal(arrays[f"{decoder}_failure"], expected_failure):
                raise ValueError(f"{decoder} failure outcomes do not equal invalid OR mismatch")
            if int(np.count_nonzero(expected_failure)) != manifest["failures"][decoder]:
                raise ValueError(f"{decoder} failure count disagrees with batch outcomes")
            if (
                int(np.count_nonzero(~arrays[f"{decoder}_syndrome_valid"]))
                != manifest["invalid"][decoder]
            ):
                raise ValueError(f"{decoder} invalid count disagrees with batch outcomes")
            if (
                int(np.count_nonzero(arrays[f"{decoder}_logical_mismatch"]))
                != manifest["logical_mismatch"][decoder]
            ):
                raise ValueError(f"{decoder} mismatch count disagrees with batch outcomes")
        for method in _HYBRIDS:
            if (
                _paired_table(arrays["baseline_failure"], arrays[f"{method}_failure"])
                != manifest["paired"][method]
            ):
                raise ValueError(f"{method} paired table disagrees with batch outcomes")
            expected_end_to_end = (
                arrays[f"{method}_latency_seconds"] + arrays[f"{method}_fno_latency_seconds"]
            )
            if not np.allclose(
                arrays[f"{method}_end_to_end_latency_seconds"],
                expected_end_to_end,
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(f"{method} end-to-end latency disagrees with batch outcomes")


def _scan_batches(
    output: Path,
    *,
    rate_index: int,
    rate: float,
    expected_indices: np.ndarray,
    source_sha256: dict[str, object],
) -> list[tuple[Path, dict[str, object]]]:
    rate_dir = output / f"rate-{rate_index:03d}"
    if not rate_dir.exists():
        return []
    for staging in rate_dir.glob(".batch-*.tmp"):
        if not staging.is_dir():
            raise ValueError(f"rate {rate_index} has a malformed batch staging artifact")
        shutil.rmtree(staging)
    batch_directories = sorted(rate_dir.glob("batch-*"))
    records: list[tuple[Path, dict[str, object]]] = []
    expected_start = 0
    for batch_index, directory in enumerate(batch_directories):
        if not directory.is_dir() or directory.name != f"batch-{batch_index:05d}":
            raise ValueError(f"rate {rate_index} evaluation batch indices are not consecutive")
        if {path.name for path in directory.iterdir()} != {"manifest.json", "outcomes.npz"}:
            raise ValueError(f"rate {rate_index} has an incomplete evaluation batch directory")
        path = directory / "manifest.json"
        manifest = _verify_batch_manifest(
            path,
            rate_index=rate_index,
            rate=rate,
            batch_index=batch_index,
            expected_start=expected_start,
            expected_indices=expected_indices,
            source_sha256=source_sha256,
        )
        expected_start = int(manifest["stop"])
        records.append((path, manifest))
    return records


def _aggregate_counts(records: list[tuple[Path, dict[str, object]]]) -> dict[str, object]:
    failures = {name: 0 for name in _DECODERS}
    invalid = {name: 0 for name in _DECODERS}
    mismatch = {name: 0 for name in _DECODERS}
    shots = 0
    for _, manifest in records:
        shots += int(manifest["shots"])
        for name in _DECODERS:
            failures[name] += int(manifest["failures"][name])
            invalid[name] += int(manifest["invalid"][name])
            mismatch[name] += int(manifest["logical_mismatch"][name])
    return {"failures": failures, "invalid": invalid, "logical_mismatch": mismatch, "shots": shots}


def _load_rate_outcomes(
    records: list[tuple[Path, dict[str, object]]],
) -> dict[str, np.ndarray]:
    batches: dict[str, list[np.ndarray]] = {}
    for path, manifest in records:
        with np.load(path.parent / str(manifest["outcomes_path"]), allow_pickle=False) as archive:
            for key in archive.files:
                batches.setdefault(key, []).append(np.array(archive[key], copy=True))
    return {
        key: np.concatenate(values) if values else np.empty(0) for key, values in batches.items()
    }


def _decoder_summary(outcomes: dict[str, np.ndarray], decoder: str) -> dict[str, object]:
    failures = outcomes[f"{decoder}_failure"]
    valid = outcomes[f"{decoder}_syndrome_valid"]
    mismatch = outcomes[f"{decoder}_logical_mismatch"]
    base = paired_decoder_summary(failures, failures)["baseline"]
    return {
        **base,
        "converged": int(np.count_nonzero(outcomes[f"{decoder}_converged"])),
        "converged_rate": float(np.mean(outcomes[f"{decoder}_converged"])),
        "correction_weight": _distribution(outcomes[f"{decoder}_correction_weight"]),
        "iterations": _distribution(outcomes[f"{decoder}_iterations"]),
        "latency_seconds": _distribution(outcomes[f"{decoder}_latency_seconds"]),
        "logical_mismatch": int(np.count_nonzero(mismatch)),
        "logical_mismatch_among_valid": int(np.count_nonzero(mismatch & valid)),
        "syndrome_invalid": int(np.count_nonzero(~valid)),
        "syndrome_valid": int(np.count_nonzero(valid)),
        "syndrome_valid_rate": float(np.mean(valid)),
    }


def _aggregate_reliability(
    records: list[tuple[Path, dict[str, object]]],
    method: str,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for bin_index in range(10):
        rows = [manifest["probability_reliability"][method][bin_index] for _, manifest in records]
        count = sum(int(row["count"]) for row in rows)
        observed = sum(int(row["observed_errors"]) for row in rows)
        probability_sum = sum(float(row["probability_sum"]) for row in rows)
        result.append(
            {
                "bin_high": (bin_index + 1) / 10,
                "bin_low": bin_index / 10,
                "count": count,
                "mean_predicted_probability": probability_sum / count if count else None,
                "observed_error_rate": observed / count if count else None,
                "observed_errors": observed,
            }
        )
    return result


def _write_rate_summary(
    output: Path,
    *,
    rate_index: int,
    rate: float,
    records: list[tuple[Path, dict[str, object]]],
    status: str,
    stop_reason: str,
    fixed_sample: bool,
    source_sha256: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    summary_path = output / f"rate-{rate_index:03d}" / "summary.json"
    if not records:
        payload: dict[str, object] = {
            "comparison_status": {method: "not_fixed_sample" for method in _HYBRIDS},
            "decoders": {},
            "error_rate": rate,
            "paired": {},
            "rate_index": rate_index,
            "shots": 0,
            "source_sha256": source_sha256,
            "status": status,
            "stop_reason": stop_reason,
        }
        _write_json_atomic(summary_path, payload)
        return summary_path, payload

    outcomes = _load_rate_outcomes(records)
    paired = {
        method: paired_decoder_summary(
            outcomes["baseline_failure"],
            outcomes[f"{method}_failure"],
        )
        for method in _HYBRIDS
    }
    decoders = {name: _decoder_summary(outcomes, name) for name in _DECODERS}
    payload = {
        "comparison_status": {
            method: paired_comparison_status(paired[method], fixed_sample=fixed_sample)
            for method in _HYBRIDS
        },
        "decoders": decoders,
        "diagnostics": {
            "residual_delta_weight": _distribution(outcomes["residual_delta_weight"]),
            "residual_proposal_weight": _distribution(outcomes["residual_proposal_weight"]),
            "residual_syndrome_weight": _distribution(outcomes["residual_syndrome_weight"]),
            **{
                f"{method}_bp_latency_seconds": _distribution(
                    outcomes[f"{method}_bp_latency_seconds"]
                )
                for method in _HYBRIDS
            },
            **{
                f"{method}_fno_latency_seconds": _distribution(
                    outcomes[f"{method}_fno_latency_seconds"]
                )
                for method in _HYBRIDS
            },
            **{
                f"{method}_end_to_end_latency_seconds": _distribution(
                    outcomes[f"{method}_end_to_end_latency_seconds"]
                )
                for method in _HYBRIDS
            },
            **{
                f"{method}_preprocessing_latency_seconds": _distribution(
                    outcomes[f"{method}_preprocessing_latency_seconds"]
                )
                for method in _HYBRIDS
            },
            **{
                f"{method}_probability_reliability": _aggregate_reliability(records, method)
                for method in _HYBRIDS
            },
        },
        "error_rate": rate,
        "paired": paired,
        "rate_index": rate_index,
        "shots": int(outcomes["baseline_failure"].size),
        "source_sha256": source_sha256,
        "status": status,
        "stop_reason": stop_reason,
    }
    _write_json_atomic(summary_path, payload)
    return summary_path, payload


def _write_progress(
    output: Path,
    *,
    source_sha256: dict[str, object],
    rates: tuple[float, ...],
    rate_records: dict[int, list[tuple[Path, dict[str, object]]]],
    status: str,
) -> None:
    rate_payload: dict[str, object] = {}
    for rate_index, rate in enumerate(rates):
        records = rate_records[rate_index]
        counts = _aggregate_counts(records)
        rate_payload[str(rate_index)] = {
            "batch_manifests": {
                str(path.relative_to(output)): sha256_file(path) for path, _ in records
            },
            "error_rate": rate,
            **counts,
        }
    _write_json_atomic(
        output / "progress.json",
        {
            "rates": rate_payload,
            "source_sha256": source_sha256,
            "status": status,
        },
    )


def _verify_existing_progress(
    output: Path,
    *,
    source_sha256: dict[str, object],
    rates: tuple[float, ...],
    rate_records: dict[int, list[tuple[Path, dict[str, object]]]],
) -> None:
    progress_path = output / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError("evaluation resume requires progress.json")
    progress = json.loads(progress_path.read_text())
    if progress.get("source_sha256") != source_sha256:
        raise ValueError("evaluation progress source provenance mismatch")
    previous_rates = progress.get("rates")
    if not isinstance(previous_rates, dict) or set(previous_rates) != {
        str(index) for index in range(len(rates))
    }:
        raise ValueError("evaluation progress rate coordinates are malformed")
    for rate_index in range(len(rates)):
        previous = previous_rates[str(rate_index)]
        if not isinstance(previous, dict):
            raise TypeError("evaluation progress rate entry must be an object")
        declared = previous.get("batch_manifests")
        if not isinstance(declared, dict):
            raise TypeError("evaluation progress batch provenance must be an object")
        discovered = {
            str(path.relative_to(output)): sha256_file(path) for path, _ in rate_records[rate_index]
        }
        for relative, digest in declared.items():
            if discovered.get(relative) != digest:
                raise ValueError("evaluation progress batch-manifest provenance mismatch")


def _verify_final_manifest(
    output: Path,
    *,
    source_sha256: dict[str, object],
    selected_calibration: dict[str, object],
    expected_rates: tuple[float, ...],
    rate_records: dict[int, list[tuple[Path, dict[str, object]]]],
    config: CampaignConfig,
) -> str:
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("complete") is not True or manifest.get("source_sha256") != source_sha256:
        raise ValueError("completed evaluation provenance mismatch")
    if manifest.get("selected_calibration") != selected_calibration:
        raise ValueError("completed evaluation calibration selection mismatch")
    if manifest.get("selection_mode") != config.selection_mode:
        raise ValueError("completed evaluation selection mode mismatch")
    if manifest.get("test_stopping_mode") != config.test_stopping_mode:
        raise ValueError("completed evaluation test stopping mode mismatch")
    if manifest.get("target_failures_active") is not (
        config.test_stopping_mode == "adaptive"
    ):
        raise ValueError("completed evaluation target-failure policy mismatch")
    status = manifest.get("status")
    if status not in {"complete", "partial_deadline"}:
        raise ValueError("completed evaluation status is invalid")
    rates = manifest.get("rates")
    expected_rate_keys = {str(index) for index in range(len(expected_rates))}
    if not isinstance(rates, dict) or set(rates) != expected_rate_keys:
        raise ValueError("completed evaluation rate coordinates are malformed")
    for rate_index, error_rate in enumerate(expected_rates):
        record = rates[str(rate_index)]
        expected_record_fields = {
            "error_rate",
            "shots",
            "status",
            "stop_reason",
            "summary_path",
            "summary_sha256",
        }
        if not isinstance(record, dict) or set(record) != expected_record_fields:
            raise ValueError("completed evaluation rate record is malformed")
        expected_summary_relative = f"rate-{rate_index:03d}/summary.json"
        if record.get("summary_path") != expected_summary_relative:
            raise ValueError("completed evaluation summary path is unsafe or mismatched")
        summary_path = output / expected_summary_relative
        verify_sha256(summary_path, str(record["summary_sha256"]), label="rate summary")
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("source_sha256") != source_sha256
            or summary.get("rate_index") != rate_index
            or summary.get("error_rate") != error_rate
            or summary.get("shots") != record.get("shots")
            or summary.get("status") != record.get("status")
            or summary.get("stop_reason") != record.get("stop_reason")
        ):
            raise ValueError("completed evaluation rate summary provenance mismatch")
        decoded_shots = int(_aggregate_counts(rate_records[rate_index])["shots"])
        recorded_shots = record.get("shots")
        if type(recorded_shots) is not int or recorded_shots < 0:
            raise ValueError("completed evaluation rate shot count is invalid")
        if (
            config.test_stopping_mode == "fixed"
            and record.get("status") == "complete"
            and (
                record.get("stop_reason") != "shot_cap"
                or recorded_shots != config.max_test_shots_per_point
            )
        ):
            raise ValueError("fixed evaluation rate must complete at the configured shot cap")
        if status == "complete" and recorded_shots != decoded_shots:
            raise ValueError("completed evaluation rate shot count disagrees with outcomes")
        if status == "partial_deadline" and recorded_shots > decoded_shots:
            raise ValueError("deadline evaluation rate shot count exceeds available outcomes")
        comparison_status = summary.get("comparison_status")
        if not isinstance(comparison_status, dict) or set(comparison_status) != set(_HYBRIDS):
            raise ValueError("completed evaluation comparison status is malformed")
        if "accuracy_compatible" in summary:
            raise ValueError("completed evaluation contains obsolete compatibility status")
        if decoded_shots:
            outcomes = _load_rate_outcomes(rate_records[rate_index])
            expected_paired = {
                method: paired_decoder_summary(
                    outcomes["baseline_failure"],
                    outcomes[f"{method}_failure"],
                )
                for method in _HYBRIDS
            }
            paired = summary.get("paired")
            if not isinstance(paired, dict) or set(paired) != set(_HYBRIDS):
                raise ValueError("completed evaluation exact paired fields are malformed")
            for method in _HYBRIDS:
                if not isinstance(paired[method], dict) or set(paired[method]) != set(
                    expected_paired[method]
                ):
                    raise ValueError("completed evaluation exact paired fields are malformed")
                if paired[method] != expected_paired[method]:
                    raise ValueError("completed evaluation paired summary disagrees with outcomes")
            fixed_sample = (
                config.test_stopping_mode == "fixed"
                and record.get("status") == "complete"
                and record.get("stop_reason") == "shot_cap"
                and recorded_shots == config.max_test_shots_per_point
            )
            expected_comparison = {
                method: paired_comparison_status(
                    expected_paired[method],
                    fixed_sample=fixed_sample,
                )
                for method in _HYBRIDS
            }
        else:
            if summary.get("paired") != {}:
                raise ValueError("empty evaluation rate has malformed paired summaries")
            expected_comparison = {method: "not_fixed_sample" for method in _HYBRIDS}
        if comparison_status != expected_comparison:
            raise ValueError("completed evaluation comparison status disagrees with outcomes")
    return status


def _finalize(
    output: Path,
    *,
    status: str,
    reason_for_unfinished: str,
    source_sha256: dict[str, object],
    rates: tuple[float, ...],
    records: dict[int, list[tuple[Path, dict[str, object]]]],
    config: CampaignConfig,
    selected_calibration: dict[str, object],
) -> None:
    rate_results: dict[str, object] = {}
    for rate_index, rate in enumerate(rates):
        counts = _aggregate_counts(records[rate_index])
        scientific_reason = test_stop_reason(
            counts["failures"],
            shots=int(counts["shots"]),
            target_failures=config.target_failures,
            shot_cap=config.max_test_shots_per_point,
            mode=config.test_stopping_mode,
        )
        stop_reason = scientific_reason or reason_for_unfinished
        rate_status = "complete" if scientific_reason is not None else status
        fixed_sample = (
            config.test_stopping_mode == "fixed"
            and rate_status == "complete"
            and stop_reason == "shot_cap"
            and int(counts["shots"]) == config.max_test_shots_per_point
        )
        summary_path, summary = _write_rate_summary(
            output,
            rate_index=rate_index,
            rate=rate,
            records=records[rate_index],
            status=rate_status,
            stop_reason=stop_reason,
            fixed_sample=fixed_sample,
            source_sha256=source_sha256,
        )
        rate_results[str(rate_index)] = {
            "error_rate": rate,
            "shots": summary["shots"],
            "status": rate_status,
            "stop_reason": stop_reason,
            "summary_path": str(summary_path.relative_to(output)),
            "summary_sha256": sha256_file(summary_path),
        }
    _write_progress(
        output,
        source_sha256=source_sha256,
        rates=rates,
        rate_records=records,
        status=status,
    )
    _write_json_atomic(
        output / "manifest.json",
        {
            "complete": True,
            "rates": rate_results,
            "selection_mode": config.selection_mode,
            "selected_calibration": selected_calibration,
            "source_sha256": source_sha256,
            "status": status,
            "target_failures_active": config.test_stopping_mode == "adaptive",
            "test_stopping_mode": config.test_stopping_mode,
        },
    )


def evaluate_hybrid_campaign(args: EvaluationRequest) -> None:
    """Verify inputs, evaluate paired decoders, and publish outcomes."""
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    if args.max_batches_this_run is not None and args.max_batches_this_run <= 0:
        raise ValueError("max-batches-this-run must be positive")
    if args.deadline_monotonic is not None and not math.isfinite(args.deadline_monotonic):
        raise ValueError("deadline-monotonic must be finite")
    if args.campaign_mode == "canonical" and args.bootstrap_samples != 10_000:
        raise ValueError("canonical evaluation requires 10000 bootstrap samples")
    if args.campaign_mode not in {"canonical", "reduced_non_scientific"}:
        raise ValueError("evaluation campaign mode is invalid")
    config = CampaignConfig.from_json(args.config)
    code_manifest_path = args.code / "code.json"
    _, hx, _, logical_x = load_campaign_code(args.code)
    rates = _verify_selection(
        args.selection,
        config_path=args.config,
        code_manifest_path=code_manifest_path,
    )
    test_shards = load_verified_shards(
        args.test,
        role="test",
        config_path=args.config,
        code_manifest_path=code_manifest_path,
    )
    calibration_shards = load_verified_shards(
        args.calibration,
        role="calibration",
        config_path=args.config,
        code_manifest_path=code_manifest_path,
    )
    _verify_shard_selection(test_shards, rates=rates, selection_path=args.selection)
    _verify_shard_selection(calibration_shards, rates=rates, selection_path=args.selection)
    if {shard.seed for shard in test_shards.shards} & {
        shard.seed for shard in calibration_shards.shards
    }:
        raise ValueError("calibration and test shard seeds must be disjoint")
    selected_models, source_sha256 = _load_selected_models(
        config_path=args.config,
        code_manifest_path=code_manifest_path,
        model_dir=args.model,
        calibration_dir=args.calibration,
        calibration_shards=calibration_shards,
        campaign_mode=args.campaign_mode,
        syndrome_checks=hx.shape[0],
    )
    selected_calibration: dict[str, object] = {
        method: {
            "model_checkpoint": selected_models[method].checkpoint,
            "model_epoch": selected_models[method].epoch,
            "parameters": {
                "alpha": selected_models[method].parameters.alpha,
                "beta": selected_models[method].parameters.beta,
                "temperature": selected_models[method].parameters.temperature,
            },
        }
        for method in _HYBRIDS
    }
    source_sha256.update(
        {
            "selection": sha256_file(args.selection),
            "test_manifest": test_shards.manifest_sha256,
            "test_shard_manifests": test_shards.shard_manifest_sha256,
        }
    )
    rate_indices = {index: test_shards.indices_for_rate(index) for index in range(len(rates))}
    for rate_index, indices in rate_indices.items():
        if indices.size < config.max_test_shots_per_point:
            raise ValueError(
                f"test shards for rate {rate_index} contain {indices.size} shots; "
                f"the configured cap requires {config.max_test_shots_per_point}"
            )

    if args.out.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing evaluation output: {args.out}")
    if not args.out.exists():
        if args.resume:
            raise FileNotFoundError("evaluation resume output does not exist")
        args.out.mkdir(parents=True)
        records = {index: [] for index in range(len(rates))}
        _write_progress(
            args.out,
            source_sha256=source_sha256,
            rates=rates,
            rate_records=records,
            status="in_progress",
        )
    records = {
        index: _scan_batches(
            args.out,
            rate_index=index,
            rate=rate,
            expected_indices=rate_indices[index],
            source_sha256=source_sha256,
        )
        for index, rate in enumerate(rates)
    }
    if args.resume:
        _verify_existing_progress(
            args.out,
            source_sha256=source_sha256,
            rates=rates,
            rate_records=records,
        )
        if (args.out / "manifest.json").is_file():
            final_status = _verify_final_manifest(
                args.out,
                source_sha256=source_sha256,
                selected_calibration=selected_calibration,
                expected_rates=rates,
                rate_records=records,
                config=config,
            )
            if final_status == "complete":
                return
        _write_progress(
            args.out,
            source_sha256=source_sha256,
            rates=rates,
            rate_records=records,
            status="in_progress",
        )

    batches_this_run = 0
    deadline_requested = False
    for rate_index, rate in enumerate(rates):
        counts = _aggregate_counts(records[rate_index])
        stop_reason = test_stop_reason(
            counts["failures"],
            shots=int(counts["shots"]),
            target_failures=config.target_failures,
            shot_cap=config.max_test_shots_per_point,
            mode=config.test_stopping_mode,
        )
        while stop_reason is None:
            if args.deadline_monotonic is not None and monotonic() >= args.deadline_monotonic:
                deadline_requested = True
                break
            if (
                args.max_batches_this_run is not None
                and batches_this_run >= args.max_batches_this_run
            ):
                _write_progress(
                    args.out,
                    source_sha256=source_sha256,
                    rates=rates,
                    rate_records=records,
                    status="in_progress",
                )
                return
            start = int(counts["shots"])
            stop = min(
                start + config.test_batch_shots,
                config.max_test_shots_per_point,
            )
            indices = rate_indices[rate_index][start:stop]
            arrays, inference_latency, probability_reliability = _decode_batch(
                indices=indices,
                rate=rate,
                shards=test_shards,
                hx=hx,
                logical_x=logical_x,
                selected_models=selected_models,
            )
            batch_index = len(records[rate_index])
            rate_dir = args.out / f"rate-{rate_index:03d}"
            rate_dir.mkdir(exist_ok=True)
            batch_dir = rate_dir / f"batch-{batch_index:05d}"
            staging_dir = rate_dir / f".batch-{batch_index:05d}.tmp"
            if batch_dir.exists() or staging_dir.exists():
                raise FileExistsError("refusing to overwrite an evaluation batch")
            staging_dir.mkdir()
            outcomes_path = staging_dir / "outcomes.npz"
            manifest_path = staging_dir / "manifest.json"
            _write_outcomes_atomic(outcomes_path, arrays)
            _write_json_atomic(
                manifest_path,
                _batch_manifest(
                    rate_index=rate_index,
                    rate=rate,
                    batch_index=batch_index,
                    start=start,
                    arrays=arrays,
                    inference_latency=inference_latency,
                    probability_reliability=probability_reliability,
                    outcomes_path=outcomes_path,
                    source_sha256=source_sha256,
                ),
            )
            os.replace(staging_dir, batch_dir)
            outcomes_path = batch_dir / "outcomes.npz"
            manifest_path = batch_dir / "manifest.json"
            records[rate_index].append((manifest_path, json.loads(manifest_path.read_text())))
            counts = _aggregate_counts(records[rate_index])
            stop_reason = test_stop_reason(
                counts["failures"],
                shots=int(counts["shots"]),
                target_failures=config.target_failures,
                shot_cap=config.max_test_shots_per_point,
                mode=config.test_stopping_mode,
            )
            batches_this_run += 1
            _write_progress(
                args.out,
                source_sha256=source_sha256,
                rates=rates,
                rate_records=records,
                status="in_progress",
            )
        if deadline_requested:
            break

    if deadline_requested:
        _finalize(
            args.out,
            status="partial_deadline",
            reason_for_unfinished="campaign_deadline",
            source_sha256=source_sha256,
            rates=rates,
            records=records,
            config=config,
            selected_calibration=selected_calibration,
        )
        return
    _finalize(
        args.out,
        status="complete",
        reason_for_unfinished="",
        source_sha256=source_sha256,
        rates=rates,
        records=records,
        config=config,
        selected_calibration=selected_calibration,
    )
