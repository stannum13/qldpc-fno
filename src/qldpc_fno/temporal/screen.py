"""Immutable orchestration for the reduced causal FNO/HiPPO screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import CSSCode, build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.models.causal_forecaster import CausalChannelForecaster, build_forecaster
from qldpc_fno.temporal.causality import (
    CausalAuditSequence,
    audit_structural_prefix_causality,
)
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.dataset import (
    build_sequence_manifest,
    read_verified_sequence,
    regenerate_and_verify,
    write_sequence,
)
from qldpc_fno.temporal.evaluation import (
    ArmEvaluation,
    BaselineSelection,
    CausalEvaluationBatch,
    FrozenArm,
    FrozenBaseline,
    PairedCausalEvaluation,
    evaluate_causal_arms,
    evaluate_oracle_sanity,
    export_frozen_predictor,
    fit_select_observation_baselines,
    freeze_learned_arm,
    reduced_factor_diagnostics,
    reduced_progression,
    restore_frozen_predictor,
    validate_frozen_arm_calibration,
)
from qldpc_fno.temporal.generator import generate_latent_sequence, sample_sequence
from qldpc_fno.temporal.seeds import REGIMES
from qldpc_fno.training.causal_sequence import (
    SequenceRoleBatch,
    build_overfit_fixture,
    overfit_causal_forecaster,
    train_causal_forecaster,
    validate_role_partition,
)

_ROLES = ("train", "validation", "calibration")
_CELLS = (("cnn", "fir"), ("fno", "fir"), ("cnn", "hippo"), ("fno", "hippo"))
_CELL_NAMES = tuple(f"{spatial}_{temporal}" for spatial, temporal in _CELLS)
_BASELINE_NAMES = ("stationary_field", "ewma", "logistic_ar")
_ARM_NAMES = (*_BASELINE_NAMES, *_CELL_NAMES)
_CAUSAL_MUTATIONS = (
    "current_future_syndromes",
    "physical_error_labels",
    "logical_labels",
    "privileged_diagnostics",
)
_RESULT_FILES = ("results.json", "timing.json")
_MANIFEST = "manifest.json"
_EVIDENCE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class _RegimeEvidence:
    selection: BaselineSelection
    predictors: tuple[FrozenBaseline | FrozenArm, ...]
    evaluation: PairedCausalEvaluation
    _token: object

    def __post_init__(self) -> None:
        if self._token is not _EVIDENCE_TOKEN:
            raise ValueError("regime evidence must be created by the screen runner")


@dataclass(frozen=True, slots=True)
class _RepositoryEvidence:
    root: Path
    commit: str


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _mapping_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _repository_evidence() -> _RepositoryEvidence:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("causal experiment requires a clean nonignored repository worktree")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _RepositoryEvidence(root=root, commit=commit)


def _require_ignored_repository_output(repository: _RepositoryEvidence, output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if not resolved.is_relative_to(repository.root):
        return
    ignored = subprocess.run(
        ["git", "-C", str(repository.root), "check-ignore", "--quiet", "--", str(resolved)],
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError("experiment output inside the repository must be git-ignored")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.ascontiguousarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(target: Path, writer: Any) -> None:
    if target.exists():
        raise FileExistsError(f"refuse to overwrite existing artifact directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-staging-", dir=target.parent))
    try:
        writer(staging)
        if not (staging / _MANIFEST).is_file():
            raise RuntimeError("staging completed without a final completion manifest")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _canonical_code() -> CSSCode:
    return build_self_lifted_product(PAPER_LP_3_7_16)


def _role_size(config: CausalExperimentConfig, role: str) -> int:
    return int(getattr(config.splits, role))


def _expected_sequence_coordinates(
    config: CausalExperimentConfig,
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (regime, role, index)
        for regime in REGIMES
        for role in _ROLES
        for index in range(_role_size(config, role))
    )


def _sequence_relative(regime: str, role: str, index: int) -> Path:
    return Path(regime) / role / f"sequence-{index:05d}"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path}")
    return value


def generate_sequence_campaign(
    *,
    config_path: Path,
    output_dir: Path,
    _repository: _RepositoryEvidence | None = None,
) -> dict[str, Any]:
    """Generate every A-E train/validation/calibration sequence immutably."""

    config = CausalExperimentConfig.from_json(config_path)
    repository = _repository or _repository_evidence()
    _require_ignored_repository_output(repository, output_dir)
    if config.artifact_mode != "reduced_non_scientific":
        raise ValueError("reduced generation requires artifact_mode reduced_non_scientific")
    code = _canonical_code()
    request_digest = _mapping_digest(
        {"config": config.to_dict(), "source_commit": repository.commit, "roles": list(_ROLES)}
    )
    completion = output_dir / _MANIFEST
    if output_dir.exists():
        if not completion.is_file():
            raise FileExistsError(f"refuse to overwrite incomplete sequence campaign: {output_dir}")
        existing = _load_json(completion)
        if existing.get("request_digest") != request_digest:
            raise FileExistsError(f"refuse to overwrite completed differing run: {output_dir}")
        verify_sequence_campaign(config_path=config_path, output_dir=output_dir, regenerate=False)
        return existing

    def write(staging: Path) -> None:
        rows: list[dict[str, object]] = []
        for regime, role, index in _expected_sequence_coordinates(config):
            latent = generate_latent_sequence(
                config, regime=regime, role=role, sequence_index=index
            )
            observed, supervision, diagnostics = sample_sequence(
                latent, bernoulli_seed=latent.seeds.bernoulli, code=code
            )
            manifest = build_sequence_manifest(config=config, latent=latent, code=code)
            relative = _sequence_relative(regime, role, index)
            write_sequence(staging / relative, observed, supervision, diagnostics, manifest)
            rows.append(
                {
                    "regime": regime,
                    "role": role,
                    "sequence_index": index,
                    "path": str(relative),
                    "manifest_sha256": sha256_file(staging / relative / _MANIFEST),
                }
            )
        root = {
            "artifact_mode": "reduced_non_scientific",
            "complete": True,
            "config": config.to_dict(),
            "request_digest": request_digest,
            "source_commit": repository.commit,
            "sequences": rows,
        }
        _write_bytes(staging / _MANIFEST, _canonical_json(root))

    _publish_directory(output_dir, write)
    return _load_json(completion)


def verify_sequence_campaign(
    *, config_path: Path, output_dir: Path, regenerate: bool
) -> dict[str, Any]:
    """Verify root membership and optionally byte-regenerate every sequence."""

    config = CausalExperimentConfig.from_json(config_path)
    code = _canonical_code()
    root = _load_json(output_dir / _MANIFEST)
    if set(root) != {
        "artifact_mode",
        "complete",
        "config",
        "request_digest",
        "source_commit",
        "sequences",
    }:
        raise ValueError("sequence campaign manifest has missing or unknown fields")
    if root.get("artifact_mode") != "reduced_non_scientific" or root.get("complete") is not True:
        raise ValueError("sequence campaign is not a completed reduced_non_scientific artifact")
    root_config = root.get("config")
    if not isinstance(root_config, Mapping) or _mapping_digest(root_config) != _mapping_digest(
        config.to_dict()
    ):
        raise ValueError("sequence campaign configuration does not match")
    expected_request = _mapping_digest(
        {
            "config": config.to_dict(),
            "source_commit": root.get("source_commit"),
            "roles": list(_ROLES),
        }
    )
    if root.get("request_digest") != expected_request:
        raise ValueError("sequence campaign request digest does not match")
    rows = root.get("sequences")
    if not isinstance(rows, list):
        raise TypeError("sequence campaign membership is missing")
    expected = _expected_sequence_coordinates(config)
    actual = tuple((row.get("regime"), row.get("role"), row.get("sequence_index")) for row in rows)
    if actual != expected:
        raise ValueError("sequence campaign does not contain exact A-E role membership")
    expected_files = {_MANIFEST}
    for row in rows:
        relative = _sequence_relative(
            str(row["regime"]), str(row["role"]), int(row["sequence_index"])
        )
        if row.get("path") != str(relative):
            raise ValueError("sequence campaign path is noncanonical")
        artifact = output_dir / relative
        if sha256_file(artifact / _MANIFEST) != row.get("manifest_sha256"):
            raise ValueError("sequence manifest SHA-256 mismatch")
        read_verified_sequence(
            artifact,
            config=config,
            code=code,
            expected_source_commit=str(root["source_commit"]),
        )
        if regenerate:
            regenerate_and_verify(artifact, config, code)
        expected_files.update(
            str(relative / name)
            for name in (*("observed.npz", "supervision.npz", "diagnostics.npz"), _MANIFEST)
        )
    actual_files = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise ValueError("sequence campaign contains missing or undeclared files")
    return root


def _walk_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return tuple(keys)


def _validate_scientific_payload(payload: Mapping[str, object]) -> None:
    required = {
        "artifact_mode",
        "scientific_scope",
        "p_value",
        "hypothesis_status",
        "source",
        "causal_audit",
        "regimes",
        "progressions",
        "factor_diagnostics",
    }
    if set(payload) != required:
        raise ValueError("scientific result payload has missing or unknown fields")
    if (
        payload["artifact_mode"] != "reduced_non_scientific"
        or payload["scientific_scope"] != "reduced_non_scientific"
    ):
        raise ValueError("scientific result must be labelled reduced_non_scientific")
    if payload["p_value"] is not None or payload["hypothesis_status"] is not None:
        raise ValueError("reduced result cannot contain p-values or hypothesis status")
    forbidden = ("latency", "timing", "wall_seconds", "setup_seconds", "decode_seconds")
    if any(any(marker in key for marker in forbidden) for key in _walk_keys(payload)):
        raise ValueError("timing fields must be isolated from the scientific payload")


def _assert_close(observed: object, expected: float, label: str) -> None:
    if type(observed) not in {int, float} or not math.isclose(
        float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"result recomputation mismatch: {label}")


def recompute_screen_result(payload: Mapping[str, object]) -> None:
    """Recompute all stored aggregate metrics from sequence and round payloads."""

    _validate_scientific_payload(payload)
    source = payload["source"]
    if not isinstance(source, Mapping) or set(source) != {
        "config_sha256",
        "sequences_manifest_sha256",
        "source_commit",
    }:
        raise ValueError("result source must contain the exact input hash set")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(source[name])) is None
        for name in ("config_sha256", "sequences_manifest_sha256")
    ):
        raise ValueError("result source contains an invalid SHA-256 digest")
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(source["source_commit"])) is None:
        raise ValueError("result source contains an invalid commit identity")
    audit = payload["causal_audit"]
    if not isinstance(audit, Mapping) or set(audit) != {
        "passed",
        "arms",
        "overfit_fixture",
    }:
        raise ValueError("result causal audit has missing or unknown fields")
    audit_arms = audit["arms"]
    overfit = audit["overfit_fixture"]
    if not isinstance(audit_arms, Mapping) or set(audit_arms) != set(REGIMES):
        raise ValueError("result causal audit does not cover exact A-E regimes")
    if not isinstance(overfit, Mapping) or set(overfit) != set(_CELL_NAMES):
        raise ValueError("result overfit audit does not cover exact learned cells")
    causal_checks: list[bool] = []
    for regime in REGIMES:
        rows = audit_arms[regime]
        if not isinstance(rows, Mapping) or set(rows) != set(_CELL_NAMES):
            raise ValueError("result causal audit does not cover exact learned cells")
        for row in rows.values():
            if not isinstance(row, Mapping) or not isinstance(row.get("checks"), list):
                raise TypeError("result causal audit row is invalid")
            if tuple(check.get("name") for check in row["checks"]) != _CAUSAL_MUTATIONS:
                raise ValueError("result causal audit has an incomplete mutation set")
            passed = all(check.get("bit_identical") is True for check in row["checks"])
            if row.get("passed") is not passed:
                raise ValueError("result causal audit pass flag is inconsistent")
            causal_checks.append(passed)
    if audit.get("passed") is not all(causal_checks):
        raise ValueError("result aggregate causal audit pass flag is inconsistent")
    regimes = payload["regimes"]
    if not isinstance(regimes, Mapping):
        raise TypeError("result regimes must be an object")
    if set(regimes) != set(REGIMES):
        raise ValueError("result does not contain the exact A-E regime set")
    arm_lookup: dict[str, Mapping[str, Mapping[str, object]]] = {}
    reference_geometry: tuple[object, object, object] | None = None
    reference_decoder_policy: object | None = None
    reference_baseline_policy: object | None = None
    for regime, regime_result in regimes.items():
        if not isinstance(regime_result, Mapping) or not isinstance(
            regime_result.get("arms"), Mapping
        ):
            raise TypeError(f"result regime {regime!r} has invalid arms")
        if set(regime_result) != {
            "selected_observation_baseline",
            "baseline_validation_nll",
            "hashes",
            "oracle_generator_sanity",
            "arms",
        }:
            raise ValueError(f"result regime {regime!r} has missing or unknown fields")
        if set(regime_result["arms"]) != set(_ARM_NAMES):
            raise ValueError(f"result regime {regime!r} does not contain the exact arm set")
        oracle = regime_result["oracle_generator_sanity"]
        if (
            not isinstance(oracle, Mapping)
            or oracle.get("scope") != "generator_sanity_only"
            or oracle.get("deployable_competition") is not False
            or oracle.get("decoder_evaluated") is not False
        ):
            raise ValueError("result oracle must be labelled nondeployable generator sanity only")
        _assert_close(
            oracle.get("overall_nll"), float(np.mean(oracle["per_sequence_nll"])), "oracle_nll"
        )
        _assert_close(
            oracle.get("overall_brier"),
            float(np.mean(oracle["per_sequence_brier"])),
            "oracle_brier",
        )
        baseline_scores = regime_result["baseline_validation_nll"]
        if not isinstance(baseline_scores, Mapping) or set(baseline_scores) != set(_BASELINE_NAMES):
            raise ValueError("result regime has an invalid baseline validation-NLL map")
        expected_selected = (
            "ewma"
            if float(baseline_scores["ewma"]) <= float(baseline_scores["logistic_ar"])
            else "logistic_ar"
        )
        if regime_result["selected_observation_baseline"] != expected_selected:
            raise ValueError("result regime baseline winner is inconsistent with validation NLL")
        regime_hashes = regime_result["hashes"]
        if not isinstance(regime_hashes, Mapping) or set(regime_hashes) != {
            "decoder_config",
            "partition",
            "partition_content",
            "baseline_fit_policy",
        }:
            raise ValueError("result regime has an invalid policy hash set")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in regime_hashes.values()
        ):
            raise ValueError("result regime contains an invalid policy digest")
        if reference_decoder_policy is None:
            reference_decoder_policy = regime_hashes["decoder_config"]
            reference_baseline_policy = regime_hashes["baseline_fit_policy"]
        elif (
            regime_hashes["decoder_config"] != reference_decoder_policy
            or regime_hashes["baseline_fit_policy"] != reference_baseline_policy
        ):
            raise ValueError("result regimes do not share canonical decoder and baseline policies")
        arm_lookup[str(regime)] = regime_result["arms"]
        reference_membership: tuple[str, ...] | None = None
        reference_partition: tuple[object, object, object] | None = None
        for arm_name, arm in regime_result["arms"].items():
            if not isinstance(arm, Mapping):
                raise TypeError(f"result arm {arm_name!r} is invalid")
            sequences = arm.get("sequence_summaries")
            rounds = arm.get("per_round_outcomes")
            if (
                not isinstance(sequences, list)
                or not sequences
                or not isinstance(rounds, list)
                or not rounds
            ):
                raise ValueError("result arm requires sequence and per-round outcomes")
            forecast = arm.get("forecast")
            decoder = arm.get("decoder")
            if not isinstance(forecast, Mapping) or not isinstance(decoder, Mapping):
                raise TypeError("result arm diagnostics are missing")
            if (
                arm.get("name") != arm_name
                or arm.get("regime") != regime
                or arm.get("role") != "validation"
            ):
                raise ValueError("result arm identity, regime, or role is inconsistent")
            membership = tuple(str(row["sequence_id"]) for row in sequences)
            if len(set(membership)) != len(membership) or any(
                re.fullmatch(r"[0-9a-f]{64}", identity) is None for identity in membership
            ):
                raise ValueError("result arm has invalid sequence identities")
            if reference_membership is None:
                reference_membership = membership
            elif membership != reference_membership:
                raise ValueError("result arms do not share exact sequence membership and order")
            hashes = arm.get("hashes")
            if not isinstance(hashes, Mapping) or set(hashes) != {
                "partition",
                "partition_content",
                "provenance",
                "evaluation_content",
                "hx",
                "hz",
                "logical_x",
            }:
                raise TypeError("result arm hashes are missing")
            if any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in hashes.values()):
                raise ValueError("result arm contains an invalid SHA-256 digest")
            geometry = (hashes.get("hx"), hashes.get("hz"), hashes.get("logical_x"))
            if reference_geometry is None:
                reference_geometry = geometry
            elif geometry != reference_geometry:
                raise ValueError("result arms do not share exact QEC geometry hashes")
            partition_evidence = (
                hashes["partition"],
                hashes["partition_content"],
                hashes["evaluation_content"],
            )
            if reference_partition is None:
                reference_partition = partition_evidence
            elif partition_evidence != reference_partition:
                raise ValueError("result arms do not share exact partition and evaluation content")
            if (
                hashes["partition"] != regime_hashes["partition"]
                or hashes["partition_content"] != regime_hashes["partition_content"]
            ):
                raise ValueError("result arm partition hashes disagree with its regime")
            counts = arm.get("parameter_count")
            if (
                not isinstance(counts, Mapping)
                or set(counts) != {"stored", "effective"}
                or any(type(counts[name]) is not int or counts[name] < 0 for name in counts)
            ):
                raise ValueError("result arm parameter accounting is invalid")
            _assert_close(
                forecast.get("overall_nll"),
                float(np.mean([row["nll"] for row in sequences])),
                "overall_nll",
            )
            _assert_close(
                forecast.get("overall_brier"),
                float(np.mean([row["brier"] for row in sequences])),
                "overall_brier",
            )
            _assert_close(
                decoder.get("overall_bler"),
                float(np.mean([row["logical_failure"] for row in rounds])),
                "overall_bler",
            )
            _assert_close(
                decoder.get("convergence_rate"),
                float(np.mean([row["converged"] for row in rounds])),
                "convergence_rate",
            )
            _assert_close(
                decoder.get("mean_iterations"),
                float(np.mean([row["iterations"] for row in rounds])),
                "mean_iterations",
            )
            _assert_close(
                decoder.get("mean_correction_weight"),
                float(np.mean([row["correction_weight"] for row in rounds])),
                "mean_correction_weight",
            )
            if decoder.get("all_syndrome_valid") is not all(
                row["syndrome_valid"] for row in rounds
            ):
                raise ValueError("result recomputation mismatch: syndrome validity")
            grouped: dict[str, list[bool]] = defaultdict(list)
            grouped_nll: dict[str, list[float]] = defaultdict(list)
            grouped_brier: dict[str, list[float]] = defaultdict(list)
            round_membership: set[tuple[str, int]] = set()
            for row in rounds:
                identity = str(row["sequence_id"])
                round_identity = (identity, int(row["round"]))
                if round_identity in round_membership:
                    raise ValueError("result arm has duplicate per-round outcomes")
                round_membership.add(round_identity)
                grouped[identity].append(bool(row["logical_failure"]))
                grouped_nll[identity].append(float(row["forecast_nll"]))
                grouped_brier[identity].append(float(row["forecast_brier"]))
            for row in sequences:
                identity = str(row["sequence_id"])
                if identity not in grouped:
                    raise ValueError("result sequence summary has no per-round outcomes")
                _assert_close(row.get("bler"), float(np.mean(grouped[identity])), "sequence_bler")
                _assert_close(row.get("nll"), float(np.mean(grouped_nll[identity])), "sequence_nll")
                _assert_close(
                    row.get("brier"), float(np.mean(grouped_brier[identity])), "sequence_brier"
                )
            if set(grouped) != set(membership):
                raise ValueError("per-round outcomes contain undeclared sequence membership")

    progressions = payload["progressions"]
    if not isinstance(progressions, Mapping):
        raise TypeError("result progressions must be an object")
    expected_progressions = tuple(regime for regime in REGIMES if regime != "stationary_iid")
    if set(progressions) != set(expected_progressions):
        raise ValueError("result does not contain the exact B-E progression set")
    for regime, progression in progressions.items():
        if not isinstance(progression, Mapping):
            raise TypeError("result progression is invalid")
        if (
            progression.get("scope") != "descriptive_reduced_non_scientific"
            or progression.get("p_value") is not None
            or progression.get("hypothesis_status") is not None
        ):
            raise ValueError("result progression carries forbidden inferential labels")
        arms = arm_lookup[str(regime)]
        selected_name = str(progression["selected_name"])
        if selected_name != regimes[regime]["selected_observation_baseline"]:
            raise ValueError("result progression does not use the frozen baseline winner")
        stationary = arms["stationary_field"]
        selected = arms[selected_name]
        stationary_forecast = stationary["forecast"]
        selected_forecast = selected["forecast"]
        stationary_decoder = stationary["decoder"]
        selected_decoder = selected["decoder"]
        nll_gain = float(stationary_forecast["overall_nll"]) - float(
            selected_forecast["overall_nll"]
        )
        bler_difference = float(selected_decoder["overall_bler"]) - float(
            stationary_decoder["overall_bler"]
        )
        _assert_close(progression.get("nll_improvement"), nll_gain, "progression_nll")
        _assert_close(progression.get("bler_difference"), bler_difference, "progression_bler")
        if progression.get("progressed") is not (nll_gain > 0.0 and bler_difference <= 0.0):
            raise ValueError("result recomputation mismatch: progression decision")

    factor = payload["factor_diagnostics"]
    if not isinstance(factor, Mapping):
        raise TypeError("factor diagnostics must be present")
    if (
        factor.get("scope") != "descriptive_reduced_non_scientific"
        or factor.get("p_value") is not None
        or factor.get("hypothesis_status") is not None
    ):
        raise ValueError("factor diagnostics carry forbidden inferential labels")

    def interaction(regime: str) -> np.ndarray:
        summaries = {
            name: np.asarray(
                [row["bler"] for row in arm_lookup[regime][name]["sequence_summaries"]],
                dtype=np.float64,
            )
            for name in _CELL_NAMES
        }
        return (summaries["cnn_fir"] - summaries["fno_fir"]) - (
            summaries["cnn_hippo"] - summaries["fno_hippo"]
        )

    in_basis = interaction("joint_in_basis")
    mismatch = interaction("joint_basis_mismatch")
    if not np.allclose(
        factor.get("in_basis_interaction"), in_basis, rtol=0.0, atol=1e-12
    ) or not np.allclose(factor.get("basis_mismatch_interaction"), mismatch, rtol=0.0, atol=1e-12):
        raise ValueError("result recomputation mismatch: factor interaction")
    in_mean = float(in_basis.mean())
    mismatch_mean = float(mismatch.mean())
    _assert_close(factor.get("in_basis_mean"), in_mean, "in_basis_mean")
    _assert_close(factor.get("basis_mismatch_mean"), mismatch_mean, "mismatch_mean")
    if factor.get("in_basis_supports_predeclared_direction") is not (in_mean < 0.0):
        raise ValueError("result recomputation mismatch: in-basis direction")
    if factor.get("basis_mismatch_retains_predeclared_direction") is not (
        in_mean < 0.0 and mismatch_mean < 0.0
    ):
        raise ValueError("result recomputation mismatch: mismatch direction")


def _publish_screen_result(
    output_dir: Path,
    scientific_payload: Mapping[str, object],
    *,
    timing: Mapping[str, object],
    evidence: Mapping[str, _RegimeEvidence],
    config_path: Path,
    sequence_dir: Path,
    repository: _RepositoryEvidence,
) -> dict[str, Any]:
    """Publish deterministic science first, isolated timings second, manifest last."""

    recompute_screen_result(scientific_payload)
    if set(evidence) != set(REGIMES):
        raise ValueError("screen evidence must cover exact A-E regimes")
    for regime in REGIMES:
        item = evidence[regime]
        if type(item) is not _RegimeEvidence or item._token is not _EVIDENCE_TOKEN:
            raise TypeError("screen publication requires runner-created typed evidence")
        paired = item.evaluation
        expected_regime = {
            "selected_observation_baseline": item.selection.selected_name,
            "baseline_validation_nll": dict(item.selection.validation_nll),
            "hashes": {
                "decoder_config": paired.decoder_config_digest,
                "partition": paired.partition_digest,
                "partition_content": paired.partition_content_digest,
                "baseline_fit_policy": item.selection.fit_policy_digest,
            },
            "oracle_generator_sanity": scientific_payload["regimes"][regime][
                "oracle_generator_sanity"
            ],
            "arms": {name: _serialize_arm(arm)[0] for name, arm in paired.arms.items()},
        }
        if _mapping_digest(expected_regime) != _mapping_digest(
            scientific_payload["regimes"][regime]
        ):
            raise ValueError("scientific result does not match runner-created typed evidence")
    result_bytes = _canonical_json(scientific_payload)
    if output_dir.exists():
        completion = output_dir / _MANIFEST
        if not completion.is_file():
            raise FileExistsError(f"refuse to overwrite incomplete screen run: {output_dir}")
        if (
            not (output_dir / "results.json").is_file()
            or (output_dir / "results.json").read_bytes() != result_bytes
        ):
            raise FileExistsError(f"refuse to overwrite completed differing run: {output_dir}")
        verify_screen_result(
            output_dir,
            config_path=config_path,
            sequence_dir=sequence_dir,
            _repository=repository,
        )
        return _load_json(completion)

    def write(staging: Path) -> None:
        _write_bytes(staging / "results.json", result_bytes)
        _write_bytes(staging / "timing.json", _canonical_json(timing))
        for regime in REGIMES:
            _write_regime_evidence(staging / "evidence" / regime, evidence[regime])
        declared = sorted(
            str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
        )
        completion = {
            "artifact_mode": "reduced_non_scientific",
            "complete": True,
            "payloads": {name: sha256_file(staging / name) for name in declared},
            "timing_is_scientific": False,
        }
        _write_bytes(staging / _MANIFEST, _canonical_json(completion))
        verify_screen_result(
            staging,
            config_path=config_path,
            sequence_dir=sequence_dir,
            _repository=repository,
        )

    _publish_directory(output_dir, write)
    return _load_json(output_dir / _MANIFEST)


def verify_screen_result(
    output_dir: Path,
    *,
    config_path: Path,
    sequence_dir: Path,
    _repository: _RepositoryEvidence | None = None,
) -> dict[str, Any]:
    """Verify hashes and recompute the deterministic scientific result payload."""

    repository = _repository or _repository_evidence()
    manifest = _load_json(output_dir / _MANIFEST)
    if set(manifest) != {"artifact_mode", "complete", "payloads", "timing_is_scientific"}:
        raise ValueError("screen completion manifest has missing or unknown fields")
    if (
        manifest.get("artifact_mode") != "reduced_non_scientific"
        or manifest.get("complete") is not True
        or manifest.get("timing_is_scientific") is not False
    ):
        raise ValueError("screen completion manifest has invalid scope")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, Mapping) or not set(_RESULT_FILES).issubset(payloads):
        raise ValueError("screen completion payload set is invalid")
    actual_files = {
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
    }
    if actual_files != {*payloads, _MANIFEST}:
        raise ValueError("screen run contains missing or undeclared files")
    for name in payloads:
        if sha256_file(output_dir / name) != payloads[name]:
            raise ValueError(f"screen payload SHA-256 mismatch: {name}")
    result = _load_json(output_dir / "results.json")
    recompute_screen_result(result)
    source = result.get("source")
    if not isinstance(source, Mapping):
        raise TypeError("screen result source hashes are missing")
    if source.get("config_sha256") != sha256_file(config_path):
        raise ValueError("screen result configuration SHA-256 mismatch")
    if source.get("sequences_manifest_sha256") != sha256_file(sequence_dir / _MANIFEST):
        raise ValueError("screen result sequence-manifest SHA-256 mismatch")
    if source.get("source_commit") != repository.commit:
        raise ValueError("screen result source commit does not match the locked environment")
    _replay_screen_result(
        output_dir=output_dir,
        config_path=config_path,
        sequence_dir=sequence_dir,
        persisted=result,
    )
    return manifest


def _load_regime_batches(
    *,
    config: CausalExperimentConfig,
    sequence_dir: Path,
    root: Mapping[str, object],
    code: CSSCode,
    regime: str,
) -> tuple[
    SequenceRoleBatch,
    SequenceRoleBatch,
    SequenceRoleBatch,
    CausalEvaluationBatch,
    dict[str, object],
]:
    by_role: dict[str, list[tuple[object, object, object, str]]] = defaultdict(list)
    rows = root["sequences"]
    if not isinstance(rows, list):
        raise TypeError("sequence campaign membership is missing")
    for row in rows:
        if row["regime"] != regime:
            continue
        artifact = sequence_dir / str(row["path"])
        observed, supervision, diagnostics, _ = read_verified_sequence(
            artifact,
            config=config,
            code=code,
            expected_source_commit=str(root["source_commit"]),
        )
        by_role[str(row["role"])].append(
            (observed, supervision, diagnostics, sha256_file(artifact / _MANIFEST))
        )

    batches: dict[str, SequenceRoleBatch] = {}
    for role in _ROLES:
        records = by_role[role]
        if len(records) != _role_size(config, role):
            raise ValueError(f"sequence campaign has incorrect {regime}/{role} membership")
        masks = [np.asarray(record[0].scored_mask) for record in records]
        if any(not np.array_equal(masks[0], mask) for mask in masks[1:]):
            raise ValueError("role sequences do not share one scored-round policy")
        batches[role] = SequenceRoleBatch(
            role=role,
            seed=config.campaign_seed,
            syndromes=torch.as_tensor(
                np.stack([record[0].syndromes for record in records])
            ).float(),
            targets=torch.as_tensor(np.stack([record[1].errors for record in records])).float(),
            scored_mask=torch.as_tensor(np.array(masks[0], copy=True), dtype=torch.bool),
            sequence_ids=tuple(record[3] for record in records),
        )
    validation_records = by_role["validation"]
    validation = batches["validation"]
    evaluation = CausalEvaluationBatch(
        regime=regime,
        role="validation",
        sequence_ids=validation.sequence_ids,
        syndromes=validation.syndromes.numpy().astype(np.uint8),
        errors=validation.targets.numpy().astype(np.uint8),
        logical_flips=np.stack([record[1].logical_flips for record in validation_records]),
        scored_mask=validation.scored_mask.numpy(),
    )
    oracle = evaluate_oracle_sanity(
        np.stack([record[2].probabilities for record in validation_records]),
        evaluation.errors,
        evaluation.scored_mask,
    )
    return batches["train"], validation, batches["calibration"], evaluation, oracle


def _audit_model(
    model: CausalChannelForecaster, evaluation: CausalEvaluationBatch
) -> dict[str, object]:
    forecast_round = max(1, evaluation.syndromes.shape[1] // 2)
    source = CausalAuditSequence(
        evaluation.syndromes[0],
        forecast_round,
        physical_errors=evaluation.errors[0],
        logical_outcomes=evaluation.logical_flips[0],
        diagnostics={"privileged": np.zeros(evaluation.syndromes.shape[1], dtype=np.uint8)},
    )
    future = np.array(evaluation.syndromes[0], copy=True)
    future[forecast_round:] ^= 1
    mutations = {
        "current_future_syndromes": CausalAuditSequence(
            future,
            forecast_round,
            physical_errors=evaluation.errors[0],
            logical_outcomes=evaluation.logical_flips[0],
            diagnostics=source.diagnostics,
        ),
        "physical_error_labels": CausalAuditSequence(
            evaluation.syndromes[0],
            forecast_round,
            physical_errors=1 - evaluation.errors[0],
            logical_outcomes=evaluation.logical_flips[0],
            diagnostics=source.diagnostics,
        ),
        "logical_labels": CausalAuditSequence(
            evaluation.syndromes[0],
            forecast_round,
            physical_errors=evaluation.errors[0],
            logical_outcomes=1 - evaluation.logical_flips[0],
            diagnostics=source.diagnostics,
        ),
        "privileged_diagnostics": CausalAuditSequence(
            evaluation.syndromes[0],
            forecast_round,
            physical_errors=evaluation.errors[0],
            logical_outcomes=evaluation.logical_flips[0],
            diagnostics={"privileged": np.ones(evaluation.syndromes.shape[1], dtype=np.uint8)},
        ),
    }
    audit = audit_structural_prefix_causality(model, source, mutations)
    return {
        "passed": audit.passed,
        "forecast_round": audit.forecast_round,
        "checks": [
            {"name": check.name, "bit_identical": check.bit_identical} for check in audit.checks
        ],
    }


def _serialize_arm(arm: ArmEvaluation) -> tuple[dict[str, object], dict[str, object]]:
    sequence_rows = [
        {
            "sequence_id": identity,
            "nll": float(arm.per_sequence_nll[index]),
            "brier": float(arm.per_sequence_brier[index]),
            "bler": float(arm.per_sequence_bler[index]),
        }
        for index, identity in enumerate(arm.evaluation_sequence_ids)
    ]
    outcomes: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    for row in arm.per_round_outcomes:
        science = dict(row)
        setup = science.pop("setup_latency_seconds")
        decode = science.pop("decode_latency_seconds")
        science["predicted_observables"] = list(science["predicted_observables"])
        science["true_logical_flips"] = list(science["true_logical_flips"])
        outcomes.append(science)
        timing_rows.append(
            {
                "sequence_id": science["sequence_id"],
                "round": science["round"],
                "setup_latency_seconds": setup,
                "decode_latency_seconds": decode,
            }
        )
    scientific = {
        "name": arm.name,
        "predictor_type": arm.predictor_type,
        "regime": arm.regime,
        "role": arm.role,
        "parameter_count": {
            "stored": arm.stored_parameters,
            "effective": arm.effective_parameters,
        },
        "forecast": {
            "overall_nll": arm.overall_nll,
            "overall_brier": arm.overall_brier,
            "ece": arm.expected_calibration_error,
            "reliability": [dict(row) for row in arm.reliability],
        },
        "decoder": {
            "overall_bler": arm.overall_bler,
            "convergence_rate": arm.convergence_rate,
            "mean_iterations": arm.mean_iterations,
            "mean_correction_weight": arm.mean_correction_weight,
            "all_syndrome_valid": arm.all_syndrome_valid,
        },
        "hashes": {
            "partition": arm.partition_digest,
            "partition_content": arm.partition_content_digest,
            "provenance": arm.provenance_digest,
            "evaluation_content": arm.evaluation_content_digest,
            "hx": arm.qec_hx_sha256,
            "hz": arm.qec_hz_sha256,
            "logical_x": arm.qec_logical_x_sha256,
        },
        "sequence_summaries": sequence_rows,
        "per_round_outcomes": outcomes,
    }
    estimator_per_round = arm.estimator_batch_seconds / len(outcomes)
    timing = {
        "scope": "engineering_measurement_no_speed_claim",
        "estimator_batch_seconds": arm.estimator_batch_seconds,
        "estimator_amortized_per_round_seconds": estimator_per_round,
        "bp_lsd_per_round_p50_seconds": arm.latency_p50_seconds,
        "bp_lsd_per_round_p95_seconds": arm.latency_p95_seconds,
        "bp_lsd_per_round_p99_seconds": arm.latency_p99_seconds,
        "end_to_end_estimated_per_round_p50_seconds": (
            estimator_per_round + arm.latency_p50_seconds
        ),
        "end_to_end_estimated_per_round_p95_seconds": (
            estimator_per_round + arm.latency_p95_seconds
        ),
        "end_to_end_estimated_per_round_p99_seconds": (
            estimator_per_round + arm.latency_p99_seconds
        ),
        "per_round": timing_rows,
    }
    return scientific, timing


def _write_regime_evidence(target: Path, evidence: _RegimeEvidence) -> None:
    if tuple(predictor.name for predictor in evidence.predictors) != _ARM_NAMES:
        raise ValueError("regime evidence does not contain the canonical predictor order")
    if tuple(evidence.evaluation.arms) != _ARM_NAMES:
        raise ValueError("regime evidence does not contain the canonical evaluated arm order")
    target.mkdir(parents=True, exist_ok=False)
    selection = {
        "selected_name": evidence.selection.selected_name,
        "validation_nll": dict(evidence.selection.validation_nll),
        "artifact_digest": evidence.selection.artifact_digest,
        "partition_digest": evidence.selection.partition_digest,
        "partition_content_digest": evidence.selection.partition_content_digest,
        "fit_policy_digest": evidence.selection.fit_policy_digest,
        "train_sequence_ids": list(evidence.selection.train_sequence_ids),
        "validation_sequence_ids": list(evidence.selection.validation_sequence_ids),
        "calibration_sequence_ids": list(evidence.selection.calibration_sequence_ids),
        "evaluation_content_digest": evidence.selection.evaluation_content_digest,
    }
    _write_bytes(target / "selection.json", _canonical_json(selection))
    for predictor in evidence.predictors:
        arm_dir = target / predictor.name
        arm_dir.mkdir()
        metadata, arrays = export_frozen_predictor(predictor)
        _write_bytes(arm_dir / "predictor.json", _canonical_json(metadata))
        for name, values in arrays.items():
            _write_npy(arm_dir / "predictor" / f"{name}.npy", values)
        evaluated = evidence.evaluation.arms[predictor.name]
        outcomes = evaluated.per_round_outcomes
        trace_metadata = {
            "sequence_ids": list(evaluated.evaluation_sequence_ids),
            "sequence_membership": [list(row) for row in evaluated.sequence_membership],
            "round_zero_prior": "causal_empty_history_unscored",
            "scored_rounds_only_for_decoder": True,
            "arrays": [
                "priors",
                "corrections",
                "predicted_observables",
                "syndrome_valid",
                "converged",
                "iterations",
            ],
        }
        _write_bytes(arm_dir / "trace.json", _canonical_json(trace_metadata))
        trace_arrays = {
            "priors": evaluated.priors,
            "corrections": evaluated.corrections,
            "predicted_observables": np.asarray(
                [row["predicted_observables"] for row in outcomes], dtype=np.uint8
            ),
            "syndrome_valid": np.asarray(
                [row["syndrome_valid"] for row in outcomes], dtype=np.bool_
            ),
            "converged": np.asarray([row["converged"] for row in outcomes], dtype=np.bool_),
            "iterations": np.asarray([row["iterations"] for row in outcomes], dtype=np.int64),
        }
        for name, values in trace_arrays.items():
            _write_npy(arm_dir / "trace" / f"{name}.npy", values)


def _load_frozen_evidence(arm_dir: Path) -> FrozenArm | FrozenBaseline:
    metadata = _load_json(arm_dir / "predictor.json")
    array_dir = arm_dir / "predictor"
    arrays = {
        path.stem: np.load(path, allow_pickle=False) for path in sorted(array_dir.glob("*.npy"))
    }
    return restore_frozen_predictor(metadata, arrays)


def _verify_trace(arm_dir: Path, arm: ArmEvaluation) -> None:
    metadata = _load_json(arm_dir / "trace.json")
    if metadata != {
        "sequence_ids": list(arm.evaluation_sequence_ids),
        "sequence_membership": [list(row) for row in arm.sequence_membership],
        "round_zero_prior": "causal_empty_history_unscored",
        "scored_rounds_only_for_decoder": True,
        "arrays": [
            "priors",
            "corrections",
            "predicted_observables",
            "syndrome_valid",
            "converged",
            "iterations",
        ],
    }:
        raise ValueError("persisted trace metadata does not match replayed membership")
    outcomes = arm.per_round_outcomes
    expected = {
        "priors": arm.priors,
        "corrections": arm.corrections,
        "predicted_observables": np.asarray(
            [row["predicted_observables"] for row in outcomes], dtype=np.uint8
        ),
        "syndrome_valid": np.asarray([row["syndrome_valid"] for row in outcomes], dtype=np.bool_),
        "converged": np.asarray([row["converged"] for row in outcomes], dtype=np.bool_),
        "iterations": np.asarray([row["iterations"] for row in outcomes], dtype=np.int64),
    }
    trace_dir = arm_dir / "trace"
    actual_files = {path.stem for path in trace_dir.glob("*.npy")}
    if actual_files != set(expected):
        raise ValueError("persisted trace has a missing or unknown array")
    for name, values in expected.items():
        observed = np.load(trace_dir / f"{name}.npy", allow_pickle=False)
        if not np.array_equal(observed, values):
            raise ValueError(f"persisted trace disagrees with exact replay: {name}")


def _replay_screen_result(
    *,
    output_dir: Path,
    config_path: Path,
    sequence_dir: Path,
    persisted: Mapping[str, object],
) -> None:
    """Reconstruct predictors and rerun exact BP-LSD against regenerated sources."""

    config = CausalExperimentConfig.from_json(config_path)
    root = verify_sequence_campaign(
        config_path=config_path,
        output_dir=sequence_dir,
        regenerate=True,
    )
    code = _canonical_code()
    logical_x = logical_x_basis(code.hx, code.hz)
    replayed_overfit: dict[str, object] = {}
    for spatial, temporal in _CELLS:
        name = f"{spatial}_{temporal}"
        fixture_model = build_forecaster(spatial=spatial, temporal=temporal, config=config)
        fixture_result = overfit_causal_forecaster(
            fixture_model,
            fixture=build_overfit_fixture(config),
            config=config,
        )
        replayed_overfit[name] = {
            "steps": fixture_result.steps,
            "nll": fixture_result.metrics.nll,
            "accuracy": fixture_result.metrics.accuracy,
        }
    replayed_evaluations: dict[str, PairedCausalEvaluation] = {}
    replayed_progressions: dict[str, object] = {}
    replayed_audits: dict[str, object] = {}
    for regime in REGIMES:
        train, validation, calibration, evaluation, oracle = _load_regime_batches(
            config=config,
            sequence_dir=sequence_dir,
            root=root,
            code=code,
            regime=regime,
        )
        partition = validate_role_partition(train, validation, calibration, regime=regime)
        selection = fit_select_observation_baselines(
            train=train,
            validation=validation,
            calibration=calibration,
            partition=partition,
        )
        predictors = tuple(
            _load_frozen_evidence(output_dir / "evidence" / regime / name) for name in _ARM_NAMES
        )
        for name in _BASELINE_NAMES:
            loaded = next(predictor for predictor in predictors if predictor.name == name)
            if loaded.artifact_digest != selection.baseline(name).artifact_digest:
                raise ValueError("persisted baseline state disagrees with exact source-data refit")
        for predictor in predictors:
            if type(predictor) is FrozenArm:
                validate_frozen_arm_calibration(
                    predictor,
                    calibration,
                    partition=partition,
                )
        paired = evaluate_causal_arms(
            evaluation,
            predictors,
            hx=code.hx,
            hz=code.hz,
            logical_x=logical_x,
        )
        replayed_evaluations[regime] = paired
        for name, arm in paired.arms.items():
            _verify_trace(output_dir / "evidence" / regime / name, arm)
        science_arms = {name: _serialize_arm(arm)[0] for name, arm in paired.arms.items()}
        replayed_regime = {
            "selected_observation_baseline": selection.selected_name,
            "baseline_validation_nll": dict(selection.validation_nll),
            "hashes": {
                "decoder_config": paired.decoder_config_digest,
                "partition": paired.partition_digest,
                "partition_content": paired.partition_content_digest,
                "baseline_fit_policy": selection.fit_policy_digest,
            },
            "oracle_generator_sanity": oracle,
            "arms": science_arms,
        }
        if _mapping_digest(replayed_regime) != _mapping_digest(persisted["regimes"][regime]):
            raise ValueError(f"persisted result disagrees with exact replay for {regime}")
        replayed_audits[regime] = {
            predictor.name: _audit_model(predictor.materialize(), evaluation)
            for predictor in predictors
            if type(predictor) is FrozenArm
        }
        if regime != "stationary_iid":
            progression = reduced_progression(
                selection=selection,
                stationary=paired.arms["stationary_field"],
                selected=paired.arms[selection.selected_name],
            )
            replayed_progressions[regime] = {
                "selected_name": progression.selected_name,
                "nll_improvement": progression.nll_improvement,
                "bler_difference": progression.bler_difference,
                "progressed": progression.progressed,
                "scope": progression.scope,
                "p_value": None,
                "hypothesis_status": None,
            }
    if _mapping_digest(replayed_progressions) != _mapping_digest(persisted["progressions"]):
        raise ValueError("persisted progressions disagree with exact Task9 replay")
    in_basis = replayed_evaluations["joint_in_basis"]
    mismatch = replayed_evaluations["joint_basis_mismatch"]
    factor = reduced_factor_diagnostics(
        in_basis_arms={name: in_basis.arms[name] for name in _CELL_NAMES},
        basis_mismatch_arms={name: mismatch.arms[name] for name in _CELL_NAMES},
    )
    replayed_factor = {
        "in_basis_interaction": factor.in_basis_interaction.tolist(),
        "basis_mismatch_interaction": factor.basis_mismatch_interaction.tolist(),
        "in_basis_mean": factor.in_basis_mean,
        "basis_mismatch_mean": factor.basis_mismatch_mean,
        "in_basis_supports_predeclared_direction": factor.in_basis_supports_predeclared_direction,
        "basis_mismatch_retains_predeclared_direction": factor.basis_mismatch_retains_predeclared_direction,
        "scope": factor.scope,
        "p_value": None,
        "hypothesis_status": None,
    }
    if _mapping_digest(replayed_factor) != _mapping_digest(persisted["factor_diagnostics"]):
        raise ValueError("persisted factor diagnostics disagree with exact Task9 replay")
    audit = persisted["causal_audit"]
    if _mapping_digest(replayed_audits) != _mapping_digest(audit["arms"]):
        raise ValueError("persisted causal audit disagrees with reconstructed predictors")
    if _mapping_digest(replayed_overfit) != _mapping_digest(audit["overfit_fixture"]):
        raise ValueError("persisted overfit fixture disagrees with deterministic replay")


def run_reduced_screen(
    *,
    config_path: Path,
    sequence_dir: Path,
    output_dir: Path,
    _repository: _RepositoryEvidence | None = None,
) -> dict[str, Any]:
    """Run the complete reduced screen without attaching confirmatory inference."""

    started = time.perf_counter()
    repository = _repository or _repository_evidence()
    _require_ignored_repository_output(repository, output_dir)
    config = CausalExperimentConfig.from_json(config_path)
    if config.artifact_mode != "reduced_non_scientific":
        raise ValueError("factor screen requires reduced_non_scientific configuration")
    root = verify_sequence_campaign(
        config_path=config_path, output_dir=sequence_dir, regenerate=False
    )
    code = _canonical_code()
    logical_x = logical_x_basis(code.hx, code.hz)
    regime_results: dict[str, object] = {}
    timing_regimes: dict[str, object] = {}
    progressions: dict[str, object] = {}
    causal_arms: dict[str, object] = {}
    evaluations: dict[str, PairedCausalEvaluation] = {}
    screen_evidence: dict[str, _RegimeEvidence] = {}
    overfit: dict[str, object] = {}
    for spatial, temporal in _CELLS:
        name = f"{spatial}_{temporal}"
        fixture_model = build_forecaster(spatial=spatial, temporal=temporal, config=config)
        fixture_result = overfit_causal_forecaster(
            fixture_model, fixture=build_overfit_fixture(config), config=config
        )
        if (
            fixture_result.metrics.nll > config.overfit_fixture.nll_threshold
            or fixture_result.metrics.accuracy < config.overfit_fixture.accuracy_threshold
        ):
            raise RuntimeError(f"{name} failed the mandatory overfit fixture")
        overfit[name] = {
            "steps": fixture_result.steps,
            "nll": fixture_result.metrics.nll,
            "accuracy": fixture_result.metrics.accuracy,
        }

    for regime in REGIMES:
        train, validation, calibration, evaluation, oracle = _load_regime_batches(
            config=config,
            sequence_dir=sequence_dir,
            root=root,
            code=code,
            regime=regime,
        )
        partition = validate_role_partition(train, validation, calibration, regime=regime)
        selection = fit_select_observation_baselines(
            train=train,
            validation=validation,
            calibration=calibration,
            partition=partition,
        )
        frozen: list[FrozenBaseline | FrozenArm] = [
            selection.baseline("stationary_field"),
            selection.baseline("ewma"),
            selection.baseline("logistic_ar"),
        ]
        audit_rows: dict[str, object] = {}
        for spatial, temporal in _CELLS:
            name = f"{spatial}_{temporal}"
            model = build_forecaster(spatial=spatial, temporal=temporal, config=config)
            trained = train_causal_forecaster(
                model,
                train=train,
                validation=validation,
                config=config,
                partition=partition,
            )
            audit_rows[name] = _audit_model(model, evaluation)
            frozen.append(
                freeze_learned_arm(
                    name=name,
                    model=model,
                    config=config,
                    partition=partition,
                    training_result=trained,
                    train=train,
                    validation=validation,
                    calibration=calibration,
                )
            )
        paired = evaluate_causal_arms(
            evaluation,
            tuple(frozen),
            hx=code.hx,
            hz=code.hz,
            logical_x=logical_x,
        )
        evaluations[regime] = paired
        screen_evidence[regime] = _RegimeEvidence(
            selection=selection,
            predictors=tuple(frozen),
            evaluation=paired,
            _token=_EVIDENCE_TOKEN,
        )
        causal_arms[regime] = audit_rows
        science_arms: dict[str, object] = {}
        timing_arms: dict[str, object] = {}
        for name, arm in paired.arms.items():
            science_arms[name], timing_arms[name] = _serialize_arm(arm)
        regime_results[regime] = {
            "selected_observation_baseline": selection.selected_name,
            "baseline_validation_nll": dict(selection.validation_nll),
            "hashes": {
                "decoder_config": paired.decoder_config_digest,
                "partition": paired.partition_digest,
                "partition_content": paired.partition_content_digest,
                "baseline_fit_policy": selection.fit_policy_digest,
            },
            "oracle_generator_sanity": oracle,
            "arms": science_arms,
        }
        timing_regimes[regime] = timing_arms
        if regime != "stationary_iid":
            progression = reduced_progression(
                selection=selection,
                stationary=paired.arms["stationary_field"],
                selected=paired.arms[selection.selected_name],
            )
            progressions[regime] = {
                "selected_name": progression.selected_name,
                "nll_improvement": progression.nll_improvement,
                "bler_difference": progression.bler_difference,
                "progressed": progression.progressed,
                "scope": progression.scope,
                "p_value": None,
                "hypothesis_status": None,
            }

    in_basis = evaluations["joint_in_basis"]
    mismatch = evaluations["joint_basis_mismatch"]
    factor = reduced_factor_diagnostics(
        in_basis_arms={name: in_basis.arms[name] for name in _CELL_NAMES},
        basis_mismatch_arms={name: mismatch.arms[name] for name in _CELL_NAMES},
    )
    factor_payload = {
        "in_basis_interaction": factor.in_basis_interaction.tolist(),
        "basis_mismatch_interaction": factor.basis_mismatch_interaction.tolist(),
        "in_basis_mean": factor.in_basis_mean,
        "basis_mismatch_mean": factor.basis_mismatch_mean,
        "in_basis_supports_predeclared_direction": factor.in_basis_supports_predeclared_direction,
        "basis_mismatch_retains_predeclared_direction": factor.basis_mismatch_retains_predeclared_direction,
        "scope": factor.scope,
        "p_value": None,
        "hypothesis_status": None,
    }
    scientific = {
        "artifact_mode": "reduced_non_scientific",
        "scientific_scope": "reduced_non_scientific",
        "p_value": None,
        "hypothesis_status": None,
        "source": {
            "config_sha256": sha256_file(config_path),
            "sequences_manifest_sha256": sha256_file(sequence_dir / _MANIFEST),
            "source_commit": repository.commit,
        },
        "causal_audit": {
            "passed": all(row["passed"] for arms in causal_arms.values() for row in arms.values()),
            "arms": causal_arms,
            "overfit_fixture": overfit,
        },
        "regimes": regime_results,
        "progressions": progressions,
        "factor_diagnostics": factor_payload,
    }
    timing = {
        "wall_seconds": time.perf_counter() - started,
        "scientific": False,
        "scope": "engineering_measurement_no_speed_claim",
        "regimes": timing_regimes,
    }
    return _publish_screen_result(
        output_dir,
        scientific,
        timing=timing,
        evidence=screen_evidence,
        config_path=config_path,
        sequence_dir=sequence_dir,
        repository=repository,
    )


__all__ = [
    "generate_sequence_campaign",
    "recompute_screen_result",
    "run_reduced_screen",
    "verify_screen_result",
    "verify_sequence_campaign",
]
