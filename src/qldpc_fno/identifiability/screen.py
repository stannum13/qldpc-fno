"""Execute and independently replay the temporal-identifiability gate.

Large arrays are streamed one sequence at a time. JSON completion markers bind
all numeric payloads and are published last. Test-only dependency injection
replaces expensive kernels without adding a production reduced mode.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.code_identity import sparse_binary_sha256
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.decoders.bplsd import BPLSDConfig, DecodeBatchResult, decode_bplsd_prior_batch
from qldpc_fno.identifiability.baseline_bundle import (
    BundleManifest,
    FrozenEstimatorBundle,
    fit_development_bundle,
    read_verified_bundle,
    write_frozen_bundle,
)
from qldpc_fno.identifiability.config import IdentifiabilityConfig, load_identifiability_config
from qldpc_fno.identifiability.endpoints import (
    EndpointBatch,
    EndpointSequence,
    calibration_by_sequence,
    expected_ce_by_sequence,
    latent_nmse_by_sequence,
    retained_syndrome_nll_by_sequence,
)
from qldpc_fno.identifiability.filters import (
    ForecastResult,
    forecast_contemporaneous,
    forecast_grid_bayes,
    forecast_known_marginal,
    forecast_latent_history,
    forecast_parity_moment,
)
from qldpc_fno.identifiability.grid import ExperimentDeadlineExceeded
from qldpc_fno.identifiability.inference import (
    DEPLOYABLE_HISTORY_ARMS,
    classify_bler_interval,
    conditional_decoder_arms,
    evaluate_identifiability,
    fixed_history_derangement,
)
from qldpc_fno.identifiability.observation import DisjointChecks, greedy_disjoint_rows
from qldpc_fno.identifiability.sequence_store import verify_campaign
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    DevelopmentPartitions,
    GeneratedSequence,
    LatentHistoryOracleInput,
    SequenceIdentity,
    TrainingTargets,
)
from qldpc_fno.metrics.clustered import studentized_sequence_interval

_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "temporal_identifiability_run"
_DEVELOPMENT_ROLES = ("train", "validation", "calibration")
_NORMAL_ARMS = (
    "known_marginal",
    "empirical_stationary",
    "ewma",
    "logistic_ar32",
    "parity_moment_ar",
    "grid_bayes",
    "latent_history_oracle",
    "contemporaneous_oracle",
)
_GRID_DIAGNOSTIC = "grid_bayes_doubled"
_DERANGED_SUFFIX = "__history_deranged"
_DERANGED_KEYS = tuple(f"{arm}{_DERANGED_SUFFIX}" for arm in DEPLOYABLE_HISTORY_ARMS)


def _forecast_keys(role: str) -> tuple[str, ...]:
    diagnostic = (_GRID_DIAGNOSTIC,) if role == "validation" else ()
    return (*_NORMAL_ARMS, *diagnostic, *_DERANGED_KEYS)


_SEQUENCE_ARRAYS = (
    "global_log_odds",
    "probabilities",
    "errors",
    "syndromes",
    "logical_flips",
    "scored_mask",
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "complete",
        "status",
        "mode",
        "claim",
        "source",
        "config",
        "code",
        "retained_checks",
        "sequences",
        "fisher_precheck",
        "development_record",
        "approval",
        "bundle",
        "completed_stages",
        "evidence",
        "inference",
        "decision",
        "decoder",
        "runtime",
        "identity_sha256",
        "content_sha256",
    }
)
_EVIDENCE_FIELDS = frozenset(
    {
        "regime",
        "role",
        "sequence_index",
        "sequence_content_sha256",
        "seeds",
        "path",
        "payload_sha256",
        "arms",
    }
)
_FISHER_FIELDS = frozenset(
    {
        "status",
        "provenance",
        "minimum_information",
        "median_information",
        "maximum_information",
        "cramer_rao_minimum",
        "cramer_rao_median",
        "cramer_rao_maximum",
        "maximum_derivative_error",
        "failure_reasons",
    }
)
_DEVELOPMENT_STAGE_NAMES = frozenset(
    {
        "source_config_code_support",
        "development_sequences",
        "frozen_bundle",
        "development_evidence",
        "inference_metadata",
    }
)
_CONFIRMATION_STAGE_NAMES = frozenset(
    {
        "source_config_code_support",
        "approved_development_bundle",
        "test_sequences",
        "confirmatory_evidence",
        "confirmatory_inference",
        "conditional_decoder",
    }
)


def _is_hex(value: object, size: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(character in "0123456789abcdef" for character in value)
    )


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


@dataclass(frozen=True, slots=True)
class ScreenSourceEvidence:
    """Clean Git commit and exact source-tree SHA-256 identity."""

    root: Path
    commit: str
    tree_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise TypeError("source root must be pathlib.Path")
        if not _is_hex(self.commit, 40):
            raise ValueError("source commit must be a lowercase 40-character Git object id")
        if not _is_hex(self.tree_sha256, 64):
            raise ValueError("source tree identity must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class ScreenDependencies:
    """In-process injection points for expensive integration-test kernels only."""

    source_evidence: Callable[[], ScreenSourceEvidence] | None = None
    verify_sequences: Callable[..., dict[str, Any]] | None = None
    code_factory: Callable[[], Any] = lambda: build_self_lifted_product(PAPER_LP_3_7_16)
    logical_x_factory: Callable[[sparse.spmatrix, sparse.spmatrix], sparse.spmatrix] = (
        logical_x_basis
    )
    require_canonical_code: bool = True
    fit_bundle: Callable[[DevelopmentPartitions, IdentifiabilityConfig], Any] = (
        fit_development_bundle
    )
    write_bundle: Callable[[Path, Any], BundleManifest] = write_frozen_bundle
    read_bundle: Callable[[Path, BundleManifest], Any] = read_verified_bundle
    forecast_sequence: Callable[..., Mapping[str, ForecastResult]] | None = None
    score_sequence: Callable[..., Mapping[str, object]] | None = None
    inference: Callable[..., dict[str, object]] = evaluate_identifiability
    decoder: Callable[..., DecodeBatchResult] = decode_bplsd_prior_batch
    process_time: Callable[[], float] = time.process_time
    wall_time: Callable[[], float] = time.perf_counter
    peak_memory_bytes: Callable[[], int] = _peak_rss_bytes


class _CpuDeadline:
    def __init__(self, *, started: float, limit: float, clock: Callable[[], float]) -> None:
        self.started = started
        self.limit = limit
        self.absolute = started + limit
        self._clock = clock

    def elapsed(self) -> float:
        value = float(self._clock()) - self.started
        if not math.isfinite(value):
            raise RuntimeError("process CPU clock returned a nonfinite value")
        return max(0.0, value)

    def check(self) -> None:
        if self.elapsed() > self.limit:
            raise ExperimentDeadlineExceeded(
                "temporal identifiability process CPU deadline exceeded"
            )


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode())
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _array_descriptor(array: np.ndarray, *, member: str) -> dict[str, object]:
    value = np.asarray(array)
    return {
        "member": member,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": _array_digest(value),
    }


def _stable_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in arrays.items():
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.ascontiguousarray(value), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(
                member, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9
            )
    return output.getvalue()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _repository_evidence() -> ScreenSourceEvidence:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("temporal identifiability publication requires a clean source tree")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "--full-tree", commit],
        check=True,
        capture_output=True,
    ).stdout
    return ScreenSourceEvidence(root, commit, hashlib.sha256(listing).hexdigest())


def _resolved_dependencies(value: ScreenDependencies | None) -> ScreenDependencies:
    dependencies = value or ScreenDependencies()
    if type(dependencies) is not ScreenDependencies:
        raise TypeError("dependencies must be the exact ScreenDependencies type")
    return dependencies


def _source(dependencies: ScreenDependencies) -> ScreenSourceEvidence:
    result = (dependencies.source_evidence or _repository_evidence)()
    if type(result) is not ScreenSourceEvidence:
        raise TypeError("source evidence provider returned an invalid value")
    return result


def _code_manifest(
    code: Any,
    logical_x: sparse.spmatrix,
    config: IdentifiabilityConfig,
    *,
    require_canonical: bool,
) -> dict[str, object]:
    value = {
        "name": str(code.name),
        "ell": int(code.ell),
        "n": int(code.n),
        "k": int(code.k),
        "hx_sha256": sparse_binary_sha256(code.hx),
        "hz_sha256": sparse_binary_sha256(code.hz),
        "logical_x_sha256": sparse_binary_sha256(logical_x),
    }
    if require_canonical:
        expected = {
            "name": config.code.name,
            "ell": config.code.ell,
            "n": config.code.n,
            "k": config.code.k,
            "hx_sha256": config.code.hx_sha256,
            "hz_sha256": config.code.hz_sha256,
        }
        if {key: value[key] for key in expected} != expected:
            raise ValueError("screen requires the canonical configured code")
    return value


def _checks_manifest(checks: DisjointChecks) -> dict[str, object]:
    value: dict[str, object] = {
        "algorithm_version": checks.algorithm_version,
        "matrix_sha256": checks.matrix_sha256,
        "row_indices": checks.row_indices.tolist(),
        "supports": [support.tolist() for support in checks.supports],
        "weights": checks.weights.tolist(),
        "covered_qubits": checks.covered_qubits.tolist(),
    }
    value["content_sha256"] = _digest(value)
    return value


def _verify_sequences(
    dependencies: ScreenDependencies,
    *,
    config_path: Path,
    sequence_dir: Path,
    roles: tuple[str, ...],
    development_record: Path | None = None,
    approval_path: Path | None = None,
) -> dict[str, Any]:
    return (dependencies.verify_sequences or verify_campaign)(
        config_path=config_path,
        output_dir=sequence_dir,
        roles=roles,
        development_record=development_record,
        approval_path=approval_path,
    )


def _require_sequence_coordinates(
    root: Mapping[str, object], config: IdentifiabilityConfig, roles: tuple[str, ...]
) -> list[dict[str, Any]]:
    if root.get("roles") != list(roles):
        raise ValueError("sequence artifact roles do not match the requested study mode")
    rows = root.get("sequences")
    if not isinstance(rows, list):
        raise TypeError("sequence artifact has no strict sequence list")
    expected = [
        (role, regime, index)
        for role in roles
        for regime in config.regimes
        for index in range(int(getattr(config.splits, role)))
    ]
    actual = [
        (row.get("role"), row.get("regime"), row.get("sequence_index"))
        for row in rows
        if isinstance(row, dict)
    ]
    if actual != expected or len(actual) != len(rows):
        raise ValueError(
            "sequence artifact has missing, extra, reordered, or overlapping identities"
        )
    return rows


def _sequence_identity(row: Mapping[str, object]) -> SequenceIdentity:
    seeds = row.get("seeds")
    if not isinstance(seeds, Mapping):
        raise TypeError("sequence row has no seed mapping")
    return SequenceIdentity(
        regime=str(row["regime"]),
        role=str(row["role"]),
        sequence_index=int(row["sequence_index"]),
        latent_seed=int(seeds["latent"]),
        bernoulli_seed=int(seeds["bernoulli"]),
        content_sha256=str(row["sequence_content_sha256"]),
    )


def _development_partitions(rows: list[dict[str, Any]]) -> DevelopmentPartitions:
    values = {
        role: tuple(_sequence_identity(row) for row in rows if row["role"] == role)
        for role in _DEVELOPMENT_ROLES
    }
    return DevelopmentPartitions(values["train"], values["validation"], values["calibration"])


def _safe_relative(row: Mapping[str, object]) -> Path:
    relative = Path(str(row["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("sequence payload path must be a safe relative path")
    return relative


def _load_sequence(sequence_dir: Path, row: Mapping[str, object], code: Any) -> GeneratedSequence:
    try:
        with np.load(sequence_dir / _safe_relative(row), allow_pickle=False) as archive:
            if set(archive.files) != set(_SEQUENCE_ARRAYS):
                raise ValueError("sequence payload arrays are incomplete")
            arrays = {name: np.array(archive[name], copy=True) for name in _SEQUENCE_ARRAYS}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("sequence payload is unreadable") from error
    errors = np.asarray(arrays["errors"], dtype=np.uint8)
    syndromes = np.asarray(arrays["syndromes"], dtype=np.uint8)
    if not np.array_equal(np.asarray(errors @ code.hx.T, dtype=np.uint8) % 2, syndromes):
        raise ValueError("sequence syndrome does not reconstruct from its physical errors")
    return GeneratedSequence(
        identity=_sequence_identity(row),
        deployable=DeployableHistory(syndromes, np.asarray(arrays["scored_mask"], dtype=np.bool_)),
        latent_oracle=LatentHistoryOracleInput(arrays["global_log_odds"]),
        contemporaneous_oracle=ContemporaneousOracleInput(arrays["probabilities"]),
        targets=TrainingTargets(errors, arrays["logical_flips"]),
    )


def _fitted_forecast(
    name: str, sequence: GeneratedSequence, bundle: FrozenEstimatorBundle
) -> ForecastResult:
    model = bundle.arm(name).materialize()
    rounds, checks = sequence.deployable.syndromes.shape
    if name == "empirical_stationary":
        predicted = model.predict(sequence_count=1, rounds=rounds)[0].reshape(rounds, -1)
    else:
        if checks != 945:
            raise ValueError("fitted temporal baselines require canonical 945-check geometry")
        observed = sequence.deployable.syndromes.reshape(1, rounds, 21, 45).astype(np.float64)
        predicted = model.predict(observed)[0].reshape(rounds, -1)
    return ForecastResult(name, predicted, None, None)


def _default_forecast_sequence(
    *,
    sequence: GeneratedSequence,
    config: IdentifiabilityConfig,
    checks: DisjointChecks,
    bundle: FrozenEstimatorBundle,
    deranged_histories: Mapping[str, DeployableHistory],
    process_cpu_deadline: float,
    include_grid_diagnostic: bool,
) -> Mapping[str, ForecastResult]:
    normal: dict[str, ForecastResult] = {
        "known_marginal": forecast_known_marginal(
            sequence.deployable, checks, config, process_cpu_deadline=process_cpu_deadline
        ),
        "empirical_stationary": _fitted_forecast("empirical_stationary", sequence, bundle),
        "ewma": _fitted_forecast("ewma", sequence, bundle),
        "logistic_ar32": _fitted_forecast("logistic_ar32", sequence, bundle),
        "parity_moment_ar": forecast_parity_moment(
            sequence.deployable, checks, config, process_cpu_deadline=process_cpu_deadline
        ),
        "grid_bayes": forecast_grid_bayes(
            sequence.deployable,
            checks,
            config,
            interior_cells=config.grid.interior_cells,
            process_cpu_deadline=process_cpu_deadline,
        ),
        "latent_history_oracle": forecast_latent_history(
            sequence.latent_oracle, config, process_cpu_deadline=process_cpu_deadline
        ),
        "contemporaneous_oracle": forecast_contemporaneous(sequence.contemporaneous_oracle, config),
    }
    values: dict[str, ForecastResult] = dict(normal)
    if include_grid_diagnostic:
        doubled = forecast_grid_bayes(
            sequence.deployable,
            checks,
            config,
            interior_cells=config.grid.doubled_interior_cells,
            process_cpu_deadline=process_cpu_deadline,
        )
        values[_GRID_DIAGNOSTIC] = dataclasses.replace(doubled, arm=_GRID_DIAGNOSTIC)
    for arm in DEPLOYABLE_HISTORY_ARMS:
        history = deranged_histories[arm]
        if arm == "grid_bayes":
            forecast = forecast_grid_bayes(
                history, checks, config, process_cpu_deadline=process_cpu_deadline
            )
        elif arm == "parity_moment_ar":
            forecast = forecast_parity_moment(
                history, checks, config, process_cpu_deadline=process_cpu_deadline
            )
        else:
            forecast = _fitted_forecast(
                arm, dataclasses.replace(sequence, deployable=history), bundle
            )
        key = f"{arm}{_DERANGED_SUFFIX}"
        values[key] = dataclasses.replace(forecast, arm=key)
    return values


def _default_score_sequence(
    *, sequence: GeneratedSequence, forecast: ForecastResult, checks: DisjointChecks
) -> Mapping[str, object]:
    endpoint = EndpointSequence(
        sequence.identity,
        sequence.deployable,
        sequence.latent_oracle,
        sequence.contemporaneous_oracle,
        forecast,
    )
    batch = EndpointBatch((endpoint,))
    calibration = calibration_by_sequence(batch)
    return {
        "expected_ce": float(expected_ce_by_sequence(batch)[0]),
        "latent_nmse": float(latent_nmse_by_sequence(batch)[0]),
        "retained_syndrome_nll": float(retained_syndrome_nll_by_sequence(batch, checks)[0]),
        "calibration": {
            "counts": calibration.counts[0].tolist(),
            "predicted_sums": calibration.predicted_sums[0].tolist(),
            "latent_sums": calibration.latent_sums[0].tolist(),
            "absolute_error": float(calibration.absolute_error[0]),
            "bin_edges": calibration.bin_edges.tolist(),
        },
    }


def _bundle_payload(path: str, manifest: BundleManifest) -> dict[str, object]:
    return {
        "path": path,
        "schema_version": manifest.schema_version,
        "metadata_sha256": manifest.metadata_sha256,
        "arrays_sha256": manifest.arrays_sha256,
        "integrity_sha256": manifest.integrity_sha256,
        "array_names": list(manifest.array_names),
    }


def _bundle_manifest(value: Mapping[str, object]) -> BundleManifest:
    if (
        set(value)
        != {
            "path",
            "schema_version",
            "metadata_sha256",
            "arrays_sha256",
            "integrity_sha256",
            "array_names",
        }
        or value["path"] != "estimator-bundle"
    ):
        raise ValueError("development bundle manifest schema is invalid")
    return BundleManifest(
        int(value["schema_version"]),
        str(value["metadata_sha256"]),
        str(value["arrays_sha256"]),
        str(value["integrity_sha256"]),
        tuple(value["array_names"]),
    )


def _history_source_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[tuple[str, str, int], dict[str, dict[str, Any]]]:
    """Plan arm-specific derangements without retaining sequence arrays."""
    result: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    groups = {(str(row["role"]), str(row["regime"])) for row in rows}
    for role, regime in groups:
        selected = [row for row in rows if row["role"] == role and row["regime"] == regime]
        for arm in DEPLOYABLE_HISTORY_ARMS:
            deranged = fixed_history_derangement(selected, seed=seed, arm=arm)
            for row, source_row in zip(selected, deranged, strict=True):
                key = (role, regime, int(row["sequence_index"]))
                result.setdefault(key, {})[arm] = source_row
    return result


def _validate_endpoint(value: Mapping[str, object]) -> dict[str, object]:
    expected = {"expected_ce", "latent_nmse", "retained_syndrome_nll", "calibration"}
    if set(value) != expected:
        raise ValueError("endpoint kernel must return the exact endpoint family")
    for name in expected - {"calibration"}:
        number = value[name]
        if isinstance(number, bool) or not isinstance(
            number, (int, float, np.integer, np.floating)
        ):
            raise TypeError("endpoint values must be numeric")
        if not math.isfinite(float(number)):
            raise ValueError("endpoint values must be finite")
        if float(number) < 0.0:
            raise ValueError("endpoint losses must be non-negative")
    calibration = value["calibration"]
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "counts",
        "predicted_sums",
        "latent_sums",
        "absolute_error",
        "bin_edges",
    }:
        raise ValueError("calibration endpoint evidence is incomplete")
    counts = calibration["counts"]
    predicted = calibration["predicted_sums"]
    latent = calibration["latent_sums"]
    edges = calibration["bin_edges"]
    if (
        not isinstance(counts, list)
        or len(counts) != 10
        or any(type(count) is not int or count < 0 for count in counts)
        or sum(counts) <= 0
    ):
        raise ValueError("calibration counts must contain ten non-negative bins")
    for sums in (predicted, latent):
        if (
            not isinstance(sums, list)
            or len(sums) != 10
            or any(
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0.0
                for number in sums
            )
            or any(float(total) > count for total, count in zip(sums, counts, strict=True))
        ):
            raise ValueError("calibration probability sums are invalid")
    expected_edges = np.linspace(1e-5, 0.25, 11).tolist()
    if edges != expected_edges:
        raise ValueError("calibration bin edges do not match the fixed policy")
    absolute_error = calibration["absolute_error"]
    expected_error = sum(
        abs(float(predicted_sum) - float(latent_sum))
        for predicted_sum, latent_sum in zip(predicted, latent, strict=True)
    ) / sum(counts)
    if (
        isinstance(absolute_error, bool)
        or not isinstance(absolute_error, (int, float))
        or not math.isfinite(float(absolute_error))
        or not math.isclose(float(absolute_error), expected_error, rel_tol=1e-12, abs_tol=1e-15)
    ):
        raise ValueError("calibration absolute error does not match its sufficient statistics")
    return dict(value)


def _evaluate_sequences(
    *,
    sequence_dir: Path,
    rows: list[dict[str, Any]],
    staging: Path,
    config: IdentifiabilityConfig,
    code: Any,
    checks: DisjointChecks,
    bundle: Any,
    dependencies: ScreenDependencies,
    deadline: _CpuDeadline,
) -> list[dict[str, object]]:
    history_sources = _history_source_rows(rows, seed=config.seeds.derangement)
    forecast_kernel = dependencies.forecast_sequence or _default_forecast_sequence
    score_kernel = dependencies.score_sequence or _default_score_sequence
    evidence: list[dict[str, object]] = []
    for row in rows:
        deadline.check()
        sequence = _load_sequence(sequence_dir, row, code)
        key = (sequence.identity.role, sequence.identity.regime, sequence.identity.sequence_index)
        deranged_histories = {
            arm: _load_sequence(sequence_dir, source_row, code).deployable
            for arm, source_row in history_sources[key].items()
        }
        forecasts = forecast_kernel(
            sequence=sequence,
            config=config,
            checks=checks,
            bundle=bundle,
            deranged_histories=deranged_histories,
            process_cpu_deadline=deadline.absolute,
            include_grid_diagnostic=sequence.identity.role == "validation",
        )
        expected_forecasts = _forecast_keys(sequence.identity.role)
        if not isinstance(forecasts, Mapping) or tuple(forecasts) != expected_forecasts:
            raise ValueError("forecast kernel must return every canonical arm in exact order")
        relative = (
            Path("evidence")
            / sequence.identity.role
            / sequence.identity.regime
            / f"sequence-{sequence.identity.sequence_index:05d}.npz"
        )
        arrays: dict[str, np.ndarray] = {}
        arms: dict[str, object] = {}
        for arm, forecast in forecasts.items():
            if type(forecast) is not ForecastResult or forecast.arm != arm:
                raise TypeError("forecast kernel returned an invalid arm result")
            forecast_member = f"{arm}__forecast"
            arrays[forecast_member] = np.asarray(forecast.probabilities)
            state_member: str | None = None
            if forecast.state_estimates is not None:
                state_member = f"{arm}__state"
                arrays[state_member] = np.asarray(forecast.state_estimates)
            endpoints = _validate_endpoint(
                score_kernel(sequence=sequence, forecast=forecast, checks=checks)
            )
            arms[arm] = {
                "forecast": _array_descriptor(arrays[forecast_member], member=forecast_member),
                "state": (
                    None
                    if state_member is None
                    else _array_descriptor(arrays[state_member], member=state_member)
                ),
                "interior_cells": forecast.interior_cells,
                "endpoints": endpoints,
            }
        payload = _stable_npz(arrays)
        _write_atomic(staging / relative, payload)
        evidence.append(
            {
                "regime": sequence.identity.regime,
                "role": sequence.identity.role,
                "sequence_index": sequence.identity.sequence_index,
                "sequence_content_sha256": sequence.identity.content_sha256,
                "seeds": dict(row["seeds"]),
                "path": str(relative),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "arms": arms,
            }
        )
        deadline.check()
    return evidence


def _source_payload(source: ScreenSourceEvidence) -> dict[str, object]:
    return {"commit": source.commit, "tree_sha256": source.tree_sha256, "clean": True}


def _identity_input(root: Mapping[str, object]) -> dict[str, object]:
    return {
        key: root[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "status",
            "mode",
            "claim",
            "source",
            "config",
            "code",
            "retained_checks",
            "sequences",
            "fisher_precheck",
            "development_record",
            "approval",
            "bundle",
            "completed_stages",
            "inference",
            "decision",
            "decoder",
            "runtime",
        )
    }


def _content_input(root: Mapping[str, object]) -> object:
    return [
        {
            "path": row["path"],
            "payload_sha256": row["payload_sha256"],
            "arms": row["arms"],
        }
        for row in root["evidence"]  # type: ignore[index]
    ]


def _stage(stages: dict[str, str], name: str, value: object) -> None:
    stages[name] = _digest(value)


def _abort(
    *,
    output_dir: Path,
    mode: str,
    source: Mapping[str, object] | None,
    config: Mapping[str, object] | None,
    stages: Mapping[str, str],
    limit: float,
    elapsed: float,
    expected_completed: Mapping[str, str] | None = None,
) -> dict[str, object]:
    root: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        "complete": False,
        "status": "aborted_no_verdict",
        "mode": mode,
        "claim": "engineering_measurement_no_speed_claim",
        "source": source,
        "config": config,
        "completed_stages": dict(stages),
        "deadline": {
            "kind": "process_cpu_seconds",
            "limit": limit,
            "elapsed": elapsed,
        },
        "decision": None,
    }
    root["identity_sha256"] = _digest({key: root[key] for key in root})
    if output_dir.exists() and expected_completed is None:
        raise FileExistsError("refusing to overwrite an existing run output")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-aborted-", dir=output_dir.parent))
    displaced: Path | None = None
    abort_confirmed = False

    def require_expected_completed(path: Path) -> None:
        manifest_path = path / "manifest.json"
        published = _load_json(manifest_path, "published run manifest")
        expected_fields = {"identity_sha256", "content_sha256", "manifest_sha256"}
        if (
            expected_completed is None
            or set(expected_completed) != expected_fields
            or published.get("complete") is not True
            or published.get("status") != "complete"
            or any(
                published.get(key) != expected_completed[key]
                for key in ("identity_sha256", "content_sha256")
            )
            or sha256_file(manifest_path) != expected_completed["manifest_sha256"]
        ):
            raise ValueError("deadline replacement target is not the exact just-published run")

    try:
        _write_atomic(staging / "manifest.json", _canonical_json(root))
        if expected_completed is not None:
            require_expected_completed(output_dir)
            displaced = Path(
                tempfile.mkdtemp(prefix=f".{output_dir.name}-expired-", dir=output_dir.parent)
            )
            displaced.rmdir()
            os.replace(output_dir, displaced)
            require_expected_completed(displaced)
        os.replace(staging, output_dir)
        published_abort = _load_json(output_dir / "manifest.json", "deadline abort manifest")
        published_files = {
            str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()
        }
        if published_abort != root or published_files != {"manifest.json"}:
            raise ValueError("deadline abort publication failed validation")
        abort_confirmed = True
        if displaced is not None:
            shutil.rmtree(displaced)
            displaced = None
    except BaseException as publication_error:
        shutil.rmtree(staging, ignore_errors=True)
        if displaced is not None and not abort_confirmed:
            if output_dir.exists():
                try:
                    current = _load_json(output_dir / "manifest.json", "failed abort manifest")
                except TypeError, ValueError:
                    current = None
                if current == root:
                    try:
                        shutil.rmtree(output_dir)
                    except OSError as cleanup_error:
                        raise RuntimeError(
                            "abort publication failed and its invalid output could not be "
                            f"removed; completed artifact retained at {displaced}, "
                            f"output path={output_dir}, cleanup error={cleanup_error!r}"
                        ) from publication_error
                else:
                    raise RuntimeError(
                        "abort publication failed; completed artifact retained at "
                        f"{displaced} because output path {output_dir} is occupied"
                    ) from publication_error
            try:
                os.replace(displaced, output_dir)
                displaced = None
            except OSError as restoration_error:
                raise RuntimeError(
                    "abort publication failed and completed artifact restoration failed; "
                    f"retained path={displaced}, output path={output_dir}, "
                    f"restoration error={restoration_error!r}"
                ) from publication_error
        raise
    return root


def _base_context(
    *,
    config_path: Path,
    dependencies: ScreenDependencies,
    deadline: _CpuDeadline,
    stages: dict[str, str],
) -> tuple[
    ScreenSourceEvidence,
    IdentifiabilityConfig,
    Any,
    sparse.csr_matrix,
    DisjointChecks,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    source = _source(dependencies)
    source_payload = _source_payload(source)
    config = load_identifiability_config(config_path)
    deadline.limit = float(config.runtime.process_cpu_seconds)
    deadline.absolute = deadline.started + deadline.limit
    config_payload = {
        "path_sha256": sha256_file(config_path),
        "identity_sha256": _digest(dataclasses.asdict(config)),
    }
    code = dependencies.code_factory()
    logical_x = dependencies.logical_x_factory(code.hx, code.hz).astype(np.uint8).tocsr()
    code_payload = _code_manifest(
        code,
        logical_x,
        config,
        require_canonical=dependencies.require_canonical_code,
    )
    checks = greedy_disjoint_rows(code.hx)
    if dependencies.require_canonical_code and (
        len(checks.row_indices),
        len(checks.covered_qubits),
    ) != (135, 1350):
        raise ValueError("canonical retained-row count or coverage changed")
    checks_payload = _checks_manifest(checks)
    _stage(
        stages,
        "source_config_code_support",
        [source_payload, config_payload, code_payload, checks_payload],
    )
    deadline.check()
    return (
        source,
        config,
        code,
        logical_x,
        checks,
        source_payload,
        config_payload,
        code_payload,
        checks_payload,
    )


def _sequence_binding(
    *,
    sequence_dir: Path,
    sequence_root: Mapping[str, object],
    roles: tuple[str, ...],
) -> dict[str, object]:
    return {
        "manifest_sha256": sha256_file(sequence_dir / "manifest.json"),
        "identity_sha256": sequence_root.get("identity_sha256"),
        "content_sha256": sequence_root.get("content_sha256"),
        "roles": list(roles),
    }


def run_development(
    *,
    config_path: Path,
    sequence_dir: Path,
    output_dir: Path,
    dependencies: ScreenDependencies | None = None,
) -> dict[str, Any]:
    """Fit, evaluate, and atomically publish development-only evidence."""
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite or resume an existing run output")
    deps = _resolved_dependencies(dependencies)
    cpu_started = float(deps.process_time())
    wall_started = float(deps.wall_time())
    deadline = _CpuDeadline(started=cpu_started, limit=21_600.0, clock=deps.process_time)
    stages: dict[str, str] = {}
    source_payload: dict[str, object] | None = None
    config_payload: dict[str, object] | None = None
    staging: Path | None = None
    published_complete = False
    completed_identity: dict[str, str] | None = None
    try:
        (
            _,
            config,
            code,
            _,
            checks,
            source_payload,
            config_payload,
            code_payload,
            checks_payload,
        ) = _base_context(
            config_path=config_path,
            dependencies=deps,
            deadline=deadline,
            stages=stages,
        )
        sequence_root = _verify_sequences(
            deps,
            config_path=config_path,
            sequence_dir=sequence_dir,
            roles=_DEVELOPMENT_ROLES,
        )
        rows = _require_sequence_coordinates(sequence_root, config, _DEVELOPMENT_ROLES)
        fisher = sequence_root.get("fisher_precheck")
        if not isinstance(fisher, Mapping) or fisher.get("status") != "passed":
            raise ValueError("development requires a passed Fisher precheck before evaluation")
        sequences_payload = _sequence_binding(
            sequence_dir=sequence_dir,
            sequence_root=sequence_root,
            roles=_DEVELOPMENT_ROLES,
        )
        _stage(stages, "development_sequences", sequences_payload)
        deadline.check()

        bundle = deps.fit_bundle(_development_partitions(rows), config)
        deadline.check()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
        )
        bundle_manifest = deps.write_bundle(staging / "estimator-bundle", bundle)
        bundle_payload = _bundle_payload("estimator-bundle", bundle_manifest)
        if getattr(bundle, "integrity_sha256", None) != bundle_manifest.integrity_sha256:
            raise ValueError("written bundle manifest does not bind the fitted bundle")
        _stage(stages, "frozen_bundle", bundle_payload)
        deadline.check()

        evidence = _evaluate_sequences(
            sequence_dir=sequence_dir,
            rows=rows,
            staging=staging,
            config=config,
            code=code,
            checks=checks,
            bundle=bundle,
            dependencies=deps,
            deadline=deadline,
        )
        _stage(stages, "development_evidence", evidence)
        inference = {
            "status": "development_only_not_confirmatory",
            "sampling_unit": "independent_sequence",
            "bootstrap": "paired_centered_rademacher_wild_bootstrap_t",
            "bootstrap_draws": config.inference.bootstrap_draws,
            "holm_family": list(DEPLOYABLE_HISTORY_ARMS),
            "delta_nll": config.inference.delta_nll,
            "test_role_opened": False,
            "grid_diagnostics": _grid_diagnostics(evidence, config),
        }
        _stage(stages, "inference_metadata", inference)
        deadline.check()
        runtime = {
            "process_cpu_seconds": deadline.elapsed(),
            "wall_seconds": max(0.0, float(deps.wall_time()) - wall_started),
            "measurement": "engineering_measurement_no_speed_claim",
            "deadline_process_cpu_seconds": config.runtime.process_cpu_seconds,
            "peak_rss_bytes": int(deps.peak_memory_bytes()),
        }
        deadline.check()
        root: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "complete": True,
            "status": "complete",
            "mode": "development",
            "claim": "engineering_measurement_no_speed_claim",
            "source": source_payload,
            "config": config_payload,
            "code": code_payload,
            "retained_checks": checks_payload,
            "sequences": sequences_payload,
            "fisher_precheck": dict(fisher),
            "development_record": None,
            "approval": None,
            "bundle": bundle_payload,
            "completed_stages": stages,
            "evidence": evidence,
            "inference": inference,
            "decision": None,
            "decoder": "not_run_in_development",
            "runtime": runtime,
        }
        root["identity_sha256"] = _digest(_identity_input(root))
        root["content_sha256"] = _digest(_content_input(root))
        _write_atomic(staging / "manifest.json", _canonical_json(root))
        completed_identity = {
            "identity_sha256": str(root["identity_sha256"]),
            "content_sha256": str(root["content_sha256"]),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        }
        deadline.check()
        os.replace(staging, output_dir)
        staging = None
        published_complete = True
        deadline.check()
        return root
    except ExperimentDeadlineExceeded:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        return _abort(
            output_dir=output_dir,
            mode="development",
            source=source_payload,
            config=config_payload,
            stages=stages,
            limit=deadline.limit,
            elapsed=deadline.elapsed(),
            expected_completed=completed_identity if published_complete else None,
        )
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is missing or unreadable") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _exact_fields(value: object, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _validate_array_descriptor(value: object, array: np.ndarray, member: str) -> None:
    descriptor = _exact_fields(
        value,
        frozenset({"member", "dtype", "shape", "sha256"}),
        "evidence array descriptor",
    )
    if descriptor != _array_descriptor(array, member=member):
        raise ValueError("evidence array descriptor or hash does not match payload")


def _validate_evidence_payloads(output_dir: Path, evidence: object) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        raise TypeError("run evidence must be a list")
    seen_paths: set[str] = set()
    for item in evidence:
        row = _exact_fields(item, _EVIDENCE_FIELDS, "run evidence row")
        path_text = row["path"]
        if not isinstance(path_text, str) or path_text in seen_paths:
            raise ValueError("run evidence paths must be unique strings")
        seen_paths.add(path_text)
        relative = Path(path_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("run evidence path must be a safe relative path")
        path = output_dir / relative
        if not path.is_file() or sha256_file(path) != row["payload_sha256"]:
            raise ValueError("run evidence payload SHA-256 mismatch")
        seeds = row["seeds"]
        if (
            not isinstance(seeds, dict)
            or set(seeds) != {"latent", "bernoulli", "filter"}
            or any(type(seed) is not int or seed < 0 for seed in seeds.values())
        ):
            raise ValueError("run evidence sequence seed streams are invalid")
        arms = row["arms"]
        if not isinstance(arms, dict) or set(arms) != set(_forecast_keys(str(row["role"]))):
            raise ValueError("run evidence has a missing, extra, or reordered arm")
        with np.load(path, allow_pickle=False) as archive:
            declared_members: set[str] = set()
            for arm_name, arm_value in arms.items():
                arm = _exact_fields(
                    arm_value,
                    frozenset({"forecast", "state", "interior_cells", "endpoints"}),
                    "run arm evidence",
                )
                forecast_member = f"{arm_name}__forecast"
                if forecast_member not in archive.files:
                    raise ValueError("evidence forecast array is missing")
                _validate_array_descriptor(
                    arm["forecast"], np.asarray(archive[forecast_member]), forecast_member
                )
                declared_members.add(forecast_member)
                state = arm["state"]
                state_member = f"{arm_name}__state"
                if state is None:
                    if state_member in archive.files:
                        raise ValueError("undeclared evidence state array is present")
                else:
                    if state_member not in archive.files:
                        raise ValueError("evidence state array is missing")
                    _validate_array_descriptor(
                        state, np.asarray(archive[state_member]), state_member
                    )
                    declared_members.add(state_member)
                if not isinstance(arm["endpoints"], Mapping):
                    raise TypeError("run endpoint evidence must be an object")
                _validate_endpoint(arm["endpoints"])
            if set(archive.files) != declared_members:
                raise ValueError("evidence payload contains undeclared arrays")
    return evidence


def _validate_run_semantics(
    value: Mapping[str, object],
    *,
    expected_mode: str,
    config: IdentifiabilityConfig,
    evidence: list[dict[str, Any]],
) -> None:
    roles = _DEVELOPMENT_ROLES if expected_mode == "development" else ("test",)
    sequence_binding = _exact_fields(
        value["sequences"],
        frozenset({"manifest_sha256", "identity_sha256", "content_sha256", "roles"}),
        "run sequence binding",
    )
    if sequence_binding["roles"] != list(roles) or any(
        not _is_hex(sequence_binding[name], 64)
        for name in ("manifest_sha256", "identity_sha256", "content_sha256")
    ):
        raise ValueError("run sequence binding is invalid")
    expected_coordinates = [
        (role, regime, index)
        for role in roles
        for regime in config.regimes
        for index in range(int(getattr(config.splits, role)))
    ]
    actual_coordinates = [(row["role"], row["regime"], row["sequence_index"]) for row in evidence]
    if actual_coordinates != expected_coordinates:
        raise ValueError("run evidence has missing, reordered, or overlapping sequence identities")

    fisher = _exact_fields(value["fisher_precheck"], _FISHER_FIELDS, "run Fisher precheck")
    if fisher["status"] != "passed" or fisher["failure_reasons"] != []:
        raise ValueError("completed run requires a passed Fisher precheck")
    if fisher["provenance"] != {
        "domain": config.seeds.fisher_domain,
        "seed": config.seeds.fisher,
        "law": config.fisher.draw_law,
        "draws": config.fisher.draws,
    }:
        raise ValueError("run Fisher precheck provenance is invalid")
    for name in _FISHER_FIELDS - {"status", "provenance", "failure_reasons"}:
        number = fisher[name]
        if type(number) not in (int, float) or not math.isfinite(float(number)):
            raise ValueError("run Fisher precheck summary is invalid")

    if expected_mode == "development":
        inference = _exact_fields(
            value["inference"],
            frozenset(
                {
                    "status",
                    "sampling_unit",
                    "bootstrap",
                    "bootstrap_draws",
                    "holm_family",
                    "delta_nll",
                    "test_role_opened",
                    "grid_diagnostics",
                }
            ),
            "development inference metadata",
        )
        if (
            inference["status"] != "development_only_not_confirmatory"
            or inference["sampling_unit"] != "independent_sequence"
            or inference["bootstrap"] != "paired_centered_rademacher_wild_bootstrap_t"
            or inference["bootstrap_draws"] != config.inference.bootstrap_draws
            or inference["holm_family"] != list(DEPLOYABLE_HISTORY_ARMS)
            or inference["delta_nll"] != config.inference.delta_nll
            or inference["test_role_opened"] is not False
        ):
            raise ValueError("development inference policy is invalid")
        diagnostics = inference["grid_diagnostics"]
        expected_diagnostics = {f"validation/{regime}" for regime in config.regimes}
        if not isinstance(diagnostics, dict) or set(diagnostics) != expected_diagnostics:
            raise ValueError("development grid diagnostics are incomplete")
        for item in diagnostics.values():
            diagnostic = _exact_fields(
                item,
                frozenset(
                    {
                        "nominal_mean_gain",
                        "doubled_mean_gain",
                        "absolute_difference",
                        "tolerance",
                        "passed",
                    }
                ),
                "development grid diagnostic",
            )
            difference = diagnostic["absolute_difference"]
            if (
                type(difference) not in (int, float)
                or not math.isfinite(float(difference))
                or diagnostic["tolerance"] != config.grid.convergence_tolerance
                or diagnostic["passed"]
                is not (float(difference) < config.grid.convergence_tolerance)
            ):
                raise ValueError("development grid diagnostic is invalid")
        if diagnostics != _grid_diagnostics(evidence, config):
            raise ValueError("development grid diagnostics do not match sequence evidence")
        if value["decision"] is not None or value["decoder"] != "not_run_in_development":
            raise ValueError("development record contains forbidden confirmatory output")
        if value["development_record"] is not None or value["approval"] is not None:
            raise ValueError("development run cannot contain approval bindings")
    else:
        inference = value["inference"]
        if not isinstance(inference, dict) or inference.get("decision") != value["decision"]:
            raise ValueError("confirmation inference and decision disagree")
        metadata = _exact_fields(
            inference.get("metadata"),
            frozenset(
                {"sampling_unit", "bootstrap", "bootstrap_draws", "holm_family", "delta_nll"}
            ),
            "confirmation inference metadata",
        )
        if metadata != {
            "sampling_unit": "independent_sequence",
            "bootstrap": "paired_centered_rademacher_wild_bootstrap_t",
            "bootstrap_draws": config.inference.bootstrap_draws,
            "holm_family": list(DEPLOYABLE_HISTORY_ARMS),
            "delta_nll": config.inference.delta_nll,
        }:
            raise ValueError("confirmation inference policy is invalid")
        if not isinstance(value["development_record"], dict) or not isinstance(
            value["approval"], dict
        ):
            raise ValueError("confirmation run is missing approved development bindings")


def _validate_completed_stages(value: Mapping[str, object], expected_mode: str) -> None:
    stages = value["completed_stages"]
    names = (
        _DEVELOPMENT_STAGE_NAMES if expected_mode == "development" else _CONFIRMATION_STAGE_NAMES
    )
    if (
        not isinstance(stages, dict)
        or set(stages) != names
        or any(not _is_hex(stage_hash, 64) for stage_hash in stages.values())
    ):
        raise ValueError("run completed-stage hashes are missing or invalid")
    expected = {
        "source_config_code_support": _digest(
            [value["source"], value["config"], value["code"], value["retained_checks"]]
        )
    }
    if expected_mode == "development":
        expected.update(
            {
                "development_sequences": _digest(value["sequences"]),
                "frozen_bundle": _digest(value["bundle"]),
                "development_evidence": _digest(value["evidence"]),
                "inference_metadata": _digest(value["inference"]),
            }
        )
    else:
        expected.update(
            {
                "approved_development_bundle": _digest(
                    [value["development_record"], value["approval"], value["bundle"]]
                ),
                "test_sequences": _digest(value["sequences"]),
                "confirmatory_evidence": _digest(value["evidence"]),
                "confirmatory_inference": _digest(value["inference"]),
                "conditional_decoder": _digest(value["decoder"]),
            }
        )
    if stages != expected:
        raise ValueError("run completed-stage hashes do not bind their declared evidence")


def _validate_complete_root(
    root: object,
    *,
    output_dir: Path,
    expected_mode: str,
    source_payload: Mapping[str, object],
    config_payload: Mapping[str, object],
    code_payload: Mapping[str, object],
    checks_payload: Mapping[str, object],
    config: IdentifiabilityConfig,
) -> dict[str, Any]:
    value = _exact_fields(root, _ROOT_FIELDS, "run completion manifest")
    if (
        value["schema_version"] != _SCHEMA_VERSION
        or value["artifact_kind"] != _ARTIFACT_KIND
        or value["complete"] is not True
        or value["status"] != "complete"
        or value["mode"] != expected_mode
        or value["claim"] != "engineering_measurement_no_speed_claim"
    ):
        raise ValueError("run completion manifest identity or state is invalid")
    if value["source"] != dict(source_payload):
        raise ValueError("run source identity does not match current clean source")
    if value["config"] != dict(config_payload):
        raise ValueError("run configuration identity does not match current config")
    if value["code"] != dict(code_payload):
        raise ValueError("run code identity does not match current canonical code")
    if value["retained_checks"] != dict(checks_payload):
        raise ValueError("run retained-row identity does not match current construction")
    if value["identity_sha256"] != _digest(_identity_input(value)):
        raise ValueError("run identity SHA-256 is invalid")
    evidence = _validate_evidence_payloads(output_dir, value["evidence"])
    _validate_run_semantics(value, expected_mode=expected_mode, config=config, evidence=evidence)
    decoder_paths = _validate_decoder_payloads(output_dir, value["decoder"])
    bundle_manifest = _bundle_manifest(value["bundle"])
    declared = {"manifest.json", *(str(row["path"]) for row in evidence), *decoder_paths}
    if expected_mode == "development":
        bundle_root = output_dir / "estimator-bundle"
        if (
            not bundle_root.is_dir()
            or sha256_file(bundle_root / "metadata.json") != bundle_manifest.metadata_sha256
            or sha256_file(bundle_root / "arrays.npz") != bundle_manifest.arrays_sha256
        ):
            raise ValueError("development bundle files do not match the completion manifest")
        declared.update({"estimator-bundle/metadata.json", "estimator-bundle/arrays.npz"})
    actual = {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    if actual != declared:
        raise ValueError("run publication contains missing or undeclared files")
    if value["content_sha256"] != _digest(_content_input(value)):
        raise ValueError("run content SHA-256 is invalid")
    _validate_completed_stages(value, expected_mode)
    runtime = _exact_fields(
        value["runtime"],
        frozenset(
            {
                "process_cpu_seconds",
                "wall_seconds",
                "measurement",
                "deadline_process_cpu_seconds",
                "peak_rss_bytes",
            }
        ),
        "run runtime",
    )
    if (
        runtime["measurement"] != "engineering_measurement_no_speed_claim"
        or runtime["deadline_process_cpu_seconds"] != config.runtime.process_cpu_seconds
        or float(runtime["process_cpu_seconds"]) > float(runtime["deadline_process_cpu_seconds"])
        or any(
            isinstance(runtime[name], bool)
            or not isinstance(runtime[name], (int, float))
            or not math.isfinite(float(runtime[name]))
            or float(runtime[name]) < 0.0
            for name in ("process_cpu_seconds", "wall_seconds")
        )
        or type(runtime["peak_rss_bytes"]) is not int
        or runtime["peak_rss_bytes"] < 0
    ):
        raise ValueError("run runtime evidence is invalid")
    value["evidence"] = evidence
    return value


def _validate_decoder_payloads(output_dir: Path, value: object) -> set[str]:
    if value == "not_run_in_development":
        return set()
    decoder = _exact_fields(
        value,
        frozenset(
            {
                "status",
                "arms",
                "configuration",
                "common_sequence_round_identity_sha256",
                "per_sequence_bler",
                "outcomes",
                "comparisons",
            }
        )
        if isinstance(value, dict) and value.get("status") == "completed"
        else frozenset({"status", "arms", "comparisons"}),
        "conditional decoder evidence",
    )
    if decoder["status"] == "not_run_by_design":
        if decoder != {"status": "not_run_by_design", "arms": [], "comparisons": {}}:
            raise ValueError("non-GO decoder evidence must be exactly not_run_by_design")
        return set()
    if decoder["status"] != "completed":
        raise ValueError("conditional decoder status is invalid")
    arms = decoder["arms"]
    if (
        not isinstance(arms, list)
        or not arms
        or arms[0] != "known_marginal"
        or arms[-1] != "contemporaneous_oracle"
    ):
        raise ValueError("conditional decoder arms are invalid")
    if len(set(arms)) != len(arms) or any(
        arm not in {"known_marginal", *DEPLOYABLE_HISTORY_ARMS, "contemporaneous_oracle"}
        for arm in arms
    ):
        raise ValueError("conditional decoder arms are duplicated or unsupported")
    if not _is_hex(decoder["common_sequence_round_identity_sha256"], 64):
        raise ValueError("decoder common membership identity is invalid")
    per_bler = decoder["per_sequence_bler"]
    if not isinstance(per_bler, dict) or set(per_bler) != set(arms):
        raise ValueError("decoder BLER evidence does not contain the exact arm set")
    sequence_counts: set[int] = set()
    for values in per_bler.values():
        if not isinstance(values, list) or not values:
            raise ValueError("decoder BLER must contain one value per sequence")
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any((numeric < 0.0) | (numeric > 1.0)):
            raise ValueError("decoder BLER values must lie in [0, 1]")
        sequence_counts.add(len(values))
    if len(sequence_counts) != 1:
        raise ValueError("decoder arms do not share the same sequence membership")
    outcomes = decoder["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != len(arms) * next(iter(sequence_counts)):
        raise ValueError("decoder outcomes are incomplete")
    expected_array_names = {
        "corrections",
        "predicted_observables",
        "syndrome_valid",
        "logical_failure",
        "converged",
        "iterations",
    }
    declared_paths: set[str] = set()
    memberships_by_arm: dict[str, list[dict[str, object]]] = {arm: [] for arm in arms}
    for item in outcomes:
        row = _exact_fields(
            item,
            frozenset(
                {
                    "arm",
                    "regime",
                    "role",
                    "sequence_index",
                    "sequence_content_sha256",
                    "membership",
                    "path",
                    "payload_sha256",
                    "arrays",
                }
            ),
            "decoder outcome",
        )
        if row["arm"] not in arms or row["regime"] != "temporal_uniform" or row["role"] != "test":
            raise ValueError("decoder outcome identity is invalid")
        memberships_by_arm[row["arm"]].append(
            {
                "role": row["role"],
                "regime": row["regime"],
                "sequence_index": row["sequence_index"],
                "sequence_content_sha256": row["sequence_content_sha256"],
                "membership": row["membership"],
            }
        )
        path = Path(str(row["path"]))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("decoder outcome path must be safe and relative")
        if str(path) in declared_paths:
            raise ValueError("decoder outcome paths must be unique")
        declared_paths.add(str(path))
        payload_path = output_dir / path
        if not payload_path.is_file() or sha256_file(payload_path) != row["payload_sha256"]:
            raise ValueError("decoder payload SHA-256 mismatch")
        descriptors = row["arrays"]
        if not isinstance(descriptors, dict) or set(descriptors) != expected_array_names:
            raise ValueError("decoder array descriptors are incomplete")
        with np.load(payload_path, allow_pickle=False) as archive:
            if set(archive.files) != expected_array_names:
                raise ValueError("decoder payload array membership is invalid")
            for name in expected_array_names:
                _validate_array_descriptor(descriptors[name], np.asarray(archive[name]), name)
    common_memberships = memberships_by_arm["known_marginal"]
    if any(memberships_by_arm[arm] != common_memberships for arm in arms[1:]) or decoder[
        "common_sequence_round_identity_sha256"
    ] != _digest(common_memberships):
        raise ValueError("decoder arms do not share the same content-bound sequence rounds")
    comparisons = decoder["comparisons"]
    if not isinstance(comparisons, dict) or set(comparisons) != set(arms[1:]):
        raise ValueError("decoder comparisons must cover every non-comparator arm")
    for arm, item in comparisons.items():
        comparison = _exact_fields(
            item,
            frozenset(
                {
                    "definition",
                    "per_sequence_differences",
                    "interval",
                    "classification",
                }
            ),
            "decoder BLER comparison",
        )
        if comparison["definition"] != f"BLER_{arm} - BLER_known_marginal" or comparison[
            "classification"
        ] not in {"BENEFIT", "HARM", "EQUIVALENT", "INCONCLUSIVE"}:
            raise ValueError("decoder BLER comparison policy is invalid")
    return declared_paths


_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "decision",
        "reviewer",
        "reviewed_at",
        "source_commit",
        "source_tree_sha256",
        "config_sha256",
        "code_sha256",
        "retained_checks_sha256",
        "development_record_sha256",
        "development_identity_sha256",
        "bundle_integrity_sha256",
        "bundle_metadata_sha256",
        "bundle_arrays_sha256",
    }
)


def _approval_binding(
    approval_path: Path,
    development_record: Path,
    development: Mapping[str, object],
) -> dict[str, object]:
    approval = _exact_fields(
        _load_json(approval_path, "manual approval"),
        _APPROVAL_FIELDS,
        "manual approval",
    )
    bundle = development["bundle"]
    assert isinstance(bundle, Mapping)
    expected = {
        "schema_version": 1,
        "kind": "temporal_identifiability_manual_approval",
        "decision": "APPROVE",
        "reviewer": approval["reviewer"],
        "reviewed_at": approval["reviewed_at"],
        "source_commit": development["source"]["commit"],
        "source_tree_sha256": development["source"]["tree_sha256"],
        "config_sha256": development["config"]["path_sha256"],
        "code_sha256": _digest(development["code"]),
        "retained_checks_sha256": development["retained_checks"]["content_sha256"],
        "development_record_sha256": sha256_file(development_record / "manifest.json"),
        "development_identity_sha256": development["identity_sha256"],
        "bundle_integrity_sha256": bundle["integrity_sha256"],
        "bundle_metadata_sha256": bundle["metadata_sha256"],
        "bundle_arrays_sha256": bundle["arrays_sha256"],
    }
    if (
        not isinstance(approval["reviewer"], str)
        or not approval["reviewer"].strip()
        or not isinstance(approval["reviewed_at"], str)
        or not approval["reviewed_at"].strip()
        or approval != expected
    ):
        raise ValueError("manual approval does not bind the exact development and bundle evidence")
    return {**approval, "approval_sha256": sha256_file(approval_path)}


def _partitions_from_run(development: Mapping[str, object]) -> DevelopmentPartitions:
    evidence = development["evidence"]
    assert isinstance(evidence, list)
    rows = [
        {
            "regime": row["regime"],
            "role": row["role"],
            "sequence_index": row["sequence_index"],
            "seeds": row["seeds"],
            "sequence_content_sha256": row["sequence_content_sha256"],
        }
        for row in evidence
    ]
    # The exact seeds live in the fitted bundle partition metadata. For the
    # independent refit, read_verified_bundle regenerates from those identities;
    # this fallback is used only by injected test fitters.
    return _development_partitions(rows)


def _validated_development(
    *,
    development_record: Path,
    approval_path: Path,
    source_payload: Mapping[str, object],
    config_payload: Mapping[str, object],
    code_payload: Mapping[str, object],
    checks_payload: Mapping[str, object],
    config: IdentifiabilityConfig,
    dependencies: ScreenDependencies,
    deadline: _CpuDeadline,
) -> tuple[dict[str, Any], dict[str, object], Any]:
    root = _validate_complete_root(
        _load_json(development_record / "manifest.json", "development record"),
        output_dir=development_record,
        expected_mode="development",
        source_payload=source_payload,
        config_payload=config_payload,
        code_payload=code_payload,
        checks_payload=checks_payload,
        config=config,
    )
    diagnostics = root["inference"]["grid_diagnostics"]
    if any(diagnostics[f"validation/{regime}"]["passed"] is not True for regime in config.regimes):
        raise ValueError("approved development validation grid convergence failed")
    approval = _approval_binding(approval_path, development_record, root)
    bundle_value = root["bundle"]
    if not isinstance(bundle_value, Mapping):
        raise TypeError("development bundle binding is invalid")
    manifest = _bundle_manifest(bundle_value)
    bundle_path = development_record / "estimator-bundle"
    if (
        sha256_file(bundle_path / "metadata.json") != manifest.metadata_sha256
        or sha256_file(bundle_path / "arrays.npz") != manifest.arrays_sha256
    ):
        raise ValueError("development bundle files do not match approved hashes")
    # Default bundle loading replays the fit internally. The explicit fit call
    # additionally makes this firewall observable to injected integration tests.
    expected_partitions = _partitions_from_run(root)
    if dependencies.read_bundle is read_verified_bundle:
        bundle = dependencies.read_bundle(bundle_path, manifest)
    else:
        fresh = dependencies.fit_bundle(expected_partitions, config)
        if getattr(fresh, "integrity_sha256", None) != manifest.integrity_sha256:
            raise ValueError("independent development refit disagrees with approved bundle")
        deadline.check()
        bundle = dependencies.read_bundle(bundle_path, manifest)
    if getattr(bundle, "integrity_sha256", None) != manifest.integrity_sha256:
        raise ValueError("verified development bundle integrity disagrees with approval")
    if (
        _digest(getattr(bundle, "config_payload", None)) != _digest(dataclasses.asdict(config))
        or getattr(bundle, "partitions", None) != expected_partitions
    ):
        raise ValueError("verified bundle config or development partitions do not match approval")
    deadline.check()
    return root, approval, bundle


def _inference_from_evidence(
    evidence: list[dict[str, object]],
    config: IdentifiabilityConfig,
    dependencies: ScreenDependencies,
    *,
    validation_grid_difference: float,
) -> dict[str, object]:
    by_regime = {
        regime: [row for row in evidence if row["regime"] == regime] for regime in config.regimes
    }

    def endpoint(row: Mapping[str, object], arm: str) -> float:
        arms = row["arms"]
        assert isinstance(arms, Mapping)
        value = arms[arm]
        assert isinstance(value, Mapping)
        endpoints = value["endpoints"]
        assert isinstance(endpoints, Mapping)
        return float(endpoints["expected_ce"])

    def gains(regime: str, arm: str) -> np.ndarray:
        return np.asarray(
            [endpoint(row, "known_marginal") - endpoint(row, arm) for row in by_regime[regime]],
            dtype=np.float64,
        )

    stationary_current = gains("stationary_iid", "contemporaneous_oracle")
    inference = dependencies.inference(
        latent_gain=gains("temporal_uniform", "latent_history_oracle"),
        deployable_gains={arm: gains("temporal_uniform", arm) for arm in DEPLOYABLE_HISTORY_ARMS},
        stationary_gains={arm: gains("stationary_iid", arm) for arm in DEPLOYABLE_HISTORY_ARMS},
        deranged_gains={
            arm: gains("temporal_uniform", f"{arm}{_DERANGED_SUFFIX}")
            for arm in DEPLOYABLE_HISTORY_ARMS
        },
        doubled_grid_gain=None,
        bootstrap_seed=config.seeds.bootstrap,
        leakage_passed=bool(np.max(np.abs(stationary_current)) <= config.inference.delta_nll),
        grid_convergence_difference=validation_grid_difference,
    )
    if not isinstance(inference, dict):
        raise TypeError("confirmatory inference kernel must return a dictionary")
    return {
        **inference,
        "metadata": {
            "sampling_unit": "independent_sequence",
            "bootstrap": "paired_centered_rademacher_wild_bootstrap_t",
            "bootstrap_draws": config.inference.bootstrap_draws,
            "holm_family": list(DEPLOYABLE_HISTORY_ARMS),
            "delta_nll": config.inference.delta_nll,
        },
    }


def _grid_diagnostics(
    evidence: list[dict[str, object]], config: IdentifiabilityConfig
) -> dict[str, object]:
    result: dict[str, object] = {}
    role = "validation"
    for regime in config.regimes:
        rows = [row for row in evidence if row["role"] == role and row["regime"] == regime]

        def ce(row: Mapping[str, object], arm: str) -> float:
            arms = row["arms"]
            assert isinstance(arms, Mapping)
            arm_value = arms[arm]
            assert isinstance(arm_value, Mapping)
            endpoints = arm_value["endpoints"]
            assert isinstance(endpoints, Mapping)
            return float(endpoints["expected_ce"])

        nominal = np.asarray([ce(row, "known_marginal") - ce(row, "grid_bayes") for row in rows])
        doubled = np.asarray(
            [ce(row, "known_marginal") - ce(row, _GRID_DIAGNOSTIC) for row in rows]
        )
        difference = abs(float(nominal.mean() - doubled.mean()))
        result[f"{role}/{regime}"] = {
            "nominal_mean_gain": float(nominal.mean()),
            "doubled_mean_gain": float(doubled.mean()),
            "absolute_difference": difference,
            "tolerance": config.grid.convergence_tolerance,
            "passed": difference < config.grid.convergence_tolerance,
        }
    return result


def _decoder_config(config: IdentifiabilityConfig) -> BPLSDConfig:
    return BPLSDConfig(
        max_iter=config.decoder.max_iter,
        bp_method=config.decoder.bp_method,
        schedule=config.decoder.schedule,
        ms_scaling_factor=config.decoder.ms_scaling_factor,
        lsd_method=config.decoder.lsd_method,
        lsd_order=config.decoder.lsd_order,
    )


def _run_decoder_diagnostic(
    *,
    sequence_dir: Path,
    rows: list[dict[str, Any]],
    staging: Path,
    evidence: list[dict[str, object]],
    selected_arms: tuple[str, ...],
    code: Any,
    logical_x: sparse.csr_matrix,
    config: IdentifiabilityConfig,
    dependencies: ScreenDependencies,
    deadline: _CpuDeadline,
) -> dict[str, object]:
    expected = conditional_decoder_arms(
        {"outcome": "GO-TEMPORAL-IDENTIFIED", "winning_arms": selected_arms[1:-1]}
    )
    if expected != selected_arms:
        raise ValueError("decoder arm selection is not the exact canonical GO policy")
    temporal_rows = [row for row in rows if row["regime"] == "temporal_uniform"]
    evidence_by_identity = {
        (str(row["role"]), str(row["regime"]), int(row["sequence_index"])): row for row in evidence
    }
    outcomes: list[dict[str, object]] = []
    per_sequence_bler = {arm: [] for arm in selected_arms}
    common_memberships: list[dict[str, object]] = []
    logical_csr = logical_x.astype(np.uint8).tocsr()
    hx_csr = code.hx.astype(np.uint8).tocsr()
    decoder_config = _decoder_config(config)
    for row in temporal_rows:
        deadline.check()
        sequence = _load_sequence(sequence_dir, row, code)
        mask = sequence.deployable.scored_mask
        membership = [int(index) for index in np.flatnonzero(mask)]
        common_memberships.append(
            {
                "role": sequence.identity.role,
                "regime": sequence.identity.regime,
                "sequence_index": sequence.identity.sequence_index,
                "sequence_content_sha256": sequence.identity.content_sha256,
                "membership": membership,
            }
        )
        selected_syndromes = sequence.deployable.syndromes[mask]
        selected_errors = sequence.targets.errors[mask]
        recomputed_labels = np.asarray(sequence.targets.errors @ logical_csr.T, dtype=np.uint8) % 2
        if not np.array_equal(recomputed_labels, sequence.targets.logical_flips):
            raise ValueError("stored logical labels do not match canonical logical operators")
        evidence_row = evidence_by_identity[
            (
                sequence.identity.role,
                sequence.identity.regime,
                sequence.identity.sequence_index,
            )
        ]
        evidence_path = staging / str(evidence_row["path"])
        with np.load(evidence_path, allow_pickle=False) as forecast_payload:
            for arm in selected_arms:
                deadline.check()
                member = f"{arm}__forecast"
                probabilities = np.asarray(forecast_payload[member], dtype=np.float64)
                if probabilities.ndim == 1:
                    channels = np.broadcast_to(
                        probabilities[:, None], sequence.targets.errors.shape
                    )
                elif probabilities.shape == sequence.targets.errors.shape:
                    channels = probabilities
                else:
                    raise ValueError("decoder forecast does not match physical field geometry")
                decoded = dependencies.decoder(
                    hx_csr,
                    selected_syndromes,
                    logical_csr,
                    error_channels=np.asarray(channels[mask], dtype=np.float64),
                    config=decoder_config,
                )
                if type(decoded) is not DecodeBatchResult:
                    raise TypeError("decoder kernel returned an invalid result")
                corrections = np.asarray(decoded.corrections, dtype=np.uint8)
                if corrections.shape != selected_errors.shape or not np.all(
                    (corrections == 0) | (corrections == 1)
                ):
                    raise ValueError("decoder corrections have invalid binary geometry")
                recomputed_valid = np.all(
                    np.asarray(hx_csr @ corrections.T, dtype=np.uint8).T % 2 == selected_syndromes,
                    axis=1,
                )
                if not np.array_equal(recomputed_valid, decoded.syndrome_valid):
                    raise RuntimeError(
                        "decoder syndrome validity disagrees with independent reconstruction"
                    )
                predicted = np.asarray(logical_csr @ corrections.T, dtype=np.uint8).T % 2
                if not np.array_equal(predicted, decoded.predicted_observables):
                    raise RuntimeError(
                        "decoder logical prediction disagrees with canonical operators"
                    )
                residual = np.bitwise_xor(selected_errors, corrections)
                residual_logical = np.asarray(logical_csr @ residual.T, dtype=np.uint8).T % 2
                logical_failure = (~recomputed_valid) | np.any(residual_logical != 0, axis=1)
                per_sequence_bler[arm].append(float(np.mean(logical_failure)))
                relative = (
                    Path("decoder") / arm / f"sequence-{sequence.identity.sequence_index:05d}.npz"
                )
                arrays = {
                    "corrections": corrections,
                    "predicted_observables": predicted,
                    "syndrome_valid": recomputed_valid.astype(np.bool_),
                    "logical_failure": logical_failure.astype(np.bool_),
                    "converged": np.asarray(decoded.converged, dtype=np.bool_),
                    "iterations": np.asarray(decoded.iterations, dtype=np.int64),
                }
                payload = _stable_npz(arrays)
                _write_atomic(staging / relative, payload)
                outcomes.append(
                    {
                        "arm": arm,
                        "regime": sequence.identity.regime,
                        "role": sequence.identity.role,
                        "sequence_index": sequence.identity.sequence_index,
                        "sequence_content_sha256": sequence.identity.content_sha256,
                        "membership": membership,
                        "path": str(relative),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "arrays": {
                            name: _array_descriptor(value, member=name)
                            for name, value in arrays.items()
                        },
                    }
                )
                deadline.check()
    comparator = np.asarray(per_sequence_bler["known_marginal"], dtype=np.float64)
    comparisons: dict[str, object] = {}
    for arm in selected_arms[1:]:
        differences = np.asarray(per_sequence_bler[arm], dtype=np.float64) - comparator
        interval = studentized_sequence_interval(
            differences,
            seed=config.seeds.bootstrap,
            null_mean=0.0,
        )
        classification = (
            classify_bler_interval(
                float(interval["two_sided_95"][0]),
                float(interval["two_sided_95"][1]),
            )
            if interval["status"] == "ok"
            else "INCONCLUSIVE"
        )
        comparisons[arm] = {
            "definition": f"BLER_{arm} - BLER_known_marginal",
            "per_sequence_differences": differences.tolist(),
            "interval": interval,
            "classification": classification,
        }
    return {
        "status": "completed",
        "arms": list(selected_arms),
        "configuration": dataclasses.asdict(decoder_config),
        "common_sequence_round_identity_sha256": _digest(common_memberships),
        "per_sequence_bler": per_sequence_bler,
        "outcomes": outcomes,
        "comparisons": comparisons,
    }


def run_confirmation(
    *,
    config_path: Path,
    sequence_dir: Path,
    output_dir: Path,
    development_record: Path,
    approval_path: Path,
    dependencies: ScreenDependencies | None = None,
) -> dict[str, Any]:
    """Execute the held-out gate after the complete pre-read firewall."""
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite or resume an existing run output")
    deps = _resolved_dependencies(dependencies)
    cpu_started = float(deps.process_time())
    wall_started = float(deps.wall_time())
    deadline = _CpuDeadline(started=cpu_started, limit=21_600.0, clock=deps.process_time)
    stages: dict[str, str] = {}
    source_payload: dict[str, object] | None = None
    config_payload: dict[str, object] | None = None
    staging: Path | None = None
    published_complete = False
    completed_identity: dict[str, str] | None = None
    try:
        (
            _,
            config,
            code,
            logical_x,
            checks,
            source_payload,
            config_payload,
            code_payload,
            checks_payload,
        ) = _base_context(
            config_path=config_path,
            dependencies=deps,
            deadline=deadline,
            stages=stages,
        )
        development, approval, bundle = _validated_development(
            development_record=development_record,
            approval_path=approval_path,
            source_payload=source_payload,
            config_payload=config_payload,
            code_payload=code_payload,
            checks_payload=checks_payload,
            config=config,
            dependencies=deps,
            deadline=deadline,
        )
        development_binding = {
            "manifest_sha256": sha256_file(development_record / "manifest.json"),
            "identity_sha256": development["identity_sha256"],
            "content_sha256": development["content_sha256"],
        }
        _stage(
            stages,
            "approved_development_bundle",
            [development_binding, approval, development["bundle"]],
        )
        deadline.check()

        # This is the first operation permitted to open test-role content.
        sequence_root = _verify_sequences(
            deps,
            config_path=config_path,
            sequence_dir=sequence_dir,
            roles=("test",),
            development_record=development_record,
            approval_path=approval_path,
        )
        rows = _require_sequence_coordinates(sequence_root, config, ("test",))
        fisher = sequence_root.get("fisher_precheck")
        if not isinstance(fisher, Mapping) or fisher.get("status") != "passed":
            raise ValueError("confirmation requires a passed Fisher precheck")
        sequence_source = sequence_root.get("source")
        if (
            not isinstance(sequence_source, Mapping)
            or sequence_source.get("commit") != source_payload["commit"]
        ):
            raise ValueError("test sequence source does not match the approved source commit")
        sequence_config = sequence_root.get("config")
        if (
            not isinstance(sequence_config, Mapping)
            or sequence_config.get("sha256") != config_payload["path_sha256"]
        ):
            raise ValueError("test sequence config does not match the approved config")
        if deps.require_canonical_code:
            if sequence_root.get("code") != code_payload:
                raise ValueError(
                    "test sequence code identity does not match current canonical code"
                )
            if sequence_root.get("retained_checks") != checks_payload:
                raise ValueError(
                    "test sequence retained rows do not match current canonical support"
                )
        sequences_payload = _sequence_binding(
            sequence_dir=sequence_dir,
            sequence_root=sequence_root,
            roles=("test",),
        )
        _stage(stages, "test_sequences", sequences_payload)
        deadline.check()

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
        )
        evidence = _evaluate_sequences(
            sequence_dir=sequence_dir,
            rows=rows,
            staging=staging,
            config=config,
            code=code,
            checks=checks,
            bundle=bundle,
            dependencies=deps,
            deadline=deadline,
        )
        _stage(stages, "confirmatory_evidence", evidence)
        inference = _inference_from_evidence(
            evidence,
            config,
            deps,
            validation_grid_difference=float(
                development["inference"]["grid_diagnostics"]["validation/temporal_uniform"][
                    "absolute_difference"
                ]
            ),
        )
        if not isinstance(inference, dict) or not isinstance(inference.get("decision"), Mapping):
            raise TypeError("confirmatory inference did not return a complete gate decision")
        decision = dict(inference["decision"])
        _stage(stages, "confirmatory_inference", inference)
        deadline.check()
        selected = conditional_decoder_arms(decision)
        if selected == "not_run_by_design":
            decoder: dict[str, object] = {
                "status": "not_run_by_design",
                "arms": [],
                "comparisons": {},
            }
        else:
            decoder = _run_decoder_diagnostic(
                sequence_dir=sequence_dir,
                rows=rows,
                staging=staging,
                evidence=evidence,
                selected_arms=selected,
                code=code,
                logical_x=logical_x,
                config=config,
                dependencies=deps,
                deadline=deadline,
            )
        _stage(stages, "conditional_decoder", decoder)
        deadline.check()
        runtime = {
            "process_cpu_seconds": deadline.elapsed(),
            "wall_seconds": max(0.0, float(deps.wall_time()) - wall_started),
            "measurement": "engineering_measurement_no_speed_claim",
            "deadline_process_cpu_seconds": config.runtime.process_cpu_seconds,
            "peak_rss_bytes": int(deps.peak_memory_bytes()),
        }
        deadline.check()
        root: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "complete": True,
            "status": "complete",
            "mode": "confirmation",
            "claim": "engineering_measurement_no_speed_claim",
            "source": source_payload,
            "config": config_payload,
            "code": code_payload,
            "retained_checks": checks_payload,
            "sequences": sequences_payload,
            "fisher_precheck": dict(fisher),
            "development_record": development_binding,
            "approval": approval,
            "bundle": development["bundle"],
            "completed_stages": stages,
            "evidence": evidence,
            "inference": inference,
            "decision": decision,
            "decoder": decoder,
            "runtime": runtime,
        }
        root["identity_sha256"] = _digest(_identity_input(root))
        root["content_sha256"] = _digest(_content_input(root))
        _write_atomic(staging / "manifest.json", _canonical_json(root))
        completed_identity = {
            "identity_sha256": str(root["identity_sha256"]),
            "content_sha256": str(root["content_sha256"]),
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        }
        deadline.check()
        os.replace(staging, output_dir)
        staging = None
        published_complete = True
        deadline.check()
        return root
    except ExperimentDeadlineExceeded:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        return _abort(
            output_dir=output_dir,
            mode="confirmation",
            source=source_payload,
            config=config_payload,
            stages=stages,
            limit=deadline.limit,
            elapsed=deadline.elapsed(),
            expected_completed=completed_identity if published_complete else None,
        )
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_identifiability_run(
    *,
    config_path: Path,
    sequence_dir: Path,
    output_dir: Path,
    development_record: Path | None = None,
    approval_path: Path | None = None,
    dependencies: ScreenDependencies | None = None,
) -> dict[str, Any]:
    """Independently regenerate fits, forecasts, endpoints, inference, and QEC."""
    if (development_record is None) != (approval_path is None):
        raise ValueError("verify requires both development record and approval, or neither")
    deps = _resolved_dependencies(dependencies)
    mode = "confirmation" if development_record is not None else "development"
    with tempfile.TemporaryDirectory(prefix="temporal-identifiability-replay-") as temporary:
        replay_output = Path(temporary) / "run"
        if mode == "development":
            replayed = run_development(
                config_path=config_path,
                sequence_dir=sequence_dir,
                output_dir=replay_output,
                dependencies=deps,
            )
        else:
            assert development_record is not None and approval_path is not None
            replayed = run_confirmation(
                config_path=config_path,
                sequence_dir=sequence_dir,
                output_dir=replay_output,
                development_record=development_record,
                approval_path=approval_path,
                dependencies=deps,
            )
        if replayed.get("status") != "complete":
            raise ExperimentDeadlineExceeded(
                "independent verification replay exceeded the process CPU deadline"
            )

        # The persisted result is deliberately not opened until regeneration,
        # refitting, filtering, endpoint scoring, inference, and QEC replay end.
        persisted = _validate_complete_root(
            _load_json(output_dir / "manifest.json", "persisted run"),
            output_dir=output_dir,
            expected_mode=mode,
            source_payload=replayed["source"],
            config_payload=replayed["config"],
            code_payload=replayed["code"],
            checks_payload=replayed["retained_checks"],
            config=load_identifiability_config(config_path),
        )
        bundle_value = persisted["bundle"]
        if not isinstance(bundle_value, Mapping):
            raise TypeError("persisted run has no strict bundle binding")
        manifest = _bundle_manifest(bundle_value)
        bundle_root = (
            output_dir / "estimator-bundle"
            if mode == "development"
            else development_record / "estimator-bundle"  # type: ignore[operator]
        )
        if (
            sha256_file(bundle_root / "metadata.json") != manifest.metadata_sha256
            or sha256_file(bundle_root / "arrays.npz") != manifest.arrays_sha256
        ):
            raise ValueError("persisted bundle files do not match run hashes")
        loaded = deps.read_bundle(bundle_root, manifest)
        if getattr(loaded, "integrity_sha256", None) != manifest.integrity_sha256:
            raise ValueError("persisted bundle disagrees with independent fit replay")

        comparable_fields = (
            "schema_version",
            "artifact_kind",
            "complete",
            "status",
            "mode",
            "claim",
            "source",
            "config",
            "code",
            "retained_checks",
            "sequences",
            "fisher_precheck",
            "development_record",
            "approval",
            "evidence",
            "inference",
            "decision",
            "decoder",
            "content_sha256",
        )
        for field in comparable_fields:
            if _digest(persisted[field]) != _digest(replayed[field]):
                raise ValueError(f"persisted {field} disagrees with independent replay")
        replay_bundle = replayed["bundle"]
        assert isinstance(replay_bundle, Mapping)
        if (
            persisted["bundle"]["integrity_sha256"] != replay_bundle["integrity_sha256"]
            or persisted["bundle"]["array_names"] != replay_bundle["array_names"]
        ):
            raise ValueError("persisted bundle state or array layout disagrees with refit")
        for stage in set(persisted["completed_stages"]) - {"frozen_bundle"}:
            if persisted["completed_stages"].get(stage) != replayed["completed_stages"].get(stage):
                raise ValueError(f"persisted completed stage {stage} disagrees with replay")
        return persisted


__all__ = [
    "ScreenDependencies",
    "ScreenSourceEvidence",
    "run_confirmation",
    "run_development",
    "verify_identifiability_run",
]
