"""Auditable recovery and tolerance-policy primitives for planar-code shots."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import qecsim
import scipy
from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import (
    PlanarCMWPMDecoder,
    PlanarCode,
    PlanarMPSDecoder,
    PlanarMWPMDecoder,
)
from scipy.stats import norm

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    PlanarCosetMasses,
    planar_mps_coset_masses,
)
from qldpc_fno.metrics.paired import paired_decoder_summary

_CMWPM_GRID = {
    "schema_version": 1,
    "factors": [1, 2, 3, 4],
    "max_iterations": [2, 4, 8],
    "box_shapes": ["t", "r"],
    "distance_algorithms": [2, 4],
}
_PLANAR_SHOT_CONFIG_KEYS = {
    "schema_version",
    "seed_domain",
    "campaign_seed",
    "code_distance",
    "error_rates",
    "shots_per_rate",
    "noise_model",
}
_PLANAR_ERROR_RATES = (0.1, 0.15)
_PLANAR_NOISE_MODEL = "qecsim_iid_depolarizing_code_capacity"
_PLANAR_SHOT_SOURCE_LABELS = (
    "src/qldpc_fno/decision/planar_shot_accuracy.py",
    "src/qldpc_fno/decision/planar_shot_data.py",
)
_CMWPM_SOURCE_LABELS = (
    "experiments/32_calibrate_planar_cmwpm.py",
    *_PLANAR_SHOT_SOURCE_LABELS,
)


def _binary_vector(value: np.ndarray, *, length: int, name: str) -> np.ndarray:
    """Return a binary vector after validating its exact dimensionality."""
    array = np.asarray(value)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, not {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} entries must be binary")
    return np.asarray(array, dtype=np.uint8)


def logical_class_recovery(
    code: PlanarCode, syndrome: np.ndarray, logical_class: int
) -> np.ndarray:
    """Return qecsim's I, X, Y, or Z representative for ``syndrome``."""
    syndrome_array = _binary_vector(syndrome, length=code.stabilizers.shape[0], name="syndrome")
    if type(logical_class) is not int or not 0 <= logical_class < 4:
        raise ValueError("logical_class must be an integer from 0 through 3")
    sample = PlanarMPSDecoder.sample_recovery(code, syndrome_array)
    paulis = (
        sample,
        sample.copy().logical_x(),
        sample.copy().logical_x().logical_z(),
        sample.copy().logical_z(),
    )
    return np.asarray(paulis[logical_class].to_bsf(), dtype=np.uint8)


def score_recovery(
    code: PlanarCode, error: np.ndarray, syndrome: np.ndarray, recovery: np.ndarray
) -> dict[str, object]:
    """Score a recovery from syndrome and residual logical commutation."""
    qubits = code.n_k_d[0]
    error_array = _binary_vector(error, length=2 * qubits, name="error")
    syndrome_array = _binary_vector(syndrome, length=code.stabilizers.shape[0], name="syndrome")
    recovery_array = _binary_vector(recovery, length=2 * qubits, name="recovery")
    recovery_syndrome = np.asarray(pt.bsp(recovery_array, code.stabilizers.T), dtype=np.uint8)
    residual = (error_array + recovery_array) % 2
    residual_syndrome = np.asarray(pt.bsp(residual, code.stabilizers.T), dtype=np.uint8)
    logical_signature = np.asarray(pt.bsp(residual, code.logicals.T), dtype=np.uint8)
    return {
        "syndrome_valid": bool(np.array_equal(recovery_syndrome, syndrome_array)),
        "recovery_syndrome": recovery_syndrome,
        "residual": residual,
        "residual_syndrome": residual_syndrome,
        "logical_signature": logical_signature,
        "logical_failure": bool(np.any(logical_signature)),
    }


def _estimated_flops(result: PlanarCosetMasses) -> int:
    if result.work is None:
        return 0
    return int(result.work.get("estimated_arithmetic_flops", 0))


def two_view_tolerance_decision(syndrome: np.ndarray, error_rate: float) -> dict[str, object]:
    """Choose an agreeing tolerance class, or a fixed-chi column fallback."""
    syndrome_array = _binary_vector(syndrome, length=40, name="syndrome")
    error_rate = float(error_rate)
    if not np.isfinite(error_rate) or not 0 < error_rate < 1:
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    views: dict[str, PlanarCosetMasses | None] = {}
    invalid_actions: dict[str, object] = {}
    invalid_work = 0
    for mode in ("columns", "rows"):
        try:
            views[mode] = planar_mps_coset_masses(
                rows=5,
                columns=5,
                syndrome=syndrome_array,
                error_rate=error_rate,
                chi=None,
                tol=0.01,
                mode=mode,
                trace_work=True,
            )
        except InvalidCosetMassError as error:
            views[mode] = None
            invalid_actions[mode] = _stable_work(error.work)
            invalid_work += int((error.work or {}).get("estimated_arithmetic_flops", 0))
    columns, rows = views["columns"], views["rows"]
    accepted = (
        columns is not None and rows is not None and columns.selected_class == rows.selected_class
    )
    fallback: PlanarCosetMasses | None = None
    selected = columns
    if not accepted:
        try:
            fallback = planar_mps_coset_masses(
                rows=5,
                columns=5,
                syndrome=syndrome_array,
                error_rate=error_rate,
                chi=8,
                tol=None,
                mode="columns",
                trace_work=True,
            )
        except InvalidCosetMassError as error:
            invalid_actions["fallback_columns"] = _stable_work(error.work)
            invalid_work += int((error.work or {}).get("estimated_arithmetic_flops", 0))
        selected = fallback
    actions = [action for action in (columns, rows, fallback) if action is not None]
    return {
        "selected_class": selected.selected_class if selected is not None else None,
        "accepted": accepted,
        "used_fallback": not accepted,
        "tolerance_columns": columns,
        "tolerance_rows": rows,
        "fallback_columns": fallback,
        "estimated_arithmetic_flops": invalid_work
        + sum(_estimated_flops(action) for action in actions),
        "invalid_views": [mode for mode in ("columns", "rows") if mode in invalid_actions],
        "invalid_action_work": invalid_actions,
    }


def cmwpm_grid(grid_path: Path) -> list[dict[str, object]]:
    """Expand the immutable CMWPM calibration grid in its declared order."""
    grid_path = Path(grid_path)
    grid = json.loads(grid_path.read_text())
    if (
        not isinstance(grid, dict)
        or set(grid) != set(_CMWPM_GRID)
        or any(not isinstance(grid[name], list) for name in _CMWPM_GRID if name != "schema_version")
    ):
        raise ValueError("CMWPM grid must contain exactly the frozen schema fields")
    if (
        type(grid["schema_version"]) is not int
        or any(type(value) is not int for value in grid["factors"])
        or any(type(value) is not int for value in grid["max_iterations"])
        or any(type(value) is not str for value in grid["box_shapes"])
        or any(type(value) is not int for value in grid["distance_algorithms"])
        or grid != _CMWPM_GRID
    ):
        raise ValueError("CMWPM grid differs from the frozen 48-candidate policy")
    return [
        {
            "factor": factor,
            "max_iterations": max_iterations,
            "box_shape": box_shape,
            "distance_algorithm": distance_algorithm,
        }
        for factor in grid["factors"]
        for max_iterations in grid["max_iterations"]
        for box_shape in grid["box_shapes"]
        for distance_algorithm in grid["distance_algorithms"]
    ]


def _calibration_shot_seed(domain: str, error_rate: float, shot_index: int) -> int:
    identity = f"{domain}|d5|p{error_rate:.6f}|i{shot_index:06d}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_shot_provenance(payload: Mapping[str, object]) -> None:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("planar calibration artifact is missing provenance")
    source_sha256 = provenance.get("source_sha256")
    if (
        not isinstance(source_sha256, Mapping)
        or set(source_sha256) != set(_PLANAR_SHOT_SOURCE_LABELS)
        or any(not _is_sha256(source_sha256.get(label)) for label in _PLANAR_SHOT_SOURCE_LABELS)
        or not isinstance(provenance.get("qecsim_version"), str)
        or not _is_sha256(provenance.get("config_sha256"))
        or not isinstance(provenance.get("git_commit"), str)
        or type(provenance.get("git_dirty")) is not bool
    ):
        raise ValueError("planar calibration artifact has malformed provenance")


def _validated_calibration_shots(
    data_path: Path, *, role: str = "calibration"
) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(data_path.read_text())
    if not isinstance(payload, dict):
        raise TypeError("planar calibration artifact must be a JSON object")
    config = payload.get("config")
    rows = payload.get("shots")
    if not isinstance(config, dict) or set(config) != _PLANAR_SHOT_CONFIG_KEYS:
        raise ValueError("planar calibration artifact has an invalid shot config")
    _validate_shot_provenance(payload)
    domain = config.get("seed_domain")
    campaign_seed = config.get("campaign_seed")
    if not isinstance(domain, str) or f"/{role}/" not in domain:
        raise ValueError(f"requires a {role}-domain shot artifact")
    if (
        type(campaign_seed) is not int
        or campaign_seed != int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
        or config.get("schema_version") != 1
        or config.get("code_distance") != 5
        or config.get("error_rates") != list(_PLANAR_ERROR_RATES)
        or type(config.get("shots_per_rate")) is not int
        or int(config["shots_per_rate"]) <= 0
        or config.get("noise_model") != _PLANAR_NOISE_MODEL
    ):
        raise ValueError("planar calibration artifact has an invalid frozen shot config")
    if not isinstance(rows, list) or len(rows) != 2 * int(config["shots_per_rate"]):
        raise ValueError("planar calibration artifact has an invalid shot count")

    code = PlanarCode(5, 5)
    model = DepolarizingErrorModel()
    expected_indices = set(range(int(config["shots_per_rate"])))
    by_rate_indices: dict[float, set[int]] = {rate: set() for rate in _PLANAR_ERROR_RATES}
    shot_ids: set[str] = set()
    sampler_seeds: set[int] = set()
    validated: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("planar calibration artifact has a malformed shot row")
        rate = row.get("error_rate")
        shot_index = row.get("shot_index")
        sampler_seed = row.get("sampler_seed")
        shot_id = row.get("shot_id")
        if (
            type(rate) not in (int, float)
            or float(rate) not in _PLANAR_ERROR_RATES
            or type(shot_index) is not int
            or type(sampler_seed) is not int
            or not isinstance(shot_id, str)
        ):
            raise ValueError("planar calibration artifact has invalid shot metadata")
        error_rate = float(rate)
        if (
            shot_index not in expected_indices
            or sampler_seed != _calibration_shot_seed(domain, error_rate, shot_index)
            or shot_id != f"d5/p{error_rate:.6f}/i{shot_index:06d}"
            or shot_id in shot_ids
            or sampler_seed in sampler_seeds
        ):
            raise ValueError("planar calibration artifact has invalid replay metadata")
        error = np.asarray(row.get("error_bsf"))
        syndrome = np.asarray(row.get("syndrome"))
        replayed_error = np.asarray(
            model.generate(code, error_rate, np.random.default_rng(sampler_seed)), dtype=np.uint8
        )
        replayed_syndrome = np.asarray(pt.bsp(replayed_error, code.stabilizers.T), dtype=np.uint8)
        if not np.array_equal(error, replayed_error) or not np.array_equal(
            syndrome, replayed_syndrome
        ):
            raise ValueError("planar calibration physical errors or syndromes do not replay")
        by_rate_indices[error_rate].add(shot_index)
        shot_ids.add(shot_id)
        sampler_seeds.add(sampler_seed)
        validated.append(
            {
                "error_rate": error_rate,
                "error": replayed_error,
                "syndrome": replayed_syndrome,
            }
        )
    if any(indices != expected_indices for indices in by_rate_indices.values()):
        raise ValueError("planar calibration artifact is missing or duplicates shot indices")
    return dict(config), validated


def _git_provenance(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _cmwpm_provenance() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = _git_provenance(repository_root)
    return {
        "source_sha256": {
            label: sha256_file(repository_root / label) for label in _CMWPM_SOURCE_LABELS
        },
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qecsim": qecsim.__version__,
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def _parameters_tuple(parameters: Mapping[str, object]) -> tuple[int, int, str, int]:
    expected_keys = {"factor", "max_iterations", "box_shape", "distance_algorithm"}
    if set(parameters) != expected_keys:
        raise ValueError("CMWPM parameters are malformed")
    factor = parameters["factor"]
    max_iterations = parameters["max_iterations"]
    box_shape = parameters["box_shape"]
    distance_algorithm = parameters["distance_algorithm"]
    if (
        type(factor) is not int
        or type(max_iterations) is not int
        or type(box_shape) is not str
        or type(distance_algorithm) is not int
    ):
        raise ValueError("CMWPM parameters are malformed")
    return factor, max_iterations, box_shape, distance_algorithm


def _selection_key(candidate: Mapping[str, object]) -> tuple[int, int, int, int, str, int]:
    parameters = candidate.get("parameters")
    pooled = candidate.get("pooled")
    per_rate = candidate.get("per_rate")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(pooled, Mapping)
        or not isinstance(per_rate, list)
    ):
        raise TypeError("CMWPM candidate summary is malformed")
    failures = pooled.get("failures")
    rate_failures = [row.get("failures") for row in per_rate if isinstance(row, Mapping)]
    if (
        type(failures) is not int
        or failures < 0
        or len(rate_failures) != len(per_rate)
        or any(type(value) is not int or value < 0 for value in rate_failures)
    ):
        raise ValueError("CMWPM candidate failures are malformed")
    factor, max_iterations, box_shape, distance_algorithm = _parameters_tuple(parameters)
    return (
        failures,
        max(rate_failures, default=0),
        factor,
        max_iterations,
        box_shape,
        distance_algorithm,
    )


def select_cmwpm_candidate(candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Choose one configuration shared across rates by the frozen total ordering."""
    if not candidates:
        raise ValueError("CMWPM selection requires at least one candidate")
    selected = min(candidates, key=_selection_key)
    return dict(selected)


def _empty_rate_summary(error_rate: float) -> dict[str, object]:
    return {
        "error_rate": error_rate,
        "shots": 0,
        "failures": 0,
        "logical_failures": 0,
        "invalid_recoveries": 0,
        "decode_exceptions": 0,
    }


def _record_recovery(
    summary: dict[str, object],
    *,
    code: PlanarCode,
    error: np.ndarray,
    syndrome: np.ndarray,
    decoder: object,
) -> None:
    summary["shots"] = int(summary["shots"]) + 1
    try:
        recovery = decoder.decode(code, syndrome)  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - decoder failures must remain measured calibration failures.
        summary["decode_exceptions"] = int(summary["decode_exceptions"]) + 1
        summary["failures"] = int(summary["failures"]) + 1
        return
    try:
        scored = score_recovery(code, error, syndrome, np.asarray(recovery))
    except TypeError, ValueError:
        summary["invalid_recoveries"] = int(summary["invalid_recoveries"]) + 1
        summary["failures"] = int(summary["failures"]) + 1
        return
    if not bool(scored["syndrome_valid"]):
        summary["invalid_recoveries"] = int(summary["invalid_recoveries"]) + 1
        summary["failures"] = int(summary["failures"]) + 1
    elif bool(scored["logical_failure"]):
        summary["logical_failures"] = int(summary["logical_failures"]) + 1
        summary["failures"] = int(summary["failures"]) + 1


def _pooled_summary(per_rate: Sequence[Mapping[str, object]]) -> dict[str, int]:
    fields = (
        "shots",
        "failures",
        "logical_failures",
        "invalid_recoveries",
        "decode_exceptions",
    )
    return {field: sum(int(summary[field]) for summary in per_rate) for field in fields}


def _evaluate_decoder(
    decoder: object, shots: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    code = PlanarCode(5, 5)
    per_rate = {rate: _empty_rate_summary(rate) for rate in _PLANAR_ERROR_RATES}
    for shot in shots:
        error_rate = float(shot["error_rate"])
        _record_recovery(
            per_rate[error_rate],
            code=code,
            error=np.asarray(shot["error"], dtype=np.uint8),
            syndrome=np.asarray(shot["syndrome"], dtype=np.uint8),
            decoder=decoder,
        )
    summaries = [per_rate[rate] for rate in _PLANAR_ERROR_RATES]
    return summaries, _pooled_summary(summaries)


def calibrate_cmwpm(grid_path: Path, data_path: Path, output_dir: Path) -> dict[str, object]:
    """Evaluate the complete CMWPM grid solely on replayed calibration shots."""
    grid_path = Path(grid_path)
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_path = output_dir / "planar_cmwpm_selection.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite CMWPM selection: {output_path}")
    candidates = cmwpm_grid(grid_path)
    calibration_data_config, shots = _validated_calibration_shots(data_path)

    candidate_rows: list[dict[str, object]] = []
    for parameters in candidates:
        factor, max_iterations, box_shape, distance_algorithm = _parameters_tuple(parameters)
        decoder = PlanarCMWPMDecoder(
            factor=factor,
            max_iterations=max_iterations,
            box_shape=box_shape,
            distance_algorithm=distance_algorithm,
        )
        per_rate, pooled = _evaluate_decoder(decoder, shots)
        candidate_rows.append({"parameters": parameters, "per_rate": per_rate, "pooled": pooled})
    mwpm_per_rate, mwpm_pooled = _evaluate_decoder(PlanarMWPMDecoder(), shots)
    selected = select_cmwpm_candidate(candidate_rows)
    payload: dict[str, object] = {
        "schema_version": 1,
        "grid": json.loads(grid_path.read_text()),
        "calibration_data_config": calibration_data_config,
        "grid_sha256": sha256_file(grid_path),
        "data_sha256": sha256_file(data_path),
        "provenance": _cmwpm_provenance(),
        "candidates": candidate_rows,
        "selected": selected,
        "mwpm_reference": {"per_rate": mwpm_per_rate, "pooled": mwpm_pooled},
    }
    write_canonical_json(output_path, payload)
    return payload


_ACCURACY_POLICY = {
    "schema_version": 1,
    "policy_id": "planar_shot_accuracy_v1",
    "required_screen_domain": "qldpc-fno/planar-shot-accuracy/screen/v1",
    "required_calibration_domain": "qldpc-fno/planar-shot-accuracy/calibration/v1",
    "screen_shots_per_rate": 2048,
    "calibration_shots_per_rate": 512,
    "code_distance": 5,
    "error_rates": [0.1, 0.15],
    "noise_model": _PLANAR_NOISE_MODEL,
    "tolerance": 0.01,
    "fallback_chi": 8,
    "reference_probability_tolerance": 1e-10,
    "reference_log_ratio_tolerance": 1e-8,
    "familywise_alpha": 0.05,
    "comparisons": 2,
    "maximum_class_mismatch_rate": 0.005,
    "work_bootstrap_replicates": 10000,
    "work_bootstrap_seed": 13158893872079179326,
}
_ACCURACY_SOURCE_LABELS = (
    *_CMWPM_SOURCE_LABELS,
    "src/qldpc_fno/decision/tensor_network.py",
    "src/qldpc_fno/metrics/paired.py",
    "experiments/33_run_planar_shot_accuracy.py",
)


def adjusted_wilson_upper(failures: int, shots: int, *, alpha: float) -> float:
    """One-sided Wilson upper limit at the supplied (already adjusted) alpha."""
    if (
        type(failures) is not int
        or type(shots) is not int
        or shots <= 0
        or not 0 <= failures <= shots
        or not 0 < alpha < 1
    ):
        raise ValueError("invalid Wilson counts or alpha")
    z = float(norm.ppf(1 - alpha))
    rate = failures / shots
    return min(
        1.0,
        (
            rate
            + z * z / (2 * shots)
            + z * math.sqrt(rate * (1 - rate) / shots + z * z / (4 * shots * shots))
        )
        / (1 + z * z / shots),
    )


def status_for_strata(mismatches: Sequence[int], *, shots: int) -> str:
    """Apply the frozen zero-mismatch and precision requirements."""
    if len(mismatches) != 2:
        raise ValueError("the gate requires both error-rate strata")
    upper = [adjusted_wilson_upper(count, shots, alpha=0.05 / 2) for count in mismatches]
    if any(mismatches):
        return "falsified_exact_outcome_preservation"
    if any(value > 0.005 for value in upper):
        return "unresolved_insufficient_precision"
    return "passed_exact_outcome_preservation"


def certify_reference(columns: np.ndarray, rows: np.ndarray) -> dict[str, object]:
    """Certify positive, uniquely winning, numerically consistent exact masses."""
    probabilities = []
    for masses in (columns, rows):
        values = np.asarray(masses, dtype=float)
        if (
            values.shape != (4,)
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
            or not np.isfinite(values.sum())
        ):
            raise ValueError("invalid exact reference masses")
        probabilities.append(values / values.sum())
    col, row = probabilities
    discrepancy = float(np.max(np.abs(col - row)))
    log_delta = np.log(col) - np.log(row)
    log_error = float(np.max(log_delta) - np.min(log_delta))
    margins = [float(np.sort(values)[-1] - np.sort(values)[-2]) for values in probabilities]
    if (
        int(np.argmax(col)) != int(np.argmax(row))
        or discrepancy > 1e-10
        or log_error > 1e-8
        or any(margin <= 2 * discrepancy for margin in margins)
    ):
        raise ValueError("ambiguous or inconsistent exact reference")
    return {
        "selected_class": int(np.argmax(col)),
        "maximum_probability_discrepancy": discrepancy,
        "maximum_log_ratio_discrepancy": log_error,
        "probability_margins": margins,
    }


def _current_sources(provenance: Mapping[str, object], labels: Sequence[str]) -> None:
    root = Path(__file__).resolve().parents[3]
    expected = {label: sha256_file(root / label) for label in labels}
    if provenance.get("source_sha256") != expected:
        raise ValueError("stale or malformed source hashes")


def _validate_selection(selection: dict[str, object]) -> tuple[dict[str, object], bool]:
    """Validate the immutable calibration boundary, without selecting on screen data."""
    config = selection.get("calibration_data_config")
    if not isinstance(config, dict) or set(config) != _PLANAR_SHOT_CONFIG_KEYS:
        raise ValueError("selection calibration config is malformed")
    domain = config.get("seed_domain")
    if (
        not isinstance(domain, str)
        or "/calibration/" not in domain
        or config.get("campaign_seed")
        != int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
        or config.get("schema_version") != 1
        or config.get("code_distance") != 5
        or config.get("error_rates") != list(_PLANAR_ERROR_RATES)
        or config.get("noise_model") != _PLANAR_NOISE_MODEL
        or type(config.get("shots_per_rate")) is not int
        or config["shots_per_rate"] <= 0
    ):
        raise ValueError("selection calibration domain/config is invalid")
    if (
        selection.get("schema_version") != 1
        or selection.get("grid") != _CMWPM_GRID
        or not _is_sha256(selection.get("data_sha256"))
        or not _is_sha256(selection.get("grid_sha256"))
    ):
        raise ValueError("selection grid/data hash is malformed")
    provenance = selection.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("selection provenance missing")
    _current_sources(provenance, _CMWPM_SOURCE_LABELS)
    if provenance.get("dependencies") != _cmwpm_provenance()["dependencies"]:
        raise ValueError("selection dependency versions do not match")
    if type(provenance.get("git_dirty")) is not bool or not isinstance(
        provenance.get("git_commit"), str
    ):
        raise ValueError("selection git provenance is malformed")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("selection candidates missing")
    allowed = {
        (factor, iterations, shape, algorithm)
        for factor in (1, 2, 3, 4)
        for iterations in (2, 4, 8)
        for shape in ("t", "r")
        for algorithm in (2, 4)
    }
    observed = [_parameters_tuple(candidate["parameters"]) for candidate in candidates]
    if len(set(observed)) != len(observed) or not set(observed) <= allowed:
        raise ValueError("selection candidate grid is inconsistent")
    for candidate in [*candidates, selection.get("mwpm_reference", {})]:
        per_rate = candidate.get("per_rate")
        if not isinstance(per_rate, list) or len(per_rate) != 2:
            raise ValueError("selection rate summaries malformed")
        for rate, summary in zip(_PLANAR_ERROR_RATES, per_rate, strict=True):
            fields = set(_empty_rate_summary(rate)) - {"error_rate"}
            if (
                set(summary) != fields | {"error_rate"}
                or summary["error_rate"] != rate
                or any(type(summary[key]) is not int or summary[key] < 0 for key in fields)
                or summary["shots"] != config["shots_per_rate"]
                or summary["failures"] > summary["shots"]
                or summary["failures"]
                != sum(
                    summary[key]
                    for key in ("logical_failures", "invalid_recoveries", "decode_exceptions")
                )
            ):
                raise ValueError("selection rate arithmetic is inconsistent")
        if candidate.get("pooled") != _pooled_summary(per_rate):
            raise ValueError("selection pooled arithmetic is inconsistent")
    if selection.get("selected") != select_cmwpm_candidate(candidates):
        raise ValueError("selection does not match frozen tie-break")
    canonical = (
        domain == _ACCURACY_POLICY["required_calibration_domain"]
        and config["shots_per_rate"] == 512
        and set(observed) == allowed
        and provenance["git_dirty"] is False
    )
    return config, canonical


def _screen_inputs(policy_path: Path, screen_path: Path, selection_path: Path) -> dict[str, object]:
    policy = json.loads(policy_path.read_text())
    if policy != _ACCURACY_POLICY:
        raise ValueError("accuracy policy differs from frozen policy")
    selection = json.loads(selection_path.read_text())
    calibration_config, calibration_canonical = _validate_selection(selection)
    config, replayed = _validated_calibration_shots(screen_path, role="screen")
    screen = json.loads(screen_path.read_text())
    _current_sources(screen["provenance"], _PLANAR_SHOT_SOURCE_LABELS)
    if screen["provenance"]["qecsim_version"] != qecsim.__version__:
        raise ValueError("screen dependency version does not match")
    domain = str(config["seed_domain"])
    if (
        "tensor" in domain
        or "confirmation" in domain
        or domain == calibration_config["seed_domain"]
    ):
        raise ValueError("forbidden or colliding screen domain")
    calibration_seeds = {
        _calibration_shot_seed(str(calibration_config["seed_domain"]), rate, index)
        for rate in _PLANAR_ERROR_RATES
        for index in range(calibration_config["shots_per_rate"])
    }
    if calibration_seeds & {shot["sampler_seed"] for shot in screen["shots"]}:
        raise ValueError("screen/calibration sampler seed collision")
    current = _cmwpm_provenance()
    canonical = bool(
        calibration_canonical
        and config["shots_per_rate"] == 2048
        and domain == policy["required_screen_domain"]
        and not screen["provenance"]["git_dirty"]
        and not current["git_dirty"]
    )
    root = Path(__file__).resolve().parents[3]
    # Canonical config hashes bind actual frozen bytes; temporary reduced configs
    # remain available for integration tests and cannot acquire canonical status.
    if canonical:
        canonical = (
            config == json.loads((root / "configs/planar_shot_screen.json").read_text())
            and calibration_config
            == json.loads((root / "configs/planar_shot_calibration.json").read_text())
            and screen["provenance"]["config_sha256"]
            == sha256_file(root / "configs/planar_shot_screen.json")
            and selection["grid_sha256"] == sha256_file(root / "configs/planar_cmwpm_grid.json")
            and sha256_file(policy_path)
            == sha256_file(root / "configs/planar_shot_accuracy_policy.json")
        )
    return {
        "policy": policy,
        "selection": selection,
        "config": config,
        "screen": screen,
        "replayed": replayed,
        "canonical": canonical,
        "input_sha256": {
            "policy": sha256_file(policy_path),
            "screen": sha256_file(screen_path),
            "selection": sha256_file(selection_path),
        },
    }


def _stable_work(value: object) -> object:
    """Keep deterministic counters and traces, excluding measured wall time."""
    if isinstance(value, dict):
        return {key: _stable_work(item) for key, item in value.items() if "wall_seconds" not in key}
    if isinstance(value, list):
        return [_stable_work(item) for item in value]
    return value


def _tensor_record(result: PlanarCosetMasses) -> dict[str, object]:
    return {
        "masses": result.masses.tolist(),
        "probabilities": result.probabilities.tolist(),
        "selected_class": result.selected_class,
        "mode": result.mode,
        "chi": result.chi,
        "tol": result.tol,
        "work": _stable_work(result.work),
    }


def _scored_arm(
    code: PlanarCode, shot: dict[str, object], recovery: np.ndarray
) -> dict[str, object]:
    scored = score_recovery(code, shot["error"], shot["syndrome"], recovery)
    return {
        "recovery_bsf": np.asarray(recovery).tolist(),
        "syndrome_valid": bool(scored["syndrome_valid"]),
        "logical_signature": scored["logical_signature"].tolist(),
        "recovery_syndrome": scored["recovery_syndrome"].tolist(),
        "residual_syndrome": scored["residual_syndrome"].tolist(),
        "logical_failure": bool(scored["logical_failure"]),
        "failure": not scored["syndrome_valid"] or bool(scored["logical_failure"]),
        "decode_exception": None,
    }


def _failed_arm(reason: str) -> dict[str, object]:
    return {
        "recovery_bsf": None,
        "syndrome_valid": False,
        "logical_signature": None,
        "recovery_syndrome": None,
        "residual_syndrome": None,
        "logical_failure": True,
        "failure": True,
        "decode_exception": reason,
    }


def _evaluate_screen(inputs: dict[str, object]) -> list[dict[str, object]]:
    code = PlanarCode(5, 5)
    cmwpm = PlanarCMWPMDecoder(**inputs["selection"]["selected"]["parameters"])
    mwpm = PlanarMWPMDecoder()
    rows = []
    for metadata, shot in zip(inputs["screen"]["shots"], inputs["replayed"], strict=True):
        rate, syndrome = shot["error_rate"], shot["syndrome"]
        tensor = {}
        invalid_tensor_work = {}
        reference_error = None
        certificate = None
        for mode in ("columns", "rows"):
            try:
                tensor[f"exact_{mode}"] = planar_mps_coset_masses(
                    rows=5,
                    columns=5,
                    syndrome=syndrome,
                    error_rate=rate,
                    chi=None,
                    tol=None,
                    mode=mode,
                    trace_work=True,
                )
            except ValueError as error:
                reference_error = type(error).__name__
                invalid_tensor_work[f"exact_{mode}"] = _stable_work(getattr(error, "work", None))
        if reference_error is None:
            try:
                certificate = certify_reference(
                    tensor["exact_columns"].masses, tensor["exact_rows"].masses
                )
            except ValueError as error:
                reference_error = str(error)
        arms = {}
        exact_class = certificate["selected_class"] if certificate else None
        arms["exact"] = (
            _scored_arm(code, shot, logical_class_recovery(code, syndrome, exact_class))
            if certificate
            else _failed_arm("invalid_exact_reference")
        )
        try:
            decision = two_view_tolerance_decision(syndrome, rate)
            policy_class = decision["selected_class"]
            arms["policy"] = (
                _scored_arm(code, shot, logical_class_recovery(code, syndrome, policy_class))
                if policy_class is not None
                else _failed_arm("invalid_policy_fallback")
            )
            for name in ("tolerance_columns", "tolerance_rows", "fallback_columns"):
                if decision[name] is not None:
                    tensor[name] = decision[name]
            policy_work = decision["estimated_arithmetic_flops"]
            for name, work in decision["invalid_action_work"].items():
                invalid_tensor_work[
                    f"tolerance_{name}" if name in ("columns", "rows") else name
                ] = work
        except ValueError as error:
            decision = {"used_fallback": True, "accepted": False}
            policy_class, policy_work = None, 0
            arms["policy"] = _failed_arm(type(error).__name__)
        fixed = tensor.get("fallback_columns")
        if "fallback_columns" in invalid_tensor_work:
            invalid_tensor_work["fixed_columns"] = invalid_tensor_work["fallback_columns"]
        elif fixed is None:
            try:
                fixed = planar_mps_coset_masses(
                    rows=5,
                    columns=5,
                    syndrome=syndrome,
                    error_rate=rate,
                    chi=8,
                    tol=None,
                    mode="columns",
                    trace_work=True,
                )
            except InvalidCosetMassError as error:
                invalid_tensor_work["fixed_columns"] = _stable_work(error.work)
        if fixed is not None:
            tensor["fixed_columns"] = fixed
        for name, decoder in (("cmwpm", cmwpm), ("mwpm", mwpm)):
            try:
                arms[name] = _scored_arm(code, shot, decoder.decode(code, syndrome))
            except Exception as error:  # noqa: BLE001 - exceptions remain measured failures.
                arms[name] = _failed_arm(type(error).__name__)
        rows.append(
            {
                "shot_id": metadata["shot_id"],
                "error_rate": rate,
                "shot_index": metadata["shot_index"],
                "sampler_seed": metadata["sampler_seed"],
                "error_bsf": shot["error"].tolist(),
                "syndrome": syndrome.tolist(),
                "reference_valid": certificate is not None,
                "reference_error": reference_error,
                "reference_certificate": certificate,
                "exact_class": exact_class,
                "policy_class": policy_class,
                "accepted": decision["accepted"],
                "used_fallback": decision["used_fallback"],
                "class_mismatch": certificate is not None and policy_class != exact_class,
                "outcome_discordance": arms["exact"]["failure"] != arms["policy"]["failure"],
                "arms": arms,
                "tensor": {name: _tensor_record(value) for name, value in tensor.items()},
                "invalid_tensor_work": invalid_tensor_work,
                "work": {
                    "policy": policy_work,
                    "fixed_chi8": _estimated_flops(fixed)
                    if fixed is not None
                    else int(
                        (invalid_tensor_work.get("fixed_columns") or {}).get(
                            "estimated_arithmetic_flops", 0
                        )
                    ),
                    "exact": sum(
                        _estimated_flops(tensor[name])
                        if name in tensor
                        else int(
                            (invalid_tensor_work.get(name) or {}).get(
                                "estimated_arithmetic_flops", 0
                            )
                        )
                        for name in ("exact_columns", "exact_rows")
                    ),
                },
            }
        )
    return rows


def _work_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Paired bootstrap of mean-work ratios, resampling within each rate."""
    groups = [
        np.array(
            [
                [row["work"][key] for key in ("policy", "fixed_chi8", "exact")]
                for row in rows
                if row["error_rate"] == rate
            ],
            dtype=float,
        )
        for rate in _PLANAR_ERROR_RATES
    ]
    groups = [group for group in groups if len(group)]
    rng = np.random.default_rng(_ACCURACY_POLICY["work_bootstrap_seed"])
    replicates = _ACCURACY_POLICY["work_bootstrap_replicates"]
    # Common sampled indices preserve pairing across all three work arms.
    samples = np.zeros((replicates, 3))
    for group in groups:
        for start in range(0, replicates, 100):
            indices = rng.integers(len(group), size=(min(100, replicates - start), len(group)))
            samples[start : start + len(indices)] += group[indices].mean(axis=1) / len(groups)
    means = np.mean([group.mean(axis=0) for group in groups], axis=0)
    comparisons = {}
    for index, name in ((1, "fixed_chi8"), (2, "exact")):
        if means[index] <= 0 or np.any(samples[:, index] <= 0):
            comparisons[name] = {"ratio": None, "savings_fraction": None, "ratio_95ci": None}
        else:
            ratios = samples[:, 0] / samples[:, index]
            interval = np.quantile(ratios, [0.025, 0.975]).tolist()
            comparisons[name] = {
                "ratio": float(means[0] / means[index]),
                "savings_fraction": float(1 - means[0] / means[index]),
                "ratio_95ci": interval,
                "savings_fraction_95ci": [1 - interval[1], 1 - interval[0]],
            }
    return {
        "estimand": "ratio of equal-rate mixture mean estimated arithmetic work",
        "descriptive_only": True,
        "resampling_unit": "physical shot, paired within error rate",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": _ACCURACY_POLICY["work_bootstrap_seed"],
        "totals": {
            key: sum(row["work"][key] for row in rows) for key in ("policy", "fixed_chi8", "exact")
        },
        "policy_relative_to": comparisons,
    }


def _summarize_screen(rows: list[dict[str, object]], *, canonical: bool) -> dict[str, object]:
    strata = []
    for rate in _PLANAR_ERROR_RATES:
        group = [row for row in rows if row["error_rate"] == rate]
        outcomes = {
            name: np.array([row["arms"][name]["failure"] for row in group], dtype=bool)
            for name in ("exact", "policy", "cmwpm", "mwpm")
        }
        mismatches = sum(row["class_mismatch"] for row in group)
        discordances = sum(row["outcome_discordance"] for row in group)
        upper = adjusted_wilson_upper(mismatches, len(group), alpha=0.05 / 2)
        paired = {
            f"{first}_vs_{second}": paired_decoder_summary(outcomes[first], outcomes[second])
            for first, second in (
                ("exact", "policy"),
                ("exact", "cmwpm"),
                ("exact", "mwpm"),
                ("policy", "cmwpm"),
            )
        }
        strata.append(
            {
                "error_rate": rate,
                "shots": len(group),
                "class_mismatches": mismatches,
                "outcome_discordances": discordances,
                "adjusted_wilson_upper": upper,
                "gate_passed": (
                    mismatches == discordances == 0
                    and upper <= 0.005
                    and all(row["reference_valid"] for row in group)
                    and all(arm["syndrome_valid"] for row in group for arm in row["arms"].values())
                ),
                "fallbacks": sum(row["used_fallback"] for row in group),
                "invalid_references": sum(not row["reference_valid"] for row in group),
                "invalid_recoveries": {
                    name: sum(not row["arms"][name]["syndrome_valid"] for row in group)
                    for name in outcomes
                },
                "arms": {
                    name: paired_decoder_summary(value, value)["baseline"]
                    for name, value in outcomes.items()
                },
                "paired": paired,
                "single_digit_counts": {
                    "arm_failures": {
                        name: int(value.sum())
                        for name, value in outcomes.items()
                        if value.sum() < 10
                    },
                    "paired_discordances": {
                        name: value["discordant_pairs"]
                        for name, value in paired.items()
                        if value["discordant_pairs"] < 10
                    },
                },
                "work": _work_summary(group),
            }
        )
    if not canonical:
        status = "unresolved_noncanonical_data"
    elif any(not row["reference_valid"] for row in rows):
        status = "invalid_exact_reference"
    elif any(not arm["syndrome_valid"] for row in rows for arm in row["arms"].values()):
        status = "invalid_recovery"
    elif any(row["class_mismatch"] or row["outcome_discordance"] for row in rows):
        status = "falsified_exact_outcome_preservation"
    elif not all(stratum["gate_passed"] for stratum in strata):
        status = "unresolved_insufficient_precision"
    else:
        status = "passed_exact_outcome_preservation"
    return {"status": status, "per_rate": strata, "work": _work_summary(rows)}


def verify_screen_result(
    result: dict[str, object],
    policy_path: Path,
    screen_path: Path,
    selection_path: Path,
    *,
    prepublication: bool = False,
) -> None:
    """Replay the complete artifact; only the runner accepts an unfinished replay flag."""
    inputs = _screen_inputs(Path(policy_path), Path(screen_path), Path(selection_path))
    expected_provenance = _screen_provenance(inputs)
    expected_integrity = {"independent_replay": not prepublication, "seed_domains_disjoint": True}
    if (
        result.get("provenance") != expected_provenance
        or result.get("integrity") != expected_integrity
    ):
        raise ValueError("screen provenance/integrity replay mismatch")
    replayed = _evaluate_screen(inputs)
    expected = _screen_result(inputs, replayed, independent_replay=not prepublication)
    # Full canonical serialization also distinguishes booleans from integer 0/1,
    # and rejects nonfinite numbers, missing fields, and unverified extra fields.
    if json.dumps(result, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("screen result independent replay mismatch")


def _screen_provenance(inputs: dict[str, object]) -> dict[str, object]:
    """Bind the producer freeze, independent of later documentation-only commits."""
    root = Path(__file__).resolve().parents[3]
    provenance = _cmwpm_provenance()
    for field in ("git_commit", "git_dirty"):
        provenance[field] = inputs["screen"]["provenance"][field]
    provenance["source_sha256"] = {
        label: sha256_file(root / label) for label in _ACCURACY_SOURCE_LABELS
    }
    provenance["input_sha256"] = inputs["input_sha256"]
    provenance["calibration_data_sha256"] = inputs["selection"]["data_sha256"]
    return provenance


def _screen_result(
    inputs: dict[str, object],
    rows: list[dict[str, object]],
    *,
    independent_replay: bool,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "canonical": inputs["canonical"],
        "policy": inputs["policy"],
        "config": inputs["config"],
        "selected_cmwpm_parameters": inputs["selection"]["selected"]["parameters"],
        "shots": rows,
        "provenance": _screen_provenance(inputs),
        "integrity": {"independent_replay": independent_replay, "seed_domains_disjoint": True},
        **_summarize_screen(rows, canonical=inputs["canonical"]),
    }


def run_planar_shot_accuracy(
    policy_path: Path,
    screen_path: Path,
    selection_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Score and independently replay the immutable held-out accuracy screen."""
    output_path = Path(output_dir) / "planar_shot_accuracy.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite screen: {output_path}")
    inputs = _screen_inputs(Path(policy_path), Path(screen_path), Path(selection_path))
    rows = _evaluate_screen(inputs)
    result = _screen_result(inputs, rows, independent_replay=False)
    verify_screen_result(result, policy_path, screen_path, selection_path, prepublication=True)
    result["integrity"]["independent_replay"] = True
    write_canonical_json(output_path, result)
    return result
