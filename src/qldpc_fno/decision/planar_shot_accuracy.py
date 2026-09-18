"""Auditable recovery and tolerance-policy primitives for planar-code shots."""

from __future__ import annotations

import hashlib
import json
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

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision.tensor_network import PlanarCosetMasses, planar_mps_coset_masses

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
    columns = planar_mps_coset_masses(
        rows=5,
        columns=5,
        syndrome=syndrome_array,
        error_rate=error_rate,
        chi=None,
        tol=0.01,
        mode="columns",
        trace_work=True,
    )
    rows = planar_mps_coset_masses(
        rows=5,
        columns=5,
        syndrome=syndrome_array,
        error_rate=error_rate,
        chi=None,
        tol=0.01,
        mode="rows",
        trace_work=True,
    )
    accepted = columns.selected_class == rows.selected_class
    fallback: PlanarCosetMasses | None = None
    selected = columns
    if not accepted:
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
        selected = fallback
    actions = (columns, rows) if fallback is None else (columns, rows, fallback)
    return {
        "selected_class": selected.selected_class,
        "accepted": accepted,
        "used_fallback": fallback is not None,
        "tolerance_columns": columns,
        "tolerance_rows": rows,
        "fallback_columns": fallback,
        "estimated_arithmetic_flops": sum(_estimated_flops(action) for action in actions),
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


def _validated_calibration_shots(data_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    if not isinstance(domain, str) or "/calibration/" not in domain:
        raise ValueError("CMWPM selection requires a calibration-domain shot artifact")
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
        if not np.array_equal(error, replayed_error) or not np.array_equal(syndrome, replayed_syndrome):
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
    if not isinstance(parameters, Mapping) or not isinstance(pooled, Mapping) or not isinstance(per_rate, list):
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
    summary: dict[str, object], *, code: PlanarCode, error: np.ndarray, syndrome: np.ndarray, decoder: object
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
    except (TypeError, ValueError):
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
        candidate_rows.append(
            {"parameters": parameters, "per_rate": per_rate, "pooled": pooled}
        )
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
