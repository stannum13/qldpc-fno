"""Independent policy execution and physical scoring for the planar confirmation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from time import perf_counter

import numpy as np
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode

from qldpc_fno.decision.adaptive_intervention_pilot import margin_gate_accepts
from qldpc_fno.decision.planar_shot_accuracy import (
    _binary_vector,
    _stable_work,
    certify_reference,
    logical_class_recovery,
    score_recovery,
)
from qldpc_fno.decision.simple_confirmation import (
    MARGIN_THRESHOLD,
    arm_order,
    event_indicators,
)
from qldpc_fno.decision.tensor_network import (
    InvalidCosetMassError,
    planar_mps_coset_masses,
)

_ACTION_SPECS: dict[str, tuple[str, int | None, float | None]] = {
    "rows_tol003": ("rows", None, 0.003),
    "columns_tol01": ("columns", None, 0.01),
    "rows_tol01": ("rows", None, 0.01),
    "columns_chi8": ("columns", 8, None),
    "exact_columns": ("columns", None, None),
    "exact_rows": ("rows", None, None),
}
_POLICY_ACTIONS = {
    "fixed_rows_tol003": ("rows_tol003",),
    "margin_columns_chi8": ("columns_tol01", "rows_tol01"),
    "fixed_columns_chi8": ("columns_chi8",),
}
_RATES = (0.1, 0.15)


def _finite_rate(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("error_rate must be finite")
    result = float(value)
    if result not in _RATES:
        raise ValueError("error_rate must be a frozen confirmation rate")
    return result


def _decomposition_cost(kind: str, rows: int, columns: int) -> int:
    q = min(rows, columns)
    if kind == "svd":
        return int(4 * rows * columns * q + 8 * q**3)
    if kind == "qr":
        return int(2 * rows * columns * q - 2 * q**3 / 3)
    raise ValueError("unknown decomposition kind")


def _metered_work(value: object) -> tuple[dict[str, object] | None, int | None]:
    """Remove host time and reconcile every recorded decomposition charge."""
    if value is None:
        return None, None
    stable = _stable_work(value)
    if not isinstance(stable, dict):
        raise TypeError("action work must be an object or null")
    required_counts = (
        "svd_calls",
        "qr_calls",
        "einsum_estimated_flops",
        "estimated_dense_decomposition_flops",
        "estimated_arithmetic_flops",
    )
    for field in required_counts:
        count = stable.get(field)
        if type(count) is not int or count < 0:
            raise ValueError(f"invalid work counter: {field}")
    attempts = stable.get("decomposition_attempts")
    if not isinstance(attempts, list):
        raise TypeError("decomposition_attempts must be a list")
    kind_counts = {"svd": 0, "qr": 0}
    attempted_cost = 0
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise TypeError("decomposition attempt must be an object")
        kind = attempt.get("decomposition_kind")
        rows, columns = attempt.get("matrix_rows"), attempt.get("matrix_columns")
        succeeded, exception_type = attempt.get("succeeded"), attempt.get("exception_type")
        if (
            kind not in kind_counts
            or type(rows) is not int
            or rows <= 0
            or type(columns) is not int
            or columns <= 0
            or type(succeeded) is not bool
            or (succeeded and exception_type is not None)
            or (not succeeded and not isinstance(exception_type, str))
        ):
            raise ValueError("malformed decomposition attempt")
        expected = _decomposition_cost(str(kind), rows, columns)
        if attempt.get("estimated_flops") != expected:
            raise ValueError("decomposition attempt has an incorrect charge")
        kind_counts[str(kind)] += 1
        attempted_cost += expected
    if stable["svd_calls"] != kind_counts["svd"] or stable["qr_calls"] != kind_counts["qr"]:
        raise ValueError("decomposition call counters do not match attempts")
    if stable["estimated_dense_decomposition_flops"] != attempted_cost:
        raise ValueError("dense decomposition total does not match attempts")
    total = int(stable["einsum_estimated_flops"]) + attempted_cost
    if stable["estimated_arithmetic_flops"] != total:
        raise ValueError("total arithmetic work does not reconcile")
    return stable, total


def _valid_masses(value: object) -> tuple[list[float], list[float], int]:
    masses = np.asarray(value, dtype=float)
    total = float(masses.sum())
    if (
        masses.shape != (4,)
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0)
        or not math.isfinite(total)
        or total <= 0
    ):
        raise InvalidCosetMassError(masses, None)
    probabilities = masses / total
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities <= 0):
        raise InvalidCosetMassError(masses, None)
    return masses.tolist(), probabilities.tolist(), int(np.argmax(probabilities))


def run_action(
    invocation_id: str,
    action_id: str,
    syndrome: np.ndarray,
    error_rate: float,
    timing: list[dict],
) -> dict:
    """Execute one fresh contraction and retain deterministic work separately from time."""
    if not isinstance(invocation_id, str) or not invocation_id:
        raise ValueError("invocation_id must be a nonempty string")
    if action_id not in _ACTION_SPECS:
        raise ValueError("unknown action")
    if not isinstance(timing, list):
        raise TypeError("timing must be a list")
    syndrome_array = _binary_vector(syndrome, length=40, name="syndrome")
    rate = _finite_rate(error_rate)
    mode, chi, tolerance = _ACTION_SPECS[action_id]
    started = perf_counter()
    record: dict[str, object]
    try:
        result = planar_mps_coset_masses(
            rows=5,
            columns=5,
            syndrome=syndrome_array,
            error_rate=rate,
            chi=chi,
            tol=tolerance,
            mode=mode,
            trace_work=True,
        )
        work, estimated = _metered_work(result.work)
        try:
            masses, probabilities, selected_class = _valid_masses(result.masses)
        except InvalidCosetMassError as error:
            error.work = result.work
            raise
        record = {
            "valid": True,
            "exception_type": None,
            "exception_message": None,
            "masses": masses,
            "probabilities": probabilities,
            "selected_class": selected_class,
            "work": work,
            "estimated_arithmetic_flops": estimated,
        }
    except InvalidCosetMassError as error:
        work, estimated = _metered_work(error.work)
        raw_masses = np.asarray(error.masses, dtype=float)
        masses = (
            raw_masses.tolist()
            if raw_masses.shape == (4,) and np.all(np.isfinite(raw_masses))
            else None
        )
        record = {
            "valid": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "masses": masses,
            "probabilities": None,
            "selected_class": None,
            "work": work,
            "estimated_arithmetic_flops": estimated,
        }
    finally:
        elapsed = perf_counter() - started
        # Unexpected exceptions also retain their engineering duration in memory;
        # callers abort rather than publishing this incomplete record.
        if "record" in locals():
            timing.append(
                {
                    "invocation_id": invocation_id,
                    "valid": bool(record["valid"]),
                    "exception_type": record["exception_type"],
                    "wall_seconds": elapsed,
                }
            )
    return {
        "invocation_id": invocation_id,
        "action_id": action_id,
        "mode": mode,
        "chi": chi,
        "tol": tolerance,
        **record,
    }


def _policy_recovery(syndrome: np.ndarray, selected_class: int | None) -> tuple[object, bool]:
    if selected_class is None:
        return None, False
    code = PlanarCode(5, 5)
    recovery = logical_class_recovery(code, syndrome, selected_class)
    recovery_syndrome = np.asarray(pt.bsp(recovery, code.stabilizers.T), dtype=np.uint8)
    return recovery.tolist(), bool(np.array_equal(recovery_syndrome, syndrome))


def _sum_work(actions: list[dict]) -> int | None:
    values = [action["estimated_arithmetic_flops"] for action in actions]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def run_policy(
    policy_id: str,
    syndrome: np.ndarray,
    error_rate: float,
    timing: list[dict],
) -> tuple[dict, list[dict]]:
    """Run one policy with no access to physical truth or the exact reference."""
    if policy_id not in _POLICY_ACTIONS:
        raise ValueError("unknown policy")
    started = perf_counter()
    actions = [
        run_action(f"{policy_id}/{name}", name, syndrome, error_rate, timing)
        for name in _POLICY_ACTIONS[policy_id]
    ]
    accepted: bool | None = None
    if policy_id == "margin_columns_chi8":
        columns, rows = actions
        accepted = margin_gate_accepts(
            columns["probabilities"],
            columns["selected_class"],
            rows["probabilities"],
            rows["selected_class"],
            threshold=MARGIN_THRESHOLD,
        )
        if not accepted:
            actions.append(
                run_action(
                    f"{policy_id}/columns_chi8",
                    "columns_chi8",
                    syndrome,
                    error_rate,
                    timing,
                )
            )
        selected = actions[0] if accepted else actions[-1]
    else:
        selected = actions[0]
    selected_class = selected["selected_class"] if selected["valid"] else None
    recovery, syndrome_valid = _policy_recovery(syndrome, selected_class)
    policy = {
        "policy_id": policy_id,
        "action_invocation_ids": [action["invocation_id"] for action in actions],
        "gate_accepted": accepted,
        "selected_class": selected_class,
        "recovery_bsf": recovery,
        "syndrome_valid": syndrome_valid,
        "logical_signature": None,
        "physical_failure": None,
        "valid_recovery": bool(selected["valid"] and syndrome_valid),
        "class_mismatch": None,
        "outcome_discordance": None,
        "estimated_arithmetic_flops": _sum_work(actions),
    }
    timing.append({"policy_id": policy_id, "wall_seconds": perf_counter() - started})
    return policy, actions


def _shot_field(shot: Mapping[str, object], field: str) -> object:
    if field not in shot:
        raise ValueError(f"shot is missing {field}")
    return shot[field]


def _score_policy(
    code: PlanarCode,
    policy: dict,
    error: np.ndarray,
    syndrome: np.ndarray,
    *,
    reference_valid: bool,
    reference_class: int | None,
    reference_failure: bool | None,
) -> None:
    if policy["valid_recovery"]:
        score = score_recovery(
            code, error, syndrome, np.asarray(policy["recovery_bsf"], dtype=np.uint8)
        )
        policy["syndrome_valid"] = bool(score["syndrome_valid"])
        policy["valid_recovery"] = bool(score["syndrome_valid"])
        policy["logical_signature"] = score["logical_signature"].tolist()
        policy["physical_failure"] = not bool(score["syndrome_valid"]) or bool(
            score["logical_failure"]
        )
    else:
        policy["logical_signature"] = None
        policy["physical_failure"] = True
    mismatch, discordance = event_indicators(
        reference_valid=reference_valid,
        policy_valid=bool(policy["valid_recovery"]),
        policy_class=policy["selected_class"] if policy["valid_recovery"] else None,
        reference_class=reference_class,
        policy_failure=policy["physical_failure"] if policy["valid_recovery"] else None,
        reference_failure=reference_failure,
    )
    policy["class_mismatch"] = mismatch
    policy["outcome_discordance"] = discordance


def evaluate_shot(shot: dict, *, timing: list[dict]) -> dict:
    """Run all independent arms/references, then add truth-dependent labels."""
    if not isinstance(shot, dict) or not isinstance(timing, list):
        raise TypeError("shot must be an object and timing must be a list")
    code = PlanarCode(5, 5)
    error = _binary_vector(_shot_field(shot, "error_bsf"), length=2 * code.n_k_d[0], name="error")
    syndrome = _binary_vector(
        _shot_field(shot, "syndrome"), length=code.stabilizers.shape[0], name="syndrome"
    )
    if not np.array_equal(np.asarray(pt.bsp(error, code.stabilizers.T), dtype=np.uint8), syndrome):
        raise ValueError("physical error and syndrome are not joined")
    rate = _finite_rate(_shot_field(shot, "error_rate"))
    index = _shot_field(shot, "shot_index")
    if type(index) is not int or index < 0:
        raise ValueError("invalid shot index")
    shot_started = perf_counter()
    event_timing: list[dict] = []
    policies: list[dict] = []
    actions: list[dict] = []
    order = arm_order(index)
    for policy_id in order:
        policy, policy_actions = run_policy(policy_id, syndrome, rate, event_timing)
        policies.append(policy)
        actions.extend(policy_actions)

    reference_order = ("columns", "rows") if index % 2 == 0 else ("rows", "columns")
    reference_actions: dict[str, dict] = {}
    for mode in reference_order:
        action = run_action(
            f"reference/exact_{mode}", f"exact_{mode}", syndrome, rate, event_timing
        )
        actions.append(action)
        reference_actions[mode] = action

    numerical_certificate: dict[str, object] | None = None
    reference_exception_type: str | None = None
    reference_exception_message: str | None = None
    columns, rows = reference_actions["columns"], reference_actions["rows"]
    if columns["valid"] and rows["valid"]:
        try:
            col = np.asarray(columns["probabilities"], dtype=float)
            row = np.asarray(rows["probabilities"], dtype=float)
            if (
                col.shape != (4,)
                or row.shape != (4,)
                or not np.all(np.isfinite(col))
                or not np.all(np.isfinite(row))
                or np.any(col <= 0)
                or np.any(row <= 0)
                or not math.isfinite(float(col.sum()))
                or not math.isfinite(float(row.sum()))
            ):
                raise ValueError("invalid normalized reference probabilities")
            numerical_certificate = certify_reference(
                np.asarray(columns["masses"], dtype=float),
                np.asarray(rows["masses"], dtype=float),
            )
            certificate_values = (
                numerical_certificate["maximum_probability_discrepancy"],
                numerical_certificate["maximum_log_ratio_discrepancy"],
                *numerical_certificate["probability_margins"],
            )
            if not all(math.isfinite(float(value)) for value in certificate_values):
                raise ValueError("nonfinite reference certificate")
        except ValueError as error_value:
            numerical_certificate = None
            reference_exception_type = type(error_value).__name__
            reference_exception_message = str(error_value)
    else:
        reference_exception_type = "invalid_reference_view"
        reference_exception_message = "one or both exact reference views are invalid"

    target: list[float] | None = None
    reference_class: int | None = None
    recovery: list[int] | None = None
    reference_syndrome_valid = False
    reference_signature: list[int] | None = None
    reference_failure: bool | None = None
    if numerical_certificate is not None:
        col = np.asarray(columns["probabilities"], dtype=float)
        row = np.asarray(rows["probabilities"], dtype=float)
        target_array = (col + row) / 2.0
        target_array /= target_array.sum()
        candidate_class = int(numerical_certificate["selected_class"])
        candidate_recovery = logical_class_recovery(code, syndrome, candidate_class)
        score = score_recovery(code, error, syndrome, candidate_recovery)
        reference_syndrome_valid = bool(score["syndrome_valid"])
        recovery = candidate_recovery.tolist()
        if reference_syndrome_valid:
            target = target_array.tolist()
            reference_class = candidate_class
            reference_signature = score["logical_signature"].tolist()
            reference_failure = bool(score["logical_failure"])
        else:
            reference_exception_type = "invalid_reference_recovery"
            reference_exception_message = "certified class produced a syndrome-invalid recovery"
    reference_certified = numerical_certificate is not None and reference_syndrome_valid
    reference = {
        "action_invocation_ids": [
            f"reference/exact_{mode}" for mode in reference_order
        ],
        "certified": reference_certified,
        "certificate": numerical_certificate,
        "exception_type": reference_exception_type,
        "exception_message": reference_exception_message,
        "probabilities": target,
        "selected_class": reference_class,
        "recovery_bsf": recovery,
        "syndrome_valid": reference_syndrome_valid,
        "logical_signature": reference_signature,
        "physical_failure": reference_failure,
        "estimated_arithmetic_flops": _sum_work(list(reference_actions.values())),
    }
    for policy in policies:
        _score_policy(
            code,
            policy,
            error,
            syndrome,
            reference_valid=reference_certified,
            reference_class=reference_class,
            reference_failure=reference_failure,
        )
    action_times = [entry for entry in event_timing if "invocation_id" in entry]
    policy_times = [entry for entry in event_timing if "policy_id" in entry]
    timing.append(
        {
            "shot_id": _shot_field(shot, "shot_id"),
            "error_rate": rate,
            "shot_index": index,
            "arm_order": list(order),
            "reference_order": list(reference_order),
            "actions": action_times,
            "policies": policy_times,
            "shot_wall_seconds": perf_counter() - shot_started,
        }
    )
    return {
        "shot_id": _shot_field(shot, "shot_id"),
        "error_rate": rate,
        "shot_index": index,
        "sampler_seed": _shot_field(shot, "sampler_seed"),
        "error_bsf": error.tolist(),
        "syndrome": syndrome.tolist(),
        "arm_order": list(order),
        "reference_order": list(reference_order),
        "actions": actions,
        "policies": policies,
        "reference": reference,
    }
