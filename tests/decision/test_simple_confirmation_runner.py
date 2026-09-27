from __future__ import annotations

import math
from collections.abc import Callable
from types import MappingProxyType

import numpy as np
import pytest
from qecsim import paulitools as pt
from qecsim.models.planar import PlanarCode

from qldpc_fno.decision import simple_confirmation as confirmation
from qldpc_fno.decision import simple_confirmation_runner as runner
from qldpc_fno.decision.simple_confirmation import MARGIN_THRESHOLD, arm_order
from qldpc_fno.decision.tensor_network import (
    InvalidContractionError,
    InvalidCosetMassError,
    PlanarCosetMasses,
)


def _work(
    value: int = 10,
    *,
    attempts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    attempts = [] if attempts is None else attempts
    dense = sum(int(attempt["estimated_flops"]) for attempt in attempts)
    svd = sum(attempt["decomposition_kind"] == "svd" for attempt in attempts)
    qr = sum(attempt["decomposition_kind"] == "qr" for attempt in attempts)
    return {
        "pairwise_contractions": 0,
        "truncation_calls": 0,
        "svd_calls": svd,
        "qr_calls": qr,
        "einsum_calls": 1,
        "einsum_estimated_flops": value,
        "pairwise_output_elements": 0,
        "decomposition_input_elements": sum(
            int(attempt["matrix_rows"]) * int(attempt["matrix_columns"])
            for attempt in attempts
        ),
        "estimated_dense_decomposition_flops": dense,
        "estimated_arithmetic_flops": value + dense,
        "decomposition_attempts": attempts,
        "peak_observed_array_elements": 0,
        "truncation_events": [],
        "contraction_sweeps": [],
        "single_shot_wall_seconds": 999.0,
    }


def _attempt(kind: str, m: int, n: int, *, succeeded: bool) -> dict[str, object]:
    q = min(m, n)
    estimated = (
        int(4 * m * n * q + 8 * q**3)
        if kind == "svd"
        else int(2 * m * n * q - 2 * q**3 / 3)
    )
    return {
        "decomposition_kind": kind,
        "matrix_rows": m,
        "matrix_columns": n,
        "estimated_flops": estimated,
        "succeeded": succeeded,
        "exception_type": None if succeeded else "LinAlgError",
    }


def _result(
    masses: tuple[float, float, float, float],
    *,
    mode: str,
    chi: int | None,
    tol: float | None,
    work: dict[str, object] | None = None,
) -> PlanarCosetMasses:
    values = np.asarray(masses, dtype=float)
    return PlanarCosetMasses(
        masses=values,
        selected_class=int(np.argmax(values)),
        rows=5,
        columns=5,
        error_rate=0.1,
        chi=chi,
        tol=tol,
        mode=mode,
        work=_work() if work is None else work,
    )


def _shot(*, error: np.ndarray | None = None, index: int = 0) -> dict[str, object]:
    code = PlanarCode(5, 5)
    error = np.zeros(2 * code.n_k_d[0], dtype=np.uint8) if error is None else error
    syndrome = np.asarray(pt.bsp(error, code.stabilizers.T), dtype=np.uint8)
    return {
        "shot_id": f"fixture/p0.100000/i{index:06d}",
        "error_rate": 0.1,
        "shot_index": index,
        "sampler_seed": 100 + index,
        "error_bsf": error.tolist(),
        "syndrome": syndrome.tolist(),
    }


def _fake_contractor(
    chooser: Callable[[str, int | None, float | None, int], object]
) -> tuple[Callable[..., PlanarCosetMasses], list[tuple[str, int | None, float | None]]]:
    calls: list[tuple[str, int | None, float | None]] = []

    def contract(**kwargs):
        mode, chi, tol = kwargs["mode"], kwargs["chi"], kwargs["tol"]
        calls.append((mode, chi, tol))
        outcome = chooser(mode, chi, tol, len(calls) - 1)
        if isinstance(outcome, BaseException):
            raise outcome
        masses = outcome
        return _result(masses, mode=mode, chi=chi, tol=tol)

    return contract, calls


def _default_chooser(mode: str, chi: int | None, tol: float | None, call: int):
    del mode, chi, tol, call
    return (0.8, 0.1, 0.06, 0.04)


def test_arm_order_cycles_declared_index_permutations() -> None:
    expected = (
        ("fixed_rows_tol003", "margin_columns_chi8", "fixed_columns_chi8"),
        ("fixed_rows_tol003", "fixed_columns_chi8", "margin_columns_chi8"),
        ("margin_columns_chi8", "fixed_rows_tol003", "fixed_columns_chi8"),
        ("margin_columns_chi8", "fixed_columns_chi8", "fixed_rows_tol003"),
        ("fixed_columns_chi8", "fixed_rows_tol003", "margin_columns_chi8"),
        ("fixed_columns_chi8", "margin_columns_chi8", "fixed_rows_tol003"),
    )
    assert tuple(arm_order(index) for index in range(6)) == expected
    counts = {order: 0 for order in expected}
    for index in range(2048):
        counts[arm_order(index)] += 1
    assert set(counts.values()) == {341, 342}
    for invalid in (-1, True, 1.0):
        with pytest.raises(ValueError, match="shot index"):
            arm_order(invalid)  # type: ignore[arg-type]


def test_valid_strict_probe_agreement_uses_exactly_six_independent_contractions(
    monkeypatch,
) -> None:
    contract, calls = _fake_contractor(_default_chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    timing: list[dict] = []

    row = runner.evaluate_shot(_shot(), timing=timing)

    assert len(calls) == 6
    margin = next(policy for policy in row["policies"] if policy["policy_id"] == "margin_columns_chi8")
    assert margin["gate_accepted"] is True
    assert margin["action_invocation_ids"] == [
        "margin_columns_chi8/columns_tol01",
        "margin_columns_chi8/rows_tol01",
    ]
    assert len(row["actions"]) == 6
    assert len(timing) == 1
    assert len(timing[0]["actions"]) == 6
    assert len(timing[0]["policies"]) == 3


def test_margin_at_threshold_escalates_without_reference_input(monkeypatch) -> None:
    at_threshold = (
        (1.0 + MARGIN_THRESHOLD) / 2.0,
        (1.0 - MARGIN_THRESHOLD) / 2.0,
        0.0,
        0.0,
    )

    def chooser(mode, chi, tol, call):
        del mode, call
        return at_threshold if tol == 0.01 and chi is None else (0.8, 0.1, 0.06, 0.04)

    contract, calls = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)

    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(policy for policy in row["policies"] if policy["policy_id"] == "margin_columns_chi8")
    assert margin["gate_accepted"] is False
    assert len(margin["action_invocation_ids"]) == 3
    assert len(calls) == 7


def test_public_threshold_rebinding_cannot_change_scientific_routing(monkeypatch) -> None:
    below_frozen_threshold = (0.55, 0.3, 0.1, 0.05)

    def chooser(mode, chi, tol, call):
        del mode, chi, call
        return below_frozen_threshold if tol == 0.01 else (0.8, 0.1, 0.06, 0.04)

    contract, calls = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    monkeypatch.setattr(confirmation, "MARGIN_THRESHOLD", 0.0)
    monkeypatch.setattr(runner, "MARGIN_THRESHOLD", 0.0)

    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(
        policy for policy in row["policies"]
        if policy["policy_id"] == "margin_columns_chi8"
    )
    assert margin["gate_accepted"] is False
    assert len(margin["action_invocation_ids"]) == 3
    assert len(calls) == 7


def test_frozen_action_and_policy_tables_reject_reachable_mutation() -> None:
    assert isinstance(runner._ACTION_SPECS, MappingProxyType)
    assert isinstance(runner._POLICY_ACTIONS, MappingProxyType)
    with pytest.raises(TypeError):
        runner._ACTION_SPECS["rows_tol003"] = ("columns", 8, None)
    with pytest.raises(TypeError):
        runner._POLICY_ACTIONS["fixed_rows_tol003"] = ("columns_chi8",)


def test_probe_disagreement_runs_fallback_separately_from_fixed_column(monkeypatch) -> None:
    chi8_results: list[PlanarCosetMasses] = []

    def contract(**kwargs):
        if kwargs["tol"] == 0.01:
            masses = (0.8, 0.1, 0.06, 0.04) if kwargs["mode"] == "columns" else (0.1, 0.8, 0.06, 0.04)
        else:
            masses = (0.8, 0.1, 0.06, 0.04)
        result = _result(masses, mode=kwargs["mode"], chi=kwargs["chi"], tol=kwargs["tol"])
        if kwargs["chi"] == 8:
            chi8_results.append(result)
        return result

    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    row = runner.evaluate_shot(_shot(), timing=[])

    assert len(chi8_results) == 2
    assert chi8_results[0] is not chi8_results[1]
    chi8_actions = [action for action in row["actions"] if action["action_id"] == "columns_chi8"]
    assert [action["invocation_id"] for action in chi8_actions] == [
        "margin_columns_chi8/columns_chi8",
        "fixed_columns_chi8/columns_chi8",
    ]


@pytest.mark.parametrize("invalid_probe_count", [1, 2])
def test_invalid_probes_are_all_attempted_and_valid_fallback_is_scored(
    monkeypatch, invalid_probe_count: int
) -> None:
    invalid_seen = 0

    def chooser(mode, chi, tol, call):
        nonlocal invalid_seen
        del mode, call
        if tol == 0.01 and invalid_seen < invalid_probe_count:
            invalid_seen += 1
            return InvalidCosetMassError(np.array([np.nan] * 4), _work(13))
        return (0.8, 0.1, 0.06, 0.04)

    contract, calls = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(policy for policy in row["policies"] if policy["policy_id"] == "margin_columns_chi8")
    assert len(calls) == 7
    assert margin["gate_accepted"] is False
    assert margin["valid_recovery"] is True
    assert margin["physical_failure"] is False
    assert margin["estimated_arithmetic_flops"] == invalid_probe_count * 13 + (3 - invalid_probe_count) * 10


def test_second_probe_only_invalid_is_charged_before_valid_fallback(monkeypatch) -> None:
    probe_count = 0

    def chooser(mode, chi, tol, call):
        nonlocal probe_count
        del mode, call
        if tol == 0.01:
            probe_count += 1
            if probe_count == 2:
                return InvalidCosetMassError(np.array([np.nan] * 4), _work(19))
        return (0.8, 0.1, 0.06, 0.04)

    contract, calls = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)

    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(
        policy for policy in row["policies"]
        if policy["policy_id"] == "margin_columns_chi8"
    )
    assert probe_count == 2
    assert len(calls) == 7
    assert margin["gate_accepted"] is False
    assert margin["valid_recovery"] is True
    assert margin["estimated_arithmetic_flops"] == 39


def test_invalid_fallback_forces_both_events_and_retains_partial_work(monkeypatch) -> None:
    def chooser(mode, chi, tol, call):
        del call
        if chi == 8 and mode == "columns":
            return InvalidContractionError(np.array([np.nan] * 4), _work(17))
        if tol == 0.01:
            return (0.8, 0.1, 0.06, 0.04) if mode == "columns" else (0.1, 0.8, 0.06, 0.04)
        return (0.8, 0.1, 0.06, 0.04)

    contract, _ = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(policy for policy in row["policies"] if policy["policy_id"] == "margin_columns_chi8")
    assert margin["valid_recovery"] is False
    assert margin["physical_failure"] is True
    assert margin["class_mismatch"] is True
    assert margin["outcome_discordance"] is True
    assert margin["estimated_arithmetic_flops"] == 37
    failed = next(
        action for action in row["actions"]
        if action["invocation_id"] == "margin_columns_chi8/columns_chi8"
    )
    assert failed["valid"] is False
    assert failed["masses"] is None
    assert failed["estimated_arithmetic_flops"] == 17


@pytest.mark.parametrize(
    "columns,rows",
    [
        ((math.nan, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0)),
        ((0.8, 0.1, 0.06, 0.04), (0.1, 0.8, 0.06, 0.04)),
        ((0.8, 0.1, 0.06, 0.04), (0.79, 0.11, 0.06, 0.04)),
        ((1.0, 1.0, 0.1, 0.1), (1.0, 1.0, 0.1, 0.1)),
        ((1e-320, 1e-320, 1e-320, 1e-320), (1e-320, 1e-320, 1e-320, 1e-320)),
    ],
)
def test_uncertified_reference_forces_both_events_for_every_policy(
    monkeypatch, columns, rows
) -> None:
    exact_count = 0

    def chooser(mode, chi, tol, call):
        nonlocal exact_count
        del call
        if chi is None and tol is None:
            exact_count += 1
            return columns if mode == "columns" else rows
        return (0.8, 0.1, 0.06, 0.04)

    contract, _ = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)

    row = runner.evaluate_shot(_shot(), timing=[])

    assert exact_count == 2
    assert row["reference"]["certified"] is False
    assert row["reference"]["selected_class"] is None
    assert row["reference"]["physical_failure"] is None
    for policy in row["policies"]:
        assert policy["class_mismatch"] is True
        assert policy["outcome_discordance"] is True


def test_accepted_agreeing_wrong_probes_are_detected(monkeypatch) -> None:
    def chooser(mode, chi, tol, call):
        del mode, chi, call
        return (0.05, 0.85, 0.06, 0.04) if tol == 0.01 else (0.85, 0.05, 0.06, 0.04)

    contract, _ = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    row = runner.evaluate_shot(_shot(), timing=[])

    margin = next(policy for policy in row["policies"] if policy["policy_id"] == "margin_columns_chi8")
    assert margin["gate_accepted"] is True
    assert margin["class_mismatch"] is True
    assert margin["physical_failure"] is True
    assert margin["outcome_discordance"] is True


def test_stabilizer_equivalent_error_is_a_physical_success(monkeypatch) -> None:
    code = PlanarCode(5, 5)
    error = np.asarray(code.stabilizers[0], dtype=np.uint8)
    contract, _ = _fake_contractor(_default_chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)

    row = runner.evaluate_shot(_shot(error=error), timing=[])

    assert all(policy["physical_failure"] is False for policy in row["policies"])
    assert row["reference"]["physical_failure"] is False


def test_distinct_wrong_logical_signatures_can_both_be_physical_failures(monkeypatch) -> None:
    def chooser(mode, chi, tol, call):
        del mode, call
        if tol == 0.003:
            return (0.05, 0.85, 0.06, 0.04)
        if tol == 0.01:
            return (0.05, 0.06, 0.85, 0.04)
        return (0.85, 0.05, 0.06, 0.04)

    contract, _ = _fake_contractor(chooser)
    monkeypatch.setattr(runner, "planar_mps_coset_masses", contract)
    row = runner.evaluate_shot(_shot(), timing=[])
    policies = {policy["policy_id"]: policy for policy in row["policies"]}

    left = policies["fixed_rows_tol003"]
    right = policies["margin_columns_chi8"]
    assert left["physical_failure"] is right["physical_failure"] is True
    assert left["class_mismatch"] is right["class_mismatch"] is True
    assert left["logical_signature"] != right["logical_signature"]


def test_failed_svd_retry_is_fully_charged_and_wall_time_is_removed(monkeypatch) -> None:
    attempts = [
        _attempt("svd", 4, 3, succeeded=False),
        _attempt("svd", 4, 3, succeeded=True),
        _attempt("qr", 4, 3, succeeded=True),
    ]
    work = _work(11, attempts=attempts)
    monkeypatch.setattr(
        runner,
        "planar_mps_coset_masses",
        lambda **kwargs: _result(
            (0.8, 0.1, 0.06, 0.04),
            mode=kwargs["mode"],
            chi=kwargs["chi"],
            tol=kwargs["tol"],
            work=work,
        ),
    )
    timing: list[dict] = []

    action = runner.run_action(
        "fixed_rows_tol003/rows_tol003",
        "rows_tol003",
        np.zeros(40, dtype=np.uint8),
        0.1,
        timing,
    )

    assert action["estimated_arithmetic_flops"] == 11 + sum(
        int(attempt["estimated_flops"]) for attempt in attempts
    )
    assert action["work"]["svd_calls"] == 2
    assert action["work"]["qr_calls"] == 1
    assert "single_shot_wall_seconds" not in action["work"]
    assert timing[0]["wall_seconds"] >= 0


@pytest.mark.parametrize("error", [TypeError("bug"), KeyError("bug")])
def test_unexpected_programming_errors_propagate(monkeypatch, error: Exception) -> None:
    monkeypatch.setattr(
        runner,
        "planar_mps_coset_masses",
        lambda **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error)):
        runner.run_action(
            "fixed_rows_tol003/rows_tol003",
            "rows_tol003",
            np.zeros(40, dtype=np.uint8),
            0.1,
            [],
        )
