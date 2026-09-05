from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from qldpc_fno.identifiability.inference import (
    DEPLOYABLE_HISTORY_ARMS,
    classify_bler_interval,
    conditional_decoder_arms,
    decide_temporal_gate,
    evaluate_identifiability,
    fixed_history_derangement,
    holm_adjust,
)

DELTA_NLL = 0.00025
ARMS = ("grid_bayes", "ewma", "logistic_ar32", "parity_moment_ar")


def test_fixed_history_derangement_is_replayed_no_fixed_point_permutation() -> None:
    histories = tuple(f"history-{index}" for index in range(64))

    first = fixed_history_derangement(histories, seed=19, arm="grid_bayes")
    second = fixed_history_derangement(histories, seed=19, arm="grid_bayes")

    assert first == second
    assert sorted(first) == sorted(histories)
    assert all(replacement != original for replacement, original in zip(first, histories))


def test_derangement_replaces_only_histories_and_has_one_control_per_arm() -> None:
    histories = tuple(f"history-{index}" for index in range(64))
    targets = tuple(np.array([index], dtype=np.int64) for index in range(64))
    target_copies = tuple(target.copy() for target in targets)

    controls = {
        arm: fixed_history_derangement(histories, seed=20260905, arm=arm)
        for arm in ARMS
    }

    assert DEPLOYABLE_HISTORY_ARMS == ARMS
    assert len(set(controls.values())) == len(ARMS)
    for control in controls.values():
        assert sorted(control) == sorted(histories)
        assert all(replacement != original for replacement, original in zip(control, histories))
    assert all(
        np.array_equal(target, original)
        for target, original in zip(targets, target_copies)
    )


@pytest.mark.parametrize("arm", ["known_marginal", "latent_history_oracle", "typo"])
def test_derangement_rejects_non_syndrome_history_arms(arm: str) -> None:
    with pytest.raises(ValueError, match="exact syndrome-history arm"):
        fixed_history_derangement(tuple(range(64)), seed=1, arm=arm)


def test_holm_adjust_uses_exact_four_arm_family_with_ordering() -> None:
    adjusted = holm_adjust(
        {
            "grid_bayes": 0.01,
            "ewma": 0.04,
            "logistic_ar32": 0.03,
            "parity_moment_ar": 0.20,
        }
    )

    assert adjusted == pytest.approx(
        {
            "grid_bayes": 0.04,
            "ewma": 0.09,
            "logistic_ar32": 0.09,
            "parity_moment_ar": 0.20,
        }
    )


def test_holm_adjust_gives_ties_the_same_adjusted_value() -> None:
    adjusted = holm_adjust(dict.fromkeys(ARMS, 0.01))

    assert adjusted == pytest.approx(dict.fromkeys(ARMS, 0.04))


def test_holm_adjust_rejects_any_family_other_than_the_exact_four_arms() -> None:
    with pytest.raises(ValueError, match="exact four-arm family"):
        holm_adjust({"grid_bayes": 0.01})


def _interval(
    *,
    lower: float = 0.001,
    upper: float = 0.0015,
    pvalue: float = 0.01,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "lower_95": lower if status == "ok" else None,
        "upper_95": upper if status == "ok" else None,
        "pvalue_greater": pvalue if status == "ok" else None,
        "status": status,
    }


def _evidence(
    *,
    latent: Mapping[str, object] | None = None,
    candidate_arms: tuple[str, ...] = ("grid_bayes",),
    control_failure: tuple[str, str] | None = None,
    degenerate: tuple[str, str] | None = None,
    leakage_passed: bool = True,
    convergence_passed: bool = True,
) -> dict[str, object]:
    arms: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        candidate = arm in candidate_arms
        arm_result = {
            "gain": _interval(
                lower=0.001 if candidate else 0.0001,
                pvalue=0.01 if candidate else 0.4,
            ),
            "holm_adjusted_pvalue": 0.04 if candidate else 0.8,
            "stationary_control": _interval(lower=-0.0001, upper=DELTA_NLL),
            "deranged_control": _interval(lower=-0.0001, upper=DELTA_NLL),
        }
        arms[arm] = arm_result
    if control_failure is not None:
        arm, control = control_failure
        arms[arm][control] = _interval(lower=0.0, upper=DELTA_NLL + 1e-12)
    if degenerate is not None:
        arm, component = degenerate
        arms[arm][component] = _interval(status="bootstrap_degenerate")
    return {
        "arms": arms,
        "controls": {
            "convergence_passed": convergence_passed,
            "leakage_passed": leakage_passed,
        },
        "delta_nll": DELTA_NLL,
        "latent": latent or _interval(),
    }


@pytest.mark.parametrize(
    ("evidence", "outcome", "winning", "limitation"),
    [
        (
            _evidence(latent=_interval(lower=DELTA_NLL)),
            "STOP-NO-PRACTICAL-CAUSAL-HEADROOM",
            (),
            None,
        ),
        (_evidence(candidate_arms=()), "INCONCLUSIVE-OBSERVER-GAP", (), None),
        (
            _evidence(candidate_arms=("grid_bayes",)),
            "GO-TEMPORAL-IDENTIFIED",
            ("grid_bayes",),
            "CURRENT-BASELINE-LIMITATION",
        ),
        (
            _evidence(candidate_arms=("ewma",)),
            "GO-TEMPORAL-IDENTIFIED",
            ("ewma",),
            "REDUCED-SCREEN-OR-SAMPLE-LIMITATION",
        ),
        (
            _evidence(candidate_arms=("logistic_ar32",)),
            "GO-TEMPORAL-IDENTIFIED",
            ("logistic_ar32",),
            "REDUCED-SCREEN-OR-SAMPLE-LIMITATION",
        ),
        (
            _evidence(candidate_arms=("parity_moment_ar",)),
            "GO-TEMPORAL-IDENTIFIED",
            ("parity_moment_ar",),
            None,
        ),
    ],
)
def test_temporal_gate_outcome_table(
    evidence: dict[str, object],
    outcome: str,
    winning: tuple[str, ...],
    limitation: str | None,
) -> None:
    decision = decide_temporal_gate(evidence)

    assert decision["outcome"] == outcome
    assert decision["winning_arms"] == winning
    assert decision["limitation"] == limitation


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (_evidence(leakage_passed=False), "leakage"),
        (_evidence(convergence_passed=False), "grid_convergence"),
        (
            _evidence(control_failure=("grid_bayes", "stationary_control")),
            "grid_bayes:stationary_control",
        ),
        (
            _evidence(control_failure=("grid_bayes", "deranged_control")),
            "grid_bayes:deranged_control",
        ),
        (
            _evidence(degenerate=("grid_bayes", "gain")),
            "grid_bayes:gain:bootstrap_degenerate",
        ),
        (
            _evidence(degenerate=("grid_bayes", "deranged_control")),
            "grid_bayes:deranged_control:bootstrap_degenerate",
        ),
    ],
)
def test_any_failed_control_or_degenerate_inference_stops_gate(
    evidence: dict[str, object], reason: str
) -> None:
    decision = decide_temporal_gate(evidence)

    assert decision["outcome"] == "STOP-INVALID-CONTROL"
    assert reason in decision["invalid_controls"]
    assert decision["winning_arms"] == ()


def test_arm_requires_effect_bound_and_holm_adjusted_pvalue() -> None:
    evidence = _evidence(candidate_arms=("grid_bayes", "ewma"))
    arms = evidence["arms"]
    assert isinstance(arms, dict)
    arms["grid_bayes"]["holm_adjusted_pvalue"] = 0.0500000001
    arms["ewma"]["gain"] = _interval(lower=DELTA_NLL)

    decision = decide_temporal_gate(evidence)

    assert decision["outcome"] == "INCONCLUSIVE-OBSERVER-GAP"
    assert decision["winning_arms"] == ()


def test_failed_control_for_a_nonwinning_arm_does_not_invalidate_a_valid_winner() -> None:
    evidence = _evidence(
        candidate_arms=("grid_bayes",),
        control_failure=("ewma", "deranged_control"),
    )

    decision = decide_temporal_gate(evidence)

    assert decision["outcome"] == "GO-TEMPORAL-IDENTIFIED"
    assert decision["winning_arms"] == ("grid_bayes",)


def _gain(mean: float, amplitude: float = 0.00012) -> np.ndarray:
    centered = np.linspace(-amplitude, amplitude, 64)
    return mean + centered


def test_evaluate_identifiability_applies_four_arm_holm_in_actual_go_path() -> None:
    deployable = {
        "grid_bayes": _gain(0.0010),
        "ewma": _gain(0.0001),
        "logistic_ar32": _gain(0.0001),
        "parity_moment_ar": _gain(0.0001),
    }
    controls = {arm: _gain(0.0, 0.00005) for arm in ARMS}

    result = evaluate_identifiability(
        latent_gain=_gain(0.0012),
        deployable_gains=deployable,
        stationary_gains=controls,
        deranged_gains=controls,
        doubled_grid_gain=deployable["grid_bayes"] + 0.00001,
        bootstrap_seed=823,
        leakage_passed=True,
    )

    raw = {
        arm: result["arms"][arm]["gain"]["pvalue_greater"] for arm in ARMS
    }
    reported_seeds = {result["latent"]["seed"]}
    for arm in ARMS:
        reported_seeds.update(
            result["arms"][arm][component]["seed"]
            for component in ("gain", "stationary_control", "deranged_control")
        )
    assert reported_seeds == {823}
    assert result["holm_adjusted_pvalues"] == pytest.approx(holm_adjust(raw))
    assert result["decision"]["outcome"] == "GO-TEMPORAL-IDENTIFIED"
    assert result["decision"]["winning_arms"] == ("grid_bayes",)
    assert result["controls"]["convergence_difference"] == pytest.approx(0.00001)
    assert result["controls"]["convergence_passed"] is True


def test_evaluate_identifiability_stops_when_one_gain_bootstrap_is_degenerate() -> None:
    deployable = {
        "grid_bayes": np.full(64, 0.001),
        "ewma": _gain(0.0010),
        "logistic_ar32": _gain(0.0001),
        "parity_moment_ar": _gain(0.0001),
    }
    controls = {arm: _gain(0.0, 0.00005) for arm in ARMS}

    result = evaluate_identifiability(
        latent_gain=_gain(0.0012),
        deployable_gains=deployable,
        stationary_gains=controls,
        deranged_gains=controls,
        doubled_grid_gain=deployable["grid_bayes"] + 0.00001,
        bootstrap_seed=823,
        leakage_passed=True,
    )

    assert result["holm_adjusted_pvalues"] == dict.fromkeys(ARMS)
    assert result["decision"]["outcome"] == "STOP-INVALID-CONTROL"
    assert result["decision"]["invalid_controls"] == (
        "grid_bayes:gain:degenerate_variance",
    )
    assert result["decision"]["winning_arms"] == ()


def test_validation_grid_difference_governs_inference_not_test_resolution_noise() -> None:
    deployable = {
        "grid_bayes": _gain(0.0010),
        "ewma": _gain(0.0001),
        "logistic_ar32": _gain(0.0001),
        "parity_moment_ar": _gain(0.0001),
    }
    controls = {arm: _gain(0.0, 0.00005) for arm in ARMS}

    result = evaluate_identifiability(
        latent_gain=_gain(0.0012),
        deployable_gains=deployable,
        stationary_gains=controls,
        deranged_gains=controls,
        doubled_grid_gain=deployable["grid_bayes"] + 0.001,
        grid_convergence_difference=0.00001,
        bootstrap_seed=823,
        leakage_passed=True,
    )

    assert result["controls"]["convergence_difference"] == pytest.approx(0.00001)
    assert result["controls"]["convergence_passed"] is True
    assert result["decision"]["outcome"] == "GO-TEMPORAL-IDENTIFIED"


def test_evaluate_identifiability_rejects_unpaired_sequence_counts() -> None:
    deployable = {arm: _gain(0.0010) for arm in ARMS}
    controls = {arm: _gain(0.0) for arm in ARMS}

    with pytest.raises(ValueError, match="identical paired sequence counts"):
        evaluate_identifiability(
            latent_gain=np.append(_gain(0.0012), 0.0012),
            deployable_gains=deployable,
            stationary_gains=controls,
            deranged_gains=controls,
            doubled_grid_gain=deployable["grid_bayes"] + 0.00001,
            bootstrap_seed=823,
            leakage_passed=True,
        )


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (-0.02, -0.0100001, "BENEFIT"),
        (-0.02, -0.01, "INCONCLUSIVE"),
        (-0.01, -0.01, "EQUIVALENT"),
        (-0.01, 0.01, "EQUIVALENT"),
        (0.01, 0.01, "EQUIVALENT"),
        (0.01, 0.02, "INCONCLUSIVE"),
        (0.0100001, 0.02, "HARM"),
        (-0.02, 0.02, "INCONCLUSIVE"),
    ],
)
def test_bler_interval_classification_has_exact_margin_boundaries(
    low: float, high: float, expected: str
) -> None:
    assert classify_bler_interval(low, high) == expected


def test_conditional_decoder_arm_policy_is_pure_and_exact() -> None:
    non_go = decide_temporal_gate(_evidence(candidate_arms=()))
    go = decide_temporal_gate(_evidence(candidate_arms=("ewma", "parity_moment_ar")))
    snapshot = dict(go)

    assert conditional_decoder_arms(non_go) == "not_run_by_design"
    assert conditional_decoder_arms(go) == (
        "known_marginal",
        "ewma",
        "parity_moment_ar",
        "contemporaneous_oracle",
    )
    assert go == snapshot
