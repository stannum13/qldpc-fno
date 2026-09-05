"""Fixed sequence-clustered inference and temporal-gate decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from qldpc_fno.metrics.clustered import studentized_sequence_interval

DEPLOYABLE_HISTORY_ARMS = (
    "grid_bayes",
    "ewma",
    "logistic_ar32",
    "parity_moment_ar",
)
DELTA_NLL = 0.00025
HOLM_ALPHA = 0.05
GRID_CONVERGENCE_TOLERANCE = 2.5e-5


def _exact_arm_mapping[T](value: Mapping[str, T], *, name: str) -> Mapping[str, T]:
    if not isinstance(value, Mapping) or set(value) != set(DEPLOYABLE_HISTORY_ARMS):
        raise ValueError(f"{name} must contain the exact four-arm family")
    return value


def fixed_history_derangement[T](
    histories: Sequence[T],
    *,
    seed: int,
    arm: str,
) -> tuple[T, ...]:
    """Replace each target sequence's history with a fixed other history.

    Only histories cross this boundary, so target labels and privileged inputs
    cannot be permuted accidentally.  Arm-specific cyclic offsets ensure the
    four canonical controls remain separate in the 64-sequence campaign.
    """
    if arm not in DEPLOYABLE_HISTORY_ARMS:
        raise ValueError("arm must be an exact syndrome-history arm")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    source = tuple(histories)
    if len(source) < 2:
        raise ValueError("history derangement requires at least two sequences")

    order = np.random.default_rng(seed).permutation(len(source))
    arm_index = DEPLOYABLE_HISTORY_ARMS.index(arm)
    shift = 1 + arm_index % (len(source) - 1)
    source_indices = np.empty(len(source), dtype=np.int64)
    source_indices[order] = np.roll(order, -shift)
    return tuple(source[int(index)] for index in source_indices)


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Adjust the exact four deployable-arm one-sided p-value family."""
    checked = _exact_arm_mapping(pvalues, name="pvalues")
    numeric: dict[str, float] = {}
    for arm in DEPLOYABLE_HISTORY_ARMS:
        value = checked[arm]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError("Holm p-values must be numeric")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
            raise ValueError("Holm p-values must lie in [0, 1]")
        numeric[arm] = numeric_value

    arm_order = {arm: index for index, arm in enumerate(DEPLOYABLE_HISTORY_ARMS)}
    ordered = sorted(numeric, key=lambda arm: (numeric[arm], arm_order[arm]))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(DEPLOYABLE_HISTORY_ARMS)
    for rank, arm in enumerate(ordered):
        running = max(running, (family_size - rank) * numeric[arm])
        adjusted[arm] = min(1.0, running)
    return {arm: adjusted[arm] for arm in DEPLOYABLE_HISTORY_ARMS}


def _inference_status(result: Mapping[str, object], *, label: str) -> str:
    status = result.get("status")
    if not isinstance(status, str):
        raise TypeError(f"{label} inference must contain a string status")
    return status


def _finite_result_number(
    result: Mapping[str, object], *, key: str, label: str
) -> float:
    value = result.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{label} inference must contain numeric {key}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} inference must contain finite {key}")
    return numeric


def decide_temporal_gate(evidence: Mapping[str, object]) -> dict[str, object]:
    """Apply the preregistered control, ceiling, Holm, and effect-size gate."""
    if not isinstance(evidence, Mapping):
        raise TypeError("gate evidence must be a mapping")
    delta = evidence.get("delta_nll")
    if delta != DELTA_NLL:
        raise ValueError(f"delta_nll must equal the fixed value {DELTA_NLL}")
    controls = evidence.get("controls")
    latent = evidence.get("latent")
    arm_value = evidence.get("arms")
    if not isinstance(controls, Mapping):
        raise TypeError("gate evidence must contain controls")
    if not isinstance(latent, Mapping):
        raise TypeError("gate evidence must contain latent inference")
    if not isinstance(arm_value, Mapping):
        raise TypeError("gate evidence must contain arm inference")
    arms = _exact_arm_mapping(arm_value, name="arms")

    invalid: list[str] = []
    leakage_passed = controls.get("leakage_passed")
    convergence_passed = controls.get("convergence_passed")
    if type(leakage_passed) is not bool:
        raise ValueError("leakage_passed must be boolean")
    if type(convergence_passed) is not bool:
        raise ValueError("convergence_passed must be boolean")
    if not leakage_passed:
        invalid.append("leakage")
    if not convergence_passed:
        invalid.append("grid_convergence")

    latent_status = _inference_status(latent, label="latent")
    if latent_status != "ok":
        invalid.append(f"latent:{latent_status}")

    winning: list[str] = []
    for arm in DEPLOYABLE_HISTORY_ARMS:
        row = arms[arm]
        if not isinstance(row, Mapping):
            raise TypeError(f"{arm} evidence must be a mapping")
        gain = row.get("gain")
        if not isinstance(gain, Mapping):
            raise TypeError(f"{arm} evidence must contain gain")
        status = _inference_status(gain, label=f"{arm}:gain")
        if status != "ok":
            invalid.append(f"{arm}:gain:{status}")
            continue
        lower = _finite_result_number(gain, key="lower_95", label=f"{arm}:gain")
        adjusted = row.get("holm_adjusted_pvalue")
        if isinstance(adjusted, (bool, np.bool_)) or not isinstance(
            adjusted, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{arm} must contain a numeric Holm-adjusted p-value")
        adjusted_value = float(adjusted)
        if not math.isfinite(adjusted_value) or not 0.0 <= adjusted_value <= 1.0:
            raise ValueError("Holm-adjusted p-values must lie in [0, 1]")
        if lower > DELTA_NLL and adjusted_value <= HOLM_ALPHA:
            winning.append(arm)

    for arm in DEPLOYABLE_HISTORY_ARMS:
        row = arms[arm]
        assert isinstance(row, Mapping)
        for component in ("stationary_control", "deranged_control"):
            result = row.get(component)
            if not isinstance(result, Mapping):
                raise TypeError(f"{arm} evidence must contain {component}")
            status = _inference_status(result, label=f"{arm}:{component}")
            if status != "ok":
                if arm in winning:
                    invalid.append(f"{arm}:{component}:{status}")
                continue
            upper = _finite_result_number(result, key="upper_95", label=f"{arm}:{component}")
            if arm in winning and upper > DELTA_NLL:
                invalid.append(f"{arm}:{component}")

    if invalid:
        return {
            "invalid_controls": tuple(invalid),
            "limitation": None,
            "outcome": "STOP-INVALID-CONTROL",
            "winning_arms": (),
        }

    latent_lower = _finite_result_number(latent, key="lower_95", label="latent")
    if latent_lower <= DELTA_NLL:
        return {
            "invalid_controls": (),
            "limitation": None,
            "outcome": "STOP-NO-PRACTICAL-CAUSAL-HEADROOM",
            "winning_arms": (),
        }

    if not winning:
        return {
            "invalid_controls": (),
            "limitation": None,
            "outcome": "INCONCLUSIVE-OBSERVER-GAP",
            "winning_arms": (),
        }

    if "ewma" in winning or "logistic_ar32" in winning:
        limitation = "REDUCED-SCREEN-OR-SAMPLE-LIMITATION"
    elif "grid_bayes" in winning:
        limitation = "CURRENT-BASELINE-LIMITATION"
    else:
        limitation = None
    return {
        "invalid_controls": (),
        "limitation": limitation,
        "outcome": "GO-TEMPORAL-IDENTIFIED",
        "winning_arms": tuple(winning),
    }


def _confirmatory_values(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must contain one value per independent sequence")
    if array.size < 64:
        raise ValueError(f"{name} must contain at least 64 independent sequences")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and complete")
    return array


def evaluate_identifiability(
    *,
    latent_gain: np.ndarray,
    deployable_gains: Mapping[str, np.ndarray],
    stationary_gains: Mapping[str, np.ndarray],
    deranged_gains: Mapping[str, np.ndarray],
    doubled_grid_gain: np.ndarray | None,
    bootstrap_seed: int,
    leakage_passed: bool,
    grid_convergence_difference: float | None = None,
) -> dict[str, object]:
    """Compute all fixed inference evidence and its actual temporal decision."""
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if type(leakage_passed) is not bool:
        raise ValueError("leakage_passed must be boolean")
    deployable = _exact_arm_mapping(deployable_gains, name="deployable_gains")
    stationary = _exact_arm_mapping(stationary_gains, name="stationary_gains")
    deranged = _exact_arm_mapping(deranged_gains, name="deranged_gains")
    checked_latent = _confirmatory_values(latent_gain, name="latent_gain")
    checked_doubled = (
        None
        if doubled_grid_gain is None
        else _confirmatory_values(doubled_grid_gain, name="doubled_grid_gain")
    )
    checked_deployable = {
        arm: _confirmatory_values(
            deployable[arm], name=f"deployable_gains[{arm}]"
        )
        for arm in DEPLOYABLE_HISTORY_ARMS
    }
    checked_stationary = {
        arm: _confirmatory_values(
            stationary[arm], name=f"stationary_gains[{arm}]"
        )
        for arm in DEPLOYABLE_HISTORY_ARMS
    }
    checked_deranged = {
        arm: _confirmatory_values(
            deranged[arm], name=f"deranged_gains[{arm}]"
        )
        for arm in DEPLOYABLE_HISTORY_ARMS
    }
    sequence_counts = {
        checked_latent.size,
        *(values.size for values in checked_deployable.values()),
        *(values.size for values in checked_stationary.values()),
        *(values.size for values in checked_deranged.values()),
    }
    if checked_doubled is not None:
        sequence_counts.add(checked_doubled.size)
    if len(sequence_counts) != 1:
        raise ValueError("all inference inputs must have identical paired sequence counts")

    def infer(values: np.ndarray) -> dict[str, object]:
        return studentized_sequence_interval(
            values,
            seed=bootstrap_seed,
            null_mean=DELTA_NLL,
        )

    latent_result = infer(checked_latent)
    arm_results: dict[str, dict[str, object]] = {}
    raw_pvalues: dict[str, float] = {}
    for arm in DEPLOYABLE_HISTORY_ARMS:
        gain = infer(checked_deployable[arm])
        stationary_control = infer(checked_stationary[arm])
        deranged_control = infer(checked_deranged[arm])
        arm_results[arm] = {
            "deranged_control": deranged_control,
            "gain": gain,
            "stationary_control": stationary_control,
        }
        if gain["status"] == "ok":
            raw = gain["pvalue_greater"]
            assert isinstance(raw, float)
            raw_pvalues[arm] = raw

    if len(raw_pvalues) == len(DEPLOYABLE_HISTORY_ARMS):
        adjusted: dict[str, float | None] = holm_adjust(raw_pvalues)
    else:
        adjusted = {arm: None for arm in DEPLOYABLE_HISTORY_ARMS}
    for arm in DEPLOYABLE_HISTORY_ARMS:
        arm_results[arm]["holm_adjusted_pvalue"] = adjusted[arm]

    nominal_grid = checked_deployable["grid_bayes"]
    if grid_convergence_difference is None:
        if checked_doubled is None:
            raise ValueError(
                "doubled_grid_gain is required unless validation convergence is supplied"
            )
        convergence_difference = abs(float(nominal_grid.mean() - checked_doubled.mean()))
    else:
        if isinstance(grid_convergence_difference, (bool, np.bool_)) or not isinstance(
            grid_convergence_difference, (int, float, np.integer, np.floating)
        ):
            raise TypeError("grid_convergence_difference must be numeric")
        convergence_difference = float(grid_convergence_difference)
        if not math.isfinite(convergence_difference) or convergence_difference < 0.0:
            raise ValueError("grid_convergence_difference must be finite and non-negative")
    controls = {
        "convergence_difference": convergence_difference,
        "convergence_passed": convergence_difference < GRID_CONVERGENCE_TOLERANCE,
        "leakage_passed": leakage_passed,
    }
    evidence: dict[str, object] = {
        "arms": arm_results,
        "controls": controls,
        "delta_nll": DELTA_NLL,
        "holm_adjusted_pvalues": adjusted,
        "latent": latent_result,
    }
    evidence["decision"] = decide_temporal_gate(evidence)
    return evidence


def conditional_decoder_arms(
    decision: Mapping[str, object],
) -> str | tuple[str, ...]:
    """Return the exact decoder policy implied by a completed primary gate."""
    if not isinstance(decision, Mapping):
        raise TypeError("decision must be a mapping")
    if decision.get("outcome") != "GO-TEMPORAL-IDENTIFIED":
        return "not_run_by_design"
    winning = decision.get("winning_arms")
    if not isinstance(winning, tuple):
        raise TypeError("GO decisions must contain an immutable winning-arm tuple")
    if (
        not winning
        or len(set(winning)) != len(winning)
        or any(arm not in DEPLOYABLE_HISTORY_ARMS for arm in winning)
    ):
        raise ValueError("GO decisions must contain exact syndrome-history winning arms")
    ordered_winners = tuple(arm for arm in DEPLOYABLE_HISTORY_ARMS if arm in winning)
    return ("known_marginal", *ordered_winners, "contemporaneous_oracle")


def classify_bler_interval(interval_low: float, interval_high: float) -> str:
    """Classify a 95% paired BLER-difference interval at the fixed 0.01 margin."""
    for value, label in ((interval_low, "interval_low"), (interval_high, "interval_high")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{label} must be numeric")
    low = float(interval_low)
    high = float(interval_high)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("BLER interval endpoints must be finite")
    if low > high:
        raise ValueError("BLER interval endpoints must be ordered")
    if high < -0.01:
        return "BENEFIT"
    if low > 0.01:
        return "HARM"
    if low >= -0.01 and high <= 0.01:
        return "EQUIVALENT"
    return "INCONCLUSIVE"
