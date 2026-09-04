from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qldpc_fno.temporal.causality import (
    CausalAuditSequence,
    ObservedHistory,
    audit_forecaster_causality,
)


class StatefulPrefixForecaster:
    def __init__(self) -> None:
        self.calls: list[ObservedHistory] = []

    def forecast(self, history: ObservedHistory) -> np.ndarray:
        self.calls.append(history)
        weights = np.arange(1, history.syndromes.shape[0] + 1, dtype=np.float64)
        return np.tensordot(weights, history.syndromes, axes=(0, 0))


def _sequence() -> CausalAuditSequence:
    syndromes = np.arange(6 * 4, dtype=np.uint8).reshape(6, 4) % 2
    return CausalAuditSequence(
        syndromes=syndromes,
        forecast_round=3,
        physical_errors=np.arange(6 * 7, dtype=np.uint8).reshape(6, 7) % 2,
        logical_outcomes=np.arange(6, dtype=np.uint8) % 2,
        diagnostics={"latent_q": np.linspace(0.01, 0.2, 6)},
    )


def test_forbidden_current_future_and_supervision_mutations_leave_forecast_identical() -> None:
    sequence = _sequence()
    changed_syndromes = sequence.syndromes.copy()
    changed_syndromes[sequence.forecast_round :] ^= 1
    mutations = {
        "current_and_future_syndromes": replace(sequence, syndromes=changed_syndromes),
        "physical_errors": replace(sequence, physical_errors=1 - sequence.physical_errors),
        "logical_outcomes": replace(sequence, logical_outcomes=1 - sequence.logical_outcomes),
        "diagnostics": replace(sequence, diagnostics={"latent_q": np.ones(6)}),
    }
    forecaster = StatefulPrefixForecaster()

    audit = audit_forecaster_causality(forecaster, sequence, mutations)

    assert audit.passed
    assert audit.forecast_round == 3
    assert audit.checked_mutations == tuple(mutations)
    assert all(check.bit_identical for check in audit.checks)
    assert len(forecaster.calls) == 5
    for history in forecaster.calls:
        assert isinstance(history, ObservedHistory)
        assert history.syndromes.shape == (3, 4)
        assert not history.syndromes.flags.writeable
        assert not hasattr(history, "physical_errors")
        assert not hasattr(history, "logical_outcomes")
        assert not hasattr(history, "diagnostics")


def test_mutating_observed_past_is_rejected_as_not_a_forbidden_field_audit() -> None:
    sequence = _sequence()
    changed = sequence.syndromes.copy()
    changed[0] ^= 1

    with pytest.raises(ValueError, match="observed syndrome prefix"):
        audit_forecaster_causality(
            StatefulPrefixForecaster(),
            sequence,
            {"past_syndrome": replace(sequence, syndromes=changed)},
        )


def test_audit_detects_forecaster_state_that_changes_repeated_fixed_prefix_output() -> None:
    class LeakyStatefulForecaster:
        def __init__(self) -> None:
            self.offset = 0

        def forecast(self, history: ObservedHistory) -> np.ndarray:
            self.offset += 1
            return history.syndromes[-1].astype(np.int64) + self.offset

    sequence = _sequence()
    audit = audit_forecaster_causality(
        LeakyStatefulForecaster(),
        sequence,
        {"logical": replace(sequence, logical_outcomes=1 - sequence.logical_outcomes)},
    )

    assert not audit.passed
    assert audit.checks[0].bit_identical is False


def test_audit_requires_at_least_one_named_mutation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        audit_forecaster_causality(StatefulPrefixForecaster(), _sequence(), {})
