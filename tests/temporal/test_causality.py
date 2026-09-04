from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.temporal.causality import (
    CausalAuditSequence,
    ObservedHistory,
    audit_structural_prefix_causality,
)
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.generator import generate_latent_sequence, sample_sequence


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

    audit = audit_structural_prefix_causality(forecaster, sequence, mutations)

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
        audit_structural_prefix_causality(
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
    audit = audit_structural_prefix_causality(
        LeakyStatefulForecaster(),
        sequence,
        {"logical": replace(sequence, logical_outcomes=1 - sequence.logical_outcomes)},
    )

    assert not audit.passed
    assert audit.checks[0].bit_identical is False


def test_audit_requires_at_least_one_named_mutation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        audit_structural_prefix_causality(StatefulPrefixForecaster(), _sequence(), {})


def test_identical_mutation_record_is_rejected_as_vacuous() -> None:
    sequence = _sequence()

    with pytest.raises(ValueError, match="identical to the source record"):
        audit_structural_prefix_causality(
            StatefulPrefixForecaster(),
            sequence,
            {"unchanged": replace(sequence)},
        )


def test_reused_forecast_buffer_cannot_retroactively_change_baseline_snapshot() -> None:
    class ReusedBufferForecaster:
        def __init__(self) -> None:
            self.buffer = np.zeros(4, dtype=np.int64)
            self.calls = 0

        def forecast(self, history: ObservedHistory) -> np.ndarray:
            self.calls += 1
            self.buffer.fill(self.calls)
            return self.buffer

    sequence = _sequence()
    audit = audit_structural_prefix_causality(
        ReusedBufferForecaster(),
        sequence,
        {"logical": replace(sequence, logical_outcomes=1 - sequence.logical_outcomes)},
    )

    assert not audit.passed
    assert audit.checks[0].bit_identical is False


def test_structural_audit_documents_external_privileged_state_limitation() -> None:
    documentation = " ".join((audit_structural_prefix_causality.__doc__ or "").split())
    assert (
        "cannot detect forecaster-captured external privileged state" in documentation
    )


def test_generated_canonical_sequence_passes_end_to_end_structural_audit() -> None:
    config = CausalExperimentConfig.from_json(
        Path("configs/causal_fno_hippo_reduced.json")
    )
    latent = generate_latent_sequence(
        config,
        regime="joint_in_basis",
        role="train",
        sequence_index=0,
    )
    observed, supervision, diagnostics = sample_sequence(
        latent,
        bernoulli_seed=latent.seeds.bernoulli,
        code=build_self_lifted_product(PAPER_LP_3_7_16),
    )
    diagnostic_fields = {
        field.name: getattr(diagnostics, field.name) for field in fields(diagnostics)
    }
    sequence = CausalAuditSequence(
        syndromes=observed.syndromes,
        forecast_round=config.rounds.burn_in,
        physical_errors=supervision.errors,
        logical_outcomes=supervision.logical_flips,
        diagnostics=diagnostic_fields,
    )
    changed_syndromes = observed.syndromes.copy()
    changed_syndromes[config.rounds.burn_in :] ^= 1
    mutations = {
        "current_and_future_syndromes": replace(
            sequence,
            syndromes=changed_syndromes,
        ),
        "physical_errors": replace(
            sequence,
            physical_errors=1 - supervision.errors,
        ),
        "logical_outcomes": replace(
            sequence,
            logical_outcomes=1 - supervision.logical_flips,
        ),
        "diagnostics": replace(
            sequence,
            diagnostics={**diagnostic_fields, "probabilities": 1 - diagnostics.probabilities},
        ),
    }

    forecaster = StatefulPrefixForecaster()
    audit = audit_structural_prefix_causality(forecaster, sequence, mutations)

    assert audit.passed
    assert forecaster.calls[0].syndromes.shape == (
        config.rounds.burn_in,
        21,
        45,
    )
    assert np.array_equal(
        forecaster.calls[0].syndromes,
        observed.syndromes[: config.rounds.burn_in],
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("physical_errors", np.zeros((5, 58, 45), dtype=np.uint8)),
        ("logical_outcomes", np.zeros((5, 744), dtype=np.uint8)),
    ],
)
def test_canonical_supervision_requires_matching_round_axis(
    field_name: str,
    value: np.ndarray,
) -> None:
    arguments = {
        "syndromes": np.zeros((6, 21, 45), dtype=np.uint8),
        "forecast_round": 3,
        "physical_errors": np.zeros((6, 58, 45), dtype=np.uint8),
        "logical_outcomes": np.zeros((6, 744), dtype=np.uint8),
    }
    arguments[field_name] = value

    with pytest.raises(ValueError, match="same round count"):
        CausalAuditSequence(**arguments)  # type: ignore[arg-type]
