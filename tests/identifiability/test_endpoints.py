from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.endpoints import (
    EndpointBatch,
    EndpointSequence,
    SequenceEndpointAccumulator,
    calibration_by_sequence,
    expected_ce_by_sequence,
    latent_nmse_by_sequence,
    retained_syndrome_nll_by_sequence,
)
from qldpc_fno.identifiability.filters import ForecastResult
from qldpc_fno.identifiability.observation import DisjointChecks, greedy_disjoint_rows
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    LatentHistoryOracleInput,
    SequenceIdentity,
    TrainingTargets,
)


def _checks() -> DisjointChecks:
    return greedy_disjoint_rows(build_self_lifted_product(PAPER_LP_3_7_16).hx)


def _identity(
    index: int = 0,
    *,
    role: str = "test",
    regime: str = "stationary_iid",
    content: str | None = None,
) -> SequenceIdentity:
    return SequenceIdentity(
        regime,
        role,
        index,
        index + 1,
        index + 11,
        content if content is not None else f"{index + 1:x}" * 64,
    )


def _sequence(
    q: np.ndarray,
    q_hat: np.ndarray,
    mask: np.ndarray,
    *,
    identity: SequenceIdentity | None = None,
    state: np.ndarray | None = None,
    syndromes: np.ndarray | None = None,
    arm: str = "ewma",
) -> EndpointSequence:
    latent = np.asarray(q, dtype=np.float64)
    rounds = latent.shape[0]
    if state is None:
        state = np.zeros(rounds)
    if syndromes is None:
        syndromes = np.zeros((rounds, 2), dtype=np.uint8)
    return EndpointSequence(
        identity=_identity() if identity is None else identity,
        observed=DeployableHistory(syndromes, mask),
        latent_state=LatentHistoryOracleInput(state),
        latent_probabilities=ContemporaneousOracleInput(latent),
        forecast=ForecastResult(arm, q_hat, None, None),
    )


def _batch(*sequences: EndpointSequence) -> EndpointBatch:
    return EndpointBatch(tuple(sequences))


def test_expected_ce_uses_latent_probabilities_and_only_scored_rounds() -> None:
    q0 = np.array([[0.1], [0.2], [0.24]])
    p0 = np.array([[0.2], [0.25], [0.25]])
    q1 = np.array([[0.24], [0.2], [0.1]])
    p1 = np.array([[0.2], [0.2], [0.2]])
    mask = np.array([False, True, True])
    batch = _batch(
        _sequence(q0, p0, mask, identity=_identity(0)),
        _sequence(q1, p1, mask, identity=_identity(1)),
    )

    observed = expected_ce_by_sequence(batch)

    expected = np.array(
        [
            np.mean(-q0[1:] * np.log(p0[1:]) - (1 - q0[1:]) * np.log1p(-p0[1:])),
            np.mean(-q1[1:] * np.log(p1[1:]) - (1 - q1[1:]) * np.log1p(-p1[1:])),
        ]
    )
    assert np.allclose(observed, expected)


@pytest.mark.parametrize(
    ("scorer", "extra"),
    [
        (expected_ce_by_sequence, ()),
        (latent_nmse_by_sequence, ()),
        (calibration_by_sequence, ()),
        (retained_syndrome_nll_by_sequence, (_checks(),)),
    ],
)
def test_every_endpoint_public_boundary_rejects_raw_arrays(scorer, extra: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="EndpointBatch"):
        scorer(np.full((2, 1), 0.1), *extra)


def test_endpoint_sequence_rejects_raw_arrays_and_sampled_targets() -> None:
    q = np.full((2, 1), 0.1)
    forecast = ForecastResult("ewma", np.full((2, 1), 0.1), None, None)
    targets = TrainingTargets(
        np.zeros((2, 1), dtype=np.uint8), np.zeros((2, 1), dtype=np.uint8)
    )

    with pytest.raises(TypeError, match="ContemporaneousOracleInput"):
        EndpointSequence(
            _identity(),
            DeployableHistory(np.zeros((2, 1), dtype=np.uint8), np.ones(2, dtype=np.bool_)),
            LatentHistoryOracleInput(np.zeros(2)),
            q,  # type: ignore[arg-type]
            forecast,
        )
    with pytest.raises(TypeError, match="ContemporaneousOracleInput"):
        EndpointSequence(
            _identity(),
            DeployableHistory(np.zeros((2, 1), dtype=np.uint8), np.ones(2, dtype=np.bool_)),
            LatentHistoryOracleInput(np.zeros(2)),
            targets,  # type: ignore[arg-type]
            forecast,
        )
    with pytest.raises(TypeError, match="ForecastResult"):
        EndpointSequence(
            _identity(),
            DeployableHistory(np.zeros((2, 1), dtype=np.uint8), np.ones(2, dtype=np.bool_)),
            LatentHistoryOracleInput(np.zeros(2)),
            ContemporaneousOracleInput(q),
            q,  # type: ignore[arg-type]
        )


def test_endpoint_batch_rejects_mixed_or_duplicate_identities() -> None:
    values = np.full((2, 1), 0.1)
    mask = np.ones(2, dtype=np.bool_)
    first = _sequence(values, values, mask, identity=_identity(0))

    with pytest.raises(ValueError, match="content-disjoint"):
        EndpointBatch((first, _sequence(values, values, mask, identity=_identity(0))))
    with pytest.raises(ValueError, match="roles"):
        EndpointBatch(
            (first, _sequence(values, values, mask, identity=_identity(1, role="calibration")))
        )
    with pytest.raises(ValueError, match="regimes"):
        EndpointBatch(
            (first, _sequence(values, values, mask, identity=_identity(1, regime="temporal_uniform")))
        )


def test_endpoint_sequence_rejects_incomplete_scored_mask() -> None:
    with pytest.raises(ValueError, match="scored"):
        _sequence(
            np.full((2, 1), 0.1),
            np.full((2, 1), 0.1),
            np.array([True, False]),
        )


def test_latent_nmse_clips_only_after_mean_log_odds_mapping() -> None:
    base = 0.0375
    q_hat = np.array([[0.24, 1e-5]])
    sequence = _sequence(
        np.full((1, 2), base),
        q_hat,
        np.array([True]),
        state=np.array([0.0]),
    )

    observed = latent_nmse_by_sequence(_batch(sequence))

    offsets = np.log(q_hat / (1 - q_hat)) - math.log(base / (1 - base))
    mapped = np.clip(offsets.mean(axis=1), -1.2, 1.2)
    expected = np.square(mapped) / (0.08**2 / (1 - 0.97**2))
    per_qubit_clipped = np.clip(offsets, -1.2, 1.2).mean(axis=1)
    assert observed == pytest.approx(expected)
    assert not np.allclose(
        observed, np.square(per_qubit_clipped) / (0.08**2 / (1 - 0.97**2))
    )


def test_retained_syndrome_nll_uses_predictive_parity_probabilities() -> None:
    checks = _checks()
    syndromes = np.zeros((2, 945), dtype=np.uint8)
    syndromes[1, checks.row_indices[0]] = 1
    q_hat = np.array([0.1, 0.2])
    sequence = _sequence(
        np.full((2, 2610), 0.1),
        q_hat,
        np.array([False, True]),
        syndromes=syndromes,
    )

    observed = retained_syndrome_nll_by_sequence(_batch(sequence), checks)

    parity = (1 - (1 - 2 * 0.2) ** 10) / 2
    expected = np.array(
        [-np.log(parity) - (len(checks.row_indices) - 1) * np.log1p(-parity)]
    ) / len(checks.row_indices)
    assert np.allclose(observed, expected)


def test_retained_syndrome_nll_accepts_typed_scalar_forecast() -> None:
    sequence = _sequence(
        np.full((2, 2610), 0.1),
        np.array([0.1, 0.2]),
        np.array([False, True]),
        syndromes=np.zeros((2, 945), dtype=np.uint8),
    )

    observed = retained_syndrome_nll_by_sequence(_batch(sequence), _checks())

    assert observed.shape == (1,)


def test_retained_syndrome_nll_rejects_noncanonical_support_container() -> None:
    sequence = _sequence(
        np.full((1, 2), 0.1), np.full((1, 2), 0.1), np.array([True])
    )
    with pytest.raises(TypeError, match="DisjointChecks"):
        retained_syndrome_nll_by_sequence(
            _batch(sequence), ((0, 1), (0,))  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field",
    [
        "matrix_sha256",
        "algorithm_version",
        "row_indices",
        "supports",
        "weights",
        "covered_qubits",
    ],
)
def test_retained_syndrome_nll_rejects_every_forged_canonical_field(field: str) -> None:
    canonical = _checks()
    forged = DisjointChecks(
        canonical.row_indices,
        canonical.supports,
        canonical.weights,
        canonical.covered_qubits,
        canonical.algorithm_version,
        canonical.matrix_sha256,
    )
    if field == "matrix_sha256":
        replacement: object = "0" * 64
    elif field == "algorithm_version":
        replacement = "greedy_disjoint_rows/forged"
    elif field == "row_indices":
        replacement = canonical.row_indices.tolist()
    elif field == "supports":
        replacement = (canonical.supports[0][::-1].copy(), *canonical.supports[1:])
    elif field == "weights":
        replacement = canonical.weights + np.arange(len(canonical.weights)) % 2
    else:
        replacement = canonical.covered_qubits[::-1].copy()
    object.__setattr__(forged, field, replacement)
    sequence = _sequence(
        np.full((1, 2610), 0.1),
        np.array([0.1]),
        np.array([True]),
        syndromes=np.zeros((1, 945), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="canonical"):
        retained_syndrome_nll_by_sequence(_batch(sequence), forged)


@pytest.mark.parametrize(
    ("syndrome_width", "qubit_width"),
    [(944, 2610), (945, 2609)],
)
def test_retained_syndrome_nll_requires_canonical_geometry(
    syndrome_width: int, qubit_width: int
) -> None:
    sequence = _sequence(
        np.full((1, qubit_width), 0.1),
        np.array([0.1]),
        np.array([True]),
        syndromes=np.zeros((1, syndrome_width), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="945.*2610"):
        retained_syndrome_nll_by_sequence(_batch(sequence), _checks())


def test_calibration_uses_exact_internal_edge_closure_and_latent_targets() -> None:
    edges = np.linspace(1e-5, 0.25, 11)
    below = np.nextafter(edges[1], -np.inf)
    predictions = np.array([[1e-5], [below], [edges[1]], [0.25]])
    latent = np.array([[0.01], [0.02], [0.03], [0.04]])
    evidence = calibration_by_sequence(
        _batch(_sequence(latent, predictions, np.ones(4, dtype=np.bool_)))
    )

    assert evidence.counts.shape == (1, 10)
    assert evidence.counts[0, 0] == 2
    assert evidence.counts[0, 1] == 1
    assert evidence.counts[0, -1] == 1
    assert evidence.predicted_sums[0, 0] == pytest.approx(1e-5 + below)
    assert evidence.latent_sums[0, 1] == pytest.approx(0.03)
    expected_error = np.abs(evidence.predicted_sums - evidence.latent_sums).sum() / 4
    assert evidence.absolute_error[0] == pytest.approx(expected_error)


def test_sequence_accumulator_accepts_only_typed_input_and_retains_bounded_state() -> None:
    sequence = _sequence(
        np.array([[0.1], [0.2], [0.24]]),
        np.array([[0.2], [0.25], [0.25]]),
        np.array([False, True, True]),
    )
    accumulator = SequenceEndpointAccumulator()

    with pytest.raises(TypeError, match="EndpointSequence"):
        accumulator.update(np.array([0.1]))  # type: ignore[arg-type]
    accumulator.update(sequence)

    assert accumulator.expected_ce == pytest.approx(expected_ce_by_sequence(_batch(sequence))[0])
    assert accumulator.count == 2
    assert accumulator.calibration_counts is not None
    assert accumulator.calibration_counts.shape == (10,)
    assert all(
        field.name
        not in {"sequence", "sequences", "latent", "forecast", "probabilities", "syndromes"}
        for field in dataclasses.fields(accumulator)
    )
