from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from qldpc_fno.models.causal_forecaster import (
    FIRState,
    GRUState,
    HiPPOState,
    PrefixShuffleIdentity,
    build_forecaster,
    deterministic_prefix_permutation,
    trainable_parameter_count,
)
from qldpc_fno.temporal.causality import (
    CausalAuditSequence,
    ObservedHistory,
    audit_structural_prefix_causality,
)
from qldpc_fno.temporal.config import CausalExperimentConfig

CONFIG_PATH = Path("configs/causal_fno_hippo_reduced.json")
PRIMARY_CELLS = (("cnn", "fir"), ("fno", "fir"), ("cnn", "hippo"), ("fno", "hippo"))
ALL_CELLS = (*PRIMARY_CELLS, ("cnn", "gru"), ("fno", "gru"))


@pytest.fixture(scope="module")
def config() -> CausalExperimentConfig:
    return CausalExperimentConfig.from_json(CONFIG_PATH)


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_factory_cells_emit_flat_probabilities_with_shared_geometry(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config)
    state = model.initial_state(batch_size=2)
    q_t, next_state = model.predict_then_update(state, torch.randn(2, 21, 45))

    assert q_t.shape == (2, 2610)
    assert torch.all(q_t >= config.generator.min_probability)
    assert torch.all(q_t <= config.generator.max_probability)
    assert next_state.completed_samples == 1
    assert model.spatial_kind == spatial
    assert model.temporal_kind == temporal
    assert model.readout.kernel_size == (1,)


def test_primary_state_shapes_and_shared_temporal_parameters(
    config: CausalExperimentConfig,
) -> None:
    fir = build_forecaster(spatial="cnn", temporal="fir", config=config)
    fir_state = fir.initial_state(batch_size=2)
    assert isinstance(fir_state, FIRState)
    assert fir_state.history.shape == (2, 32, 32, 45)
    assert fir.temporal.taps.shape == (32, 32)

    hippo = build_forecaster(spatial="cnn", temporal="hippo", config=config)
    hippo_state = hippo.initial_state(batch_size=2)
    assert isinstance(hippo_state, HiPPOState)
    assert hippo_state.memory.shape == (2, 32, 45, 16)
    assert hippo.temporal.readout_coefficients.shape == (32, 16)

    gru = build_forecaster(spatial="cnn", temporal="gru", config=config)
    gru_state = gru.initial_state(batch_size=2)
    assert isinstance(gru_state, GRUState)
    assert gru_state.hidden.shape == (2, 45, 16)
    assert gru.temporal.cell.input_size == 32
    assert gru.temporal.cell.hidden_size == 16


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_predict_then_update_excludes_current_syndrome(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    torch.manual_seed(9)
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
    state = model.initial_state(batch_size=2)
    first = torch.randn(2, 21, 45)
    second = first.clone()
    second[:, :, 7] += 10

    q_first, state_first = model.predict_then_update(state, first)
    q_second, state_second = model.predict_then_update(state, second)

    torch.testing.assert_close(q_first, q_second, rtol=0, atol=0)
    assert not torch.equal(state_first.tensor, state_second.tensor)


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_round_zero_uses_stationary_base_probability(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
    q_zero, _ = model.predict_then_update(
        model.initial_state(batch_size=1), torch.randn(1, 21, 45)
    )

    torch.testing.assert_close(
        q_zero,
        torch.full_like(q_zero, config.generator.base_probability),
        rtol=0,
        atol=1e-7,
    )


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_circular_shift_equivariance(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    torch.manual_seed(12)
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
    syndrome = torch.randn(2, 21, 45)
    state = model.initial_state(batch_size=2)
    _, next_state = model.predict_then_update(state, syndrome)
    q_next, _ = model.predict_then_update(next_state, syndrome * 0)

    shifted_state = model.initial_state(batch_size=2)
    _, shifted_state = model.predict_then_update(shifted_state, torch.roll(syndrome, 11, -1))
    shifted_q, _ = model.predict_then_update(shifted_state, syndrome * 0)

    expected = torch.roll(q_next.reshape(2, 58, 45), 11, -1).reshape(2, 2610)
    torch.testing.assert_close(shifted_q, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_batch_replay_equals_separate_sequences_and_reset_is_fresh(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    torch.manual_seed(22)
    model = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
    syndromes = torch.randn(2, 5, 21, 45)

    batched = model.predict_sequence(syndromes, burn_in=2)
    separate = torch.cat(
        [model.predict_sequence(syndromes[index : index + 1], burn_in=2).probabilities for index in range(2)]
    )

    torch.testing.assert_close(batched.probabilities, separate, rtol=2e-5, atol=2e-6)
    assert batched.scored_mask.tolist() == [False, False, True, True, True]
    fresh = model.initial_state(batch_size=2)
    assert fresh.completed_samples == 0
    assert torch.count_nonzero(fresh.tensor) == 0
    assert not any(name.startswith("_cached") for name, _ in model.named_buffers())


def test_fir_prefix_shuffle_is_deterministic_and_prefix_only() -> None:
    identity = PrefixShuffleIdentity(
        campaign_seed=20260904,
        regime="joint_in_basis",
        role="validation",
        sequence_index=3,
    )
    first = deterministic_prefix_permutation(32, identity=identity, forecast_round=41)
    second = deterministic_prefix_permutation(32, identity=identity, forecast_round=41)

    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(32))
    assert not torch.equal(first, deterministic_prefix_permutation(32, identity=identity, forecast_round=42))
    with pytest.raises(ValueError, match="available prefix"):
        deterministic_prefix_permutation(33, identity=identity, forecast_round=32)


def test_prefix_shuffle_diagnostic_is_rejected_for_recurrent_models(
    config: CausalExperimentConfig,
) -> None:
    syndrome = torch.zeros(1, 4, 21, 45)
    identity = PrefixShuffleIdentity(7, "joint_in_basis", "validation", 0)
    for temporal in ("hippo", "gru"):
        model = build_forecaster(spatial="cnn", temporal=temporal, config=config)
        with pytest.raises(ValueError, match="FIR-only"):
            model.predict_sequence(syndrome, prefix_shuffle_identities=(identity,))


def test_fir_prefix_shuffle_replay_is_deterministic_and_changes_only_lag_order(
    config: CausalExperimentConfig,
) -> None:
    torch.manual_seed(27)
    model = build_forecaster(spatial="cnn", temporal="fir", config=config).eval()
    syndromes = torch.randn(1, 12, 21, 45)
    identity = PrefixShuffleIdentity(20260904, "joint_in_basis", "validation", 1)

    ordinary = model.predict_sequence(syndromes).probabilities
    first = model.predict_sequence(
        syndromes, prefix_shuffle_identities=(identity,)
    ).probabilities
    second = model.predict_sequence(
        syndromes, prefix_shuffle_identities=(identity,)
    ).probabilities

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first[:, 0], ordinary[:, 0], rtol=0, atol=0)
    assert not torch.equal(first[:, 3:], ordinary[:, 3:])


def test_exact_real_scalar_parameter_counts(config: CausalExperimentConfig) -> None:
    expected = {
        ("cnn", "fir"): 11_482,
        ("cnn", "hippo"): 10_970,
        ("cnn", "gru"): 13_402,
        ("fno", "fir"): 54_906,
        ("fno", "hippo"): 54_394,
        ("fno", "gru"): 56_826,
    }
    observed = {
        cell: trainable_parameter_count(build_forecaster(spatial=cell[0], temporal=cell[1], config=config))
        for cell in ALL_CELLS
    }
    assert observed == expected


@pytest.mark.parametrize(("spatial", "temporal"), ALL_CELLS)
def test_concrete_recreated_models_exclude_privileged_fields_and_spy_only_prefix(
    config: CausalExperimentConfig, spatial: str, temporal: str
) -> None:
    torch.manual_seed(31)
    reference = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
    weights = copy.deepcopy(reference.state_dict())
    syndromes = np.arange(7 * 21 * 45, dtype=np.float32).reshape(7, 21, 45) % 2
    source = CausalAuditSequence(
        syndromes=syndromes,
        forecast_round=4,
        physical_errors=np.zeros((7, 58, 45), dtype=np.uint8),
        logical_outcomes=np.zeros((7, 3), dtype=np.uint8),
        diagnostics={"latent_q": np.zeros((7, 58, 45)), "event": [False] * 7},
    )
    mutations = {
        "current_and_future": CausalAuditSequence(
            syndromes=np.concatenate((syndromes[:4], 1 - syndromes[4:])),
            forecast_round=4,
            physical_errors=np.ones((7, 58, 45), dtype=np.uint8),
            logical_outcomes=np.ones((7, 3), dtype=np.uint8),
            diagnostics={"latent_q": np.ones((7, 58, 45)), "event": [True] * 7},
        )
    }
    captured: list[torch.Tensor] = []

    class RecreatedForecaster:
        def forecast(self, history: ObservedHistory) -> np.ndarray:
            concrete = build_forecaster(spatial=spatial, temporal=temporal, config=config).eval()
            concrete.load_state_dict(weights)
            hook = concrete.spatial.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().clone())
            )
            try:
                return concrete.forecast(history)
            finally:
                hook.remove()

    audit = audit_structural_prefix_causality(RecreatedForecaster(), source, mutations)

    assert audit.passed
    assert len(captured) == 8
    expected_prefix = torch.from_numpy(syndromes[:4])
    for offset, actual in enumerate(captured):
        torch.testing.assert_close(actual, expected_prefix[offset % 4].unsqueeze(0))


def test_factory_rejects_unknown_axes(config: CausalExperimentConfig) -> None:
    with pytest.raises(ValueError, match="spatial"):
        build_forecaster(spatial="attention", temporal="fir", config=config)
    with pytest.raises(ValueError, match="temporal"):
        build_forecaster(spatial="cnn", temporal="lstm", config=config)
