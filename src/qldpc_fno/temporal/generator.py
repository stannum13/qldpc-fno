"""Causal repeated code-capacity sequence generation for the LP benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Protocol

import numpy as np
from scipy import sparse

from qldpc_fno.campaign.code_identity import validate_campaign_code_identity
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.seeds import SequenceSeeds, sequence_seed_tuple


class SamplingCode(Protocol):
    name: str
    ell: int
    n: int
    k: int
    hx: sparse.spmatrix
    hz: sparse.spmatrix


def _readonly(array: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


class _ImmutableArrays:
    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field.name, _readonly(value))


@dataclass(frozen=True, slots=True)
class LatentSequence(_ImmutableArrays):
    """Complete simulator state; never a deployable forecaster input."""

    regime: str
    role: str
    sequence_index: int
    seeds: SequenceSeeds
    burn_in: int
    probabilities: np.ndarray
    global_log_odds: np.ndarray
    spatial_log_odds: np.ndarray
    base_spatial_log_odds: np.ndarray
    channel_offsets: np.ndarray
    event_active: np.ndarray
    event_onset: np.ndarray
    event_termination: np.ndarray
    event_center: np.ndarray
    event_age: np.ndarray
    event_width: np.ndarray
    event_step: np.ndarray


@dataclass(frozen=True, slots=True)
class CausalObservedSequence(_ImmutableArrays):
    """Fields available to a causal forecaster at deployment."""

    syndromes: np.ndarray
    scored_mask: np.ndarray
    ell: int
    syndrome_channels: int


@dataclass(frozen=True, slots=True)
class CausalSupervision(_ImmutableArrays):
    """Simulator labels available only to training and evaluation code."""

    errors: np.ndarray
    logical_flips: np.ndarray


@dataclass(frozen=True, slots=True)
class SimulatorDiagnostics(_ImmutableArrays):
    """Privileged latent state available only for generator diagnostics."""

    probabilities: np.ndarray
    global_log_odds: np.ndarray
    spatial_log_odds: np.ndarray
    channel_offsets: np.ndarray
    event_active: np.ndarray
    event_onset: np.ndarray
    event_termination: np.ndarray
    event_center: np.ndarray
    event_age: np.ndarray
    event_width: np.ndarray
    event_step: np.ndarray


@dataclass(frozen=True, slots=True)
class GeneratedSequence:
    """One role-batch member with simulator state kept separate from public fields."""

    latent: LatentSequence
    observed: CausalObservedSequence
    supervision: CausalSupervision
    diagnostics: SimulatorDiagnostics


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _global_process(
    rounds: int, config: CausalExperimentConfig, rng: np.random.Generator
) -> np.ndarray:
    values = np.zeros(rounds, dtype=np.float64)
    generator = config.generator
    for round_index in range(1, rounds):
        innovation = rng.normal(0.0, generator.global_innovation_std)
        values[round_index] = np.clip(
            generator.global_ar_coefficient * values[round_index - 1] + innovation,
            -generator.global_clip,
            generator.global_clip,
        )
    return values


def _sample_frequency_phase(
    config: CausalExperimentConfig, rng: np.random.Generator
) -> tuple[int, float]:
    low, high = config.generator.spatial_frequency_range
    return int(rng.integers(low, high + 1)), float(rng.uniform(0.0, 2.0 * math.pi))


def _cosine_field(*, ell: int, frequency: int, phase: float, amplitude: float) -> np.ndarray:
    coordinates = np.arange(ell, dtype=np.float64)
    return amplitude * np.cos(2.0 * math.pi * frequency * coordinates / ell + phase)


def _channel_offsets(
    config: CausalExperimentConfig,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    channels = config.code.n // config.code.ell
    if regime in {"stationary_iid", "temporal_uniform"}:
        return np.zeros(channels, dtype=np.float64)
    offsets = rng.normal(0.0, config.generator.channel_offset_std, size=channels)
    return offsets - np.mean(offsets)


def _empty_event_arrays(rounds: int) -> dict[str, np.ndarray]:
    return {
        "event_active": np.zeros(rounds, dtype=np.bool_),
        "event_onset": np.zeros(rounds, dtype=np.bool_),
        "event_termination": np.zeros(rounds, dtype=np.bool_),
        "event_center": np.full(rounds, -1, dtype=np.int16),
        "event_age": np.full(rounds, -1, dtype=np.int16),
        "event_width": np.zeros(rounds, dtype=np.int16),
        "event_step": np.zeros(rounds, dtype=np.int8),
    }


def _joint_in_basis_spatial(
    config: CausalExperimentConfig,
    rng: np.random.Generator,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    generator = config.generator
    ell = config.code.ell
    frequency, phase = _sample_frequency_phase(config, rng)
    base = _cosine_field(
        ell=ell,
        frequency=frequency,
        phase=phase,
        amplitude=generator.joint_base_amplitude,
    )
    spatial = np.repeat(base[None, :], rounds, axis=0)
    events = _empty_event_arrays(rounds)
    active = False
    center = -1
    age = -1
    coordinates = np.arange(ell, dtype=np.float64)

    for round_index in range(rounds):
        if not active and rng.random() < generator.burst_start_probability:
            active = True
            center = int(rng.integers(0, ell))
            age = 0
            events["event_onset"][round_index] = True
        if not active:
            continue

        amplitude = generator.burst_amplitude * math.exp(-age / generator.burst_decay_rounds)
        profile = amplitude * np.exp(
            generator.burst_profile_concentration
            * (np.cos(2.0 * math.pi * (coordinates - center) / ell) - 1.0)
        )
        spatial[round_index] += profile
        events["event_active"][round_index] = True
        events["event_center"][round_index] = center
        events["event_age"][round_index] = age

        terminate = age >= generator.burst_max_age or amplitude < generator.burst_min_amplitude
        events["event_termination"][round_index] = terminate
        age += 1
        if terminate:
            active = False
            center = -1
            age = -1

    return spatial, base, events


def _joint_basis_mismatch_spatial(
    config: CausalExperimentConfig,
    rng: np.random.Generator,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    generator = config.generator
    ell = config.code.ell
    spatial = np.zeros((rounds, ell), dtype=np.float64)
    base = np.zeros(ell, dtype=np.float64)
    events = _empty_event_arrays(rounds)
    active = False
    center = -1
    age = -1
    duration_remaining = 0
    width = 0
    step = 0

    for round_index in range(rounds):
        if not active and rng.random() < generator.burst_start_probability:
            active = True
            center = int(rng.integers(0, ell))
            duration_remaining = int(
                rng.integers(generator.mismatch_duration_min, generator.mismatch_duration_max + 1)
            )
            width = int(rng.integers(generator.mismatch_width_min, generator.mismatch_width_max + 1))
            step = int(rng.integers(generator.mismatch_step_min, generator.mismatch_step_max + 1))
            age = 0
            events["event_onset"][round_index] = True
        if not active:
            continue

        left = -(width // 2)
        sites = (center + np.arange(left, left + width)) % ell
        spatial[round_index, sites] = generator.burst_amplitude
        events["event_active"][round_index] = True
        events["event_center"][round_index] = center
        events["event_age"][round_index] = age
        events["event_width"][round_index] = width
        events["event_step"][round_index] = step

        center = (center + step) % ell
        age += 1
        duration_remaining -= 1
        terminate = duration_remaining == 0
        events["event_termination"][round_index] = terminate
        if terminate:
            active = False
            center = -1
            age = -1
            width = 0
            step = 0

    return spatial, base, events


def generate_latent_sequence(
    config: CausalExperimentConfig,
    *,
    regime: str,
    role: str,
    sequence_index: int,
) -> LatentSequence:
    """Generate one latent sequence using only its domain-separated latent RNG."""
    config.validate()
    if regime not in config.regimes:
        raise ValueError(f"regime is not enabled by configuration: {regime}")
    seeds = sequence_seed_tuple(
        config.campaign_seed,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
    )
    rng = np.random.default_rng(seeds.latent)
    rounds = config.rounds.burn_in + config.rounds.scored
    ell = config.code.ell
    offsets = _channel_offsets(config, regime, rng)
    events = _empty_event_arrays(rounds)
    global_values = np.zeros(rounds, dtype=np.float64)
    base = np.zeros(ell, dtype=np.float64)

    if regime == "stationary_iid":
        spatial = np.zeros((rounds, ell), dtype=np.float64)
    elif regime == "static_spatial_latent":
        frequency, phase = _sample_frequency_phase(config, rng)
        base = _cosine_field(
            ell=ell,
            frequency=frequency,
            phase=phase,
            amplitude=config.generator.static_spatial_amplitude,
        )
        spatial = np.repeat(base[None, :], rounds, axis=0)
    elif regime == "temporal_uniform":
        global_values = _global_process(rounds, config, rng)
        spatial = np.zeros((rounds, ell), dtype=np.float64)
    elif regime == "joint_in_basis":
        global_values = _global_process(rounds, config, rng)
        spatial, base, events = _joint_in_basis_spatial(config, rng, rounds)
    elif regime == "joint_basis_mismatch":
        global_values = _global_process(rounds, config, rng)
        spatial, base, events = _joint_basis_mismatch_spatial(config, rng, rounds)
    else:  # defensive: sequence_seed_tuple already validates canonical names
        raise ValueError(f"unsupported temporal regime: {regime}")

    base_log_odds = math.log(
        config.generator.base_probability / (1.0 - config.generator.base_probability)
    )
    logits = (
        base_log_odds
        + global_values[:, None, None]
        + spatial[:, None, :]
        + offsets[None, :, None]
    )
    probabilities = np.clip(
        _sigmoid(logits),
        config.generator.min_probability,
        config.generator.max_probability,
    )
    if regime == "stationary_iid":
        probabilities.fill(config.generator.base_probability)

    return LatentSequence(
        regime=regime,
        role=role,
        sequence_index=sequence_index,
        seeds=seeds,
        burn_in=config.rounds.burn_in,
        probabilities=probabilities,
        global_log_odds=global_values,
        spatial_log_odds=spatial,
        base_spatial_log_odds=base,
        channel_offsets=offsets,
        **events,
    )


@lru_cache(maxsize=1)
def _canonical_logical_x() -> sparse.csr_matrix:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    return logical_x_basis(code.hx, code.hz)


def _logical_x_for_code(code: SamplingCode) -> sparse.csr_matrix:
    validate_campaign_code_identity(
        {"name": code.name, "ell": code.ell, "n": code.n, "k": code.k},
        code.hx,
        code.hz,
    )
    return _canonical_logical_x()


def sample_sequence(
    latent: LatentSequence,
    *,
    bernoulli_seed: int,
    code: SamplingCode,
) -> tuple[CausalObservedSequence, CausalSupervision, SimulatorDiagnostics]:
    """Sample errors and derive syndrome/logical labels from one shared code identity."""
    if type(bernoulli_seed) is not int or bernoulli_seed < 0:
        raise ValueError("bernoulli_seed must be a non-negative integer")
    if bernoulli_seed != latent.seeds.bernoulli:
        raise ValueError("bernoulli_seed does not match latent sequence identity")
    rounds, qubit_channels, ell = latent.probabilities.shape
    if ell != code.ell or qubit_channels * ell != code.n:
        raise ValueError("latent probability geometry does not match the sampling code")
    if code.hx.shape[1] != code.n or code.hz.shape[1] != code.n:
        raise ValueError("code matrices do not match the declared block length")
    if code.hx.shape[0] % ell:
        raise ValueError("check rows must be an integer number of ring channels")

    logical_x = _logical_x_for_code(code)
    rng = np.random.default_rng(bernoulli_seed)
    errors = (rng.random(latent.probabilities.shape) < latent.probabilities).astype(np.uint8)
    errors_flat = errors.reshape(rounds, code.n)
    syndromes_flat = np.asarray(code.hx @ errors_flat.T, dtype=np.uint8).T % 2
    logical_flips = np.asarray(logical_x @ errors_flat.T, dtype=np.uint8).T % 2
    syndrome_channels = code.hx.shape[0] // ell
    scored_mask = np.arange(rounds) >= latent.burn_in

    observed = CausalObservedSequence(
        syndromes=syndromes_flat.reshape(rounds, syndrome_channels, ell),
        scored_mask=scored_mask,
        ell=ell,
        syndrome_channels=syndrome_channels,
    )
    supervision = CausalSupervision(errors=errors, logical_flips=logical_flips)
    diagnostics = SimulatorDiagnostics(
        probabilities=latent.probabilities,
        global_log_odds=latent.global_log_odds,
        spatial_log_odds=latent.spatial_log_odds,
        channel_offsets=latent.channel_offsets,
        event_active=latent.event_active,
        event_onset=latent.event_onset,
        event_termination=latent.event_termination,
        event_center=latent.event_center,
        event_age=latent.event_age,
        event_width=latent.event_width,
        event_step=latent.event_step,
    )
    return observed, supervision, diagnostics


def generate_role_batch(
    config: CausalExperimentConfig,
    *,
    regime: str,
    role: str,
    code: SamplingCode,
) -> tuple[GeneratedSequence, ...]:
    """Generate exactly the configured independent sequence membership for one role."""
    role_sizes = {
        "train": config.splits.train,
        "validation": config.splits.validation,
        "calibration": config.splits.calibration,
        "test": config.splits.test,
    }
    if role not in role_sizes:
        raise ValueError(f"unsupported temporal role: {role}")
    members: list[GeneratedSequence] = []
    for sequence_index in range(role_sizes[role]):
        latent = generate_latent_sequence(
            config,
            regime=regime,
            role=role,
            sequence_index=sequence_index,
        )
        observed, supervision, diagnostics = sample_sequence(
            latent,
            bernoulli_seed=latent.seeds.bernoulli,
            code=code,
        )
        members.append(
            GeneratedSequence(
                latent=latent,
                observed=observed,
                supervision=supervision,
                diagnostics=diagnostics,
            )
        )
    return tuple(members)
