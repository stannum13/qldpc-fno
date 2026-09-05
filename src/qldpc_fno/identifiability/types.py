"""Noninterchangeable data boundaries for temporal identifiability."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields

import numpy as np

from qldpc_fno.identifiability.config import REGIMES, ROLES

_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch


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


def _binary_matrix(array: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional array")
    if not np.all((value == 0) | (value == 1)):
        raise ValueError(f"{label} must be binary")
    return value


@dataclass(frozen=True, slots=True)
class SequenceIdentity:
    """RNG and payload identity for exactly one generated sequence."""

    regime: str
    role: str
    sequence_index: int
    latent_seed: int
    bernoulli_seed: int
    content_sha256: str | None

    def __post_init__(self) -> None:
        if self.regime not in REGIMES:
            raise ValueError(f"unsupported sequence regime: {self.regime}")
        if self.role not in ROLES:
            raise ValueError(f"unsupported sequence role: {self.role}")
        for label, value in (
            ("sequence_index", self.sequence_index),
            ("latent_seed", self.latent_seed),
            ("bernoulli_seed", self.bernoulli_seed),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if self.latent_seed == self.bernoulli_seed:
            raise ValueError("latent and Bernoulli seeds must be separate")
        if self.content_sha256 is not None and _SHA256(self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class DeployableHistory(_ImmutableArrays):
    """Only observations a deployed forecaster may receive."""

    syndromes: np.ndarray
    scored_mask: np.ndarray

    def __post_init__(self) -> None:
        syndromes = _binary_matrix(self.syndromes, "syndromes")
        mask = np.asarray(self.scored_mask)
        if mask.ndim != 1 or mask.shape[0] != syndromes.shape[0]:
            raise ValueError("scored_mask must have one entry per syndrome round")
        if mask.dtype != np.bool_:
            raise ValueError("scored_mask must have boolean dtype")
        object.__setattr__(self, "syndromes", _readonly(syndromes, dtype=np.uint8))
        object.__setattr__(self, "scored_mask", _readonly(mask, dtype=np.bool_))


@dataclass(frozen=True, slots=True)
class LatentHistoryOracleInput(_ImmutableArrays):
    """Privileged causal history available only to the latent oracle."""

    global_log_odds: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.global_log_odds)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("global_log_odds must be a finite one-dimensional array")
        object.__setattr__(self, "global_log_odds", _readonly(values, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class ContemporaneousOracleInput(_ImmutableArrays):
    """Noncausal current-round probabilities for the scoring ceiling."""

    probabilities: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.probabilities)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("probabilities must be a finite two-dimensional array")
        if not np.all((values > 0.0) & (values < 0.5)):
            raise ValueError("probabilities must be strictly between zero and one half")
        object.__setattr__(self, "probabilities", _readonly(values, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class TrainingTargets(_ImmutableArrays):
    """Simulator labels unavailable to deployable forecasters."""

    errors: np.ndarray
    logical_flips: np.ndarray

    def __post_init__(self) -> None:
        errors = _binary_matrix(self.errors, "errors")
        logical = _binary_matrix(self.logical_flips, "logical_flips")
        if errors.shape[0] != logical.shape[0]:
            raise ValueError("errors and logical_flips must have the same round count")
        object.__setattr__(self, "errors", _readonly(errors, dtype=np.uint8))
        object.__setattr__(self, "logical_flips", _readonly(logical, dtype=np.uint8))


def require_deployable_history(value: object) -> DeployableHistory:
    """Enforce the exact public type at every deployable entry point."""
    if type(value) is not DeployableHistory:
        raise TypeError("forecasters require the exact DeployableHistory type")
    return value


@dataclass(frozen=True, slots=True)
class GeneratedSequence:
    """Orchestration-only aggregate; never a forecaster input."""

    identity: SequenceIdentity
    deployable: DeployableHistory
    latent_oracle: LatentHistoryOracleInput
    contemporaneous_oracle: ContemporaneousOracleInput
    targets: TrainingTargets

    def __post_init__(self) -> None:
        if self.identity.content_sha256 is None:
            raise ValueError("generated sequence identity must be content-bound")
        rounds = self.deployable.syndromes.shape[0]
        if (
            self.latent_oracle.global_log_odds.shape[0] != rounds
            or self.contemporaneous_oracle.probabilities.shape[0] != rounds
            or self.targets.errors.shape[0] != rounds
            or self.targets.logical_flips.shape[0] != rounds
        ):
            raise ValueError("generated sequence containers must have the same round count")
        if self.contemporaneous_oracle.probabilities.shape != self.targets.errors.shape:
            raise ValueError("probability and error arrays must have identical shapes")


@dataclass(frozen=True, slots=True)
class DevelopmentPartitions:
    """Content-bound identities permitted during pre-confirmatory development."""

    train: tuple[SequenceIdentity, ...]
    validation: tuple[SequenceIdentity, ...]
    calibration: tuple[SequenceIdentity, ...]

    def __post_init__(self) -> None:
        expected_roles = (
            ("train", self.train),
            ("validation", self.validation),
            ("calibration", self.calibration),
        )
        seen_keys: set[tuple[str, str, int]] = set()
        seen_content: set[str] = set()
        for expected_role, identities in expected_roles:
            if type(identities) is not tuple:
                raise TypeError("development identity partitions must be immutable tuples")
            for identity in identities:
                if type(identity) is not SequenceIdentity:
                    raise TypeError("development partitions require exact SequenceIdentity values")
                if identity.role != expected_role or identity.role == "test":
                    raise ValueError(f"identity role must match the {expected_role} partition")
                if identity.content_sha256 is None:
                    raise ValueError("development identities must be content-bound")
                key = (identity.regime, identity.role, identity.sequence_index)
                if key in seen_keys or identity.content_sha256 in seen_content:
                    raise ValueError("development partitions must be pairwise-disjoint")
                seen_keys.add(key)
                seen_content.add(identity.content_sha256)
