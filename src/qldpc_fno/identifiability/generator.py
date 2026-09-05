"""Scalar stationary and clipped-AR sequence generation."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Protocol

import numpy as np
from scipy import sparse

from qldpc_fno.campaign.code_identity import sparse_binary_sha256, validate_campaign_code_identity
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.config import IdentifiabilityConfig
from qldpc_fno.identifiability.seeds import identifiability_seed
from qldpc_fno.identifiability.types import (
    ContemporaneousOracleInput,
    DeployableHistory,
    GeneratedSequence,
    LatentHistoryOracleInput,
    SequenceIdentity,
    TrainingTargets,
)


class SamplingCode(Protocol):
    name: str
    ell: int
    n: int
    k: int
    hx: sparse.spmatrix
    hz: sparse.spmatrix


_CANONICAL_LOGICAL_X_SHA256 = "ae804e39bb61a745aa08875a3c8c1d004e2e493e8f472b86eb206c6c80e6501a"


@lru_cache(maxsize=1)
def _canonical_logical_x() -> sparse.csr_matrix:
    canonical = build_self_lifted_product(PAPER_LP_3_7_16)
    logical_x = logical_x_basis(canonical.hx, canonical.hz).astype(np.uint8).tocsr()
    if sparse_binary_sha256(logical_x) != _CANONICAL_LOGICAL_X_SHA256:
        raise ValueError("logical-X identity is not canonical")
    return logical_x


def _validate_code(code: SamplingCode) -> sparse.csr_matrix:
    validate_campaign_code_identity(
        {"name": code.name, "ell": code.ell, "n": code.n, "k": code.k},
        code.hx,
        code.hz,
    )
    return _canonical_logical_x()


def _global_process(
    config: IdentifiabilityConfig,
    *,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    rounds = config.rounds.burn_in + config.rounds.scored
    global_log_odds = np.zeros(rounds, dtype=np.float64)
    if regime == "stationary_iid":
        return global_log_odds
    for round_index in range(1, rounds):
        innovation = rng.normal(0.0, config.dynamics.innovation_std)
        global_log_odds[round_index] = np.clip(
            config.dynamics.ar_coefficient * global_log_odds[round_index - 1] + innovation,
            -config.dynamics.clip,
            config.dynamics.clip,
        )
    return global_log_odds


def _content_sha256(
    *,
    regime: str,
    role: str,
    sequence_index: int,
    latent_seed: int,
    bernoulli_seed: int,
    arrays: tuple[np.ndarray, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"qldpc-fno/temporal-identifiability/payload/v1:{regime}:{role}:"
        f"{sequence_index}:{latent_seed}:{bernoulli_seed}".encode()
    )
    for array in arrays:
        canonical = np.ascontiguousarray(array)
        digest.update(canonical.dtype.str.encode())
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def generate_scalar_sequence(
    config: IdentifiabilityConfig,
    *,
    regime: str,
    role: str,
    sequence_index: int,
    code: SamplingCode,
) -> GeneratedSequence:
    """Generate one canonical scalar sequence from identity-derived RNG streams."""
    config.validate()
    if regime not in config.regimes:
        raise ValueError(f"unsupported identifiability regime: {regime}")
    if role not in config.roles:
        raise ValueError(f"unsupported identifiability role: {role}")
    logical_x = _validate_code(code)
    latent_seed = identifiability_seed(
        config,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
        stream="latent",
    )
    bernoulli_seed = identifiability_seed(
        config,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
        stream="bernoulli",
    )
    global_log_odds = _global_process(
        config,
        regime=regime,
        rng=np.random.default_rng(latent_seed),
    )
    base_logit = math.log(
        config.dynamics.base_probability / (1.0 - config.dynamics.base_probability)
    )
    scalar_probabilities = 1.0 / (1.0 + np.exp(-(base_logit + global_log_odds)))
    scalar_probabilities = np.clip(
        scalar_probabilities,
        config.dynamics.probability_clip[0],
        config.dynamics.probability_clip[1],
    )
    probabilities = np.broadcast_to(
        scalar_probabilities[:, None],
        (global_log_odds.shape[0], code.n),
    ).copy()
    if regime == "stationary_iid":
        probabilities.fill(config.dynamics.base_probability)

    errors = (
        np.random.default_rng(bernoulli_seed).random(probabilities.shape) < probabilities
    ).astype(np.uint8)
    syndromes = np.asarray(errors @ code.hx.T, dtype=np.uint8) % 2
    logical_flips = np.asarray(errors @ logical_x.T, dtype=np.uint8) % 2
    scored_mask = np.arange(global_log_odds.shape[0]) >= config.rounds.burn_in
    content_sha256 = _content_sha256(
        regime=regime,
        role=role,
        sequence_index=sequence_index,
        latent_seed=latent_seed,
        bernoulli_seed=bernoulli_seed,
        arrays=(global_log_odds, probabilities, errors, syndromes, logical_flips, scored_mask),
    )
    return GeneratedSequence(
        identity=SequenceIdentity(
            regime=regime,
            role=role,
            sequence_index=sequence_index,
            latent_seed=latent_seed,
            bernoulli_seed=bernoulli_seed,
            content_sha256=content_sha256,
        ),
        deployable=DeployableHistory(syndromes=syndromes, scored_mask=scored_mask),
        latent_oracle=LatentHistoryOracleInput(global_log_odds=global_log_odds),
        contemporaneous_oracle=ContemporaneousOracleInput(probabilities=probabilities),
        targets=TrainingTargets(errors=errors, logical_flips=logical_flips),
    )
