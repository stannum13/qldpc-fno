"""Immutable, content-addressed artifacts for causal temporal sequences."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.code_identity import (
    sparse_binary_sha256,
    validate_campaign_code_identity,
)
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import CSSCode, build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.temporal.config import CausalExperimentConfig
from qldpc_fno.temporal.generator import (
    CausalObservedSequence,
    CausalSupervision,
    LatentSequence,
    SamplingCode,
    SimulatorDiagnostics,
    generate_latent_sequence,
    sample_sequence,
)
from qldpc_fno.temporal.seeds import sequence_seed_tuple

ARTIFACT_VERSION = 1
GENERATOR_VERSION = "causal-code-capacity-v1"
PAYLOAD_NAMES = ("observed.npz", "supervision.npz", "diagnostics.npz")
DECLARED_FILES = frozenset((*PAYLOAD_NAMES, "manifest.json"))
IDENTITY_FIELDS = frozenset(
    {
        "artifact_version",
        "artifact_mode",
        "generator_version",
        "source_commit",
        "config_sha256",
        "code",
        "identity",
        "seeds",
    }
)
CANONICAL_CODE_FIELDS = frozenset({"name", "ell", "n", "k", "hx_sha256", "hz_sha256"})
CANONICAL_CODE_IDENTITY = {"name": "lp_3_7_16", "ell": 45, "n": 2610, "k": 744}


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@lru_cache(maxsize=1)
def _canonical_validation_material() -> tuple[CSSCode, object]:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    return code, logical_x_basis(code.hx, code.hz)


def _validate_manifest_identity(manifest: Mapping[str, object], *, completed: bool) -> None:
    expected_fields = IDENTITY_FIELDS | ({"payloads"} if completed else set())
    if set(manifest) != expected_fields:
        label = "completed" if completed else "uncompleted"
        raise ValueError(f"{label} sequence manifest has missing or unknown fields")
    if type(manifest.get("artifact_version")) is not int or manifest["artifact_version"] != 1:
        raise ValueError("unsupported or missing sequence artifact version")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("sequence manifest generator version is invalid")

    code_identity = manifest.get("code")
    if not isinstance(code_identity, Mapping) or set(code_identity) != CANONICAL_CODE_FIELDS:
        raise ValueError("manifest must contain the canonical code identity")
    if any(
        type(code_identity.get(key)) is not type(value) or code_identity.get(key) != value
        for key, value in CANONICAL_CODE_IDENTITY.items()
    ):
        raise ValueError("manifest must contain the canonical code identity")
    canonical_code, _ = _canonical_validation_material()
    expected_hashes = {
        "hx_sha256": sparse_binary_sha256(canonical_code.hx),
        "hz_sha256": sparse_binary_sha256(canonical_code.hz),
    }
    if any(
        type(code_identity.get(key)) is not str or code_identity.get(key) != value
        for key, value in expected_hashes.items()
    ):
        raise ValueError("manifest canonical code identity has invalid matrix hashes")

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {"regime", "role", "sequence_index"}:
        raise ValueError("sequence identity is invalid")
    if (
        identity.get("regime") not in {
            "stationary_iid",
            "static_spatial_latent",
            "temporal_uniform",
            "joint_in_basis",
            "joint_basis_mismatch",
        }
        or identity.get("role") not in {"train", "validation", "calibration", "test"}
        or type(identity.get("sequence_index")) is not int
        or identity["sequence_index"] < 0
    ):
        raise ValueError("sequence identity is invalid")
    seeds = manifest.get("seeds")
    if (
        not isinstance(seeds, Mapping)
        or set(seeds) != {"latent", "bernoulli"}
        or any(type(seeds.get(name)) is not int or seeds[name] < 0 for name in seeds)
    ):
        raise ValueError("sequence seed tuple is invalid")


def build_sequence_manifest(
    *, config: CausalExperimentConfig, latent: LatentSequence, code: SamplingCode
) -> dict[str, object]:
    """Build the identity portion of a sequence manifest before payload publication."""
    config.validate()
    expected_seeds = sequence_seed_tuple(
        config.campaign_seed,
        regime=latent.regime,
        role=latent.role,
        sequence_index=latent.sequence_index,
    )
    if latent.seeds != expected_seeds:
        raise ValueError("latent sequence seed tuple does not match its identity")
    if (code.name, code.ell, code.n) != (
        config.code.name,
        config.code.ell,
        config.code.n,
    ):
        raise ValueError("sampling code metadata does not match configuration")
    validate_campaign_code_identity(
        {"name": code.name, "ell": code.ell, "n": code.n, "k": code.k},
        code.hx,
        code.hz,
    )
    config_payload = config.to_dict()
    return {
        "artifact_version": ARTIFACT_VERSION,
        "artifact_mode": config.artifact_mode,
        "generator_version": GENERATOR_VERSION,
        "source_commit": _source_commit(),
        "config_sha256": _mapping_sha256(config_payload),
        "code": {
            "name": code.name,
            "ell": code.ell,
            "n": code.n,
            "k": code.k,
            "hx_sha256": sparse_binary_sha256(code.hx),
            "hz_sha256": sparse_binary_sha256(code.hz),
        },
        "identity": {
            "regime": latent.regime,
            "role": latent.role,
            "sequence_index": latent.sequence_index,
        },
        "seeds": {"latent": latent.seeds.latent, "bernoulli": latent.seeds.bernoulli},
    }


def _observed_arrays(sequence: CausalObservedSequence) -> dict[str, np.ndarray]:
    return {
        "syndromes": sequence.syndromes,
        "scored_mask": sequence.scored_mask,
        "ell": np.asarray(sequence.ell, dtype=np.int64),
        "syndrome_channels": np.asarray(sequence.syndrome_channels, dtype=np.int64),
    }


def _supervision_arrays(sequence: CausalSupervision) -> dict[str, np.ndarray]:
    return {"errors": sequence.errors, "logical_flips": sequence.logical_flips}


def _diagnostic_arrays(sequence: SimulatorDiagnostics) -> dict[str, np.ndarray]:
    return {
        "probabilities": sequence.probabilities,
        "global_log_odds": sequence.global_log_odds,
        "spatial_log_odds": sequence.spatial_log_odds,
        "channel_offsets": sequence.channel_offsets,
        "event_active": sequence.event_active,
        "event_onset": sequence.event_onset,
        "event_termination": sequence.event_termination,
        "event_center": sequence.event_center,
        "event_age": sequence.event_age,
        "event_width": sequence.event_width,
        "event_step": sequence.event_step,
    }


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(np.asarray(array).shape), "dtype": np.asarray(array).dtype.str}
        for name, array in arrays.items()
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_sequence_contract(
    observed: CausalObservedSequence,
    supervision: CausalSupervision,
    diagnostics: SimulatorDiagnostics,
    manifest: Mapping[str, object],
) -> None:
    _validate_manifest_identity(manifest, completed="payloads" in manifest)
    identity = manifest["identity"]
    regime = identity["regime"]
    rounds = observed.syndromes.shape[0]
    round_dimensions = [
        observed.scored_mask.shape[0],
        supervision.errors.shape[0],
        supervision.logical_flips.shape[0],
        diagnostics.probabilities.shape[0],
        diagnostics.global_log_odds.shape[0],
        diagnostics.spatial_log_odds.shape[0],
        diagnostics.event_active.shape[0],
        diagnostics.event_onset.shape[0],
        diagnostics.event_termination.shape[0],
        diagnostics.event_center.shape[0],
        diagnostics.event_age.shape[0],
        diagnostics.event_width.shape[0],
        diagnostics.event_step.shape[0],
    ]
    if rounds <= 0 or any(value != rounds for value in round_dimensions):
        raise ValueError("sequence payload round dimensions must agree")
    if (observed.ell, observed.syndrome_channels) != (45, 21) or observed.syndromes.shape != (
        rounds,
        21,
        45,
    ):
        raise ValueError("observed syndromes must have canonical geometry (rounds,21,45)")
    if supervision.errors.shape != (rounds, 58, 45):
        raise ValueError("errors must have canonical physical-error geometry (rounds,58,45)")
    if supervision.logical_flips.shape != (rounds, 744):
        raise ValueError("logical flips must have canonical width k=744")
    if diagnostics.probabilities.shape != supervision.errors.shape:
        raise ValueError("probability and physical-error geometries must agree")
    if diagnostics.spatial_log_odds.shape != (rounds, observed.ell):
        raise ValueError("spatial diagnostic geometry is inconsistent")
    if diagnostics.channel_offsets.shape != (supervision.errors.shape[1],):
        raise ValueError("channel-offset diagnostic geometry is inconsistent")
    expected_dtypes = {
        "syndromes": (observed.syndromes, np.dtype(np.uint8)),
        "scored_mask": (observed.scored_mask, np.dtype(np.bool_)),
        "errors": (supervision.errors, np.dtype(np.uint8)),
        "logical_flips": (supervision.logical_flips, np.dtype(np.uint8)),
        "probabilities": (diagnostics.probabilities, np.dtype(np.float64)),
        "event_active": (diagnostics.event_active, np.dtype(np.bool_)),
        "event_onset": (diagnostics.event_onset, np.dtype(np.bool_)),
        "event_termination": (diagnostics.event_termination, np.dtype(np.bool_)),
        "global_log_odds": (diagnostics.global_log_odds, np.dtype(np.float64)),
        "spatial_log_odds": (diagnostics.spatial_log_odds, np.dtype(np.float64)),
        "channel_offsets": (diagnostics.channel_offsets, np.dtype(np.float64)),
        "event_center": (diagnostics.event_center, np.dtype(np.int16)),
        "event_age": (diagnostics.event_age, np.dtype(np.int16)),
        "event_width": (diagnostics.event_width, np.dtype(np.int16)),
        "event_step": (diagnostics.event_step, np.dtype(np.int8)),
    }
    for name, (array, expected) in expected_dtypes.items():
        if array.dtype != expected:
            raise ValueError(f"{name} dtype must be {expected}")

    for name, array in {
        "syndromes": observed.syndromes,
        "errors": supervision.errors,
        "logical flips": supervision.logical_flips,
    }.items():
        if not np.all((array == 0) | (array == 1)):
            raise ValueError(f"{name} must be binary")
    if not np.all(np.isfinite(diagnostics.probabilities)):
        raise ValueError("probabilities must be finite")
    if not np.all((diagnostics.probabilities >= 1e-5) & (diagnostics.probabilities <= 0.25)):
        raise ValueError("probabilities must lie within [1e-5,0.25]")
    for name, array in {
        "global log odds": diagnostics.global_log_odds,
        "spatial log odds": diagnostics.spatial_log_odds,
        "channel offsets": diagnostics.channel_offsets,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")

    mask = observed.scored_mask
    if not np.any(mask) or np.any(mask[:-1] & ~mask[1:]):
        raise ValueError("scored mask must be a nonempty suffix of complete rounds")

    active = diagnostics.event_active
    onset = diagnostics.event_onset
    termination = diagnostics.event_termination
    center = diagnostics.event_center
    age = diagnostics.event_age
    width = diagnostics.event_width
    step = diagnostics.event_step
    if np.any(onset & (~active | (age != 0))):
        raise ValueError("event onset must imply an active age-zero event")
    if np.any(termination & ~active):
        raise ValueError("event termination must occur on an active event")
    if np.any(active & ((center < 0) | (center >= 45) | (age < 0))):
        raise ValueError("active event labels have invalid center or age")
    if np.any(~active & ((center != -1) | (age != -1) | (width != 0) | (step != 0))):
        raise ValueError("inactive event labels must use canonical sentinels")
    if regime not in {"joint_in_basis", "joint_basis_mismatch"} and np.any(active):
        raise ValueError("event labels are forbidden outside joint regimes")
    if regime == "joint_in_basis" and np.any(active & ((width != 0) | (step != 0))):
        raise ValueError("smooth bursts cannot carry interval width or movement")
    if regime == "joint_basis_mismatch" and np.any(
        active & (((width < 3) | (width > 9)) | ((step < -1) | (step > 1)))
    ):
        raise ValueError("mismatch event width or movement is invalid")

    canonical_code, logical_x = _canonical_validation_material()
    errors_flat = supervision.errors.reshape(rounds, 2610)
    expected_syndromes = np.asarray(canonical_code.hx @ errors_flat.T, dtype=np.uint8).T % 2
    if not np.array_equal(observed.syndromes.reshape(rounds, 945), expected_syndromes):
        raise ValueError("syndromes do not match canonical errors and parity checks")
    expected_logicals = np.asarray(logical_x @ errors_flat.T, dtype=np.uint8).T % 2
    if not np.array_equal(supervision.logical_flips, expected_logicals):
        raise ValueError("logical flips do not match canonical errors and logical operators")


def write_sequence(
    path: Path,
    observed: CausalObservedSequence,
    supervision: CausalSupervision,
    diagnostics: SimulatorDiagnostics,
    manifest: Mapping[str, object],
) -> None:
    """Publish payloads atomically and publish the completion manifest last."""
    target = Path(path)
    _validate_sequence_contract(observed, supervision, diagnostics, manifest)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"refuse to overwrite sequence artifact: {target}") from error

    payload_arrays = {
        "observed.npz": _observed_arrays(observed),
        "supervision.npz": _supervision_arrays(supervision),
        "diagnostics.npz": _diagnostic_arrays(diagnostics),
    }
    for name in PAYLOAD_NAMES:
        _write_npz_atomic(target / name, payload_arrays[name])

    completed = dict(manifest)
    if "payloads" in completed:
        raise ValueError("uncompleted manifest must not contain payload metadata")
    completed["payloads"] = {
        name: {
            "sha256": sha256_file(target / name),
            "arrays": _array_manifest(payload_arrays[name]),
        }
        for name in PAYLOAD_NAMES
    }
    # This rename is the completion marker and deliberately occurs after every payload.
    _write_json_atomic(target / "manifest.json", completed)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file() or any(not (path / name).is_file() for name in PAYLOAD_NAMES):
        raise ValueError(f"incomplete sequence artifact: {path}")
    actual_files = {entry.name for entry in path.iterdir()}
    if actual_files != DECLARED_FILES:
        raise ValueError(f"sequence artifact contains undeclared files: {sorted(actual_files - DECLARED_FILES)}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sequence manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("sequence manifest must be an object")  # noqa: TRY004
    _validate_manifest_identity(manifest, completed=True)
    payloads = manifest.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise ValueError("sequence manifest has an invalid payload set")
    return manifest


def _load_verified_npz(
    path: Path, metadata: Mapping[str, object]
) -> dict[str, np.ndarray]:
    expected_hash = metadata.get("sha256")
    if not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
        raise ValueError(f"{path.name} SHA-256 mismatch")
    array_metadata = metadata.get("arrays")
    if not isinstance(array_metadata, Mapping):
        raise ValueError(f"{path.name} array metadata is missing")  # noqa: TRY004
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(array_metadata):
                raise ValueError(f"{path.name} array set mismatch")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "mismatch" in str(error):
            raise
        raise ValueError(f"unable to load verified payload: {path.name}") from error
    for name, array in arrays.items():
        expected = array_metadata[name]
        if not isinstance(expected, Mapping):
            raise ValueError(f"{path.name}:{name} array metadata is invalid")  # noqa: TRY004
        if list(array.shape) != expected.get("shape"):
            raise ValueError(f"{path.name}:{name} shape mismatch")
        if array.dtype.str != expected.get("dtype"):
            raise ValueError(f"{path.name}:{name} dtype mismatch")
    return arrays


def _validate_expected_provenance(
    manifest: Mapping[str, object],
    *,
    config: CausalExperimentConfig,
    code: SamplingCode,
    expected_source_commit: str | None,
) -> tuple[str, str, int, int]:
    config.validate()
    if manifest.get("artifact_mode") != config.artifact_mode:
        raise ValueError("artifact mode does not match expected configuration")
    if manifest.get("config_sha256") != _mapping_sha256(config.to_dict()):
        raise ValueError("configuration hash does not match expected configuration")

    expected_commit = _source_commit() if expected_source_commit is None else expected_source_commit
    manifest_commit = manifest.get("source_commit")
    if type(manifest_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", manifest_commit) is None:
        raise ValueError("source commit must be exactly 40 lowercase hex characters")
    if type(expected_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected source commit must be exactly 40 lowercase hex characters")
    if manifest_commit != expected_commit:
        raise ValueError("source commit does not match expected commit")

    validate_campaign_code_identity(
        {"name": code.name, "ell": code.ell, "n": code.n, "k": code.k},
        code.hx,
        code.hz,
    )
    if (code.name, code.ell, code.n, code.k) != (
        config.code.name,
        config.code.ell,
        config.code.n,
        config.code.k,
    ):
        raise ValueError("expected code does not match expected configuration")
    code_manifest = manifest["code"]
    expected_code_manifest = {
        "name": code.name,
        "ell": code.ell,
        "n": code.n,
        "k": code.k,
        "hx_sha256": sparse_binary_sha256(code.hx),
        "hz_sha256": sparse_binary_sha256(code.hz),
    }
    if code_manifest != expected_code_manifest:
        raise ValueError("manifest code identity does not match expected canonical code")

    identity = manifest["identity"]
    regime = identity["regime"]
    role = identity["role"]
    sequence_index = identity["sequence_index"]
    if regime not in config.regimes:
        raise ValueError("sequence regime is not enabled by expected configuration")
    role_sizes = {
        "train": config.splits.train,
        "validation": config.splits.validation,
        "calibration": config.splits.calibration,
        "test": config.splits.test,
    }
    if sequence_index >= role_sizes[role]:
        raise ValueError("sequence index is outside expected role membership")
    expected_seeds = sequence_seed_tuple(
        config.campaign_seed,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
    )
    if manifest["seeds"] != {
        "latent": expected_seeds.latent,
        "bernoulli": expected_seeds.bernoulli,
    }:
        raise ValueError("sequence seed tuple does not match expected configuration")
    return regime, role, sequence_index, expected_seeds.bernoulli


def read_verified_sequence(
    path: Path,
    *,
    config: CausalExperimentConfig,
    code: SamplingCode,
    expected_source_commit: str | None = None,
) -> tuple[CausalObservedSequence, CausalSupervision, SimulatorDiagnostics, dict[str, object]]:
    """Verify content and provenance against explicit expected experiment inputs."""
    target = Path(path)
    manifest = _load_manifest(target)
    _validate_expected_provenance(
        manifest,
        config=config,
        code=code,
        expected_source_commit=expected_source_commit,
    )
    loaded = {
        name: _load_verified_npz(target / name, manifest["payloads"][name])
        for name in PAYLOAD_NAMES
    }
    observed_arrays = loaded["observed.npz"]
    supervision_arrays = loaded["supervision.npz"]
    diagnostic_arrays = loaded["diagnostics.npz"]
    observed = CausalObservedSequence(
        syndromes=observed_arrays["syndromes"],
        scored_mask=observed_arrays["scored_mask"],
        ell=int(observed_arrays["ell"]),
        syndrome_channels=int(observed_arrays["syndrome_channels"]),
    )
    supervision = CausalSupervision(
        errors=supervision_arrays["errors"],
        logical_flips=supervision_arrays["logical_flips"],
    )
    diagnostics = SimulatorDiagnostics(**diagnostic_arrays)
    _validate_sequence_contract(observed, supervision, diagnostics, manifest)
    return observed, supervision, diagnostics, manifest


def regenerate_and_verify(
    path: Path, config: CausalExperimentConfig, code: SamplingCode
) -> None:
    """Regenerate a sequence and require byte-identical payloads and manifest."""
    target = Path(path)
    _, _, _, manifest = read_verified_sequence(target, config=config, code=code)
    regime, role, sequence_index, bernoulli_seed = _validate_expected_provenance(
        manifest,
        config=config,
        code=code,
        expected_source_commit=None,
    )
    latent = generate_latent_sequence(
        config,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
    )
    observed, supervision, diagnostics = sample_sequence(
        latent, bernoulli_seed=bernoulli_seed, code=code
    )
    regenerated_manifest = build_sequence_manifest(config=config, latent=latent, code=code)
    with tempfile.TemporaryDirectory(prefix="qldpc-fno-regenerate-", dir=target.parent) as temporary:
        regenerated = Path(temporary) / "sequence"
        write_sequence(
            regenerated,
            observed,
            supervision,
            diagnostics,
            regenerated_manifest,
        )
        for name in (*PAYLOAD_NAMES, "manifest.json"):
            if (target / name).read_bytes() != (regenerated / name).read_bytes():
                raise ValueError(f"regenerated artifact is not byte-identical: {name}")
