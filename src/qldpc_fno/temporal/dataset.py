"""Immutable, content-addressed artifacts for causal temporal sequences."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.code_identity import (
    sparse_binary_sha256,
    validate_campaign_code_identity,
)
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


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
) -> None:
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
    if observed.syndromes.shape != (rounds, observed.syndrome_channels, observed.ell):
        raise ValueError("observed syndrome geometry is inconsistent")
    if supervision.errors.ndim != 3 or supervision.errors.shape[2] != observed.ell:
        raise ValueError("supervision error geometry is inconsistent")
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
    }
    for name, (array, expected) in expected_dtypes.items():
        if array.dtype != expected:
            raise ValueError(f"{name} dtype must be {expected}")


def write_sequence(
    path: Path,
    observed: CausalObservedSequence,
    supervision: CausalSupervision,
    diagnostics: SimulatorDiagnostics,
    manifest: Mapping[str, object],
) -> None:
    """Publish payloads atomically and publish the completion manifest last."""
    target = Path(path)
    _validate_sequence_contract(observed, supervision, diagnostics)
    if set(manifest) != IDENTITY_FIELDS:
        raise ValueError("uncompleted sequence manifest has missing or unknown fields")
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
    if not isinstance(manifest, dict) or manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError("unsupported or missing sequence artifact version")
    required = IDENTITY_FIELDS | {"payloads"}
    if set(manifest) != required:
        raise ValueError("sequence manifest has missing or unknown fields")
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


def read_verified_sequence(
    path: Path,
) -> tuple[CausalObservedSequence, CausalSupervision, SimulatorDiagnostics, dict[str, object]]:
    """Verify all hashes and array contracts before constructing immutable sequence types."""
    target = Path(path)
    manifest = _load_manifest(target)
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
    _validate_sequence_contract(observed, supervision, diagnostics)
    return observed, supervision, diagnostics, manifest


def _verify_regeneration_identity(
    manifest: Mapping[str, object], config: CausalExperimentConfig, code: SamplingCode
) -> tuple[str, str, int, int]:
    expected_config_hash = _mapping_sha256(config.to_dict())
    if manifest.get("config_sha256") != expected_config_hash:
        raise ValueError("configuration hash does not match sequence artifact")
    code_manifest = manifest.get("code")
    if not isinstance(code_manifest, Mapping):
        raise ValueError("code identity is missing from sequence manifest")  # noqa: TRY004
    if (
        code_manifest.get("hx_sha256") != sparse_binary_sha256(code.hx)
        or code_manifest.get("hz_sha256") != sparse_binary_sha256(code.hz)
    ):
        raise ValueError("code matrix hash does not match sequence artifact")
    identity = manifest.get("identity")
    seeds = manifest.get("seeds")
    if not isinstance(identity, Mapping) or not isinstance(seeds, Mapping):
        raise ValueError("sequence identity or seeds are missing")  # noqa: TRY004
    regime = identity.get("regime")
    role = identity.get("role")
    sequence_index = identity.get("sequence_index")
    if not isinstance(regime, str) or not isinstance(role, str) or type(sequence_index) is not int:
        raise ValueError("sequence identity is invalid")
    expected_seeds = sequence_seed_tuple(
        config.campaign_seed,
        regime=regime,
        role=role,
        sequence_index=sequence_index,
    )
    if seeds != {"latent": expected_seeds.latent, "bernoulli": expected_seeds.bernoulli}:
        raise ValueError("sequence seed tuple does not match identity")
    return regime, role, sequence_index, expected_seeds.bernoulli


def regenerate_and_verify(
    path: Path, config: CausalExperimentConfig, code: SamplingCode
) -> None:
    """Regenerate a sequence and require byte-identical payloads and manifest."""
    target = Path(path)
    _, _, _, manifest = read_verified_sequence(target)
    regime, role, sequence_index, bernoulli_seed = _verify_regeneration_identity(
        manifest, config, code
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
