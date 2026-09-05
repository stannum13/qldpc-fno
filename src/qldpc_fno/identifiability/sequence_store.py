"""Immutable role-separated publication for temporal-identifiability sequences.

The public functions in this module deliberately accept only the canonical
configuration.  ``SequenceStoreDependencies`` exists solely to make the
in-process integration tests fast; the command-line entry point never exposes
an alternate or reduced experiment mode.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.code_identity import sparse_binary_sha256
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
from qldpc_fno.identifiability.config import IdentifiabilityConfig, load_identifiability_config
from qldpc_fno.identifiability.generator import SamplingCode, generate_scalar_sequence
from qldpc_fno.identifiability.observation import (
    DisjointChecks,
    greedy_disjoint_rows,
    run_fisher_precheck,
)
from qldpc_fno.identifiability.seeds import identifiability_seed
from qldpc_fno.identifiability.types import (
    DevelopmentPartitions,
    GeneratedSequence,
    SequenceIdentity,
)

_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "temporal_identifiability_role_sequences"
_ROLES = ("train", "validation", "calibration", "test")
_DEVELOPMENT_ROLES = ("train", "validation", "calibration")
_ARRAY_NAMES = (
    "global_log_odds",
    "probabilities",
    "errors",
    "syndromes",
    "logical_flips",
    "scored_mask",
)
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "complete",
        "source",
        "config",
        "code",
        "retained_checks",
        "roles",
        "fisher_precheck",
        "approval",
        "development_record",
        "sequences",
        "identity_sha256",
        "content_sha256",
    }
)
_SEQUENCE_FIELDS = frozenset(
    {
        "regime",
        "role",
        "sequence_index",
        "path",
        "seeds",
        "sequence_content_sha256",
        "payload_sha256",
        "arrays",
    }
)
_FISHER_FIELDS = frozenset(
    {
        "status",
        "provenance",
        "minimum_information",
        "median_information",
        "maximum_information",
        "cramer_rao_minimum",
        "cramer_rao_median",
        "cramer_rao_maximum",
        "maximum_derivative_error",
        "failure_reasons",
    }
)
_COMMIT = re.compile(r"[0-9a-f]{40}").fullmatch
_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch
_RUN_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "decision",
        "reviewer",
        "reviewed_at",
        "source_commit",
        "source_tree_sha256",
        "config_sha256",
        "code_sha256",
        "retained_checks_sha256",
        "development_record_sha256",
        "development_identity_sha256",
        "bundle_integrity_sha256",
        "bundle_metadata_sha256",
        "bundle_arrays_sha256",
    }
)
_RUN_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "complete",
        "status",
        "mode",
        "claim",
        "source",
        "config",
        "code",
        "retained_checks",
        "sequences",
        "fisher_precheck",
        "development_record",
        "approval",
        "bundle",
        "completed_stages",
        "evidence",
        "inference",
        "decision",
        "decoder",
        "runtime",
        "identity_sha256",
        "content_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class RepositoryEvidence:
    """Clean repository identity required for a scientific publication."""

    root: Path
    commit: str


@dataclass(frozen=True, slots=True)
class SequenceStoreDependencies:
    """Internal injection points used by fast integration tests only."""

    code_factory: Callable[[], SamplingCode] = lambda: build_self_lifted_product(PAPER_LP_3_7_16)
    sequence_factory: Callable[..., GeneratedSequence] = generate_scalar_sequence
    fisher_precheck: Callable[..., object] = run_fisher_precheck
    repository_evidence: Callable[[], RepositoryEvidence] | None = None
    logical_x_factory: Callable[[sparse.spmatrix, sparse.spmatrix], sparse.spmatrix] = logical_x_basis
    require_canonical_code: bool = True
    read_run_bundle: Callable[[Path, object], object] | None = None


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _require_exact_fields(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _repository_evidence() -> RepositoryEvidence:
    root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("temporal identifiability publication requires a clean source tree")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if _COMMIT(commit) is None:
        raise RuntimeError("unable to establish exact Git source identity")
    return RepositoryEvidence(root=root, commit=commit)


def _dependencies(dependencies: SequenceStoreDependencies | None) -> SequenceStoreDependencies:
    return dependencies or SequenceStoreDependencies(repository_evidence=_repository_evidence)


def _evidence(dependencies: SequenceStoreDependencies) -> RepositoryEvidence:
    provider = dependencies.repository_evidence or _repository_evidence
    evidence = provider()
    if not isinstance(evidence, RepositoryEvidence) or _COMMIT(evidence.commit) is None:
        raise RuntimeError("repository evidence must contain a clean 40-character source commit")
    return evidence


def _parse_roles(roles: Sequence[str]) -> tuple[str, ...]:
    parsed = tuple(role for part in roles for role in part.split(",") if role)
    if not parsed or any(role not in _ROLES for role in parsed):
        raise ValueError(f"roles must be a nonempty subset of {_ROLES!r}")
    if len(set(parsed)) != len(parsed):
        raise ValueError("roles must not overlap or repeat")
    return tuple(role for role in _ROLES if role in parsed)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical.dtype.str.encode())
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _sequence_arrays(sequence: GeneratedSequence) -> dict[str, np.ndarray]:
    return {
        "global_log_odds": np.asarray(sequence.latent_oracle.global_log_odds),
        "probabilities": np.asarray(sequence.contemporaneous_oracle.probabilities),
        "errors": np.asarray(sequence.targets.errors),
        "syndromes": np.asarray(sequence.deployable.syndromes),
        "logical_flips": np.asarray(sequence.targets.logical_flips),
        "scored_mask": np.asarray(sequence.deployable.scored_mask),
    }


def _sequence_content_sha256(
    *,
    regime: str,
    role: str,
    sequence_index: int,
    latent_seed: int,
    bernoulli_seed: int,
    arrays: Mapping[str, np.ndarray],
) -> str:
    """Recompute the generator's ordered array identity digest."""
    digest = hashlib.sha256()
    digest.update(
        f"qldpc-fno/temporal-identifiability/payload/v1:{regime}:{role}:"
        f"{sequence_index}:{latent_seed}:{bernoulli_seed}".encode()
    )
    for name in _ARRAY_NAMES:
        array = np.ascontiguousarray(arrays[name])
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(np.asarray(arrays[name]).shape),
            "dtype": np.asarray(arrays[name]).dtype.str,
            "sha256": _array_sha256(np.asarray(arrays[name])),
        }
        for name in _ARRAY_NAMES
    }


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Make a stable npz payload (including fixed zip timestamps)."""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in _ARRAY_NAMES:
            raw = io.BytesIO()
            np.lib.format.write_array(raw, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _code_manifest(
    code: SamplingCode,
    dependencies: SequenceStoreDependencies,
    config: IdentifiabilityConfig,
) -> dict[str, object]:
    logical_x = dependencies.logical_x_factory(code.hx, code.hz)
    value = {
        "name": code.name,
        "ell": int(code.ell),
        "n": int(code.n),
        "k": int(code.k),
        "hx_sha256": sparse_binary_sha256(code.hx),
        "hz_sha256": sparse_binary_sha256(code.hz),
        "logical_x_sha256": sparse_binary_sha256(logical_x),
    }
    if dependencies.require_canonical_code:
        expected = {
            "name": config.code.name,
            "ell": config.code.ell,
            "n": config.code.n,
            "k": config.code.k,
            "hx_sha256": config.code.hx_sha256,
            "hz_sha256": config.code.hz_sha256,
        }
        if {key: value[key] for key in expected} != expected:
            raise ValueError("sequence publication requires the canonical configured code")
    return value


def _checks_manifest(checks: DisjointChecks) -> dict[str, object]:
    value = {
        "algorithm_version": checks.algorithm_version,
        "matrix_sha256": checks.matrix_sha256,
        "row_indices": checks.row_indices.tolist(),
        "supports": [support.tolist() for support in checks.supports],
        "weights": checks.weights.tolist(),
        "covered_qubits": checks.covered_qubits.tolist(),
    }
    value["content_sha256"] = _digest(value)
    return value


def _fisher_manifest(report: object, config: IdentifiabilityConfig) -> dict[str, object]:
    """Serialize the complete precheck record without inventing any values."""
    if isinstance(report, Mapping):
        value = dict(report)
    else:
        provenance = getattr(report, "provenance", None)
        value = {
            "status": getattr(report, "status", None),
            "provenance": {
                "domain": getattr(provenance, "domain", None),
                "seed": getattr(provenance, "seed", None),
                "law": getattr(provenance, "law", None),
                "draws": getattr(provenance, "draws", None),
            },
            "minimum_information": getattr(report, "minimum_information", None),
            "median_information": getattr(report, "median_information", None),
            "maximum_information": getattr(report, "maximum_information", None),
            "cramer_rao_minimum": getattr(report, "cramer_rao_minimum", None),
            "cramer_rao_median": getattr(report, "cramer_rao_median", None),
            "cramer_rao_maximum": getattr(report, "cramer_rao_maximum", None),
            "maximum_derivative_error": getattr(report, "maximum_derivative_error", None),
            "failure_reasons": list(getattr(report, "failure_reasons", ())),
        }
    return _validate_fisher(value, config)


def _sequence_relative(role: str, regime: str, index: int) -> Path:
    return Path(role) / regime / f"sequence-{index:05d}.npz"


def _expected_coordinates(config: IdentifiabilityConfig, roles: tuple[str, ...]) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (role, regime, index)
        for role in roles
        for regime in config.regimes
        for index in range(int(getattr(config.splits, role)))
    )


def _sequence_record(
    sequence: GeneratedSequence,
    arrays: Mapping[str, np.ndarray],
    relative: Path,
    payload: bytes,
    config: IdentifiabilityConfig,
) -> dict[str, object]:
    identity = sequence.identity
    expected_latent = identifiability_seed(
        config,
        regime=identity.regime,
        role=identity.role,
        sequence_index=identity.sequence_index,
        stream="latent",
    )
    expected_bernoulli = identifiability_seed(
        config,
        regime=identity.regime,
        role=identity.role,
        sequence_index=identity.sequence_index,
        stream="bernoulli",
    )
    if identity.latent_seed != expected_latent or identity.bernoulli_seed != expected_bernoulli:
        raise ValueError("generated sequence seeds are not bound to its role identity")
    if identity.content_sha256 is None:
        raise ValueError("generated sequence must be content-bound")
    if identity.content_sha256 != _sequence_content_sha256(
        regime=identity.regime,
        role=identity.role,
        sequence_index=identity.sequence_index,
        latent_seed=identity.latent_seed,
        bernoulli_seed=identity.bernoulli_seed,
        arrays=arrays,
    ):
        raise ValueError("generated sequence content hash does not match its arrays and seed streams")
    return {
        "regime": identity.regime,
        "role": identity.role,
        "sequence_index": identity.sequence_index,
        "path": str(relative),
        "seeds": {
            "latent": identity.latent_seed,
            "bernoulli": identity.bernoulli_seed,
            "filter": identifiability_seed(
                config,
                regime=identity.regime,
                role=identity.role,
                sequence_index=identity.sequence_index,
                stream="filter",
            ),
        },
        "sequence_content_sha256": identity.content_sha256,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "arrays": _array_manifest(arrays),
    }


def _identity_input(root: Mapping[str, object]) -> dict[str, object]:
    return {
        key: root[key]
        for key in (
            "schema_version",
            "artifact_kind",
            "source",
            "config",
            "code",
            "retained_checks",
            "roles",
            "fisher_precheck",
            "approval",
            "development_record",
        )
    } | {
        "sequences": [
            {
                key: row[key]
                for key in ("regime", "role", "sequence_index", "path", "seeds", "sequence_content_sha256")
            }
            for row in root["sequences"]  # type: ignore[index]
        ]
    }


def _content_input(root: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        {
            "path": row["path"],
            "payload_sha256": row["payload_sha256"],
            "arrays": row["arrays"],
        }
        for row in root["sequences"]  # type: ignore[index]
    ]


def _approval_binding(approval_path: Path, development_record: Path) -> dict[str, object]:
    try:
        approval = json.loads(approval_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("manual approval record is unreadable") from error
    development = _load_root(development_record / "manifest.json")
    if development.get("artifact_kind") == "temporal_identifiability_run":
        value = _require_exact_fields(
            approval,
            _RUN_APPROVAL_FIELDS,
            "manual approval record",
        )
        try:
            bundle = development["bundle"]
            source = development["source"]
            config = development["config"]
            retained = development["retained_checks"]
            expected = {
                "schema_version": 1,
                "kind": "temporal_identifiability_manual_approval",
                "decision": "APPROVE",
                "reviewer": value["reviewer"],
                "reviewed_at": value["reviewed_at"],
                "source_commit": source["commit"],
                "source_tree_sha256": source["tree_sha256"],
                "config_sha256": config["path_sha256"],
                "code_sha256": _digest(development["code"]),
                "retained_checks_sha256": retained["content_sha256"],
                "development_record_sha256": sha256_file(
                    development_record / "manifest.json"
                ),
                "development_identity_sha256": development["identity_sha256"],
                "bundle_integrity_sha256": bundle["integrity_sha256"],
                "bundle_metadata_sha256": bundle["metadata_sha256"],
                "bundle_arrays_sha256": bundle["arrays_sha256"],
            }
        except (KeyError, TypeError) as error:
            raise ValueError("approved development run is incomplete") from error
        if (
            not isinstance(value["reviewer"], str)
            or not value["reviewer"].strip()
            or not isinstance(value["reviewed_at"], str)
            or not value["reviewed_at"].strip()
            or value != expected
        ):
            raise ValueError("manual approval does not bind the exact development run")
        return {
            "approval_sha256": sha256_file(approval_path),
            "development_record_sha256": value["development_record_sha256"],
            "development_identity_sha256": value["development_identity_sha256"],
            "approver": value["reviewer"],
            "approved_at": value["reviewed_at"],
        }
    value = _require_exact_fields(
        approval,
        frozenset(
            {
                "schema_version",
                "kind",
                "approved",
                "approver",
                "approved_at",
                "development_record_sha256",
                "development_identity_sha256",
            }
        ),
        "manual approval record",
    )
    if (
        value["schema_version"] != 1
        or value["kind"] != "temporal_identifiability_manual_approval"
        or value["approved"] is not True
        or not isinstance(value["approver"], str)
        or not value["approver"].strip()
        or not isinstance(value["approved_at"], str)
        or not value["approved_at"].strip()
    ):
        raise ValueError("manual approval record is not an auditable approval")
    record_hash = _require_sha(value["development_record_sha256"], "development approval hash")
    identity_hash = _require_sha(value["development_identity_sha256"], "development approval identity")
    if record_hash != sha256_file(development_record / "manifest.json"):
        raise ValueError("manual approval development record hash does not match")
    return {
        "approval_sha256": sha256_file(approval_path),
        "development_record_sha256": record_hash,
        "development_identity_sha256": identity_hash,
        "approver": value["approver"],
        "approved_at": value["approved_at"],
    }


def _validate_checks(value: object, checks: DisjointChecks) -> None:
    actual = _require_exact_fields(
        value,
        frozenset(
            {
                "algorithm_version",
                "matrix_sha256",
                "row_indices",
                "supports",
                "weights",
                "covered_qubits",
                "content_sha256",
            }
        ),
        "retained checks",
    )
    expected = _checks_manifest(checks)
    if actual != expected:
        raise ValueError("retained-check construction or hash does not match canonical code")


def _validate_fisher(value: object, config: IdentifiabilityConfig) -> dict[str, Any]:
    fisher = _require_exact_fields(value, _FISHER_FIELDS, "Fisher precheck")
    if fisher["status"] not in {"passed", "precheck_failed"}:
        raise ValueError("Fisher precheck status is invalid")
    expected_provenance = {
        "domain": config.seeds.fisher_domain,
        "seed": config.seeds.fisher,
        "law": config.fisher.draw_law,
        "draws": config.fisher.draws,
    }
    if fisher["provenance"] != expected_provenance:
        raise ValueError("Fisher precheck provenance is not bound to the configuration")
    numbers = _FISHER_FIELDS - {"status", "provenance", "failure_reasons"}
    if any(type(fisher[name]) not in (int, float) or not np.isfinite(fisher[name]) for name in numbers):
        raise ValueError("Fisher precheck summary is invalid")
    if not isinstance(fisher["failure_reasons"], list) or any(
        not isinstance(reason, str) for reason in fisher["failure_reasons"]
    ):
        raise ValueError("Fisher precheck failure reasons are invalid")
    return fisher


def _validate_manifest(
    root: object,
    *,
    config: IdentifiabilityConfig,
    roles: tuple[str, ...],
    evidence: RepositoryEvidence,
    code_manifest: Mapping[str, object],
    checks: DisjointChecks,
) -> dict[str, Any]:
    manifest = _require_exact_fields(root, _ROOT_FIELDS, "completion manifest")
    if (
        manifest["schema_version"] != _SCHEMA_VERSION
        or manifest["artifact_kind"] != _ARTIFACT_KIND
        or manifest["complete"] is not True
    ):
        raise ValueError("completion manifest is not a completed temporal-identifiability artifact")
    if manifest["source"] != {"commit": evidence.commit, "clean": True}:
        raise ValueError("completion manifest source identity does not match clean source tree")
    config_binding = _require_exact_fields(
        manifest["config"], frozenset({"sha256", "identity_sha256"}), "configuration binding"
    )
    expected_config = {
        "sha256": _require_sha(config_binding["sha256"], "configuration hash"),
        "identity_sha256": _digest(config_to_dict(config)),
    }
    if config_binding != expected_config:
        raise ValueError("completion manifest configuration binding is invalid")
    if manifest["code"] != dict(code_manifest):
        raise ValueError("completion manifest code identity is invalid")
    _validate_checks(manifest["retained_checks"], checks)
    if manifest["roles"] != list(roles):
        raise ValueError("completion manifest role membership is invalid")
    _validate_fisher(manifest["fisher_precheck"], config)
    if not isinstance(manifest["sequences"], list):
        raise TypeError("completion manifest sequence list is invalid")
    actual_coordinates: list[tuple[object, object, object]] = []
    paths: set[str] = set()
    for row in manifest["sequences"]:
        sequence = _require_exact_fields(row, _SEQUENCE_FIELDS, "sequence manifest entry")
        actual_coordinates.append((sequence["role"], sequence["regime"], sequence["sequence_index"]))
        if not isinstance(sequence["path"], str) or sequence["path"] in paths:
            raise ValueError("sequence manifest paths must be unique")
        paths.add(sequence["path"])
        if sequence["path"] != str(
            _sequence_relative(str(sequence["role"]), str(sequence["regime"]), int(sequence["sequence_index"]))
        ):
            raise ValueError("sequence manifest path is noncanonical")
        seeds = _require_exact_fields(sequence["seeds"], frozenset({"latent", "bernoulli", "filter"}), "sequence seeds")
        if any(type(seed) is not int or seed < 0 for seed in seeds.values()):
            raise ValueError("sequence seed streams are invalid")
        regime, role, index = str(sequence["regime"]), str(sequence["role"]), sequence["sequence_index"]
        if type(index) is not int:
            raise ValueError("sequence index is invalid")
        for stream in ("latent", "bernoulli", "filter"):
            if seeds[stream] != identifiability_seed(config, regime=regime, role=role, sequence_index=index, stream=stream):
                raise ValueError("sequence seed stream does not match its identity")
        _require_sha(sequence["sequence_content_sha256"], "sequence content hash")
        _require_sha(sequence["payload_sha256"], "payload hash")
        arrays = _require_exact_fields(sequence["arrays"], frozenset(_ARRAY_NAMES), "array manifest")
        for metadata in arrays.values():
            item = _require_exact_fields(metadata, frozenset({"shape", "dtype", "sha256"}), "array metadata")
            if not isinstance(item["shape"], list) or not all(type(size) is int and size >= 0 for size in item["shape"]):
                raise ValueError("array shape metadata is invalid")
            if not isinstance(item["dtype"], str):
                raise TypeError("array dtype metadata is invalid")
            _require_sha(item["sha256"], "array content hash")
    if tuple(actual_coordinates) != _expected_coordinates(config, roles):
        raise ValueError("completion manifest has missing, extra, or overlapping role sequences")
    if manifest["identity_sha256"] != _digest(_identity_input(manifest)):
        raise ValueError("completion manifest identity hash is invalid")
    if manifest["content_sha256"] != _digest(_content_input(manifest)):
        raise ValueError("completion manifest content hash is invalid")
    return manifest


def config_to_dict(config: IdentifiabilityConfig) -> dict[str, object]:
    """Convert the frozen config to the exact JSON-compatible canonical form."""
    # The public loader deliberately makes this structure immutable, and the
    # raw config-file hash below binds its source representation as well.
    from dataclasses import asdict

    value = asdict(config)
    value["regimes"] = list(config.regimes)
    for section in ("dynamics", "inference"):
        for key, item in value[section].items():
            if isinstance(item, tuple):
                value[section][key] = list(item)
    for key, item in value["baselines"].items():
        if isinstance(item, tuple):
            if key == "arm_aliases":
                value["baselines"][key] = dict(item)
            else:
                value["baselines"][key] = list(item)
    return value


def _load_root(path: Path) -> dict[str, Any]:
    try:
        root = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("completion manifest is missing or invalid") from error
    return root


def _validate_payload(path: Path, row: Mapping[str, object]) -> None:
    if not path.is_file() or sha256_file(path) != row["payload_sha256"]:
        raise ValueError("sequence payload SHA-256 mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != _ARRAY_NAMES:
                raise ValueError("sequence payload array membership is invalid")
            for name in _ARRAY_NAMES:
                array = np.asarray(archive[name])
                metadata = row["arrays"][name]  # type: ignore[index]
                if list(array.shape) != metadata["shape"] or array.dtype.str != metadata["dtype"]:
                    raise ValueError("sequence payload shape or dtype does not match manifest")
                if _array_sha256(array) != metadata["sha256"]:
                    raise ValueError("sequence payload array content hash mismatch")
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        if isinstance(error, ValueError) and "mismatch" in str(error):
            raise
        raise ValueError("sequence payload is unreadable") from error


def _run_development_partitions(root: Mapping[str, object]) -> DevelopmentPartitions:
    evidence = root["evidence"]
    if not isinstance(evidence, list):
        raise TypeError("approved development run evidence is invalid")

    def identity(row: Mapping[str, object]) -> SequenceIdentity:
        seeds = row["seeds"]
        if not isinstance(seeds, Mapping):
            raise TypeError("approved development sequence seeds are invalid")
        return SequenceIdentity(
            regime=str(row["regime"]),
            role=str(row["role"]),
            sequence_index=int(row["sequence_index"]),
            latent_seed=int(seeds["latent"]),
            bernoulli_seed=int(seeds["bernoulli"]),
            content_sha256=str(row["sequence_content_sha256"]),
        )

    return DevelopmentPartitions(
        *(tuple(identity(row) for row in evidence if row["role"] == role) for role in _DEVELOPMENT_ROLES)
    )


def _validate_run_development_record(
    development_record: Path,
    *,
    config_path: Path,
    dependencies: SequenceStoreDependencies,
    evidence: RepositoryEvidence,
    binding: Mapping[str, object],
) -> None:
    from qldpc_fno.identifiability.baseline_bundle import (
        BundleManifest,
        read_verified_bundle,
    )
    from qldpc_fno.identifiability.screen import _validate_complete_root

    config = load_identifiability_config(config_path)
    root = _require_exact_fields(
        _load_root(development_record / "manifest.json"),
        _RUN_ROOT_FIELDS,
        "development run manifest",
    )
    if (
        root["schema_version"] != 1
        or root["artifact_kind"] != "temporal_identifiability_run"
        or root["complete"] is not True
        or root["status"] != "complete"
        or root["mode"] != "development"
        or root["claim"] != "engineering_measurement_no_speed_claim"
        or root["decision"] is not None
        or root["decoder"] != "not_run_in_development"
        or root["approval"] is not None
        or root["development_record"] is not None
    ):
        raise ValueError("approved development run state is invalid")
    source = _require_exact_fields(
        root["source"], frozenset({"commit", "tree_sha256", "clean"}), "run source"
    )
    if source["commit"] != evidence.commit or source["clean"] is not True:
        raise ValueError("approved development run source is not the current clean commit")
    _require_sha(source["tree_sha256"], "run source tree hash")
    config_binding = _require_exact_fields(
        root["config"], frozenset({"path_sha256", "identity_sha256"}), "run config"
    )
    expected_config_identity = _digest(dataclasses.asdict(config))
    if config_binding != {
        "path_sha256": sha256_file(config_path),
        "identity_sha256": expected_config_identity,
    }:
        raise ValueError("approved development run config is not current")
    code = dependencies.code_factory()
    if root["code"] != _code_manifest(code, dependencies, config):
        raise ValueError("approved development run code identity is not current")
    checks = greedy_disjoint_rows(code.hx)
    _validate_checks(root["retained_checks"], checks)
    fisher = _fisher_manifest(dependencies.fisher_precheck(config, checks), config)
    if fisher["status"] != "passed" or root["fisher_precheck"] != fisher:
        raise ValueError("approved development run Fisher evidence does not replay")
    _validate_complete_root(
        root,
        output_dir=development_record,
        expected_mode="development",
        source_payload=source,
        config_payload=config_binding,
        code_payload=root["code"],
        checks_payload=root["retained_checks"],
        config=config,
    )
    if (
        root["identity_sha256"] != binding["development_identity_sha256"]
        or sha256_file(development_record / "manifest.json")
        != binding["development_record_sha256"]
    ):
        raise ValueError("approved development binding does not match run")
    bundle = _require_exact_fields(
        root["bundle"],
        frozenset(
            {
                "path",
                "schema_version",
                "metadata_sha256",
                "arrays_sha256",
                "integrity_sha256",
                "array_names",
            }
        ),
        "development run bundle",
    )
    if bundle["path"] != "estimator-bundle" or not isinstance(bundle["array_names"], list):
        raise ValueError("development run bundle layout is invalid")
    manifest = BundleManifest(
        bundle["schema_version"],
        _require_sha(bundle["metadata_sha256"], "bundle metadata hash"),
        _require_sha(bundle["arrays_sha256"], "bundle arrays hash"),
        _require_sha(bundle["integrity_sha256"], "bundle integrity hash"),
        tuple(bundle["array_names"]),
    )
    reader = dependencies.read_run_bundle or read_verified_bundle
    loaded = reader(development_record / "estimator-bundle", manifest)
    if loaded.integrity_sha256 != manifest.integrity_sha256:
        raise ValueError("approved development bundle refit disagrees with stored state")
    if (
        _digest(getattr(loaded, "config_payload", None)) != _digest(dataclasses.asdict(config))
        or getattr(loaded, "partitions", None) != _run_development_partitions(root)
    ):
        raise ValueError("approved development bundle config or partitions do not match run")


def _validate_development_record(
    development_record: Path,
    approval_path: Path,
    *,
    config_path: Path,
    dependencies: SequenceStoreDependencies,
    evidence: RepositoryEvidence,
) -> dict[str, object]:
    binding = _approval_binding(approval_path, development_record)
    candidate = _load_root(development_record / "manifest.json")
    if candidate.get("artifact_kind") == "temporal_identifiability_run":
        _validate_run_development_record(
            development_record,
            config_path=config_path,
            dependencies=dependencies,
            evidence=evidence,
            binding=binding,
        )
        return binding
    development = _verify(
        config_path=config_path,
        output_dir=development_record,
        roles=_DEVELOPMENT_ROLES,
        dependencies=dependencies,
        evidence=evidence,
        approval_path=None,
        development_record=None,
    )
    if development["identity_sha256"] != binding["development_identity_sha256"]:
        raise ValueError("manual approval development identity does not match")
    if development["fisher_precheck"]["status"] != "passed":
        raise ValueError("test-role generation requires a passed Fisher precheck in development record")
    return binding


def _verify(
    *,
    config_path: Path,
    output_dir: Path,
    roles: tuple[str, ...],
    dependencies: SequenceStoreDependencies,
    evidence: RepositoryEvidence,
    approval_path: Path | None,
    development_record: Path | None,
) -> dict[str, Any]:
    config = load_identifiability_config(config_path)
    code = dependencies.code_factory()
    code_manifest = _code_manifest(code, dependencies, config)
    checks = greedy_disjoint_rows(code.hx)
    fisher = _fisher_manifest(dependencies.fisher_precheck(config, checks), config)
    test_requested = "test" in roles
    binding: dict[str, object] | None = None
    # This must run before opening the test completion marker or test payloads.
    if test_requested:
        if approval_path is None or development_record is None:
            raise ValueError("test-role verification requires manual approval and development record")
        if fisher["status"] != "passed":
            raise ValueError("test-role verification requires a passed current Fisher precheck")
        binding = _validate_development_record(
            development_record,
            approval_path,
            config_path=config_path,
            dependencies=dependencies,
            evidence=evidence,
        )
    root = _validate_manifest(
        _load_root(output_dir / "manifest.json"),
        config=config,
        roles=roles,
        evidence=evidence,
        code_manifest=code_manifest,
        checks=checks,
    )
    if root["fisher_precheck"] != fisher:
        raise ValueError("Fisher precheck report does not exactly match independent replay")
    if root["config"]["sha256"] != sha256_file(config_path):
        raise ValueError("completion manifest exact config-file hash does not match")
    if test_requested:
        assert binding is not None
        if root["approval"] != binding or root["development_record"] != {
            "sha256": binding["development_record_sha256"],
            "identity_sha256": binding["development_identity_sha256"],
        }:
            raise ValueError("test-role approval binding does not match its completed manifest")
    elif root["approval"] is not None or root["development_record"] is not None:
        raise ValueError("development-role publication must not carry test approval bindings")
    declared = {"manifest.json"}
    for row in root["sequences"]:
        relative = Path(row["path"])
        _validate_payload(output_dir / relative, row)
        declared.add(str(relative))
        sequence = dependencies.sequence_factory(
            config,
            regime=row["regime"],
            role=row["role"],
            sequence_index=row["sequence_index"],
            code=code,
        )
        arrays = _sequence_arrays(sequence)
        regenerated = _sequence_record(sequence, arrays, relative, _npz_bytes(arrays), config)
        if regenerated != row:
            raise ValueError("sequence regeneration is not byte-identical to the published record")
    actual = {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    if actual != declared:
        raise ValueError("sequence publication contains missing or undeclared files")
    return root


def generate_campaign(
    *,
    config_path: Path,
    output_dir: Path,
    roles: Sequence[str],
    approval_path: Path | None = None,
    development_record: Path | None = None,
    dependencies: SequenceStoreDependencies | None = None,
) -> dict[str, Any]:
    """Publish role-isolated payloads and the completion manifest last."""
    selected_roles = _parse_roles(roles)
    deps = _dependencies(dependencies)
    evidence = _evidence(deps)
    config = load_identifiability_config(config_path)
    test_requested = "test" in selected_roles
    if test_requested != (approval_path is not None and development_record is not None):
        raise ValueError("test-role generation requires both manual approval and development record")
    if not test_requested and (approval_path is not None or development_record is not None):
        raise ValueError("approval and development record are only valid for test-role generation")
    code = deps.code_factory()
    code_manifest = _code_manifest(code, deps, config)
    checks = greedy_disjoint_rows(code.hx)
    if deps.require_canonical_code and (len(checks.supports), int(checks.weights.sum())) != (135, 1350):
        raise ValueError("canonical retained rows no longer match preregistered count and coverage")
    fisher = _fisher_manifest(deps.fisher_precheck(config, checks), config)
    binding: dict[str, object] | None = None
    if test_requested:
        # Do not read or create test output until both Fisher gates and the
        # independently replayed development record are valid.
        if fisher["status"] != "passed":
            raise ValueError("test-role generation requires a passed current Fisher precheck")
        binding = _validate_development_record(
            development_record,
            approval_path,
            config_path=config_path,
            dependencies=deps,
            evidence=evidence,
        )
    if output_dir.exists():
        if not (output_dir / "manifest.json").is_file():
            raise FileExistsError("refuse to overwrite incomplete sequence publication")
        try:
            existing = _verify(
                config_path=config_path,
                output_dir=output_dir,
                roles=selected_roles,
                dependencies=deps,
                evidence=evidence,
                approval_path=approval_path,
                development_record=development_record,
            )
        except (TypeError, ValueError) as error:
            raise FileExistsError("refuse to overwrite completed differing or corrupted publication") from error
        return existing
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent))
    try:
        records: list[dict[str, object]] = []
        for role in selected_roles:
            role_staging = staging / f".{role}-staging"
            for _, regime, index in _expected_coordinates(config, (role,)):
                sequence = deps.sequence_factory(config, regime=regime, role=role, sequence_index=index, code=code)
                arrays = _sequence_arrays(sequence)
                relative = _sequence_relative(role, regime, index)
                payload = _npz_bytes(arrays)
                _write_atomic(role_staging / relative.relative_to(role), payload)
                records.append(_sequence_record(sequence, arrays, relative, payload, config))
            os.replace(role_staging, staging / role)
        root: dict[str, object] = {
            "schema_version": _SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "complete": True,
            "source": {"commit": evidence.commit, "clean": True},
            "config": {"sha256": sha256_file(config_path), "identity_sha256": _digest(config_to_dict(config))},
            "code": code_manifest,
            "retained_checks": _checks_manifest(checks),
            "roles": list(selected_roles),
            "fisher_precheck": fisher,
            "approval": binding,
            "development_record": (
                {"sha256": binding["development_record_sha256"], "identity_sha256": binding["development_identity_sha256"]}
                if binding is not None
                else None
            ),
            "sequences": records,
        }
        root["identity_sha256"] = _digest(_identity_input(root))
        root["content_sha256"] = _digest(_content_input(root))
        # The root manifest is the sole completion marker and is written last.
        _write_atomic(staging / "manifest.json", _canonical_json(root))
        # Replay every staged payload before the staging directory becomes a
        # visible completed artifact. A nondeterministic factory therefore
        # cannot leave a final manifest behind.
        verified = _verify(
            config_path=config_path,
            output_dir=staging,
            roles=selected_roles,
            dependencies=deps,
            evidence=evidence,
            approval_path=approval_path,
            development_record=development_record,
        )
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verified


def verify_campaign(
    *,
    config_path: Path,
    output_dir: Path,
    roles: Sequence[str],
    approval_path: Path | None = None,
    development_record: Path | None = None,
    dependencies: SequenceStoreDependencies | None = None,
) -> dict[str, Any]:
    """Verify immutable manifest bindings, hashes, and deterministic replay."""
    selected_roles = _parse_roles(roles)
    deps = _dependencies(dependencies)
    return _verify(
        config_path=config_path,
        output_dir=output_dir,
        roles=selected_roles,
        dependencies=deps,
        evidence=_evidence(deps),
        approval_path=approval_path,
        development_record=development_record,
    )
