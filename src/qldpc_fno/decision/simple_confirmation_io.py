"""Provenance-bound I/O for the frozen simple planar confirmation.

Preflight is deliberately metadata-only.  The only functions allowed to open a
physical-shot domain are :func:`generate` and :func:`run`, and production use of
those functions requires an independently issued release receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import qecsim
import scipy
from qecsim.graphtools import blossom5
from qecsim.models.planar import PlanarCode

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.decision import planar_shot_data
from qldpc_fno.decision.planar_shot_data import _validate_shot_manifest
from qldpc_fno.decision.simple_confirmation import (
    load_config,
    load_shot_config,
    summarize,
    validate_result_rows,
)
from qldpc_fno.decision.simple_confirmation_runner import evaluate_shot

_CONTRACT_ID = "simple_planar_confirmation_v1"
_SCIENTIFIC_DOMAIN = "qldpc-fno/simple-planar-confirmation/v1"
_BOOTSTRAP_DOMAIN = "qldpc-fno/simple-planar-confirmation/bootstrap/v1"
_CONTRACT_PATH = "docs/adaptive-computation-simple-confirmation-contract.md"
_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/qldpc_fno/decision/simple_confirmation.py",
    "src/qldpc_fno/decision/simple_confirmation_runner.py",
    "src/qldpc_fno/decision/simple_confirmation_io.py",
    "src/qldpc_fno/decision/planar_shot_data.py",
    "src/qldpc_fno/decision/planar_shot_accuracy.py",
    "src/qldpc_fno/decision/tensor_network.py",
    "src/qldpc_fno/decision/adaptive_intervention_pilot.py",
    "src/qldpc_fno/decision/adaptive_diagnostics.py",
    "src/qldpc_fno/metrics/paired.py",
    "src/qldpc_fno/artifacts.py",
)
_RETIRED_DOMAINS = (
    "qldpc-fno/adaptive-intervention-pilot/v1",
    "qldpc-fno/adaptive-intervention-pilot/v2",
    "qldpc-fno/adaptive-intervention-pilot/test-fixture/v1",
    "qldpc-fno/simple-planar-confirmation/test-fixture/v1",
)
_RELEASE_KEYS = {
    "schema_version",
    "contract_id",
    "contract_sha256",
    "approved_commit",
    "source_sha256",
    "config_bindings",
    "runtime",
    "code_identity",
    "historical_domains",
    "review_bindings",
    "unopened_domain",
    "expected_shots",
    "decision",
}
_GENERATION_MANIFEST_KEYS = {
    "schema_version",
    "contract_id",
    "producer_commit",
    "config_bindings",
    "source_sha256",
    "runtime",
    "code_identity",
    "historical_domains",
    "release_binding",
    "shots_binding",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> object:
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _git_identity(root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _committed_digest(root: Path, commit: str, label: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{label}"],
        capture_output=True,
        check=False,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _relative_path(root: Path, path: Path) -> str:
    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("bound file must be inside the repository") from error
    if not label or ".." in Path(label).parts:
        raise ValueError("invalid repository-relative binding path")
    return label


def _binding(root: Path, path: Path) -> dict[str, object]:
    label = _relative_path(root, path)
    return {"path": label, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _source_hashes(root: Path) -> dict[str, str]:
    missing = [label for label in _SOURCE_PATHS if not (root / label).is_file()]
    if missing:
        raise ValueError(f"missing scientific source inventory: {missing}")
    return {label: sha256_file(root / label) for label in _SOURCE_PATHS}


def _runtime_identity(root: Path) -> dict[str, object]:
    available = bool(blossom5.available())
    binary = Path(blossom5._libpypm()._name) if available else None
    return {
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "qecsim": qecsim.__version__,
            "networkx": nx.__version__,
        },
        "matching_backend": {
            "name": "blossom5" if available else "networkx",
            "blossom5_available": available,
            "blossom5_binary_sha256": sha256_file(binary) if binary is not None else None,
            "lockfile_sha256": sha256_file(root / "uv.lock"),
        },
        "blas": {
            "numpy_configuration": np.show_config(mode="dicts"),
            "scipy_configuration": scipy.show_config(mode="dicts"),
        },
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_KEYS},
    }


def _matrix_digest(matrix: np.ndarray) -> str:
    value = np.ascontiguousarray(matrix, dtype=np.uint8)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _code_identity() -> dict[str, object]:
    code = PlanarCode(5, 5)
    stabilizers = np.asarray(code.stabilizers, dtype=np.uint8)
    logicals = np.asarray(code.logicals, dtype=np.uint8)
    if stabilizers.shape != (40, 82) or logicals.shape != (2, 82):
        raise ValueError("unexpected planar-code matrix shapes")
    n, k, _distance = code.n_k_d
    return {
        "family": "qecsim_planar",
        "distance": 5,
        "n": int(n),
        "k": int(k),
        "stabilizers_sha256": _matrix_digest(stabilizers),
        "logicals_sha256": _matrix_digest(logicals),
    }


def _collect_domains(value: object, *, path: tuple[str, ...] = ()) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.endswith("seed_domain"):
                if not isinstance(child, str) or not child:
                    raise ValueError(f"invalid seed domain at {'.'.join((*path, key))}")
                found.append(child)
            found.extend(_collect_domains(child, path=(*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_collect_domains(child, path=(*path, str(index))))
    return found


def _historical_domain_inventory(
    config_root: Path, *, excluded_paths: frozenset[Path]
) -> dict[str, dict[str, object]]:
    """Recursively bind every JSON config that declares a seed domain."""
    root = Path(config_root)
    excluded = {path.resolve() for path in excluded_paths}
    result: dict[str, dict[str, object]] = {}
    for path in sorted(root.rglob("*.json")):
        if path.resolve() in excluded:
            continue
        try:
            value = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid historical config JSON: {path}") from error
        domains = sorted(_collect_domains(value))
        if not domains:
            continue
        label = path.relative_to(root).as_posix()
        if len(domains) != len(set(domains)):
            raise ValueError(f"duplicate historical seed domain in {label}")
        result[label] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "domains": domains,
        }
    return result


def _historical_domains(root: Path, analysis_path: Path, shots_path: Path) -> dict[str, object]:
    contract_declarations = {
        analysis_path.resolve(),
        shots_path.resolve(),
        (root / "configs/simple_planar_confirmation.json").resolve(),
        (root / "configs/simple_planar_confirmation_shots.json").resolve(),
    }
    inventory = _historical_domain_inventory(
        root / "configs",
        excluded_paths=frozenset(contract_declarations),
    )
    encountered = {
        domain
        for record in inventory.values()
        for domain in record["domains"]  # type: ignore[union-attr]
    }
    reserved = sorted(set(_RETIRED_DOMAINS) | encountered)
    if _SCIENTIFIC_DOMAIN in reserved or _BOOTSTRAP_DOMAIN in encountered:
        raise ValueError("confirmation or bootstrap domain collides with historical inventory")
    return {"configs": inventory, "reserved_domains": reserved}


def _historical_sampler_seeds(search_root: Path) -> set[int]:
    """Inventory already persisted planar-shot seeds without deriving any."""
    seeds: set[int] = set()
    for path in sorted(Path(search_root).rglob("planar_shots.json")):
        value = _strict_json(path)
        if not isinstance(value, dict) or not isinstance(value.get("shots"), list):
            raise TypeError(f"malformed historical planar-shot artifact: {path}")
        for shot in value["shots"]:
            if not isinstance(shot, dict) or type(shot.get("sampler_seed")) is not int:
                raise ValueError(f"malformed historical sampler seed: {path}")
            seed = shot["sampler_seed"]
            if seed in seeds:
                raise ValueError(f"duplicate historical sampler seed: {seed}")
            seeds.add(seed)
    return seeds


def preflight(config_path: Path, shot_config_path: Path) -> dict:
    """Build a metadata-only release candidate; never derive a shot coordinate."""
    root = _root()
    config_path, shot_config_path = Path(config_path), Path(shot_config_path)
    analysis = load_config(config_path)
    shots = load_shot_config(shot_config_path, fixture=False)
    commit, dirty = _git_identity(root)
    source = _source_hashes(root)
    config_bindings = {
        "analysis": _binding(root, config_path),
        "shots": _binding(root, shot_config_path),
    }
    committed = all(
        _committed_digest(root, commit, label) == digest for label, digest in source.items()
    )
    committed = committed and all(
        _committed_digest(root, commit, binding["path"]) == binding["sha256"]
        for binding in config_bindings.values()
    )
    committed = committed and _committed_digest(root, commit, "uv.lock") == sha256_file(
        root / "uv.lock"
    )
    if analysis["required_shot_seed_domain"] != shots["seed_domain"]:
        raise ValueError("analysis and shot domains disagree")
    return {
        "schema_version": 1,
        "contract_id": _CONTRACT_ID,
        "contract_sha256": sha256_file(root / _CONTRACT_PATH),
        "approved_commit": commit,
        "source_sha256": source,
        "config_bindings": config_bindings,
        "runtime": _runtime_identity(root),
        "code_identity": _code_identity(),
        "historical_domains": _historical_domains(root, config_path, shot_config_path),
        "unopened_domain": {
            "domain": _SCIENTIFIC_DOMAIN,
            "campaign_seed": shots["campaign_seed"],
            "coordinates_derived": False,
        },
        "expected_shots": 4096,
        "git_dirty": dirty,
        "committed_closure": committed,
    }


def _validate_file_binding(binding: object, *, base: Path, label: str) -> None:
    if not isinstance(binding, dict) or set(binding) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"malformed {label} binding")
    path_value = binding["path"]
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"invalid {label} binding path")
    candidate = Path(path_value)
    path = candidate if candidate.is_absolute() else base / candidate
    if (
        not path.is_file()
        or path.stat().st_size != binding["size_bytes"]
        or sha256_file(path) != binding["sha256"]
    ):
        raise ValueError(f"{label} binding bytes changed")


def validate_release(receipt_path: Path, preflight_record: dict) -> dict:
    """Validate an immutable independent authorization against current metadata."""
    path = Path(receipt_path)
    receipt = _strict_json(path)
    if not isinstance(receipt, dict) or set(receipt) != _RELEASE_KEYS:
        raise ValueError("release receipt has wrong fields")
    reviews = receipt.get("review_bindings")
    if not isinstance(reviews, dict) or set(reviews) != {"contract", "implementation"}:
        raise ValueError("release receipt review bindings are incomplete")
    for name, binding in reviews.items():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "size_bytes", "sha256", "verdict"}
            or binding.get("verdict") != "accepted"
        ):
            raise ValueError(f"invalid {name} review binding")
        _validate_file_binding(
            {key: binding[key] for key in ("path", "size_bytes", "sha256")},
            base=_root(),
            label=f"{name} review",
        )
    expected = {
        key: value
        for key, value in preflight_record.items()
        if key not in {"git_dirty", "committed_closure"}
    }
    expected["review_bindings"] = reviews
    expected["decision"] = "approved_to_open"
    if receipt != expected:
        raise ValueError("release receipt does not match current preflight")
    if (
        preflight_record.get("git_dirty") is not False
        or preflight_record.get("committed_closure") is not True
    ):
        raise ValueError("release receipt cannot authorize dirty or uncommitted inputs")
    return receipt


def _release_receipt_for_test(
    record: dict, *, contract_review: Path, implementation_review: Path
) -> dict:
    """Construct receipt-shaped data for validator tests; never writes authority."""

    def review(path: Path) -> dict[str, object]:
        try:
            label = _relative_path(_root(), path)
        except ValueError:
            label = str(path.resolve())
        return {
            "path": label,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "verdict": "accepted",
        }

    receipt = {
        key: value for key, value in record.items() if key not in {"git_dirty", "committed_closure"}
    }
    receipt["review_bindings"] = {
        "contract": review(contract_review),
        "implementation": review(implementation_review),
    }
    receipt["decision"] = "approved_to_open"
    return receipt


def _failure_path(out: Path) -> Path:
    return out.with_name(f"{out.name}.failure.json")


def _write_failure_receipt(
    out: Path, *, error: BaseException, approved_hashes: Mapping[str, str], stage: str
) -> None:
    receipt = {
        "schema_version": 1,
        "exception_type": type(error).__name__,
        "approved_hashes": dict(approved_hashes),
        "intended_destination": str(out),
        "stage": stage,
    }
    target = _failure_path(out)
    try:
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_canonical_bytes(receipt))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_tree(path: Path) -> None:
    for child in sorted(path.rglob("*")):
        if child.is_file() and not child.is_symlink():
            with child.open("rb") as handle:
                os.fsync(handle.fileno())
    directory = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_directory(
    out: Path,
    populate: Callable[[Path], None],
    *,
    approved_hashes: Mapping[str, str] | None = None,
    validate: Callable[[Path], None] | None = None,
) -> None:
    """Publish a complete new directory while owning an exclusive sibling lock."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(f"{out.name}.lock")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing existing destination: {out}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    lock_stat = os.fstat(descriptor)
    stage_name = "lock"
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"destination appeared while locked: {out}")
        with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temporary:
            stage = Path(temporary)
            stage_name = "populate"
            populate(stage)
            stage_name = "validate"
            if validate is not None:
                validate(stage)
            _fsync_tree(stage)
            if out.exists() or out.is_symlink():
                raise FileExistsError(f"destination appeared before publication: {out}")
            stage_name = "rename"
            os.replace(stage, out)
            parent_descriptor = os.open(out.parent, os.O_RDONLY)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    except BaseException as error:
        _write_failure_receipt(
            out,
            error=error,
            approved_hashes=approved_hashes or {},
            stage=stage_name,
        )
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = lock.stat(follow_symlinks=False)
            if current.st_dev == lock_stat.st_dev and current.st_ino == lock_stat.st_ino:
                lock.unlink()
        except FileNotFoundError:
            pass


def _base_provenance(
    config_path: Path,
    shot_config_path: Path,
    *,
    release_binding: dict[str, object] | None,
) -> dict[str, object]:
    root = _root()
    commit, dirty = _git_identity(root)
    return {
        "producer_commit": commit,
        "git_dirty": dirty,
        "source_sha256": _source_hashes(root),
        "config_bindings": {
            "analysis": _binding(root, config_path),
            "shots": _binding(root, shot_config_path),
        },
        "runtime": _runtime_identity(root),
        "code_identity": _code_identity(),
        "historical_domains": _historical_domains(root, config_path, shot_config_path),
        "release_binding": release_binding,
    }


def _assert_base_provenance_current(
    baseline: Mapping[str, object], config_path: Path, shot_config_path: Path
) -> None:
    release = baseline.get("release_binding")
    if release is not None and not isinstance(release, dict):
        raise ValueError("malformed release binding in provenance")
    current = _base_provenance(
        Path(config_path),
        Path(shot_config_path),
        release_binding=release,
    )
    if any(baseline.get(key) != value for key, value in current.items()):
        raise ValueError("scientific source/input provenance changed during publication")


def _approved_hashes(
    provenance: Mapping[str, object], *, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    source = provenance.get("source_sha256")
    configs = provenance.get("config_bindings")
    if not isinstance(source, dict) or not isinstance(configs, dict):
        raise TypeError("malformed provenance hash closure")
    result = {f"source:{label}": str(digest) for label, digest in source.items()}
    for name, binding in configs.items():
        if not isinstance(binding, dict) or not isinstance(binding.get("sha256"), str):
            raise TypeError("malformed config provenance binding")
        result[f"config:{name}"] = binding["sha256"]
    if extra is not None:
        result.update(extra)
    return result


def _coordinate_seeds(config: Mapping[str, object]) -> list[int]:
    return [
        planar_shot_data._shot_seed(
            str(config["seed_domain"]), error_rate=float(rate), shot_index=index
        )
        for rate in config["error_rates"]  # type: ignore[union-attr]
        for index in range(int(config["shots_per_rate"]))
    ]


def _validate_generation_directory(stage: Path) -> None:
    if {path.name for path in stage.iterdir()} != {"planar_shots.json", "generation_manifest.json"}:
        raise ValueError("generation output has unexpected file closure")
    manifest = _strict_json(stage / "generation_manifest.json")
    if not isinstance(manifest, dict) or set(manifest) != _GENERATION_MANIFEST_KEYS:
        raise ValueError("generation manifest fields do not match the frozen schema")
    binding = manifest.get("shots_binding")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"path", "size_bytes", "sha256"}
        or binding.get("path") != "planar_shots.json"
        or binding.get("size_bytes") != (stage / "planar_shots.json").stat().st_size
        or binding.get("sha256") != sha256_file(stage / "planar_shots.json")
    ):
        raise ValueError("generation manifest has stale shot hash")
    shots = _strict_json(stage / "planar_shots.json")
    if not isinstance(shots, dict) or set(shots) != {"config", "shots", "provenance"}:
        raise ValueError("generated planar-shot artifact has wrong fields")
    _validate_shot_manifest(shots, _root())


def generate(
    config_path: Path,
    shot_config_path: Path,
    out: Path,
    *,
    release_path: Path | None,
    fixture: bool = False,
) -> dict:
    """Generate immutable joined shots and a separate wrapper provenance manifest."""
    config_path, shot_config_path, out = Path(config_path), Path(shot_config_path), Path(out)
    load_config(config_path)
    shots_config = load_shot_config(shot_config_path, fixture=fixture)
    release_binding: dict[str, object] | None = None
    if fixture:
        if release_path is not None:
            raise ValueError("fixture generation cannot consume production release authority")
    else:
        if release_path is None:
            raise ValueError("production generation requires a release receipt")
        record = preflight(config_path, shot_config_path)
        validate_release(release_path, record)
        release_binding = _binding(_root(), Path(release_path))
        seeds = _coordinate_seeds(shots_config)
        if len(seeds) != len(set(seeds)):
            raise RuntimeError("confirmation sampler seed collision")
        historical = _historical_sampler_seeds(_root())
        fixture_config = load_shot_config(
            _root() / "configs/simple_planar_confirmation_fixture_shots.json",
            fixture=True,
        )
        reserved = historical | set(_coordinate_seeds(fixture_config))
        collision = next((seed for seed in seeds if seed in reserved), None)
        if collision is not None:
            raise RuntimeError(f"confirmation sampler seed collides with reserved seed {collision}")
    provenance = _base_provenance(config_path, shot_config_path, release_binding=release_binding)

    def populate(stage: Path) -> None:
        sampler = stage / "sampler"
        payload = planar_shot_data.generate_planar_shots(shot_config_path, sampler)
        source = sampler / "planar_shots.json"
        shutil.move(source, stage / "planar_shots.json")
        sampler.rmdir()
        manifest = {
            "schema_version": 1,
            "contract_id": _CONTRACT_ID,
            **{
                key: provenance[key]
                for key in (
                    "producer_commit",
                    "config_bindings",
                    "source_sha256",
                    "runtime",
                    "code_identity",
                    "historical_domains",
                    "release_binding",
                )
            },
            "shots_binding": {
                "path": "planar_shots.json",
                "size_bytes": (stage / "planar_shots.json").stat().st_size,
                "sha256": sha256_file(stage / "planar_shots.json"),
            },
        }
        write_canonical_json(stage / "generation_manifest.json", manifest)
        result.update({"shots": payload, "manifest": manifest})

    def validate(stage: Path) -> None:
        _validate_generation_directory(stage)
        _assert_base_provenance_current(provenance, config_path, shot_config_path)

    result: dict[str, Any] = {}
    approved = _approved_hashes(provenance)
    _publish_directory(out, populate, approved_hashes=approved, validate=validate)
    return result


def _host_identity() -> dict[str, object]:
    runtime = _runtime_identity(_root())
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "thread_environment": runtime["thread_environment"],
        "blas": runtime["blas"],
    }


def run(
    config_path: Path,
    shot_config_path: Path,
    shots_path: Path,
    out: Path,
    *,
    release_path: Path | None,
    fixture: bool = False,
) -> dict:
    """Execute every frozen arm and publish raw, summary, and timing atomically."""
    config_path, shot_config_path, shots_path, out = map(
        Path, (config_path, shot_config_path, shots_path, out)
    )
    load_config(config_path)
    load_shot_config(shot_config_path, fixture=fixture)
    if fixture:
        if release_path is not None:
            raise ValueError("fixture run cannot consume production release authority")
        release_binding = None
    else:
        if release_path is None:
            raise ValueError("production run requires a release receipt")
        record = preflight(config_path, shot_config_path)
        validate_release(release_path, record)
        release_binding = _binding(_root(), Path(release_path))
    try:
        _validate_generation_directory(shots_path.parent)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
        raise ValueError("shots must come from a valid two-file generation directory") from error
    shot_payload = json.loads(shots_path.read_text())
    if not isinstance(shot_payload, dict) or set(shot_payload) != {"config", "shots", "provenance"}:
        raise ValueError("invalid planar-shot artifact")
    _validate_shot_manifest(shot_payload, _root())
    expected_config = load_shot_config(shot_config_path, fixture=fixture)
    if shot_payload["config"] != expected_config:
        raise ValueError("shot artifact config does not match requested run")
    rows: list[dict] = []
    shot_timings: list[dict] = []
    started = time.perf_counter()
    for shot in shot_payload["shots"]:
        rows.append(evaluate_shot(shot, timing=shot_timings))
    run_seconds = time.perf_counter() - started
    validate_result_rows(rows, fixture=fixture)
    summary = summarize(rows, fixture=fixture)
    provenance = _base_provenance(config_path, shot_config_path, release_binding=release_binding)
    provenance.update(
        {
            "shots_binding": {
                "path": shots_path.name,
                "size_bytes": shots_path.stat().st_size,
                "sha256": sha256_file(shots_path),
            },
            "generation_manifest_binding": None,
        }
    )
    manifest_path = shots_path.with_name("generation_manifest.json")
    if manifest_path.is_file():
        provenance["generation_manifest_binding"] = {
            "path": manifest_path.name,
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    raw = {
        "schema_version": 1,
        "contract_id": _CONTRACT_ID,
        "status": "reduced_non_scientific" if fixture else "complete_pending_replay",
        "scientific_eligible": bool(not fixture and summary["decision"]["execution_eligible"]),
        "config": load_config(config_path),
        "shot_config": expected_config,
        "shot_provenance": shot_payload["provenance"],
        "rows": rows,
        "summary": summary,
        "provenance": provenance,
    }
    timing = {
        "schema_version": 1,
        "contract_id": _CONTRACT_ID,
        "host": _host_identity(),
        "shots": shot_timings,
        "run_wall_seconds": run_seconds,
    }

    def populate(stage: Path) -> None:
        write_canonical_json(stage / "result.json", raw)
        write_canonical_json(stage / "summary.json", summary)
        write_canonical_json(stage / "timing.json", timing)

    def validate(stage: Path) -> None:
        if {path.name for path in stage.iterdir()} != {
            "result.json",
            "summary.json",
            "timing.json",
        }:
            raise ValueError("run output has unexpected file closure")
        persisted = json.loads((stage / "result.json").read_text())
        validate_result_rows(persisted["rows"], fixture=fixture)
        if persisted["summary"] != summarize(persisted["rows"], fixture=fixture):
            raise ValueError("persisted summary does not replay")
        _assert_base_provenance_current(provenance, config_path, shot_config_path)
        if sha256_file(shots_path) != provenance["shots_binding"]["sha256"]:
            raise ValueError("joined shot input changed during publication")

    _publish_directory(
        out,
        populate,
        approved_hashes=_approved_hashes(
            provenance, extra={"input:shots": sha256_file(shots_path)}
        ),
        validate=validate,
    )
    return {"raw": raw, "summary": summary, "timing": timing}
