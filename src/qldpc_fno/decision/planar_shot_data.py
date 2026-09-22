"""Immutable, replayable physical-error shots for planar accuracy screening."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import qecsim
import scipy
from qecsim import paulitools as pt
from qecsim.graphtools import blossom5
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode

from qldpc_fno.artifacts import sha256_file, write_canonical_json

_CONFIG_KEYS = {
    "schema_version",
    "seed_domain",
    "campaign_seed",
    "code_distance",
    "error_rates",
    "shots_per_rate",
    "noise_model",
}
_ERROR_RATES = [0.1, 0.15]
_NOISE_MODEL = "qecsim_iid_depolarizing_code_capacity"
_SOURCE_LABELS = (
    "src/qldpc_fno/decision/planar_shot_data.py",
    "src/qldpc_fno/decision/planar_shot_accuracy.py",
    "src/qldpc_fno/decision/tensor_network.py",
    "src/qldpc_fno/metrics/paired.py",
    "src/qldpc_fno/artifacts.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_provenance(root: Path) -> dict[str, object]:
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
    }


def _valid_commit(root: Path, commit: object) -> None:
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("invalid producer commit provenance")
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("producer commit provenance is not available")


def _committed_digest(root: Path, commit: str, label: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{label}"], capture_output=True, check=False
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _file_binding(root: Path, path: Path, commit: str) -> dict[str, object]:
    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
        scope = "repository"
    except ValueError:
        label, scope = path.name, "external_diagnostic"
    digest = sha256_file(path)
    return {
        "path": label,
        "scope": scope,
        "sha256": digest,
        "committed": scope == "repository" and _committed_digest(root, commit, label) == digest,
    }


def _validate_binding(binding: object, root: Path, commit: str, digest: str) -> bool:
    if not isinstance(binding, dict) or set(binding) != {"path", "scope", "sha256", "committed"}:
        raise ValueError("malformed config/input provenance binding")
    label = binding["path"]
    if (
        not isinstance(label, str)
        or not label
        or Path(label).is_absolute()
        or ".." in Path(label).parts
        or binding["scope"] not in {"repository", "external_diagnostic"}
        or binding["sha256"] != digest
        or type(binding["committed"]) is not bool
    ):
        raise ValueError("invalid config/input provenance binding")
    committed = (
        binding["scope"] == "repository" and _committed_digest(root, commit, label) == digest
    )
    if binding["committed"] != committed:
        raise ValueError("config/input committed provenance mismatch")
    return committed


def _validate_producer(provenance: dict[str, object], root: Path, labels: tuple[str, ...]) -> bool:
    _valid_commit(root, provenance.get("git_commit"))
    if type(provenance.get("git_dirty")) is not bool:
        raise ValueError("invalid dirty-state provenance")
    for key, expected in _runtime_provenance(root).items():
        if provenance.get(key) != expected:
            raise ValueError(f"dependency/backend provenance mismatch: {key}")
    expected_sources = {label: sha256_file(root / label) for label in labels}
    if provenance.get("source_sha256") != expected_sources:
        raise ValueError("stale source provenance hashes")
    committed_sources = all(
        _committed_digest(root, provenance["git_commit"], label) == digest
        for label, digest in expected_sources.items()
    )
    lock_committed = (
        _committed_digest(root, provenance["git_commit"], "uv.lock")
        == provenance["matching_backend"]["lockfile_sha256"]
    )
    if not provenance["git_dirty"] and not (committed_sources and lock_committed):
        raise ValueError("clean producer provenance does not match committed scientific sources")
    return not provenance["git_dirty"] and committed_sources and lock_committed


def _validate_shot_manifest(payload: dict[str, object], root: Path) -> bool:
    provenance = payload["provenance"]
    clean = _validate_producer(provenance, root, _SOURCE_LABELS)
    content = provenance.get("config_content")
    if (
        not isinstance(content, str)
        or hashlib.sha256(content.encode()).hexdigest() != provenance.get("config_sha256")
        or json.loads(content) != payload["config"]
        or provenance.get("qecsim_version") != qecsim.__version__
    ):
        raise ValueError("config/dependency provenance does not match artifact")
    binding = {
        "path": provenance.get("config_path"),
        "scope": provenance.get("config_scope"),
        "sha256": provenance.get("config_sha256"),
        "committed": provenance.get("config_committed"),
    }
    committed = _validate_binding(
        binding, root, provenance["git_commit"], provenance["config_sha256"]
    )
    if binding["scope"] == "repository":
        config_path = root / binding["path"]
        if not config_path.is_file() or config_path.read_text() != content:
            raise ValueError("config path/content provenance is stale")
    return clean and committed


def _load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise ValueError("planar shot config must contain exactly the schema fields")
    if config["schema_version"] != 1:
        raise ValueError("planar shot schema_version must be 1")
    domain = config["seed_domain"]
    if not isinstance(domain, str) or not domain:
        raise ValueError("seed_domain must be a nonempty string")
    expected_campaign_seed = int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big")
    if (
        type(config["campaign_seed"]) is not int
        or config["campaign_seed"] != expected_campaign_seed
    ):
        raise ValueError("campaign_seed must be the SHA-256 derivation of seed_domain")
    if type(config["code_distance"]) is not int or config["code_distance"] != 5:
        raise ValueError("code_distance must be exactly 5")
    if config["error_rates"] != _ERROR_RATES:
        raise ValueError("error_rates must be exactly [0.1, 0.15]")
    if type(config["shots_per_rate"]) is not int or config["shots_per_rate"] <= 0:
        raise ValueError("shots_per_rate must be a positive integer")
    if config["noise_model"] != _NOISE_MODEL:
        raise ValueError(f"noise_model must be {_NOISE_MODEL}")
    return config


def _shot_seed(domain: str, *, error_rate: float, shot_index: int) -> int:
    if (
        os.environ.get("QLDPC_FNO_CONFIRMATION_TESTING") == "1"
        and domain.startswith("qldpc-fno/simple-planar-confirmation/")
        and "/test-fixture/" not in domain
    ):
        raise ValueError("scientific confirmation seed forbidden in test mode")
    identity = f"{domain}|d5|p{error_rate:.6f}|i{shot_index:06d}"
    return int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")


def _git_provenance(repository_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _source_hashes(repository_root: Path) -> dict[str, str]:
    return {label: sha256_file(repository_root / label) for label in _SOURCE_LABELS}


def generate_planar_shots(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Generate one immutable, physical-error paired planar-shot artifact."""
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = _load_config(config_path)
    domain = str(config["seed_domain"])
    if (
        os.environ.get("QLDPC_FNO_CONFIRMATION_TESTING") == "1"
        and domain.startswith("qldpc-fno/simple-planar-confirmation/")
        and "/test-fixture/" not in domain
    ):
        raise ValueError("scientific confirmation seed forbidden in test mode")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

    repository_root = _repository_root()
    git_commit, git_dirty = _git_provenance(repository_root)
    binding = _file_binding(repository_root, config_path, git_commit)
    provenance = {
        "qecsim_version": qecsim.__version__,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "config_sha256": binding["sha256"],
        "config_path": binding["path"],
        "config_scope": binding["scope"],
        "config_committed": binding["committed"],
        "config_content": config_path.read_text(),
        "source_sha256": _source_hashes(repository_root),
        **_runtime_provenance(repository_root),
    }

    code = PlanarCode(5, 5)
    model = DepolarizingErrorModel()
    shots: list[dict[str, object]] = []
    sampler_seeds: set[int] = set()
    for error_rate_value in _ERROR_RATES:
        error_rate = float(error_rate_value)
        for shot_index in range(int(config["shots_per_rate"])):
            sampler_seed = _shot_seed(
                str(config["seed_domain"]), error_rate=error_rate, shot_index=shot_index
            )
            if sampler_seed in sampler_seeds:
                raise RuntimeError("sampler seed collision within planar shot artifact")
            sampler_seeds.add(sampler_seed)
            error = np.asarray(
                model.generate(code, error_rate, np.random.default_rng(sampler_seed)),
                dtype=np.uint8,
            )
            syndrome = np.asarray(pt.bsp(error, code.stabilizers.T), dtype=np.uint8)
            shots.append(
                {
                    "shot_id": f"d5/p{error_rate:.6f}/i{shot_index:06d}",
                    "error_rate": error_rate,
                    "shot_index": shot_index,
                    "sampler_seed": sampler_seed,
                    "error_bsf": error.tolist(),
                    "syndrome": syndrome.tolist(),
                }
            )

    payload: dict[str, object] = {
        "config": config,
        "shots": shots,
        "provenance": provenance,
    }
    write_canonical_json(output_dir / "planar_shots.json", payload)
    return payload
