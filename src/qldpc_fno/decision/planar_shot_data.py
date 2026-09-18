"""Immutable, replayable physical-error shots for planar accuracy screening."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import qecsim
from qecsim import paulitools as pt
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
)


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
    if type(config["campaign_seed"]) is not int or config["campaign_seed"] != expected_campaign_seed:
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
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")

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
                model.generate(code, error_rate, np.random.default_rng(sampler_seed)), dtype=np.uint8
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

    repository_root = Path(__file__).resolve().parents[3]
    git_commit, git_dirty = _git_provenance(repository_root)
    payload: dict[str, object] = {
        "config": config,
        "shots": shots,
        "provenance": {
            "qecsim_version": qecsim.__version__,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "config_sha256": sha256_file(config_path),
            "source_sha256": _source_hashes(repository_root),
        },
    }
    write_canonical_json(output_dir / "planar_shots.json", payload)
    return payload
