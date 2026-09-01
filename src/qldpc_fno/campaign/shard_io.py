"""Strict verification and batch-wise access for immutable campaign shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.seeds import derive_seed
from qldpc_fno.campaign.shards import validate_campaign_code
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.stim.b8 import read_b8_rows


@dataclass(frozen=True, slots=True)
class VerifiedShard:
    directory: Path
    manifest_path: Path
    manifest_sha256: str
    start: int
    stop: int
    error_rate: float
    rate_index: int
    shard_index: int
    seed: int
    dimensions: dict[str, int]


@dataclass(frozen=True, slots=True)
class VerifiedShardSet:
    root: Path
    role: str
    manifest_sha256: str
    shards: tuple[VerifiedShard, ...]

    @property
    def shots(self) -> int:
        return self.shards[-1].stop if self.shards else 0

    @property
    def shard_manifest_sha256(self) -> dict[str, str]:
        return {
            str(shard.manifest_path.relative_to(self.root)): shard.manifest_sha256
            for shard in self.shards
        }

    def read(self, filename: str, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("batch indices must be one-dimensional")
        if np.any(requested < 0) or np.any(requested >= self.shots):
            raise ValueError("batch index is out of bounds")
        if filename not in {"dets.b8", "errors.b8", "obs_actual.b8"}:
            raise ValueError(f"unsupported campaign shard payload: {filename}")
        if not self.shards:
            raise ValueError("verified shard set is empty")
        width = self.shards[0].dimensions[filename]
        output = np.empty((requested.size, width), dtype=np.uint8)
        for shard in self.shards:
            positions = np.flatnonzero((requested >= shard.start) & (requested < shard.stop))
            if positions.size == 0:
                continue
            local_rows = requested[positions] - shard.start
            output[positions] = read_b8_rows(
                shard.directory / filename,
                rows=local_rows,
                shots=shard.stop - shard.start,
                bits_per_shot=width,
            )
        return output

    def error_rates(self, indices: np.ndarray) -> np.ndarray:
        requested = np.asarray(indices, dtype=np.int64)
        if requested.ndim != 1 or np.any(requested < 0) or np.any(requested >= self.shots):
            raise ValueError("batch index is out of bounds")
        rates = np.empty(requested.size, dtype=np.float64)
        for shard in self.shards:
            positions = np.flatnonzero((requested >= shard.start) & (requested < shard.stop))
            rates[positions] = shard.error_rate
        return rates

    def indices_for_rate(self, rate_index: int) -> np.ndarray:
        """Return global shot indices belonging to one verified noise rate."""
        batches = [
            np.arange(shard.start, shard.stop, dtype=np.int64)
            for shard in self.shards
            if shard.rate_index == rate_index
        ]
        if not batches:
            raise ValueError(f"campaign shards do not contain rate_index {rate_index}")
        return np.concatenate(batches)


def deterministic_stratified_split(
    shards: VerifiedShardSet,
    *,
    training_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Assign train/validation shots independently within every represented rate."""
    if type(training_seed) is not int or training_seed < 0:
        raise ValueError("training_seed must be a non-negative integer")
    rates = sorted({shard.rate_index for shard in shards.shards})
    train_batches: list[np.ndarray] = []
    validation_batches: list[np.ndarray] = []
    per_rate: list[dict[str, object]] = []
    for rate_index in rates:
        indices = shards.indices_for_rate(rate_index)
        ranked = sorted(
            indices.tolist(),
            key=lambda index: hashlib.sha256(
                f"{training_seed}:{rate_index}:{index}:validation".encode()
            ).digest(),
        )
        validation_count = max(1, indices.size // 4) if indices.size >= 2 else 0
        validation = np.array(sorted(ranked[:validation_count]), dtype=np.int64)
        train = np.array(sorted(ranked[validation_count:]), dtype=np.int64)
        if train.size == 0:
            raise ValueError(f"rate_index {rate_index} has no fitting shots")
        train_batches.append(train)
        if validation.size:
            validation_batches.append(validation)
        error_rates = {
            shard.error_rate for shard in shards.shards if shard.rate_index == rate_index
        }
        if len(error_rates) != 1:
            raise ValueError(f"rate_index {rate_index} has inconsistent error rates")
        per_rate.append(
            {
                "error_rate": error_rates.pop(),
                "rate_index": rate_index,
                "total_shots": int(indices.size),
                "train_shots": int(train.size),
                "validation_shots": int(validation.size),
            }
        )
    if not validation_batches:
        raise ValueError("stratified training requires at least one rate with two shots")
    train_indices = np.concatenate(train_batches)
    validation_indices = np.concatenate(validation_batches)

    def indices_sha256(values: np.ndarray) -> str:
        return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()

    metadata: dict[str, object] = {
        "derivation": {
            "algorithm": "sha256_rank_within_rate",
            "payload": "{training_seed}:{rate_index}:{global_index}:validation",
            "validation_count": "max(1, floor(rate_shots / 4)) when rate_shots >= 2",
        },
        "per_rate": per_rate,
        "train_indices_sha256": indices_sha256(train_indices),
        "train_shots": int(train_indices.size),
        "validation_indices_sha256": indices_sha256(validation_indices),
        "validation_shots": int(validation_indices.size),
    }
    return train_indices, validation_indices, metadata


def load_campaign_code(
    code_dir: Path,
) -> tuple[dict[str, object], sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    """Load and independently validate the canonical campaign code artifact."""
    manifest_path = code_dir / "code.json"
    metadata = json.loads(manifest_path.read_text())
    hx_path = code_dir / "hx.npz"
    hz_path = code_dir / "hz.npz"
    verify_sha256(hx_path, str(metadata["hx_sha256"]), label="Hx")
    verify_sha256(hz_path, str(metadata["hz_sha256"]), label="Hz")
    hx = sparse.load_npz(hx_path).astype(np.uint8).tocsr()
    hz = sparse.load_npz(hz_path).astype(np.uint8).tocsr()
    validate_campaign_code(metadata, hx, hz)
    logical_x = logical_x_basis(hx, hz).astype(np.uint8).tocsr()
    if logical_x.shape != (744, 2610):
        raise ValueError("campaign logical X matrix dimensions must be (744, 2610)")
    return metadata, hx, hz, logical_x


def load_verified_shards(
    root: Path,
    *,
    role: str,
    config_path: Path,
    code_manifest_path: Path,
) -> VerifiedShardSet:
    """Verify a complete role publication and every artifact it declares."""
    completion_path = root / "manifest.json"
    completion = json.loads(completion_path.read_text())
    if completion.get("complete") is not True or completion.get("role") != role:
        raise ValueError(f"shards do not belong to a completed {role} publication")
    declared = completion.get("shards")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("completion manifest must declare at least one shard")
    discovered = {str(path.relative_to(root)) for path in root.glob("rate-*/shard-*/samples.json")}
    if discovered != set(declared):
        raise ValueError("completion manifest shard set does not match the role directory")

    config_sha256 = sha256_file(config_path)
    config = CampaignConfig.from_json(config_path)
    code_sha256 = sha256_file(code_manifest_path)
    shards: list[VerifiedShard] = []
    offset = 0
    expected_dimensions: dict[str, int] | None = None
    coordinates: set[tuple[int, int]] = set()
    seeds: set[int] = set()
    rates_by_index: dict[int, float] = {}
    shard_indices_by_rate: dict[int, set[int]] = {}
    for relative in sorted(declared):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("shard manifest path escapes the role directory")
        manifest_path = root / relative_path
        verify_sha256(manifest_path, str(declared[relative]), label=f"{role} shard manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("role") != role or manifest.get("format") != "b8":
            raise ValueError(f"shard manifest is not a {role} b8 artifact")
        rate_index = manifest.get("rate_index")
        shard_index = manifest.get("shard_index")
        seed = manifest.get("seed")
        if type(rate_index) is not int or rate_index < 0:
            raise ValueError("shard rate_index must be a non-negative integer")
        if type(shard_index) is not int or shard_index < 0:
            raise ValueError("shard shard_index must be a non-negative integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("shard seed must be a non-negative integer")
        coordinate = (rate_index, shard_index)
        if coordinate in coordinates:
            raise ValueError(f"duplicate shard coordinate: {coordinate}")
        coordinates.add(coordinate)
        if seed in seeds:
            raise ValueError(f"duplicate shard seed: {seed}")
        seeds.add(seed)
        expected_path = Path(f"rate-{rate_index:03d}") / f"shard-{shard_index:05d}"
        if manifest.get("path") != str(expected_path) or relative_path.parent != expected_path:
            raise ValueError("shard manifest path provenance mismatch")
        expected_seed = derive_seed(
            config.campaign_seed,
            p_index=rate_index,
            role=role,
            shard_index=shard_index,
        )
        if seed != expected_seed:
            raise ValueError(f"shard seed does not match the derived seed for {role} {coordinate}")
        shots = int(manifest["shots"])
        if shots <= 0:
            raise ValueError("campaign shards must contain positive shot counts")
        dimensions = {key: int(value) for key, value in manifest["dimensions"].items()}
        if dimensions != {"dets.b8": 945, "errors.b8": 2610, "obs_actual.b8": 744}:
            raise ValueError("campaign shard dimensions do not match lp_3_7_16")
        if expected_dimensions is not None and dimensions != expected_dimensions:
            raise ValueError("campaign shards have inconsistent dimensions")
        expected_dimensions = dimensions
        payload_sha256 = manifest.get("sha256")
        expected_payloads = {"dets.b8", "errors.b8", "obs_actual.b8"}
        if not isinstance(payload_sha256, dict) or set(payload_sha256) != expected_payloads:
            raise ValueError(
                "shard SHA-256 payloads must be exactly dets.b8, errors.b8, and obs_actual.b8"
            )
        source = manifest.get("source_sha256")
        if not isinstance(source, dict):
            raise TypeError("shard manifest is missing source SHA-256 provenance")
        if source.get("config") != config_sha256:
            raise ValueError("campaign configuration SHA-256 mismatch in shard provenance")
        if source.get("code_manifest") != code_sha256:
            raise ValueError("code manifest SHA-256 mismatch in shard provenance")
        for filename, expected in payload_sha256.items():
            verify_sha256(manifest_path.parent / filename, str(expected), label=filename)
        verify_sha256(manifest_path.parent / "model.dem", str(source["dem"]), label="DEM")
        rate = float(manifest["error_rate"])
        if not 0.0 < rate < 0.5:
            raise ValueError("campaign shard error_rate must be between zero and 0.5")
        previous_rate = rates_by_index.setdefault(rate_index, rate)
        if previous_rate != rate:
            raise ValueError(f"rate_index {rate_index} has inconsistent error rates")
        shard_indices_by_rate.setdefault(rate_index, set()).add(shard_index)
        shards.append(
            VerifiedShard(
                directory=manifest_path.parent,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                start=offset,
                stop=offset + shots,
                error_rate=rate,
                rate_index=rate_index,
                shard_index=shard_index,
                seed=seed,
                dimensions=dimensions,
            )
        )
        offset += shots
    if set(rates_by_index) != set(range(len(rates_by_index))):
        raise ValueError("campaign shard rate indices must be consecutive from zero")
    for rate_index, shard_indices in shard_indices_by_rate.items():
        if shard_indices != set(range(len(shard_indices))):
            raise ValueError(f"shard indices for rate {rate_index} must be consecutive from zero")
    return VerifiedShardSet(root, role, sha256_file(completion_path), tuple(shards))
