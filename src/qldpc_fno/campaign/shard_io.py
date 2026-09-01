"""Strict verification and batch-wise access for immutable campaign shards."""

from __future__ import annotations

import hashlib
import json
import math
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

_TEACHER_BITS_PER_SHOT = 2610


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


def verify_teacher_chunk(
    manifest_path: Path,
    *,
    chunk_index: int,
    start: int,
    stop: int,
    source: dict[str, object],
    decoder: dict[str, object],
) -> dict[str, object]:
    """Verify one atomically published teacher chunk and its packed rows."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"teacher chunk manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if set(manifest) != {
        "bits_per_shot",
        "chunk_index",
        "decoder",
        "path",
        "sha256",
        "shots",
        "source",
        "start",
        "stop",
    }:
        raise ValueError("teacher chunk manifest fields do not match the declared schema")
    integer_fields = ("bits_per_shot", "chunk_index", "shots", "start", "stop")
    if any(type(manifest.get(field)) is not int for field in integer_fields):
        raise ValueError("teacher chunk coordinates must be exact integers")
    if not isinstance(manifest.get("path"), str) or not isinstance(manifest.get("sha256"), str):
        raise TypeError("teacher chunk path and SHA-256 must be strings")
    expected = {
        "bits_per_shot": _TEACHER_BITS_PER_SHOT,
        "chunk_index": chunk_index,
        "decoder": decoder,
        "path": f"chunk-{chunk_index:05d}.b8",
        "shots": stop - start,
        "source": source,
        "start": start,
        "stop": stop,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"teacher chunk {chunk_index} has mismatched {key}")
    chunk_path = manifest_path.parent / manifest["path"]
    if not chunk_path.is_file():
        raise FileNotFoundError(f"teacher chunk data is missing: {chunk_path}")
    expected_size = (stop - start) * math.ceil(_TEACHER_BITS_PER_SHOT / 8)
    if chunk_path.stat().st_size != expected_size:
        raise ValueError(f"teacher chunk {chunk_index} size mismatch")
    verify_sha256(chunk_path, manifest["sha256"], label=f"teacher chunk {chunk_index}")
    return manifest


def verify_teacher_chunks(
    model_dir: Path,
    *,
    expected_shots: int,
    chunk_shots: int,
    source: dict[str, object],
    decoder: dict[str, object],
) -> dict[str, str]:
    """Verify the exact complete manifest/data set for a packed teacher cache."""
    if type(expected_shots) is not int or expected_shots <= 0:
        raise ValueError("expected teacher shots must be a positive integer")
    if type(chunk_shots) is not int or not 0 < chunk_shots <= 2_048:
        raise ValueError("teacher chunk_shots must be an integer between 1 and 2048")
    if not isinstance(source, dict) or not isinstance(decoder, dict):
        raise TypeError("teacher chunk source and decoder provenance must be objects")
    chunk_dir = model_dir / "teacher_chunks"
    expected_count = (expected_shots + chunk_shots - 1) // chunk_shots
    expected_manifest_names = {
        f"chunk-{chunk_index:05d}.json" for chunk_index in range(expected_count)
    }
    expected_data_names = {f"chunk-{chunk_index:05d}.b8" for chunk_index in range(expected_count)}
    discovered_manifest_names = {path.name for path in chunk_dir.glob("chunk-*.json")}
    if discovered_manifest_names != expected_manifest_names:
        raise ValueError("teacher chunk manifest set is incomplete or contains extras")
    discovered_data_names = {path.name for path in chunk_dir.glob("chunk-*.b8")}
    if discovered_data_names != expected_data_names:
        raise ValueError("teacher chunk data set is incomplete or contains extras")

    chunks: dict[str, str] = {}
    for chunk_index in range(expected_count):
        start = chunk_index * chunk_shots
        stop = min(start + chunk_shots, expected_shots)
        manifest_path = chunk_dir / f"chunk-{chunk_index:05d}.json"
        verify_teacher_chunk(
            manifest_path,
            chunk_index=chunk_index,
            start=start,
            stop=stop,
            source=source,
            decoder=decoder,
        )
        chunks[str(manifest_path.relative_to(model_dir))] = sha256_file(manifest_path)
    return chunks


def load_verified_teacher_artifact(
    model_dir: Path,
    *,
    expected_metadata_sha256: str | None,
    expected_shots: int,
) -> dict[str, object]:
    """Verify the final packed teacher cache and its hash-bound metadata."""
    if type(expected_shots) is not int or expected_shots <= 0:
        raise ValueError("expected teacher shots must be a positive integer")
    metadata_path = model_dir / "teacher.json"
    if expected_metadata_sha256 is not None:
        verify_sha256(metadata_path, expected_metadata_sha256, label="teacher metadata")
    metadata = json.loads(metadata_path.read_text())
    expected_fields = {
        "bits_per_shot",
        "chunk_shots",
        "chunks",
        "decoder",
        "positive_counts_by_channel",
        "sha256",
        "shots",
        "source",
    }
    if set(metadata) != expected_fields:
        raise ValueError("teacher metadata fields do not match the declared schema")
    if metadata.get("bits_per_shot") != _TEACHER_BITS_PER_SHOT:
        raise ValueError("teacher bits_per_shot does not match lp_3_7_16")
    if type(metadata.get("shots")) is not int or metadata["shots"] != expected_shots:
        raise ValueError("teacher shots do not match training provenance")
    chunk_shots = metadata.get("chunk_shots")
    if type(chunk_shots) is not int or not 0 < chunk_shots <= 2_048:
        raise ValueError("teacher chunk_shots must be an integer between 1 and 2048")
    chunks = metadata.get("chunks")
    if (
        not isinstance(chunks, dict)
        or not chunks
        or any(
            not isinstance(path, str) or not isinstance(digest, str)
            for path, digest in chunks.items()
        )
    ):
        raise ValueError("teacher chunks must be a non-empty path-to-SHA-256 mapping")
    if not isinstance(metadata.get("decoder"), dict) or not isinstance(
        metadata.get("source"), dict
    ):
        raise TypeError("teacher decoder and source provenance must be objects")
    if not isinstance(metadata.get("sha256"), str):
        raise TypeError("teacher correction SHA-256 must be a string")
    positive_counts = metadata.get("positive_counts_by_channel")
    if (
        not isinstance(positive_counts, list)
        or len(positive_counts) != 58
        or any(type(count) is not int or count < 0 for count in positive_counts)
    ):
        raise ValueError("teacher positive counts must be 58 non-negative integers")

    verified_chunks = verify_teacher_chunks(
        model_dir,
        expected_shots=expected_shots,
        chunk_shots=chunk_shots,
        source=metadata["source"],
        decoder=metadata["decoder"],
    )
    if chunks != verified_chunks:
        raise ValueError("teacher metadata does not declare the exact verified chunks")

    cache_path = model_dir / "teacher_corrections.b8"
    if not cache_path.is_file():
        raise FileNotFoundError(f"teacher correction cache is missing: {cache_path}")
    expected_size = expected_shots * math.ceil(_TEACHER_BITS_PER_SHOT / 8)
    if cache_path.stat().st_size != expected_size:
        raise ValueError(
            f"teacher correction cache size mismatch: expected {expected_size} bytes, "
            f"found {cache_path.stat().st_size}"
        )
    verify_sha256(cache_path, str(metadata["sha256"]), label="teacher correction cache")
    return metadata


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
        shots = manifest.get("shots")
        if type(shots) is not int or shots <= 0:
            raise ValueError("campaign shard shots must be a positive integer")
        raw_dimensions = manifest.get("dimensions")
        if not isinstance(raw_dimensions, dict) or any(
            type(value) is not int or value < 0 for value in raw_dimensions.values()
        ):
            raise ValueError("shard dimensions must contain exact non-negative integers")
        dimensions = dict(raw_dimensions)
        if dimensions != {"dets.b8": 945, "errors.b8": 2610, "obs_actual.b8": 744}:
            raise ValueError("campaign shard dimensions do not match lp_3_7_16")
        for field, filename in (
            ("num_detectors", "dets.b8"),
            ("num_errors", "errors.b8"),
            ("num_observables", "obs_actual.b8"),
        ):
            value = manifest.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"shard {field} must be an exact non-negative integer")
            if value != dimensions[filename]:
                raise ValueError(f"shard {field} disagrees with dimensions.{filename}")
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
