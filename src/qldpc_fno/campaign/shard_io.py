"""Strict verification and batch-wise access for immutable campaign shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256
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
    code_sha256 = sha256_file(code_manifest_path)
    shards: list[VerifiedShard] = []
    offset = 0
    expected_dimensions: dict[str, int] | None = None
    for relative in sorted(declared):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("shard manifest path escapes the role directory")
        manifest_path = root / relative_path
        verify_sha256(manifest_path, str(declared[relative]), label=f"{role} shard manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("role") != role or manifest.get("format") != "b8":
            raise ValueError(f"shard manifest is not a {role} b8 artifact")
        if manifest.get("path") != str(relative_path.parent):
            raise ValueError("shard manifest path provenance mismatch")
        shots = int(manifest["shots"])
        if shots <= 0:
            raise ValueError("campaign shards must contain positive shot counts")
        dimensions = {key: int(value) for key, value in manifest["dimensions"].items()}
        if dimensions != {"dets.b8": 945, "errors.b8": 2610, "obs_actual.b8": 744}:
            raise ValueError("campaign shard dimensions do not match lp_3_7_16")
        if expected_dimensions is not None and dimensions != expected_dimensions:
            raise ValueError("campaign shards have inconsistent dimensions")
        expected_dimensions = dimensions
        source = manifest.get("source_sha256")
        if not isinstance(source, dict):
            raise TypeError("shard manifest is missing source SHA-256 provenance")
        if source.get("config") != config_sha256:
            raise ValueError("campaign configuration SHA-256 mismatch in shard provenance")
        if source.get("code_manifest") != code_sha256:
            raise ValueError("code manifest SHA-256 mismatch in shard provenance")
        for filename, expected in manifest["sha256"].items():
            verify_sha256(manifest_path.parent / filename, str(expected), label=filename)
        verify_sha256(manifest_path.parent / "model.dem", str(source["dem"]), label="DEM")
        rate = float(manifest["error_rate"])
        if not 0.0 < rate < 0.5:
            raise ValueError("campaign shard error_rate must be between zero and 0.5")
        shards.append(
            VerifiedShard(
                directory=manifest_path.parent,
                manifest_path=manifest_path,
                manifest_sha256=sha256_file(manifest_path),
                start=offset,
                stop=offset + shots,
                error_rate=rate,
                dimensions=dimensions,
            )
        )
        offset += shots
    return VerifiedShardSet(root, role, sha256_file(completion_path), tuple(shards))
