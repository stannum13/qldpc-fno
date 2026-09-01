"""Pilot-point selection and immutable, role-separated campaign shards."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import stim

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.seeds import derive_seed
from qldpc_fno.stim.sample import sample_dem_shard

_ROLES = frozenset({"pilot", "train", "calibration", "test"})
_MAX_SHARD_SHOTS = 2_048


def select_noise_points(pilot_rows: Iterable[Mapping[str, object]]) -> Sequence[float]:
    """Select a deterministic useful noise range from baseline pilot results.

    The two lowest-noise points remain as controls. The selected range extends one
    measured point beyond a zero-failure prefix, then retains measured points up to
    a 50% baseline block-error rate. At the first majority-failure point, its
    midpoint with the preceding point is used instead.
    """
    parsed: dict[float, tuple[int, int]] = {}
    for row in pilot_rows:
        rate = float(row["error_rate"])
        errors = int(row["block_errors"])
        shots = int(row["shots"])
        if not math.isfinite(rate) or not 0.0 < rate < 0.5:
            raise ValueError("pilot error rates must be finite and between 0 and 0.5")
        if shots <= 0 or errors < 0 or errors > shots:
            raise ValueError("pilot error counts must be between zero and shots")
        result = (errors, shots)
        if rate in parsed and parsed[rate] != result:
            raise ValueError(f"conflicting pilot rows for error rate {rate}")
        parsed[rate] = result
    if not parsed:
        raise ValueError("at least one pilot row is required")

    rows = [(rate, *parsed[rate]) for rate in sorted(parsed)]
    selected = {rate for rate, _, _ in rows[:2]}

    last_zero = -1
    for index, (_, errors, _) in enumerate(rows):
        if errors == 0:
            last_zero = index
        else:
            break
    if last_zero >= 0:
        selected.update(rate for rate, _, _ in rows[: min(last_zero + 2, len(rows))])

    for index, (rate, errors, shots) in enumerate(rows):
        if errors / shots <= 0.5:
            selected.add(rate)
            continue
        if index > 0:
            selected.add(round((rows[index - 1][0] + rate) / 2, 15))
        break
    return tuple(sorted(selected))


def write_role_shards(
    *,
    role: str,
    rates: Sequence[float],
    shots_per_rate: int,
    shard_size: int,
    campaign_seed: int,
    output_dir: Path,
    dem_factory: Callable[[float], stim.DetectorErrorModel],
    source_code_sha256: str,
    source_artifact_sha256: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Sample and publish one immutable completion manifest for a campaign role."""
    if role not in _ROLES:
        raise ValueError(f"unsupported campaign shard role: {role}")
    if output_dir.name != role:
        raise ValueError(f"output must be the {role!r} role directory")
    completion_path = output_dir / "manifest.json"
    if completion_path.exists():
        raise FileExistsError(f"completion manifest already exists: {completion_path}")
    if shots_per_rate <= 0:
        raise ValueError("shots_per_rate must be positive")
    if shard_size <= 0 or shard_size > _MAX_SHARD_SHOTS:
        raise ValueError(f"shard_size must be between 1 and {_MAX_SHARD_SHOTS}")
    if not rates:
        raise ValueError("at least one noise rate is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, object]] = []
    manifest_paths: list[Path] = []
    for rate_index, rate_value in enumerate(rates):
        rate = float(rate_value)
        if not math.isfinite(rate) or not 0.0 < rate < 0.5:
            raise ValueError("noise rates must be finite and between 0 and 0.5")
        for shard_index, offset in enumerate(range(0, shots_per_rate, shard_size)):
            shots = min(shard_size, shots_per_rate - offset)
            seed = derive_seed(
                campaign_seed, p_index=rate_index, role=role, shard_index=shard_index
            )
            shard_path = Path(f"rate-{rate_index:03d}") / f"shard-{shard_index:05d}"
            shard_dir = output_dir / shard_path
            if (shard_dir / "samples.json").exists():
                raise FileExistsError(f"shard manifest already exists: {shard_dir / 'samples.json'}")
            dem = dem_factory(rate)
            shard_dir.mkdir(parents=True, exist_ok=True)
            dem_path = shard_dir / "model.dem"
            dem.to_file(dem_path)
            sampled = sample_dem_shard(dem, shots=shots, seed=seed, output_dir=shard_dir)
            manifest: dict[str, object] = {
                **sampled,
                "dimensions": {
                    "dets.b8": dem.num_detectors,
                    "errors.b8": dem.num_errors,
                    "obs_actual.b8": dem.num_observables,
                },
                "error_rate": rate,
                "path": str(shard_path),
                "rate_index": rate_index,
                "role": role,
                "shard_index": shard_index,
                "sha256": {
                    "dets.b8": sampled["sha256"]["detections"],
                    "errors.b8": sampled["sha256"]["errors"],
                    "obs_actual.b8": sampled["sha256"]["observables_actual"],
                },
                "source_sha256": {
                    **(source_artifact_sha256 or {}),
                    "code_manifest": source_code_sha256,
                    "dem": sha256_file(dem_path),
                },
            }
            manifest_path = shard_dir / "samples.json"
            write_canonical_json(manifest_path, manifest)
            manifests.append(manifest)
            manifest_paths.append(manifest_path)

    write_canonical_json(
        completion_path,
        {
            "complete": True,
            "role": role,
            "shards": {
                str(path.relative_to(output_dir)): sha256_file(path) for path in manifest_paths
            },
        },
    )
    return manifests
