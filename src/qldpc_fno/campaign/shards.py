"""Pilot-point selection and immutable, role-separated campaign shards."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import stim
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.campaign.seeds import derive_seed
from qldpc_fno.stim.b8 import write_b8

_ROLES = frozenset({"pilot", "train", "calibration", "test"})
_MAX_SHARD_SHOTS = 2_048
_PILOT_EXTENSION_FACTOR = 1.5
_PILOT_RATE_CAP = 0.08


def run_pilot_grid(
    configured_rates: Sequence[float],
    evaluate: Callable[[float, int], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Evaluate configured rates, extending geometrically while every result is zero."""
    rates = sorted({float(rate) for rate in configured_rates})
    if not rates:
        raise ValueError("at least one configured pilot rate is required")
    rows: list[dict[str, object]] = []
    for rate_index, rate in enumerate(rates):
        rows.append({**evaluate(rate, rate_index), "error_rate": rate, "rate_index": rate_index})

    while all(int(row["block_errors"]) == 0 for row in rows) and rates[-1] < _PILOT_RATE_CAP:
        next_rate = min(
            _PILOT_RATE_CAP,
            round(rates[-1] * _PILOT_EXTENSION_FACTOR, 12),
        )
        if next_rate <= rates[-1]:
            raise RuntimeError("pilot extension did not advance the noise rate")
        rate_index = len(rates)
        rates.append(next_rate)
        rows.append(
            {
                **evaluate(next_rate, rate_index),
                "error_rate": next_rate,
                "rate_index": rate_index,
            }
        )
    return rows


def allocate_total_shots(rates: Sequence[float], *, total_shots: int) -> tuple[int, ...]:
    """Split a total cap across sorted rates, assigning remainder to lower rates."""
    if total_shots <= 0:
        raise ValueError("total_shots must be positive")
    if not rates:
        raise ValueError("at least one rate is required")
    quotient, remainder = divmod(total_shots, len(rates))
    return tuple(quotient + (index < remainder) for index in range(len(rates)))


def validate_campaign_code(
    metadata: Mapping[str, object],
    hx: sparse.spmatrix,
    hz: sparse.spmatrix,
) -> None:
    """Reject artifacts other than the canonical lp_3_7_16 campaign code."""
    expected = {"name": "lp_3_7_16", "ell": 45, "n": 2610, "k": 744}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("campaign CLIs require the canonical lp_3_7_16 code metadata")
    if hx.shape != (945, 2610) or hz.shape != (945, 2610):
        raise ValueError("campaign code matrix dimensions must both be (945, 2610)")


@contextmanager
def atomic_role_directory(output_dir: Path, *, role: str) -> Iterator[Path]:
    """Stage a role tree privately and publish it only after its final manifest exists."""
    if output_dir.name != role:
        raise ValueError(f"output must be the {role!r} role directory")
    completion_path = output_dir / "manifest.json"
    if completion_path.exists():
        raise FileExistsError(f"completion manifest already exists: {completion_path}")
    if output_dir.exists():
        raise FileExistsError(f"campaign role output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{role}-staging-", dir=output_dir.parent))
    try:
        yield staging
        if not (staging / "manifest.json").is_file():
            raise RuntimeError("role staging completed without a final manifest")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


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
    shots_per_rate: int | Sequence[int],
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
    if shard_size <= 0 or shard_size > _MAX_SHARD_SHOTS:
        raise ValueError(f"shard_size must be between 1 and {_MAX_SHARD_SHOTS}")
    if not rates:
        raise ValueError("at least one noise rate is required")
    sorted_rates = tuple(sorted(float(rate) for rate in rates))
    if isinstance(shots_per_rate, int):
        shot_counts = (shots_per_rate,) * len(sorted_rates)
    else:
        shot_counts = tuple(shots_per_rate)
    if len(shot_counts) != len(sorted_rates) or any(shots <= 0 for shots in shot_counts):
        raise ValueError("shots_per_rate must provide one positive count per rate")
    for rate in sorted_rates:
        if not math.isfinite(rate) or not 0.0 < rate < 0.5:
            raise ValueError("noise rates must be finite and between 0 and 0.5")

    manifests: list[dict[str, object]] = []
    with atomic_role_directory(output_dir, role=role) as staging:
        manifest_paths: list[Path] = []
        for rate_index, (rate, rate_shots) in enumerate(zip(sorted_rates, shot_counts, strict=True)):
            for shard_index, offset in enumerate(range(0, rate_shots, shard_size)):
                shots = min(shard_size, rate_shots - offset)
                seed = derive_seed(
                    campaign_seed, p_index=rate_index, role=role, shard_index=shard_index
                )
                shard_path = Path(f"rate-{rate_index:03d}") / f"shard-{shard_index:05d}"
                shard_dir = staging / shard_path
                dem = dem_factory(rate)
                shard_dir.mkdir(parents=True)
                dem_path = shard_dir / "model.dem"
                dem.to_file(dem_path)
                dets, obs, errors = dem.compile_sampler(seed=seed).sample(
                    shots, bit_packed=False, return_errors=True
                )
                if errors is None:
                    raise RuntimeError("Stim did not return requested error-mechanism labels")
                packed = {
                    "dets.b8": np.asarray(dets, dtype=np.uint8),
                    "errors.b8": np.asarray(errors, dtype=np.uint8),
                    "obs_actual.b8": np.asarray(obs, dtype=np.uint8),
                }
                for filename, values in packed.items():
                    write_b8(shard_dir / filename, values)
                manifest: dict[str, object] = {
                    "bit_order": "little",
                    "dimensions": {
                        "dets.b8": dem.num_detectors,
                        "errors.b8": dem.num_errors,
                        "obs_actual.b8": dem.num_observables,
                    },
                    "error_rate": rate,
                    "format": "b8",
                    "num_detectors": dem.num_detectors,
                    "num_errors": dem.num_errors,
                    "num_observables": dem.num_observables,
                    "path": str(shard_path),
                    "rate_index": rate_index,
                    "role": role,
                    "seed": seed,
                    "shard_index": shard_index,
                    "sha256": {
                        filename: sha256_file(shard_dir / filename) for filename in packed
                    },
                    "shots": shots,
                    "source_sha256": {
                        **(source_artifact_sha256 or {}),
                        "code_manifest": source_code_sha256,
                        "dem": sha256_file(dem_path),
                    },
                    "stim_version": stim.__version__,
                }
                manifest_path = shard_dir / "samples.json"
                write_canonical_json(manifest_path, manifest)
                manifests.append(manifest)
                manifest_paths.append(manifest_path)

        write_canonical_json(
            staging / "manifest.json",
            {
                "complete": True,
                "role": role,
                "shards": {
                    str(path.relative_to(staging)): sha256_file(path) for path in manifest_paths
                },
            },
        )
    return manifests
