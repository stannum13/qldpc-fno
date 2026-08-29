from __future__ import annotations

from pathlib import Path

import stim

from qldpc_fno.artifacts import sha256_file, write_canonical_json
from qldpc_fno.stim.b8 import write_b8


def sample_dem_shard(
    dem: stim.DetectorErrorModel,
    *,
    shots: int,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Sample a DEM once and persist every label required for exact replay."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    dets, obs, errors = dem.compile_sampler(seed=seed).sample(
        shots,
        bit_packed=False,
        return_errors=True,
    )
    if errors is None:
        raise RuntimeError("Stim did not return requested error-mechanism labels")
    paths = {
        "detections": output_dir / "dets.b8",
        "observables_actual": output_dir / "obs_actual.b8",
        "errors": output_dir / "errors.b8",
    }
    write_b8(paths["detections"], dets)
    write_b8(paths["observables_actual"], obs)
    write_b8(paths["errors"], errors)
    manifest: dict[str, object] = {
        "bit_order": "little",
        "format": "b8",
        "num_detectors": dem.num_detectors,
        "num_errors": dem.num_errors,
        "num_observables": dem.num_observables,
        "seed": seed,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "shots": shots,
        "stim_version": stim.__version__,
    }
    write_canonical_json(output_dir / "samples.json", manifest)
    return manifest
