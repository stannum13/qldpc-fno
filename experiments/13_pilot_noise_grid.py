from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import stim
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.seeds import derive_seed
from qldpc_fno.campaign.shards import (
    atomic_role_directory,
    run_pilot_grid,
    select_noise_points,
    validate_campaign_code,
)
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.metrics.decoding import score_observable_predictions
from qldpc_fno.stim.b8 import write_b8
from qldpc_fno.stim.dem import build_z_error_dem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    code_manifest_path = args.code / "code.json"
    code_metadata = json.loads(code_manifest_path.read_text())
    hx_path = args.code / "hx.npz"
    hz_path = args.code / "hz.npz"
    verify_sha256(hx_path, code_metadata["hx_sha256"], label="Hx")
    verify_sha256(hz_path, code_metadata["hz_sha256"], label="Hz")
    hx = sparse.load_npz(hx_path).tocsr()
    hz = sparse.load_npz(hz_path).tocsr()
    validate_campaign_code(code_metadata, hx, hz)
    logical_x = logical_x_basis(hx, hz)
    if logical_x.shape != (744, 2610):
        raise ValueError("campaign logical X matrix dimensions must be (744, 2610)")

    code_manifest_sha256 = sha256_file(code_manifest_path)
    config_sha256 = sha256_file(args.config)
    with atomic_role_directory(args.out, role="pilot") as staging:
        shard_manifest_paths: list[Path] = []

        def evaluate(rate: float, rate_index: int) -> dict[str, object]:
            dem = build_z_error_dem(hx, logical_x, error_rate=rate)
            seed = derive_seed(
                config.campaign_seed,
                p_index=rate_index,
                role="pilot",
                shard_index=0,
            )
            shard_path = Path(f"rate-{rate_index:03d}") / "shard-00000"
            shard_dir = staging / shard_path
            shard_dir.mkdir(parents=True)
            dem_path = shard_dir / "model.dem"
            dem.to_file(dem_path)
            dets, obs, errors = dem.compile_sampler(seed=seed).sample(
                config.pilot_shots_per_point,
                bit_packed=False,
                return_errors=True,
            )
            if errors is None:
                raise RuntimeError("Stim did not return requested error-mechanism labels")
            packed = {
                "dets.b8": dets,
                "errors.b8": errors,
                "obs_actual.b8": obs,
            }
            for filename, values in packed.items():
                write_b8(shard_dir / filename, values)
            shard_manifest = {
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
                "role": "pilot",
                "seed": seed,
                "shard_index": 0,
                "sha256": {
                    filename: sha256_file(shard_dir / filename) for filename in packed
                },
                "shots": config.pilot_shots_per_point,
                "source_sha256": {
                    "code_manifest": code_manifest_sha256,
                    "config": config_sha256,
                    "dem": sha256_file(dem_path),
                },
                "stim_version": stim.__version__,
            }
            shard_manifest_path = shard_dir / "samples.json"
            write_canonical_json(shard_manifest_path, shard_manifest)
            shard_manifest_paths.append(shard_manifest_path)

            result = decode_bplsd_batch(hx, dets, logical_x, error_rate=rate)
            score = score_observable_predictions(obs, result.predicted_observables)
            return {
                **score,
                "converged": int(np.count_nonzero(result.converged)),
                "convergence_rate": float(np.mean(result.converged)),
                "latency_mean_seconds": float(np.mean(result.latency_seconds)),
                "latency_seconds": float(np.sum(result.latency_seconds)),
                "seed": seed,
                "syndrome_valid": int(np.count_nonzero(result.syndrome_valid)),
                "syndrome_valid_rate": float(np.mean(result.syndrome_valid)),
            }

        rows = run_pilot_grid(config.noise_grid, evaluate)
        selection_path = staging / "selection.json"
        write_canonical_json(
            selection_path,
            {
                "pilot_rows": rows,
                "selected_noise_points": list(select_noise_points(rows)),
                "source_sha256": {
                    "code_manifest": code_manifest_sha256,
                    "config": config_sha256,
                },
            },
        )
        write_canonical_json(
            staging / "manifest.json",
            {
                "complete": True,
                "role": "pilot",
                "selection_sha256": sha256_file(selection_path),
                "shards": {
                    str(path.relative_to(staging)): sha256_file(path)
                    for path in shard_manifest_paths
                },
            },
        )


if __name__ == "__main__":
    main()
