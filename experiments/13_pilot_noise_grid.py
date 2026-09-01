from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shards import (
    atomic_role_directory,
    run_pilot_grid,
    sample_pilot_point_shards,
    select_noise_points,
    validate_campaign_code,
)
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.metrics.decoding import score_observable_predictions
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
            manifests, dets, obs = sample_pilot_point_shards(
                dem=dem,
                rate=rate,
                rate_index=rate_index,
                shots=config.pilot_shots_per_point,
                campaign_seed=config.campaign_seed,
                staging=staging,
                source_code_sha256=code_manifest_sha256,
                source_artifact_sha256={"config": config_sha256},
            )
            shard_manifest_paths.extend(
                staging / str(manifest["path"]) / "samples.json" for manifest in manifests
            )

            result = decode_bplsd_batch(hx, dets, logical_x, error_rate=rate)
            score = score_observable_predictions(
                obs,
                result.predicted_observables,
                syndrome_valid=result.syndrome_valid,
            )
            return {
                **score,
                "converged": int(np.count_nonzero(result.converged)),
                "convergence_rate": float(np.mean(result.converged)),
                "latency_mean_seconds": float(np.mean(result.latency_seconds)),
                "latency_seconds": float(np.sum(result.latency_seconds)),
                "seeds": [manifest["seed"] for manifest in manifests],
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
