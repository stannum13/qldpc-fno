from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256, write_canonical_json
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shards import select_noise_points, write_role_shards
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.decoders.bplsd import decode_bplsd_batch
from qldpc_fno.metrics.decoding import score_observable_predictions
from qldpc_fno.stim.b8 import read_b8
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
    logical_x = logical_x_basis(hx, hz)

    manifests = write_role_shards(
        role="pilot",
        rates=config.noise_grid,
        shots_per_rate=config.pilot_shots_per_point,
        shard_size=config.pilot_shots_per_point,
        campaign_seed=config.campaign_seed,
        output_dir=args.out,
        dem_factory=lambda rate: build_z_error_dem(hx, logical_x, error_rate=rate),
        source_code_sha256=sha256_file(code_manifest_path),
        source_artifact_sha256={"config": sha256_file(args.config)},
    )
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        shard_dir = args.out / str(manifest["path"])
        shots = int(manifest["shots"])
        syndromes = read_b8(shard_dir / "dets.b8", shots=shots, bits_per_shot=hx.shape[0])
        actual = read_b8(
            shard_dir / "obs_actual.b8", shots=shots, bits_per_shot=logical_x.shape[0]
        )
        result = decode_bplsd_batch(
            hx, syndromes, logical_x, error_rate=float(manifest["error_rate"])
        )
        score = score_observable_predictions(actual, result.predicted_observables)
        rows.append(
            {
                **score,
                "converged": int(np.count_nonzero(result.converged)),
                "convergence_rate": float(np.mean(result.converged)),
                "error_rate": manifest["error_rate"],
                "latency_mean_seconds": float(np.mean(result.latency_seconds)),
                "latency_seconds": float(np.sum(result.latency_seconds)),
                "syndrome_valid": int(np.count_nonzero(result.syndrome_valid)),
                "syndrome_valid_rate": float(np.mean(result.syndrome_valid)),
            }
        )

    write_canonical_json(
        args.out / "selection.json",
        {
            "pilot_rows": rows,
            "selected_noise_points": list(select_noise_points(rows)),
            "source_sha256": {
                "code_manifest": sha256_file(code_manifest_path),
                "config": sha256_file(args.config),
                "pilot_manifest": sha256_file(args.out / "manifest.json"),
            },
        },
    )


if __name__ == "__main__":
    main()
