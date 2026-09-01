from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shards import write_role_shards
from qldpc_fno.codes.gf2 import logical_x_basis
from qldpc_fno.stim.dem import build_z_error_dem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "calibration", "test"), required=True)
    parser.add_argument("--shots-per-rate", type=int)
    parser.add_argument("--shard-size", type=int, default=2_048)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = CampaignConfig.from_json(args.config)
    selection = json.loads(args.selection.read_text())
    rates = selection["selected_noise_points"]
    if not isinstance(rates, list) or not rates:
        raise ValueError("selection must contain non-empty selected_noise_points")

    code_manifest_path = args.code / "code.json"
    code_metadata = json.loads(code_manifest_path.read_text())
    hx_path = args.code / "hx.npz"
    hz_path = args.code / "hz.npz"
    verify_sha256(hx_path, code_metadata["hx_sha256"], label="Hx")
    verify_sha256(hz_path, code_metadata["hz_sha256"], label="Hz")
    hx = sparse.load_npz(hx_path).tocsr()
    hz = sparse.load_npz(hz_path).tocsr()
    logical_x = logical_x_basis(hx, hz)

    default_shots = {
        "train": config.train_shots_cap,
        "calibration": config.calibration_shots_cap,
        "test": config.test_batch_shots,
    }
    shots_per_rate = args.shots_per_rate or default_shots[args.role]
    if shots_per_rate > default_shots[args.role]:
        raise ValueError(f"shots_per_rate exceeds the configured {args.role} limit")
    write_role_shards(
        role=args.role,
        rates=rates,
        shots_per_rate=shots_per_rate,
        shard_size=args.shard_size,
        campaign_seed=config.campaign_seed,
        output_dir=args.out,
        dem_factory=lambda rate: build_z_error_dem(hx, logical_x, error_rate=rate),
        source_code_sha256=sha256_file(code_manifest_path),
        source_artifact_sha256={
            "config": sha256_file(args.config),
            "selection": sha256_file(args.selection),
        },
    )


if __name__ == "__main__":
    main()
