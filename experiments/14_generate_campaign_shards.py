from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import sparse

from qldpc_fno.artifacts import sha256_file, verify_sha256
from qldpc_fno.campaign.config import CampaignConfig
from qldpc_fno.campaign.shards import (
    allocate_total_shots,
    validate_campaign_code,
    write_role_shards,
)
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
    code_manifest_path = args.code / "code.json"
    code_manifest_sha256 = sha256_file(code_manifest_path)
    pilot_manifest_path = args.selection.parent / "manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text())
    if pilot_manifest.get("complete") is not True or pilot_manifest.get("role") != "pilot":
        raise ValueError("selection does not belong to a completed pilot publication")
    verify_sha256(
        args.selection,
        str(pilot_manifest["selection_sha256"]),
        label="selection",
    )
    selection = json.loads(args.selection.read_text())
    selection_sources = selection.get("source_sha256")
    if not isinstance(selection_sources, dict):
        raise TypeError("selection is missing source SHA-256 provenance")
    if selection_sources.get("config") != sha256_file(args.config):
        raise ValueError("campaign configuration SHA-256 mismatch in selection provenance")
    if selection_sources.get("code_manifest") != code_manifest_sha256:
        raise ValueError("code manifest SHA-256 mismatch in selection provenance")
    rates = selection["selected_noise_points"]
    if not isinstance(rates, list) or not rates:
        raise ValueError("selection must contain non-empty selected_noise_points")
    rates = sorted(float(rate) for rate in rates)

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

    default_shots = {
        "train": config.train_shots_cap,
        "calibration": config.calibration_shots_cap,
        "test": config.max_test_shots_per_point,
    }
    requested_shots = (
        args.shots_per_rate if args.shots_per_rate is not None else default_shots[args.role]
    )
    if requested_shots <= 0 or requested_shots > default_shots[args.role]:
        raise ValueError(f"requested shots exceed the configured {args.role} limit")
    if args.role in {"train", "calibration"}:
        shot_counts: int | tuple[int, ...] = allocate_total_shots(
            rates, total_shots=requested_shots
        )
    else:
        shot_counts = requested_shots
    write_role_shards(
        role=args.role,
        rates=rates,
        shots_per_rate=shot_counts,
        shard_size=args.shard_size,
        campaign_seed=config.campaign_seed,
        output_dir=args.out,
        dem_factory=lambda rate: build_z_error_dem(hx, logical_x, error_rate=rate),
        source_code_sha256=code_manifest_sha256,
        source_artifact_sha256={
            "config": sha256_file(args.config),
            "pilot_manifest": sha256_file(pilot_manifest_path),
            "selection": sha256_file(args.selection),
        },
    )


if __name__ == "__main__":
    main()
