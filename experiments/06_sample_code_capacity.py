from __future__ import annotations

import argparse
import json
from pathlib import Path

import stim

from qldpc_fno.artifacts import verify_sha256, write_canonical_json
from qldpc_fno.stim.sample import sample_dem_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if not args.dem.is_file():
        raise FileNotFoundError(f"DEM file does not exist or is not a file: {args.dem}")
    manifest_path = args.dem.with_name("dem.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DEM manifest does not exist: {manifest_path}")
    metadata = json.loads(manifest_path.read_text())
    verify_sha256(args.dem, metadata["model_sha256"], label="detector error model")
    dem = stim.DetectorErrorModel.from_file(args.dem)
    manifest = sample_dem_shard(dem, shots=args.shots, seed=args.seed, output_dir=args.out)
    manifest["source_dem_sha256"] = metadata["model_sha256"]
    write_canonical_json(args.out / "samples.json", manifest)


if __name__ == "__main__":
    main()
