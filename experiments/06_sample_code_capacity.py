from __future__ import annotations

import argparse
from pathlib import Path

import stim

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
    dem = stim.DetectorErrorModel.from_file(args.dem)
    sample_dem_shard(dem, shots=args.shots, seed=args.seed, output_dir=args.out)


if __name__ == "__main__":
    main()
