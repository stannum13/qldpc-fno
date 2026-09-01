from __future__ import annotations

import argparse
from pathlib import Path

from qldpc_fno.campaign.evaluation import EvaluationRequest, evaluate_hybrid_campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--campaign-mode",
        choices=("canonical", "reduced_non_scientific"),
        default="canonical",
    )
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--max-batches-this-run", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate_hybrid_campaign(EvaluationRequest(**vars(args)))


if __name__ == "__main__":
    main()
