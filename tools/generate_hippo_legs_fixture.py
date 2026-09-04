"""Generate the independent float64 HiPPO-LegS recurrence fixture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _generator(order: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(order, dtype=np.float64)
    scale = np.sqrt(2.0 * indices + 1.0)
    generator = np.zeros((order, order), dtype=np.float64)
    for row in range(order):
        generator[row, row] = -(row + 1.0)
        for column in range(row):
            generator[row, column] = -scale[row] * scale[column]
    return generator, scale


def _fixture() -> dict[str, object]:
    order = 4
    inputs = np.asarray([0.25, -0.5, 1.25, 0.75, -1.0], dtype=np.float64)
    generator, input_vector = _generator(order)
    identity = np.eye(order, dtype=np.float64)
    state = np.zeros(order, dtype=np.float64)
    records: list[dict[str, object]] = []

    for completed_sample_index, value in enumerate(inputs, start=1):
        scaled_generator = generator / completed_sample_index
        scaled_input = input_vector / completed_sample_index
        lhs = identity - scaled_generator / 2.0
        transition = np.linalg.solve(lhs, identity + scaled_generator / 2.0)
        injection = np.linalg.solve(lhs, scaled_input)
        state = transition @ state + injection * value
        records.append(
            {
                "abar": transition.tolist(),
                "bbar": injection.tolist(),
                "completed_sample_index": completed_sample_index,
                "state": state.tolist(),
            }
        )

    return {
        "description": "Independent float64 bilinear HiPPO-LegS recurrence oracle",
        "generator": generator.tolist(),
        "input_vector": input_vector.tolist(),
        "inputs": inputs.tolist(),
        "order": order,
        "records": records,
        "schema_version": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(float(np.finfo(np.float64).max)):
        raise RuntimeError("float64 is unavailable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_fixture(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
