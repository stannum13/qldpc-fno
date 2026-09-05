"""Generate an independent float64 oracle for scalar syndrome likelihood tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

_FIXTURE_PATH = Path("tests/identifiability/fixtures/scalar_likelihood_float64.json")
_BASE_PROBABILITY = 0.0375
_STATES = np.array([-1.2, -0.4, 0.0, 0.7, 1.2], dtype=np.float64)
_WEIGHTS = np.array([2, 3, 10], dtype=np.int64)
_SYNDROME = np.array([1, 0, 1], dtype=np.uint8)
_STEP = 1e-6


def _physical_probability(states: np.ndarray) -> np.ndarray:
    base_logit = math.log(_BASE_PROBABILITY / (1.0 - _BASE_PROBABILITY))
    return np.clip(expit(base_logit + states), 1e-5, 0.25)


def _parity_probability(probability: np.ndarray, weight: int) -> np.ndarray:
    return 0.5 * (1.0 - np.power(1.0 - 2.0 * probability, weight))


def _derivative(states: np.ndarray, weight: int) -> np.ndarray:
    probability = _physical_probability(states)
    return (
        weight * np.power(1.0 - 2.0 * probability, weight - 1) * probability * (1.0 - probability)
    )


def _payload() -> dict[str, object]:
    probabilities = _physical_probability(_STATES)
    parity = np.stack([_parity_probability(probabilities, int(weight)) for weight in _WEIGHTS])
    analytic = np.stack([_derivative(_STATES, int(weight)) for weight in _WEIGHTS])
    finite_difference = np.stack(
        [
            (
                _parity_probability(_physical_probability(_STATES + _STEP), int(weight))
                - _parity_probability(_physical_probability(_STATES - _STEP), int(weight))
            )
            / (2.0 * _STEP)
            for weight in _WEIGHTS
        ]
    )
    likelihood = np.sum(
        np.where(_SYNDROME[:, None] == 1, np.log(parity), np.log1p(-parity)), axis=0
    )
    fisher = np.sum(analytic**2 / (parity * (1.0 - parity)), axis=0)
    return {
        "schema_version": 1,
        "implementation": "independent_float64_direct_formula/v1",
        "base_probability": _BASE_PROBABILITY,
        "finite_difference_step": _STEP,
        "states": _STATES.tolist(),
        "physical_probabilities": probabilities.tolist(),
        "weights": _WEIGHTS.tolist(),
        "syndrome": _SYNDROME.tolist(),
        "parity_one_probabilities": parity.tolist(),
        "log_likelihood": likelihood.tolist(),
        "analytic_derivatives": analytic.tolist(),
        "finite_difference_derivatives": finite_difference.tolist(),
        "fisher_information": fisher.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(_payload(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not _FIXTURE_PATH.is_file() or _FIXTURE_PATH.read_text() != rendered:
            return 1
        return 0
    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
