"""Validated configuration for a hybrid-decoder accuracy campaign."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    """Immutable accuracy-campaign policy loaded from a canonical JSON file."""

    campaign_seed: int
    noise_grid: tuple[float, ...]
    pilot_shots_per_point: int
    train_shots_cap: int
    calibration_shots_cap: int
    test_batch_shots: int
    max_test_shots_per_point: int
    target_failures: int
    training_epochs: int
    training_batch_size: int
    training_learning_rate: float
    training_seed: int
    checkpoint_every_epochs: int
    cloud_cpu: int
    cloud_memory: str
    cloud_timeout_seconds: int
    checkpoint_grace_seconds: int

    _INTEGER_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "campaign_seed",
            "pilot_shots_per_point",
            "train_shots_cap",
            "calibration_shots_cap",
            "test_batch_shots",
            "max_test_shots_per_point",
            "target_failures",
            "training_epochs",
            "training_batch_size",
            "training_seed",
            "checkpoint_every_epochs",
            "cloud_cpu",
            "cloud_timeout_seconds",
            "checkpoint_grace_seconds",
        }
    )
    _FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(__annotations__) - {
        "_INTEGER_FIELDS",
        "_FIELD_NAMES",
    }

    @classmethod
    def from_json(cls, path: Path) -> CampaignConfig:
        """Load and validate the exact campaign policy represented by *path*."""
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"unable to read campaign configuration: {path}") from error

        if not isinstance(payload, dict):
            raise TypeError("campaign configuration must be a JSON object")

        unknown_fields = set(payload) - cls._FIELD_NAMES
        missing_fields = cls._FIELD_NAMES - set(payload)
        if unknown_fields:
            raise ValueError(f"unknown fields: {sorted(unknown_fields)}")
        if missing_fields:
            raise ValueError(f"missing fields: {sorted(missing_fields)}")

        for field in cls._INTEGER_FIELDS:
            if type(payload[field]) is not int:
                raise ValueError(f"{field} must be an integer")
        if type(payload["training_learning_rate"]) not in (int, float):
            raise ValueError("training_learning_rate must be numeric")
        if not math.isfinite(payload["training_learning_rate"]):
            raise ValueError("training_learning_rate must be finite")
        if not isinstance(payload["cloud_memory"], str) or not payload["cloud_memory"]:
            raise ValueError("cloud_memory must be a non-empty string")
        if not isinstance(payload["noise_grid"], list) or not payload["noise_grid"]:
            raise ValueError("noise_grid must be a non-empty list")
        if any(type(probability) not in (int, float) for probability in payload["noise_grid"]):
            raise ValueError("noise_grid values must be numeric")
        if any(not math.isfinite(probability) for probability in payload["noise_grid"]):
            raise ValueError("noise_grid values must be finite")

        config = cls(
            noise_grid=tuple(payload["noise_grid"]),
            **{field: payload[field] for field in cls._FIELD_NAMES - {"noise_grid"}},
        )
        config._validate()
        return config

    def _validate(self) -> None:
        if any(getattr(self, field) <= 0 for field in self._INTEGER_FIELDS):
            raise ValueError("integer campaign limits and seeds must be positive")
        if self.training_learning_rate <= 0:
            raise ValueError("training_learning_rate must be positive")
        if any(probability <= 0 or probability >= 0.5 for probability in self.noise_grid):
            raise ValueError("noise_grid probabilities must be between 0 and 0.5")
        if any(left >= right for left, right in zip(self.noise_grid, self.noise_grid[1:])):
            raise ValueError("noise_grid probabilities must be strictly increasing")
        if self.target_failures > self.max_test_shots_per_point:
            raise ValueError("target_failures must not exceed max_test_shots_per_point")
        if self.checkpoint_grace_seconds >= self.cloud_timeout_seconds:
            raise ValueError("checkpoint_grace_seconds must be less than cloud_timeout_seconds")
