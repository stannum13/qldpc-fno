from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stim

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.shards import select_noise_points, write_role_shards
from qldpc_fno.stim.b8 import read_b8


def _row(rate: float, errors: int, *, shots: int = 256) -> dict[str, object]:
    return {"error_rate": rate, "block_errors": errors, "shots": shots}


def test_selection_extends_one_point_beyond_zero_failures() -> None:
    rows = [_row(0.003, 0), _row(0.005, 0), _row(0.008, 4), _row(0.012, 20)]

    assert select_noise_points(rows)[:3] == (0.003, 0.005, 0.008)


def test_selection_retains_two_low_noise_controls() -> None:
    rows = [_row(0.003, 0), _row(0.005, 0), _row(0.008, 200)]

    assert select_noise_points(rows)[:2] == (0.003, 0.005)


def test_selection_inserts_midpoint_before_majority_failure() -> None:
    rows = [_row(0.003, 1), _row(0.005, 20), _row(0.009, 140)]

    assert select_noise_points(rows) == (0.003, 0.005, 0.007)


def test_selection_is_sorted_deduplicated_and_order_independent() -> None:
    rows = [_row(0.012, 180), _row(0.003, 0), _row(0.008, 40), _row(0.005, 2)]

    expected = (0.003, 0.005, 0.008, 0.01)
    assert select_noise_points(rows) == expected
    assert select_noise_points(reversed(rows)) == expected


def test_write_role_shards_are_role_separated_hashed_and_replayable(tmp_path: Path) -> None:
    rates = (0.1, 0.2)
    manifests: list[dict[str, object]] = []
    for role in ("pilot", "train", "calibration", "test"):
        role_dir = tmp_path / role
        role_manifests = write_role_shards(
            role=role,
            rates=rates,
            shots_per_rate=8,
            shard_size=8,
            campaign_seed=20260901,
            output_dir=role_dir,
            dem_factory=lambda rate: stim.DetectorErrorModel(f"error({rate}) D0 L0"),
            source_code_sha256="c" * 64,
        )
        manifests.extend(role_manifests)

    assert len({manifest["seed"] for manifest in manifests}) == len(manifests)
    for manifest in manifests:
        shard_dir = tmp_path / str(manifest["role"]) / str(manifest["path"])
        dem_path = shard_dir / "model.dem"
        dem = stim.DetectorErrorModel.from_file(dem_path)
        replay = dem.compile_sampler(seed=int(manifest["seed"])).sample(
            int(manifest["shots"]), return_errors=True
        )
        for filename, values in zip(
            ("dets.b8", "obs_actual.b8", "errors.b8"), replay, strict=True
        ):
            packed_path = shard_dir / filename
            assert sha256_file(packed_path) == manifest["sha256"][filename]
            bits = int(manifest["dimensions"][filename])
            assert np.array_equal(
                read_b8(packed_path, shots=int(manifest["shots"]), bits_per_shot=bits), values
            )


def test_write_role_shards_refuses_cross_role_path_and_completed_output(tmp_path: Path) -> None:
    kwargs = {
        "role": "train",
        "rates": (0.1,),
        "shots_per_rate": 1,
        "shard_size": 1,
        "campaign_seed": 9,
        "dem_factory": lambda rate: stim.DetectorErrorModel(f"error({rate}) D0 L0"),
        "source_code_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="role directory"):
        write_role_shards(output_dir=tmp_path / "test", **kwargs)

    output_dir = tmp_path / "train"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text(json.dumps({"complete": True}))
    with pytest.raises(FileExistsError, match="completion manifest"):
        write_role_shards(output_dir=output_dir, **kwargs)
