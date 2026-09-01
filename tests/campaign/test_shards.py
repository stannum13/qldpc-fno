from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import stim
from scipy import sparse

from qldpc_fno.artifacts import sha256_file
from qldpc_fno.campaign.shards import (
    allocate_total_shots,
    run_pilot_grid,
    sample_pilot_point_shards,
    select_noise_points,
    validate_campaign_code,
    write_role_shards,
)
from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import PAPER_LP_3_7_16
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


def test_all_zero_pilot_grid_extends_geometrically_to_exact_cap() -> None:
    observed: list[tuple[float, int]] = []

    def evaluate(rate: float, rate_index: int) -> dict[str, object]:
        observed.append((rate, rate_index))
        return _row(rate, 0)

    rows = run_pilot_grid((0.003, 0.025), evaluate)

    assert [(row["error_rate"], row["rate_index"]) for row in rows] == [
        (0.003, 0),
        (0.025, 1),
        (0.0375, 2),
        (0.05625, 3),
        (0.08, 4),
    ]
    assert observed == [(0.003, 0), (0.025, 1), (0.0375, 2), (0.05625, 3), (0.08, 4)]


def test_total_shots_are_allocated_by_sorted_rate_with_remainder_first() -> None:
    with pytest.raises(ValueError, match="sorted"):
        allocate_total_shots((0.2, 0.1, 0.3), total_shots=8)
    assert allocate_total_shots((0.1, 0.2, 0.3), total_shots=8) == (3, 3, 2)


def test_allocated_shard_manifests_sum_to_exact_total(tmp_path: Path) -> None:
    rates = (0.1, 0.2, 0.3)
    manifests = write_role_shards(
        role="train",
        rates=rates,
        shots_per_rate=allocate_total_shots(rates, total_shots=8),
        shard_size=2,
        campaign_seed=7,
        output_dir=tmp_path / "train",
        dem_factory=lambda rate: stim.DetectorErrorModel(f"error({rate}) D0 L0"),
        source_code_sha256="c" * 64,
    )

    assert sum(int(manifest["shots"]) for manifest in manifests) == 8
    assert [manifest["error_rate"] for manifest in manifests if manifest["shard_index"] == 0] == [
        0.1,
        0.2,
        0.3,
    ]


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


def test_write_role_shards_failure_leaves_no_published_output_and_can_retry(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "train"
    calls = 0

    def fail_second_rate(rate: float) -> stim.DetectorErrorModel:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return stim.DetectorErrorModel(f"error({rate}) D0 L0")

    kwargs = {
        "role": "train",
        "rates": (0.1, 0.2),
        "shots_per_rate": 1,
        "shard_size": 1,
        "campaign_seed": 9,
        "source_code_sha256": "c" * 64,
    }
    with pytest.raises(RuntimeError, match="injected failure"):
        write_role_shards(output_dir=output_dir, dem_factory=fail_second_rate, **kwargs)
    assert not output_dir.exists()

    manifests = write_role_shards(
        output_dir=output_dir,
        dem_factory=lambda rate: stim.DetectorErrorModel(f"error({rate}) D0 L0"),
        **kwargs,
    )
    assert len(manifests) == 2
    assert (output_dir / "manifest.json").is_file()


def test_campaign_code_validation_requires_canonical_metadata_and_shapes() -> None:
    canonical = {
        "name": "lp_3_7_16",
        "ell": 45,
        "n": 2610,
        "k": 744,
    }
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    validate_campaign_code(canonical, code.hx, code.hz)

    with pytest.raises(ValueError, match="canonical lp_3_7_16"):
        validate_campaign_code({**canonical, "name": "tiny"}, code.hx, code.hz)
    with pytest.raises(ValueError, match="matrix dimensions"):
        validate_campaign_code(canonical, sparse.csr_matrix((1, 2)), code.hz)


def test_campaign_code_validation_rejects_spoofed_same_shape_css_code() -> None:
    canonical = {"name": "lp_3_7_16", "ell": 45, "n": 2610, "k": 744}
    spoofed_hx = sparse.csr_matrix((945, 2610), dtype=np.uint8)
    spoofed_hz = sparse.csr_matrix((945, 2610), dtype=np.uint8)

    with pytest.raises(ValueError, match="matrix identity"):
        validate_campaign_code(canonical, spoofed_hx, spoofed_hz)


def test_pilot_point_over_2048_shots_is_split_and_aggregated(tmp_path: Path) -> None:
    dem = stim.DetectorErrorModel("error(0.1) D0 L0")

    manifests, detections, observables = sample_pilot_point_shards(
        dem=dem,
        rate=0.1,
        rate_index=2,
        shots=2_049,
        campaign_seed=17,
        staging=tmp_path,
        source_code_sha256="c" * 64,
        source_artifact_sha256={"config": "f" * 64},
    )

    assert [manifest["shots"] for manifest in manifests] == [2_048, 1]
    assert [manifest["shard_index"] for manifest in manifests] == [0, 1]
    assert len({manifest["seed"] for manifest in manifests}) == 2
    assert detections.shape == (2_049, 1)
    assert observables.shape == (2_049, 1)
