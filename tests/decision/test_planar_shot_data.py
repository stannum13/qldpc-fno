from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qecsim import paulitools as pt
from qecsim.models.generic import DepolarizingErrorModel
from qecsim.models.planar import PlanarCode

from qldpc_fno.decision.planar_shot_data import generate_planar_shots


def _config(tmp_path: Path, domain: str, *, shots_per_rate: int = 2) -> Path:
    payload = {
        "schema_version": 1,
        "seed_domain": domain,
        "campaign_seed": int.from_bytes(hashlib.sha256(domain.encode()).digest()[:8], "big"),
        "code_distance": 5,
        "error_rates": [0.1, 0.15],
        "shots_per_rate": shots_per_rate,
        "noise_model": "qecsim_iid_depolarizing_code_capacity",
    }
    path = tmp_path / f"{domain.replace('/', '-')}.json"
    path.write_text(json.dumps(payload))
    return path


def test_generator_validates_seed_code_rates_count_and_noise_model(tmp_path: Path) -> None:
    config = _config(tmp_path, "qldpc-fno/planar-shot-test/validation/v1")
    payload = json.loads(config.read_text())

    invalid_values = (
        ("campaign_seed", payload["campaign_seed"] + 1),
        ("code_distance", 3),
        ("error_rates", [0.1]),
        ("shots_per_rate", 0),
        ("noise_model", "unsupported"),
    )
    for key, value in invalid_values:
        invalid = {**payload, key: value}
        invalid_path = tmp_path / f"invalid-{key}.json"
        invalid_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            generate_planar_shots(invalid_path, tmp_path / f"out-{key}")


def test_generator_records_replayable_physical_errors_and_relative_provenance(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1")
    output_dir = tmp_path / "shots"

    payload = generate_planar_shots(config, output_dir)

    assert payload["config"] == json.loads(config.read_text())
    assert (output_dir / "planar_shots.json").exists()
    assert len(payload["shots"]) == 4
    assert len({row["sampler_seed"] for row in payload["shots"]}) == 4
    assert {
        "shot_id",
        "error_rate",
        "shot_index",
        "sampler_seed",
        "error_bsf",
        "syndrome",
    } <= set(payload["shots"][0])

    model = DepolarizingErrorModel()
    code = PlanarCode(5, 5)
    for row in payload["shots"]:
        expected_seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"qldpc-fno/planar-shot-test/calibration/v1|d5|"
                    f"p{row['error_rate']:.6f}|i{row['shot_index']:06d}"
                ).encode()
            ).digest()[:8],
            "big",
        )
        assert row["sampler_seed"] == expected_seed
        error = model.generate(code, row["error_rate"], np.random.default_rng(row["sampler_seed"]))
        assert error.tolist() == row["error_bsf"]
        assert pt.bsp(error, code.stabilizers.T).tolist() == row["syndrome"]

    source_hashes = payload["provenance"]["source_sha256"]
    assert "src/qldpc_fno/decision/planar_shot_data.py" in source_hashes
    assert all(not Path(label).is_absolute() for label in source_hashes)


def test_generator_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    config = _config(tmp_path, "qldpc-fno/planar-shot-test/no-overwrite/v1", shots_per_rate=1)
    output_dir = tmp_path / "shots"
    generate_planar_shots(config, output_dir)

    with pytest.raises(FileExistsError):
        generate_planar_shots(config, output_dir)


def test_calibration_and_screen_domains_have_disjoint_shot_seeds(tmp_path: Path) -> None:
    calibration = _config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1", shots_per_rate=1)
    screen = _config(tmp_path, "qldpc-fno/planar-shot-test/screen/v1", shots_per_rate=1)

    calibration_payload = generate_planar_shots(calibration, tmp_path / "calibration")
    screen_payload = generate_planar_shots(screen, tmp_path / "screen")

    calibration_seeds = {row["sampler_seed"] for row in calibration_payload["shots"]}
    screen_seeds = {row["sampler_seed"] for row in screen_payload["shots"]}
    assert calibration_seeds.isdisjoint(screen_seeds)


def test_generator_is_byte_replayable(tmp_path: Path) -> None:
    config = _config(tmp_path, "qldpc-fno/planar-shot-test/replay/v1", shots_per_rate=1)
    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_planar_shots(config, first)
    generate_planar_shots(config, second)

    assert (first / "planar_shots.json").read_bytes() == (second / "planar_shots.json").read_bytes()


def test_shots_freeze_complete_science_config_and_matching_backend(tmp_path: Path) -> None:
    import networkx
    from qecsim.graphtools import blossom5

    config = _config(tmp_path, "qldpc-fno/planar-shot-test/calibration/v1")
    payload = generate_planar_shots(config, tmp_path / "shots")
    provenance = payload["provenance"]
    assert set(provenance["source_sha256"]) >= {
        "src/qldpc_fno/decision/planar_shot_data.py",
        "src/qldpc_fno/decision/planar_shot_accuracy.py",
        "src/qldpc_fno/decision/tensor_network.py",
        "src/qldpc_fno/metrics/paired.py",
        "src/qldpc_fno/artifacts.py",
    }
    assert provenance["config_content"] == config.read_text()
    assert provenance["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert not Path(provenance["config_path"]).is_absolute()
    assert provenance["dependencies"]["networkx"] == networkx.__version__
    backend = provenance["matching_backend"]
    assert backend["blossom5_available"] == blossom5.available()
    assert backend["name"] == ("blossom5" if blossom5.available() else "networkx")
    root = Path(__file__).resolve().parents[2]
    assert backend["lockfile_sha256"] == hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
