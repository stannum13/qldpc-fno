from __future__ import annotations

import importlib.util
from pathlib import Path

CLI_PATH = Path("experiments/24_run_mass_sampling.py")


def _cli_module():
    spec = importlib.util.spec_from_file_location("mass_sampling_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_mass_sampling_artifact(tmp_path: Path) -> None:
    module = _cli_module()

    payload = module.run(
        [
            "--decision-config",
            "configs/prediction_to_decision.json",
            "--sampling-config",
            "configs/mass_sampling.json",
            "--out",
            str(tmp_path),
        ]
    )

    assert payload["schema_version"] == 1
    assert (tmp_path / "mass_sampling.json").is_file()
