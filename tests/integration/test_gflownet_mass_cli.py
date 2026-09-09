from __future__ import annotations

import importlib.util
import json
from pathlib import Path

CLI_PATH = Path("experiments/25_run_gflownet_mass.py")


def _cli_module():
    spec = importlib.util.spec_from_file_location("gflownet_mass_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_runs_reduced_config(tmp_path: Path) -> None:
    config = json.loads(Path("configs/gflownet_mass.json").read_text())
    config.update(
        {
            "training_seeds": config["training_seeds"][:1],
            "sampling_replicate_seeds": config["sampling_replicate_seeds"][:1],
            "hidden_width": 8,
            "training_steps": 2,
            "training_syndrome_indices": [0],
            "heldout_syndrome_indices": [1],
            "sample_budgets": [4],
            "training_fields": config["training_fields"][:1],
        }
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    module = _cli_module()

    payload = module.run(
        [
            "--config",
            str(config_path),
            "--decision-config",
            "configs/prediction_to_decision.json",
            "--baseline",
            "evidence/mass-sampling-baselines/mass_sampling.json",
            "--out",
            str(tmp_path / "out"),
        ]
    )

    assert payload["run_label"] == "reduced_non_scientific"
