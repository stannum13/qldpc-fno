from __future__ import annotations

import importlib.util
from pathlib import Path

CLI_PATH = Path("experiments/23_run_prediction_to_decision.py")
CONFIG_PATH = Path("configs/prediction_to_decision.json")


def _cli_module():
    spec = importlib.util.spec_from_file_location("prediction_to_decision_cli", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_exact_action_table(tmp_path: Path) -> None:
    module = _cli_module()

    payload = module.run(["--config", str(CONFIG_PATH), "--out", str(tmp_path / "result")])

    assert payload["exact_enumeration"] is True
    assert (tmp_path / "result" / "action_table.json").is_file()
