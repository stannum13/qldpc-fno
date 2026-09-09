from __future__ import annotations

import subprocess
import sys


def test_tensor_network_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "experiments/26_run_tensor_network_reference.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--out" in result.stdout
