"""Reduced CLI fixtures exercise orchestration only, never scientific evidence."""

from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

from qldpc_fno.decision import adaptive_intervention_pilot as pilot
from tests.decision.test_adaptive_intervention_pilot import (
    _PILOT_CONFIG,
    _artifact_fixture,
    _fake_contractions,
)

CLI = Path("experiments/35_run_adaptive_intervention_pilot.py")


def test_reduced_cli_replay_is_byte_identical_and_never_advances(tmp_path, monkeypatch):
    assert CLI.is_file(), "pilot CLI is missing"
    spec = importlib.util.spec_from_file_location("pilot_cli", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    artifact, _ = _artifact_fixture(tmp_path)
    outputs = [tmp_path / "first", tmp_path / "second"]
    for index, out in enumerate(outputs):
        _fake_contractions(monkeypatch)
        ticks = itertools.count(step=(index + 1) * 0.125)
        monkeypatch.setattr(pilot, "perf_counter", lambda ticks=ticks: next(ticks), raising=False)
        result = module.run(
            [
                "--config",
                str(_PILOT_CONFIG),
                "--shots",
                str(artifact),
                "--out",
                str(out),
                "--non-scientific-fixture",
            ]
        )
        assert result["status"] == "reduced_non_scientific"
        assert result["scientific_eligible"] is False
        assert result["summary"]["advancement"]["margin_gate_clause"] is False
        assert result["summary"]["advancement"]["heterogeneous_efficiency_clause"] is False
    for filename in ("intervention_pilot.json", "summary.json"):
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    assert (outputs[0] / "timing.json").is_file(), "timing sidecar is missing"
    assert (outputs[0] / "timing.json").read_bytes() != (outputs[1] / "timing.json").read_bytes()


def test_cli_invalid_artifact_exits_nonzero_without_published_output(tmp_path):
    assert CLI.is_file(), "pilot CLI is missing"
    artifact, payload = _artifact_fixture(tmp_path)
    payload["shots"][0]["syndrome"][0] ^= 1
    artifact.write_text(json.dumps(payload))
    out = tmp_path / "result"
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--config",
            str(_PILOT_CONFIG),
            "--shots",
            str(artifact),
            "--out",
            str(out),
            "--non-scientific-fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "replay" in completed.stderr
    assert not out.exists()
