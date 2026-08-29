import json
import subprocess
import sys
from pathlib import Path

from qldpc_fno.artifacts import build_manifest, sha256_file, write_canonical_json


def test_canonical_json_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_canonical_json(path, {"z": 1, "a": [2, 3]})
    assert path.read_text() == '{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    assert sha256_file(path) == "276f0d2b3c61bc2c2d97bf2dc188c9294882cff2557f40a1b37b94f0873922d2"


def test_manifest_hashes_inputs_and_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.bin"
    output_path = tmp_path / "output.bin"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")

    manifest = build_manifest(
        command=["experiment", "--flag"],
        inputs=[input_path],
        outputs=[output_path],
        parameters={"shots": 8},
    )

    assert manifest["command"] == ["experiment", "--flag"]
    assert manifest["parameters"] == {"shots": 8}
    assert manifest["inputs"][str(input_path)] == sha256_file(input_path)
    assert manifest["outputs"][str(output_path)] == sha256_file(output_path)
    assert manifest["git_commit"]
    assert manifest["python"]


def test_source_lock_cli_writes_verified_versions(tmp_path: Path) -> None:
    output_path = tmp_path / "source_lock.json"
    subprocess.run(
        [sys.executable, "experiments/00_lock_sources.py", "--out", str(output_path)],
        check=True,
    )
    lock = json.loads(output_path.read_text())
    assert lock["paper"]["version"] == "2603.28627v1"
    assert lock["stim"]["version"] == "1.16.0"
    assert lock["ldpc"]["version"] == "2.4.1"
    assert lock["willow"]["doi"] == "10.5281/zenodo.13273331"
