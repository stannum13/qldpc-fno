import json
import subprocess
import sys
from pathlib import Path

from scipy import sparse


def test_build_and_validate_code_clis(tmp_path: Path) -> None:
    code_dir = tmp_path / "code"
    subprocess.run(
        [sys.executable, "experiments/01_build_lp_codes.py", "--out", str(code_dir)],
        check=True,
    )
    hx = sparse.load_npz(code_dir / "hx.npz")
    hz = sparse.load_npz(code_dir / "hz.npz")
    assert hx.shape == (945, 2610)
    assert hz.shape == (945, 2610)

    subprocess.run(
        [sys.executable, "experiments/02_validate_lp_codes.py", "--code", str(code_dir)],
        check=True,
    )
    checks = json.loads((code_dir / "checks.json").read_text())
    assert checks["valid"] is True
    assert checks["k"] == 744
