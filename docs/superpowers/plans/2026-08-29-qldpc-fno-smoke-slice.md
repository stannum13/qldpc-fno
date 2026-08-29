# qLDPC-FNO Smoke Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one reproducible local experiment that reconstructs the paper's `lp(3,7)_16` code, samples its Z-error code-capacity model with Stim, obtains BP-LSD correction targets, overfits a tiny 1D FNO, and scores held-out logical predictions.

**Architecture:** The implementation separates immutable scientific artifacts from reusable code. Sparse GF(2) code construction feeds an exact Stim detector error model; packed samples feed a pinned BP-LSD teacher; deterministic ring-coordinate tensors feed a small spectral model. Each numbered experiment is a thin CLI over tested library functions and writes a manifest plus machine-readable outputs.

**Tech Stack:** Python 3.14, `uv`, NumPy 2.4.1, SciPy 1.17.1, Stim 1.16.0, `ldpc` 2.4.1, PyTorch 2.11.0, pytest 9.1.1, Ruff 0.16.5.

## Global Constraints

- Decode only the Z-error sector in this smoke slice: syndrome `s = Hx e_Z mod 2`, logical flips from logical-X supports.
- Use the paper's `lp(3,7)_16` seed at ring order 45 and label its distance only as `d <= 16`.
- Use exactly 21 syndrome channels and 58 correction channels over the ring coordinate.
- Persist Stim outputs as byte-aligned little-endian `b8` files with sidecar manifests.
- Pin Stim to 1.16.0 and `ldpc` to 2.4.1; save the generated `uv.lock`.
- Use one fixed Stim sampling call per shard and store the resulting raw files; seeded regeneration alone is not exact cross-platform replay.
- Train on BP-LSD corrections, not sampled physical-error representatives.
- Keep generated artifacts under ignored `artifacts/`; never commit sampled data or model checkpoints.
- Run locally. Do not create Google Cloud resources in this plan.
- Commit after every task whose tests pass.

---

## Planned File Structure

```text
pyproject.toml                         dependency and tool configuration
uv.lock                               exact environment lock
.gitignore                            generated environment and artifact exclusions
configs/smoke_lp_3_7_16.json          single source of smoke-run parameters
src/qldpc_fno/artifacts.py            canonical JSON, hashing, and run manifests
src/qldpc_fno/codes/seeds.py          checked-in paper seed specifications
src/qldpc_fno/codes/ring.py           monomial circulants and ring Kronecker products
src/qldpc_fno/codes/lifted_product.py LP check construction and code metadata
src/qldpc_fno/codes/gf2.py            rank, quotient basis, and binary checks
src/qldpc_fno/stim/dem.py             exact code-capacity DEM construction
src/qldpc_fno/stim/b8.py              packed Stim file read/write helpers
src/qldpc_fno/stim/sample.py          deterministic DEM shard sampling
src/qldpc_fno/decoders/bplsd.py       pinned BP-LSD batch adapter
src/qldpc_fno/data/ring_fields.py     bit vectors to ring-field tensors
src/qldpc_fno/models/fno1d.py         spectral layer and small ring FNO
src/qldpc_fno/training/overfit.py     deterministic tiny-set training
src/qldpc_fno/metrics/decoding.py     validity and logical-error scoring
experiments/00_lock_sources.py        write verified source registry
experiments/01_build_lp_codes.py      build `lp(3,7)_16` artifacts
experiments/02_validate_lp_codes.py   run scientific invariants
experiments/05_build_code_capacity_dem.py
experiments/06_sample_code_capacity.py
experiments/07_decode_bplsd.py
experiments/08_tensorize_ring_fields.py
experiments/10_overfit_tiny_models.py
experiments/12_evaluate_in_size.py
scripts/run_smoke.sh                  execute the vertical slice in order
tests/                                unit and integration tests mirroring `src/`
```

### Task 1: Reproducible project and artifact contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/qldpc_fno/__init__.py`
- Create: `src/qldpc_fno/artifacts.py`
- Create: `tests/test_artifacts.py`
- Create: `experiments/00_lock_sources.py`

**Interfaces:**
- Produces: `sha256_file(path: Path) -> str`
- Produces: `write_canonical_json(path: Path, value: Mapping[str, object]) -> None`
- Produces: `build_manifest(*, command: list[str], inputs: list[Path], outputs: list[Path], parameters: Mapping[str, object]) -> dict[str, object]`

- [ ] **Step 1: Write the artifact tests**

```python
from pathlib import Path
import json

from qldpc_fno.artifacts import sha256_file, write_canonical_json


def test_canonical_json_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_canonical_json(path, {"z": 1, "a": [2, 3]})
    assert path.read_text() == '{\n  "a": [\n    2,\n    3\n  ],\n  "z": 1\n}\n'
    assert sha256_file(path) == "276f0d2b3c61bc2c2d97bf2dc188c9294882cff2557f40a1b37b94f0873922d2"
```

- [ ] **Step 2: Run the test and confirm the package is absent**

Run: `uv run pytest tests/test_artifacts.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'qldpc_fno'`.

- [ ] **Step 3: Add the pinned project configuration**

```toml
[project]
name = "qldpc-fno"
version = "0.1.0"
requires-python = ">=3.14,<3.15"
dependencies = [
  "ldpc==2.4.1",
  "numpy==2.4.1",
  "scipy==1.17.1",
  "stim==1.16.0",
  "torch==2.11.0",
]

[dependency-groups]
dev = ["pytest==9.1.1", "ruff==0.16.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py314"
```

Generate the lock with `uv lock`.

- [ ] **Step 4: Implement canonical artifact helpers**

```python
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_manifest(*, command, inputs, outputs, parameters):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "command": command,
        "git_commit": commit,
        "inputs": {str(p): sha256_file(p) for p in inputs},
        "outputs": {str(p): sha256_file(p) for p in outputs},
        "parameters": dict(parameters),
        "platform": platform.platform(),
        "python": sys.version,
    }
```

Add `.venv/`, `artifacts/`, `__pycache__/`, `.pytest_cache/`, and `.ruff_cache/` to `.gitignore`. Add a source-lock CLI that writes the verified arXiv, Stim tag, `ldpc` version, and Willow DOI to a caller-supplied output path.

- [ ] **Step 5: Verify formatting, tests, and source locking**

Run: `uv run ruff check . && uv run pytest tests/test_artifacts.py -q && uv run python experiments/00_lock_sources.py --out artifacts/smoke/source_lock.json`

Expected: Ruff passes, one test passes, and `source_lock.json` lists arXiv `2603.28627v1`, Stim `1.16.0`, `ldpc` `2.4.1`, and DOI `10.5281/zenodo.13273331`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src/qldpc_fno experiments/00_lock_sources.py tests/test_artifacts.py
git commit -m "build: initialize reproducible experiment project"
```

### Task 2: Ring algebra and paper seed registry

**Files:**
- Create: `src/qldpc_fno/codes/__init__.py`
- Create: `src/qldpc_fno/codes/seeds.py`
- Create: `src/qldpc_fno/codes/ring.py`
- Create: `tests/codes/test_ring.py`
- Create: `tests/codes/test_seeds.py`

**Interfaces:**
- Produces: `LPSeed(name: str, ell: int, exponents: tuple[tuple[int, ...], ...], reported_n: int, reported_k: int, distance_upper_bound: int)`
- Produces: `PAPER_LP_3_7_16: LPSeed`
- Produces: `circulant_shift(ell: int, exponent: int) -> scipy.sparse.csr_matrix`
- Produces: `ring_identity(size: int) -> numpy.ndarray`
- Produces: `ring_kron(left: numpy.ndarray, right: numpy.ndarray, ell: int) -> numpy.ndarray`
- Produces: `expand_ring_matrix(blocks: numpy.ndarray, ell: int) -> scipy.sparse.csr_matrix`

- [ ] **Step 1: Write exact ring tests**

```python
import numpy as np

from qldpc_fno.codes.ring import circulant_shift, expand_ring_matrix, ring_identity, ring_kron


def test_circulant_shift_maps_column_forward() -> None:
    matrix = circulant_shift(5, 2).toarray()
    assert np.array_equal(matrix @ np.eye(5, dtype=np.uint8)[:, 1], [0, 0, 0, 1, 0])


def test_ring_kron_adds_monomial_exponents() -> None:
    left = np.array([[1, -1], [-1, 0]])
    right = ring_identity(2)
    actual = ring_kron(left, right, ell=5)
    expected = np.array([[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 0, -1], [-1, -1, -1, 0]])
    assert np.array_equal(actual, expected)
    assert expand_ring_matrix(np.array([[2]]), 5).shape == (5, 5)
```

- [ ] **Step 2: Run tests to verify missing implementations**

Run: `uv run pytest tests/codes/test_ring.py tests/codes/test_seeds.py -q`

Expected: collection fails because `qldpc_fno.codes.ring` and `seeds` do not exist.

- [ ] **Step 3: Implement ring operations with `-1` as the zero polynomial**

```python
def ring_kron(left: np.ndarray, right: np.ndarray, ell: int) -> np.ndarray:
    out = np.full(
        (left.shape[0] * right.shape[0], left.shape[1] * right.shape[1]),
        -1,
        dtype=np.int64,
    )
    for i, j in zip(*np.nonzero(left >= 0), strict=True):
        for u, v in zip(*np.nonzero(right >= 0), strict=True):
            out[i * right.shape[0] + u, j * right.shape[1] + v] = (
                int(left[i, j]) + int(right[u, v])
            ) % ell
    return out


def circulant_shift(ell: int, exponent: int) -> sparse.csr_matrix:
    columns = np.arange(ell)
    rows = (columns + exponent) % ell
    return sparse.csr_matrix((np.ones(ell, dtype=np.uint8), (rows, columns)), shape=(ell, ell))
```

Implement `expand_ring_matrix` with `scipy.sparse.bmat`, inserting `None` for `-1` entries and `circulant_shift` otherwise.

- [ ] **Step 4: Check in the exact `lp(3,7)_16` seed**

```python
PAPER_LP_3_7_16 = LPSeed(
    name="lp_3_7_16",
    ell=45,
    exponents=(
        (29, 21, 31, 15, 37, 25, 27),
        (13, 25, 19, 26, 11, 18, 29),
        (31, 2, 27, 32, 41, 41, 18),
    ),
    reported_n=2610,
    reported_k=744,
    distance_upper_bound=16,
)
```

Test every exponent, ring order, and reported parameter against literal expected values.

- [ ] **Step 5: Run tests and commit**

Run: `uv run ruff check . && uv run pytest tests/codes/test_ring.py tests/codes/test_seeds.py -q`

Expected: all ring and seed tests pass.

```bash
git add src/qldpc_fno/codes tests/codes
git commit -m "feat: add lifted-product ring primitives"
```

### Task 3: Lifted-product checks and scientific validation

**Files:**
- Create: `src/qldpc_fno/codes/lifted_product.py`
- Create: `src/qldpc_fno/codes/gf2.py`
- Create: `tests/codes/test_lifted_product.py`
- Create: `experiments/01_build_lp_codes.py`
- Create: `experiments/02_validate_lp_codes.py`

**Interfaces:**
- Consumes: `LPSeed`, ring operations from Task 2
- Produces: `CSSCode(name: str, ell: int, hx: csr_matrix, hz: csr_matrix, n: int, k: int)`
- Produces: `build_self_lifted_product(seed: LPSeed) -> CSSCode`
- Produces: `gf2_rank(matrix: spmatrix) -> int`
- Produces: `validate_css(code: CSSCode) -> dict[str, object]`

- [ ] **Step 1: Write a tiny algebra fixture and paper-shape test**

```python
import numpy as np

from qldpc_fno.codes.lifted_product import build_self_lifted_product
from qldpc_fno.codes.seeds import LPSeed, PAPER_LP_3_7_16


def test_self_lifted_product_commutes_on_tiny_seed() -> None:
    seed = LPSeed("tiny", 5, ((0, 1),), 25, 9, 2)
    code = build_self_lifted_product(seed)
    assert code.hx.shape == (10, 25)
    assert code.hz.shape == (10, 25)
    product = code.hx @ code.hz.T
    assert np.all(product.data % 2 == 0)


def test_paper_code_shapes_and_dimension() -> None:
    code = build_self_lifted_product(PAPER_LP_3_7_16)
    assert code.hx.shape == (945, 2610)
    assert code.hz.shape == (945, 2610)
    assert code.n == 2610
    assert code.k == 744
    assert set(np.diff(code.hx.indptr)) == {10}
```

- [ ] **Step 2: Run the tests and observe the missing constructor**

Run: `uv run pytest tests/codes/test_lifted_product.py -q`

Expected: collection fails because `lifted_product.py` is absent.

- [ ] **Step 3: Implement the ring-level LP formula**

```python
def build_self_lifted_product(seed: LPSeed) -> CSSCode:
    a = np.asarray(seed.exponents, dtype=np.int64)
    r, n_a = a.shape
    a_dagger = (-a.T) % seed.ell
    hx_ring = np.hstack(
        [ring_kron(a, ring_identity(n_a), seed.ell),
         ring_kron(ring_identity(r), a_dagger, seed.ell)]
    )
    hz_ring = np.hstack(
        [ring_kron(ring_identity(n_a), a, seed.ell),
         ring_kron(a_dagger, ring_identity(r), seed.ell)]
    )
    hx = expand_ring_matrix(hx_ring, seed.ell).astype(np.uint8)
    hz = expand_ring_matrix(hz_ring, seed.ell).astype(np.uint8)
    n = hx.shape[1]
    k = n - gf2_rank(hx) - gf2_rank(hz)
    return CSSCode(seed.name, seed.ell, hx, hz, n, k)
```

`gf2_rank` delegates to `ldpc.mod2.rank(matrix, method="sparse")`. `validate_css` computes dimensions, ranks, row weights, the sparse commutator modulo 2, and one-ring-shift equivariance. It must reduce sparse-product data modulo 2 before deciding whether a commutator is zero; integer sparse multiplication alone is not a GF(2) check.

- [ ] **Step 4: Implement artifact CLIs**

`01_build_lp_codes.py` saves `hx.npz`, `hz.npz`, and `code.json`. `02_validate_lp_codes.py` reloads them, runs `validate_css`, writes `checks.json`, and exits nonzero unless `n == 2610`, `k == 744`, both check matrices have shape `945x2610`, every check has weight 10, and CSS commutation is exact.

- [ ] **Step 5: Verify the real code and commit**

Run:

```bash
uv run pytest tests/codes/test_lifted_product.py -q
uv run python experiments/01_build_lp_codes.py --out artifacts/smoke/code
uv run python experiments/02_validate_lp_codes.py --code artifacts/smoke/code
```

Expected: tests pass and `artifacts/smoke/code/checks.json` contains `"valid": true` and `"k": 744`.

```bash
git add src/qldpc_fno/codes experiments/01_build_lp_codes.py experiments/02_validate_lp_codes.py tests/codes/test_lifted_product.py
git commit -m "feat: reconstruct and validate paper lp code"
```

### Task 4: Logical-X quotient basis and exact Stim DEM

**Files:**
- Modify: `src/qldpc_fno/codes/gf2.py`
- Create: `src/qldpc_fno/stim/__init__.py`
- Create: `src/qldpc_fno/stim/dem.py`
- Create: `tests/codes/test_gf2.py`
- Create: `tests/stim/test_dem.py`
- Create: `experiments/05_build_code_capacity_dem.py`

**Interfaces:**
- Consumes: `hx`, `hz`, physical error probability
- Produces: `quotient_basis(subspace: spmatrix, superspace_kernel: spmatrix) -> csr_matrix`
- Produces: `logical_x_basis(hx: spmatrix, hz: spmatrix) -> csr_matrix`
- Produces: `build_z_error_dem(hx: spmatrix, logical_x: spmatrix, error_rate: float) -> stim.DetectorErrorModel`

- [ ] **Step 1: Write quotient and DEM tests on a repetition-style CSS fixture**

```python
import numpy as np
from scipy import sparse

from qldpc_fno.codes.gf2 import quotient_basis
from qldpc_fno.stim.dem import build_z_error_dem


def test_quotient_basis_adds_only_independent_rows() -> None:
    sub = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    sup = sparse.csr_matrix([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
    logical = quotient_basis(sub, sup)
    assert logical.shape == (1, 3)
    assert np.array_equal(logical.toarray(), [[0, 0, 1]])


def test_dem_matches_direct_binary_products() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    dem = build_z_error_dem(hx, logical_x, error_rate=0.1)
    dets, obs, errors = dem.compile_sampler(seed=7).sample(64, return_errors=True)
    assert np.array_equal(dets, (errors @ hx.T.toarray()) % 2)
    assert np.array_equal(obs, (errors @ logical_x.T.toarray()) % 2)
```

- [ ] **Step 2: Run tests and verify the basis functions are absent**

Run: `uv run pytest tests/codes/test_gf2.py tests/stim/test_dem.py -q`

Expected: collection fails on missing imports.

- [ ] **Step 3: Implement bitset quotient elimination**

Convert each sparse row to a Python integer using little-endian packed bytes. Seed a pivot dictionary with an independent basis of `subspace`; reduce each row of `superspace_kernel`; retain the original row exactly when its reduction introduces a new pivot. Return retained rows as CSR.

```python
def _add_to_bit_basis(value: int, pivots: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        if pivot not in pivots:
            pivots[pivot] = value
            return True
        value ^= pivots[pivot]
    return False
```

`logical_x_basis` computes `ldpc.mod2.nullspace(hz, method="sparse")` and takes its quotient by the row basis of `hx`. Require exactly `k = n - rank(hx) - rank(hz)` returned rows and verify `logical_x @ hz.T == 0 mod 2`.

- [ ] **Step 4: Implement one-error-mechanism-per-qubit DEM construction**

For column `q`, append one `error(p)` instruction targeting every detector in `hx[:, q]` and every logical observable in `logical_x[:, q]`. Use Stim target objects, not string interpolation:

```python
targets = [stim.target_relative_detector_id(int(i)) for i in hx[:, q].nonzero()[0]]
targets += [stim.target_logical_observable_id(int(i)) for i in logical_x[:, q].nonzero()[0]]
dem.append("error", error_rate, targets)
```

- [ ] **Step 5: Build and verify the real DEM**

Run: `uv run python experiments/05_build_code_capacity_dem.py --code artifacts/smoke/code --error-rate 0.005 --out artifacts/smoke/dem`

Expected: `model.dem` has 2610 error mechanisms, 945 detectors, and 744 observables; `dem.json` records error rate `0.005` and all input/output hashes.

- [ ] **Step 6: Run tests and commit**

Run: `uv run ruff check . && uv run pytest tests/codes/test_gf2.py tests/stim/test_dem.py -q`

Expected: all tests pass.

```bash
git add src/qldpc_fno/codes/gf2.py src/qldpc_fno/stim experiments/05_build_code_capacity_dem.py tests/codes/test_gf2.py tests/stim/test_dem.py
git commit -m "feat: build exact code-capacity detector model"
```

### Task 5: Packed sampling and replay

**Files:**
- Create: `src/qldpc_fno/stim/b8.py`
- Create: `src/qldpc_fno/stim/sample.py`
- Create: `tests/stim/test_b8.py`
- Create: `tests/stim/test_sample.py`
- Create: `experiments/06_sample_code_capacity.py`

**Interfaces:**
- Produces: `write_b8(path: Path, values: np.ndarray) -> None`
- Produces: `read_b8(path: Path, *, shots: int, bits_per_shot: int) -> np.ndarray`
- Produces: `sample_dem_shard(dem: stim.DetectorErrorModel, *, shots: int, seed: int, output_dir: Path) -> dict[str, object]`

- [ ] **Step 1: Write byte-roundtrip and replay tests**

```python
def test_b8_round_trip_non_byte_aligned(tmp_path: Path) -> None:
    values = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 1, 1]], dtype=np.uint8)
    path = tmp_path / "values.b8"
    write_b8(path, values)
    assert np.array_equal(read_b8(path, shots=2, bits_per_shot=5), values)


def test_sample_manifest_and_replay(tmp_path: Path) -> None:
    dem = stim.DetectorErrorModel("error(0.25) D0 L0")
    manifest = sample_dem_shard(dem, shots=32, seed=123, output_dir=tmp_path)
    assert manifest["shots"] == 32
    assert manifest["seed"] == 123
    assert read_b8(tmp_path / "dets.b8", shots=32, bits_per_shot=1).shape == (32, 1)
```

- [ ] **Step 2: Run tests and confirm missing helpers**

Run: `uv run pytest tests/stim/test_b8.py tests/stim/test_sample.py -q`

Expected: collection fails on missing modules.

- [ ] **Step 3: Implement packed file helpers and one-call sampling**

`write_b8` packs unpacked arrays with `np.packbits(bitorder="little", axis=1)` and writes C-order bytes. `read_b8` checks the exact expected byte count, reshapes to `(shots, ceil(bits_per_shot / 8))`, unpacks little-endian, and slices padding bits.

`sample_dem_shard` calls:

```python
dets, obs, errors = dem.compile_sampler(seed=seed).sample(
    shots, bit_packed=False, return_errors=True
)
```

It writes `dets.b8`, `obs_actual.b8`, `errors.b8`, and `samples.json` with dimensions, format, seed, Stim version, and SHA-256 values.

- [ ] **Step 4: Generate the smoke shard**

Run: `uv run python experiments/06_sample_code_capacity.py --dem artifacts/smoke/dem/model.dem --shots 512 --seed 20260829 --out artifacts/smoke/samples`

Expected: three packed files exist; the manifest reports 512 shots, 945 detectors, 744 observables, and 2610 error mechanisms.

- [ ] **Step 5: Test and commit**

Run: `uv run ruff check . && uv run pytest tests/stim/test_b8.py tests/stim/test_sample.py -q`

```bash
git add src/qldpc_fno/stim experiments/06_sample_code_capacity.py tests/stim
git commit -m "feat: add replayable stim sample shards"
```

### Task 6: BP-LSD teacher and logical scoring

**Files:**
- Create: `src/qldpc_fno/decoders/__init__.py`
- Create: `src/qldpc_fno/decoders/bplsd.py`
- Create: `src/qldpc_fno/metrics/__init__.py`
- Create: `src/qldpc_fno/metrics/decoding.py`
- Create: `tests/decoders/test_bplsd.py`
- Create: `tests/metrics/test_decoding.py`
- Create: `experiments/07_decode_bplsd.py`

**Interfaces:**
- Produces: `decode_bplsd_batch(hx, syndromes, logical_x, *, error_rate: float) -> DecodeBatchResult`
- Produces: `score_observable_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, object]`

- [ ] **Step 1: Write decoder and score tests**

```python
def test_bp_lsd_corrections_match_syndrome() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    result = decode_bplsd_batch(hx, syndromes, logical_x, error_rate=0.05)
    assert np.array_equal((result.corrections @ hx.T.toarray()) % 2, syndromes)
    assert result.predicted_observables.shape == (2, 1)


def test_block_error_counts_any_wrong_observable() -> None:
    actual = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.uint8)
    predicted = np.array([[0, 0], [1, 1], [0, 1]], dtype=np.uint8)
    metrics = score_observable_predictions(actual, predicted)
    assert metrics["shots"] == 3
    assert metrics["block_errors"] == 2
```

- [ ] **Step 2: Run tests and verify missing decoder adapter**

Run: `uv run pytest tests/decoders/test_bplsd.py tests/metrics/test_decoding.py -q`

Expected: collection fails on missing modules.

- [ ] **Step 3: Implement the pinned teacher**

Construct one `ldpc.BpLsdDecoder` with:

```python
decoder = BpLsdDecoder(
    hx,
    error_rate=error_rate,
    max_iter=100,
    bp_method="minimum_sum",
    ms_scaling_factor=0.0,
    schedule="serial",
    lsd_method="LSD_E",
    lsd_order=5,
)
```

Decode each syndrome, verify `(hx @ correction) % 2 == syndrome`, and compute predicted observable flips as `(correction @ logical_x.T) % 2`. Record per-shot latency and convergence.

- [ ] **Step 4: Implement logical metrics with a Wilson interval**

Return block-error count, block-error rate, 95% Wilson lower/upper bounds, exact-match rate across observables, and shot count. A shot is a block error if any actual/predicted observable differs.

- [ ] **Step 5: Decode the smoke shard**

Run: `uv run python experiments/07_decode_bplsd.py --code artifacts/smoke/code --samples artifacts/smoke/samples --error-rate 0.005 --out artifacts/smoke/bplsd`

Expected: `corrections.b8`, `obs_predicted.b8`, `decode.json`, and `metrics.json` exist; every correction is syndrome-valid.

- [ ] **Step 6: Test and commit**

Run: `uv run ruff check . && uv run pytest tests/decoders tests/metrics -q`

```bash
git add src/qldpc_fno/decoders src/qldpc_fno/metrics experiments/07_decode_bplsd.py tests/decoders tests/metrics
git commit -m "feat: add bp-lsd teacher and logical metrics"
```

### Task 7: Exact ring-field tensorization

**Files:**
- Create: `src/qldpc_fno/data/__init__.py`
- Create: `src/qldpc_fno/data/ring_fields.py`
- Create: `tests/data/test_ring_fields.py`
- Create: `experiments/08_tensorize_ring_fields.py`

**Interfaces:**
- Produces: `to_ring_field(bits: np.ndarray, *, channels: int, ell: int) -> np.ndarray`
- Produces: `from_ring_field(field: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Write shape, roundtrip, and shift tests**

```python
def test_ring_field_roundtrip_and_shift() -> None:
    bits = np.arange(2 * 3 * 5, dtype=np.uint8).reshape(2, 15) % 2
    field = to_ring_field(bits, channels=3, ell=5)
    assert field.shape == (2, 3, 5)
    assert np.array_equal(from_ring_field(field), bits)
    shifted_bits = np.roll(field, 1, axis=-1).reshape(2, 15)
    assert np.array_equal(from_ring_field(np.roll(field, 1, axis=-1)), shifted_bits)
```

- [ ] **Step 2: Run test to confirm missing tensorizer**

Run: `uv run pytest tests/data/test_ring_fields.py -q`

Expected: collection fails on missing module.

- [ ] **Step 3: Implement strict reshape-only mappings**

Reject non-binary input and require `bits.shape[1] == channels * ell`. Return C-contiguous `float32` fields for PyTorch and `uint8` flattened bits for inverse mapping.

- [ ] **Step 4: Tensorize the smoke source**

Run: `uv run python experiments/08_tensorize_ring_fields.py --samples artifacts/smoke/samples --corrections artifacts/smoke/bplsd/corrections.b8 --ell 45 --out artifacts/smoke/tensors`

Expected: `syndromes.npy` has shape `(512, 21, 45)`, `corrections.npy` has shape `(512, 58, 45)`, and `split.json` assigns the first 384 shots to train and the final 128 to test without shuffling.

- [ ] **Step 5: Test and commit**

Run: `uv run ruff check . && uv run pytest tests/data/test_ring_fields.py -q`

```bash
git add src/qldpc_fno/data experiments/08_tensorize_ring_fields.py tests/data
git commit -m "feat: map qldpc samples to ring fields"
```

### Task 8: Tiny 1D FNO and deterministic overfit trainer

**Files:**
- Create: `src/qldpc_fno/models/__init__.py`
- Create: `src/qldpc_fno/models/fno1d.py`
- Create: `src/qldpc_fno/training/__init__.py`
- Create: `src/qldpc_fno/training/overfit.py`
- Create: `tests/models/test_fno1d.py`
- Create: `tests/training/test_overfit.py`
- Create: `experiments/10_overfit_tiny_models.py`

**Interfaces:**
- Produces: `SpectralConv1d(in_channels: int, out_channels: int, modes: int)`
- Produces: `RingFNO(in_channels: int = 21, out_channels: int = 58, width: int = 32, modes: int = 12, depth: int = 2)`
- Produces: `overfit_fno(inputs: np.ndarray, targets: np.ndarray, *, steps: int, seed: int) -> tuple[RingFNO, dict[str, object]]`

- [ ] **Step 1: Write shape and gradient tests**

```python
def test_ring_fno_preserves_ring_length_and_outputs_corrections() -> None:
    model = RingFNO(in_channels=21, out_channels=58, width=16, modes=8, depth=2)
    inputs = torch.randn(4, 21, 45)
    outputs = model(inputs)
    assert outputs.shape == (4, 58, 45)
    outputs.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_modes_cannot_exceed_rfft_width() -> None:
    model = RingFNO(in_channels=2, out_channels=3, width=4, modes=20, depth=1)
    with pytest.raises(ValueError, match="modes"):
        model(torch.zeros(1, 2, 9))
```

- [ ] **Step 2: Run tests and confirm model modules are missing**

Run: `uv run pytest tests/models/test_fno1d.py tests/training/test_overfit.py -q`

Expected: collection fails on missing modules.

- [ ] **Step 3: Implement the spectral convolution**

```python
def forward(self, values: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.rfft(values, dim=-1)
    if self.modes > spectrum.shape[-1]:
        raise ValueError(f"modes={self.modes} exceeds rFFT width={spectrum.shape[-1]}")
    out = torch.zeros(
        values.shape[0], self.out_channels, spectrum.shape[-1],
        dtype=spectrum.dtype, device=values.device,
    )
    out[..., : self.modes] = torch.einsum(
        "bim,iom->bom", spectrum[..., : self.modes], self.weight
    )
    return torch.fft.irfft(out, n=values.shape[-1], dim=-1)
```

Each FNO block sums one spectral convolution and one learned 1x1 circular-field projection, then applies GELU. Lift 21 channels to width with `Conv1d(..., kernel_size=1)` and project width to 58 logits.

- [ ] **Step 4: Implement deterministic weighted-BCE training**

Set NumPy and PyTorch seeds, enable deterministic algorithms, train on the first 128 training shots for 600 full-batch Adam steps at learning rate `3e-3`, and compute per-output `pos_weight = negatives / max(positives, 1)`, clipped to `[1, 100]`. Save initial/final loss and exact teacher-bit accuracy.

- [ ] **Step 5: Run the overfit experiment**

Run: `uv run python experiments/10_overfit_tiny_models.py --tensors artifacts/smoke/tensors --shots 128 --steps 600 --seed 17 --out artifacts/smoke/fno`

Expected: `model.pt`, `train_metrics.json`, and `model.json` exist; final weighted BCE is lower than initial weighted BCE, teacher-bit accuracy exceeds 0.99, and at least 90% of the 128 training corrections reproduce their syndromes exactly. If either accuracy gate fails, stop before Task 9 and diagnose training, thresholding, or tensor mappings.

- [ ] **Step 6: Test and commit**

Run: `uv run ruff check . && uv run pytest tests/models tests/training -q`

```bash
git add src/qldpc_fno/models src/qldpc_fno/training experiments/10_overfit_tiny_models.py tests/models tests/training
git commit -m "feat: add tiny ring fno overfit experiment"
```

### Task 9: Held-out decoder evaluation

**Files:**
- Create: `experiments/12_evaluate_in_size.py`
- Create: `tests/integration/test_smoke_evaluation.py`
- Modify: `src/qldpc_fno/metrics/decoding.py`

**Interfaces:**
- Consumes: frozen FNO, test syndrome fields, `hx`, logical-X basis, actual observables
- Produces: `evaluate_correction_logits(logits, *, hx, syndromes, logical_x, actual_observables) -> dict[str, object]`

- [ ] **Step 1: Write an evaluation test separating validity from logical accuracy**

```python
def test_invalid_correction_is_counted_even_if_logical_bits_match() -> None:
    hx = sparse.csr_matrix([[1, 1, 0]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    logits = np.array([[-10.0, -10.0, -10.0]])
    metrics = evaluate_correction_logits(
        logits,
        hx=hx,
        syndromes=np.array([[1]], dtype=np.uint8),
        logical_x=logical_x,
        actual_observables=np.array([[0]], dtype=np.uint8),
    )
    assert metrics["syndrome_valid"] == 0
    assert metrics["block_errors"] == 1
```

- [ ] **Step 2: Run the test and verify the evaluator is absent**

Run: `uv run pytest tests/integration/test_smoke_evaluation.py -q`

Expected: import fails for `evaluate_correction_logits`.

- [ ] **Step 3: Implement correction scoring**

Threshold logits at zero, flatten `(58, 45)` to 2610 correction bits, check every syndrome, and compute predicted logical flips. Count any syndrome-invalid shot as a block error even when its logical bits happen to match. Report syndrome-validity rate, logical block-error rate, Wilson interval, exact observable-match rate, and model inference latency.

- [ ] **Step 4: Evaluate the held-out 128 shots**

Run: `uv run python experiments/12_evaluate_in_size.py --code artifacts/smoke/code --samples artifacts/smoke/samples --tensors artifacts/smoke/tensors --model artifacts/smoke/fno/model.pt --out artifacts/smoke/evaluation`

Expected: `metrics.json` clearly separates teacher-bit accuracy, syndrome validity, and logical block error. This smoke run is not required to beat BP-LSD; it is required to execute correctly and disclose invalid corrections.

- [ ] **Step 5: Test and commit**

Run: `uv run ruff check . && uv run pytest tests/integration/test_smoke_evaluation.py tests/metrics -q`

```bash
git add experiments/12_evaluate_in_size.py src/qldpc_fno/metrics/decoding.py tests/integration/test_smoke_evaluation.py
git commit -m "feat: evaluate learned qldpc corrections"
```

### Task 10: One-command smoke loop and verification

**Files:**
- Create: `configs/smoke_lp_3_7_16.json`
- Create: `scripts/run_smoke.sh`
- Create: `tests/integration/test_smoke_cli.py`
- Create: `README.md`

**Interfaces:**
- Consumes: all CLIs from Tasks 1-9
- Produces: a complete `artifacts/smoke/` tree and final `artifacts/smoke/evaluation/metrics.json`

- [ ] **Step 1: Write a subprocess integration test using reduced shots and steps**

The test invokes `scripts/run_smoke.sh` with `SMOKE_SHOTS=16` and `SMOKE_STEPS=2`, then asserts that source lock, code checks, DEM, sample manifest, teacher metrics, tensor split, model metrics, and evaluation metrics exist and contain their declared dimensions.

- [ ] **Step 2: Run it before the script exists**

Run: `uv run pytest tests/integration/test_smoke_cli.py -q`

Expected: failure because `scripts/run_smoke.sh` is absent.

- [ ] **Step 3: Add the canonical configuration**

```json
{
  "code": "lp_3_7_16",
  "ell": 45,
  "error_rate": 0.005,
  "sample_seed": 20260829,
  "shots": 512,
  "train_shots": 384,
  "overfit_shots": 128,
  "training_seed": 17,
  "training_steps": 600,
  "fno_width": 32,
  "fno_modes": 12,
  "fno_depth": 2
}
```

- [ ] **Step 4: Implement the fail-fast shell orchestrator**

Use `#!/usr/bin/env bash` and `set -euo pipefail`. Read `SMOKE_SHOTS` and `SMOKE_STEPS` only as optional test overrides; invoke the numbered experiment scripts in numeric order. Refuse to overwrite an existing output directory unless `SMOKE_OUTPUT` points to a new path.

- [ ] **Step 5: Document exact commands and scientific boundaries**

README sections must cover environment creation, `uv run pytest`, `bash scripts/run_smoke.sh`, artifact layout, the Z-error-only scope, the fact that this is code-capacity rather than the paper's circuit-level reproduction, and the rule that `16` is a distance upper bound.

- [ ] **Step 6: Run complete verification**

Run:

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -q
SMOKE_OUTPUT=artifacts/final-smoke bash scripts/run_smoke.sh
git status --short
```

Expected: Ruff and all tests pass; the full 512-shot/600-step smoke run completes; `artifacts/final-smoke/evaluation/metrics.json` exists; Git status lists no generated artifacts because `artifacts/` is ignored.

- [ ] **Step 7: Commit**

```bash
git add configs scripts tests/integration/test_smoke_cli.py README.md
git commit -m "feat: add one-command qldpc fno smoke loop"
```

## Plan Completion Check

Before starting the family-transfer plan, verify all of the following from fresh artifacts:

- the reconstructed paper code has `n=2610`, `k=744`, and commuting checks;
- the exact DEM has 2610 error mechanisms, 945 detectors, and 744 observables;
- every BP-LSD teacher correction is syndrome-valid;
- tensor mappings round-trip exactly with shapes `(shots, 21, 45)` and `(shots, 58, 45)`;
- the tiny FNO passes its overfit gate;
- held-out evaluation counts invalid corrections as failures;
- the complete loop is reproducible from the committed source plus stored raw `b8` artifacts.

The next separate plan implements experiments 03-04 and 09-16: controlled-family construction, spectral audit, matched controls, naive paper-family transfer, and operator-conditioned transfer.
