# Planar Shot-Accuracy Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-locked, fresh-shot planar-code screen that tests whether the two-view tolerance policy preserves unrestricted tensor-decoder logical outcomes and compares both descriptively with calibrated CMWPM.

**Architecture:** Separate immutable shot generation, recovery/scoring primitives, baseline calibration, and held-out evaluation. Calibration selects one CMWPM configuration without touching screen shots; the evaluator independently replays physical errors, symplectic syndromes, tensor decisions, recoveries, logical failures, paired statistics, and work totals. Canonical execution is frozen and pushed before either new seed domain is sampled.

**Tech Stack:** Python 3.14, NumPy, SciPy, qecsim 1.0b9, pytest, Ruff, canonical JSON artifacts.

## Global Constraints

- Use `PlanarCode(5, 5)` at physical error rates `0.10` and `0.15` under qecsim i.i.d. depolarizing code-capacity noise.
- Generate 512 calibration shots and 2,048 screen shots per rate from distinct SHA-256-derived seed domains.
- Generate syndromes only with `qecsim.paulitools.bsp(error, code.stabilizers.T)`.
- Count failure modulo the stabilizer group through residual logical commutation, never raw correction-string equality.
- Freeze one CMWPM configuration shared across both rates before opening screen data.
- The primary gate is zero policy/reference class mismatches in both screen strata, adjusted one-sided Wilson upper bound below `0.005`, and zero logical-failure discordances.
- Do not claim posterior calibration, FPGA latency, threshold behavior, qLDPC transfer, or state-of-the-art decoding.
- Commit and push the complete evaluator/config freeze before generating calibration or screen artifacts.

---

### Task 1: Add audited planar recovery and shot-scoring primitives

**Files:**
- Create: `src/qldpc_fno/decision/planar_shot_accuracy.py`
- Create: `tests/decision/test_planar_shot_accuracy.py`

**Interfaces:**
- Produces: `logical_class_recovery(code: PlanarCode, syndrome: np.ndarray, logical_class: int) -> np.ndarray`.
- Produces: `score_recovery(code: PlanarCode, error: np.ndarray, syndrome: np.ndarray, recovery: np.ndarray) -> dict[str, object]`.
- Produces: `two_view_tolerance_decision(syndrome: np.ndarray, error_rate: float) -> dict[str, object]`.
- Consumes: existing `planar_mps_coset_masses` and qecsim `PlanarMPSDecoder.sample_recovery`.

- [ ] **Step 1: Write failing class-to-recovery and modulo-stabilizer tests**

Test all four qecsim class positions and require each recovery to reproduce the
input syndrome. Multiply a valid recovery by every stabilizer and require the
logical result to remain unchanged. Include a deliberately logical-shifted
recovery that remains syndrome-valid but changes the failure outcome.

```python
for logical_class in range(4):
    recovery = logical_class_recovery(code, syndrome, logical_class)
    assert np.array_equal(pt.bsp(recovery, code.stabilizers.T), syndrome)

scored = score_recovery(code, error, syndrome, recovery)
assert scored["syndrome_valid"] is True
assert scored["logical_failure"] == bool(
    np.any(pt.bsp((error + recovery) % 2, code.logicals.T))
)
```

- [ ] **Step 2: Run the new tests and observe import/behavior failure**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py`

Expected: FAIL because `planar_shot_accuracy` does not exist.

- [ ] **Step 3: Implement the minimal audited primitives**

Use qecsim's exact recovery ordering:

```python
sample = PlanarMPSDecoder.sample_recovery(code, syndrome)
paulis = (
    sample,
    sample.copy().logical_x(),
    sample.copy().logical_x().logical_z(),
    sample.copy().logical_z(),
)
return np.asarray(paulis[logical_class].to_bsf(), dtype=np.uint8)
```

`score_recovery` validates all shapes, checks recovery syndrome equality, forms
the binary residual, and reports both residual syndrome and logical signature.
`two_view_tolerance_decision` runs columns/rows at tolerance `0.01`, accepts only
valid agreement, otherwise runs column `chi=8`, and sums every executed action's
`estimated_arithmetic_flops`.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py tests/decision/test_tensor_network.py`

Run: `uv run ruff check src/qldpc_fno/decision/planar_shot_accuracy.py tests/decision/test_planar_shot_accuracy.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc_fno/decision/planar_shot_accuracy.py tests/decision/test_planar_shot_accuracy.py
git commit -m "feat: add audited planar shot scoring"
```

---

### Task 2: Generate immutable correctly paired calibration and screen shots

**Files:**
- Create: `src/qldpc_fno/decision/planar_shot_data.py`
- Create: `tests/decision/test_planar_shot_data.py`
- Create: `experiments/31_generate_planar_shots.py`
- Create: `configs/planar_shot_calibration.json`
- Create: `configs/planar_shot_screen.json`

**Interfaces:**
- Produces: `generate_planar_shots(config_path: Path, output_dir: Path) -> dict[str, object]`.
- Produces one `planar_shots.json` with config, shot rows, and relative-path source hashes.
- Each shot row contains `shot_id`, `error_rate`, `shot_index`, `sampler_seed`, `error_bsf`, and `syndrome`.

- [ ] **Step 1: Write failing config, replay, and separation tests**

Require SHA-256 derivation of `campaign_seed`, exact code/rate/count validation,
refusal to overwrite output, distinct calibration/screen domains, and replay:

```python
error = model.generate(code, row["error_rate"], np.random.default_rng(row["sampler_seed"]))
assert error.tolist() == row["error_bsf"]
assert pt.bsp(error, code.stabilizers.T).tolist() == row["syndrome"]
```

Also assert provenance keys are repository-relative names rather than absolute
checkout paths.

- [ ] **Step 2: Run the new tests and observe failure**

Run: `uv run pytest -q tests/decision/test_planar_shot_data.py`

Expected: FAIL because the generator does not exist.

- [ ] **Step 3: Implement deterministic generation and CLI**

Derive each per-shot seed as the unsigned big-endian integer in the first eight
bytes of `SHA256("{domain}|d5|p{error_rate:.6f}|i{shot_index:06d}")`. Require
uniqueness within each artifact; held-out evaluation additionally requires the
calibration and screen seed sets to be disjoint. Generate one
physical error, compute the symplectic syndrome, and serialize both. Record
qecsim version, git commit, dirty status, config hash, and source hashes under
stable labels such as `src/qldpc_fno/decision/planar_shot_data.py`.

The canonical configs use:

```json
{"seed_domain":"qldpc-fno/planar-shot-accuracy/calibration/v1","campaign_seed":11897243249319388773,"shots_per_rate":512}
```

and

```json
{"seed_domain":"qldpc-fno/planar-shot-accuracy/screen/v1","campaign_seed":10662168810357808346,"shots_per_rate":2048}
```

with common schema fields `code_distance: 5`, `error_rates: [0.1, 0.15]`, and
`noise_model: "qecsim_iid_depolarizing_code_capacity"`.

- [ ] **Step 4: Verify reduced generation without opening canonical domains**

Tests must create temporary domain names and tiny counts. Do not execute either
tracked canonical config in this task.

Run: `uv run pytest -q tests/decision/test_planar_shot_data.py`

Run: `uv run ruff check src/qldpc_fno/decision/planar_shot_data.py experiments/31_generate_planar_shots.py tests/decision/test_planar_shot_data.py`

Expected: all pass and no `evidence/planar-shot-*` paths exist.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc_fno/decision/planar_shot_data.py tests/decision/test_planar_shot_data.py experiments/31_generate_planar_shots.py configs/planar_shot_calibration.json configs/planar_shot_screen.json
git commit -m "feat: add paired planar shot generation"
```

---

### Task 3: Add calibration-only CMWPM selection

**Files:**
- Modify: `src/qldpc_fno/decision/planar_shot_accuracy.py`
- Modify: `tests/decision/test_planar_shot_accuracy.py`
- Create: `experiments/32_calibrate_planar_cmwpm.py`
- Create: `configs/planar_cmwpm_grid.json`

**Interfaces:**
- Produces: `calibrate_cmwpm(grid_path: Path, data_path: Path, output_dir: Path) -> dict[str, object]`.
- Produces: `planar_cmwpm_selection.json` with every candidate's per-rate/pooled failures and one deterministic selected config.
- Consumes only calibration shots from Task 2.

- [ ] **Step 1: Write failing selection and information-boundary tests**

Use synthetic candidate summaries to prove ordering by pooled failures, then
worst-rate failures, then parameter tuple. Require one shared config across rates.
Reject a data artifact whose domain is the screen domain or whose physical errors
do not replay.

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py -k cmwpm`

Expected: FAIL because calibration functions do not exist.

- [ ] **Step 3: Implement the frozen grid and evaluator**

Expand exactly 48 candidates from factors `[1,2,3,4]`, iterations `[2,4,8]`,
box shapes `["t","r"]`, and distance algorithms `[2,4]`. For every calibration
shot call `PlanarCMWPMDecoder(...).decode`, score modulo stabilizers, and retain
exceptions/invalid recoveries as explicit failures. Evaluate ordinary
`PlanarMWPMDecoder` in the same artifact but never let it enter CMWPM selection.

- [ ] **Step 4: Verify calibration on reduced temporary data**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py -k 'cmwpm or calibration'`

Run: `uv run ruff check src/qldpc_fno/decision/planar_shot_accuracy.py experiments/32_calibrate_planar_cmwpm.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc_fno/decision/planar_shot_accuracy.py tests/decision/test_planar_shot_accuracy.py experiments/32_calibrate_planar_cmwpm.py configs/planar_cmwpm_grid.json
git commit -m "feat: calibrate planar correlated matching"
```

---

### Task 4: Add held-out tensor/matching screen and statistical gate

**Files:**
- Modify: `src/qldpc_fno/decision/planar_shot_accuracy.py`
- Modify: `tests/decision/test_planar_shot_accuracy.py`
- Create: `experiments/33_run_planar_shot_accuracy.py`
- Create: `configs/planar_shot_accuracy_policy.json`

**Interfaces:**
- Produces: `run_planar_shot_accuracy(policy_path: Path, screen_path: Path, selection_path: Path, output_dir: Path) -> dict[str, object]`.
- Produces: `planar_shot_accuracy.json` with per-shot outcomes, per-rate gates, paired comparisons, work summaries, and provenance.
- Consumes `paired_decoder_summary` from `src/qldpc_fno/metrics/paired.py`.

- [ ] **Step 1: Write failing evaluator integrity and gate tests**

Tests must cover unique-reference rejection, policy agreement/fallback, recovery
syndrome validation, class mismatch, exact/policy BLER discordance, Wilson gate
at 0 versus 1 failures in 2,048 shots, exact paired tables, data/selection hash
tampering, screen/calibration domain collision, stale source hashes, reduced-run
noncanonical labeling, and stable relative provenance paths.

```python
assert adjusted_wilson_upper(0, 2048, alpha=0.05 / 2) < 0.005
assert status_for_strata([0, 0], shots=2048) == "passed_exact_outcome_preservation"
assert status_for_strata([0, 1], shots=2048) == "falsified_exact_class_preservation"
```

- [ ] **Step 2: Run the focused tests and observe failure**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py -k 'screen or gate or integrity'`

Expected: FAIL because held-out evaluation is absent.

- [ ] **Step 3: Implement exact/policy/CMWPM/MWPM evaluation**

For every screen shot independently replay the physical error and syndrome. Run
unrestricted row and column tensor references, require probability difference
at most `1e-10` and maximum pairwise log-ratio difference at most `1e-8`, require
the same winning class, and require each top/runner-up probability margin to
exceed twice the maximum row/column probability discrepancy,
run the two-view policy, reconstruct both recoveries, and score all four arms.
Independently recompute stored classes, probabilities, work counters, recovery
signatures, and aggregate statistics before allowing canonical status.

The policy config freezes `familywise_alpha: 0.05`, `comparisons: 2`,
`maximum_class_mismatch_rate: 0.005`, `work_bootstrap_replicates: 10000`, and
`work_bootstrap_seed: 13158893872079179326`.

- [ ] **Step 4: Verify the complete reduced pipeline**

Generate temporary calibration and screen domains with 2 shots per rate, limit
the CMWPM grid to two candidates, and require end-to-end noncanonical output.

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py tests/decision/test_planar_shot_data.py`

Run: `uv run ruff check src/qldpc_fno/decision/planar_shot_accuracy.py experiments/33_run_planar_shot_accuracy.py tests/decision/test_planar_shot_accuracy.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc_fno/decision/planar_shot_accuracy.py tests/decision/test_planar_shot_accuracy.py experiments/33_run_planar_shot_accuracy.py configs/planar_shot_accuracy_policy.json
git commit -m "feat: freeze planar shot accuracy gate"
```

---

### Task 5: Document, independently review, verify, and push the freeze

**Files:**
- Modify: `README.md`
- Create: `docs/planar-shot-accuracy-contract.md`

**Interfaces:**
- Documents exact commands, sample roles, baseline scope, statistical endpoints,
  failure scoring, expected artifacts, and explicit nonclaims.

- [ ] **Step 1: Add the contract and README experiment status**

State clearly that no result exists yet. Link the design, config files, erratum,
and future artifact locations. Explain in simple technical English why equal raw
corrections are unnecessary and why the residual logical signature is decisive.

- [ ] **Step 2: Run targeted and repository-wide verification**

Run: `uv run pytest -q tests/decision/test_planar_shot_accuracy.py tests/decision/test_planar_shot_data.py tests/decision/test_tensor_network.py`

Run: `uv run ruff check .`

Run: `git diff --check`

Expected: all pass.

- [ ] **Step 3: Request independent pre-sampling review**

The reviewer must inspect claim extraction, code/description agreement,
information symmetry, baseline grid/selection, modulo-stabilizer scoring,
statistical power, provenance, and the canonical/reduced boundary. Resolve every
BLOCKER or MAJOR finding before continuing.

- [ ] **Step 4: Commit and push the complete freeze**

```bash
git add README.md docs/planar-shot-accuracy-contract.md
git commit -m "docs: freeze planar shot accuracy contract"
git push origin research/adaptive-inference-world-model
```

Record the pushed commit in a local run note. Confirm `git status --short` is
empty and remote tracking is synchronized. **Do not generate data before this
step succeeds.**

---

### Task 6: Open calibration, freeze the baseline, then open the screen

**Files:**
- Create by command: `evidence/planar-shot-calibration/planar_shots.json`
- Create by command: `evidence/planar-cmwpm-calibration/planar_cmwpm_selection.json`
- Create by command: `evidence/planar-shot-screen/planar_shots.json`
- Create by command: `evidence/planar-shot-accuracy/planar_shot_accuracy.json`
- Create: `docs/planar-shot-accuracy-results.md`
- Modify: `README.md`

**Interfaces:**
- Consumes only the pushed freeze from Task 5.
- Produces the empirical screen and a claim-bounded results page.

- [ ] **Step 1: Generate calibration shots**

```bash
uv run python experiments/31_generate_planar_shots.py \
  --config configs/planar_shot_calibration.json \
  --out evidence/planar-shot-calibration
git add evidence/planar-shot-calibration
git commit -m "data: freeze planar calibration shots"
git push origin research/adaptive-inference-world-model
```

Expected: 1,024 correctly paired calibration shots and zero replay failures.
The calibration data must be committed and pushed before selection starts;
confirm the working tree is clean at this first barrier.

- [ ] **Step 2: Select and freeze calibrated CMWPM**

```bash
uv run python experiments/32_calibrate_planar_cmwpm.py \
  --grid configs/planar_cmwpm_grid.json \
  --data evidence/planar-shot-calibration/planar_shots.json \
  --out evidence/planar-cmwpm-calibration
git add evidence/planar-cmwpm-calibration
git commit -m "calibration: freeze planar matching selection"
git push origin research/adaptive-inference-world-model
```

Review candidate counts and verify that the selected tuple follows the declared
ordering. The second barrier commits and pushes selection before screen
generation, with all selection inputs already committed at the first barrier.

- [ ] **Step 3: Generate untouched screen shots**

```bash
uv run python experiments/31_generate_planar_shots.py \
  --config configs/planar_shot_screen.json \
  --out evidence/planar-shot-screen
git add evidence/planar-shot-screen
git commit -m "data: freeze held-out planar screen shots"
git push origin research/adaptive-inference-world-model
```

Expected: 4,096 correctly paired screen shots from the distinct screen domain.
The third barrier commits and pushes these shots before evaluation starts.

- [ ] **Step 4: Run the held-out screen**

```bash
uv run python experiments/33_run_planar_shot_accuracy.py \
  --policy configs/planar_shot_accuracy_policy.json \
  --screen evidence/planar-shot-screen/planar_shots.json \
  --selection evidence/planar-cmwpm-calibration/planar_cmwpm_selection.json \
  --out evidence/planar-shot-accuracy
git add evidence/planar-shot-accuracy
git commit -m "results: freeze planar shot accuracy evaluation"
git push origin research/adaptive-inference-world-model
```

Expected: a canonical status determined only by the frozen gate, with complete
per-rate failures, paired discordances, and work totals.
The fourth barrier publishes the scored result; subsequent interpretation and
documentation commits do not replace any of the four producer barriers.

- [ ] **Step 5: Independently review the evidence before writing conclusions**

The reviewer must replay a sample of seeds and every aggregate, verify all
recovery syndrome/logical checks, confirm calibration/screen separation, inspect
failure counts and intervals, and classify every proposed empirical claim as
SUPPORTED, UNSUPPORTED, or UNVERIFIABLE.

- [ ] **Step 6: Write claim-bounded results and verify**

Report the primary gate first, followed by exact/policy BLER, calibrated CMWPM,
MWPM, paired discordances, failure-count limitations, fallbacks, work estimates,
and all nonclaims. Do not call a descriptive matching difference superior unless
the frozen paired evidence and failure counts support that wording.

Run: `uv run pytest -q`

Run: `uv run ruff check . && git diff --check`

Expected: complete suite and static checks pass.

- [ ] **Step 7: Commit and push results**

```bash
git add evidence/planar-shot-calibration evidence/planar-cmwpm-calibration evidence/planar-shot-screen evidence/planar-shot-accuracy README.md docs/planar-shot-accuracy-results.md
git commit -m "results: evaluate planar shot accuracy"
git push origin research/adaptive-inference-world-model
```

Confirm the worktree is clean and local/remote branch tips match.
