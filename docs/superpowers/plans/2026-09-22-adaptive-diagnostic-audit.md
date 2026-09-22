# Adaptive Diagnostic Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a compact, reproducible development audit of the five proposed adaptive-computation signals using the immutable 4,096-shot planar physical-shot artifact.

**Architecture:** A pure analysis module converts one frozen shot row into a compact record with inference-visible features, paid post-refinement diagnostics, and offline labels kept in separate namespaces. A CLI verifies the source artifact's published identity, derives all rows and aggregate audits, and writes a new development artifact without altering the original result. A results document explains feature availability, common-mode failures, matched benign cases, and the limits of an eight-event exploratory analysis.

**Tech Stack:** Python 3.14, NumPy, SciPy, pytest, the repository artifact helpers, canonical JSON.

## Global Constraints

- Treat `evidence/planar-shot-accuracy/planar_shot_accuracy.json` and its published hashes as immutable.
- Existing revealed shots are development-open and cannot confirm a retuned rule.
- Features available before a decision, paid post-refinement diagnostics, exact-reference labels, and physical-outcome labels must be structurally separated.
- An exact logical-class mismatch is not automatically an additional physical-shot failure.
- Invalid masses or actions remain explicit and are never repaired by clipping or absolute values.
- The audit may describe discrimination on these 4,096 shots but may not claim calibrated safety, out-of-sample prediction, speed, or hardware value.

---

### Task 1: Probability and spectral diagnostic primitives

**Files:**
- Create: `src/qldpc_fno/decision/adaptive_diagnostics.py`
- Create: `tests/decision/test_adaptive_diagnostics.py`

**Interfaces:**
- Consumes: four-class probability vectors and serialized tensor-action dictionaries.
- Produces: `probability_margin`, `total_variation`, `jensen_shannon_divergence`, `maximum_log_ratio_discrepancy`, and `summarize_spectra`.

- [ ] **Step 1: Write failing metric tests**

Add tests using `p=[0.7,0.2,0.08,0.02]` and `q=[0.6,0.3,0.08,0.02]`. Require margin `0.5`, total variation `0.1`, symmetric finite Jensen–Shannon divergence, zero self-divergence, zero self log-ratio discrepancy, and rejection of negative, zero-sum, non-four-class, or non-finite vectors.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py -q`

Expected: collection fails because `qldpc_fno.decision.adaptive_diagnostics` does not exist.

- [ ] **Step 3: Implement validated probability primitives**

Create `_probabilities(value)` that requires shape `(4,)`, finite nonnegative entries, positive finite sum, normalizes the vector, and returns float64. Implement:

```python
def probability_margin(probabilities: Sequence[float]) -> float: ...
def total_variation(left: Sequence[float], right: Sequence[float]) -> float: ...
def jensen_shannon_divergence(left: Sequence[float], right: Sequence[float]) -> float: ...
def maximum_log_ratio_discrepancy(left: Sequence[float], right: Sequence[float]) -> float | None: ...
```

The log-ratio function returns `None` when either normalized vector has a zero entry; it does not clip.

- [ ] **Step 4: Write failing spectral-summary tests**

Use a real-shaped fixture with two truncation events and three spectral summaries. Require counts, maximum discarded squared-weight fraction, mean spectral entropy, maximum retained rank, and the action's already-charged estimated arithmetic work. Require `available=False` for a missing or invalid action.

- [ ] **Step 5: Implement spectral aggregation**

Implement:

```python
def summarize_spectra(action: Mapping[str, object] | None) -> dict[str, object]: ...
```

Traverse `action["work"]["truncation_events"][*]["spectral_summaries"]`, validate numeric values, and return only aggregate statistics. Do not expose a spectrum as pre-action information.

- [ ] **Step 6: Run focused tests and lint**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py -q`

Run: `uv run ruff check src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py`

Expected: all pass.

- [ ] **Step 7: Commit the primitives**

```bash
git add src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py
git commit -m "feat: add adaptive diagnostic primitives"
```

### Task 2: Compact per-shot records and information ledger

**Files:**
- Modify: `src/qldpc_fno/decision/adaptive_diagnostics.py`
- Modify: `tests/decision/test_adaptive_diagnostics.py`

**Interfaces:**
- Consumes: one serialized shot from the frozen planar evaluator.
- Produces: `extract_diagnostic_row(shot)` and `diagnostic_availability_ledger()`.

- [ ] **Step 1: Write a failing per-shot extraction test**

Build a minimal shot fixture containing exact row/column, tolerance row/column, fixed-chi8, arms, work, and reference certificate. Require the result to contain:

```python
{
    "identity": {"shot_id", "shot_index", "error_rate"},
    "inference_features": {
        "tolerance_class_agreement",
        "tolerance_total_variation",
        "tolerance_jensen_shannon",
        "tolerance_log_ratio_discrepancy",
        "tolerance_column_margin",
        "tolerance_row_margin",
        "tolerance_minimum_margin",
        "tolerance_column_spectra",
        "tolerance_row_spectra",
        "cheap_estimated_arithmetic_flops",
    },
    "post_refinement_diagnostics": {
        "column_to_fixed_chi8_total_variation",
        "fixed_chi8_to_reference_total_variation",
        "fixed_chi8_class_matches_reference",
    },
    "reference_labels": {
        "exact_class",
        "policy_class",
        "class_mismatch",
        "exact_margin",
        "column_to_reference_total_variation",
        "row_to_reference_total_variation",
    },
    "physical_outcome_labels": {
        "outcome_discordance",
        "exact_failure",
        "policy_failure",
        "change_helped_realized_outcome",
        "change_harmed_realized_outcome",
    },
}
```

Assert that raw errors, syndromes, recoveries, exact probabilities, and physical outcomes never occur inside `inference_features`.

- [ ] **Step 2: Run the focused test and verify it fails because the interface is absent**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py -q`

- [ ] **Step 3: Implement compact extraction**

Use tolerance views for cheap features, exact-column probabilities for offline reference labels, and fixed-chi8 only under paid post-refinement diagnostics. Record invalid/missing views explicitly; do not impute their distributions. Derive helped/harmed only from paired exact/policy realized failures.

- [ ] **Step 4: Write a failing availability-ledger test**

Require five entries named `cross_view_disagreement`, `decision_margin`, `cross_scale_stability`, `discarded_representation`, and `calibrated_ood_score`. Require stages `cheap_action_complete`, `refinement_complete`, and `unavailable_without_fitted_calibrator`; require charged feature fields for each available signal.

- [ ] **Step 5: Implement the availability ledger**

Return static, machine-readable entries stating earliest availability, field dependencies, and whether the signal is deployable from this artifact. Mark the OOD score unavailable and cross-scale stability as paid post-refinement information.

- [ ] **Step 6: Run focused tests and lint**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py -q`

Run: `uv run ruff check src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py`

- [ ] **Step 7: Commit compact extraction**

```bash
git add src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py
git commit -m "feat: extract planar adaptive diagnostics"
```

### Task 3: Exploratory audit and source consistency checks

**Files:**
- Modify: `src/qldpc_fno/decision/adaptive_diagnostics.py`
- Modify: `tests/decision/test_adaptive_diagnostics.py`

**Interfaces:**
- Consumes: compact diagnostic rows and the source artifact's aggregate sections.
- Produces: `rank_auc`, `select_margin_matched_controls`, and `build_diagnostic_audit`.

- [ ] **Step 1: Write failing exact-rank AUC tests**

Require AUC `1.0` for perfectly separated scores, `0.0` for reversed scores, and `0.5` for all ties. Reject a label vector without both classes.

- [ ] **Step 2: Implement tie-aware rank AUC**

Implement `rank_auc(scores, labels)` using average ranks and the Mann–Whitney identity. Keep the score direction explicit in the returned audit.

- [ ] **Step 3: Write failing matched-control tests**

Create two rates with mismatch and benign rows. Require one unique benign accepted control per mismatch, same error rate, minimal absolute cheap-margin difference, and deterministic shot-ID tie breaking. Exclude fallback rows and prior controls.

- [ ] **Step 4: Implement deterministic margin matching**

Implement:

```python
def select_margin_matched_controls(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]: ...
```

Return mismatch/control IDs, margins, and paired differences for cheap view-disagreement and discarded-representation signals. Describe these as matched development examples, not a causal estimate.

- [ ] **Step 5: Write failing whole-audit tests**

Use a small synthetic artifact. Require the builder to reject disagreement between derived shot counts, per-rate mismatch/outcome counts, and source work totals. Require an output containing source identity, rows, availability ledger, source consistency checks, common-mode failures, matched controls, exploratory AUCs for all declared scores, and explicit nonclaims.

- [ ] **Step 6: Implement the audit builder**

Implement:

```python
def build_diagnostic_audit(source: Mapping[str, object], *, source_sha256: str) -> dict[str, object]: ...
```

Compute all derived counts from compact rows. Compare them to the frozen `per_rate` and `work.totals` data. Limit the primary feature audit to accepted cheap agreements when analysing common-mode failure. Provide pooled and rate-stratified AUCs when both labels occur; return `None` with a reason otherwise. Mark all discrimination results development-only because the eight outcomes were already opened.

- [ ] **Step 7: Run focused tests and lint**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py -q`

Run: `uv run ruff check src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py`

- [ ] **Step 8: Commit the exploratory audit**

```bash
git add src/qldpc_fno/decision/adaptive_diagnostics.py tests/decision/test_adaptive_diagnostics.py
git commit -m "feat: audit adaptive diagnostic signals"
```

### Task 4: CLI, derived artifact, and scientific exposition

**Files:**
- Create: `experiments/34_audit_planar_diagnostics.py`
- Create: `tests/integration/test_adaptive_diagnostics_cli.py`
- Create: `evidence/adaptive-computation/diagnostic-audit-v1/diagnostic_audit.json`
- Create: `docs/adaptive-computation-diagnostic-audit.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the immutable planar accuracy JSON and its manifest.
- Produces: a canonical compact audit and a reader-facing development report.

- [ ] **Step 1: Write a failing CLI integration test**

Create a minimal valid source fixture and manifest with matching SHA-256. Invoke the experiment script with `--artifact`, `--manifest`, and `--out`. Require the output file, exact source hash, source-consistency pass, and refusal to overwrite an existing output.

- [ ] **Step 2: Run the CLI test and verify it fails because the script is absent**

Run: `uv run pytest tests/integration/test_adaptive_diagnostics_cli.py -q`

- [ ] **Step 3: Implement the CLI**

The script parses paths, verifies the raw artifact against `artifact_binding.raw.sha256` and size in the publication manifest, loads the JSON, calls `build_diagnostic_audit`, and writes `diagnostic_audit.json` with `write_canonical_json`. It refuses an existing output directory or file.

- [ ] **Step 4: Run focused tests and lint**

Run: `uv run pytest tests/decision/test_adaptive_diagnostics.py tests/integration/test_adaptive_diagnostics_cli.py -q`

Run: `uv run ruff check src/qldpc_fno/decision/adaptive_diagnostics.py experiments/34_audit_planar_diagnostics.py tests/decision/test_adaptive_diagnostics.py tests/integration/test_adaptive_diagnostics_cli.py`

- [ ] **Step 5: Generate the development artifact once**

Run:

```bash
uv run python experiments/34_audit_planar_diagnostics.py \
  --artifact evidence/planar-shot-accuracy/planar_shot_accuracy.json \
  --manifest evidence/planar-shot-accuracy/manifest.json \
  --out evidence/adaptive-computation/diagnostic-audit-v1
```

Expected: one canonical JSON derived from 4,096 shots; source identity and all consistency checks pass.

- [ ] **Step 6: Write the results document from the artifact**

Document the observed diagnostic distributions, the eight common-mode cases and matched benign controls, AUCs with their score directions, feature timing/cost, and strongest counterexample. State that the data are development-open, only eight mismatches exist, thresholds remain unfrozen, and no safety/performance claim follows.

- [ ] **Step 7: Link the audit from README**

Add one status paragraph pointing readers to the compact artifact and report. Preserve the existing falsification statement and claim boundary.

- [ ] **Step 8: Verify derived evidence and repository tests**

Run:

```bash
uv run pytest tests/decision/test_adaptive_diagnostics.py tests/integration/test_adaptive_diagnostics_cli.py -q
uv run pytest -q
uv run ruff check .
git diff --check
```

Expected: all tests and lint pass. Independently load the output, recompute its source hash, and check that all eight source mismatches appear exactly once.

- [ ] **Step 9: Commit and push the completed first audit**

```bash
git add README.md docs/adaptive-computation-diagnostic-audit.md \
  experiments/34_audit_planar_diagnostics.py \
  evidence/adaptive-computation/diagnostic-audit-v1/diagnostic_audit.json \
  src/qldpc_fno/decision/adaptive_diagnostics.py \
  tests/decision/test_adaptive_diagnostics.py \
  tests/integration/test_adaptive_diagnostics_cli.py
git commit -m "results: publish adaptive diagnostic audit"
git push origin main
```

## Self-review

- Tasks 1–4 cover the first three items of the reader's executable tranche. The 64-shot action-grid pilot remains a separate follow-on plan because it performs new expensive solver runs and requires the audit's finalized state/action schema.
- No placeholder implementation or undefined interface is used.
- Function names are consistent across tasks.
- The plan keeps reference and outcome information outside inference features and charges post-refinement diagnostics at their actual stage.
