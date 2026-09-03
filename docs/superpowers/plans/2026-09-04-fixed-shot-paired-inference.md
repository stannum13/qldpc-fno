# Fixed-Shot Paired-Inference Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace invalid paired-bootstrap/compatibility reporting with exact discordant-pair evidence and add a provenance-locked, single-rate, fixed-shot disconfirming campaign.

**Architecture:** Keep decoder and dataset generation unchanged. Introduce exact paired inference as a small metrics API, make rate selection and test stopping explicit in the strict campaign configuration, and thread those policies through pilot publication, evaluation, orchestration, and generated summaries. Preserve adaptive execution only as diagnostic output; only a complete fixed-shot result receives a directional comparison status.

**Tech Stack:** Python 3.14, NumPy, SciPy `binomtest`, Stim, PyTorch, `ldpc`, pytest, Bash, canonical JSON artifacts, Git/GCS provenance.

## Global Constraints

- Continue to study only the `lp(3,7)_16` independent-Z code-capacity model with perfect syndrome information.
- Keep uniform BP-LSD, FNO soft-prior BP-LSD, and FNO proposal plus residual BP-LSD as the three decoder arms.
- Do not compare raw correction strings; a failure remains an invalid syndrome or any logical-observable mismatch.
- Do not introduce BP-LSD hyperparameter tuning, noninferiority, equivalence, speed, threshold, circuit-level, Willow, or code-family claims.
- Keep configuration strict: new fields are required and unknown fields remain errors.
- Preserve train/calibration/test role-separated sampler seeds and immutable artifact hashes.
- A complete fixed test consumes exactly `max_test_shots_per_point`; partial/deadline and adaptive results receive `not_fixed_sample`.
- Use exact McNemar/binomial inference on discordant pairs and label the Clopper-Pearson interval as conditional on discordance.
- Follow red-green-refactor for every production behavior change.
- Do not rewrite the immutable reduced Cloud artifact produced by commit `0ad764b0d7372904a3222fb54f0ee4ec003d226d`.

---

### Task 1: Replace paired bootstrap statistics with exact discordant-pair inference

**Files:**
- Modify: `tests/metrics/test_paired.py`
- Modify: `src/qldpc_fno/metrics/paired.py`

**Interfaces:**
- Consumes: two equal-length boolean arrays containing paired baseline and hybrid block-failure outcomes.
- Produces: `paired_decoder_summary(baseline_failures, hybrid_failures, *, alpha=0.05) -> dict[str, object]`.
- Produces: `paired_comparison_status(summary, *, fixed_sample: bool, alpha=0.05) -> str`.
- Removes: `bootstrap_seed`, `samples`, percentile-bootstrap fields, and `accuracy_compatible`.

- [ ] **Step 1: Replace bootstrap-oriented tests with exact-inference failing tests**

Add tests that assert the complete public schema and boundary behavior:

```python
from qldpc_fno.metrics.paired import paired_comparison_status, paired_decoder_summary


def test_all_hybrid_only_discordances_have_nonzero_exact_uncertainty() -> None:
    summary = paired_decoder_summary(
        np.zeros(8, dtype=np.bool_),
        np.ones(8, dtype=np.bool_),
    )

    assert summary["hybrid_only_failure"] == 8
    assert summary["baseline_only_failure"] == 0
    assert summary["discordant_pairs"] == 8
    assert summary["mcnemar_exact_pvalue_two_sided"] == pytest.approx(0.0078125)
    assert summary["mcnemar_exact_pvalue_harm"] == pytest.approx(0.00390625)
    assert summary["mcnemar_exact_pvalue_benefit"] == 1.0
    assert 0.0 < summary["hybrid_harm_share_given_discordance_95ci_low"] < 1.0
    assert summary["hybrid_harm_share_given_discordance_95ci_high"] == 1.0
    assert paired_comparison_status(summary, fixed_sample=True) == "harm_detected"


def test_all_baseline_only_discordances_detect_benefit() -> None:
    summary = paired_decoder_summary(
        np.ones(8, dtype=np.bool_),
        np.zeros(8, dtype=np.bool_),
    )
    assert paired_comparison_status(summary, fixed_sample=True) == "benefit_detected"


def test_balanced_discordances_are_inconclusive() -> None:
    baseline = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.bool_)
    hybrid = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.bool_)
    summary = paired_decoder_summary(baseline, hybrid)
    assert paired_comparison_status(summary, fixed_sample=True) == "inconclusive"


def test_no_discordances_have_null_conditional_interval() -> None:
    outcomes = np.array([0, 1, 0, 1], dtype=np.bool_)
    summary = paired_decoder_summary(outcomes, outcomes)
    assert summary["discordant_pairs"] == 0
    assert summary["hybrid_harm_share_given_discordance"] is None
    assert summary["hybrid_harm_share_given_discordance_95ci_low"] is None
    assert summary["hybrid_harm_share_given_discordance_95ci_high"] is None
    assert paired_comparison_status(summary, fixed_sample=True) == "no_discordances"


def test_adaptive_or_partial_comparison_has_no_fixed_sample_status() -> None:
    summary = paired_decoder_summary(
        np.zeros(8, dtype=np.bool_),
        np.ones(8, dtype=np.bool_),
    )
    assert paired_comparison_status(summary, fixed_sample=False) == "not_fixed_sample"
```

Retain shape, dtype, empty-input, and alpha-validation tests. Delete assertions about deterministic bootstrap seeds and `accuracy_compatible`.

- [ ] **Step 2: Run the new metric tests and verify the expected failure**

Run:

```bash
uv run pytest tests/metrics/test_paired.py -q
```

Expected: collection or assertion failure because `paired_comparison_status` does not exist and `paired_decoder_summary` still requires bootstrap arguments.

- [ ] **Step 3: Implement exact paired statistics**

In `src/qldpc_fno/metrics/paired.py`, import `binomtest` from `scipy.stats`, retain `_failure_outcomes` and `_wilson_summary`, and implement this behavior:

```python
def paired_decoder_summary(
    baseline_failures: np.ndarray,
    hybrid_failures: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, object]:
    baseline = _failure_outcomes(baseline_failures, name="baseline")
    hybrid = _failure_outcomes(hybrid_failures, name="hybrid")
    if baseline.shape != hybrid.shape:
        raise ValueError("baseline and hybrid outcomes must have equal shape")
    if baseline.size == 0:
        raise ValueError("paired outcomes must contain at least one shot")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    both_succeed = int(np.count_nonzero(~baseline & ~hybrid))
    baseline_only = int(np.count_nonzero(baseline & ~hybrid))
    hybrid_only = int(np.count_nonzero(~baseline & hybrid))
    both_fail = int(np.count_nonzero(baseline & hybrid))
    shots = int(baseline.size)
    discordant = baseline_only + hybrid_only

    if discordant:
        two_sided = float(binomtest(hybrid_only, discordant, 0.5).pvalue)
        harm = float(binomtest(hybrid_only, discordant, 0.5, alternative="greater").pvalue)
        benefit = float(binomtest(hybrid_only, discordant, 0.5, alternative="less").pvalue)
        interval = binomtest(hybrid_only, discordant).proportion_ci(
            confidence_level=1.0 - alpha,
            method="exact",
        )
        harm_share: float | None = hybrid_only / discordant
        low: float | None = float(interval.low)
        high: float | None = float(interval.high)
    else:
        two_sided = harm = benefit = 1.0
        harm_share = low = high = None

    return {
        "baseline": _wilson_summary(baseline),
        "baseline_only_failure": baseline_only,
        "block_error_delta": (hybrid_only - baseline_only) / shots,
        "both_fail": both_fail,
        "both_succeed": both_succeed,
        "discordant_pairs": discordant,
        "hybrid": _wilson_summary(hybrid),
        "hybrid_harm_share_given_discordance": harm_share,
        "hybrid_harm_share_given_discordance_95ci_high": high,
        "hybrid_harm_share_given_discordance_95ci_low": low,
        "hybrid_only_failure": hybrid_only,
        "mcnemar_exact_pvalue_benefit": benefit,
        "mcnemar_exact_pvalue_harm": harm,
        "mcnemar_exact_pvalue_two_sided": two_sided,
        "shots": shots,
    }


def paired_comparison_status(
    paired_summary: Mapping[str, object],
    *,
    fixed_sample: bool,
    alpha: float = 0.05,
) -> str:
    if not isinstance(fixed_sample, (bool, np.bool_)):
        raise TypeError("fixed_sample must be boolean")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if not fixed_sample:
        return "not_fixed_sample"
    baseline_only = int(paired_summary["baseline_only_failure"])
    hybrid_only = int(paired_summary["hybrid_only_failure"])
    discordant = int(paired_summary["discordant_pairs"])
    pvalue = float(paired_summary["mcnemar_exact_pvalue_two_sided"])
    if discordant == 0:
        return "no_discordances"
    if pvalue <= alpha and hybrid_only > baseline_only:
        return "harm_detected"
    if pvalue <= alpha and baseline_only > hybrid_only:
        return "benefit_detected"
    return "inconclusive"
```

Validate that counts are non-negative, add up to `shots`, and that the p-value is finite and in `[0,1]` before assigning status.

- [ ] **Step 4: Run metric tests and verify green**

Run:

```bash
uv run pytest tests/metrics/test_paired.py -q
```

Expected: all paired-metric tests pass.

- [ ] **Step 5: Commit the exact-statistics change**

```bash
git add src/qldpc_fno/metrics/paired.py tests/metrics/test_paired.py
git commit -m "fix: use exact paired decoder inference"
```

---

### Task 2: Add strict selection and stopping modes plus the disconfirming config

**Files:**
- Modify: `tests/campaign/test_config.py`
- Modify: all inline campaign-config fixtures in `tests/integration/test_campaign_data_clis.py`, `tests/integration/test_campaign_training_clis.py`, `tests/integration/test_hybrid_evaluation_cli.py`, and `tests/integration/test_accuracy_campaign_cli.py`
- Modify: `src/qldpc_fno/campaign/config.py`
- Modify: `configs/accuracy_campaign.json`
- Modify: `configs/accuracy_campaign_cloud_reduced.json`
- Create: `configs/accuracy_disconfirm_p0375.json`

**Interfaces:**
- Produces: `CampaignConfig.selection_mode: str` with values `pilot|fixed`.
- Produces: `CampaignConfig.test_stopping_mode: str` with values `adaptive|fixed`.
- Produces: a strict committed configuration for the fixed `p=0.0375` experiment.

- [ ] **Step 1: Write failing strict-schema tests**

Update the base fixture payloads with:

```python
"selection_mode": "pilot",
"test_stopping_mode": "adaptive",
```

Then add:

```python
def test_config_accepts_fixed_selection_and_stopping(tmp_path: Path) -> None:
    payload = _canonical_payload()
    payload["noise_grid"] = [0.0375]
    payload["selection_mode"] = "fixed"
    payload["test_stopping_mode"] = "fixed"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    config = CampaignConfig.from_json(path)
    assert config.selection_mode == "fixed"
    assert config.test_stopping_mode == "fixed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_mode", "manual", "selection_mode"),
        ("test_stopping_mode", "deadline", "test_stopping_mode"),
    ],
)
def test_config_rejects_invalid_campaign_modes(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    payload = _canonical_payload()
    payload[field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        CampaignConfig.from_json(path)


def test_disconfirming_config_is_strict_and_fixed() -> None:
    config = CampaignConfig.from_json(Path("configs/accuracy_disconfirm_p0375.json"))
    assert config.noise_grid == (0.0375,)
    assert config.selection_mode == "fixed"
    assert config.test_stopping_mode == "fixed"
    assert config.max_test_shots_per_point == 2048
```

- [ ] **Step 2: Run config tests and verify red**

```bash
uv run pytest tests/campaign/test_config.py -q
```

Expected: failure because `CampaignConfig` does not define either mode and the disconfirming file is absent.

- [ ] **Step 3: Implement and validate the new fields**

Add required dataclass fields:

```python
selection_mode: str
test_stopping_mode: str
```

After constructing the dataclass in `_validate`, enforce:

```python
if self.selection_mode not in {"pilot", "fixed"}:
    raise ValueError("selection_mode must be 'pilot' or 'fixed'")
if self.test_stopping_mode not in {"adaptive", "fixed"}:
    raise ValueError("test_stopping_mode must be 'adaptive' or 'fixed'")
```

Do not place either string field in `_INTEGER_FIELDS`. Preserve the existing positive-integer and `target_failures <= max_test_shots_per_point` rules.

- [ ] **Step 4: Update committed and fixture configurations**

Set the canonical and reduced Cloud configs to:

```json
"selection_mode": "pilot",
"test_stopping_mode": "adaptive"
```

Create `configs/accuracy_disconfirm_p0375.json` with the exact payload in the approved design specification. Add the same two fields to every inline strict-schema fixture, choosing `pilot/adaptive` unless a test explicitly exercises fixed behavior.

- [ ] **Step 5: Run config and fixture-collection tests**

```bash
uv run pytest tests/campaign/test_config.py tests/integration/test_campaign_data_clis.py --collect-only -q
```

Expected: collection succeeds and all config tests pass.

- [ ] **Step 6: Commit the campaign-policy schema**

```bash
git add configs src/qldpc_fno/campaign/config.py tests/campaign/test_config.py tests/integration
git commit -m "feat: declare campaign selection and stopping modes"
```

---

### Task 3: Publish fixed rate selections without pilot sampling

**Files:**
- Modify: `tests/integration/test_campaign_data_clis.py`
- Modify: `experiments/13_pilot_noise_grid.py`
- Modify: `src/qldpc_fno/campaign/evaluation.py`
- Modify: `experiments/14_generate_campaign_shards.py`

**Interfaces:**
- Consumes: `CampaignConfig.selection_mode` and `noise_grid`.
- Produces: the existing `pilot/selection.json` and `pilot/manifest.json` artifact paths for both selection modes.
- Adds selection fields: `selection_mode` and `evidence_role`.
- Preserves downstream selection and hash verification.

- [ ] **Step 1: Write a failing fixed-selection CLI test**

Add an integration test that builds the code, invokes stage 13 with a one-rate fixed config, and asserts:

```python
selection = json.loads((pilot_dir / "selection.json").read_text())
manifest = json.loads((pilot_dir / "manifest.json").read_text())
assert selection["selection_mode"] == "fixed"
assert selection["selected_noise_points"] == [0.0375]
assert selection["pilot_rows"] == []
assert selection["evidence_role"] == "predeclared_selection_not_evidence"
assert manifest["shards"] == {}
assert not list(pilot_dir.glob("rate-*"))
```

Extend the existing pilot test to assert:

```python
assert selection["selection_mode"] == "pilot"
assert selection["evidence_role"] == "selection_only_not_held_out"
```

Add a tampering test that changes `selected_noise_points` after publication and confirms stage 14 rejects the selection hash.

- [ ] **Step 2: Run the new fixed-selection test and verify red**

```bash
uv run pytest tests/integration/test_campaign_data_clis.py -q -k "fixed_selection"
```

Expected: failure because stage 13 always samples and applies `select_noise_points`.

- [ ] **Step 3: Branch stage 13 on the declared selection mode**

After validating the code and opening `atomic_role_directory`, create the selection payload as follows:

```python
if config.selection_mode == "fixed":
    rows: list[dict[str, object]] = []
    selected = list(config.noise_grid)
    evidence_role = "predeclared_selection_not_evidence"
else:
    rows = run_pilot_grid(config.noise_grid, evaluate)
    selected = list(select_noise_points(rows))
    evidence_role = "selection_only_not_held_out"

write_canonical_json(
    selection_path,
    {
        "evidence_role": evidence_role,
        "pilot_rows": rows,
        "selected_noise_points": selected,
        "selection_mode": config.selection_mode,
        "source_sha256": {
            "code_manifest": code_manifest_sha256,
            "config": config_sha256,
        },
    },
)
```

Keep `shard_manifest_paths` empty in fixed mode. Do not construct a DEM or call BP-LSD in that branch.

- [ ] **Step 4: Enforce mode/rate consistency downstream**

In selection verification used by stages 14 and 17, require:

```python
if selection.get("selection_mode") != config.selection_mode:
    raise ValueError("selection mode does not match campaign configuration")
if config.selection_mode == "fixed" and tuple(raw_rates) != config.noise_grid:
    raise ValueError("fixed selection rates do not match configured noise_grid")
```

Require the exact evidence-role string appropriate to the mode. Preserve config/code hashes and pilot publication hash verification.

- [ ] **Step 5: Run data-stage integration tests**

```bash
uv run pytest tests/integration/test_campaign_data_clis.py tests/campaign/test_shards.py -q
```

Expected: fixed and pilot selection tests pass; existing pilot adaptation remains unchanged.

- [ ] **Step 6: Commit fixed-selection publication**

```bash
git add experiments/13_pilot_noise_grid.py experiments/14_generate_campaign_shards.py src/qldpc_fno/campaign/evaluation.py tests/integration/test_campaign_data_clis.py
git commit -m "feat: publish predeclared fixed noise selections"
```

---

### Task 4: Enforce fixed test stopping and emit exact paired statuses

**Files:**
- Modify: `tests/metrics/test_paired.py`
- Modify: `tests/integration/test_hybrid_evaluation_cli.py`
- Modify: `src/qldpc_fno/metrics/paired.py`
- Modify: `src/qldpc_fno/campaign/evaluation.py`

**Interfaces:**
- Produces: `test_stop_reason(failure_counts, *, shots, target_failures, shot_cap, mode) -> str | None`.
- Consumes: exact `paired_decoder_summary` and `paired_comparison_status` from Task 1.
- Produces: per-method `comparison_status` in each completed rate summary.
- Removes from new summaries: `accuracy_compatible` and paired bootstrap fields.

- [ ] **Step 1: Write failing fixed-stopping unit tests**

Rename `adaptive_stop_reason` tests to target the mode-aware API and add:

```python
def test_fixed_stop_ignores_failure_target_until_shot_cap() -> None:
    counts = {"baseline": 8, "soft_prior": 8, "residual": 8}
    assert test_stop_reason(
        counts,
        shots=8,
        target_failures=1,
        shot_cap=16,
        mode="fixed",
    ) is None
    assert test_stop_reason(
        counts,
        shots=16,
        target_failures=1,
        shot_cap=16,
        mode="fixed",
    ) == "shot_cap"


def test_adaptive_stop_preserves_target_failure_behavior() -> None:
    counts = {"baseline": 2, "soft_prior": 2, "residual": 2}
    assert test_stop_reason(
        counts,
        shots=4,
        target_failures=2,
        shot_cap=16,
        mode="adaptive",
    ) == "target_failures"
```

Add invalid-mode coverage.

- [ ] **Step 2: Write a failing evaluation integration test**

Use a fixture with `test_stopping_mode="fixed"`, `test_batch_shots=1`,
`max_test_shots_per_point=2`, and `target_failures=1`. Patch or fixture the decoder outcomes so every arm fails the first shot. Assert evaluation still writes two shots and:

```python
assert summary["shots"] == 2
assert summary["stop_reason"] == "shot_cap"
assert "accuracy_compatible" not in summary
assert summary["comparison_status"]["soft_prior"] in {
    "harm_detected", "benefit_detected", "inconclusive", "no_discordances"
}
assert "block_error_delta_95ci_low" not in summary["paired"]["soft_prior"]
```

Add a partial fixed evaluation assertion:

```python
assert partial_summary["comparison_status"]["soft_prior"] == "not_fixed_sample"
```

Add resume-verification tests that remove one required exact paired field from a
rate summary and that mark a fixed result complete below the configured shot cap.
Both mutations must be rejected before any new batch is decoded.

- [ ] **Step 3: Run the focused tests and verify red**

```bash
uv run pytest tests/metrics/test_paired.py tests/integration/test_hybrid_evaluation_cli.py -q -k "fixed_stop or fixed_stopping or partial_fixed"
```

Expected: failure because stopping is always adaptive and summaries still contain compatibility/bootstrap fields.

- [ ] **Step 4: Implement mode-aware stopping**

Replace `adaptive_stop_reason` with:

```python
def test_stop_reason(
    failure_counts: Mapping[str, object],
    *,
    shots: int,
    target_failures: int,
    shot_cap: int,
    mode: str,
) -> str | None:
    # Preserve all existing count/shot validation.
    if mode not in {"adaptive", "fixed"}:
        raise ValueError("test stopping mode must be 'adaptive' or 'fixed'")
    if mode == "adaptive" and all(
        int(failure_counts[name]) >= target_failures for name in _DECODER_NAMES
    ):
        return "target_failures"
    if shots >= shot_cap:
        return "shot_cap"
    return None
```

Use this helper at initial scan, after each batch, and during finalization with
`mode=config.test_stopping_mode`.

- [ ] **Step 5: Replace compatibility output with exact comparison status**

Change `_write_rate_summary` to accept `fixed_sample: bool`, call
`paired_decoder_summary` without bootstrap arguments, and emit:

```python
"comparison_status": {
    method: paired_comparison_status(paired[method], fixed_sample=fixed_sample)
    for method in _HYBRIDS
},
```

Compute `fixed_sample` only when all are true:

```python
config.test_stopping_mode == "fixed"
rate_status == "complete"
stop_reason == "shot_cap"
int(counts["shots"]) == config.max_test_shots_per_point
```

For empty/partial summaries, return `not_fixed_sample` for both hybrids. Add
`selection_mode`, `test_stopping_mode`, and
`target_failures_active = config.test_stopping_mode == "adaptive"` to the final
evaluation manifest.

- [ ] **Step 6: Run metric and evaluation tests**

```bash
uv run pytest tests/metrics/test_paired.py tests/integration/test_hybrid_evaluation_cli.py -q
```

Expected: all tests pass and no new evaluation fixture contains `accuracy_compatible` or percentile-bootstrap fields.

- [ ] **Step 7: Commit fixed evaluation semantics**

```bash
git add src/qldpc_fno/metrics/paired.py src/qldpc_fno/campaign/evaluation.py tests/metrics/test_paired.py tests/integration/test_hybrid_evaluation_cli.py
git commit -m "fix: require fixed shots for paired comparison status"
```

---

### Task 5: Remove obsolete bootstrap controls through the orchestration stack

**Files:**
- Modify: `tests/campaign/test_inputs.py`
- Modify: `tests/campaign/test_runner.py`
- Modify: `tests/integration/test_cloud_launcher.py`
- Modify: `tests/integration/test_accuracy_campaign_cli.py`
- Modify: `src/qldpc_fno/campaign/inputs.py`
- Modify: `src/qldpc_fno/campaign/evaluation.py`
- Modify: `src/qldpc_fno/campaign/runner.py`
- Modify: `experiments/17_evaluate_hybrid_decoders.py`
- Modify: `scripts/run_accuracy_campaign.sh`
- Modify: `scripts/launch_cloud_campaign.sh`

**Interfaces:**
- Removes: `CampaignInputRequest.bootstrap_samples`, `EvaluationRequest.bootstrap_samples`, CLI `--bootstrap-samples`, environment `CAMPAIGN_BOOTSTRAP_SAMPLES`, and run-mode `execution_controls.bootstrap_samples`.
- Preserves: `calibration_grid_limit` as the only reduced calibration-selection execution control.
- Produces: run-mode schema version `3` so old bootstrap-bearing inputs cannot resume silently.

- [ ] **Step 1: Update tests to require bootstrap-free requests and contracts**

Remove `bootstrap_samples` from request constructors and expected manifests. Add assertions:

```python
assert "bootstrap_samples" not in run_mode["execution_controls"]
assert run_mode["schema_version"] == 3
assert "CAMPAIGN_BOOTSTRAP_SAMPLES" not in launcher_environment
```

Update canonical-input validation tests so canonical mode only rejects a non-null calibration grid limit. Update runner/evaluation CLI tests to invoke no bootstrap option.

- [ ] **Step 2: Run orchestration tests and verify red**

```bash
uv run pytest tests/campaign/test_inputs.py tests/campaign/test_runner.py tests/integration/test_cloud_launcher.py tests/integration/test_accuracy_campaign_cli.py -q
```

Expected: failures from the existing required bootstrap constructor/CLI/environment fields.

- [ ] **Step 3: Remove bootstrap fields from Python requests and runner plumbing**

Delete `bootstrap_samples` from both request dataclasses, validation branches, expected input mode, runner constructor, runner CLI, stage-17 invocation, `_finalize`, and `_write_rate_summary`. Set run-mode `schema_version` to `3`.

The canonical validation becomes:

```python
if request.campaign_mode == "canonical":
    if request.calibration_grid_limit is not None:
        raise ValueError("canonical campaign controls cannot reduce calibration")
```

The stage-17 CLI no longer defines or forwards `--bootstrap-samples`.

- [ ] **Step 4: Remove shell and Cloud bootstrap controls**

Delete:

```text
CAMPAIGN_BOOTSTRAP_SAMPLES
bootstrap_samples
--bootstrap-samples
```

from the local runner, launcher environment, execution contract, dry-run output,
and reduced usage text. Do not change Cloud resources, timeout policy, canonical
fail-closed behavior, or calibration-grid controls.

- [ ] **Step 5: Run orchestration tests and verify green**

```bash
uv run pytest tests/campaign/test_inputs.py tests/campaign/test_runner.py tests/integration/test_cloud_launcher.py tests/integration/test_accuracy_campaign_cli.py -q
```

Expected: all tests pass with schema version 3 and no bootstrap control.

- [ ] **Step 6: Commit the orchestration cleanup**

```bash
git add src/qldpc_fno/campaign experiments/17_evaluate_hybrid_decoders.py scripts tests/campaign tests/integration
git commit -m "refactor: remove paired bootstrap campaign controls"
```

---

### Task 6: Generate exact-evidence summaries and expose the fixed disconfirming profile

**Files:**
- Modify: `tests/campaign/test_runner.py`
- Modify: `tests/integration/test_accuracy_campaign_cli.py`
- Modify: `src/qldpc_fno/campaign/runner.py`
- Modify: `scripts/run_accuracy_campaign.sh`

**Interfaces:**
- Consumes: new per-rate `comparison_status` and exact paired fields.
- Produces: Markdown tables with exact discordant-pair evidence and no compatibility/noninferiority wording.
- Produces: `bash scripts/run_accuracy_campaign.sh --disconfirm [--resume]` selecting only `configs/accuracy_disconfirm_p0375.json`.

- [ ] **Step 1: Write failing generated-summary tests**

Replace fixture `accuracy_compatible` data with:

```python
"comparison_status": {"soft_prior": "inconclusive", "residual": "harm_detected"},
"paired": {
    "soft_prior": {
        "baseline_only_failure": 1,
        "hybrid_only_failure": 2,
        "discordant_pairs": 3,
        "block_error_delta": 0.01,
        "mcnemar_exact_pvalue_two_sided": 1.0,
        "hybrid_harm_share_given_discordance": 2 / 3,
        "hybrid_harm_share_given_discordance_95ci_low": 0.09429932405024608,
        "hybrid_harm_share_given_discordance_95ci_high": 0.9915962413403874,
    }
}
```

Assert the generated Markdown contains `comparison status`, `discordant pairs`,
`exact McNemar`, and `conditional on discordance`, and does not contain
`accuracy-compatible`, `noninferior`, `equivalent`, or `paired 95% interval`.

- [ ] **Step 2: Write a failing disconfirm-profile shell test**

Extend the local CLI integration test to run with the stage execution guard:

```bash
CAMPAIGN_OUTPUT=<fresh temp path> \
CAMPAIGN_FAIL_ON_STAGE_EXECUTION=1 \
bash scripts/run_accuracy_campaign.sh --disconfirm
```

Assert the published inputs name `accuracy_disconfirm_p0375.json`, contain
`selection_mode="fixed"` and `test_stopping_mode="fixed"`, and reach the expected
guard without accepting an arbitrary config path. Add rejection tests for
`--disconfirm` combined with `CAMPAIGN_REDUCED=1` and for unknown arguments.

- [ ] **Step 3: Run summary/profile tests and verify red**

```bash
uv run pytest tests/campaign/test_runner.py tests/integration/test_accuracy_campaign_cli.py -q -k "markdown or disconfirm"
```

Expected: generated Markdown still uses compatibility language and the script rejects `--disconfirm`.

- [ ] **Step 4: Update generated Markdown**

Replace the overview column with `paired comparison status`. For each hybrid print:

```text
Delta; baseline-only failures; hybrid-only failures; discordant pairs;
exact two-sided McNemar p-value; hybrid-harm share conditional on discordance;
95% Clopper-Pearson interval conditional on discordance.
```

When status is `not_fixed_sample`, print a sentence stating that paired inference
is diagnostic because the result is adaptive, incomplete, or both. Do not select a
winner from timing data.

- [ ] **Step 5: Add the allow-listed disconfirm profile**

Extend argument parsing to accept only `--resume` and `--disconfirm`, in either
order without duplicates. When `--disconfirm` is present:

```bash
canonical_config="$repo_root/configs/accuracy_disconfirm_p0375.json"
output="${CAMPAIGN_OUTPUT:-artifacts/accuracy-disconfirm-p0375}"
```

Reject `CAMPAIGN_REDUCED=1`. Keep `CAMPAIGN_CONFIG` unsupported, run
`verify_canonical_checkout` against the selected committed configuration, pass the
same file as canonical and effective config, and retain resumability and all
overwrite/symlink safeguards. Permit `CAMPAIGN_FAIL_ON_STAGE_EXECUTION=1` only for
the existing reduced test profile or the allow-listed disconfirm profile, never for
the ordinary canonical campaign; this gives the integration test a bounded stop
after input verification without opening arbitrary configuration selection.

- [ ] **Step 6: Run summary/profile tests and verify green**

```bash
uv run pytest tests/campaign/test_runner.py tests/integration/test_accuracy_campaign_cli.py -q
```

Expected: all tests pass; profile input publication is provenance-locked to the
committed fixed configuration.

- [ ] **Step 7: Commit summary and entry-point behavior**

```bash
git add src/qldpc_fno/campaign/runner.py scripts/run_accuracy_campaign.sh tests/campaign/test_runner.py tests/integration/test_accuracy_campaign_cli.py
git commit -m "feat: add fixed-shot disconfirming campaign profile"
```

---

### Task 7: Align scientific documentation and verify the complete repair

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-methodology.md`
- Modify: `docs/reproducibility.md`
- Modify: `docs/calibration-throughput-benchmark.md` only if it currently presents the censored 645-second observation as a runtime estimate
- Test: complete repository test suite

**Interfaces:**
- Documents: exact paired statistics, conditional-interval meaning, fixed versus adaptive evidence, the asymmetric disconfirming run, and remaining prerequisites.
- Preserves: existing source locks, reduced-result warning, Cloud safety boundary, and accuracy-before-speed policy.

- [ ] **Step 1: Update the methodology contract**

Replace bootstrap/compatibility language with these explicit statements:

```text
The paired test conditions on discordant shots and uses an exact binomial
(McNemar) test under equal paired failure probability. Its Clopper-Pearson interval
describes the share of discordances in which only the hybrid fails; it is not a
confidence interval for the marginal block-error-rate difference.

An inconclusive paired test is not evidence of equivalence or noninferiority.
Adaptive and deadline-truncated evaluations report paired quantities only as
diagnostics. Directional comparison statuses are assigned only after a complete,
predeclared fixed-shot evaluation.
```

Document both config modes and the inactive role of `target_failures` in fixed
mode.

- [ ] **Step 2: Add the exact engineer command and interpretation boundary**

In the README and reproducibility guide add:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm
```

and the resume form:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm --resume
```

State that a detected harm result falsifies this candidate at `p=0.0375`, while an
inconclusive or benefit result only permits the next experiment. Name baseline
tuning, at least three training seeds, and a larger confirmatory test as remaining
requirements before a positive accuracy claim.

- [ ] **Step 3: Remove stale executable-language references**

Run:

```bash
rg -n "accuracy.compat|bootstrap.samples|paired bootstrap|paired 95% interval|roughly 92|92 hours" README.md docs src experiments scripts tests configs
```

Expected: matches remain only in the historical approved specification/plan where
the removed behavior is described, or in explicit migration text. Remove all stale
claims from live methodology, commands, and generated-output tests.

- [ ] **Step 4: Run targeted scientific-path tests**

```bash
uv run pytest -q \
  tests/metrics/test_paired.py \
  tests/campaign/test_config.py \
  tests/campaign/test_inputs.py \
  tests/campaign/test_runner.py \
  tests/integration/test_campaign_data_clis.py \
  tests/integration/test_hybrid_evaluation_cli.py \
  tests/integration/test_accuracy_campaign_cli.py \
  tests/integration/test_cloud_launcher.py
```

Expected: all targeted tests pass.

- [ ] **Step 5: Run static checks and the full suite**

```bash
uv run ruff check .
uv run pytest -q
```

Expected: Ruff exits zero and the complete test suite passes.

- [ ] **Step 6: Verify repository state and review the final diff**

```bash
git diff --check
git status --short
git log --oneline -8
```

Expected: no whitespace errors; only the intended documentation changes remain
uncommitted before the final documentation commit; prior tasks appear as atomic
commits.

- [ ] **Step 7: Commit documentation**

```bash
git add README.md docs/experiment-methodology.md docs/reproducibility.md docs/calibration-throughput-benchmark.md
git commit -m "docs: define fixed-shot decoder evidence"
```

- [ ] **Step 8: Perform final verification after the commit**

```bash
uv run ruff check .
uv run pytest -q
git status --short
```

Expected: Ruff and all tests pass; the worktree is clean.
