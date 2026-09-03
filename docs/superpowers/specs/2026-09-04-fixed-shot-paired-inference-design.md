# Fixed-Shot Paired-Inference Repair

## Purpose

The current accuracy campaign preserves paired decoder outcomes but summarizes the
hybrid-minus-baseline difference with an empirical percentile bootstrap. At small
sample sizes this can return a zero-width interval. For example, eight
hybrid-only failures and no other discordances produce `[1.0, 1.0]`, even though
eight observations cannot determine the population risk difference exactly.

The campaign also labels a hybrid `accuracy_compatible` whenever the lower bound
of that interval is not above zero. That rule treats failure to detect harm as
evidence of compatibility. It is not an equivalence or noninferiority test because
the campaign declares neither a degradation margin nor a procedure that controls
the corresponding error rate.

This change has two goals:

1. replace the invalid paired summary and ambiguous compatibility label with exact,
   explicitly directional evidence; and
2. make a predeclared one-rate, fixed-shot disconfirming experiment executable
   through the existing campaign stages.

The change does not add BP-LSD hyperparameter selection, claim noninferiority,
authorize a speed comparison, or turn reduced execution checks into scientific
results.

## Scope

The scientific system remains the `lp(3,7)_16` independent-Z code-capacity
experiment with perfect syndrome information. The same three decoder arms remain:

1. uniform-prior BP-LSD;
2. FNO soft-prior BP-LSD; and
3. thresholded FNO proposal followed by residual BP-LSD repair.

All arms continue to receive identical test shots. Logical failure continues to
mean either an invalid final syndrome or a mismatch in at least one logical
observable. Correction strings are not compared directly.

The following work is deferred:

- calibration-only BP-LSD baseline tuning;
- a formal paired noninferiority margin and confidence interval for the marginal
  risk difference;
- anytime-valid inference for adaptively stopped campaigns;
- multiple-training-seed aggregation;
- circuit-level, repeated-round, Willow, code-family-transfer, and hardware-latency
  experiments.

## Statistical contract

### Paired outcomes

For each test shot define the baseline and hybrid block-failure indicators
`B, H in {0, 1}`. Preserve the complete paired 2 by 2 table:

```text
both_succeed:          B=0, H=0
baseline_only_failure: B=1, H=0
hybrid_only_failure:   B=0, H=1
both_fail:             B=1, H=1
```

The marginal point estimate remains:

```text
block_error_delta = mean(H - B)
                  = (hybrid_only_failure - baseline_only_failure) / shots
```

This point estimate is descriptive. The implementation will no longer attach an
empirical percentile-bootstrap confidence interval to it.

### Exact discordant-pair inference

Only discordant pairs distinguish the two methods. Let:

```text
n_discordant = baseline_only_failure + hybrid_only_failure
```

Under the paired equal-performance null, either decoder is equally likely to be
the failing member of a discordant pair. Use `scipy.stats.binomtest` with null
probability `0.5` to report:

- an exact two-sided McNemar p-value;
- an exact one-sided p-value for excess hybrid harm;
- an exact one-sided p-value for hybrid benefit; and
- a 95% Clopper-Pearson interval for
  `hybrid_only_failure / n_discordant`.

The Clopper-Pearson interval is explicitly named
`hybrid_harm_share_given_discordance_95ci`. It is conditional on a discordance and
must not be described as an interval for the marginal block-error-rate delta.

When there are no discordant pairs:

- both exact p-values are `1.0`;
- the two-sided p-value is `1.0`;
- the conditional share is `null`;
- its interval bounds are `null`; and
- the comparison status is `no_discordances`.

### Comparison status

Remove `accuracy_compatible`. For a complete fixed-shot comparison, assign one of
four descriptive statuses at unadjusted per-comparison `alpha = 0.05`:

- `harm_detected`: the exact two-sided p-value is at most `0.05` and
  `hybrid_only_failure > baseline_only_failure`;
- `benefit_detected`: the exact two-sided p-value is at most `0.05` and
  `baseline_only_failure > hybrid_only_failure`;
- `inconclusive`: discordances exist but the exact two-sided test is not
  significant; or
- `no_discordances`: the observed paired failure indicators are identical.

For an adaptive or incomplete comparison, report the paired counts and p-values as
diagnostics but set `comparison_status` to `not_fixed_sample`. No status means
noninferiority or equivalence. A positive decoder claim across multiple methods or
rates requires a separately declared multiplicity policy and is outside this
change.

Individual decoder block-error rates retain their 95% Wilson intervals. Those
intervals are valid as fixed-sample marginal intervals, not as confidence
sequences.

## Campaign configuration

Add two required string fields to `CampaignConfig`:

```text
selection_mode: "pilot" | "fixed"
test_stopping_mode: "adaptive" | "fixed"
```

Unknown values fail validation. Existing committed canonical and reduced
configurations are updated explicitly rather than receiving implicit defaults.
This keeps the configuration hash a complete description of the campaign policy.

### Selection modes

`selection_mode = "pilot"` preserves the existing pilot behavior, including
geometric extension and data-dependent selection.

`selection_mode = "fixed"` treats the configured `noise_grid` as the complete,
predeclared selected-rate list. The stage-13 command still publishes
`pilot/selection.json` and `pilot/manifest.json` so every downstream provenance
check remains intact, but it performs no sampling and no baseline decoding. Its
selection artifact records:

```json
{
  "selection_mode": "fixed",
  "selected_noise_points": [0.0375],
  "pilot_rows": [],
  "evidence_role": "predeclared_selection_not_evidence"
}
```

The config and code-manifest hashes remain mandatory. A fixed selection can
contain one rate. A pilot selection continues to require whatever controls the
existing selection algorithm needs.

### Stopping modes

`test_stopping_mode = "adaptive"` preserves the current target-failure-or-shot-cap
behavior. Its paired inferential status is always `not_fixed_sample`.

`test_stopping_mode = "fixed"` ignores `target_failures` while collecting test
outcomes and stops only after exactly `max_test_shots_per_point` for each selected
rate. A deadline may still produce a resumable `partial_deadline` artifact, but a
partial rate receives `comparison_status = "not_fixed_sample"` and cannot support
a fixed-sample claim.

The `target_failures` field remains required for schema stability and is labelled
inactive in fixed mode. Configuration validation continues to require it to be
positive and no larger than the shot cap.

## One-rate disconfirming configuration

Add `configs/accuracy_disconfirm_p0375.json` with:

```json
{
  "campaign_seed": 20260904,
  "noise_grid": [0.0375],
  "selection_mode": "fixed",
  "pilot_shots_per_point": 1,
  "train_shots_cap": 10000,
  "calibration_shots_cap": 2048,
  "calibration_decode_shots_cap": 128,
  "calibration_shortlist_per_method": 1,
  "test_batch_shots": 256,
  "max_test_shots_per_point": 2048,
  "target_failures": 200,
  "test_stopping_mode": "fixed",
  "training_epochs": 60,
  "training_batch_size": 128,
  "training_learning_rate": 0.001,
  "training_seed": 1701,
  "checkpoint_every_epochs": 1,
  "cloud_cpu": 8,
  "cloud_memory": "32Gi",
  "cloud_timeout_seconds": 28800,
  "checkpoint_grace_seconds": 2700
}
```

`pilot_shots_per_point` remains present because the strict shared schema requires
it, but fixed selection never consumes it. The calibration shortlist and decoded
subset are deliberately small: this experiment is designed to reject a bad
candidate cheaply, not support a positive learned-decoder claim.

The initial execution is local and resumable. Existing canonical Cloud launch
gates remain unchanged. Completing this configuration with one training seed can
falsify this candidate at this operating point. Surviving it only justifies the
next experiment; it does not demonstrate robustness across optimization seeds.

## Artifact changes

Each per-rate evaluation summary replaces:

```text
accuracy_compatible
block_error_delta_95ci_low
block_error_delta_95ci_high
bootstrap_samples
bootstrap_seed
```

with:

```text
comparison_status
block_error_delta
both_succeed
both_fail
baseline_only_failure
hybrid_only_failure
discordant_pairs
mcnemar_exact_pvalue_two_sided
mcnemar_exact_pvalue_harm
mcnemar_exact_pvalue_benefit
hybrid_harm_share_given_discordance
hybrid_harm_share_given_discordance_95ci_low
hybrid_harm_share_given_discordance_95ci_high
```

The evaluation manifest records `selection_mode`, `test_stopping_mode`, and the
inactive/active status of `target_failures`. Existing results from another commit
are not resumed because the repository commit and source hashes are already part
of campaign provenance.

The generated Markdown summary prints paired counts, exact p-values, and the
conditional interval with its full name. It contains no “compatible,”
“noninferior,” or “equivalent” label. Adaptive and partial results are visibly
marked diagnostic.

## Data flow

The fixed experiment reuses the existing stages:

```text
validated LP code
  -> predeclared fixed selection publication
  -> role-separated train/calibration/test Stim shards
  -> BP-LSD teacher and conditional FNO training
  -> calibration-only hybrid selection
  -> exactly 2,048 paired test shots
  -> marginal Wilson intervals + exact discordant-pair inference
  -> generated result summary
```

No test outcome participates in model, calibration, rate, or stopping selection.
All three decoder arms continue to consume the same test indices in the same
order.

## Error handling and resumability

- Invalid configuration modes fail before campaign artifacts are created.
- Fixed selection refuses a selection artifact whose selected rates differ from
  the configured `noise_grid`.
- Fixed evaluation refuses a completed status unless every rate has exactly the
  configured shot count.
- A deadline interruption publishes partial progress but not a fixed-sample
  comparison status.
- Resumption preserves already materialized test shots and immutable batch
  outcomes.
- Mixing artifacts produced under different selection or stopping modes fails the
  existing hash/provenance checks.
- A summary missing the new paired fields fails verification rather than silently
  falling back to the old bootstrap schema.

## Testing

### Paired statistics

Unit tests cover:

- eight hybrid-only failures: finite-width exact conditional interval, harm
  detected, and no `[1.0, 1.0]` interval;
- eight baseline-only failures: benefit detected;
- balanced discordances: inconclusive;
- no discordances: null conditional share/interval and `no_discordances`;
- deterministic results with no bootstrap seed;
- invalid shapes and non-boolean inputs.

### Configuration and selection

Unit and integration tests cover:

- accepted and rejected `selection_mode` values;
- accepted and rejected `test_stopping_mode` values;
- committed configurations loading under the strict schema;
- one-point fixed selection remaining exactly one point;
- pilot mode retaining its current adaptive selection behavior;
- fixed selection provenance and tamper rejection.

### Evaluation

Integration tests cover:

- fixed mode ignoring an early target-failure count;
- fixed mode reaching the exact shot cap;
- adaptive mode preserving current stopping behavior;
- partial fixed evaluation reporting `not_fixed_sample`;
- generated JSON and Markdown omitting compatibility/bootstrap language;
- the disconfirming configuration reaching the existing downstream stage
  interfaces in a bounded fixture.

The full test suite and Ruff must pass before any campaign run is recommended.

## Documentation

Update the README and methodology to state:

- paired evidence is based on exact discordant-pair inference;
- the conditional interval is not a marginal risk-difference interval;
- `inconclusive` is not evidence of noninferiority;
- adaptive campaigns produce diagnostic conventional summaries only;
- the one-rate configuration is an asymmetric disconfirming experiment;
- a surviving candidate still requires baseline tuning, multiple training seeds,
  and a larger confirmatory test before a positive accuracy claim;
- timing remains out of scope.

The earlier reduced Cloud result remains labelled non-scientific. Its existing
immutable artifact is not rewritten.

## Acceptance criteria

The repair is complete when:

- the empirical paired percentile bootstrap is no longer used by campaign
  evaluation;
- `accuracy_compatible` is absent from new evaluation and summary artifacts;
- eight all-in-one-direction discordances cannot produce a zero-width population
  confidence interval;
- fixed selection publishes exactly the configured one-rate list without sampling;
- fixed evaluation consumes exactly the configured shot count unless interrupted;
- interrupted fixed evaluations cannot be presented as fixed-sample evidence;
- `configs/accuracy_disconfirm_p0375.json` passes strict validation;
- the disconfirming path uses different train, calibration, and test sampler roles;
- no speed, equivalence, noninferiority, threshold, or code-family claim is enabled;
  and
- targeted tests, the full test suite, and Ruff pass.
