# Experiment methodology

This document describes the implemented accuracy campaign. For setup,
artifacts, and safe restart behavior, see [Reproducibility](reproducibility.md).

## Research question

The accuracy campaign tests whether a noise-conditioned ring FNO can supply useful
structure to BP-LSD on `lp(3,7)_16` independent-Z code-capacity noise. The intended
held-out comparison fixes data and decoder settings while varying the prior path:

1. uniform-prior BP-LSD;
2. calibrated FNO soft-prior BP-LSD; and
3. a calibrated hard FNO proposal followed by residual BP-LSD.

Preparation, training, calibration, and paired held-out evaluation are implemented.
Calibration results remain model-selection artifacts, not test-set evidence; only
the untouched test role supports the final comparison.

## Code and noise model

`experiments/01_build_lp_codes.py` reconstructs the source-locked lifted-product
CSS code. `experiments/02_validate_lp_codes.py` checks its hashes, commutation,
dimensions, rank-derived `k`, and row weights against the encoded source metadata.

The canonical campaign admits only this code identity:

| Quantity | Value |
| --- | ---: |
| `n` | 2,610 |
| `k` | 744 |
| `Hx` shape | 945 × 2,610 |
| `Hz` shape | 945 × 2,610 |
| `ell` | 45 |
| `Hx` and `Hz` row weight | 10 |
| Published distance field | upper bound 16 |

The Stim DEM contains one independent Z-error mechanism per physical qubit. At a
noise point `p`, every mechanism has probability `p`; its targets are the affected
X-check detectors and logical-X observables. Syndrome measurement is perfect.

## Noise-point selection and data roles

The committed campaign grid is
`[0.003, 0.005, 0.008, 0.012, 0.018, 0.025]`, with 256 pilot shots at each point.
Uniform-prior BP-LSD supplies pilot measurements. In the current implementation,
pilot `block_errors` count either syndrome-invalid corrections or observable
mismatches; syndrome validity is also reported separately. If every configured
point has zero block failures, the
pilot extends geometrically by a factor of 1.5, up to `p = 0.08`, until an
block failure is measured or the cap is reached.

The `selection_mode` is either `pilot` or `fixed`. In `pilot` mode, selection
retains the two lowest measured points, extends one point beyond an initial
zero-failure prefix, retains measured points through 50% baseline block-error
rate, and inserts the midpoint before the first majority-failure point.
`selection.json` records the measurements, selected points, and source hashes.
These rows select the campaign's noise range; they are not a final decoder
comparison. In `fixed` mode, the selected points are exactly the predeclared
`noise_grid`; its selection record is provenance, not evidence. Held-out
evaluation applies the same invalid-as-failure rule to all three methods.
Pilot publications are checked with an exact row schema, deterministic shard
coordinates and seeds, finite internally consistent timing, and a BP-LSD replay.
Evaluation persists a versioned verification receipt; its digest is part of each
batch's immutable provenance so bounded resumes can rehash the publication
without repeating the decoder replay.

Subsequent samples are immutable, role-separated shards:

- **train:** BP-LSD teacher generation and model fitting;
- **calibration:** checkpoint and probability-parameter selection; and
- **test:** untouched samples for paired held-out evaluation.

The default train and calibration caps (50,000 and 10,000 total shots) are divided
across selected rates, with any remainder assigned to lower rates. Calibration
proxy screening uses that calibration role; expensive hybrid decoding uses a
deterministic, rate-stratified subset capped at 512 shots. Test generation
materializes the configured cap of 200,000 shots per selected rate in immutable
shards; held-out evaluation later consumes those shots in 2,048-shot batches.
Each shard is at most 2,048 shots and derives its seed from the campaign seed, rate
index, role, and shard index. No role reuses another role's shard.

## Uniform BP-LSD configuration

All three paths use the same pinned BP-LSD implementation:

| Setting | Value |
| --- | --- |
| BP method | `minimum_sum` |
| Schedule | `serial` |
| Maximum iterations | 100 |
| Minimum-sum scaling | 0.0 |
| LSD method | `LSD_E` |
| LSD order | 5 |

For the baseline and teacher, the error channel is uniform at the shard's physical
error rate. Teacher generation rejects any syndrome-invalid BP-LSD correction.

## Conditional FNO

The cyclic lift gives the model a natural one-dimensional periodic coordinate:

- syndrome input: 945 bits → `(21, 45)`;
- noise input: one `(1, 45)` channel filled with the shard error rate's log-odds;
- total input: `(22, 45)`;
- correction target: 2,610 BP-LSD bits → `(58, 45)`.

The fixed model has width 32, 12 Fourier modes, and two spectral/local blocks. It
uses weighted binary cross-entropy, Adam with learning rate `0.001`, batch size
128, and seed 1701 for 60 configured epochs. A deterministic, noise-stratified
train/validation split is derived from the training shards. Every epoch produces
a checkpoint candidate and validation NLL.

## Calibration and hybrid decoders

Calibration is deterministic and two-stage. It never reads the test role.

1. For every saved epoch and all 48 fixed `(alpha, beta, temperature)` tuples,
   compute calibration-only proxies without invoking BP-LSD: correction NLL,
   threshold-proposal invalid count, and mean residual-syndrome weight. FNO logits
   are evaluated once per checkpoint over the full calibration role.
2. Independently shortlist four candidates for each method. Soft-prior ranking
   uses NLL. Residual ranking uses proposal invalid count, residual-syndrome
   weight, then NLL. The union contains at most eight candidates.
3. Run both hybrid decoders for that union on a maximum of 512 calibration shots.
   The subset is selected by SHA-256 ranking within each error rate followed by
   round-robin allocation, and its seed, per-rate counts, and index hash are
   recorded.

The parameters transform the FNO logits while conditioning on the physical noise
rate, then clip the resulting probabilities away from zero and one. Proxy ranking
is only a computational screen: final soft and residual winners are each
restricted to their own independently constructed shortlist and chosen using
their actual hybrid-decoding outcomes.

**Soft-prior path.** BP-LSD receives the calibrated per-qubit probabilities and
decodes the original syndrome.

**Residual path.** Probabilities at least `0.5` form a hard proposal. The decoder
computes its residual syndrome, sets each prior from the proposal's uncertainty
`min(q, 1 - q)`, decodes a delta, and XORs the delta with the proposal.

Each method selects its calibration candidate independently by this lexicographic
rule:

1. prefer candidates with no invalid corrections;
2. minimize block errors;
3. minimize NLL;
4. prefer earlier epoch; then
5. break remaining ties by calibration parameter values.

Measured FNO inference time is diagnostic and never selects a winner. Each
invocation completes one bounded checkpoint-wide screen or one shortlisted hybrid
decode. `progress.json` binds the exact config, calibration shard manifests,
model/checkpoints, screening policy, decode subset, shortlists, and ordered work;
resume rejects any divergence.

The audited pre-policy capacity diagnostic in
[Calibration throughput benchmark](calibration-throughput-benchmark.md) did not
complete one real eight-shot candidate within 10m45s. The earlier 4.2167
candidate-shots/s estimate is withdrawn. FNO-only screening avoids hybrid decoder
calls but has no assigned runtime, and the hybrid stage has no validated
conservative bound. Canonical Cloud execution is therefore fail-closed until
killable decoder-unit timeouts and a representative worst-case 8-vCPU benchmark
are reviewed.

## Metrics: accuracy before speed

A predicted correction `c` for syndrome `s` is valid when
`Hx @ c mod 2 = s`. A shot is a logical block failure when any predicted logical
observable differs from the sampled observable. Syndrome-invalid learned
corrections count as block failures even if their observable bits happen to match.

The reporting order is:

1. syndrome-invalid count and syndrome-valid rate;
2. logical block-error count and rate;
3. 95% Wilson interval for the block-error rate where the scoring stage implements
   it;
4. exact-observable-match rate and decoder diagnostics;
5. teacher-bit accuracy and NLL as model diagnostics; and
6. measured latency, with batch size and timed components.

Teacher-bit accuracy cannot establish decoding correctness because valid
corrections are not unique and a small bitwise difference can violate a check.
Calibration accuracy cannot establish generalization because it is used for
selection. The evaluator scores all three frozen methods on the same untouched
test shots and reports paired disagreement. Its `test_stopping_mode` is either
`adaptive` or `fixed`: adaptive mode stops a rate after every decoder reaches
`target_failures`, or at the shot cap; fixed mode stops only at the shot cap.
`target_failures` is inactive in fixed mode.

## Paired held-out evaluation

At each selected physical error rate, all three decoders receive the same test
shots in the same order. Each shot records syndrome validity, logical mismatch,
the combined block-failure outcome, convergence, correction weight, iterations,
and decoder latency. Hybrid records additionally separate FNO, preprocessing,
BP-LSD, and end-to-end time. This pairing matters: the comparison asks which
decoder fails on a particular error instance, not only whether two aggregate
rates happen to be close.
Persisted FNO latency totals are derived from the same per-shot arrays written to
the batch archive. Verification uses a scale-aware floating-point tolerance when
reading batches produced around an atomic rename boundary; timing remains
diagnostic and cannot affect an accuracy status.

For each hybrid, the summary includes the full 2 × 2 disagreement table against
uniform BP-LSD and the hybrid-minus-baseline block-error-rate delta. Each
decoder's individual block-error rate receives a Wilson interval where reported.

The paired test conditions on discordant shots and uses an exact binomial
(McNemar) test under equal paired failure probability. Its Clopper-Pearson interval
describes the share of discordances in which only the hybrid fails; it is not a
confidence interval for the marginal block-error-rate difference.

An inconclusive paired test is not evidence of equivalence or noninferiority.
Adaptive and deadline-truncated evaluations report paired quantities only as
diagnostics. Directional comparison statuses are assigned only after a complete,
predeclared fixed-shot evaluation. Latency does not participate in those statuses.

Evaluation consumes the pre-generated test role in deterministic batches. In
adaptive mode, a noise point stops when all three decoders have reached the
configured failure target or when its shot cap is reached. In fixed mode, it runs
to the shot cap regardless of failures. A campaign deadline may publish a
resumable `partial_deadline` result, but that status is not equivalent to
scientific completion. Immutable batch outcomes allow later resumption without
changing already decoded shots.

The `accuracy_disconfirm_p0375` profile is deliberately asymmetric: it has one
predeclared rate, `p=0.0375`, and is intended to detect candidate harm. A detected
harm result falsifies this candidate at that rate. An inconclusive or benefit
result only permits the next experiment; baseline tuning, at least three training
seeds, and a larger confirmatory test remain required before a positive accuracy
claim.

## Threats to interpretation

- Results apply to one code, one error sector, and an independent code-capacity
  noise family.
- When pilot mode is used, pilot adaptation chooses the studied noise range from
  baseline observations.
- BP-LSD teacher labels are one valid representative, not a unique ground truth
  correction.
- Calibration screens all checkpoints/parameters and decodes shortlisted
  candidates on calibration-only data; proxy quality can omit a genuinely better
  hybrid. Only the separate test role can support held-out comparison.
- Adaptive and deadline-truncated paired quantities are diagnostic, not
  anytime-valid or directional evidence.
- The conditional Clopper-Pearson interval is not an interval for the marginal
  block-error-rate difference.
- A single-rate disconfirming run is asymmetric and cannot establish a positive
  accuracy claim.
- CPU timings include different components across stages and depend on machine
  load, batching, and software versions.
- Willow surface-code hardware data is outside this methodology and must not be
  interpreted as part of the qLDPC campaign.

See the source-locked references in the [README](../README.md#sources).
