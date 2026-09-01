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

## Pilot selection and data roles

The committed campaign grid is
`[0.003, 0.005, 0.008, 0.012, 0.018, 0.025]`, with 256 pilot shots at each point.
Uniform-prior BP-LSD supplies pilot measurements. In the current implementation,
pilot `block_errors` count either syndrome-invalid corrections or observable
mismatches; syndrome validity is also reported separately. If every configured
point has zero block failures, the
pilot extends geometrically by a factor of 1.5, up to `p = 0.08`, until an
block failure is measured or the cap is reached.

Selection always retains the two lowest measured points. It extends one point
beyond an initial zero-failure prefix, retains measured points through 50%
baseline block-error rate, and inserts the midpoint before the first
majority-failure point. `selection.json` records the measurements, selected points,
and source hashes. These rows select the campaign's noise range; they are not a
final decoder comparison. Held-out evaluation applies the same
invalid-as-failure rule to all three methods.

Subsequent samples are immutable, role-separated shards:

- **train:** BP-LSD teacher generation and model fitting;
- **calibration:** checkpoint and probability-parameter selection; and
- **test:** untouched samples for paired held-out evaluation.

The default train and calibration caps (50,000 and 10,000 total shots) are divided
across selected rates, with any remainder assigned to lower rates. The default
test batch contains 2,048 shots at each selected rate. Each shard is at most 2,048
shots and derives its seed from the campaign seed, rate index, role, and shard
index. No role reuses another role's shard.

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

Calibration evaluates every saved epoch checkpoint over a fixed parameter grid.
The parameters transform the FNO logits while conditioning on the physical noise
rate, then clip the resulting probabilities away from zero and one.

**Soft-prior path.** BP-LSD receives the calibrated per-qubit probabilities and
decodes the original syndrome.

**Residual path.** Probabilities at least `0.5` form a hard proposal. The decoder
computes its residual syndrome, sets each prior from the proposal's uncertainty
`min(q, 1 - q)`, decodes a delta, and XORs the delta with the proposal.

Each method selects its calibration candidate independently by this lexicographic
rule:

1. prefer candidates with zero invalid corrections;
2. minimize block errors;
3. minimize NLL;
4. minimize measured FNO inference time;
5. prefer earlier epoch; then
6. break remaining ties by calibration parameter values.

The first four criteria represent scientific priorities; the final fields make
selection deterministic.

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
test shots, reports paired disagreement and confidence intervals, and stops a
noise point only at the configured failure target or shot cap.

## Paired held-out evaluation

At each selected physical error rate, all three decoders receive the same test
shots in the same order. Each shot records syndrome validity, logical mismatch,
the combined block-failure outcome, convergence, correction weight, iterations,
and decoder latency. Hybrid records additionally separate FNO, preprocessing,
BP-LSD, and end-to-end time. This pairing matters: the comparison asks which
decoder fails on a particular error instance, not only whether two aggregate
rates happen to be close.

For each hybrid, the summary includes the full 2 × 2 disagreement table against
uniform BP-LSD and the hybrid-minus-baseline block-error-rate delta. A
deterministic shot-level bootstrap gives the delta's 95% interval; each decoder's
individual block-error rate receives a Wilson interval. A hybrid is marked
`accuracy_compatible` only when every correction is syndrome-valid and the
paired delta interval is not strictly above zero. Latency does not participate in
that gate.

Sampling proceeds in deterministic batches. A noise point stops when all three
decoders have reached the configured failure target or when its shot cap is
reached. A campaign deadline may publish a resumable `partial_deadline` result,
but that status is not equivalent to scientific completion. Immutable batch
outcomes allow later resumption without changing already decoded shots.

## Threats to interpretation

- Results apply to one code, one error sector, and an independent code-capacity
  noise family.
- Pilot adaptation chooses the studied noise range from baseline observations.
- BP-LSD teacher labels are one valid representative, not a unique ground truth
  correction.
- Calibration searches checkpoints and parameters on calibration data; only the
  separate test role can support held-out comparison.
- CPU timings include different components across stages and depend on machine
  load, batching, and software versions.
- Willow surface-code hardware data is outside this methodology and must not be
  interpreted as part of the qLDPC campaign.

See the source-locked references in the [README](../README.md#sources).
