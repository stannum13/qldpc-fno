# Accuracy-First Hybrid qLDPC Decoder Campaign

## Purpose

The first `lp(3,7)_16` smoke experiment established a useful negative result. A
small Fourier neural operator (FNO) reproduced 99.38% of held-out BP-LSD correction
bits, but none of its 128 held-out corrections satisfied the measured syndrome and
every shot was a logical block failure. Per-bit imitation is therefore not a valid
decoder objective by itself.

This campaign asks a narrower and more consequential question:

> Can an FNO supply useful soft information to an algebraic decoder while BP-LSD
> preserves exact syndrome validity and competitive logical accuracy?

Accuracy is the primary objective. Runtime and iteration counts are recorded to
prepare a later speed-focused campaign, but no speed claim is permitted until the
hybrid matches the baseline's logical performance within statistical uncertainty.

## Scope

The campaign remains on the paper's `lp(3,7)_16` code with ring order `ell=45`.
It uses the existing independent-Z code-capacity model with perfect measurements.
It does not claim circuit-level, neutral-atom, temporal, Willow, or cross-code-size
performance. Those are later experiments after the in-size decoder is correct.

Three decoders are compared on identical held-out error samples:

1. Uniform-prior BP-LSD baseline.
2. FNO-conditioned soft-prior BP-LSD.
3. Hard FNO proposal followed by residual-syndrome BP-LSD repair.

A syndrome-aware learned world model is explicitly deferred. It is a separate
research direction rather than an extra loss term in this campaign.

## Dataset and split

### Noise grid

Begin with a deterministic pilot grid:

```text
p in {0.003, 0.005, 0.008, 0.012, 0.018, 0.025}
```

The pilot samples 256 shots per point and records baseline BP-LSD failures and
latency. Points at which every baseline shot succeeds are retained as low-noise
controls, but the full campaign prioritizes points that produce measurable errors.
If every point has zero failures, extend the grid geometrically up to `p=0.08`. If
the baseline block-error rate exceeds 50%, insert a midpoint below that point and
do not increase the noise further.

### Independent roles

Every shot is assigned before decoding to one immutable role and is never reused:

- training: BP-LSD correction labels for FNO fitting;
- calibration: temperature and prior-mixing selection;
- test: final decoder comparison only.

Seeds are derived from `(campaign_seed, p_index, role, shard_index)` using a stable
SHA-256 scheme. Each role uses separate seeds. The packed error, detection, and
observable files remain the source of truth.

### Adaptive sizes

Default caps are:

- up to 50,000 training shots across the selected grid;
- up to 10,000 calibration shots;
- test batches of 2,048 shots per noise point;
- stop test collection for a point after 200 baseline block failures and 200
  failures for each hybrid, or after 200,000 shots, whichever happens first.

The wall-clock deadline can stop collection earlier. Every partial result is valid
and reports its achieved shot/error counts and Wilson confidence intervals.

## Conditional FNO

The existing cyclic representation is retained. A broadcast noise channel is
added, so the input is the 21 syndrome fields plus `logit(p)` at every ring site.
The model outputs one logit for each of the 58 correction channels at each of 45
ring positions.

Training labels are the syndrome-valid corrections selected by the pinned BP-LSD
teacher. Weighted binary cross-entropy remains a diagnostic imitation loss. Model
selection instead uses calibration-set hybrid logical error and validity:

1. reject any checkpoint that produces a syndrome-invalid final hybrid correction;
2. among valid checkpoints, minimize calibrated hybrid block-error rate;
3. break statistical ties using negative log likelihood, then inference latency.

The model is trained once across the selected noise grid. Noise-conditioned input
prevents a separate model per physical error rate and makes interpolation testable.

## Decoder A: uniform-prior BP-LSD

The baseline preserves the pinned configuration:

- minimum-sum BP;
- serial schedule;
- 100 maximum iterations;
- `ms_scaling_factor=0.0`;
- exhaustive LSD order 5.

The channel is the uniform physical error rate `p`. Per-shot correction, logical
prediction, syndrome validity, convergence, BP iteration count, and latency are
recorded.

## Decoder B: FNO-conditioned soft-prior BP-LSD

For each syndrome, the FNO produces logits `z_i`. Calibration fits two scalar
parameters on the calibration split only:

```text
q_i = sigmoid(alpha * z_i / temperature + beta * logit(p))
```

Probabilities are clipped to `[1e-5, 1-1e-5]` and passed to BP-LSD through
`update_channel_probs(q)`. The parameter grid is deliberately small and fixed in
the manifest. The test split is evaluated once using the selected parameters.

This is a heuristic posterior-informed decoder because the FNO has already seen
the syndrome. The campaign does not describe `q_i` as an independent physical
noise channel. Its value is empirical: whether it guides BP-LSD to syndrome-valid,
logically correct representatives more often than a uniform prior.

## Decoder C: hard proposal with residual repair

Threshold calibrated probabilities to obtain an FNO proposal `c0`. Compute:

```text
residual = syndrome XOR (Hx @ c0 mod 2)
uncertainty_i = min(q_i, 1 - q_i)
```

BP-LSD decodes the residual using the uncertainty vector. The final correction is
`c = c0 XOR delta`. A final syndrome check is mandatory. A failed or invalid repair
is a block failure.

This method tests whether the FNO can remove most of the correction structure and
leave an easier residual decoding problem. It records residual syndrome weight,
repair weight, BP iterations, and latency, but accuracy remains the selection
criterion.

## Metrics and decision rules

Report for every `(noise rate, decoder)` pair:

- shots, block failures, block-error rate, and 95% Wilson interval;
- syndrome-valid count and rate;
- logical-mismatch count among all shots and among valid shots;
- BP convergence and iteration-count distribution;
- correction and residual weights;
- FNO inference, BP-LSD, and end-to-end latency distributions;
- calibration parameters and probability reliability bins.

Paired outcomes are retained because all decoders see identical shots. Report the
paired disagreement table and a paired bootstrap interval for the change in block
error. No hybrid is called accurate unless:

1. syndrome validity is 100% on collected test shots;
2. its block-error confidence interval is statistically compatible with or below
   baseline BP-LSD at the declared operating points;
3. calibration and test artifacts have disjoint seeds and verified provenance.

If both hybrids fail, the result is still complete: characterize whether failures
come from calibration, logical representative selection, or residual repair.

## Cloud execution

### Local pilot

Run unit/integration tests and a reduced end-to-end shard locally. The pilot must
verify the chosen probability formula, exact residual identity, packed replay, and
artifact-chain hashes before cloud resources are created.

### Cloud Run Job

Use one CPU-only Cloud Run Job in the authenticated GCP project. The container is
built from the pinned `uv.lock` and the exact Git commit. Defaults:

- 8 vCPUs;
- 32 GiB memory;
- 8-hour task timeout;
- one task and zero retries for the first campaign;
- no GPU;
- region chosen from available low-friction project regions;
- a unique campaign ID on every launch.

The job executes the pilot selection, dataset generation, training, calibration,
and adaptive evaluation sequentially. CPU-bound decoding may use bounded worker
processes; model training uses CPU inside Cloud Run. This intentionally favors a
simple resumable first campaign over distributed orchestration.

### Persistence and resumption

All durable outputs go to a campaign-specific Cloud Storage prefix. Each stage
writes to a temporary object prefix, validates hashes, and publishes a completion
manifest last. On restart, a stage is skipped only when its completion manifest and
all declared hashes verify.

Checkpoint at least every 2,048 decoded shots and every training epoch. A monotonic
deadline leaves ten minutes for the final checkpoint and summary. The job creates
no long-lived VM. Cloud Run job configuration and container references are retained
for reproducibility; a cleanup command removes them when requested.

### Cost and safety boundary

The launcher prints the selected project, region, CPU, memory, timeout, Git commit,
bucket, and exact command, then requires an explicit `--execute` flag. The job has
an 8-hour hard timeout and cannot allocate a GPU. It never deletes pre-existing
buckets, repositories, images, or campaign prefixes. A unique prefix prevents
overwrites.

## Atomic commands and artifacts

New commands are thin wrappers around tested library functions:

```text
13_pilot_noise_grid.py
14_generate_campaign_shards.py
15_train_conditional_fno.py
16_calibrate_hybrid_priors.py
17_evaluate_hybrid_decoders.py
18_summarize_accuracy_campaign.py
```

Cloud support is separated from scientific logic:

```text
scripts/run_accuracy_campaign.sh
scripts/build_cloud_image.sh
scripts/launch_cloud_campaign.sh
Dockerfile
```

The final summary contains no timing-only winner. It names the accuracy-compatible
decoders, their confidence intervals, and the evidence required before a later
speed phase.

## README contract

The repository README becomes a progressive technical narrative:

1. **Two-minute explanation:** quantum errors, syndromes, why 99% bit accuracy can
   still mean a completely broken decoder, and the central experiment.
2. **Engineer path:** exact commands, artifact tree, resumability, local versus GCP
   execution, expected resource use, and cleanup.
3. **Scientist path:** code construction, noise model, decoder definitions,
   calibration protocol, statistical tests, limitations, and reproducibility.
4. **Results:** machine-generated tables copied from the immutable campaign summary;
   no claims beyond completed data.
5. **Roadmap:** circuit-level noise, Willow temporal data, code-size transfer, FPGA
   emulation, and the separate syndrome-aware world-model direction.

Simple language comes first. Equations and implementation detail follow without
replacing the plain-English explanation.

## Acceptance criteria

The campaign implementation is complete when:

- local tests cover probability calibration, residual algebra, validity gates,
  provenance, resumption, and paired metrics;
- a reduced local campaign completes from one command;
- a dry-run cloud command reports all billable resources without creating them;
- an executed Cloud Run campaign writes a final verified summary or a valid partial
  summary on deadline;
- every final correction is explicitly syndrome-checked;
- baseline and both hybrid decoders are evaluated on identical held-out shots;
- README instructions reproduce the local path and explain the scientific boundary
  to all three target audiences;
- commit history uses concise technical descriptions of repository changes and results.

## Release-hardening amendment (2026-09-01)

Final review found that the original exhaustive calibration grid and ten-minute
shutdown grace could not support the stated cloud boundary. The benchmark in
[`docs/calibration-throughput-benchmark.md`](../../calibration-throughput-benchmark.md)
measured the real canonical-code hybrid path before this policy change.

Calibration is now deterministic and two-stage. Stage 1 evaluates every declared
checkpoint and parameter tuple on calibration-role shots using only FNO inference,
correction NLL, threshold-proposal syndrome validity, and residual syndrome
weight. It creates separate four-candidate shortlists: soft-prior ranks by NLL;
residual ranks by proposal validity and residual weight, then NLL. Stage 2 runs
both hybrid decoders on the union of those lists, at most eight candidates, using
a 512-shot calibration-only subset selected by SHA-256 ranking independently
within every noise rate. Soft-prior and residual winners are selected independently
from their own declared shortlists. Test-role artifacts are neither read nor
referenced by calibration.

The screen policy, subset derivation and SHA-256, shortlist membership, config,
model, checkpoint, code, and calibration-shard hashes are part of progress and
selection provenance. Resume accepts only an exact ordered prefix of this policy;
any policy or source change is rejected.

The canonical campaign is not claimed to fit one execution. The pre-change
benchmark proves only that the expensive revised calibration second stage has a
conservative planning bound near 23 minutes; it does not validate whole-campaign
speed. Canonical cloud creation therefore requires explicit acknowledgement that
multiple executions may be necessary, and later executions reuse the same job and
immutable store through a dedicated resume command.

Cloud Run keeps an eight-hour outer task timeout. New scientific work stops at
7h15m at the latest, reserving at least 45 minutes for a small immutable partial
summary and, time permitting, checkpoint snapshots. Absolute monotonic deadlines
are propagated into publication and materialization operations; a partial summary
is attempted before optional large snapshots.

Cloud build input is a Git archive of the exact committed runtime paths, not the
working tree. The Docker ignore policy is default-deny as defense in depth, and
an ignored sentinel regression must prove it is absent from the submitted archive.
Runner-managed artifacts use `inputs/`, `pilot/`, `shards/`, `training/`,
`calibration/`, `evaluation/`, and `summary/`. Manual stage layouts are documented
separately. `source-lock.json` is not a runner-managed input and must not be listed
in the runner artifact tree.
