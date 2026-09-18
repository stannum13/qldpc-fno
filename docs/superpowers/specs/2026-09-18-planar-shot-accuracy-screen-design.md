# Fresh planar shot-accuracy screen design

## Purpose

The existing two-view tensor rule has a locked conditional result: on syndromes
drawn from an i.i.d. depolarizing marginal, it usually reaches the same logical
class as an unrestricted tensor contraction with much less estimated arithmetic
work. A historical syndrome-convention defect prevents those samples from being
used for end-to-end logical block-error rate (BLER). This experiment asks the
next, narrower question on entirely fresh data:

> When each sampled Pauli error is correctly paired with its symplectic syndrome,
> does the two-view tolerance rule preserve the exact tensor decoder's shot-level
> logical outcome, and how do both compare descriptively with a calibrated
> converging-matching decoder?

This is an accuracy screen. It makes no FPGA latency, circuit-level noise,
threshold, qLDPC-transfer, or calibrated-posterior claim.

## Alternatives considered

1. **Run all six historical strata immediately.** This offers continuity but
   spends substantial exact-contraction work on low-event strata before the
   recovery/class mapping has passed a real-shot test.
2. **Recommended: a focused distance-5 screen at `p=0.10` and `p=0.15`.** These
   points produced enough failures in the post-hoc diagnostic to expose logical
   differences, while remaining within the already audited exact planar tensor
   implementation.
3. **Skip exact inference and compare only with matching.** This is cheaper but
   cannot test the central claim that the adaptive contraction preserves the
   exact logical decision.

The implementation uses option 2. A later six-stratum confirmation is permitted
only after this screen and a new preregistration.

## Fixed sample roles

All seeds are derived independently as the unsigned big-endian integer encoded
by the first eight bytes of
`SHA256("{domain}|d5|p{error_rate:.6f}|i{shot_index:06d}")`. The generator
requires every child seed to be unique within an artifact and disjoint across
the calibration and screen artifacts. The evaluator reconstructs the literal
identity string and seed for every role/rate/index. The source/config freeze is
committed and pushed before either domain is opened.

- calibration domain: `qldpc-fno/planar-shot-accuracy/calibration/v1`;
- screen domain: `qldpc-fno/planar-shot-accuracy/screen/v1`;
- code: qecsim unrotated `PlanarCode(5, 5)`;
- channel: qecsim i.i.d. depolarizing code-capacity noise;
- error rates: `0.10` and `0.15`;
- calibration shots: 512 per rate;
- screen shots: 2,048 per rate.

Calibration selects a single correlated-matching configuration shared by both
rates. Screen shots are never inspected during baseline selection. Samples store
the deterministic seed, physical binary-symplectic error, and syndrome. Every
reader regenerates the error and requires
`qecsim.paulitools.bsp(error, code.stabilizers.T) == syndrome`.

## Decoder arms

### Unrestricted tensor reference

The existing unrestricted MPS contraction returns four logical-coset masses in
qecsim's `I, X, Y, Z` order. Both unrestricted row and column contractions must
select the same winner, differ by at most `1e-10` in normalized probability and
`1e-8` in maximum pairwise log-mass ratio, and each top-versus-runner-up
probability margin must exceed twice the maximum row/column probability
discrepancy. Otherwise the shot invalidates canonical evaluation as numerically
ambiguous. The certified highest-mass class is converted to a recovery by
starting from `PlanarMPSDecoder.sample_recovery` and applying no logical, logical
X, logical X then Z, or logical Z respectively. Finite positive masses and a
unique maximum are required; an invalid or tied reference invalidates the shot.

### Two-view tolerance policy

Run column and row contractions at tolerance `0.01`. Accept the column class only
when both contractions are valid and agree. Otherwise run column `chi=8`. Charge
both tolerance views and any fallback to the policy. This is the already frozen
symbolic policy; it is not retuned on calibration or screen data.

### Strong classical baseline

Calibrate qecsim `PlanarCMWPMDecoder` over the Cartesian grid:

- `factor in {1, 2, 3, 4}`;
- `max_iterations in {2, 4, 8}`;
- `box_shape in {"t", "r"}`;
- `distance_algorithm in {2, 4}`.

Choose one shared configuration by, in order: minimum pooled calibration
failures, minimum worst-rate failure count, then lexicographic order of
`(factor, max_iterations, box_shape, distance_algorithm)`. Also run ordinary
`PlanarMWPMDecoder` as a named reference. The writeup calls the selected baseline
"calibrated CMWPM within the declared grid," not globally optimal matching.

## Correctness and logical scoring

Every recovery must reproduce the observed syndrome under the symplectic product.
For sampled error `e` and recovery `r`, form `residual = (e + r) mod 2`.

- syndrome validity requires `bsp(residual, stabilizers.T)` to be all zero;
- logical success requires `bsp(residual, logicals.T)` to be all zero;
- any invalid recovery is a logical failure and an integrity error.

Raw correction-string equality is never used. Every arm returns one joint
binary-symplectic recovery. Tensor inference models the depolarizing Pauli
distribution directly and CMWPM iteratively couples primal and dual matches;
ordinary MWPM is retained as a conventional factorized reference and does not
exploit the X/Z correlation.

## Endpoints and interpretation

The independent unit is one physical shot. The primary screen endpoint is the
two-view policy's selected-class mismatch rate against unrestricted tensor
inference, separately at each physical error rate. A pass requires zero class
mismatches in both 2,048-shot strata and a familywise-adjusted one-sided Wilson
upper bound below `0.005`, splitting alpha `0.05` across the two strata. The
evaluator also requires zero policy-versus-exact logical-failure discordances.

Secondary outputs are:

- Wilson 95% intervals for each arm's BLER;
- exact paired McNemar tables for exact tensor versus calibrated CMWPM, exact
  tensor versus MWPM, and policy versus calibrated CMWPM;
- class mismatch, fallback, and invalid-recovery counts;
- paired estimated arithmetic work for the policy, fixed column `chi=8`, and
  unrestricted tensor inference;
- 10,000 within-rate paired-bootstrap replicates for descriptive work-ratio
  intervals, using a separately frozen seed.

The matching comparisons are secondary and descriptive. A significant screen
result is not a threshold or state-of-the-art claim. Single-digit discordant or
failure counts are printed prominently and cannot support a superiority claim.

## Artifacts and provenance

The experiment produces immutable canonical JSON artifacts for calibration
shots, CMWPM selection, screen shots, and scored results. Each contains the
literal config, source/config hashes, dependency versions, and git commit. The
evaluator independently replays seeds, syndromes, decoder parameters, tensor
classes, recoveries, logical outcomes, aggregate counts, and work totals.

Canonical status additionally requires a clean committed tree, the exact frozen
configs, unseen screen domain, expected shot counts, and matching source hashes.
Each producer must consume committed input artifacts: generate calibration shots
then commit and push; run selection then commit and push; generate screen shots
then commit and push; evaluate then commit and push the result. Each stage begins
with a clean tree. Shot manifests freeze generator, evaluator, tensor, paired
metric, and artifact-writer sources before sampling. Calibration provenance,
config bytes/path/digest, committed input identities, dependency versions,
NetworkX/Blossom5 backend identity, lockfile digest, and any native Blossom5
binary digest are validated and propagated through selection to the screen.
Existing v1 tensor-confirmation domains are forbidden. Partial, reduced, or
tampered data are scored only as nonconfirmatory diagnostics.

## Failure handling

- A tied or invalid unrestricted reference invalidates the canonical run.
- A CMWPM exception or syndrome-invalid recovery counts as baseline failure and
  is reported; it does not silently remove a shot.
- A tensor-policy invalidity triggers its declared `chi=8` fallback. An invalid
  fallback invalidates canonical evaluation.
- Interrupted generation resumes only from complete, hash-verified shards;
  artifacts are never overwritten in place.
- Any methodological defect found after screen sampling retires the screen
  domain and requires a new versioned domain.

## Progression rule

Only a passing primary class/outcome gate justifies designing the larger
six-stratum confirmation or hardware schedule. A failure instead triggers an
error-case analysis of contraction direction, tolerance, and fallback; it does
not trigger post-hoc threshold tuning on the screen set. World models, RL,
GFlowNets, FNO, and HiPPO are outside this experiment.
