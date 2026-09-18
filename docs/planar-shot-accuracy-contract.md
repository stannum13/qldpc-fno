# Planar physical-shot accuracy contract

## Status

**No planar physical-shot accuracy result exists yet.** Neither the calibration
domain nor the held-out screen domain has been opened, and no artifact named
below exists. This document freezes the experiment that may create those
artifacts; it is not a results page.

The design is the [planar shot-accuracy screen design](superpowers/specs/2026-09-18-planar-shot-accuracy-screen-design.md).
It responds to the [symplectic syndrome erratum](symplectic-syndrome-erratum.md):
historical planar tensor data had the right syndrome marginal under symmetric
depolarizing noise, but did not generally pair an original sampled physical
error with its syndrome. Those historical results therefore do not establish
shot-level BLER. This new screen generates and replays correctly paired errors
and syndromes with `qecsim.paulitools.bsp(error, code.stabilizers.T)`.

## Frozen question and samples

On an unrotated qecsim `PlanarCode(5, 5)` under i.i.d. depolarizing
code-capacity noise, does the fixed two-view tolerance policy preserve the
unrestricted tensor decoder's per-shot logical decision? The screen also
describes how both tensor arms compare with matching decoders on those same
physical shots.

The two sample roles are intentionally separate:

| Role | Frozen domain | Rates | Shots per rate | Permitted use |
|---|---|---:|---:|---|
| Calibration | `qldpc-fno/planar-shot-accuracy/calibration/v1` | 0.10, 0.15 | 512 | Select one CMWPM configuration only. |
| Held-out screen | `qldpc-fno/planar-shot-accuracy/screen/v1` | 0.10, 0.15 | 2,048 | Evaluate the frozen policy and selected baseline. |

For each rate and shot index, the generator takes the first eight bytes of
`SHA256("{domain}|d5|p{error_rate:.6f}|i{shot_index:06d}")` as an unsigned
big-endian seed. The artifact stores that seed, the binary-symplectic physical
error, and its symplectic syndrome. Readers regenerate each one, require the
same syndrome, require unique seeds within an artifact, and require the two
roles' seeds to be disjoint. Calibration must finish and its selected baseline
must be frozen before the screen domain is opened.

The exact inputs are [calibration shots](../configs/planar_shot_calibration.json),
[screen shots](../configs/planar_shot_screen.json),
[CMWPM grid](../configs/planar_cmwpm_grid.json), and the
[accuracy policy](../configs/planar_shot_accuracy_policy.json). The source and
these exact config bytes must be committed and pushed before either domain is
sampled.

## Decoders and information boundary

The unrestricted tensor reference runs both row and column MPS contractions.
Their normalized four-class masses must have the same unique winning class,
probabilities within `1e-10`, pairwise log-mass ratios within `1e-8`, and a
winning margin greater than twice the row/column probability discrepancy. A
tied, invalid, or inconsistent reference invalidates canonical evaluation.

The fixed policy runs row and column contractions at tolerance `0.01`. It
accepts the column class only if both valid views choose that class; otherwise
it runs a column contraction at `chi=8`. The policy is charged for both
tolerance views on every shot and for any fallback. It is not retuned on
calibration or screen data. A fixed column `chi=8` tensor calculation is a
descriptive work comparator.

The strong classical arm is qecsim `PlanarCMWPMDecoder`, calibrated only on the
calibration role over all 48 combinations of:

- `factor`: 1, 2, 3, 4;
- `max_iterations`: 2, 4, 8;
- `box_shape`: `"t"`, `"r"`;
- `distance_algorithm`: 2, 4.

One configuration shared by both rates is selected by fewest pooled calibration
failures, then fewest failures at its worse rate, then lexicographic
`(factor, max_iterations, box_shape, distance_algorithm)`. It must be described
only as **calibrated CMWPM within the declared grid**, not as globally optimal
matching. Ordinary `PlanarMWPMDecoder` is also evaluated as a named
factorized-reference arm; it is not part of CMWPM selection. Neither matching
result controls the primary policy/reference gate.

## What counts as a decoding failure

Different correction bit strings do not need to be equal. Two corrections can
differ by a stabilizer: their raw strings change, but they have exactly the same
effect on the encoded information. Comparing strings would incorrectly call
that harmless difference an error.

Instead, for sampled error `e` and decoder recovery `r`, the evaluator forms
`residual = (e + r) mod 2`. It first requires the recovery to reproduce the
observed syndrome. It then computes the residual's commutation signature with
the code logical operators. With syndrome validity established, an all-zero
residual logical signature means the residual is a stabilizer and has no
logical effect; any nonzero entry means a logical failure.
An invalid recovery is both an integrity error and a counted failure. CMWPM or
MWPM exceptions remain explicit baseline failures. A failed policy fallback, or
an invalid exact reference, prevents canonical certification rather than
silently dropping a shot.

## Endpoints and gate

The independent statistical unit is one physical shot. The primary endpoint is
the policy's selected-class mismatch rate against the certified unrestricted
tensor reference, separately at `p=0.10` and `p=0.15`. A canonical pass requires
all of the following in each 2,048-shot stratum:

- zero class mismatches;
- a one-sided Wilson upper bound no greater than `0.005`, using alpha `0.025`
  (`0.05` familywise alpha split across the two rates);
- zero exact-policy logical-failure discordances;
- valid references and syndrome-valid recoveries for every arm.

The resulting canonical statuses are
`passed_exact_outcome_preservation`,
`falsified_exact_outcome_preservation`,
`unresolved_insufficient_precision`, `invalid_exact_reference`, or
`invalid_recovery`; reduced or otherwise noncanonical input is labelled
`unresolved_noncanonical_data`.

Secondary, descriptive outputs are Wilson 95% BLER intervals for exact tensor,
policy, CMWPM, and MWPM; exact paired McNemar summaries for exact versus policy,
exact versus CMWPM, exact versus MWPM, and policy versus CMWPM; fallback and
invalidity counts; and estimated arithmetic-work comparisons. Work compares
the policy with fixed `chi=8` and unrestricted tensor inference using 10,000
within-rate paired-bootstrap replicates and the frozen bootstrap seed. It is an
arithmetic estimate, not a timing measurement.

## Expected artifacts and commands

After the freeze is pushed, use fresh output directories in this order:

```bash
uv run python experiments/31_generate_planar_shots.py \
  --config configs/planar_shot_calibration.json \
  --out evidence/planar-shot-calibration

uv run python experiments/32_calibrate_planar_cmwpm.py \
  --grid configs/planar_cmwpm_grid.json \
  --data evidence/planar-shot-calibration/planar_shots.json \
  --out evidence/planar-cmwpm-calibration

uv run python experiments/31_generate_planar_shots.py \
  --config configs/planar_shot_screen.json \
  --out evidence/planar-shot-screen

uv run python experiments/33_run_planar_shot_accuracy.py \
  --policy configs/planar_shot_accuracy_policy.json \
  --screen evidence/planar-shot-screen/planar_shots.json \
  --selection evidence/planar-cmwpm-calibration/planar_cmwpm_selection.json \
  --out evidence/planar-shot-accuracy
```

The expected future immutable files are
[`planar-shot-calibration/planar_shots.json`](../evidence/planar-shot-calibration/planar_shots.json),
[`planar-cmwpm-calibration/planar_cmwpm_selection.json`](../evidence/planar-cmwpm-calibration/planar_cmwpm_selection.json),
[`planar-shot-screen/planar_shots.json`](../evidence/planar-shot-screen/planar_shots.json),
and
[`planar-shot-accuracy/planar_shot_accuracy.json`](../evidence/planar-shot-accuracy/planar_shot_accuracy.json).
They are currently absent. Producers refuse to overwrite an existing output;
the evaluator records input/config/source hashes, dependency versions, git
provenance, per-shot errors, syndromes, classes, recoveries, failures, and work,
then independently replays the complete result. Canonical status additionally
requires exact frozen configs, expected counts, matching current source hashes,
disjoint domains, and clean producer/evaluator provenance.

## Explicit nonclaims

Before results exist, this contract makes no accuracy, BLER, matching,
work-saving, or hardware claim. Even a passing screen would establish only the
specified distance-5, code-capacity, i.i.d.-depolarizing physical-shot endpoint
and exact-policy preservation. It would not establish calibrated posterior
masses, a globally optimal matching decoder, circuit-level-noise performance,
threshold behavior, qLDPC transfer, FPGA resource fit, latency, throughput,
backlog stability, learned-controller value, FNO/HiPPO/world-model value, or
state-of-the-art decoding.
