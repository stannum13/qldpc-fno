# Locked confirmation: local-spectrum consensus

> **Syndrome pairing erratum:** under symmetric depolarizing noise, the stored
> syndromes have the intended marginal law but generally do not match the
> original seeded physical errors. This is a conditional logical-class and
> estimated-work result, not shot-level decoder BLER. See the
> [full erratum](symplectic-syndrome-erratum.md).

## Result

A two-view rule frozen before sampling selected the same logical class as the
unrestricted tensor-network reference on all 960 independent planar-code
syndromes. It used 89.02% less estimated arithmetic work than fixed column
`chi=8`; the preregistered one-sided 95% paired-bootstrap lower bound on the
saving is 87.46%.

This is a *logical-decision* result under code-capacity i.i.d. depolarizing
noise, not a calibrated-posterior, end-to-end decoder-error-rate, or FPGA-latency
result. The selected posterior masses fail the existing strict pairwise
log-ratio criterion on 868 of 960 cases. That divergence is part of the result,
not an omitted ablation.

## Why the rule is interesting

Maximum-likelihood decoding compares the *total probability mass* of four
logical error classes. Contracting the corresponding planar tensor network
with a large matrix-product-state bond dimension can be expensive. A fixed
rank cap treats every syndrome and every local bond as equally hard.

The earlier `chi=4` consensus confirmation showed that a second contraction
direction can detect ambiguity more cheaply than always increasing rank. A
subsequent development sweep found another simple lever: qecsim's normalized
singular-value tolerance retains different ranks at different bonds, even
though the tolerance itself is fixed for the whole sweep. This experiment
tests the combined conservative rule on a completely separate seed domain.

## Frozen experiment

The policy and evaluator were committed at `78166f8` on 15 September 2026 at
16:24 IST, before any samples from the confirmation domain were generated. The
policy is in
[`configs/tensor_tolerance_consensus_policy.json`](../configs/tensor_tolerance_consensus_policy.json):

1. Run column and row contractions at tolerance `0.01` on every syndrome.
2. If both return valid positive masses and agree on the winning logical class,
   accept the column result.
3. Otherwise, run and return column `chi=8`.

The fixed comparator runs column `chi=8` for every syndrome. Both tolerance
views and any fallback are charged to the policy's work. There is no learned
controller, early terminal result from a partial tensor, or state-dependent
change of tolerance within a sweep.

The untouched seed domain contains 160 independent syndromes with the
depolarizing-channel marginal for
each distance/error-rate point: distances 3 and 5 crossed with `p=0.05`,
`0.10`, and `0.15`. Each base syndrome has a transposed companion for symmetry
auditing, not a second independent statistical unit. The resulting artifact
contains 960 base draws, 1,920 orientations, and 7,680 action outcomes.

The historical scorer replays deterministic seeds and the original generator's
syndrome convention, independently recomputes
the row/column unrestricted reference and every action's probabilities and
primitive work counters, checks the declared reference tolerances, and
exhaustively enumerates stabilizer cosets for every distance-3 context. All
reference contexts are valid. Maximum distance-3 enumeration log-ratio
discrepancy is `2.85e-13`; maximum unrestricted row/column discrepancy is
`8.88e-15`. Transpose audits found no solver-validity, selected-class, or
estimated-work mismatch.

## Safety and work gates

Both policy and fixed comparator had zero exact-reference logical-class
failures in each of six strata:

| Distance | Error rate | Policy failures | Fixed failures | Adjusted one-sided Wilson upper |
|---:|---:|---:|---:|---:|
| 3 | 0.05 | 0/160 | 0/160 | 4.169% |
| 3 | 0.10 | 0/160 | 0/160 | 4.169% |
| 3 | 0.15 | 0/160 | 0/160 | 4.169% |
| 5 | 0.05 | 0/160 | 0/160 | 4.169% |
| 5 | 0.10 | 0/160 | 0/160 | 4.169% |
| 5 | 0.15 | 0/160 | 0/160 | 4.169% |

Familywise alpha `0.05` is split across 12 predeclared comparisons. Every
one-sided Wilson upper bound is below the frozen 5% unsafe-rate ceiling. Zero
observed failures is *not* proof of zero true failure probability; the adjusted
interval is the supported statement. One failure in any stratum would have
failed this contract.

| Work measure | Result |
|---|---:|
| Fixed column `chi=8` | 2,519,523,840 estimated FLOPs |
| Two-view tolerance policy, including fallbacks | 276,663,487 estimated FLOPs |
| Observed saving | 89.019% |
| One-sided 95% paired-bootstrap lower bound | 87.465% |
| `chi=8` fallbacks | 22/960 |

The work interval resamples base draws independently within each of six strata
for 10,000 replicates with the seed frozen in the policy config. Its minimum
lower-bound saving gate was 5%. These are reproducible arithmetic estimates,
including full local SVD costs before truncation, not measured cycles or host
timings. Stratum-specific observed savings range from 32.03% at distance 3,
`p=0.15`, to 94.15% at distance 5, `p=0.05`; the pooled saving is influenced by
the much larger fixed work at distance 5.

## The accuracy/fidelity split

The two tolerance views disagreed on 22 cases, which were escalated. On all
938 agreements, the shared winning class matched the unrestricted reference.
On the same fresh data, column tolerance alone made 12 class mistakes and row
tolerance alone made 10; the disagreements exposed those errors. The fixed
fallback made no class mistakes.

Yet 868 final outputs were *not* faithful posterior-mass approximations under
the `0.05` maximum pairwise log-ratio criterion. Even fixed `chi=8` had one
strict-fidelity failure on this larger sample. The policy should therefore be
used, if at all, to select a logical class with a conservative fallback. It
cannot be used as a calibrated soft posterior without a separate fidelity gate
or higher-accuracy mass computation.

## Claim boundary and next test

This confirmation establishes a code-capacity, distance-3/5, exact-reference
logical-class and arithmetic-work result for one unrotated planar code instance
per distance and the stated rates. It does not establish actual physical-shot
logical block-error rate against MWPM, repeated noisy-syndrome performance,
threshold scaling, transfer to qLDPC codes, FPGA memory fit, single-shot
latency, throughput, or backlog stability.

The next accuracy test must count logical failure modulo the stabilizer group
against sampled physical errors, with a strong matching baseline on the same
shots. Only after that should the arithmetic saving be mapped to a target
hardware schedule. For any learned adaptive bond controller, this symbolic
two-view rule—not fixed `chi=8` alone—is now the relevant compute/accuracy
baseline. A world model or RL policy remains unjustified until a separately
audited resumable contraction engine exposes real action-conditioned state
transitions.

## Evidence and reproduction

The complete fresh data are
[`evidence/tensor-tolerance-confirmation-data/tensor_policy_data.json`](../evidence/tensor-tolerance-confirmation-data/tensor_policy_data.json)
(SHA-256 `7f7113a56ca50ab2453a0a2743d398b2f58908dfbf8259a3d6786126ad3eae43`).
The scored decisions and statistics are
[`evidence/tensor-tolerance-confirmation/tensor_tolerance_consensus.json`](../evidence/tensor-tolerance-confirmation/tensor_tolerance_consensus.json)
(SHA-256 `7a84b5431f9326e491c3bf5ddfb4185405353523959872bea8a992e7c3d29359`).

To replay the **historical scientific result**, use the pre-sampling freeze
commit `78166f8`, not current HEAD. Current generation uses corrected
symplectic syndrome extraction; the revealed v1 domain cannot be certified
again under new source code. The JSON embeds absolute source paths, so a
checkout at another path can reproduce numerical results without reproducing
these byte-identical hashes.

```bash
uv sync --frozen
uv run python experiments/27_generate_tensor_policy_data.py \
  --config configs/tensor_tolerance_confirmation_data.json \
  --out evidence/tensor-tolerance-confirmation-data
uv run python experiments/30_run_tensor_tolerance_consensus.py \
  --policy configs/tensor_tolerance_consensus_policy.json \
  --data evidence/tensor-tolerance-confirmation-data/tensor_policy_data.json \
  --out evidence/tensor-tolerance-confirmation
```

Use fresh output directories for reruns; these scripts refuse to overwrite an
existing artifact.
