# Locked confirmation of two-view tensor consensus

## Result in one sentence

A rule frozen before sampling selected the exact-reference logical class on all
960 independent planar-code cases while using 60.26% less estimated arithmetic
work than a fixed column contraction at `chi=8`; the preregistered one-sided 95%
lower confidence bound on the saving is 59.44%.

This is evidence for safe adaptive approximate inference in the tested
code-capacity setting. It is not an FPGA latency result, a circuit-level decoder
result, or a claim about qLDPC codes.

## The problem

Tensor-network maximum-likelihood decoding estimates the posterior mass of each
logical error class. Retaining a larger matrix-product-state bond dimension
usually costs more arithmetic but can protect the logical decision from
truncation error. A fixed large bond dimension pays that cost on every syndrome.

The discovery study showed that the cheapest sufficient contraction varied by
instance, but an oracle used the exact answer to choose it. This confirmation
study asks the deployable question: can a rule using only approximate solver
outputs save work on new syndromes without changing the logical decision?

## Frozen rule

The complete rule and statistical contract were committed in
[`configs/tensor_consensus_policy.json`](../configs/tensor_consensus_policy.json)
before the confirmation seed domain was sampled.

- At distance 3, run a column contraction at `chi=4`.
- At distance 5, run column and row contractions at `chi=4`.
- If both distance-5 contractions are valid and select the same logical class,
  accept their consensus.
- Otherwise, run the fixed fallback: column contraction at `chi=8`.

The comparator always runs the same column `chi=8` contraction. Agreement is a
stopping certificate supplied by two geometrically distinct approximations; it
does not inspect the high-accuracy reference.

## Untouched confirmation data

The separate confirmation seed domain is
`qldpc-fno/tensor-consensus-confirmation/v1`. It contains 160 independent
physical-channel draws in each of six strata:

| Distance | Physical error rates | Independent draws per rate |
|---:|---|---:|
| 3 | 0.05, 0.10, 0.15 | 160 |
| 5 | 0.05, 0.10, 0.15 | 160 |

Each of the 960 draws is paired with its transposed syndrome for an exact
equivariance audit. The transpose is not counted as another independent sample.
All four required actions are evaluated for both orientations, producing 1,920
contexts and 7,680 action outcomes.

Reference safeguards passed:

- zero invalid reference contexts;
- maximum unrestricted row/column probability discrepancy `1.44e-15`;
- maximum unrestricted row/column log-ratio discrepancy `8.88e-15`;
- maximum distance-3 enumeration probability discrepancy `5.93e-14`;
- maximum distance-3 enumeration log-ratio discrepancy `2.81e-13`;
- zero solver-validity or mass-feasibility transpose mismatches.

The data artifact records the exact config and source hashes. The evaluator also
regenerates every physical error, syndrome, transpose, feature vector, action
identity, and deterministic seed before scoring the policy.

## Accuracy gate

The independent unit is the original physical-channel draw. In every stratum,
both the adaptive policy and the fixed `chi=8` comparator had zero logical-class
failures out of 160 draws.

| Distance | Error rate | Policy failures | Fixed failures | Adjusted Wilson upper bound |
|---:|---:|---:|---:|---:|
| 3 | 0.05 | 0/160 | 0/160 | 0.04169 |
| 3 | 0.10 | 0/160 | 0/160 | 0.04169 |
| 3 | 0.15 | 0/160 | 0/160 | 0.04169 |
| 5 | 0.05 | 0/160 | 0/160 | 0.04169 |
| 5 | 0.10 | 0/160 | 0/160 | 0.04169 |
| 5 | 0.15 | 0/160 | 0/160 | 0.04169 |

The bound is a one-sided Wilson upper interval with familywise alpha 0.05 split
across 12 predeclared comparisons: policy and comparator in six strata. The
frozen safety ceiling was 0.05, so every stratum passes. Zero observed failures
does not mean zero true failure probability; the interval is the claim.

## Work gate

The work metric is the deterministic arithmetic estimate defined in the tensor
reference study. For the policy it includes every branch actually executed:
both `chi=4` views at distance 5 and the `chi=8` fallback when required.

| Quantity | Result |
|---|---:|
| Fixed `columns_chi8` work | 2,519,523,840 FLOPs |
| Two-view policy work | 1,001,381,232 FLOPs |
| Point saving | 60.255% |
| One-sided 95% lower bound | 59.436% |
| Distance-5 fallbacks | 4/480 |

The interval uses 10,000 paired bootstrap replicates, resampling independent base
draws within each distance/error-rate stratum with the seed frozen in the policy
config. Its estimand is an equal mixture of the six strata. The preregistered
minimum lower-bound saving was 5%, which is exceeded by a wide margin.

The policy and its transpose selected symmetry-related logical classes and used
identical estimated work for all 960 paired checks.

## What this changes

The result supports *consensus as an adaptive-compute primitive*. A second,
differently ordered approximation is much cheaper than immediately increasing
the bond dimension, and disagreement identifies a small ambiguous tail that can
be escalated conservatively.

It also changes the role of machine learning in this project. A learned policy
is no longer compared only with fixed `chi`. It must improve on this symbolic
rule—for example, by predicting when the row view can be skipped, allocating bond
dimension within a contraction, or reducing tail work—without weakening the
logical-class safety gate. A world model becomes relevant only after contraction
actions expose genuine sequential state transitions that this one-step rule
cannot represent.

## Claim boundary

Established here:

- locked, out-of-sample logical-class confirmation for the stated six strata;
- familywise-adjusted per-stratum unsafe-rate bounds below 5%;
- a paired lower confidence bound of 59.44% on estimated arithmetic-work saving;
- rare, explicitly counted fallback use; and
- exact transpose-consistency checks.

Not established:

- logical error rate under repeated noisy syndrome extraction;
- performance at distance greater than 5 or on a different code family;
- superiority to MWPM, BP-LSD, or another end-to-end decoder;
- wall-clock, FPGA, ASIC, throughput, or backlog improvement;
- posterior-mass fidelity on every accepted instance; or
- a benefit from FNO, HiPPO, attention, GFlowNet, RL, or a world model.

## Reproduce

The policy freeze is commit `9ff1fa2`. From that source state, generate the locked
data and evaluate it with:

```bash
uv sync --frozen
uv run python experiments/27_generate_tensor_policy_data.py \
  --config configs/tensor_consensus_confirmation_data.json \
  --out evidence/tensor-consensus-confirmation-data
uv run python experiments/28_run_tensor_consensus.py \
  --policy configs/tensor_consensus_policy.json \
  --data evidence/tensor-consensus-confirmation-data/tensor_policy_data.json \
  --out evidence/tensor-consensus-confirmation
```

The confirmation data SHA-256 is
`33bf0947113260dc676cf1aeb9b4df85484ef98fa3b0630eb978462fa1046cb5`.
The scored result SHA-256 is
`462a098506d91c571637dba2ad9a92090d43637a2e8e8ed731885428b35c5c5c`.
