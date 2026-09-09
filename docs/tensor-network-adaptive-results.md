# Adaptive tensor-network discovery result

## Why this experiment exists

Maximum-likelihood decoding chooses the logical error class with the largest
*total* posterior probability. That sum over many degenerate physical errors is
a partition function. For planar surface codes it can be represented as a
two-dimensional tensor network and contracted approximately with a matrix
product state (MPS). The retained MPS bond dimension, `chi`, controls a real
accuracy-versus-work tradeoff.

The practical question is not merely whether a large `chi` is accurate. It is
whether every syndrome needs the same `chi` and the same contraction direction.
If easy syndromes can be identified before expensive work is performed, a
controller could spend high bond dimension only on the ambiguous tail.

This repository tests that prerequisite before building RL, a transformer, or a
renormalization policy.

## Reference method

The implementation pins
[qecsim 1.0b9](https://github.com/qecsim/qecsim), whose `PlanarMPSDecoder`
implements the planar tensor-network decoder based on Bravyi, Suchara, and
Vargo's [maximum-likelihood surface-code algorithm](https://arxiv.org/abs/1405.4883).
The audited upstream commit is
`24d6b8a320b292461b66b68fe4fba40c9ddc2257`.

Two independent paths establish the reference:

1. qecsim contracts the four logical cosets in `I, X, Y, Z` order.
2. On distance 3, our oracle separately enumerates every stabilizer
   representative in all four cosets: `4 * 2^12 = 16,384` physical errors per
   syndrome.

Across the frozen study, the largest probability difference between unrestricted
MPS contraction and independent enumeration is `5.93e-14`. Distance 5 also uses
unrestricted MPS contraction, which qecsim defines as exact up to floating-point
roundoff. No finite `chi` result is silently called exact.

## Frozen discovery experiment

The canonical configuration is
[`configs/tensor_network_reference.json`](../configs/tensor_network_reference.json).
It contains:

- unrotated planar surface codes at distances 3 and 5;
- code-capacity i.i.d. depolarizing noise at `p = 0.05, 0.10, 0.15`;
- 16 deterministic channel draws per distance/rate point, or 96 paired
  instances total;
- column, row, and averaged row/column contraction;
- `chi = 1, 2, 4, 8, 16`;
- one warm-up and three unbatched timing repetitions per action.

Every action sees the same syndrome and channel rate. The physical error that
generated the syndrome is never used by the contraction or action oracle.
Repeated syndromes are retained because they are repeated draws from the stated
channel distribution, not independent code instances invented after inspection.

An action is feasible only when it:

1. returns strictly positive finite mass for all four logical classes;
2. selects the same logical class as the unrestricted reference; and
3. has maximum pairwise log-mass-ratio error at most `0.05`.

The work comparison uses an explicit arithmetic estimate: NumPy's reported
einsum FLOP count plus dense-decomposition estimates of
`4*m*n*r + 8*r^3` for SVD and
`2*m*n*r - 2*r^3/3` for QR, where `r = min(m, n)`. This is a reproducible
algorithmic estimate, not an FPGA cycle count. Peak observed array elements and
raw host wall times are retained separately.

## Result

The cheapest single action feasible on all 96 instances is:

```text
column contraction, chi = 8
total estimated work = 251,952,384 FLOPs
mean estimated work  =   2,624,504 FLOPs / instance
```

The per-instance feasible-action oracle uses:

| Action | Instances |
|---|---:|
| columns, `chi=2` | 1 |
| rows, `chi=2` | 3 |
| columns, `chi=4` | 54 |
| rows, `chi=4` | 7 |
| average, `chi=4` | 1 |
| columns, `chi=8` | 30 |

Its total estimated work is `175,746,910` FLOPs, or `1,830,697` per instance.
That is a **30.25% reduction** from the best fixed feasible action under the same
logical-class and posterior-fidelity criterion.

This passes the preregistered discovery gate of at least 5% oracle savings and
more than one oracle action. It establishes *available adaptive value*. It does
not establish that a deployable policy can recover that value.

## Stress signals that matter

The experiment also found failure modes a controller must respect:

- Eight of 1,440 low-`chi` contractions produced a nonpositive logical-coset
  mass. They are recorded as invalid solver outcomes; no clipping or absolute
  value is applied.
- Total-variation error increased with larger `chi` in 14 of 288 ordered
  action sequences.
- Log-mass-ratio error was nonmonotonic in 6 of 288 sequences.

Thus a safe anytime rule cannot assume that one more unit of bond dimension
always improves the approximation. It needs a calibrated failure or regret
estimate and a conservative fallback.

## Sensitivity to the accuracy requirement

The `0.05` threshold was frozen before the canonical result. A diagnostic sweep
over the same raw contractions gives:

| Maximum log-ratio error | Best fixed action | Oracle work reduction |
|---:|---|---:|
| 0.001 | columns, `chi=16` | 65.36% |
| 0.01 | columns, `chi=8` | 8.76% |
| 0.05 | columns, `chi=8` | 30.25% |
| 0.10 | columns, `chi=8` | 47.17% |
| 0.50 | columns, `chi=8` | 69.91% |
| 1.00 | columns, `chi=8` | 85.17% |

Adaptive value remains above the 5% discovery threshold throughout this range.
The savings are not monotone because tightening the criterion can change both
the best fixed comparator and the oracle action set.

## What is and is not established

Established in this frozen discovery set:

- exact logical-coset agreement at distance 3;
- a paired, instance-dependent accuracy/work frontier at distances 3 and 5;
- a 30.25% oracle arithmetic-work opportunity at the primary criterion;
- real numerical failures and nonmonotonic approximation paths at low `chi`.

Not established:

- out-of-sample policy savings;
- circuit-level or noisy-syndrome decoding;
- qLDPC generalization;
- logical error-rate improvement over MWPM, BP-LSD, or another decoder;
- FPGA latency, throughput, memory fit, or streaming backlog stability;
- benefit from RL, attention, FNO, HiPPO, or a world model.

The next experiment is therefore a prediction test, not a larger architecture.
Using only information available before an action—syndrome geometry, channel
rate, code distance, and cheap contraction diagnostics—we will compare a fixed
cascade, a small calibrated classifier, and a shallow decision tree on untouched
instances. A world model or RL controller becomes justified only if sequential
actions add value beyond that contextual decision.

## Reproduce

```bash
uv sync --frozen
uv run python experiments/26_run_tensor_network_reference.py \
  --config configs/tensor_network_reference.json \
  --out evidence/tensor-network-reference
```

The complete 1,440-row artifact, including raw masses, references, work counters,
timings, invalid actions, oracle choices, and source/config hashes, is
[`evidence/tensor-network-reference/tensor_network_reference.json`](../evidence/tensor-network-reference/tensor_network_reference.json).
