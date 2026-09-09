# Logical-mass sampling baselines

## Why this experiment exists

The exact Steane-code gate found a syndrome for which the most probable physical
error belongs to the wrong maximum-posterior logical class. A useful generative
method must therefore estimate *probability mass*, not merely produce diverse
valid errors. This experiment establishes two nonlearned baselines before a
GFlowNet is trained.

- Conditional rejection draws independent physical errors from the true channel
  and keeps only those with the requested syndrome. Its retained samples are
  independent and exactly distributed, but it can discard substantial work.
- Affine Metropolis represents each valid error uniquely by three stabilizer
  coefficients and one logical coefficient. It proposes uniformly among all 15
  nonzero affine displacements. The proposal is symmetric and irreducible, so
  the likelihood-ratio acceptance rule has the exact conditional posterior as
  its stationary distribution.

The comparison covers both frozen channels and all eight syndromes: 16 exact
target distributions. Each stochastic row uses one deterministic, domain-separated
seed. The results are reproducible diagnostics, not uncertainty-resolved claims
about an average over random seeds.

## Equal retained samples

| Retained samples | Rejection mean TV | Metropolis mean TV | Rejection top-class accuracy | Metropolis top-class accuracy |
|---:|---:|---:|---:|---:|
| 32 | 0.04800 | 0.14716 | 0.9375 | 0.8750 |
| 128 | 0.02075 | 0.07036 | 1.0000 | 0.8750 |
| 512 | 0.01140 | 0.04203 | 1.0000 | 1.0000 |
| 2,048 | 0.00900 | 0.02646 | 1.0000 | 1.0000 |

Rejection is better on this axis because its retained samples are independent.
The axis is not a compute comparison. Producing 2,048 retained rejection samples
requires a mean of `23,935.8` physical draws and syndrome evaluations, while the
Metropolis row uses `4,352` transition proposals plus setup and target-weight
evaluations.

## Equal complete-configuration proposals

| Complete proposals | Rejection mean TV | Metropolis mean TV | Rejection coverage | Metropolis coverage |
|---:|---:|---:|---:|---:|
| 512 | 0.05146 | 0.07077 | 0.320 | 0.352 |
| 2,048 | 0.03215 | 0.03307 | 0.488 | 0.586 |
| 8,192 | 0.01355 | 0.01411 | 0.613 | 0.762 |
| 32,768 | 0.00981 | 0.00828 | 0.766 | 0.875 |

At equal complete-proposal counts the methods are much closer. Metropolis has
lower mean TV error at the largest frozen budget and explores more of the
16-state fiber. One seed per state is insufficient to claim that this rank is
stable. The result instead defines a useful confirmation target: a learned
sampler must be compared with both methods across repeated seeds and the whole
frontier.

## Work accounting

No scalar operation count is used. Every row separately records one-time
candidate materialization, setup syndrome evaluation, affine terminal
materialization, physical-channel draws, sampling-time syndrome evaluation,
target-log-weight evaluation, affine transition proposals, burn-in transitions,
and retained samples. The equal-proposal ladder matches completed stochastic
proposals but does not assert that a channel draw, a parity calculation, and a
neural-network step have equal hardware cost.

Logical effective sample size uses a Geyer initial-positive monotone paired
autocovariance estimate. A trace that never changes logical class has undefined
empirical ESS and is stored as `null`; summary tables report how many states had
an available ESS instead of silently dropping failures.

## Consequence for the next model

A trajectory-balance GFlowNet now has a precise task. Its terminal state is one
unique affine coefficient vector, its reward is the exact physical likelihood,
and samples are aggregated into the two logical classes. It advances only if it
improves total-variation error or decision accuracy over these baselines at
matched retained outputs and matched complete-terminal evaluations. Higher
diversity alone is not a passing result.

## Reproduce

```bash
uv run python experiments/24_run_mass_sampling.py \
  --decision-config configs/prediction_to_decision.json \
  --sampling-config configs/mass_sampling.json \
  --out artifacts/mass-sampling-baselines
uv run pytest -q tests/decision/test_sampling.py \
  tests/decision/test_sampling_study.py \
  tests/integration/test_mass_sampling_cli.py
```

The tracked [complete sampling artifact](../evidence/mass-sampling-baselines/mass_sampling.json)
has SHA-256 digest
`d2abebffd1fbc9cfa46139b6f32bc6e8370850ea13bc5d834ea37381dfab70d8`.

## Claim boundary

This is a small-code independent-Z sampling experiment with exact targets. It
does not test qLDPC scaling, circuit-level noise, learned generation, real-time
throughput, or FPGA latency. The stochastic summaries contain one seed for each
state and budget; apparent method gaps require repeated-seed confirmation.
