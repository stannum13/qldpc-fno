# Trajectory-balance logical-mass discovery

## Question

Can a small conditional GFlowNet amortize syndrome-conditioned sampling? The
target is not merely to generate valid physical errors. Its terminal distribution
must reproduce the exact posterior mass of each physical error and, after
aggregation, each logical class.

The experiment uses a unique four-step trajectory for the Steane code: three
stabilizer coefficients followed by one logical coefficient. Fixed action order
gives each of the 16 syndrome-compatible physical errors exactly one trajectory,
so the trajectory-balance objective does not learn a multiplicity artifact. The
unnormalized terminal reward is the exact independent-error likelihood.

## Discovery design

Five development fields and syndrome indices 0–5 supply every terminal reward
for training. Syndrome indices 6–7 and the two fields from the exact decision
gate are inspected discovery contexts. Three independently initialized models
are trained for 2,000 full-terminal steps. Three separate sampling-replicate
streams are reused across model seeds so model and Monte Carlo variation remain
identifiable.

This run is labelled `discovery_nonconfirmatory`. The inspected contexts cannot
become a confirmation set after model changes. Gate 4 remains explicitly
`unresolved_requires_fresh_paired_baselines`.

## Exact induced-distribution diagnostic

Because there are only 16 terminals, the forward policy's complete induced
distribution can be enumerated without sampling noise.

| Context split | Rows | Mean physical TV | Mean logical TV | Top-class accuracy |
|---|---:|---:|---:|---:|
| Training fit | 90 | 0.0932 | 0.0693 | 0.9556 |
| Unseen syndromes | 30 | 0.7192 | 0.2472 | 0.7333 |
| Unseen fields | 48 | 0.3533 | 0.1862 | 0.7292 |

The model does not fit all development distributions closely and generalizes
poorly, especially to syndrome bit patterns absent from training. This is not a
sampling-budget failure. On unseen fields, sampled mean logical TV is `0.1986`
at 32 terminals, `0.1862` at 128, `0.1851` at 512, and `0.1863` at 2,048. The
large-budget result converges toward the biased learned distribution rather than
the exact posterior.

## What this falsifies—and what it does not

This disconfirms the utility of the current generic MLP trajectory-balance model
as an amortized posterior sampler. It does not falsify GFlowNets in QEC. The
coefficient-to-error map contains parity interactions, and holding out entire
syndrome bit patterns is a hard combinatorial extrapolation problem. More width
or post-hoc training after inspecting these results would not turn this same run
into confirmation.

The result also explains why “valid and diverse samples” is an insufficient
metric. Every generated terminal is syndrome-valid by construction, yet its
logical mass can remain badly wrong. Exact induced-distribution TV catches the
failure before a large decoding campaign.

## Cheapest responsible next comparison

If this branch is revisited, model selection must use only development data. A
direct autoregressive maximum-likelihood student is the most useful control: it
has the same policy factorization but minimizes posterior cross-entropy instead
of trajectory balance. A symmetry-aware model can then expose affine parity
features explicitly. Only if one of those models fits development contexts and
passes a frozen validation rule should new spatial fields be generated from a
predeclared law for confirmation.

The confirmation must rerun rejection and affine Metropolis with multiple paired
sampler seeds. It must report both retained-terminal and complete-proposal axes.
Wall-clock or FPGA claims require timed implementations; this discovery artifact
records batched policy invocations and per-terminal action-logit evaluations but
does not convert them into latency.

## Reproduce

```bash
uv run python experiments/25_run_gflownet_mass.py \
  --config configs/gflownet_mass.json \
  --decision-config configs/prediction_to_decision.json \
  --baseline evidence/mass-sampling-baselines/mass_sampling.json \
  --out artifacts/gflownet-mass-discovery
uv run pytest -q tests/decision/test_gflownet.py \
  tests/decision/test_gflownet_study.py \
  tests/integration/test_gflownet_mass_cli.py
```

The tracked [discovery artifact](../evidence/gflownet-mass-discovery/gflownet_mass.json)
has SHA-256 digest
`3e8ca84d4b339014d0f2b137ae46f1a78fbb81e1c948ce4d08c6632f389cb465`.

## Claim boundary

This is a discovery-only, small-code, independent-Z experiment. Development
uses exhaustive terminal rewards, so it demonstrates an objective and
generalization diagnostic rather than a scalable training recipe. It provides no
qLDPC, circuit-level, threshold, hardware-throughput, or latency result. The
nonlearned baseline artifact has one stochastic realization per state and budget;
no claim of a statistically confirmed method difference is made.
