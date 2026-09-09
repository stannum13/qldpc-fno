# Exact prediction-to-decision gate

## Question

Before training another decoder, this experiment asks which information and
which computation can change the logical decision. It separates three effects:

1. replacing a nominal uniform error rate with the correct global mean;
2. replacing it with the correct per-qubit spatial field; and
3. selecting the most probable logical class by summing its physical-error
   probabilities instead of selecting the class of one most probable error.

The calculation uses the Steane `[[7,1,3]]` CSS code under independent Z noise.
All 128 physical Z-error strings are enumerated. Every syndrome fiber is also
constructed from a unique four-bit affine coordinate: three independent Z
stabilizer coefficients and one logical-Z coefficient.

## Result

| True channel | Nominal uniform physical MAP | Correct spatial physical MAP | Correct spatial coset MAP |
|---|---:|---:|---:|
| Uniform, `p_i = 0.08` | 0.092043 | 0.092043 | 0.092043 |
| Frozen heterogeneous field | 0.298857 | 0.258952 | **0.255921** |

The entries are exact expected logical-failure probabilities under each true
channel, not Monte Carlo estimates. On the heterogeneous field, correct spatial
reliabilities reduce expected failure by `0.039905` with physical MAP. Exact
logical-coset aggregation contributes a further `0.003031`, for a total absolute
improvement of `0.042936` relative to nominal physical MAP. Supplying only the
correct global mean changes neither decisions nor risk in this experiment.

One syndrome exposes why aggregation matters. For syndrome `011`, the single
most probable error is in relative logical class 0, but the exact posterior mass
is `0.484079` for class 0 and `0.515921` for class 1. Physical MAP therefore has
conditional logical risk `0.515921`; coset MAP selects class 1 and has risk
`0.484079`. This syndrome occurs with probability `0.095178` under the frozen
heterogeneous channel.

## Architecture gates

- Probability-mass inference is open. Fourteen channel/syndrome states satisfy
  the frozen material-mass criterion, and coset aggregation changes one
  decision. Sampling or contraction methods therefore have a real quantity to
  approximate.
- Learned action selection is closed on this environment. With an equal mixture
  of the two declared channels and syndrome as the observable state, the fixed
  action `correct_spatial/coset_map` matches the per-syndrome oracle. Its oracle
  improvement over the best constant action is exactly zero within the frozen
  tolerance.
- Adaptive tensor contraction remains undecided. It requires a separate local
  surface-code experiment demonstrating an instance-dependent accuracy-cost
  frontier across contraction order and bond dimension.

This points to a narrower next experiment than “train an RL decoder.” First test
whether a GFlowNet or conventional sampler estimates logical-class mass more
accurately than rejection or Metropolis sampling at the same terminal-sample
budget. RL becomes relevant only after a larger sequential inference problem
exhibits nonzero policy headroom.

## Reproduce

```bash
uv sync --locked
uv run python experiments/23_run_prediction_to_decision.py \
  --config configs/prediction_to_decision.json \
  --out artifacts/prediction-to-decision-exact
uv run pytest -q tests/decision tests/integration/test_prediction_to_decision_cli.py
```

The tracked [action table](../evidence/prediction-to-decision-exact/action_table.json)
has SHA-256 digest
`803bac816111a808000333dd71ada0a0fca7ac8c9520d536c4af4a6511ea5811`.
It contains all 96 channel/syndrome/action rows, the assumed and true class
posteriors, selected errors, conditional risks, code and source hashes, and the
declared abstract operation-count model.

## Claim boundary

This is an exact small-code sensitivity calculation for independent Z noise. It
is not a sampled threshold result, a qLDPC result, a circuit-level decoding
result, evidence about Willow hardware, or a latency measurement. The spatial
field is supplied as an oracle; a deployable estimator has not yet recovered it
from syndrome history. Operation counts are deterministic abstract primitives,
not FLOPs or FPGA timing.
