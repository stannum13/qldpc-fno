# Local-spectrum bond-allocation discovery

## Question

The confirmed two-view policy reduced estimated arithmetic work by deciding when
to accept `chi=4` and when to fall back to `chi=8`. qecsim exposes a simpler
control that should be tested before learning: retain a singular direction only
when its singular value, normalized by the largest local singular value, exceeds
a fixed tolerance.

The tolerance is fixed for a whole contraction, but the retained rank can vary
from bond to bond. It is therefore an adaptive bond-allocation rule, not a
resumable controller or an RL policy.

## Passive contraction trace

The work tracer now records each qecsim partial sweep and truncation. For every
local SVD it reports the full and retained ranks, spectral entropy, and the
fraction of squared singular weight discarded locally. It also records pre/post
bond dimensions, tensor elements, the normalization factor, and arithmetic-work
counters.

Tests require traced and untraced coset masses to be bit-identical. Sweep work
plus terminal inner-product work must exactly equal the aggregate arithmetic
estimate. The local discarded-weight fraction is a diagnostic; it is not a
bound on the final coset-mass or logical-decision error.

## Discovery sweep

The frozen config
[`configs/tensor_tolerance_discovery.json`](../configs/tensor_tolerance_discovery.json)
reuses the 96 syndromes from the earlier tensor-network discovery table. This is
deliberately a development-set analysis, not fresh confirmation.

Column and row contractions were evaluated at nine tolerances from `1e-8` to
`0.5`. The reference remains the same unrestricted contraction. Fixed column
`chi=8`, which had zero logical-class failures on this table, is the arithmetic
comparator.

## Results

Every tolerance action returned finite positive masses. Decision accuracy and
posterior fidelity separated sharply:

| Mode | Tolerance | Logical-class failures | Log-ratio fidelity failures | Work saving vs. fixed `chi=8` |
|---|---:|---:|---:|---:|
| columns | `1e-8` | 0/96 | 7/96 | 33.49% |
| columns | `1e-6` | 0/96 | 12/96 | 58.16% |
| columns | `1e-4` | 0/96 | 48/96 | 84.42% |
| columns | `1e-3` | 0/96 | 75/96 | 92.44% |
| columns | `1e-2` | 1/96 | 91/96 | 96.64% |
| rows | `1e-8` | 0/96 | 9/96 | 33.88% |
| rows | `1e-6` | 0/96 | 16/96 | 57.81% |
| rows | `1e-4` | 0/96 | 44/96 | 83.59% |
| rows | `1e-3` | 0/96 | 68/96 | 92.24% |
| rows | `1e-2` | 0/96 | 87/96 | 96.29% |

Fidelity failure means maximum pairwise log-coset-mass-ratio error greater than
the existing `0.05` criterion. A contraction can therefore make the correct
logical decision while badly misestimating losing-class probability mass. This
study supports decision-specific early acceptance, not calibrated soft output.

The most aggressive single action with zero observed decision failures is row
tolerance `0.01`, but selecting it after inspecting this table is post-hoc. A
more conservative candidate uses both row and column tolerance `0.01`, accepts
only agreement, and otherwise runs fixed column `chi=8`. On this discovery table
that candidate has zero failures, one fallback, and a descriptive 90.88% work
saving. If its column probabilities are returned on agreement, 90/96 outputs
still fail the log-ratio criterion. Those numbers select a class-only rule; they
do not confirm it or restore posterior fidelity.

## What happens next

The two-tolerance-view rule must be committed with its seed domain, safety
interval, and paired work test before new data are generated. A failure on that
new domain falsifies the frozen rule. Passing would establish a stronger simple
baseline for subsequent learned allocation.

No world model or RL claim opens here. qecsim's current API supplies deterministic
fixed-tolerance sweeps, not a state in which an intervention changes later
available actions. A separately audited resumable contraction engine is required
before action-conditioned transition prediction is meaningful.

The complete 1,728-row trace is
[`evidence/tensor-tolerance-discovery/tensor_tolerance_discovery.json`](../evidence/tensor-tolerance-discovery/tensor_tolerance_discovery.json).
Its SHA-256 is
`2b7cb43c41721947b1102936310ad7b424c357ddfb0f2e389e4c3ad8b9a7909b`.

## Reproduce

```bash
uv sync --frozen
uv run python experiments/29_run_tensor_tolerance_discovery.py \
  --config configs/tensor_tolerance_discovery.json \
  --reference evidence/tensor-network-reference/tensor_network_reference.json \
  --out evidence/tensor-tolerance-discovery
```
