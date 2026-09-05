# Reduced causal forecasting screen

## Scientific question

Can recent syndrome history improve the per-qubit error probabilities supplied to a
conventional qLDPC decoder, and do Fourier spatial features and HiPPO temporal memory help
specifically when the simulated error channel has matching spatial and temporal structure?

This is a **causal channel-forecasting** experiment. At round `t`, a model sees syndromes only
from rounds before `t` and predicts the physical-error probabilities for round `t`. The current
syndrome and those probabilities then go to the same BP-LSD decoder in every arm. The neural
network does not emit a correction and does not replace BP-LSD. This distinction matters:
forecast NLL measures whether the probability model describes the channel, while logical
block-error rate (BLER) measures the downstream decoder outcome.

The screen used the `lp(3,7)_16` lifted-product code (`n = 2610`, `k = 744`, `ell = 45`) and an
independent-Z, perfect-syndrome code-capacity simulator. Its purpose was to decide whether this
exact observation-only formulation showed enough basic headroom to justify a larger run.

## Experimental design

The simulator generated five controlled regimes. They progressively introduce structure that a
spatial encoder, a temporal memory, or both could exploit:

| Regime | Channel structure | Intended control |
|---|---|---|
| `stationary_iid` | Constant error probability `0.0375` at every qubit and round | Neither spatial nor temporal structure |
| `static_spatial_latent` | A fixed low-frequency cosine field around the ring, plus per-channel offsets | Spatial structure without time variation |
| `temporal_uniform` | A clipped AR(1) global log-odds process, uniform around the ring | Temporal structure without spatial variation |
| `joint_in_basis` | AR(1) temporal variation plus a low-frequency cosine field and localized, exponentially decaying circular bursts | Joint structure aligned with the Fourier/long-memory inductive biases |
| `joint_basis_mismatch` | AR(1) temporal variation plus finite-width top-hat events that can move around the ring | Joint structure deliberately less aligned with a low-mode Fourier basis |

The learned comparison is a crossed 2 x 2 design:

| Name | Spatial encoder | Temporal memory |
|---|---|---|
| `cnn_fir` | Circular CNN | Finite impulse-response history |
| `cnn_hippo` | Circular CNN | HiPPO-LegS state |
| `fno_fir` | Ring Fourier neural operator | Finite impulse-response history |
| `fno_hippo` | Ring Fourier neural operator | HiPPO-LegS state |

The FNO retained 12 Fourier modes. FIR used 32 history steps, and HiPPO used order 16. Learned
models had 10,970-11,482 effective parameters for the CNN cells and 52,346-52,858 effective
parameters for the FNO cells. All arms fed per-round priors into BP-LSD configured with serial
minimum-sum BP (100 iterations maximum) and order-5 `LSD_E`.

Three observation-only reference models were also fitted: a stationary per-qubit field, an EWMA
circular logistic model, and a 32-lag circular logistic autoregression. Selection between EWMA and
the autoregression used validation NLL. The stationary field remained the prerequisite control:
the selected history-aware baseline had to improve NLL over it without worsening BLER before the
neural interaction question was worth scaling.

## Sample size and evidence boundary

The configuration generated exactly two train, two validation, and two calibration sequences per
regime: 30 sequences in total. Every sequence contained 8 burn-in rounds and 16 scored rounds, or
24 rounds total. Reported forecast and decoder metrics reuse the two validation sequences for
evaluation, giving only 32 scored round-decisions per regime and arm. There was no independent
test split (`test = 0`).

The result is explicitly labelled `reduced_non_scientific`. It contains no p-value, confidence
interval, or hypothesis verdict. BLER therefore moves in increments of `1/32 = 0.03125`, and the
numbers below are descriptive execution evidence, not estimates suitable for a scientific claim.

## Observation-only reference results

These are the exact validation NLL values used by the reference-model selection code. EWMA was
selected over logistic AR in all five regimes. The stationary field was not a selection candidate;
it was the prerequisite control.

| Regime | Stationary field | EWMA | Logistic AR | Selected history-aware model |
|---|---:|---:|---:|---|
| `stationary_iid` | 0.15830474723986407 | 0.15851480143234334 | 0.33966103604265700 | EWMA |
| `static_spatial_latent` | 0.17706132023919421 | 0.17767969716294680 | 0.40430975118290420 | EWMA |
| `temporal_uniform` | 0.14182429943623520 | 0.14201413078425480 | 0.29006434932058230 | EWMA |
| `joint_in_basis` | 0.21158301078447223 | 0.21230546259229519 | 0.50158170689892760 | EWMA |
| `joint_basis_mismatch` | 0.15914934754219426 | 0.15941905277787560 | 0.34718614298538064 | EWMA |

The decisive prerequisite comparison was recomputed on the scored validation rounds as
`stationary NLL - selected NLL`, with BLER difference defined as
`selected BLER - stationary BLER`:

| Regime | Selected model | NLL improvement | BLER difference | Prerequisite passed |
|---|---|---:|---:|---|
| `static_spatial_latent` | EWMA | -0.0006056672841143196 | -0.03125 | No |
| `temporal_uniform` | EWMA | -0.0002223820985045033 | 0.06250 | No |
| `joint_in_basis` | EWMA | -0.0005086500895234436 | -0.03125 | No |
| `joint_basis_mismatch` | EWMA | -0.00033043625613440875 | 0.03125 | No |

Every NLL improvement was negative: the chosen history-aware reference was slightly worse than a
stationary per-qubit field. Consequently, none of the four nonstationary regimes met the declared
observation-only headroom prerequisite.

## Learned-arm results

The table reports exact overall forecast NLL and downstream BP-LSD BLER on the same two validation
sequences. Each BLER is based on 32 scored rounds.

| Regime | CNN + FIR NLL / BLER | CNN + HiPPO NLL / BLER | FNO + FIR NLL / BLER | FNO + HiPPO NLL / BLER |
|---|---:|---:|---:|---:|
| `stationary_iid` | 0.15837965729220710 / 0.18750 | 0.15828730711692274 / 0.15625 | 0.15889569874899623 / 0.25000 | 0.15832635659225547 / 0.18750 |
| `static_spatial_latent` | 0.17720346095204714 / 0.50000 | 0.17708369664829776 / 0.40625 | 0.17786052109809478 / 0.43750 | 0.17713020651703698 / 0.46875 |
| `temporal_uniform` | 0.14508740029203000 / 0.06250 | 0.14495197111140415 / 0.09375 | 0.14541776358352698 / 0.09375 | 0.14503939582140310 / 0.12500 |
| `joint_in_basis` | 0.21640510812042477 / 0.78125 | 0.21631892317544720 / 0.81250 | 0.21701685462483897 / 0.78125 | 0.21641651258847694 / 0.84375 |
| `joint_basis_mismatch` | 0.15743677392010746 / 0.18750 | 0.15739637439777737 / 0.18750 | 0.15797029258271494 / 0.18750 | 0.15743427218941564 / 0.18750 |

The per-sequence BLER interaction was defined as:

```text
(CNN+FIR - FNO+FIR) - (CNN+HiPPO - FNO+HiPPO)
```

For `joint_in_basis`, the two sequence values were `[0.0, 0.0625]`, with mean `0.03125`.
For `joint_basis_mismatch`, they were `[0.0, 0.0]`, with mean `0.0`. The declared supporting
direction was negative, so neither descriptive result had that direction. With two reused
validation sequences and no uncertainty estimate, this interaction is a diagnostic only.

## Correctness and integrity checks

Several checks constrain what can be inferred from the run:

- All four model families passed a deterministic tiny-data overfit test. Final NLL / bit accuracy
  was `0.0029237980488687754 / 1.0` for CNN+FIR,
  `0.014405708760023117 / 0.9994612336158752` for CNN+HiPPO,
  `0.014173733070492744 / 0.9994612336158752` for FNO+FIR, and
  `0.006638688966631889 / 1.0` for FNO+HiPPO. This checks trainability, not generalization.
- Structural causal audits changed current/future syndromes, physical-error labels, logical
  labels, and privileged diagnostics after a forecast boundary. At forecast round 12, the
  prediction remained bit-identical under each forbidden-field mutation for all four learned
  cells in every regime.
- Every reported decoder arm reconstructed syndromes and logical labels from the same error arrays
  and shared the same `Hx`, `Hz`, logical basis, role membership, decoder policy, and baseline-fit
  policy hashes. Every correction was syndrome-valid.
- Train, validation, and calibration membership was identity-bound and disjoint. Predictor state,
  calibration, evaluation content, QEC geometry, and policies were SHA-256-bound in the evidence.
- The completion record marked 532 payloads complete and stored hashes for each payload. Independent
  replay verification exited successfully; its local log recorded 379.53 s elapsed time.

## Runtime

The primary screen execution recorded 366.95387629186735 s of wall time. The enclosing local run
recorded 726.31 s elapsed and a maximum resident-set measurement of 1,128,333,312 bytes; the replay
verification recorded 379.53 s elapsed and 1,229,553,664 bytes. These are engineering observations
from one local software environment. The timing artifact is explicitly marked
`engineering_measurement_no_speed_claim`: inference was batched, BP-LSD setup and decode were
measured in software, and no FPGA, ASIC, streaming-backlog, or deployment-latency claim follows.

## Interpretation and decision

This run answered its narrow decision question: the selected non-neural history-aware baseline,
EWMA, did not beat the stationary per-qubit field on forecast NLL in any nonstationary regime.
Under the declared decision rule, scaling this exact formulation to a larger causal comparison was
therefore not justified. The crossed CNN/FNO x FIR/HiPPO results remain useful diagnostics, but they
cannot rescue a formulation whose simpler temporal prerequisite failed.

This screen **does not establish the absence of an effect**. The sample contains only two
validation sequences per regime, reuses them for reported evaluation, has no test set, and supplies
no inferential uncertainty. A different observation model, stronger temporal reference, longer
sequences, more frequent or more identifiable transients, or a reformulation that predicts a
decoder-relevant residual could behave differently. The supported conclusion is only that more
compute on this exact setup was not warranted by its own prerequisite.

## Reproducing the machinery

The numeric evidence described above lives in ignored local artifacts and is **not available in a
fresh clone**. In this working copy it consists of the result payload, completion hashes, isolated
timings, and run/replay logs. The committed repository contains the configuration, generators,
models, evaluator, integrity checks, and command-line entry points needed to create a new result.

Generate a fresh sequence set in a new output directory:

```bash
uv run python experiments/19_generate_causal_sequences.py generate \
  --config configs/causal_fno_hippo_reduced.json \
  --out artifacts/causal-sequences-new
```

Verify it, including deterministic regeneration:

```bash
uv run python experiments/19_generate_causal_sequences.py verify \
  --config configs/causal_fno_hippo_reduced.json \
  --out artifacts/causal-sequences-new \
  --regenerate
```

Run the crossed comparison into another new directory, then replay-verify the persisted evidence:

```bash
uv run python experiments/20_run_causal_factor_screen.py run \
  --config configs/causal_fno_hippo_reduced.json \
  --sequences artifacts/causal-sequences-new \
  --out artifacts/causal-screen-new

uv run python experiments/20_run_causal_factor_screen.py verify \
  --config configs/causal_fno_hippo_reduced.json \
  --sequences artifacts/causal-sequences-new \
  --out artifacts/causal-screen-new
```

Relevant committed files:

- [Reduced configuration](../configs/causal_fno_hippo_reduced.json)
- [Sequence command](../experiments/19_generate_causal_sequences.py)
- [Comparison command](../experiments/20_run_causal_factor_screen.py)
- [Sequence generator](../src/qldpc_fno/temporal/generator.py)
- [Model factory](../src/qldpc_fno/models/causal_forecaster.py)
- [Training implementation](../src/qldpc_fno/training/causal_sequence.py)
- [Reference models](../src/qldpc_fno/temporal/baselines.py)
- [Evaluation implementation](../src/qldpc_fno/temporal/evaluation.py)
- [Artifact publication and replay verification](../src/qldpc_fno/temporal/screen.py)
- [End-to-end command tests](../tests/integration/test_causal_screen_cli.py)
