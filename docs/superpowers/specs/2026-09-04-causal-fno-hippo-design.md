# Causal FNO-HiPPO Channel Estimation Design

## Scope and claim boundary

The completed fixed-shot experiment showed that always-on FNO-derived priors
harm BP-LSD under stationary independent-Z noise. On 2,048 held-out shots for
`lp(3,7)_16` at `p=0.0375`, uniform BP-LSD failed 387 times, while the FNO
soft-prior and residual-repair arms failed 835 and 819 times.

This design moves learning to a task where the decoder lacks information:
forecasting a changing channel from syndrome history strictly before the round
being decoded.

The research question is:

> Does a Fourier spatial encoder and a HiPPO temporal memory have a measurable
> interaction when forecasting causally recoverable spatiotemporal noise, and
> does that interaction improve a fixed BP-LSD decoder?

The work has two separately reviewed boundaries:

1. **Mechanism discovery:** exact generator, causal data path, faithful HiPPO
   primitives, crossed architecture screen, and power calculation. This stage
   cannot emit a positive decoder claim.
2. **Frozen confirmation:** an immutable configuration fixes the surviving
   arms, independent-sequence count, estimands, seeds, and tests before test
   sequences are generated.

Real hardware, circuit-level detector events, Willow, FPGA synthesis, attention,
code-size transfer, and multi-pass candidate portfolios remain separate phases.

## Hypotheses

### H1: useful causal channel information exists

At least one observation-only causal baseline—EWMA or logistic AR—improves
BP-LSD over a stationary train-marginal prior. If both fail, the syndrome history
has not demonstrated recoverable channel information and neural comparisons
have no meaningful target.

### H2: causal learning improves the decoder

On the frozen joint spatiotemporal regime, the selected FNO-HiPPO model has lower
sequence-averaged block-logical error than the strongest surviving non-neural
causal baseline on identical sequences.

### H3: spatial-temporal complementarity exists

In the crossed architecture family, replacing a circular CNN with an FNO has a
larger benefit when paired with HiPPO than when paired with fixed finite memory.
The predeclared interaction contrast for a per-sequence loss `L` is:

```text
I = [L(CNN, FIR) - L(FNO, FIR)]
  - [L(CNN, HiPPO) - L(FNO, HiPPO)]
```

The sign convention is fixed before confirmation. Pairwise wins alone cannot
establish the interaction. H3 is the one-sided alternative `I<0`; its estimator
is the mean of the per-sequence interaction contrasts.

### H4: the effect is not only an in-basis construction

Any complementarity supported on the smooth Fourier/polynomial generator must
also retain the same numerical direction on a frozen basis-mismatch regime. H4
is a descriptive scope check in this phase, not a confirmatory test. A result
supported only in-basis is reported only as an in-basis mechanism demonstration;
a generalization claim requires a separately powered mismatch replication.

### H5: stationary-noise safety

Two stationary checks answer different questions and are not conflated:

1. a model fitted only on `stationary_iid` is a negative control for adaptive
   headroom and is expected not to beat stationary BP-LSD; and
2. the frozen checkpoint selected on `joint_in_basis`, including its unchanged
   calibration, is deployed without refitting on
   stationary sequences. H5 tests this deployment-shift arm for noninferiority
   to stationary BP-LSD with an absolute block-error margin of `0.005`.

Failure of the second check rejects deployment even if the joint-regime result
is positive.

## What this phase is—and is not

Every round is a code-capacity Z-error sample conditioned on a latent channel.
Syndrome measurements are perfect, and rounds do not share a Pauli frame. This
is repeated causal channel estimation over code-capacity samples, not a
circuit-level fault-tolerant memory experiment. The term `syndrome history` is
used throughout; `detector history` is reserved for later noisy-measurement work.

Synthetic training may use sampled physical errors as labels. Hardware does not
reveal those labels; a later hardware phase would require simulator pretraining,
weak labels, syndrome likelihood, or latent-variable fitting.

## Causal protocol

For round `t`, every deployable estimator consumes only public static metadata
and syndromes `s_0,...,s_(t-1)`, then emits per-qubit probabilities `q_t`.
BP-LSD receives `q_t` and current syndrome `s_t`.

```text
past syndromes -> channel forecaster -> q_t
                                         |
current syndrome s_t ----------------> BP-LSD -> correction c_t
```

The forecaster cannot receive the current error, syndrome, observable flip,
decoder outcome, latency, convergence state, or any future value. Round zero
uses the stationary probability estimated from training sequences. A causality
test mutates every forbidden current/future field while holding history fixed
and requires bit-identical `q_t`.

## Code geometry and representation

All regimes use the existing `lp(3,7)_16` code with `ell=45`. Flat vectors are
reshaped channel-major onto the exact lifted-ring ordering already verified by
the repository. Cyclic shift equivariance is tested for the FNO and circular-CNN
encoders. A complex FFT is treated as an inductive bias over `Z_ell`, not as an
algebraic diagonalization theorem over `F_2`.

## Frozen discovery generator

The discovery generator is configured by a strict JSON schema. All probabilities
are computed in log-odds and clipped to `[1e-5, 0.25]`. For qubit channel `c`,
ring coordinate `x`, sequence `j`, and round `t`:

```text
q[j,t,c,x] = sigmoid(logit(p0) + global[j,t]
                     + spatial[j,t,x] + channel_offset[j,c])
p0         = 0.0375
```

`channel_offset` is identically zero in `stationary_iid` and
`temporal_uniform`. In `static_spatial_latent`, `joint_in_basis`, and
`joint_basis_mismatch`, draw one `Normal(0,0.10^2)` value per qubit channel and
subtract the exact across-channel mean, once per sequence. Every baseline
estimates its stationary scalar or field from train only. Therefore a different
realized marginal cannot unfairly advantage a learned arm. Generator tests
require zero spatial variance in regimes A/C and nonzero spatial variance in
B/D/E before Bernoulli sampling.

Latent-state randomness and Bernoulli shot noise use separate seeds derived from
`(campaign_seed, regime, role, sequence_index, stream)`. Errors are sampled
independently conditional on `q`; syndromes and logical flips are computed from
the same committed parity checks and logical operators for every arm.

### Regime A: `stationary_iid`

`global=0` and `spatial=0` for every round.

### Regime B: `static_spatial_latent`

The field is constant within a sequence but unknown to the estimator:

```text
spatial[x] = 0.80 * cos(2*pi*k*x/ell + phase)
k          ~ Uniform{1,2,3}
phase      ~ Uniform[0,2*pi)
```

Temporal averaging can legitimately help infer this static latent field; this is
not described as a memory-free task.

### Regime C: `temporal_uniform`

`spatial=0` and:

```text
global[t+1] = clip(0.97*global[t] + epsilon[t], -1.20, 1.20)
epsilon     ~ Normal(0, 0.08^2)
global[0]   = 0
```

### Regime D: `joint_in_basis`

The global term follows Regime C. The spatial term contains a persistent low
mode plus transient hot regions:

```text
base[x]       = 0.45*cos(2*pi*k*x/ell + phase), k ~ Uniform{1,2,3}
burst_start   ~ Bernoulli(0.02) when no burst is active
burst_center  ~ Uniform{0,...,ell-1}
amplitude(age)= 1.20*exp(-age/32)
burst_profile = amplitude(age)*exp(4*(cos(2*pi*(x-center)/ell)-1))
spatial       = base + burst_profile
```

The exact round transition is: inspect the state carried from the previous
round; if inactive, draw the start Bernoulli and, on success, create an event
with `age=0` and its sampled center; emit the field for that age; record the
diagnostic labels; increment age; then terminate the event for the *next* round
when the emitted age was `127` or its amplitude was below `0.02`. Thus the onset
round emits age zero and exactly one termination rule is applied after emission.
Overlapping bursts are disabled. A golden fixture forces a start and records the
onset, final-active, and first-inactive rounds. The onset, center, age, and
termination are simulator-only labels and never estimator inputs.

### Regime E: `joint_basis_mismatch`

The global AR process is unchanged. Spatial events are sharp cyclic intervals,
not low Fourier modes:

```text
start probability = 0.02 when inactive
duration          ~ Uniform{8,...,96}
width             ~ Uniform{3,...,9} ring sites
amplitude          = 1.20 until an abrupt end
center[t+1]        = center[t] + step mod ell
step              ~ Uniform{-1,0,1}
```

Event rates are reported for every split. Generator parameters never change in
response to model performance.

Regime E uses the same inspect/start/emit/record/advance/terminate ordering.
The onset round emits at the sampled center and counts as duration round one.
After each active emission, the center advances by the sampled step for the
next round and the remaining duration decreases by one. The event is inactive
on the round after exactly `duration` emissions. A golden fixture covers onset,
movement, the final active round, and the first inactive round.

## Discovery data sizes

Each sequence contains a 32-round burn-in followed by 256 scored rounds.
Discovery uses:

- train: 32 independent sequences per regime;
- validation: 16 independent sequences per regime;
- calibration: 16 independent sequences per regime;
- test: none.

Reduced integration fixtures use two sequences, eight burn-in rounds, and 16
scored rounds and are labelled `reduced_non_scientific`.

The confirmation sequence count is not chosen from test outcomes. For every
confirmatory contrast—H1, H2, H3, and H5—compute the standard deviation of its
validation-sequence difference and its one-sided 95% chi-square upper confidence
bound `sd_upper`. Then compute:

```text
n_contrast = ceil(((z_0.975 + z_0.80) * sd_upper / 0.005)^2)
n          = max(64, n_H1_EWMA, n_H1_logistic, n_H2, n_H3, n_H5)
```

If `n>512`, confirmation is declared infeasible under the current budget; it is
not silently capped and cannot emit a claim. The resulting integer, every
variance and bound, validation artifact hash, and formula are committed in the
confirmation config. Confirmation test sequences are generated only after an
adversarial review approves that commit.

## Information-symmetric model matrix

All fitted arms use the same training-sequence membership, causal history, output
parameterization, validation budget, and calibration role. Calibration is never
used for fitting representation weights.

One separate model is trained per regime; no regime-specific checkpoint is
reused as evidence for another regime. Allowed estimator inputs are earlier
syndromes, check/qubit channel-type one-hot metadata, cyclic ring coordinates,
and the public completed-round index. Sequence IDs, seeds, latent parameters,
spatial frequency/phase, global state, event state, and simulator diagnostic
labels are forbidden features.

### Non-neural baselines

1. **Stationary scalar:** train-set marginal probability.
2. **Stationary field:** for physical qubit `i`, let `empirical_q_i` be its
   train-set error frequency and `scalar_q` the train-set marginal across all
   qubits. Use `q_i=lambda*empirical_q_i+(1-lambda)*scalar_q`, selecting
   `lambda` from `{0,0.25,0.5,0.75,1}` by validation NLL and breaking ties
   toward the smaller value. Fit one scalar temperature on calibration, then
   clip to `[1e-5,0.25]`.
3. **EWMA:** maintain one causal exponentially weighted syndrome field of shape
   `(21,45)`. Select decay from `{0.5,0.8,0.9,0.97,0.99}` on validation. A
   circular logistic `Conv1d(21,58,kernel_size=5)` with per-output intercept maps
   that field to 2,610 probabilities. Fit its weights on train only.
4. **Observation-only logistic AR:** stack the previous 32 syndrome fields into
   `(672,45)` and apply a circular logistic
   `Conv1d(672,58,kernel_size=3)` with per-output intercept. Missing prehistory is
   zero padded and accompanied by no extra mask. For both logistic mappings,
   select L2 from `{1e-4,1e-3,1e-2,1e-1}` on validation, use deterministic
   full-batch L-BFGS with at most 500 iterations and gradient tolerance `1e-8`,
   fit on train only, then fit a scalar temperature on calibration and clip final
   probabilities to `[1e-5,0.25]`. These baselines establish whether variation is
   recoverable without a neural spatial-temporal architecture.
5. **Privileged latent oracle:** true `q_t`; reported only as an unattainable
   channel-information ceiling.

A known-dynamics approximate Bayesian filter is deliberately deferred until a
Gate-2 signal exists. Its observation likelihood and state parameterization must
receive a separate review before it can replace logistic AR as the strongest
non-neural confirmation baseline.

### Crossed learned family

The causal spatial encoder is crossed with the causal temporal operator:

| Spatial encoder | Temporal operator |
| --- | --- |
| circular CNN | learned 32-tap finite impulse response (`FIR`) |
| FNO | learned 32-tap `FIR` |
| circular CNN | HiPPO-LegS |
| FNO | HiPPO-LegS |

Every input round has shape `(B,21,45)`. A single factory swaps only
`spatial_operator` and `temporal_operator`:

- the circular CNN applies `Conv1d(21,32,kernel_size=5,padding_mode="circular")`,
  GELU, then `Conv1d(32,32,kernel_size=5,padding_mode="circular")` and GELU;
- the FNO lifts `21 -> 32`, applies two residual spectral blocks retaining the
  first 12 rFFT modes, and returns a real physical-ring tensor `(B,32,45)`;
- the FIR keeps exactly 32 spatial embeddings and applies one learned scalar tap
  per hidden channel and lag, shared across ring positions, producing
  `(B,32,45)`;
- HiPPO keeps a state `(B,32,45,16)`, applies the same fixed LegS `Abar/Bbar`
  independently at every hidden-channel/ring site, and uses learned coefficients
  `C[hidden,order]` shared across ring positions to produce `(B,32,45)`;
- a shared pointwise `Conv1d(32,58,kernel_size=1)` produces logits
  `(B,58,45)`, flattened channel-major to 2,610 probabilities.

All cells use hidden width 32 and GELU in their spatial path. Equal-width is the
primary comparison and exact parameter counts are reported. A secondary
approximately equal-parameter comparison may adjust the shared hidden width;
no extra layer is introduced solely to equalize counts, and parameter count is
never treated as proof of equal capacity.

Two controls are required after the crossed screen:

- FNO-GRU and CNN-GRU. At each ring site, a single GRU shared across positions
  maps the 32-channel spatial embedding into a 16-dimensional state initialized
  to zero at each sequence boundary; a shared linear projection maps state 16
  back to 32 channels before the common pointwise readout. The GRU predicts from
  its previous state and only then incorporates the completed current round;
- an FIR-only prefix-shuffled-history diagnostic. At prediction `t`, indices in
  `[max(0,t-32),t)` are deterministically permuted using a hash of the public
  campaign seed, regime, role, sequence index, and `t`. No current or future
  index can enter the permutation, and the causal index audit is unchanged. It
  is not applied to full-history recurrent states and is not a confirmatory arm.

### HiPPO contract

The primary measure is time-varying HiPPO-LegS. For order `N`, the continuous
generator and input vector are:

```text
G[n,k] = -sqrt(2n+1)*sqrt(2k+1) when n > k
G[n,n] = -(n + 1)
G[n,k] = 0                       when n < k
B[n]   = sqrt(2n+1)
```

For completed sample index `r=1,2,...`, use the official bilinear update:

```text
G_r       = G / r
B_r       = B / r
Abar_r    = solve(I - G_r/2, I + G_r/2)
Bbar_r    = solve(I - G_r/2, B_r)
h_r       = Abar_r @ h_(r-1) + Bbar_r * u_r
h_0       = 0
```

This pins the sign, normalization, sample timing, initialization, and
discretization to the original implementation's `HiPPO_LegS` bilinear path.
Golden recurrence vectors are generated independently in float64 from these
equations and committed; the production recurrence must match them step by step.
A separate numerical diagnostic compares the finite-order reconstructed memory
to direct high-precision shifted-Legendre projections on smooth test functions.
That approximation check has convergence tolerances rather than requiring exact
coefficient equality. The implementation cannot be called faithful LegS until
both tests pass and states and gradients remain finite for orders `8`, `16`, and
`32` over 4,096 steps.

HiPPO-LegT with bilinear discretization and `theta in {8,32,128}` is a possible
Gate-3 secondary ablation, deferred until its equations, sampling convention,
and independent golden fixtures receive a separate design review. It is not a
Gate-1 or Gate-2 implementation requirement. A constant learned
HiPPO-initialized LTI state matrix is not reported as LegS. EWMA supplies the
exponential-memory control.

## Optimization

Learned models minimize next-error Bernoulli NLL using AdamW with learning rate
`1e-3`, weight decay `1e-4`, batch size four complete sequences, maximum 60
epochs, gradient norm cap `1.0`, and no test access. Checkpoints are selected by
validation NLL; ties choose the earlier epoch. Calibration fits one temperature
on the calibration role.

Primary training seed is `1701`. Seeds `1702` and `1703` are robustness runs only
after a model survives discovery; no best-seed selection is permitted. The
deployed confirmation checkpoint is the seed-1701 validation winner.

Every stateful estimator resets at a sequence boundary. Burn-in rounds update
state but do not contribute scored losses. Prediction always precedes the update
with round `t`: for EWMA, `m_0=0`, `q_t=mapper(m_t)`, and
`m_(t+1)=decay*m_t+(1-decay)*s_t`. FIR, HiPPO, and GRU obey the identical
predict-then-update convention, so `s_t` cannot influence `q_t`.

## Statistical contract

Independent sequences are the sampling unit. Correlated rounds are never treated
as independent observations for inferential intervals or p-values.

For sequence `j`, compute each arm's mean block-error loss and paired loss
difference. Primary inference uses:

- a one-sided paired sequence-level t-test with `alpha=0.025` and at least 64
  independent sequences;
- a studentized wild-cluster bootstrap-t sensitivity analysis with 100,000
  Rademacher draws and seed `20260904`, described as asymptotic rather than exact;
- 10,000 whole-sequence cluster-bootstrap replicates with seed `20260905` for
  BLER and difference intervals;
- round-level error counts and discordance tables as descriptive diagnostics.

For a paired contrast let `d_j` be the sequence-level difference, `mu0=0` for
benefit tests and `mu0=0.005` for H5 noninferiority, and
`T_obs=(mean(d)-mu0)/(sd(d)/sqrt(n))`. The paired t-test uses the corresponding
lower tail. For each wild bootstrap-t draw use `e_j=d_j-mean(d)`,
`d*_j=mu0+xi_j*e_j` for independent Rademacher `xi_j`, and
`T*_j=(mean(d*)-mu0)/(sd(d*)/sqrt(n))`. Reject and resample zero-variance draws,
with at most 1,000,000 total draws to obtain 100,000 valid draws; failure to do
so produces status `bootstrap_degenerate` and prohibits a claim. The p-value is
`(1 + count(T_star <= T_obs))/(100000 + 1)`. If observed `sd(d)==0`, both t and
bootstrap inference receive status `degenerate_variance`, no claim status is
allowed, and the constant observed difference is reported descriptively.
The cluster-bootstrap interval resamples complete sequence indices with
replacement 10,000 times and uses percentile `2.5%` and `97.5%` endpoints.
Incomplete sequences are rejected; no missing-round imputation is permitted.

The hierarchy is:

1. H1 chooses the better of predeclared EWMA and logistic AR on validation, then
   compares that frozen baseline with the stationary field;
2. H2 FNO-HiPPO versus the best frozen non-neural causal baseline;
3. H3 tests the per-sequence interaction contrast against `I<0`, then uses an
   intersection-union test requiring FNO-HiPPO
   to beat both FNO-FIR and CNN-HiPPO at one-sided `0.025`;
4. Holm-adjusted secondary comparisons against CNN-FIR and both GRUs;
5. H5 deployment-shift stationary-i.i.d. noninferiority is a co-primary safety
   gate, not conditional on the GRU comparisons. It is supported only when the
   upper 95% cluster confidence bound for frozen joint-checkpoint-minus-stationary
   BLER is below `0.005`.

Failure at one level stops confirmatory interpretation below it. A nonsignificant
superiority test is never called noninferiority.

## Metrics

Decoder metrics:

- sequence-clustered block-logical error and difference intervals;
- per-sequence paired loss distributions;
- descriptive round-level paired 2x2 tables;
- syndrome validity, BP convergence, iterations, and correction weight.

Forecast metrics:

- Bernoulli NLL and Brier score by sequence;
- fixed-bin expected calibration error and reliability tables;
- calibration slope/intercept;
- simulator-only correlation with latent `q_t`;
- onset, persistence, and recovery curves with whole-sequence intervals.

For burst diagnostics, onset is the persisted first active round, termination is
the first inactive round after an event, and recovery is the first of eight
consecutive rounds for which the absolute model-minus-stationary NLL difference
is no greater than `1.05` times the mean absolute difference over the 16
pre-onset rounds. Censored events are reported separately and excluded from
recovery-time means. An event is censored if another event begins before the
eight-round recovery condition has been established.
No conditional claim is made with fewer than 30 complete test events.

Operational diagnostics are batch-one estimator latency, BP latency, p50/p95/p99
end-to-end latency, state bytes, parameter count, and continuous-replay backlog.
They support no FPGA or real-time claim in this phase.

## Artifact and verification contract

Every stage publishes immutable payloads followed by a completion manifest that
binds configuration, source commit, code matrices, generator version, all seed
derivations, sequence identities, model and optimizer state, calibration,
causal feature indices, decoder configuration, and per-round outcomes.

Verification reconstructs every forecaster input from strictly earlier rounds,
checks role-disjoint sequence and seed tuples, regenerates synthetic sequences
from their manifests, and recomputes sequence-level statistics from verified
outcomes. Current/future syndromes, errors, logical labels, and post-decoding
fields are forbidden model features by schema and mutation tests.

Discovery, reduced, incomplete, adaptively sized, or post-hoc outputs cannot emit
a primary claim status. Confirmation refuses to generate test data unless its
configuration contains the reviewed validation-variance hash and frozen power
calculation.

## Gated execution

### Gate 1: primitives

Implement strict generator configs, all five regimes, role-separated sequence
sampling, direct-regeneration verification, HiPPO-LegS primitives, and
cluster-aware metrics. An adversarial reviewer must clear causality, recurrence,
and statistical-unit tests.

### Gate 2: reduced factor-isolation screen

Train the four crossed cells and basic baselines on reduced data. Each learned
arm must overfit a deterministic fixture. All regimes use the privileged oracle
only as a generator sanity check. Regimes B-E must show observation-only causal
headroom: EWMA or logistic AR must improve over the stationary-field baseline
before neural work proceeds. Regime A is expected to show no adaptive headroom.

### Gate 3: discovery campaign

Run the exact discovery sizes, crossed models, non-neural filters, and basis-
mismatch regime. Select no result using decoder test outcomes because no test
role exists. Use validation differences to calculate the confirmation sequence
count and freeze the surviving comparisons.

### Gate 4: confirmation-config review

An independent QEC reviewer audits information symmetry, power, multiplicity,
model selection, runtime, and manifests. Only a clean review permits generation
of confirmation test sequences.

### Gate 5: fixed confirmation

Run identical test sequences through every frozen arm and report the hierarchical
decision procedure plus the co-primary H5 safety gate. Three-seed robustness is
launched only for a surviving primary model. A final claim review reads code,
configs, logs, seeds, manifests, and reconstructed statistics before
documentation calls any hypothesis supported.

## Decision rules

- Neither EWMA nor logistic AR has gain: retire the regime as causally
  unidentifiable from the available history.
- EWMA or logistic AR matches the learned models: favor the simple estimator.
- No crossed interaction: reject FNO-HiPPO complementarity.
- Interaction only in `joint_in_basis`: report an in-basis mechanism result.
- CNN-HiPPO wins: Fourier structure is unnecessary.
- FNO-FIR wins: HiPPO memory is unnecessary.
- GRU wins at matched information/capacity: use GRU and retire the HiPPO claim.
- FNO-HiPPO survives every gate: proceed to chronological hardware data, then
  latency engineering.
