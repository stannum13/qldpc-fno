# Temporal Syndrome-History Identifiability Design

## Decision this study makes

The reduced causal FNO-HiPPO screen did not show that an observation-only
history-aware baseline could improve on a stationary field. Scaling its neural
models would confound two questions:

1. does past syndrome data contain useful information about the next-round
   channel; and
2. if it does, which architecture estimates that information efficiently?

This architecture-free study answers the first question in the cheapest
identifiable special case:

> Under a known scalar time-varying error generator, how much of the next-round
> physical error probability can be recovered from strictly past syndromes?

The result is a temporal gate. Spatial generators and FNO, HiPPO, attention, or
other learned forecasters remain out of scope until this gate passes. Their
design will receive a separate preregistration rather than being partially
specified here.

## Claim boundary

This is a synthetic code-capacity identifiability experiment on the canonical
`lp(3,7)_16` code. Physical Z errors are conditionally independent given a
latent scalar, syndrome measurements are perfect, and Pauli frames do not carry
between rounds. It is not a circuit-level memory, hardware, Willow, threshold,
or real-time decoding experiment.

The primary endpoint measures estimation of simulator probabilities. BP-LSD is
a secondary downstream diagnostic. A positive result supports only the claim
that the specified syndrome history contains recoverable predictive information
under the specified generator. A failed estimator does not prove that no such
information exists outside the validated observer class.

## Causal timing

At scored round `t`, every deployable estimator receives only public code and
generator metadata plus observations through `t-1`. It predicts `q_hat_t`
before `syndrome_t`, `error_t`, or any diagnostic for round `t` is available.

```text
syndrome_0 ... syndrome_(t-1) -> filter -> q_hat_t
latent state_t --------------------------> q_t  # scoring only
```

The privileged causal ceiling follows the same timing but receives the true
latent state through `t-1` and integrates the known transition to round `t`.
The contemporaneous ceiling receives true `q_t`. Neither is deployable.

## Generator

The study reuses the existing `temporal_uniform` generator without changing its
dynamics. All qubits share one latent log-odds offset:

```text
q[t,i]     = sigmoid(logit(0.0375) + g[t])
g[t+1]     = clip(0.97*g[t] + epsilon[t], -1.20, 1.20)
epsilon[t] ~ Normal(0, 0.08^2)
g[0]       = 0
```

`stationary_iid`, with `g[t]=0`, is the leakage and false-positive control.
The code, generator implementation, probability clipping, and fixed BP-LSD
policy are unchanged from the completed reduced screen. This study receives a
new domain-separated campaign seed.

The scalar problem is deliberately favorable to classical filtering. If a
validated generator-matched estimator cannot recover useful signal here, a
larger neural architecture search is not justified.

## Exact retained-row likelihood

For parity-check row `j` with qubit support `N(j)`, conditionally independent
Bernoulli physical errors imply

```text
P(s_j=1 | q_t) = (1 - product(i in N(j), 1 - 2*q[t,i])) / 2.
```

Rows that share qubits are correlated. Their Bernoulli marginals must not be
multiplied and called an exact likelihood.

The primary likelihood uses a deterministic row-disjoint subset of `Hx`:

1. traverse canonical CSR rows in ascending index order;
2. retain a row exactly when its support is disjoint from every retained row;
3. bind the row indices, supports, matrix hash, and algorithm version into the
   manifest.

On the current canonical matrix this rule retains 135 weight-10 rows covering
1,350 distinct qubits. Code verifies these values; they are not input constants.
Conditional on the scalar latent field, these syndrome bits are mutually
independent, so their product is an exact likelihood for the retained data.

Using all rows as a product of marginals is allowed only as the explicitly named
`composite_full_rows` diagnostic. It cannot determine the primary result.

## Local identifiability precheck

Before sequential filtering, compute the scalar Bernoulli Fisher information

```text
F(g) = sum(j) [d p_j(g) / d g]^2 / [p_j(g)*(1-p_j(g))].
```

Check the analytic derivative against central finite differences over 10,000
prior draws. Report the minimum, median, and maximum finite information and the
corresponding Cramer-Rao bounds. Stop before confirmatory generation if any
value is nonfinite or nonpositive, or if derivatives disagree beyond fixed
tolerances. This establishes local sensitivity of the retained measurement; it
does not establish finite-sequence recovery.

## Estimator ladder

All estimators predict before observing the current round.

1. `known_marginal`: open-loop prior predictive mean with no observations.
2. `empirical_stationary`: per-qubit mean fitted on the training role.
3. `ewma`: the existing observation-only EWMA baseline.
4. `logistic_ar32`: the existing fitted 32-lag circular logistic baseline.
5. `parity_moment_ar`: a parameter-free filter that inverts the weight-10
   parity moment
   `P(s_j=1|p) = [1-(1-2p)^10]/2` and applies the known scalar AR transition.
6. `grid_bayes`: a generator-matched one-dimensional Bayesian filter over the
   clipped AR(1) state, using the exact retained-row likelihood.
7. `latent_history_oracle`: a privileged causal ceiling that sees true latent
   states only through `t-1` and integrates the known transition to predict `t`.
8. `contemporaneous_oracle`: true `q_t`; a noncausal ceiling only.

The nominal grid has 2,048 equal-width interior cells on `(-1.20, 1.20)` plus
separate atoms at `-1.20` and `+1.20`. An interior cell is represented by its
midpoint. Transition probabilities into every interior cell are Gaussian-CDF
differences at its edges; the two atoms receive the exact probability below or
above the clipping boundary. Prediction integrates probabilities at the
interior midpoints and boundary atoms. The doubled grid uses 4,096 interior
cells and the identical convention. It must change mean `G_grid` by less than
`2.5e-5` nats/qubit/round on validation data. Round zero is the exact point mass
`g[0]=0`, not a discretized initialization.

Every syndrome-history arm also has a `history_deranged` control: a fixed
derangement replaces each sequence's history with another sequence's history
while its scored latent target remains unchanged. Privileged latent ceilings
are not deranged. Deranged outputs are controls, not scientific replicates.

## Primary estimands

The primary loss is expected Bernoulli cross-entropy against the true latent
probability, averaged over all physical qubits and scored rounds within a
sequence:

```text
CE(q, q_hat) = mean(-q*log(q_hat) - (1-q)*log(1-q_hat)).
```

This integrates out sampled-error label noise. Let lower CE be better:

```text
G_grid     = CE(q, known_marginal) - CE(q, grid_bayes)
G_ewma     = CE(q, known_marginal) - CE(q, ewma)
G_logistic = CE(q, known_marginal) - CE(q, logistic_ar32)
G_moment   = CE(q, known_marginal) - CE(q, parity_moment_ar)
G_latent   = CE(q, known_marginal) - CE(q, latent_history_oracle)
G_current  = CE(q, known_marginal) - CE(q, contemporaneous_oracle)
G_deranged[arm] = CE(q, known_marginal)
                   - CE(q, history_deranged[arm])
```

Secondary endpoints are latent-state normalized MSE, retained-syndrome
predictive NLL, calibration, and BP-LSD BLER using the predicted global field.
Decoder outcomes cannot overturn a failed primary identifiability gate.

## Data and splits

Each sequence has 64 burn-in rounds followed by 128 scored rounds.

- train: 8 independent sequences per regime, used only by fitted baselines;
- validation: 8 independent sequences per regime, used for fixed baseline
  settings and numerical diagnostics;
- calibration: 8 independent sequences per regime, used only for declared
  calibration transforms, if any;
- test: 64 new independent sequences per regime, generated only after the
  implementation, thresholds, and review record are frozen.

The study contains only `stationary_iid` and `temporal_uniform`. No development
sequence, fitted field, or selection outcome is reused as a test sampling unit.
Every role, sequence, latent stream, Bernoulli stream, and filter stream has an
identity-derived seed. The sequence is the inferential unit; qubits, rounds,
checks, and numerical-resolution runs are not replicates.

## Confirmatory hypotheses and decisions

Predeclare `delta_NLL = 0.00025` nats/qubit/round, approximately 10% of the
completed reduced screen's descriptive contemporaneous-oracle gap in
`temporal_uniform`. For each test sequence, average loss over qubits and scored
rounds, then compare paired values across the 64 sequence identities.

- **Numerical validity:** doubled grid resolution changes mean `G_grid` by less
  than `2.5e-5`.
- **Negative controls:** the stationary-regime gain and temporal-regime
  `G_deranged[arm]` for every candidate winning arm have upper one-sided 95%
  bounds no greater than `delta_NLL`.
- **Detectable causal ceiling:** the lower one-sided 95% bound for `G_latent`
  exceeds `delta_NLL`.
- **Recoverable syndrome information:** the lower one-sided 95% bound for at
  least one of `G_grid`, `G_ewma`, `G_logistic`, or `G_moment` exceeds
  `delta_NLL`, and that arm's deranged-history control remains null.

Use a paired studentized bootstrap over whole sequences with 10,000 draws. The
four deployable-arm gain tests form one Holm-corrected family. Comparisons among
arms are descriptive and paired unless separately predeclared. Report every
interval and adjusted value. If bootstrap behavior is degenerate, report it and
do not substitute qubits or rounds as independent samples.

Outcomes are deliberately asymmetric:

- if the privileged causal ceiling does not clear `delta_NLL`, report
  `STOP-NO-PRACTICAL-CAUSAL-HEADROOM` for this generator;
- if the ceiling clears but no validated syndrome-history arm clears the
  threshold, report `INCONCLUSIVE-OBSERVER-GAP`, not “syndrome history has no
  information”;
- if any validated syndrome-history arm clears the threshold and its control,
  report `GO-TEMPORAL-IDENTIFIED` and permit design of a separate spatial gate;
- within a `GO`, report `CURRENT-BASELINE-LIMITATION` only if `grid_bayes`
  clears while EWMA and logistic AR do not, and report
  `REDUCED-SCREEN-OR-SAMPLE-LIMITATION` if EWMA or logistic AR clears on the new
  independent test. Other winning-arm combinations are reported literally;
- if a leakage, derangement, or convergence control fails, report
  `STOP-INVALID-CONTROL`.

## Conditional BP-LSD diagnostic

Run BP-LSD only after `GO-TEMPORAL-IDENTIFIED`. Compare `known_marginal`, every
syndrome-history arm that passed the primary gate, and the
contemporaneous-oracle prior on identical test rounds. For each arm, define the
paired difference as `BLER_arm - BLER_known_marginal` and
classify its 95% interval against the fixed absolute margin `0.01`:

- `BENEFIT` if the upper endpoint is below `-0.01`;
- `HARM` if the lower endpoint is above `+0.01`;
- `EQUIVALENT` if the whole interval lies within `[-0.01, +0.01]`;
- `INCONCLUSIVE` otherwise.

If the contemporaneous oracle is `BENEFIT` while the causal filter is not, the
causal forecast was insufficiently decoder-relevant. If both are `EQUIVALENT`,
the conclusion is limited to insensitivity to this regime's global-rate
variation. Other combinations are reported literally. NLL identifiability and
downstream BLER are nested questions, not mutually exclusive mechanisms.

## Artifact and replay contract

The command-line run publishes immutable artifacts only after all payloads
exist. The completion manifest binds:

- exact Git source tree and configuration hashes;
- canonical `Hx`, `Hz`, and logical-operator identities;
- row-disjoint construction and retained supports;
- generator parameters, split identities, and every RNG stream;
- latent states, probabilities, errors, syndromes, and scored masks;
- estimator predictions, states, and numerical diagnostics;
- primary per-sequence losses and aggregate inference;
- conditional BP-LSD configuration, corrections, syndrome-validity checks, and
  logical outcomes;
- runtime and memory measurements labelled engineering-only.

A standalone verifier regenerates sequence payloads, recomputes the retained
likelihood, replays deterministic filters, reconstructs syndrome and logical
labels, and recomputes every primary result. Confirmation refuses a dirty source
tree, noncanonical code, mutable output path, or missing development approval.

## Runtime gate

Development measures an end-to-end local projection before confirmation. The
target is at most two CPU-hours and the hard stop is six CPU-hours for the full
confirmatory run. Runtime reduction may lower grid resolution only before the
configuration is frozen and only if the doubled-resolution tolerance still
passes. It may not reduce sequence count after seeing confirmatory outcomes.

## Explicitly deferred work

- spatial or joint spatial-temporal latent generators;
- learned FNO, CNN, HiPPO, GRU, Transformer, or attention comparisons;
- circuit-level detector error models and noisy measurements;
- hardware or Willow data adaptation;
- code-size transfer and additional qLDPC instances;
- FPGA synthesis, quantization, throughput, or latency claims;
- decoder ensembles or multi-pass consensus.

Those experiments become meaningful only after this study establishes temporal
information in syndrome history.
