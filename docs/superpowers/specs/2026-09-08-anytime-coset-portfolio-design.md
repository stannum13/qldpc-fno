# Anytime Logical-Coset Portfolio: Experiment Design

**Status:** design frozen before implementation. No result is claimed here.

## Purpose

This experiment asks whether learned models are more useful as sources of
decoder diversity than as replacement decoders. The proposed system keeps
belief propagation and localized-statistics decoding responsible for algebraic
validity. It uses several controlled decoder views to produce syndrome-valid
corrections, compares those candidates in logical-coset space, and spends extra
work only on shots that remain ambiguous.

The intended contribution is not another unconstrained neural correction
string. It is a calibrated control plane for hardware-friendly qLDPC decoders:

1. generate a cheap candidate;
2. stop when its evidence is sufficient;
3. otherwise request another diverse view;
4. combine candidates by logical class, not raw bit-string agreement.

## Why this experiment follows from the repository

The completed fixed-shot experiment at `p=0.0375` rejected both implemented
FNO hybrids as standalone decoders. Nominal BP-LSD failed on 387 of 2,048 held
out shots, while the soft-prior and residual-repair arms failed on 835 and 819.

After that claim was frozen, an exploratory analysis found a different signal.
The union of the three syndrome-valid candidate sets failed on 343 shots, and a
minimum-Hamming-weight rule with a fixed baseline-first tie order failed on 344.
The rule rescued 43 nominal failures without introducing a nominal-only
regression. It selected nominal BP-LSD on 1,839 shots, and one of the FNO-derived
candidates on the remaining 209. There were 1,325 minimum-weight ties.

These numbers came from an already opened test set. They are motivation only:
they may not train a selector, choose a threshold, or support a new empirical
claim. They establish one falsifiable question for fresh data: does FNO-derived
candidate diversity persist after comparison with strong BP-generated
diversity?

## Claim boundary

The first admissible claim is deliberately narrow:

> On the canonical `lp(3,7)_16` independent-Z code-capacity task at
> `p=0.0375`, frozen FNO-derived BP views do or do not add correct logical
> classes beyond the source paper's five BP-LSD views.

It is not a circuit-level, threshold, code-family, Willow, neutral-atom,
real-time, FPGA, or cross-size claim. Those require later gates.

## Common code and outcome definition

Every arm consumes the identical committed `Hx`, syndrome rows, logical-X
operator basis, physical error rate, and side information. A correction is
syndrome-valid exactly when

```text
Hx @ correction mod 2 == syndrome.
```

For this Z-error experiment, its logical signature is

```text
logical_x @ correction mod 2.
```

Candidates for the same syndrome are in the same relative logical class when
their signatures agree. The exact binary logical-X matrix is persisted and
hashed alongside `Hx` and `Hz`; validation checks its shape, rank and
completeness, commutation, and byte identity at scoring time. Decoder success is
scored only after joining a candidate signature with the shot's actual
observable flips. Raw correction agreement is never used as a correctness
criterion.

## Candidate generators

Gate 1a runs a faithful reconstruction of the source paper's published
five-view BP-LSD specification before asking whether FNO adds value. The paper
does not publish its stochastic schedule realizations or random seeds, so this
is not claimed to be bit-identical to the authors' run. Every BP-LSD instance
uses the pinned `ldpc==2.4.1` build, float64 channel probabilities, min-sum BP,
serial scheduling, 100 iterations, `ms_scaling_factor=0`, `LSD_E`, and order 5.

| ID | View |
|---|---|
| `b0_nominal` | uniform prior `p` |
| `b1_random_serial` | uniform prior `p`, deterministic randomized serial schedule |
| `b2_optimistic` | uniform prior `0.8p` |
| `b3_pessimistic` | uniform prior `1.2p` |
| `b4_thermal` | per-qubit prior `p(1+eta_i)`, `eta_i ~ Normal(0, 0.04)` |
| `f0_soft` | frozen soft-prior FNO followed by the same BP-LSD |
| `f1_residual` | frozen thresholded FNO proposal followed by the same residual BP-LSD |

Perturbed probabilities are clipped to `[1e-5, 0.5-1e-5]`. The thermal field is
resampled once per shot and remains fixed for all BP/LSD work on that shot. The
randomized arm constructs one decoder per shot and uses only `ldpc==2.4.1`'s
native `random_schedule_seed`; it does not pass `serial_schedule_order`. A
deterministic fixture must characterize whether the native order changes across
iterations and prove identical corrections, convergence, and iteration counts
under replay before the first development decode. Random schedule seeds and
thermal fields use independent SHA-256-derived seed domains and coordinates
`(campaign_seed, shard_index, shot_index, arm_id)` with NumPy `PCG64`. The
native schedule seed, characterized package behavior, and thermal-field hash are
retained.

All arms retain invalid candidates as evidence. Invalid candidates are excluded
from portfolio selection and count as failures when their arm is evaluated
alone.

## Outcome-blind candidate artifact

Candidate generation consumes a syndrome-only input root and publishes and
hash-closes an artifact before any scorer opens physical errors or actual
observable flips. The syndrome-only reader rejects `errors.b8`,
`obs_actual.b8`, outcome fields, and paths outside that root; candidate
generation receives no scoring-root path. For every shot and arm it stores:

- shot and syndrome identity;
- packed correction and packed logical signature;
- syndrome validity;
- Hamming weight and nominal LLR cost;
- convergence, BP iterations, and whether LSD ran;
- setup, decode, and end-to-end engineering timings;
- arm configuration, schedule, thermal-field, and FNO provenance hashes.

Exact duplicate corrections are stored once with all generating-arm provenance.
The artifact contains no physical error string, actual observable flip,
candidate correctness flag, or feature derived from an outcome. A separate
scorer performs the outcome join.

Every arm is judged by the same nominal cost

```text
C(c) = sum_i c_i * log((1-p)/p).
```

For the uniform channel this is proportional to Hamming weight. The frozen tie
order is cost, arm order `b0` through `f1`, then lexicographically packed
correction bytes.

## Data firewall

The prior 2,048-shot artifact is permanently development-open. The new runner
must reject its sampling seed, manifest hash, and any input rooted below
`artifacts/accuracy-disconfirm-p0375`.

Gate 1a uses campaign seed `2026090801`, 10,000 fixed shots, immutable shards of
at most 2,048 shots, no target-failure stopping, and a clean source commit. Seed
domains for sampling, decoder schedules, thermal perturbations, bootstrap, and
later selector splits are disjoint. Training, calibration, and confirmation
roles never share sampler identities.

## Gate 1a: incremental candidate diversity

Let `O5` indicate that every syndrome-valid source-five candidate has the wrong
logical signature, and `O7` the equivalent event for all seven candidates. The
primary estimand is

```text
Delta_FNO = P(O5 = 1 and O7 = 0).
```

This measures shots rescued only by adding the two frozen FNO-derived views.
Report an exact one-sided 95% Clopper-Pearson lower bound for this primary
unconditional rate. Report the conditional rescue fraction among source-five
oracle failures, with its own exact interval, as descriptive only; it is not a
second route to `GO`. Also report leave-one-arm-out unique rescues, distinct
corrections and signatures, pairwise failure dependence, and deployable selector
BLERs as secondary outcomes.

With 10,000 shots, 118 rescues are the minimum for the one-sided exact lower
bound to exceed one percentage point. Gate 1a is `GO` only when:

1. the one-sided 95% lower bound on the unconditional rescue rate is at least
   `0.01`;
2. both FNO arms contribute at least one leave-one-arm-out unique rescue; and
3. all information-symmetry and artifact-integrity checks pass.

Otherwise the FNO portfolio thread stops. The reproduced source-five portfolio
remains a valid conventional baseline.

Before spending fresh confirmation shots, all seven arms are rerun on a
syndrome-only projection of the already opened 2,048 shots as a development-only
disconfirmation. Existing candidate outcomes cannot be reused because they do
not persist corrections or logical signatures. Invalid candidates are excluded
from both oracle sets. Phase A stops Gate 1a exactly when fewer than 21 of the
2,048 shots satisfy `O5=1 and O7=0`, meaning the observed unconditional rescue
fraction is below `0.010000`. There is no interval-based exception.

## Gate 1b: Relay-BP comparison

Relay-BP is a required modern baseline for a state-of-the-art claim, but it does
not block the cheaper source-five disconfirmation. Its adapter must use the same
`Hx`, syndromes, priors, and logical operators; pin a released implementation;
replay deterministically; and receive tuning effort on development data before
confirmation.

The frozen Relay development search uses 4,096 new development shots and the 16
Cartesian configurations formed by `gamma0 in {0.05, 0.10}`,
`pre_iter in {40, 80}`, `num_sets in {30, 60}`, and
`gamma_dist_interval in {(-0.24, 0.66), (-0.18, 0.54)}`, with
`set_max_iter=60` and `stop_nconv=1`. Selection minimizes BLER, then mean BP
iterations, then the listed lexicographic parameter order. Confirmation compares
against the better, under this frozen development rule, of the selected
configuration and the documented default
`(0.10, 80, 60, (-0.24, 0.66), 60, 1)`. The implementation version, trial count,
sample role, metric, compute consumed, and all results are retained even when
tuning makes performance worse.

Gate 1b asks whether FNO adds oracle headroom beyond source-five plus tuned
Relay-BP on campaign seed `2026090802`, with 10,000 fixed shots, shards of at
most 2,048 shots, and no adaptive stopping. It uses the same primary estimand,
118-rescue minimum, interval, and `GO` rule as Gate 1a. If a deterministic,
comparable Relay-BP adapter cannot be built, the claim remains explicitly scoped
to BP-LSD portfolios.

## Gate 2: select a logical class

Gate 2 opens only after candidate headroom exists. Separate selector training,
calibration, and confirmation samples compare:

1. source-paper minimum cost;
2. all-arm minimum cost;
3. logical-signature plurality;
4. cost-weighted logical-signature voting; and
5. a regularized logistic ranker over outcome-blind candidate-set features.

The fixed sample roles are 8,000 selector-training shots under seed
`2026090811`, 4,000 calibration shots under `2026090812`, and 20,000
confirmation shots under `2026090813`.

The learned ranker may see arm support, correction multiplicity, cost summaries,
cost gaps, convergence counts, and iteration summaries. It may not see errors,
actual observable flips, or future candidates at inference. Selector choice is
frozen on calibration. Confirmation uses paired exact inference against the
source-paper minimum-cost rule.

## Gate 3: anytime control

Only after Gate 2 establishes selectable value does a controller decide which
view to run next and when to stop. Comparators include the complete frozen
portfolio, the fixed source-five order, two-view signature agreement, calibrated
confidence thresholds, and a shallow next-arm controller.

Gate 3 uses 8,000 controller-training shots under campaign seed `2026090819`,
4,000 threshold-calibration shots under `2026090820`, and 20,000 fixed
confirmation shots under `2026090821`. Arm order, confidence thresholds, and
the shallow controller are frozen before confirmation is opened.
The reference is the complete frozen portfolio executing every available view
and applying the Gate 2-selected logical-class rule. For shot `i`, let
`W_calls(i)` be the number of decoder views executed and `W_bp(i)` the sum of BP
iterations over those views. The sole Gate 3 success route requires all of:

- one-sided 95% upper confidence bound on
  `BLER(anytime) - BLER(complete)` below `0.005`; and
- one-sided 95% upper confidence bound on the mean decoder-call ratio below
  `0.5`; and
- one-sided 95% upper confidence bound on the mean BP-iteration ratio below
  `0.5`.

The paired BLER bound uses a tested Tango score interval for paired proportions.
The two work estimands are ratios of means,
`mean(W_anytime)/mean(W_complete)`, not means of per-shot ratios. The complete
portfolio executes at least one view, so both denominators are positive. Each of
100,000 paired bootstrap resamples draws common shot indices, recomputes both
means and their ratio, and uses the one-sided percentile upper bound. Its seed
comes from `qldpc-fno/portfolio/gate3-bootstrap/v1` and the frozen Gate 3
campaign seed. Because every condition is required, this is an
intersection-union gate rather than alternative unadjusted success routes.

Decoder calls and BP iterations are the only hardware-independent gating
quantities. BP edge updates, LSD invocations and available cluster-growth work,
FNO calls and estimated MACs, candidate count, and timing are reported as a work
vector rather than collapsed into an arbitrary scalar. Passing Gate 3 supports
only a decoder-call and BP-iteration reduction claim. Any total-compute,
throughput, or real-time claim additionally requires lower measured warmed
latency and acceptable backlog, including observed LSD work. "Matched work" is
not used as a claim.

Report calls, BP iterations, LSD invocations, FNO invocations, candidates,
logical signatures, work to stopping, p50/p95/p99/p99.9 warmed single-shot
latency, throughput, and simulated backlog. Batched GPU inference is never
presented as single-shot FPGA latency.

## Role of FNO and HiPPO

FNO is justified here only as a structured perturbation or candidate generator.
It is retired if it supplies no incremental correct logical classes beyond
source-five and Relay-BP.

HiPPO is not part of Gates 1 or 2. It enters only after a separate privileged
oracle shows that past-only, spatially heterogeneous relative risks improve
logical error or BP work. At that point FNO may estimate low-frequency spatial
modes and HiPPO may carry their causal temporal state. A global scalar drift
forecast alone is insufficient justification because it mostly changes common
LLR scale rather than reliability ordering.

## Final stop rule

The project does not rescue an architectural preference. If strong BP views
eliminate FNO's unique headroom, FNO is retired from decoding. If candidate
headroom exists but no deployable selector closes it, the result is an oracle
analysis, not a decoder. If an anytime controller does not meet both accuracy
and work requirements, no real-time claim is made.
