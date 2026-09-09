# Anytime Logical-Coset Portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or `executing-plans` to implement this plan
> task by task. Steps use checkbox syntax for tracking.

**Goal:** test whether FNO-derived BP views add useful logical-coset diversity
beyond fair qLDPC baselines, then build a calibrated anytime selector only if
that headroom survives.

**Design:**
`docs/superpowers/specs/2026-09-08-anytime-coset-portfolio-design.md`

**Tech stack:** Python 3.14, NumPy, SciPy, Stim, ldpc, PyTorch, pytest.

## Phase 0: exact prediction-to-decision gate

### Task 0a: implement exact CSS posterior enumeration

- [x] Add a `qldpc_fno.decision` package with a canonical Steane Z-error
  problem, independent Bernoulli likelihoods, physical-MAP decoding, exact
  logical-coset posterior aggregation, and Bayes logical risk.
- [x] Test syndrome identities, stabilizer invariance, logical toggling,
  posterior normalization, and a frozen physical-MAP/coset-MAP counterexample
  before implementing each behavior.
- [x] Use only an independent stabilizer basis and assert one canonical
  coefficient representation per affine-space member.

### Task 0b: produce the sensitivity and exhaustive-action artifact

- [x] Add a deterministic CLI that evaluates nominal-uniform, correct-global,
  and correct-spatial priors under frozen uniform and heterogeneous true
  channels for every syndrome.
- [x] Store the true syndrome probability, selected physical error and logical
  class, inferred class probabilities, conditional Bayes risk, operation count,
  input hashes, and code identity in JSON.
- [x] Summarize how often a better prior changes the physical error, logical
  class, and Bayes risk. Label exact enumeration separately from any sampled
  logical-error estimate.
- [x] Run the artifact twice into different temporary directories and require
  byte-identical JSON.

### Task 0c: decide which advanced branches have oracle headroom

- [x] Measure the best possible improvement from choosing among the frozen
  inference actions with access to the true channel, and the improvement from
  exact coset aggregation over physical MAP.
- [x] Open GFlowNet work only if multiple physical configurations contribute
  material posterior mass and class aggregation changes at least one frozen
  decision.
- [x] Open adaptive contraction work only when a tensor-network reference on a
  local code exposes an accuracy-cost frontier across `chi`.
- [x] Open learned action selection only if action values vary across observable
  states; otherwise publish the invariance or constant-policy result.

## Phase A: cheapest disconfirmation on opened data

### Task 1: freeze identities and schemas

- Add a strict portfolio config with the canonical code, `p=0.0375`, source-five
  BP-LSD parameters, FNO checkpoint hashes, seed domains, fixed arm order, and
  development-only input identities.
- Define typed candidate, candidate-set, logical-signature, arm-provenance, and
  timing records.
- Define an outcome-blind sharded artifact and separate outcome join.
- Materialize and hash the exact logical-X matrix used by signature generation
  and scoring; validate shape, binary rank/completeness, commutation, and byte
  identity.
- Publish a syndrome-only input root whose reader has no scoring-root path and
  rejects label-bearing files or outcome-derived metadata.
- Reject unknown fields, duplicate shot identities, unsafe paths, mismatched
  matrices/operators, old confirmation roles, and incomplete arms.
- Test every rejection before implementing the writer.

### Task 2: reproduce the source-paper five views

- Extend BP-LSD construction with the pinned package's native
  `random_schedule_seed`, `random_serial_schedule=True`, one decoder seed per
  shot, and no explicit serial order. Characterize and freeze the package's
  across-iteration behavior in a deterministic fixture.
- Implement deterministic per-shot thermal priors and `0.8p`/`1.2p` views.
- Verify every correction against the common syndrome.
- Record exact setup and decoding work without making a speed claim.
- Unit-test channel vectors, seed separation, schedule hashes, clipping, and
  exact replay.

### Task 3: publish candidate evidence

- Rerun all seven views on a syndrome-only projection of the already opened
  2,048 shots. Reuse only immutable model, config, and sample provenance; the
  old outcome files do not contain corrections or logical signatures.
- Persist packed corrections and logical signatures for newly decoded views.
- Hash-close candidate shards before the scorer opens outcomes.
- Independently replay at least one shard and verify every syndrome and logical
  signature from the stored correction.

### Task 4: run the opened-data oracle audit

- Compute source-five and seven-arm oracle failure masks, marginal FNO rescue,
  leave-one-arm-out rescues, correction/signature multiplicity, failure
  dependence, and minimum-cost selectors.
- Label every output `development_only_opened_test`.
- Obtain an independent QEC/statistical review.
- Stop fresh Gate 1a when fewer than 21 of 2,048 shots are rescued only by the
  two FNO views, with invalid candidates excluded and no exception.

## Phase B: fresh candidate-diversity gate

### Task 5: generate a new fixed-shot sample

- Use the frozen campaign seed `2026090801` and 10,000-shot cap after Phase A
  review permits the run.
- Generate immutable shards from one common sampler and source commit.
- Reject the prior test seed, manifest, and artifact root.
- Verify payload hashes and regenerate a sample of shards independently before
  decoding.

### Task 6: execute all seven arms

- Run source-five and both frozen FNO-derived views on identical shots.
- Publish outcome-blind candidates first; publish actual observable flips in a
  separately access-controlled scoring artifact.
- Resume only at verified shard boundaries and never overwrite a complete
  shard.
- Record syndrome validity, cost, convergence, iterations, LSD use, and
  engineering timings for every arm and shot.

### Task 7: decide Gate 1a

- Join outcomes only after candidate publication is complete.
- Compute the exact Clopper-Pearson bound for incremental FNO rescue and all
  frozen secondary diagnostics.
- Independently replay all aggregate counts from shards.
- Obtain claim-level QEC/statistical review before writing public documentation.
- `GO` only under the design's exact thresholds; otherwise retire FNO from the
  portfolio thread.

## Phase C: strong baseline and selection

### Task 8: integrate and tune Relay-BP

- Pin Relay-BP as an optional dependency and document its license/version.
- Implement a code-capacity adapter with common syndrome and logical scoring.
- Reproduce a documented default configuration first.
- Reproduce released reference examples, measure a development timing slice,
  then freeze an affordable search that includes `stop_nconv` values 1, 5, and
  9. Use exactly 4,096 development shots; select by BLER, mean graph-message
  updates, then fixed parameter order.
- Retain the documented default and compare against the better of default and
  development-selected configurations under the frozen rule.
- Require deterministic replay, validity, and matched-information review.

### Task 9: run Gate 1b

- Use a new fixed seed and compare source-five plus Relay-BP with and without
  frozen FNO candidates.
- Use seed `2026090802`, 10,000 fixed shots, shards of at most 2,048, no
  adaptive stopping, and the Gate 1a estimand, 118-rescue minimum, interval, and
  stop rules.
- Scope the result to BP-LSD if Relay-BP cannot be made comparable.

### Task 10: implement nonlearned logical-coset selectors

- Implement source-paper minimum cost, all-arm minimum cost, signature
  plurality, and cost-weighted signature voting.
- Add adversarial fixtures where an invalid low-cost correction shares a
  signature with valid candidates; prove it cannot affect voting,
  deduplication, agreement, or stopping.
- Freeze all tie orders and numerical transforms.
- Prove selectors consume only the outcome-blind artifact.
- Evaluate first on development and calibration roles.

### Task 11: implement the regularized ranker

- Fit only candidate-set summaries that exist at deployment time.
- Keep sampler seeds disjoint across train, calibration, and confirmation.
- Calibrate probabilities and abstention thresholds on calibration only.
- Compare calibration BLER and work under the fixed selection rule.

### Task 12: confirm Gate 2

- Run the frozen selector on 20,000 fresh paired shots.
- Report paired BLER difference, exact McNemar inference, oracle gap closed, and
  all validity controls.
- Proceed only after independent replay and review.

## Phase D: anytime control and realistic scope

### Task 13: characterize an anytime policy

- Use 8,000 controller-training shots under seed `2026090819` and 4,000
  threshold-calibration shots under seed `2026090820`.
- Measure unique rescue against the complete work vector for each next view.
- Freeze arm order and confidence thresholds on training/calibration only.
- Implement full-portfolio, agreement-stop, confidence-stop, and shallow
  next-arm policies.
- Test that no policy observes future-arm or outcome information.

### Task 14: confirm accuracy versus work

- Use 20,000 fresh paired shots under seed `2026090821` and the design's
  intersection-union rule: a
  Tango paired-proportion bound for BLER plus paired bootstrap bounds for both
  ratios of mean decoder calls and mean BP iterations against the complete
  frozen portfolio.
- Report tail work, warmed single-shot latency, throughput, and backlog without
  extrapolating to FPGA timing.
- Stop if either accuracy or work requirement fails.

### Task 15: move to circuit-level evidence

- Build one common Stim circuit/DEM for every arm, with identical extraction
  rounds, final boundary, heralding flags, X/Z treatment, and observables.
- Compare against tuned Relay-BP and the source-paper ensemble.
- Repeat across more than one code instance before making family or scaling
  claims.

### Task 16: open the spatial-history branch only through an oracle

- Simulate past-only heterogeneous `q[i,t]` with known ground-truth priors.
- Measure whether privileged relative priors improve BLER or decoder work.
- Only after that gate compare exact/Kalman/particle filters with
  FNO-plus-HiPPO/S4 estimators.
- Keep drift estimation, candidate generation, and anytime selection as
  separately ablated components.

## Phase E: probability-mass inference engines

### Task 17: implement exact-space sampling baselines

- [x] Express each syndrome-consistent Steane error through one affine-basis
  coefficient vector and test the bijection exhaustively.
- [x] Add conditional rejection sampling and Metropolis sampling baselines with
  exact operation counters and deterministic seed domains.
- [x] Compare sampled logical-class probabilities against Task 0a enumeration
  using total variation, worst-class error, top-class accuracy, coverage, and
  effective sample size over a frozen sample-budget ladder.

### Task 18: test a trajectory-balance GFlowNet

- [x] Build a binary coefficient-assignment environment whose terminal state is
  a unique syndrome-valid physical error and whose reward is its exact channel
  probability.
- [x] Test terminal validity, trajectory multiplicity, reward identity, and
  logical aggregation before training.
- [ ] Train on development channels and compare on held-out syndromes and
  spatial fields at each budget in Task 17.
- [ ] Stop the branch unless it improves logical-class posterior estimation at
  matched evaluated terminal objects; diversity without posterior accuracy is
  recorded as a negative result.

### Task 19: establish a tensor-network reference

- [x] Construct or adapt a surface-code coset partition-function tensor network
  with exact small-distance and high-`chi` references.
- [x] Sweep fixed contraction orders and a frozen `chi` ladder while recording
  logical class, log-mass-ratio error, FLOPs, peak elements, and latency.
- [x] Verify monotonicity is measured rather than assumed: a larger `chi` may
  change numerical error nonmonotonically under different contraction orders.
- [x] Stop before RL if no meaningful per-instance accuracy-cost variation is
  present.

### Task 20: implement adaptive coarse-graining and bond allocation

- [x] Define the first observable finite actions over contraction order and bond
  dimension, with no reference-only policy features.
- [ ] Extend observable contraction states and finite actions to region,
  scale factor, and next bond dimension.
- [ ] Add fixed-`chi`, fixed geometric renormalization, and hyper-optimized
  contraction comparators.
- [x] Create complete action-outcome tables on small networks using the same
  high-`chi` reference, without exposing that reference to policy features.
- [x] Confirm a frozen two-view rule against fixed `chi=8` on an untouched seed
  domain at matched logical-class accuracy and estimated arithmetic work.
- [x] Trace local singular spectra and test fixed-tolerance adaptive bond ranks
  as a required nonlearned baseline on the discovery table.
- [ ] Evaluate multistep adaptive rules at matched logical accuracy and compute,
  including peak memory and worst-case work.

## Phase F: world model, RL, and symbolic distillation

### Task 21: fit an action-conditioned hypergraph world model

- [ ] Encode Tanner nodes, checks, partition regions, scale, singular spectra,
  discarded weight, current logical gaps, and remaining budget with masks tied
  to the hypergraph partition hierarchy.
- [ ] Predict next observable solver state, logical-gap change, selected-class
  change, approximation error, FLOPs, and memory with calibrated uncertainty.
- [ ] Compare scalar, node/check, short-cycle, regional, and cross-scale models
  in that order; stop increasing interaction order when held-out action-value
  prediction does not improve.
- [ ] Test transfer across syndrome distributions, noise fields, code distances,
  and unseen code instances where the underlying inference engine is defined.

### Task 22: train and confirm the computation policy

- [ ] Start with supervised counterfactual action values from complete logged
  action tables; compare a fixed cascade, syndrome-weight rule, and calibrated
  myopic value model.
- [ ] Add limited-depth model-based planning only when earlier actions alter the
  state and value of later actions.
- [ ] Evaluate logical error, regret to the per-instance oracle, mean and tail
  work, peak memory, latency, and continuous-arrival backlog over the complete
  accuracy-work frontier.
- [ ] Never reward predicted entropy reduction by itself; score realized
  logical outcomes and reference approximation error.

### Task 23: distill and verify a symbolic policy

- [ ] Search a bounded rule grammar over deployable state features with hard
  syndrome-validity, work-budget, and declared symmetry constraints.
- [ ] Freeze the rule list on calibration data and compare with the learned
  controller on untouched confirmation data.
- [ ] Exhaustively verify every reachable Gate 0 state and report the rule's
  accuracy-work loss relative to the learned policy.
- [ ] Retain the symbolic policy only if it satisfies the frozen approximation
  tolerance and every hard constraint.

## Verification at every task boundary

Each task ends with focused tests, Ruff, `git diff --check`, an atomic commit,
and an independent reviewer. Scientific runs additionally require clean source
identity, immutable artifacts, exact replay, claim-level review, and a manual
approval record before opening the next role.
