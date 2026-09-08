# Anytime Logical-Coset Portfolio Implementation Plan

**Goal:** test whether FNO-derived BP views add useful logical-coset diversity
beyond fair qLDPC baselines, then build a calibrated anytime selector only if
that headroom survives.

**Design:**
`docs/superpowers/specs/2026-09-08-anytime-coset-portfolio-design.md`

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
  `random_schedule_seed`, one decoder seed per shot, and no explicit serial
  order. Characterize and freeze the package's across-iteration behavior in a
  deterministic fixture.
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
- Run the design's fixed 16-configuration search on exactly 4,096 development
  shots; select by BLER, mean iterations, then fixed parameter order.
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

## Verification at every task boundary

Each task ends with focused tests, Ruff, `git diff --check`, an atomic commit,
and an independent reviewer. Scientific runs additionally require clean source
identity, immutable artifacts, exact replay, claim-level review, and a manual
approval record before opening the next role.
