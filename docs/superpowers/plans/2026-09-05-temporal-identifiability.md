# Temporal Syndrome-History Identifiability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, replay-verify, and run the preregistered architecture-free test
of whether past qLDPC syndromes identify a changing scalar error rate.

**Architecture:** A new `identifiability` package owns a strict experiment
contract, structurally separated public and privileged inputs, an independently
reproducible scalar generator, the exact likelihood on a deterministic
row-disjoint check set, causal classical filters, and sequence-clustered
decisions. Two small CLIs generate immutable role-separated data and run or
verify the study; held-out test generation requires a manually approved,
content-bound development record and frozen estimator bundle.

**Tech Stack:** Python 3.14, NumPy, SciPy, PyTorch only for the existing fitted
baselines, canonical JSON/NPZ, `ldpc` BP-LSD, pytest, Ruff.

## Global Constraints

- The normative design is
  `docs/superpowers/specs/2026-09-05-syndrome-identifiability-design.md`.
- Use only canonical `lp(3,7)_16`, independent-Z code-capacity errors, perfect
  syndromes, and the existing fixed BP-LSD policy.
- The primary study contains only `stationary_iid` and `temporal_uniform`.
- A round-`t` deployable prediction may use syndromes only through `t-1`.
- Deployable APIs accept `DeployableHistory` only. Privileged ceilings have
  distinct exact input types and cannot enter deployable registries.
- Retained-row likelihood is exact only for the deterministic row-disjoint set;
  all-row marginal products are named composite likelihoods.
- The independent sequence is the inferential unit. Never treat rounds, qubits,
  checks, numerical grids, or filter seeds as replicates.
- Primary loss is latent expected Bernoulli cross-entropy, not sampled-error
  cross-entropy.
- Nominal/refined grids have 2,048/4,096 interior cells plus two clipping atoms;
  required mean-gain agreement is `<2.5e-5` nats/qubit/round.
- `delta_NLL=0.00025`; the test role contains 64 independent sequences per
  regime with 64 burn-in and 128 scored rounds.
- A winning arm requires `lower_95 > delta_NLL`, Holm-adjusted one-sided
  `p<=0.05`, a stationary-control upper bound `<=delta_NLL`, a matching
  derangement upper bound `<=delta_NLL`, and nondegenerate bootstrap output.
- Do not generate the test role until a reviewer approves the source commit,
  frozen config hash, immutable development record, and self-contained fitted
  estimator bundle. This is a manual approval gate, not a cryptographic trust
  boundary.
- Do not run BP-LSD unless the primary result is
  `GO-TEMPORAL-IDENTIFIED`.
- Numeric artifacts remain ignored; committed prose must distinguish local
  evidence from files available in a fresh clone.
- Enforce a six-process-CPU-hour deadline. An overrun publishes only an
  `aborted_no_verdict` record and cannot emit a scientific decision.
- Every production change follows red-green-refactor and receives independent
  review before the next task.

---

### Task 1: Define the strict contract, seeds, and scalar sequence generator

**Files:**

- Create: `src/qldpc_fno/identifiability/__init__.py`
- Create: `src/qldpc_fno/identifiability/config.py`
- Create: `src/qldpc_fno/identifiability/seeds.py`
- Create: `src/qldpc_fno/identifiability/types.py`
- Create: `src/qldpc_fno/identifiability/generator.py`
- Create: `configs/temporal_identifiability.json`
- Create: `tests/identifiability/test_config.py`
- Create: `tests/identifiability/test_generator.py`
- Create: `tests/identifiability/test_seeds.py`

**Interfaces:**

- Produces:
  `load_identifiability_config(path: Path) -> IdentifiabilityConfig`,
  `identifiability_seed(config, *, regime, role, sequence_index, stream) -> int`,
  and
  `generate_scalar_sequence(config, *, regime, role, sequence_index, code) -> GeneratedSequence`.
- Produces exact, noninterchangeable `DeployableHistory`,
  `LatentHistoryOracleInput`, `ContemporaneousOracleInput`, `TrainingTargets`,
  and `SequenceIdentity` types. Deployable functions require
  `type(value) is DeployableHistory`; a combined sequence container is confined
  to orchestration and never accepted by a forecaster.

- [ ] **Step 1: Write strict schema and seed tests.** Cover missing/unknown
  keys, exact regimes and roles, `8/8/8/64` split, `64/128` rounds,
  `0.97/0.08/1.20` dynamics, fixed grid and inference thresholds, canonical
  decoder settings, `21,600` process-CPU-second deadline, deterministic seed
  replay, and pairwise stream separation.
  Freeze the 4,096-cell open-loop comparator, Fisher stationary-normal draw law,
  `1e-6` finite-difference step, `1e-8/1e-6` tolerances, NMSE formula, and 10
  calibration bins exactly as written in the spec.
  Assert the four exact campaign/bootstrap/derangement/Fisher seed values and
  their SHA-256 domain-string derivation from the normative spec.
  The baseline policy is fully explicit: empirical stationary shrinkage fixed
  to `[1.0]` with no validation selection; EWMA decays
  `[.5,.8,.9,.97,.99]`, kernel 5; logistic
  lags 32, kernel 3; L2 `[1e-4,1e-3,1e-2,1e-1]`; LBFGS maximum 500; sorted-grid
  first-minimum tie rule with `1e-12` tolerance; identity calibration and
  probability clipping `[1e-5,.25]`; canonical arm aliases.

- [ ] **Step 2: Run the schema tests and observe the missing-module failure.**

  ```bash
  uv run pytest tests/identifiability/test_config.py \
    tests/identifiability/test_seeds.py -q
  ```

  Expected: collection fails because `qldpc_fno.identifiability` does not exist.

- [ ] **Step 3: Implement only the strict config dataclasses and seed
  derivation.** Do not create `types.py` or generator containers yet. Reject
  booleans as integers, nonfinite floats, mutable regime lists, noncanonical
  code metadata, altered BP-LSD settings, and any artifact label other than
  `confirmatory_gate`. The public loader accepts canonical scientific sizes
  only. Tests inject fake kernels around a canonical config; no reduced mode is
  selectable by a production CLI. Development versus confirmation is a CLI
  stage, not a mutable scientific-config field.

- [ ] **Step 4: Write type-boundary and generator tests before implementation.**
  Require the public and two privileged containers to have disjoint fields and
  be noninterchangeable by exact type checks. Require a
  `DevelopmentPartitions` constructor to accept only content-bound,
  pairwise-disjoint train/validation/calibration identities and reject any test
  identity. Independently
  reconstruct `q`, `Hx @ error mod 2`, and logical flips. Require `g[0]=0`, a
  constant stationary latent, identical probabilities across all 2,610 qubits,
  separate latent/Bernoulli streams, exact scored masks, immutable arrays, and
  byte-identical regeneration.

- [ ] **Step 5: Run the generator tests and observe failures for missing
  behavior.**

  ```bash
  uv run pytest tests/identifiability/test_generator.py -q
  ```

- [ ] **Step 6: Implement the exact input/container types,
  `DevelopmentPartitions`, and minimal scalar generator.** Copy no event or
  spatial logic. Use the same transition order as the existing temporal
  generator and validate canonical `Hx`, `Hz`, and logical-X identities.

- [ ] **Step 7: Prove equation-level equivalence to the existing generator.** A
  test-local reference loop consumes an explicitly seeded NumPy generator and
  independently applies the existing inspect/transition order. Assert exact
  latent and probability equality for both regimes before independently
  reconstructing syndromes and logical flips. The production generator must not
  call the old generator or the test reference.

- [ ] **Step 8: Run focused tests and lint.**

  ```bash
  uv run pytest tests/identifiability/test_config.py \
    tests/identifiability/test_seeds.py \
    tests/identifiability/test_generator.py -q
  uv run ruff check src/qldpc_fno/identifiability \
    tests/identifiability
  ```

- [ ] **Step 9: Commit.**

  ```bash
  git add configs/temporal_identifiability.json \
    src/qldpc_fno/identifiability tests/identifiability
  git commit -m "feat: define temporal identifiability data"
  ```

- [ ] **Step 10: Independent review.** Audit causal timing, generator
  equivalence, seed separation, code identity, role sizes, and privileged-field
  boundaries before Task 2.

---

### Task 2: Implement the exact retained-row likelihood and Fisher precheck

**Files:**

- Create: `src/qldpc_fno/identifiability/observation.py`
- Create: `tools/generate_scalar_likelihood_fixture.py`
- Create: `tests/identifiability/fixtures/scalar_likelihood_float64.json`
- Create: `tests/identifiability/test_observation.py`

**Interfaces:**

- Consumes: canonical CSR `Hx` and scalar/vector physical probabilities.
- Produces:
  `greedy_disjoint_rows(hx) -> DisjointChecks`,
  `parity_one_probability(q, weight) -> ndarray`,
  `retained_log_likelihood(syndrome, state_grid, checks, config) -> ndarray`,
  `scalar_fisher_information(state, checks, config) -> float`, and
  `run_fisher_precheck(config, checks) -> FisherReport`.

- [ ] **Step 1: Write failing row-selection tests.** Require ascending greedy
  traversal, no shared qubit in retained supports, complete rejection of a
  deliberately overlapping toy row, stable CSR canonicalization, and canonical
  values of 135 rows, weight 10, and 1,350 covered qubits.

- [ ] **Step 2: Run the row tests and verify the expected missing-symbol
  failures.**

  ```bash
  uv run pytest tests/identifiability/test_observation.py \
    -k 'disjoint or canonical' -q
  ```

- [ ] **Step 3: Implement `DisjointChecks` and greedy selection.** Store
  read-only row indices, CSR supports, weights, covered qubits, algorithm
  version, and the canonical sparse-matrix hash.

- [ ] **Step 4: Write failing likelihood tests.** Enumerate all physical-error
  strings for toy weight-2 and weight-3 checks and compare the parity formula.
  Compare retained log likelihood with direct products, test `logaddexp`
  stability at probability bounds, and prove that the API rejects overlapping
  rows when `likelihood_kind="exact_disjoint"`.

- [ ] **Step 5: Generate an independent float64 fixture.** The tool implements
  the formula directly, does not import `qldpc_fno.identifiability.observation`,
  and stores parity probabilities, likelihoods, analytic derivatives, finite
  differences, and Fisher values at fixed states.

- [ ] **Step 6: Implement likelihood and analytic Fisher functions, then run
  the fixture and property tests.**

  ```bash
  uv run python tools/generate_scalar_likelihood_fixture.py --check
  uv run pytest tests/identifiability/test_observation.py -q
  ```

- [ ] **Step 7: Write failing Fisher-gate tests.** Require exactly 10,000
  identity-seeded prior draws, recorded draw provenance, finite positive
  information, derivative agreement at the configured absolute/relative
  tolerances, min/median/max information and Cramer-Rao output, and a hard
  `precheck_failed` result for nonfinite, nonpositive, or tolerance-violating
  fixtures. Return a typed failed result that later generation code must consume.

- [ ] **Step 8: Implement the deterministic Fisher precheck and pass its tests.**

- [ ] **Step 9: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest tests/identifiability/test_observation.py -q
  uv run ruff check src/qldpc_fno/identifiability/observation.py \
    tests/identifiability/test_observation.py \
    tools/generate_scalar_likelihood_fixture.py
  git add src/qldpc_fno/identifiability/observation.py \
    tools/generate_scalar_likelihood_fixture.py \
    tests/identifiability/fixtures/scalar_likelihood_float64.json \
    tests/identifiability/test_observation.py
  git commit -m "feat: add exact scalar syndrome likelihood"
  ```

- [ ] **Step 10: Independent review.** A numerical/QEC reviewer checks row
  independence, formula derivation, finite-difference independence, and the
  exact-versus-composite terminology.

---

### Task 3: Implement and validate the clipped AR grid transition

**Files:**

- Create: `src/qldpc_fno/identifiability/grid.py`
- Create: `tools/generate_clipped_grid_fixture.py`
- Create: `tests/identifiability/fixtures/clipped_grid_float64.json`
- Create: `tests/identifiability/test_grid.py`

**Interfaces:**

- Consumes: `IdentifiabilityConfig`.
- Produces:
  `build_clipped_ar_grid(config, *, interior_cells) -> ClippedARGrid`,
  `transition_distribution(grid, posterior) -> ndarray`, and
  `integrate_probability(grid, distribution) -> float`.

- [ ] **Step 1: Write failing grid-construction tests.** Require 2,048 interior
  midpoints plus two separate boundary atoms, exact cell edges, nonnegative
  normalized CDF-difference transitions, explicit clipping-tail mass, and an
  exact round-zero point mass at `g=0`.

- [ ] **Step 2: Run the grid tests and observe missing-symbol failures.**

  ```bash
  uv run pytest tests/identifiability/test_grid.py \
    -k 'grid or boundary or transition' -q
  ```

- [ ] **Step 3: Create an independent small-grid fixture.** Use direct SciPy
  Gaussian CDF differences and exhaustive matrix propagation for 8 and 16
  interior cells. The fixture script must not import production filters.

- [ ] **Step 4: Write failing propagation tests before production code.** Match
  the independent 8/16-cell fixture, require mass conservation, nonnegative
  output, correct left/right atom mass, deterministic repeated calls, and an
  enforced process-CPU deadline that raises `ExperimentDeadlineExceeded`.

- [ ] **Step 5: Implement the minimal stable transition.** Match the direct
  fixture before optimizing. Any matrix-free optimization must retain a direct
  small-grid comparison and a configured absolute mass/error tolerance. Stream
  batches rather than materializing all arm predictions.

- [ ] **Step 6: Run focused tests and lint, then commit.**

  ```bash
  uv run python tools/generate_clipped_grid_fixture.py --check
  uv run pytest tests/identifiability/test_grid.py -q
  uv run ruff check src/qldpc_fno/identifiability/grid.py \
    tests/identifiability/test_grid.py \
    tools/generate_clipped_grid_fixture.py
  git add src/qldpc_fno/identifiability/grid.py \
    tools/generate_clipped_grid_fixture.py \
    tests/identifiability/fixtures/clipped_grid_float64.json \
    tests/identifiability/test_grid.py
  git commit -m "feat: add clipped ar grid transition"
  ```

- [ ] **Step 7: Independent review.** A numerical reviewer audits the clipped
  transition, fixture independence, stability, deadline, memory complexity, and
  feasibility at 2,048/4,096 cells.

---

### Task 4: Implement typed causal filters and privileged ceilings

**Files:**

- Create: `src/qldpc_fno/identifiability/filters.py`
- Create: `tests/identifiability/test_filters.py`

**Interfaces:**

- Deployable filters accept only `DeployableHistory` and `DisjointChecks`.
- Privileged functions separately accept `LatentHistoryOracleInput` or
  `ContemporaneousOracleInput`.
- Produces `ForecastResult` plus
  `forecast_known_marginal`, `forecast_parity_moment`, `forecast_grid_bayes`,
  `forecast_latent_history`, and `forecast_contemporaneous`.

- [ ] **Step 1: Write all filter behavior tests before implementation.** Cover
  no-observation equivalence to the spec's fixed 4,096-cell open-loop
  `known_marginal`, the constrained parity-moment MLE at both latent boundaries
  and for fractions above `0.5`, small-grid Bayes updates, exact latent-history transition
  integration, contemporaneous identity, finite bounded probabilities,
  batch-versus-separate equivalence, and type rejection for every privileged or
  combined input passed to a deployable arm.

- [ ] **Step 2: Write causal timing tests before implementation.** Mutating the
  current/future syndrome, error, latent, or logical label must leave `q_hat_t`
  bit-identical. Mutating `syndrome_(t-1)` must be able to change it. Resetting a
  sequence identity clears posterior state.

- [ ] **Step 3: Write convergence and streaming tests before implementation.**
  Require `abs(mean_G_2048-mean_G_4096)`, refuse nonfinite/degenerate output,
  and assert sequence-at-a-time evaluation does not retain full predictions for
  all arms and derangements simultaneously.

- [ ] **Step 4: Run the tests and verify failures are caused by missing filter
  behavior.**

  ```bash
  uv run pytest tests/identifiability/test_filters.py -q
  ```

- [ ] **Step 5: Implement the minimal filters in forecast-then-update order.**
  Use log-space observation updates, the reviewed grid transition, exact input
  types, and streaming per-sequence outputs.

- [ ] **Step 6: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest tests/identifiability/test_grid.py \
    tests/identifiability/test_filters.py -q
  uv run ruff check src/qldpc_fno/identifiability/filters.py \
    tests/identifiability/test_filters.py
  git add src/qldpc_fno/identifiability/filters.py \
    tests/identifiability/test_filters.py
  git commit -m "feat: add typed causal scalar filters"
  ```

- [ ] **Step 7: Independent review.** Audit forecast/update timing, input
  privilege, state resets, grid use, convergence computation, and streaming
  memory behavior.

---

### Task 5: Freeze fitted baselines and compute all per-sequence endpoints

**Files:**

- Create: `src/qldpc_fno/identifiability/baseline_bundle.py`
- Create: `src/qldpc_fno/identifiability/endpoints.py`
- Create: `tests/identifiability/test_baseline_bundle.py`
- Create: `tests/identifiability/test_endpoints.py`

**Interfaces:**

- Consumes: `DevelopmentPartitions`, the exact baseline policy, and streaming
  `ForecastResult` objects.
- Produces:
  `fit_development_bundle(partitions, config) -> FrozenEstimatorBundle`,
  `write_frozen_bundle(path, bundle) -> BundleManifest`,
  `read_verified_bundle(path, manifest) -> FrozenEstimatorBundle`,
  `expected_ce_by_sequence(q, q_hat, mask) -> ndarray`,
  `latent_nmse_by_sequence(...) -> ndarray`,
  `retained_syndrome_nll_by_sequence(...) -> ndarray`, and
  `calibration_by_sequence(...) -> CalibrationEvidence`.

- [ ] **Step 1: Write failing partition and baseline-policy tests.** Require
  content-bound pairwise-disjoint train/validation/calibration identities,
  reject test-role content, assert every candidate grid/kernel/iteration/tie
  rule from config is passed to the existing fitters, and require identity
  calibration for every arm. Burn-in syndromes may update state, but only scored
  train/validation/calibration rounds may enter their respective losses.

- [ ] **Step 2: Write failing frozen-bundle tests.** Require safe non-pickle
  arrays, canonical arm aliases, all fitted weights/biases/settings, role
  identities and content hashes, policy hash, deterministic byte replay, and
  rejection of missing, renamed, rehashed, or tampered parameters.

- [ ] **Step 3: Run bundle tests and observe missing behavior.**

  ```bash
  uv run pytest tests/identifiability/test_baseline_bundle.py -q
  ```

- [ ] **Step 4: Implement fit, serialization, and exact reload.** The confirmation
  path receives the bundle and never refits against test data. Replay regenerates
  development roles, refits with deterministic Torch settings, and requires
  byte-identical safe arrays plus identical selected settings.

- [ ] **Step 5: Write endpoint tests before implementation.** Compare latent
  expected CE, latent-state normalized MSE, retained-syndrome predictive NLL,
  and calibration-bin count/sum/error against hand calculations. Require one
  value per sequence, prove sampled physical errors cannot affect latent CE,
  and reject mixed sequence identities or incomplete scored masks. Use the
  exact NMSE mapping and stationary-variance denominator plus the 10 fixed
  `[1e-5,0.25]` bin edges and closure rules from the spec.

- [ ] **Step 6: Implement streaming endpoint accumulators.** Persist only
  per-sequence aggregates and the one currently evaluated prediction stream;
  do not retain all normal and deranged arm tensors simultaneously.

- [ ] **Step 7: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest tests/identifiability/test_baseline_bundle.py \
    tests/identifiability/test_endpoints.py -q
  uv run ruff check src/qldpc_fno/identifiability/baseline_bundle.py \
    src/qldpc_fno/identifiability/endpoints.py \
    tests/identifiability/test_baseline_bundle.py \
    tests/identifiability/test_endpoints.py
  git add src/qldpc_fno/identifiability/baseline_bundle.py \
    src/qldpc_fno/identifiability/endpoints.py \
    tests/identifiability/test_baseline_bundle.py \
    tests/identifiability/test_endpoints.py
  git commit -m "feat: freeze temporal estimator evidence"
  ```

- [ ] **Step 8: Independent review.** Audit role exclusion, exact baseline
  policy, deterministic fit replay, bundle completeness, latent rather than
  sampled-error scoring, secondary endpoints, and bounded memory.

---

### Task 6: Add derangement, clustered inference, and gate decisions

**Files:**

- Create: `src/qldpc_fno/identifiability/inference.py`
- Create: `tests/identifiability/test_inference.py`
- Modify: `src/qldpc_fno/metrics/clustered.py`
- Modify: `tests/metrics/test_clustered.py`

**Interfaces:**

- Produces `fixed_history_derangement`,
  `studentized_sequence_interval`, `holm_adjust`,
  `evaluate_identifiability`, `decide_temporal_gate`, and
  `classify_bler_interval`.

- [ ] **Step 1: Write failing derangement tests.** Require a true permutation
  with no fixed points, deterministic replay, unchanged targets, history-only
  replacement, and a separate control for each syndrome-history arm.

- [ ] **Step 2: Write failing bootstrap tests.** Fix the bootstrap-t statistic,
  centering under `delta_NLL` for gain tests, one-sided lower/upper and two-sided
  intervals, exactly 10,000 whole-sequence draws, deterministic seeds,
  degenerate output, and rejection of fewer than 64 confirmatory sequences.

- [ ] **Step 3: Write failing Holm tests.** Cover ordering and ties for the exact
  four-arm family and require adjusted `p<=0.05` in addition to
  `lower_95>delta_NLL`.

- [ ] **Step 4: Write table-driven gate tests before implementation.** Cover no
  causal ceiling, observer gap, each winning arm, control failure, bootstrap
  degeneracy, convergence failure, current-baseline limitation,
  reduced-screen/sample limitation, and exact `GO-TEMPORAL-IDENTIFIED`
  prerequisites.

- [ ] **Step 5: Write BLER boundary and pure arm-policy tests.** Test exact
  endpoints at `-0.01/+0.01`; `conditional_decoder_arms(decision)` returns
  `not_run_by_design` for non-GO evidence and exactly the comparator, winning
  arms, and oracle for GO. Task 8 tests actual decoder-factory non-invocation.

- [ ] **Step 6: Run all tests and verify expected missing-symbol failures.**

  ```bash
  uv run pytest tests/metrics/test_clustered.py \
    tests/identifiability/test_inference.py -q
  ```

- [ ] **Step 7: Implement derangement, bootstrap-t, Holm, and decision logic.**
  Never replace degenerate inference with qubit- or round-level samples.

- [ ] **Step 8: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest tests/metrics/test_clustered.py \
    tests/identifiability/test_inference.py -q
  uv run ruff check src/qldpc_fno/metrics/clustered.py \
    src/qldpc_fno/identifiability/inference.py \
    tests/metrics/test_clustered.py tests/identifiability/test_inference.py
  git add src/qldpc_fno/metrics/clustered.py \
    src/qldpc_fno/identifiability/inference.py \
    tests/metrics/test_clustered.py tests/identifiability/test_inference.py
  git commit -m "feat: decide temporal identifiability gate"
  ```

- [ ] **Step 9: Independent review.** A statistical reviewer checks sampling
  units, bootstrap construction and tails, multiplicity in the actual GO path,
  effect thresholds, controls, and nested NLL/BLER outcomes.

---

### Task 7: Publish and verify immutable role-separated sequences

**Files:**

- Create: `src/qldpc_fno/identifiability/sequence_store.py`
- Create: `experiments/21_generate_temporal_identifiability.py`
- Create: `tests/integration/test_temporal_identifiability_generation_cli.py`

**Interfaces:**

- Produces `21...py generate|verify --config PATH --out PATH --roles ...`.
- Test-role generation additionally requires `--approval PATH` and
  `--development-record PATH`, whose hashes and identities must agree.

- [ ] **Step 1: Write failing manifest-schema tests.** Enumerate mandatory code,
  source, config, role, sequence, seed-stream, retained-support, array
  shape/dtype, content-hash, and completion fields. Reject every single missing,
  renamed, rehashed, extra, or tampered field and any role overlap.

- [ ] **Step 2: Write failing in-process CLI tests.** Use the canonical config
  with dependency-injected fast fake kernels; production argument parsing has no
  reduced mode. Exercise generation, deterministic regeneration, refusal to
  overwrite, byte corruption, partial publication, dirty-source test
  generation, missing approval/development record, and every bound-hash
  mismatch. A failed or missing Fisher precheck in the bound development record
  must prevent test-role generation.

- [ ] **Step 3: Run the CLI tests and observe missing-command failures.**

  ```bash
  uv run pytest \
    tests/integration/test_temporal_identifiability_generation_cli.py -q
  ```

- [ ] **Step 4: Implement atomic role-separated sequence publication.** Write
  payloads to temporary paths, hash them, atomically rename them, and publish
  the completion manifest last. Verification independently regenerates every
  requested sequence.

- [ ] **Step 5: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest \
    tests/integration/test_temporal_identifiability_generation_cli.py \
    tests/identifiability/test_generator.py -q
  uv run ruff check src/qldpc_fno/identifiability/sequence_store.py \
    experiments/21_generate_temporal_identifiability.py \
    tests/integration/test_temporal_identifiability_generation_cli.py
  git add src/qldpc_fno/identifiability/sequence_store.py \
    experiments/21_generate_temporal_identifiability.py \
    tests/integration/test_temporal_identifiability_generation_cli.py
  git commit -m "feat: publish temporal identifiability sequences"
  ```

- [ ] **Step 6: Independent review.** Audit manifest completeness, atomicity,
  regeneration, role/seed separation, manual-approval bindings, and the absence
  of a production reduced-mode escape hatch.

---

### Task 8: Execute, conditionally decode, and replay the study

**Files:**

- Create: `src/qldpc_fno/identifiability/screen.py`
- Create: `experiments/22_run_temporal_identifiability.py`
- Create: `tests/integration/test_temporal_identifiability_run_cli.py`

**Interfaces:**

- Produces `22...py development|confirmation|verify --config PATH
  --sequences PATH --out PATH [--development-record PATH --approval PATH]`.
- Development publishes the frozen estimator bundle and evidence record.
- Confirmation must verify and load that bundle before opening test payloads.

- [ ] **Step 1: Write failing development integration tests.** Require Fisher
  pass before evaluation, exact fitted bundle publication, all normal and
  per-arm deranged predictions/states, grid diagnostics, every per-sequence
  primary/secondary endpoint, inference metadata, source/config/code/support
  hashes, process CPU and wall time, and
  `engineering_measurement_no_speed_claim`.

- [ ] **Step 2: Write failing confirmation-firewall tests.** Prove the command
  validates source, config, code, development record, bundle, and manual
  approval before opening test data. Replay must regenerate development roles,
  refit, and compare bundle arrays/settings before it reads confirmation output.

- [ ] **Step 3: Write failing conditional-decoder tests.** A non-GO decision
  must construct no decoder and record `not_run_by_design`. A GO runs exactly
  `known_marginal`, every primary-passing syndrome arm, and
  `contemporaneous_oracle` on identical test sequence/round identities. Verify
  every correction syndrome, reconstruct logical failure modulo stabilizers,
  and compute paired sequence BLER intervals.

- [ ] **Step 4: Write failing deadline and abort tests.** Exceeding six process
  CPU hours through an injected monotonic CPU clock must stop work, omit a
  verdict, and atomically publish `aborted_no_verdict` with completed-stage
  hashes. Resume is forbidden unless the exact partial-state policy is later
  separately specified.

- [ ] **Step 5: Run tests and observe missing-command failures.**

  ```bash
  uv run pytest tests/integration/test_temporal_identifiability_run_cli.py -q
  ```

- [ ] **Step 6: Implement development, confirmation, and replay paths.** Stream
  sequence/arm evaluation, load the safe frozen bundle for confirmation, enforce
  all firewalls before reading test content, and publish the strict completion
  manifest last.

- [ ] **Step 7: Independently recompute all evidence in verify mode.** Recreate
  sequences, fits, filters, controls, endpoints, bootstrap/Holm decisions, and
  any QEC results; reject numeric or hash differences.

- [ ] **Step 8: Run focused tests and lint, then commit.**

  ```bash
  uv run pytest tests/integration/test_temporal_identifiability_run_cli.py \
    tests/identifiability tests/metrics/test_clustered.py -q
  uv run ruff check src/qldpc_fno/identifiability \
    experiments/22_run_temporal_identifiability.py \
    tests/integration/test_temporal_identifiability_run_cli.py
  git diff --check
  git add src/qldpc_fno/identifiability/screen.py \
    experiments/22_run_temporal_identifiability.py \
    tests/integration/test_temporal_identifiability_run_cli.py
  git commit -m "feat: orchestrate temporal identifiability gate"
  ```

- [ ] **Step 9: Broad adversarial review.** Review all Tasks 1-8 using the QEC
  rubric: code identity, common sequences, causal information symmetry,
  privileged types, exact likelihood, frozen fits, split isolation,
  sequence-level inference, multiplicity, conditional BP-LSD, deadline,
  manifest schema, replay completeness, and claim labels.

---

### Task 9: Run and review the development-only gate

**Files:**

- Local ignored output: `artifacts/temporal-identifiability-development/`
- Local ignored approval: `artifacts/temporal-identifiability-development/APPROVAL.json`
- Local frozen bundle:
  `artifacts/temporal-identifiability-development/run/estimator-bundle/`
- Modify only if evidence requires a bug fix: Task 1-8 files and their tests.

**Interfaces:**

- Consumes the exact committed source and config; produces no confirmatory
  result and cannot unlock itself.

- [ ] **Step 1: Verify the repository before execution.**

  ```bash
  uv run pytest -q
  uv run ruff check .
  git diff --check
  test -z "$(git status --porcelain)"
  ```

- [ ] **Step 2: Generate only train, validation, and calibration roles.**

  ```bash
  uv run python experiments/21_generate_temporal_identifiability.py generate \
    --config configs/temporal_identifiability.json \
    --out artifacts/temporal-identifiability-development/sequences \
    --roles train validation calibration
  ```

- [ ] **Step 3: Execute and independently replay the development run.**

  ```bash
  uv run python experiments/22_run_temporal_identifiability.py development \
    --config configs/temporal_identifiability.json \
    --sequences artifacts/temporal-identifiability-development/sequences \
    --out artifacts/temporal-identifiability-development/run
  uv run python experiments/22_run_temporal_identifiability.py verify \
    --config configs/temporal_identifiability.json \
    --sequences artifacts/temporal-identifiability-development/sequences \
    --out artifacts/temporal-identifiability-development/run
  ```

- [ ] **Step 4: Review numerical and runtime gates.** A fresh reviewer inspects
  raw manifests and logs, Fisher checks, grid normalization, 2,048/4,096 gain
  agreement, derangement behavior, baseline settings, causal audits, replay,
  peak memory, and projected confirmation duration.

- [ ] **Step 5: Resolve review findings test-first.** Any source fix invalidates
  the development artifact. Commit the fix, delete only the exact ignored
  development directory, rerun Steps 1-4, and never alter scientific thresholds
  in response to performance.

- [ ] **Step 6: Obtain a manual approval record.** The reviewer, not the run
  command, writes `APPROVAL.json` containing `APPROVE` or `REJECT`, reviewer
  identity, and every bound source/config/code/support/development/bundle hash.
  This is an auditable workflow gate, not a cryptographic signature. Only
  `APPROVE` permits Task 10.

---

### Task 10: Freeze, confirm, document, and publish

**Files:**

- Local ignored output: `artifacts/temporal-identifiability-confirmation/`
- Modify: `README.md`
- Create: `docs/temporal-identifiability-results.md`
- Test: `tests/integration/test_temporal_identifiability_run_cli.py`

**Interfaces:**

- Consumes the approved development record and untouched test identities.
- Produces a replay-verified confirmatory result with one of the exact gate
  decisions and an optional conditional BP-LSD diagnostic.

- [ ] **Step 1: Generate the test role only after approval.**

  ```bash
  uv run python experiments/21_generate_temporal_identifiability.py generate \
    --config configs/temporal_identifiability.json \
    --out artifacts/temporal-identifiability-confirmation/sequences \
    --roles test \
    --approval artifacts/temporal-identifiability-development/APPROVAL.json \
    --development-record artifacts/temporal-identifiability-development/run
  ```

- [ ] **Step 2: Run confirmation once.**

  ```bash
  uv run python experiments/22_run_temporal_identifiability.py confirmation \
    --config configs/temporal_identifiability.json \
    --sequences artifacts/temporal-identifiability-confirmation/sequences \
    --out artifacts/temporal-identifiability-confirmation/run \
    --approval artifacts/temporal-identifiability-development/APPROVAL.json \
    --development-record artifacts/temporal-identifiability-development/run
  ```

- [ ] **Step 3: Replay independently before interpretation.**

  ```bash
  uv run python experiments/22_run_temporal_identifiability.py verify \
    --config configs/temporal_identifiability.json \
    --sequences artifacts/temporal-identifiability-confirmation/sequences \
    --out artifacts/temporal-identifiability-confirmation/run \
    --approval artifacts/temporal-identifiability-development/APPROVAL.json \
    --development-record artifacts/temporal-identifiability-development/run
  ```

- [ ] **Step 4: Obtain a claim-level QEC/statistics review.** Review every
  empirical sentence against config, logs, per-sequence values, intervals,
  adjusted tests, syndrome validity, and the conditional decoder rule. Do not
  interpret before this review returns `APPROVE`.

- [ ] **Step 5: Write public documentation from approved evidence.** Explain the
  scientific question in plain technical English, report all controls and
  uncertainty, distinguish causal and contemporaneous ceilings, state the exact
  decision, scope any BLER result to global-rate variation, and disclose that
  numeric artifacts are local/ignored.

- [ ] **Step 6: Verify documentation and repository.**

  ```bash
  uv run pytest -q
  uv run ruff check .
  git diff --check
  git status --short
  ```

- [ ] **Step 7: Commit and push only reviewed claims.**

  ```bash
  git add README.md docs/temporal-identifiability-results.md
  git commit -m "docs: report temporal identifiability result"
  git push origin main
  ```

- [ ] **Step 8: Decide the next experiment from the frozen outcome.** A `GO`
  opens a separately reviewed spatial identifiability design. A stop or
  inconclusive result opens only the remedy named by the preregistered decision;
  it does not trigger an unplanned neural sweep or a second test seed.
