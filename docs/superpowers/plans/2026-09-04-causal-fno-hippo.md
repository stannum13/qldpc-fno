# Causal FNO-HiPPO Gate 1-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `subagent-driven-development` task by task. Each production change follows
> red-green-refactor and receives an independent review before the next gate.

**Goal:** Build and run a reduced, causality-audited 2×2 experiment that detects
implementation failure or absence of a reduced descriptive signal before any
expensive discovery campaign. Only frozen confirmation can statistically
support or falsify H2/H3.

**Architecture:** Generate immutable repeated code-capacity sequences for the
existing `lp(3,7)_16` geometry. A forecaster predicts the next round's 2,610
physical error probabilities from completed syndromes only. Cross circular CNN
versus FNO spatial encoders with finite-window FIR versus faithful HiPPO-LegS
temporal operators, compare them with stationary/EWMA/logistic causal baselines,
and pass every predicted prior through the same fixed BP-LSD implementation.

**Tech stack:** Python 3.14, NumPy, SciPy, PyTorch, Stim-compatible code
identities, `ldpc`, pytest, Ruff, canonical JSON/NPZ artifacts.

## Global constraints

- The normative design is
  `docs/superpowers/specs/2026-09-04-causal-fno-hippo-design.md`.
- Preserve the existing fixed-shot campaign and its public APIs.
- Use errors only as supervised training targets and evaluation labels; they are
  never deployable inputs.
- Predict before updating state with the current syndrome.
- Reset state at sequence boundaries; update through burn-in; score only later
  rounds.
- Compare all decoder arms on identical errors, syndromes, and logical flips.
- Count decoder failure modulo the existing logical operators, never by raw
  correction equality.
- Gate-2 output is `reduced_non_scientific` and cannot support a scientific
  claim.
- Do not launch Gate 3 unless the reduced headroom and overfit gates pass and a
  reviewer approves the artifacts.

---

### Task 1: Add strict causal experiment configuration and seed identities

**Files:**

- Create: `src/qldpc_fno/temporal/__init__.py`
- Create: `src/qldpc_fno/temporal/config.py`
- Create: `src/qldpc_fno/temporal/seeds.py`
- Create: `configs/causal_fno_hippo_reduced.json`
- Create: `tests/temporal/test_config.py`
- Create: `tests/temporal/test_seeds.py`

**Contract:**

- Strict dataclasses represent code identity, five named regimes, split sizes,
  burn-in/scored rounds, generator parameters, model widths/modes/order, decoder
  parameters, optimizer settings, and artifact mode.
- Unknown/missing fields and invalid probabilities, ranges, or role names fail.
- `reduced_non_scientific` is mandatory in the reduced config.
- Seed tuples derive independently from campaign seed, regime, role, sequence,
  and stream (`latent` or `bernoulli`) and are pairwise disjoint.

- [ ] Write failing schema, round-trip, invalid-value, and seed-disjointness tests.
- [ ] Run `uv run pytest tests/temporal/test_config.py tests/temporal/test_seeds.py -q`
  and record the expected import failure.
- [ ] Implement the minimum strict parser and deterministic seed derivation.
- [ ] Run the focused tests and `uv run ruff check` on the new files.
- [ ] Commit as `feat: define causal experiment contract`.
- [ ] Independent reviewer checks claim labels, strictness, and split isolation.

---

### Task 2: Implement exact temporal regimes and reproducible sequence artifacts

**Files:**

- Create: `src/qldpc_fno/temporal/generator.py`
- Create: `src/qldpc_fno/temporal/dataset.py`
- Create: `tests/temporal/test_generator.py`
- Create: `tests/temporal/test_dataset.py`

**Interfaces:**

```python
generate_latent_sequence(config, *, regime, role, sequence_index) -> LatentSequence
sample_sequence(latent, *, bernoulli_seed, code) -> tuple[
    CausalObservedSequence, CausalSupervision, SimulatorDiagnostics
]
write_sequence(path, observed, supervision, diagnostics, manifest) -> None
read_verified_sequence(path) -> tuple[
    CausalObservedSequence, CausalSupervision, SimulatorDiagnostics, dict[str, object]
]
regenerate_and_verify(path, config, code) -> None
```

The generator returns three immutable, physically separate types:

```python
CausalObservedSequence  # syndromes, allowed static metadata, scored mask
CausalSupervision       # physical error targets and logical labels
SimulatorDiagnostics    # latent q and event labels
```

Forecasters may accept only `CausalObservedSequence` or an explicitly extracted
syndrome tensor/state. Training receives supervision separately. The manifest
binds the strict config hash, code-matrix hash, seed tuple, every payload's
shape/dtype, and hashes.

- [ ] Write failing golden tests for all five regimes, including zero/nonzero
  spatial variance and Regime D/E onset/final/post-event boundaries.
- [ ] Add tests that independently reconstruct syndrome and logical flips from
  the existing matrices and reject one-byte artifact corruption.
- [ ] Implement regimes exactly in the design's log-odds parameterization with
  distinct latent and Bernoulli RNGs.
- [ ] Implement channel-major `(channels,45)` mapping and role-separated batch
  generation.
- [ ] Publish payloads to temporary paths, hash and atomically rename them, then
  write the completion manifest last. Refuse to overwrite a completed artifact;
  reject payload-only, interrupted, or manifest-mismatched directories.
- [ ] Verify direct regeneration is byte-identical.
- [ ] Commit as `feat: generate causal qldpc sequences`.
- [ ] Independent reviewer audits transition order, code identity, and forbidden
  information fields.

---

### Task 3: Implement and independently validate HiPPO memory primitives

**Files:**

- Create: `src/qldpc_fno/models/hippo.py`
- Create: `tools/generate_hippo_legs_fixture.py`
- Create: `tests/models/fixtures/hippo_legs_float64.json`
- Create: `tests/models/test_hippo.py`

**Interfaces:**

```python
legs_transition(order: int, step: int, *, dtype) -> tuple[Tensor, Tensor]
class HiPPOLegSMemory(nn.Module): ...
```

- [ ] Write failing tests for the signed lower-triangular LegS generator and the
  bilinear `Abar/Bbar` update at several completed-sample indices.
- [ ] Produce committed golden numbers from a standalone float64 test script,
  not by calling production code; test that the script does not import
  `qldpc_fno.models.hippo`.
- [ ] Test recurrence agreement step-by-step and shifted-Legendre projection
  convergence as separate properties.
- [ ] Test finite states and gradients for orders 8/16/32 over 4,096 steps.
- [ ] Implement batched `(B,H,ell,N)` state updates without materializing a
  position-specific transition.
- [ ] Run focused tests, then commit as `feat: add faithful hippo memory`.
- [ ] Independent literature reviewer checks equations, sign, timing, and the
  scope of the word “faithful.”
- [ ] Keep LegT out of Gate 1-2; it requires a separate reviewed numerical
  contract before any Gate-3 secondary ablation.

---

### Task 4: Add sequence-clustered inference and causal audit utilities

**Files:**

- Create: `src/qldpc_fno/metrics/clustered.py`
- Create: `src/qldpc_fno/temporal/causality.py`
- Create: `tests/metrics/test_clustered.py`
- Create: `tests/temporal/test_causality.py`

**Interfaces:**

```python
paired_sequence_inference(differences, *, mu0, seeds, draws) -> dict[str, object]
cluster_percentile_interval(values, *, seed, draws) -> dict[str, object]
audit_structural_prefix_causality(forecaster, sequence, forbidden_mutations) -> Audit
```

- [ ] Test lower-tail paired t inference, the explicitly centered wild
  bootstrap-t, deterministic whole-sequence bootstrap intervals, and declared
  `degenerate_variance`/`bootstrap_degenerate` outcomes.
- [ ] Test that round duplication changes descriptive counts but not the number
  of inferential units.
- [ ] Test mutation of current/future syndromes, errors, logical outcomes, and
  diagnostics leaves `q_t` bit-identical for fixed past history.
- [ ] Implement the minimum statistics and mutation-audit APIs.
- [ ] Treat this utility as a structural prefix audit only: it cannot rule out
  privileged state captured outside its input. Task 5 must recreate/reset each
  concrete forecaster from identical weights and spy on the actual prediction
  path to close that integration gap.
- [ ] Commit as `feat: add causal and clustered audits`.
- [ ] Independent statistical reviewer checks sampling units and tails.

---

### Task 4b: Clear the combined Gate-1 implementation

**Files:**

- Create: `artifacts/causal-gate-1-review/READY.json` only after approval

- [ ] Run all Task 1-4 focused tests, `uv run pytest -q`, `uv run ruff check .`,
  and `git diff --check`.
- [ ] Directly regenerate every golden generator fixture.
- [ ] Run a concrete mutation audit against a deliberately stateful test
  forecaster.
- [ ] Obtain independent recurrence/literature and clustered-statistics reviews.
- [ ] Have one adversarial reviewer inspect the combined generator, recurrence,
  causality, and inferential contract and return `READY` or `NOT_READY`.
- [ ] Record reviewer identity, source commit, commands, outputs, and hashes in
  the Gate-1 artifact. Do not begin Task 5 unless status is `READY`.

---

### Task 5: Build the crossed forecaster factory

**Files:**

- Modify: `src/qldpc_fno/models/fno1d.py` only if a reusable ring block is needed
- Create: `src/qldpc_fno/models/causal_forecaster.py`
- Create: `tests/models/test_causal_forecaster.py`

**Interfaces:**

```python
build_forecaster(spatial="cnn|fno", temporal="fir|hippo|gru", config=...) -> nn.Module
forecaster.predict_then_update(history_or_state, syndrome_t) -> tuple[q_t, new_state]
```

- [ ] Write failing shape and parameter-sharing tests for all four primary cells
  and both GRU controls.
- [ ] Assert circular shift equivariance of CNN/FNO, shared ring-site temporal
  parameters, exact FIR history 32, and HiPPO state order 16.
- [ ] Assert that changing `syndrome_t` cannot change emitted `q_t`, but does
  change the next state.
- [ ] Run the shared forbidden-field causal auditor on all four primary and both
  GRU forecasters. A spy test proves the forward object exposes no physical
  error, logical, latent-probability, event-label, or future field.
- [ ] Test reset and burn-in timing, batch-versus-separate sequence equivalence,
  and absence of cached state crossing sequence identities.
- [ ] Implement one factory whose only primary-cell switches are the spatial and
  temporal operators specified in the design.
- [ ] Report exact trainable parameter counts and add an FIR-only deterministic
  prefix-shuffle diagnostic.
- [ ] Commit as `feat: build crossed causal forecasters`.
- [ ] Independent architecture reviewer checks the 2×2 isolation and tensor path.

---

### Task 6: Implement stationary, EWMA, and logistic-AR comparators

**Files:**

- Create: `src/qldpc_fno/temporal/baselines.py`
- Create: `tests/temporal/test_baselines.py`

- [ ] Write failing tests for train-only stationary shrinkage, deterministic tie
  rules, EWMA predict-before-update timing, sequence reset, burn-in updates,
  32-lag zero padding, L2 selection, temperature calibration, and clipping.
- [ ] Implement stationary scalar/field, EWMA plus circular logistic mapper, and
  circular logistic AR with deterministic full-batch L-BFGS.
- [ ] Add a privileged `q_t` oracle whose type is rejected by deployable-model
  registries and permitted only in generator diagnostics.
- [ ] Commit as `feat: add causal channel baselines`.
- [ ] Independent baseline reviewer checks for information or tuning asymmetry.

---

### Task 7: Add the per-round-prior BP-LSD adapter

**Files:**

- Modify: `src/qldpc_fno/decoders/bplsd.py`
- Modify: `tests/decoders/test_bplsd.py`

**Interface:**

```python
decode_bplsd_prior_batch(
    hx,
    syndromes,
    logical_x,
    *,
    error_channels,  # (rounds, n)
    config=BPLSDConfig(
        max_iter=100,
        bp_method="minimum_sum",
        ms_scaling_factor=0.0,
        schedule="serial",
        lsd_method="LSD_E",
        lsd_order=5,
    ),
) -> DecodeBatchResult
```

- [ ] Test shape, finite-value, and strict `(0,0.5)` prior validation.
- [ ] Refactor the scalar and vector-prior APIs through the same frozen
  `BPLSDConfig` and decoder-construction path. Test constant prior rows reproduce
  corrections, predicted observables, syndrome validity, convergence, and
  iterations from the scalar API bit-for-bit.
- [ ] Spy on independent decoder construction/calls to prove distinct rows reach
  distinct rounds; do not rely on untested in-place prior mutation.
- [ ] Test all arms receive identical pinned BP/LSD parameters and reuse the
  existing syndrome-validity and logical-observable scoring path.
- [ ] Commit as `feat: decode with per-round priors`.
- [ ] Independent QEC reviewer checks degeneracy-aware failure scoring and fixed
  decoder parity between arms.

---

### Task 8: Add causal sequence training and calibration

**Files:**

- Create: `src/qldpc_fno/training/causal_sequence.py`
- Create: `tests/training/test_causal_sequence.py`

- [ ] Define a separate deterministic, causally learnable, translation-equivariant
  overfit fixture in the reduced config. It has two sequences, eight burn-in and
  16 scored rounds, seed `1801`; each syndrome input is constant around the ring
  and constant through time, with balanced on/off channels across the pair; target
  output channel `c` is the fixed value of syndrome channel `c mod 21` from the
  preceding round, repeated around the ring. It contains no Bernoulli sampling,
  is never decoded, and is never reported as scientific data. Use raw (unclipped)
  sigmoid training outputs, AdamW learning rate `1e-2`, at most 2,000 steps,
  scored Bernoulli NLL `<=0.03`, and thresholded target accuracy `>=0.995`.
  Every learned cell must meet both thresholds with the same fixture and budget;
  the test first verifies the rule is representable by an ideal reference for
  each temporal path.
- [ ] Test full-sequence batching, separation of observed inputs and supervision,
  train/validation/calibration role isolation, validation checkpoint selection,
  scalar temperature fitting, and absence of a test role.
- [ ] Implement the fixed discovery optimizer separately from the explicit
  overfit-fixture override.
- [ ] Commit as `feat: train causal sequence forecasters`.
- [ ] Independent training reviewer checks checkpoint selection and role leakage.

---

### Task 9: Add paired causal evaluation

**Files:**

- Create: `src/qldpc_fno/temporal/evaluation.py`
- Create: `tests/temporal/test_evaluation.py`

- [ ] Test that all frozen arms predict and decode identical sequence membership
  through `decode_bplsd_prior_batch` and all corrections are syndrome-valid.
- [ ] Fit/select EWMA and logistic AR using validation NLL only, freeze the lower
  NLL arm with ties resolved in favor of EWMA, then compare it with stationary
  field on the same validation sequences.
- [ ] Define reduced progression signal per B-E as strictly lower overall mean
  validation NLL and non-worse overall mean validation BLER. Record per-sequence
  values too. This is descriptive and has no p-value or hypothesis status.
- [ ] Emit parameter counts, forecast metrics, decoder diagnostics, per-round
  outcomes, and descriptive interaction contrasts.
- [ ] Commit as `feat: evaluate causal decoder priors`.
- [ ] Independent QEC/statistical reviewer checks information symmetry and the
  exact reduced estimand.

---

### Task 10: Add immutable reduced CLI orchestration and verification

**Files:**

- Create: `experiments/18_generate_causal_sequences.py`
- Create: `experiments/19_run_causal_factor_screen.py`
- Create: `tests/integration/test_causal_screen_cli.py`

- [ ] Emit per-round outcomes, sequence summaries, causal-audit result, parameter
  counts, forecast metrics, decoder diagnostics, hashes, and the mandatory
  `reduced_non_scientific` label.
- [ ] Add direct-regeneration and result-recomputation verification commands.
- [ ] Run the reduced CLI twice and require hash-identical scientific payloads
  apart from declared timing fields.
- [ ] Publish payload-first and completion-manifest-last without overwriting
  completed runs.
- [ ] Commit as `feat: orchestrate reduced causal screen`.
- [ ] Independent QEC reviewer reads code, config, logs, seeds, and artifacts in
  the user's ten-part review order.

---

### Task 11: Run Gate 2 and make the stop/go decision

**Files:**

- Create under ignored artifacts: `artifacts/causal-fno-hippo-reduced/`
- Modify only after verified evidence: `README.md`
- Create only after verified evidence: `docs/causal-fno-hippo-results.md`

- [ ] Run focused tests after each task, then `uv run pytest -q`,
  `uv run ruff check .`, and `git diff --check`.
- [ ] Generate reduced train/validation/calibration sequences for A-E.
- [ ] Verify artifacts before fitting any model.
- [ ] Fit/select stationary/EWMA/logistic arms on validation NLL, freeze the
  winner, then enforce the exact descriptive B-E progression signal from Task 9.
- [ ] Overfit deterministic fixtures, then fit the four crossed cells.
- [ ] Compute the descriptive interaction contrast and basis-mismatch direction;
  do not attach confirmatory p-values to reduced output.
- [ ] Ask an independent adversarial reviewer for `PROCEED`, `REVISE`, or
  `STOP-NO-REDUCED-SIGNAL`.
- [ ] If `PROCEED`, write and review a separate Gate-3 discovery plan and runtime
  estimate before launch. If not, document the failed prerequisite or absent
  reduced signal and stop
  rather than broadening the model search.
- [ ] Commit only evidence-backed documentation; push after the complete test
  suite and final review pass.

## Gate-2 success definition

The reduced screen succeeds only when all of the following are true:

1. every artifact regenerates and all causal mutation audits pass;
2. every learned cell overfits the deterministic fixture;
3. the frozen observation-only baseline has strictly lower overall mean
   validation NLL and non-worse overall mean validation BLER than stationary
   field in each of B-E;
4. all decoder corrections are syndrome-valid and logical failure is evaluated
   modulo the committed logical operators;
5. the four crossed cells finish on identical sequence membership; and
6. the reviewer finds no blocker in implementation, information symmetry,
   baseline strength, or claim labeling.

This gate establishes only that the experiment is executable and informative.
It may falsify deterministic prerequisites such as causality, regeneration,
syndrome validity, or fixture learnability. It cannot statistically support or
falsify FNO superiority, HiPPO superiority, their interaction, decoder
improvement, hardware relevance, or publishability.
