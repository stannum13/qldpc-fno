# Adaptive Intervention Pilot Implementation Plan

> **Execution:** Follow `subagent-driven-development` and test-driven development.
> Do not generate the scientific seed domain until the preregistration review of
> the design and committed configs has no unresolved blocker.

**Goal:** Produce an auditable 128-shot development pilot that measures which
planar tensor-contraction interventions improve a trusted posterior, for what
fully charged arithmetic cost, before freezing the first margin-gate
confirmation contract.

**Architecture:** Reuse the existing immutable physical-shot generator and
planar MPS instrumentation. Add a separate intervention-pilot module that
validates a frozen action config, evaluates every action on every joined shot,
certifies unrestricted row/column references, scores recoveries modulo logical
commutation, and derives gate/oracle summaries. Keep the raw scientific artifact
outside Git and publish a checksummed manifest, compact summary, and result note.

**Stack:** Python, NumPy, qecsim, pytest, Ruff, canonical JSON artifacts.

---

## Task 1: Freeze and validate both pilot configs

**Files:**

- Create: `configs/adaptive_intervention_pilot_shots.json`
- Create: `configs/adaptive_intervention_pilot.json`
- Create: `src/qldpc_fno/decision/adaptive_intervention_pilot.py`
- Create: `tests/decision/test_adaptive_intervention_pilot.py`

1. Write failing tests for exact config keys, schema version, action identity and
   order, unique action tuples, required two probe actions, fixed threshold,
   certificate tolerances, expected shot domain/count/rates, and derived campaign
   seed.
2. Run:

   ```bash
   uv run pytest -q tests/decision/test_adaptive_intervention_pilot.py -k config
   ```

   Expect failure because the module/configs do not exist.
3. Implement strict immutable config parsing. Freeze the 14 actions in the order
   declared by the design. Reject extra fields and non-finite values.
4. Add the shot-generator config with 64 shots at each of `0.10` and `0.15` in
   seed domain `qldpc-fno/adaptive-intervention-pilot/v1` and campaign seed
   `16980117767564665917`.
5. Rerun the focused tests and Ruff.
6. Commit: `feat: freeze adaptive intervention pilot`.

## Task 2: Implement pure scoring and policy primitives

**Files:**

- Modify: `src/qldpc_fno/decision/adaptive_intervention_pilot.py`
- Modify: `tests/decision/test_adaptive_intervention_pilot.py`

1. Write failing tests for:
   - posterior normalization and probability margin;
   - TV, Jensen-Shannon divergence, and reference-conditional excess risk;
   - natural-log JS behavior with zero components;
   - symmetric unrestricted-reference posterior construction;
   - strict margin-gate acceptance, including equality and invalid probes;
   - full independent-run composite work;
   - positive, zero, and negative candidate benefit;
   - the `1e-6` positive-gain and unique-winner separation floors and `1e-12`
     deterministic tie rule;
   - deterministic oracle selection, including no-positive-gain and ties;
   - repair/introduction of class mismatch and physical outcome discordance.
2. Run the focused tests and observe the expected failures.
3. Implement small pure functions. Reuse stable diagnostic metrics where their
   semantics match; do not make exact labels available to the gate.
4. Rerun focused tests and Ruff.
5. Commit: `feat: score intervention opportunities`.

## Task 3: Evaluate a shot with complete failure accounting

**Files:**

- Modify: `src/qldpc_fno/decision/adaptive_intervention_pilot.py`
- Modify: `tests/decision/test_adaptive_intervention_pilot.py`

1. Write failing tests using deterministic fake contractions for:
   - every action executed exactly once in frozen order;
   - both unrestricted views certified using the existing certificate;
   - recovery scoring through `logical_class_recovery` and `score_recovery`;
   - invalid action exception class and partial work retained;
   - invalid reference excluded from target metrics but fully charged;
   - any invalid reference forcing both advancement clauses false;
   - spectra recorded only after their action has run;
   - the exact frozen pre-action feature set and separate label namespace.
2. Implement action dispatch through `planar_mps_coset_masses(trace_work=True)`.
   Preserve complete stable work traces in raw output and derive compact spectral
   summaries. Do not catch programming/configuration errors as measured numerical
   invalidity.
3. Rerun focused tests and Ruff.
4. Commit: `feat: evaluate adaptive interventions`.

## Task 4: Add artifact validation, summaries, and CLI

**Files:**

- Modify: `src/qldpc_fno/decision/adaptive_intervention_pilot.py`
- Create: `experiments/35_run_adaptive_intervention_pilot.py`
- Modify: `tests/decision/test_adaptive_intervention_pilot.py`
- Create: `tests/integration/test_adaptive_intervention_pilot_cli.py`
- Modify: `.gitignore`

1. Write failing tests for:
   - shot artifact config/provenance and exact domain/count validation;
   - physical error/syndrome replay consistency;
   - disjointness from all declared historical domains;
   - deterministic result summaries and action denominators;
   - refusal to overwrite output;
   - dirty/uncommitted scientific provenance rejection;
   - byte-identical reduced replay;
   - CLI nonzero failure with no partial published output.
2. Implement the runner and CLI. Raw output goes to
   `artifacts/adaptive-intervention-pilot-v1/intervention_pilot.json`, already
   covered by the repository's ignored `artifacts/` directory.
3. Derive a compact summary without errors, syndromes, recovery strings, or full
   work traces. The summary must bind raw/config/shot hashes and reproduce counts
   and charged work.
4. Run focused unit/integration tests and Ruff.
5. Commit: `feat: add adaptive intervention pilot runner`.

## Task 5: Preregistration review checkpoint

**Files reviewed:** design, plan, configs, runner, tests.

1. Request an independent scientific review before creating the pilot shots.
2. Require explicit decisions on reference validity, logical scoring,
   information availability, action completeness, invalid-work charging,
   development-only scope, and the cheapest disconfirming check.
3. Resolve every BLOCKER and MAJOR finding with tests and atomic commits.
4. Re-run focused tests and Ruff. Require a clean committed tree.

## Task 6: Generate and run the bounded scientific pilot

**Produces:**

- `artifacts/adaptive-intervention-pilot-v1/shots/planar_shots.json`
- `artifacts/adaptive-intervention-pilot-v1/result/intervention_pilot.json`
- `evidence/adaptive-computation/intervention-pilot-v1/summary.json`
- `evidence/adaptive-computation/intervention-pilot-v1/manifest.json`

1. Verify clean Git status and record the exact commit.
2. Generate immutable joined shots:

   ```bash
   uv run python experiments/31_generate_planar_shots.py \
     --config configs/adaptive_intervention_pilot_shots.json \
     --out artifacts/adaptive-intervention-pilot-v1/shots
   ```

3. Run the full action grid:

   ```bash
   uv run python experiments/35_run_adaptive_intervention_pilot.py \
     --config configs/adaptive_intervention_pilot.json \
     --shots artifacts/adaptive-intervention-pilot-v1/shots/planar_shots.json \
     --out artifacts/adaptive-intervention-pilot-v1/result
   ```

4. Do not change the config, code, action set, or sample count after inspecting
   outcomes. Any execution defect requires a versioned new domain or a documented
   non-outcome-dependent repair reviewed before rerun.
5. Build the compact summary and manifest from the immutable raw output. Verify
   their reconstruction and hashes before adding them to Git.

## Task 7: Interpret, review, and publish the development evidence

**Files:**

- Create: `docs/adaptive-computation-intervention-pilot.md`
- Modify: `README.md`
- Add: compact summary and manifest from Task 6.

1. Report all frozen endpoints, denominators, invalid actions, total and per-action
   work, wall-time observations, oracle opportunity, negative gains, and strongest
   counterexamples. State that wall time is research-host timing, not deployment
   latency.
2. Apply the design's advance rule. If heterogeneous positive opportunity is
   absent, explicitly stop learned-controller work and retain a fixed fallback.
3. Request an independent artifact/replay review. Regenerate the compact summary
   from raw output and require byte identity.
4. Run:

   ```bash
   uv run pytest -q
   uv run ruff check .
   git diff --check
   ```

5. Commit the compact evidence and documentation. Push `main` only after all
   checks pass.

## Task 8: Freeze the next confirmation contract

Use the pilot only to choose among the predeclared branches:

- simple unchanged margin gate plus fixed column `chi=8` fallback; or
- unchanged margin gate plus a small, fully specified refinement action family,
  if the pilot establishes heterogeneous positive opportunity.

The contract must fix independent seed domains, exact estimands, fallback,
familywise error treatment, sample size/event-rate bound, work endpoint, and
failure criteria before confirmation samples are generated. It receives a new
design, implementation plan, and preregistration review; this pilot does not
silently become confirmation.
