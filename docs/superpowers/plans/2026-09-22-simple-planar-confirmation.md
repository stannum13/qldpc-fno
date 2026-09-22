# Simple Planar Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and independently review a frozen 4,096-shot comparison of
two simple planar policies before opening their fresh confirmation domain.

**Architecture:** Three new focused modules separate frozen configuration and
pure statistics, independent policy/reference execution, and provenance/replay
publication. Reuse the existing planar sampler, tensor-work instrumentation,
reference certificate, and physical recovery scorer. Keep nondeterministic host
timing outside deterministic scientific artifacts.

**Tech Stack:** Python `>=3.14,<3.15`, NumPy 2.4.1, SciPy 1.17.1, qecsim 1.0b9,
pytest 9.1.1, Ruff 0.16.5, the committed `uv.lock`, strict canonical JSON.

## Global Constraints

- This plan is documentation only in Task 8; do not implement or sample now.
- The [frozen contract](../../adaptive-computation-simple-confirmation-contract.md)
  is normative; the [design](../specs/2026-09-22-simple-planar-confirmation-design.md)
  explains architecture. Resolve contradictions before release.
- Scientific domain: `qldpc-fno/simple-planar-confirmation/v1`; campaign identity
  `2409828766515432030`; exactly 2,048 shots per rate at `[0.10,0.15]`.
- Fixtures: `qldpc-fno/simple-planar-confirmation/test-fixture/v1`, identity
  `3697382327853009454`, exactly two shots per rate; no scientific coordinates in
  tests, including subprocesses. Parsing constants/domain-only hashes is allowed.
- Bootstrap: separate domain `qldpc-fno/simple-planar-confirmation/bootstrap/v1`,
  seed `2713269656809941213`, PCG64, 10,000 replicates, linear .05 quantile.
- Policies: `fixed_rows_tol003`, `margin_columns_chi8`; strong comparator:
  `fixed_columns_chi8`. Preserve margin threshold `0.30710401263493464` strictly.
- Eight primary one-sided CP bounds use `alpha=.00625` and pass at `U<=.005`.
  No sample extension, outcome-driven action choice, superiority test, or learning.
- Invalid/uncertified reference or invalid final recovery means both endpoint
  events; no denominator deletion. Failed reference certification also blocks
  positive scientific status.
- Margin work claim: paired mean relative saving, equal-rate mixture, bootstrap
  lower bound **greater than .50**; approximate secondary analysis outside the
  exact primary family. Fixed rows work remains descriptive.
- Preserve the pilot and its evidence. `rows_tol003` was selected post hoc, and
  learned-controller work under the TV objective is stopped.
- No confirmation coordinate derivation before independent contract and committed
  implementation/preregistration reviews accept the exact release identities.
- Each implementation task records an observed red test, green result, targeted
  lint/diff check, and atomic commit. No task runs scientific data to test code.

---

## Files, interfaces, and immutable schemas

Create `simple_confirmation.py` for constants/config parsing, event indicators,
CP bounds, bootstrap, schemas, summaries, and decisions. Create
`simple_confirmation_runner.py` for policy actions/reference/physical scoring and
timing. Create `simple_confirmation_io.py` for preflight, release, generator
wrapper, provenance, publication, and replay. All three live in
`src/qldpc_fno/decision/`. Create CLI
`experiments/36_run_simple_planar_confirmation.py` with exactly `preflight`,
`generate`, `run`, and `verify` subcommands.

Create three configs named `simple_planar_confirmation.json`,
`simple_planar_confirmation_shots.json`, and
`simple_planar_confirmation_fixture_shots.json`. Add tests in
`tests/decision/test_simple_confirmation.py`,
`tests/decision/test_simple_confirmation_runner.py`,
`tests/decision/test_simple_confirmation_io.py`, and
`tests/integration/test_simple_confirmation_cli.py`; create `tests/conftest.py`
for the session-wide seed firewall. The only anticipated existing scientific
file change is a narrowly scoped test-mode guard in `planar_shot_data.py`.
Do not refactor the pilot or change its numerical algorithms.

Public interfaces to implement, with no scientific side effects except where
explicitly named:

```python
# simple_confirmation.py
def load_config(path: Path) -> dict[str, object]: ...
def load_shot_config(path: Path, *, fixture: bool) -> dict[str, object]: ...
def primary_upper(events: int, shots: int) -> float: ...
def event_indicators(*, reference_valid: bool, policy_valid: bool,
                     policy_class: int | None, reference_class: int | None,
                     policy_failure: bool, reference_failure: bool | None
                     ) -> tuple[bool, bool]: ...
def arm_order(shot_index: int) -> tuple[str, str, str]: ...
def bootstrap_saving(paired: np.ndarray) -> dict[str, object]: ...
def validate_result_rows(rows: list[dict], *, fixture: bool) -> None: ...
def summarize(rows: list[dict], *, fixture: bool) -> dict[str, object]: ...

# simple_confirmation_runner.py
def run_action(invocation_id: str, action_id: str, syndrome: np.ndarray,
               error_rate: float, timing: list[dict]) -> dict: ...
def run_policy(policy_id: str, syndrome: np.ndarray, error_rate: float,
               timing: list[dict]) -> tuple[dict, list[dict]]: ...
def evaluate_shot(shot: dict, *, timing: list[dict]) -> dict: ...

# simple_confirmation_io.py
def preflight(config_path: Path, shot_config_path: Path) -> dict: ...
def validate_release(receipt_path: Path, preflight_record: dict) -> dict: ...
def generate(config_path: Path, shot_config_path: Path, out: Path, *,
             release_path: Path | None, fixture: bool = False) -> dict: ...
def run(config_path: Path, shot_config_path: Path, shots_path: Path, out: Path,
        *, release_path: Path | None, fixture: bool = False) -> dict: ...
def verify(result_path: Path, receipt_out: Path, *, fixture: bool = False) -> dict: ...
```

These are interface declarations, not stub implementations. Every implementation
must provide all specified behavior and validators before its task is complete.
`run_policy` receives no sampled error, exact target, or timer-derived feature;
`evaluate_shot` adds labels afterwards. `preflight` and `validate_release` must
not call a coordinate-seed function, RNG, sampler, or contraction.

Use the contract's exact raw/action/policy/reference/primary/timing field sets.
Freeze these additional nested summary schemas (all keys required, null only
where specified; rate rows ordered `.10,.15`, policies in declared order):

| Section | Exact fields |
| --- | --- |
| `sample_counts[]` | `error_rate, expected, observed, complete` |
| `invalid_counts[]` | `error_rate, reference_uncertified, invalid_recoveries, invalid_actions, unmetered_actions`; last three maps have every declared policy or invocation identity, including zero-count fallback entries. |
| `policies[]` | `policy_id, error_rate, shots, valid_recoveries, physical_failures, physical_failure_rate` for all three arms. |
| `comparator[]` | `error_rate, shots, class_mismatches, outcome_discordances, class_mismatch_rate, outcome_discordance_rate` for fixed columns only. |
| `coverage[]` | `error_rate, shots, accepted, escalated, accepted_class_mismatches, accepted_outcome_discordances, coverage, accepted_class_mismatch_rate, accepted_outcome_discordance_rate`; conditional rates null at zero acceptance. |
| `paired_physical[]` | `error_rate, left, right, shots, both_succeed, left_only_fails, right_only_fails, both_fail, uncertified_reference`; three policy pairs and each policy versus reference. |
| `work` | `by_rate, equal_rate, bootstrap`; by-rate rows have `error_rate, shots, policy_totals, policy_means, reference_total, study_total, total_work_ratios`; equal-rate has `policy_means, reference_mean, study_mean, total_work_ratios`. |
| `work.bootstrap` | `status, estimand, replicates, seed, bit_generator, quantile_method, estimate, lower_bound, threshold, passed, unavailable_reason`; no replicate dump is required, but replay regenerates every replicate. |
| `decision` | `execution_eligible, policy_gate_status, both_policy_gates_passed, margin_work_gate_passed, reasons` |

Policy maps have the three literal arm keys. The `total_work_ratios` map has
`fixed_rows_tol003_over_fixed_columns_chi8` and
`margin_columns_chi8_over_fixed_columns_chi8`; null on zero or unknown denominator.
Work null propagation never discards a missing contribution. Add reference BLER
to `policies[]` with `policy_id="reference"`, `valid_recoveries=certified count`,
`physical_failures=uncertified count + certified physical failures`, and label
that row's rate as the all-shot system-failure rate in prose; separately obtain
certified physical BLER from the reference-pair cells and certified denominator.
All per-rate rates have an equal-rate arithmetic mean in the reporting layer;
do not insert a pseudo-rate into the primary family.

Run output has `status="complete_pending_replay"` or
`"reduced_non_scientific"`. `scientific_eligible` means complete approved
production execution with certified references and metered work; it does not
assert replay success. `policy_gate_status` values are
`"gates_passed_pending_replay"`, `"failed_discrepancy_budget"`,
`"blocked_execution"`, or `"fixture_only"`. Mathematical gate values stay
visible even when blocked. Positive final statuses occur only in the independent
verification receipt; raw artifacts are immutable and never rewritten to mark
verification complete.

Release receipt exact keys: `schema_version, contract_id, contract_sha256,
approved_commit, source_sha256, config_bindings, runtime, code_identity,
historical_domains, review_bindings, unopened_domain, expected_shots,
decision`. `decision` must be `"approved_to_open"`, `expected_shots=4096`, and
the domain must be the production domain. `review_bindings` contains exactly
`contract` and `implementation`, each `{path,size_bytes,sha256,verdict}` with
`verdict="accepted"`. Bind the clean implementation commit and all sources;
the receipt may be written afterwards outside Git, avoiding a self-hash cycle.

Final verification receipt exact keys: `schema_version, contract_id,
producer_commit, verifier_commit, raw_sha256, summary_sha256, timing_sha256,
checks, policy_status, margin_work_status, scientific_eligible`.
`checks` has Boolean `file_closure, provenance, joined_shot_replay,
contraction_replay, physical_scoring, work_reconciliation, summary_bytes,
bootstrap_replay, timing_integrity`; every one must be true. Final per-policy
statuses are `passed_discrepancy_budget`, `failed_discrepancy_budget`,
`blocked_execution`, or `fixture_only`; work statuses are `passed_secondary`,
`descriptive_only`, or `fixture_only`.

Raw `provenance` exact keys: `producer_commit, git_dirty, source_sha256,
config_bindings, runtime, code_identity, historical_domains, release_binding,
shots_binding, generation_manifest_binding`. Every file binding is
`{path,size_bytes,sha256}` with repository-relative path, exact bytes, and no
parent traversal. `config_bindings` has exactly `analysis` and `shots`.
`source_sha256` maps the reviewed exact source paths to digests; require their
bytes to match both the producing commit and current runtime source.
`historical_domains` has `configs` (path to `{size_bytes,sha256,domains}`) and
`reserved_domains` (sorted retired/fixture domain strings).
`code_identity` is `{family,distance,n,k,stabilizers_sha256,logicals_sha256}`;
hash the C-contiguous `uint8` matrices in documented row order and verify their
fixed shapes before hashing. `runtime` has `dependencies,matching_backend,blas,
thread_environment`; preserve existing dependency/backend keys, use
`blas={numpy_configuration,scipy_configuration}` for captured build identities,
and record `OMP_NUM_THREADS,OPENBLAS_NUM_THREADS,MKL_NUM_THREADS,VECLIB_MAXIMUM_THREADS`
as present strings or null in `thread_environment`. Host time measurements do
not enter this identity.

Do not extend the old planar-shot producer's fixed source map and then claim it
passes the old validator unchanged. Preserve its exact `{config,provenance,shots}`
artifact and create a separate `generation_manifest.json` alongside it, with
exact keys `schema_version,contract_id,producer_commit,config_bindings,
source_sha256,runtime,code_identity,historical_domains,release_binding,shots_binding`.
The wrapper's full transitive provenance lives in that manifest and is bound
again in run provenance. The generation directory has exactly these two files.
The raw run's `shot_provenance` preserves the original sampler provenance.

Timing `host` exact keys are `node,platform,machine,process_id,thread_environment,
blas`. Each timing shot has `shot_id,error_rate,shot_index,arm_order,
reference_order,actions,policies,shot_wall_seconds`. Action timing rows have
`invocation_id,valid,exception_type,wall_seconds`; policy timing rows have
`policy_id,wall_seconds`. Include finite nonnegative failed-action durations,
every actual invocation exactly once, and no nonexistent skipped fallback.
Check nested durations against their containing policy/shot/run durations with
tolerance `max(1e-9,1e-9*parent_seconds)`; no time can change a scientific value.

`decision.policy_gate_status` has exactly the two selected-policy keys.
`work.bootstrap.estimand` is `"equal_rate_mean_paired_relative_saving"`;
its `status` is `"available"`, `"unavailable"`, or `"fixture_only"`.
Unavailable reason is null only when available. Primary/eligibility failures
can suppress the final work claim even when its numerical interval is available.
For final status, `work.bootstrap.passed` is true only when the strict numerical
gate, the margin primary gates, and execution eligibility all hold; label that
value pending independent replay until a verification receipt exists.

## Task 1: Freeze strict configuration and the fixture firewall

**Files:** create the three configs, `simple_confirmation.py`,
`tests/decision/test_simple_confirmation.py`, and `tests/conftest.py`; modify
`src/qldpc_fno/decision/planar_shot_data.py` only for the test-mode domain guard.
**Consumes:** contract constants and the existing seven-field sampler schema.
**Produces:** `load_config`, `load_shot_config`, immutable constants, global
fixture guard. No generation is part of this task.

- [ ] Write tests that reject every changed constant, unknown/missing/duplicate
  key, NaN/Infinity, float/Boolean shot counts, wrong domain/campaign identity,
  fixture domain in scientific mode, scientific domain in fixture mode, changed
  action order/parameter, and insufficient declared source inventory. Include:

  ```python
  @pytest.mark.parametrize("count", [2, 2047, 2049, True, 2048.0])
  def test_scientific_count_is_frozen(tmp_path, count):
      payload = dict(FROZEN_SHOT_CONFIG, shots_per_rate=count)
      path = tmp_path / "shots.json"
      path.write_text(json.dumps(payload))
      with pytest.raises(ValueError):
          load_shot_config(path, fixture=False)

  def test_fixture_cannot_derive_production_seed(monkeypatch):
      def forbidden_hash(*args, **kwargs):
          pytest.fail("must reject scientific test coordinates before hashing")
      monkeypatch.setattr(planar_shot_data.hashlib, "sha256", forbidden_hash)
      with pytest.raises(ValueError, match="confirmation.*test"):
          planar_shot_data._shot_seed(
              SCIENTIFIC_DOMAIN, error_rate=.1, shot_index=0
          )
  ```

- [ ] Run `uv run pytest -q tests/decision/test_simple_confirmation.py -k 'config or seed'`.
  Observe missing module/API failure. The hash tripwire above is mandatory even
  during the first red cycle: a missing product guard must fail before computing
  a scientific seed. For generation tests, also install sampler/RNG tripwires.
- [ ] Implement `FROZEN_SHOT_CONFIG` exactly:

  ```python
  FROZEN_SHOT_CONFIG = {
      "schema_version": 1,
      "seed_domain": "qldpc-fno/simple-planar-confirmation/v1",
      "campaign_seed": 2409828766515432030,
      "code_distance": 5,
      "error_rates": [0.1, 0.15],
      "shots_per_rate": 2048,
      "noise_model": "qecsim_iid_depolarizing_code_capacity",
  }
  FIXTURE_SHOT_CONFIG = {
      **FROZEN_SHOT_CONFIG,
      "seed_domain": "qldpc-fno/simple-planar-confirmation/test-fixture/v1",
      "campaign_seed": 3697382327853009454,
      "shots_per_rate": 2,
  }
  ```

  Freeze analysis config exact keys `schema_version, contract_id,
  required_shot_seed_domain, fixture_seed_domain, code_distance, noise_model,
  required_error_rates, required_shots_per_rate, policy_ids, comparator_id,
  actions, margin_threshold, reference_probability_tolerance,
  reference_log_ratio_tolerance, reference_margin_discrepancy_factor,
  primary_endpoints, family_alpha, primary_alpha, discrepancy_budget,
  bootstrap_domain, bootstrap_seed, bootstrap_replicates, bootstrap_generator,
  bootstrap_quantile, bootstrap_quantile_method, minimum_work_saving,
  order_rule, invalidity_rule, status_rule`. Values are the contract literals;
  `order_rule="lexicographic_permutations_index_mod_6"`,
  `invalidity_rule="count_both_events_reference_blocks_positive"`,
  `status_rule="immutable_pending_independent_replay"`. `actions` is the exact
  six-item table `rows_tol003`, `columns_tol01`, `rows_tol01`, `columns_chi8`,
  `exact_columns`, `exact_rows`, each with `action_id,mode,chi,tol`; actual
  invocation order is separately determined by the arm schedule.
- [ ] Add `pytest_configure` in `tests/conftest.py` to set
  `QLDPC_FNO_CONFIRMATION_TESTING=1` before test collection; restore its prior
  value in `pytest_unconfigure`. Use this narrow guard at the beginning of
  `_shot_seed` and immediately after config load in `generate_planar_shots`:

  ```python
  if (
      os.environ.get("QLDPC_FNO_CONFIRMATION_TESTING") == "1"
      and domain.startswith("qldpc-fno/simple-planar-confirmation/")
      and "/test-fixture/" not in domain
  ):
      raise ValueError("scientific confirmation seed forbidden in test mode")
  ```

  The new IO wrapper must perform the same rejection before release validation
  or RNG construction. Subprocess tests inherit the flag. Test direct generator,
  coordinate helper, and new wrapper entry points; only fixture coordinates may
  reach the sampler. Test domain comparisons/hash-only parsing independently.
- [ ] Run the focused tests plus existing planar-shot/pilot config tests and
  Ruff on touched Python files; run `git diff --check`. Commit only these task
  files with `feat: freeze simple confirmation configs and fixture firewall`.

## Task 2: Implement exact primary statistics and paired bootstrap

**Files:** modify `simple_confirmation.py` and its test file.
**Consumes:** immutable constants and complete paired vectors.
**Produces:** `primary_upper`, `event_indicators`, `bootstrap_saving`; no IO/RNG
for physical shots. The bootstrap RNG is the only random operation here.

- [ ] Add independent binomial-CDF inversion tests for `k=0,1,2,3,n`, monotonicity,
  strict integer/range validation, and all invalid-event combinations:

  ```python
  @pytest.mark.parametrize("events, expected", [
      (0, 0.0024750442291897544),
      (1, 0.003498836715834498),
      (2, 0.004385297715096555),
      (3, 0.005205093018026918),
  ])
  def test_primary_upper(events, expected):
      actual = primary_upper(events, 2048)
      assert actual == pytest.approx(expected, abs=1e-15)
      assert (actual <= .005) == (events <= 2)
      assert sum(math.comb(2048, j) * actual**j * (1-actual)**(2048-j)
                 for j in range(events+1)) == pytest.approx(.00625, abs=1e-13)

  def test_two_wrong_classes_can_share_failure_outcome():
      assert event_indicators(
          reference_valid=True, policy_valid=True,
          policy_class=1, reference_class=2,
          policy_failure=True, reference_failure=True,
      ) == (True, False)
  ```

- [ ] Run `uv run pytest -q tests/decision/test_simple_confirmation.py -k 'upper or event or bootstrap'`
  and observe failures for the unimplemented functions.
- [ ] Implement these exact numerical kernels, adding strict type/finite/shape
  validators before their numerical bodies:

  ```python
  def primary_upper(events: int, shots: int) -> float:
      if type(events) is not int or type(shots) is not int:
          raise ValueError("events and shots must be integers")
      if shots <= 0 or not 0 <= events <= shots:
          raise ValueError("invalid binomial counts")
      return 1.0 if events == shots else float(
          scipy.stats.beta.ppf(1 - .00625, events + 1, shots - events)
      )

  def event_indicators(*, reference_valid, policy_valid, policy_class,
                       reference_class, policy_failure, reference_failure):
      if not reference_valid or not policy_valid:
          return True, True
      return policy_class != reference_class, policy_failure != reference_failure
  ```

  Valid labels require integer classes in `[0,3]` and Boolean failures; nulls are
  permitted only for invalid labels. The event function checks validity before
  comparing nullable fields. Preserve Boolean-versus-signature distinctions.
- [ ] Define paired numeric vectors in this exact column order:
  `margin_work,column_work,rows_work,margin_failure,column_failure,rows_failure,
  reference_failure,margin_M,margin_O,rows_M,rows_O`. Rows are ordered within
  `.10,.15` strata. Validate shape `(2,2048,11)`, finite entries, nonnegative
  integer-valued costs and strictly positive comparator cost before bootstrap;
  wrapper validation blocks invalid/unmetered contributing actions first.
  Implement the resampling core without flattening strata or splitting arms:

  ```python
  rng = np.random.Generator(np.random.PCG64(2713269656809941213))
  replicates = np.empty(10000, dtype=np.float64)
  for b in range(10000):
      means = []
      for stratum in range(2):
          indices = rng.integers(0, 2048, size=2048)
          selected = paired[stratum, indices, :]
          means.append(float(np.mean(1 - selected[:, 0] / selected[:, 1])))
      replicates[b] = (means[0] + means[1]) / 2
  estimate = float(np.mean(1 - paired[:, :, 0] / paired[:, :, 1]))
  lower = float(np.quantile(replicates, .05, method="linear"))
  ```

  Return the exact `work.bootstrap` keys, with `passed=(lower>.50)` initially
  only the numerical gate; `summarize` applies primary/eligibility restrictions.
  Unavailable analysis uses null estimate/lower, `passed=false`, exact cause,
  and preserves all declared algorithm metadata. Fixtures never run an
  inferential bootstrap and return `status="fixture_only"`.
- [ ] Add deterministic synthetic paired-vector tests: constant 75% saving gives
  `.75` estimate/lower; exactly 50% does not pass; negative saving stays negative;
  unequal rate distributions retain equal weights; variable comparator cost
  distinguishes mean ratios from ratio of totals; repeated calls match; swapping
  an arm destroys a hand-constructed pairing check. Verify all complete vector
  columns are indexed together using a deterministic injected RNG spy. Test
  invalid/zero/missing costs suppress inference without dropping observations.
- [ ] Run tests, Ruff, diff check; commit `feat: add exact confirmation gates and paired work bootstrap`.

## Task 3: Execute independent arms and certify/scored joined shots

**Files:** create `simple_confirmation_runner.py` and
`tests/decision/test_simple_confirmation_runner.py`; add `arm_order` to the pure
module. **Consumes:** fixed action table, event function, existing tensor API.
**Produces:** strict evaluation rows and separate timing entries.

- [ ] Add fake-contraction tests before implementation. Use fixture-domain shots
  or a hand-constructed all-zero physical error and derived syndrome; record each
  call's mode/chi/tol and return `PlanarCosetMasses` with controlled positive
  four-class masses and full synthetic stable trace. Test these exact cases:

  | Case | Required evidence |
  | --- | --- |
  | Valid strict agreement | Two margin actions; no fallback; total six contractions. |
  | Margin at threshold | Fallback executes; total seven; no reference input to decision. |
  | Probe disagreement | Fallback independent from the fixed-column invocation. |
  | First/second/both probes invalid | Both probes attempted; fallback charged; final valid recovery scored normally. |
  | Invalid fallback | Both primary events true and full partial work retained. |
  | Invalid exact mass, class disagreement, tolerance excess, ambiguous tie | Reference uncertified; both events true for every policy. |
  | Normalization underflow or nonfinite log differences | Reject before declaring a certificate, including NaN comparison fallthrough. |
  | Two agreeing but wrong probe classes | Accepted mismatch and physical transition detected. |
  | Equivalent recovery plus stabilizer | Physical success unchanged. |
  | Distinct wrong logical signatures | Physical-failure Boolean can agree while class mismatch is true. |
  | Failed SVD then retry | Every attempted shape reconciles with charged totals. |
  | Unexpected TypeError/KeyError | Propagates as fatal execution failure. |

- [ ] Run `uv run pytest -q tests/decision/test_simple_confirmation_runner.py`;
  observe missing runner failures.
- [ ] Implement arm order exactly:

  ```python
  POLICY_IDS = ("fixed_rows_tol003", "margin_columns_chi8", "fixed_columns_chi8")
  ORDERS = tuple(itertools.permutations(POLICY_IDS))

  def arm_order(shot_index: int) -> tuple[str, str, str]:
      if type(shot_index) is not int or shot_index < 0:
          raise ValueError("invalid shot index")
      return ORDERS[shot_index % 6]
  ```

  `itertools.permutations` follows the declared numeric index order, not sorting
  policy-name strings. Assert each order count is 341 or 342 over 2,048 indices.
- [ ] Implement `run_action` using `planar_mps_coset_masses` with `trace_work=True`.
  Always retain masses/nulls, exception class/message/nulls, stable trace, cost,
  normalized probabilities, class, and unique invocation ID. Reuse
  `planar_shot_accuracy._stable_work` to remove wall time from scientific traces.
  Catch only declared numerical invalidity. Record elapsed time in a `finally`
  block for success or failure; unknown work stays null. Do not infer zero work
  from `work is None`. Reconcile every decomposition and full action cost.
- [ ] Implement policy dispatch with these complete action sequences:

  ```python
  # Every call invokes run_action anew; no shared cache exists.
  if policy_id == "fixed_rows_tol003":
      action_ids = ["rows_tol003"]
  elif policy_id == "fixed_columns_chi8":
      action_ids = ["columns_chi8"]
  elif policy_id == "margin_columns_chi8":
      action_ids = ["columns_tol01", "rows_tol01"]
  else:
      raise ValueError("unknown policy")
  actions = [run_action(f"{policy_id}/{name}", name, syndrome, error_rate, timing)
             for name in action_ids]
  if policy_id == "margin_columns_chi8":
      col, row = actions
      accepted = margin_gate_accepts(
          col["probabilities"], col["selected_class"],
          row["probabilities"], row["selected_class"],
          threshold=0.30710401263493464,
      )
      if not accepted:
          actions.append(run_action(f"{policy_id}/columns_chi8",
                                    "columns_chi8", syndrome, error_rate, timing))
      selected = actions[0] if accepted else actions[-1]
  else:
      accepted, selected = None, actions[0]
  ```

  Import `margin_gate_accepts` from the audited pilot pure helper and bind that
  source transitively; do not call its pilot runner or oracles. Build the
  recovery and syndrome-validity result before stopping whole-policy timing;
  defer residual error/logical scoring to `evaluate_shot`. Sum all attempted
  action work for policy cost with null propagation.
- [ ] Implement `evaluate_shot`: validate error/syndrome joining; call all three
  policy blocks in `arm_order`; run both references after them with parity order;
  certify the reference with its full ambiguity guard; construct target/recovery;
  use `score_recovery` for every final policy and valid reference; add primary
  events; preserve original coordinates/error/syndrome and all ordered actions.
  Action failure does not skip another planned independent action. Reference
  failure yields null target labels plus forced event flags.
  Before accepting the existing certificate result, explicitly require finite
  positive normalized probabilities and finite certificate quantities. Mark the
  final reference certified only if its reconstructed recovery is syndrome valid;
  retain a successful numerical certificate diagnostically if recovery fails.
- [ ] Run focused tests plus `tests/decision/test_tensor_network.py` and
  `tests/decision/test_planar_shot_accuracy.py`; check Ruff/diff. Commit
  `feat: execute paired simple planar policies with complete accounting`.

## Task 4: Validate raw rows and construct all summaries

**Files:** modify pure module/tests. **Consumes:** strict evaluation rows from
Task 3, CP/bootstrap functions from Task 2. **Produces:** `validate_result_rows`
and `summarize`, including all nested schemas and pending-replay decisions.

- [ ] Write tests with complete synthetic 2,048-row-per-rate vectors and no
  physical sampling. Cover event counts `0,1,2,3`, exactly eight rows, one policy
  passing while the other fails, every reference invalidity, coverage zero,
  accepted invalid reference, invalid probe followed by good fallback, missing
  shot/action/trace, duplicate index, and unknown/null work. Include this gate
  assertion after constructing rows with three events in one chosen cell:

  ```python
  assert len(summary["primary"]) == 8
  target = summary["primary"][0]
  assert target["events"] == 3 and target["shots"] == 2048
  assert target["upper_bound"] == pytest.approx(0.005205093018026918)
  assert target["gate_passed"] is False
  assert summary["decision"]["policy_gate_status"]["fixed_rows_tol003"] == (
      "failed_discrepancy_budget"
  )
  ```

- [ ] Run pure-summary tests red. Implement exact schema/record validators before
  aggregation: strong types, declared action IDs and order, valid probability
  sums/classes, joined membership, nested certificate/work shape, logical-score
  reconstruction, primary events consistent with invalidity, and no extra keys.
  Do not trust persisted event flags without recomputing them.
- [ ] Construct primary rows using the ordered product of two selected policies,
  two rates, and two endpoint names. Gate on all 2,048 physical shots. Construct
  comparator/coverage/paired counts and all work sections using the frozen
  schemas. Each paired table must sum to 2,048 including its uncertified-reference
  cell. Check exact action sums, policy totals, references, and study totals.
- [ ] Apply status precedence: fixture-only; blocked execution/reference/work;
  failed discrepancy gate; otherwise gates passed pending replay. Show numerical
  gate rows regardless of status. The work claim additionally needs valid
  contributing actions, positive comparator work, passing margin gates, and
  strict bootstrap lower bound greater than .50. Keep fixed-rows work descriptive.
- [ ] Check per-rate/equal-rate estimands with deliberately unequal acceptance
  counts so a pooled accepted-set proportion cannot replace an equal-rate mean.
  Check missing reference physical outcome is not inserted into ordinary failure
  cells, and all-shot reference system failure is labeled separately.
- [ ] Run pure/runner tests and Ruff/diff; commit
  `feat: summarize frozen confirmation endpoints and decisions`.

## Task 5: Implement release preflight, provenance, and atomic publication

**Files:** create IO module and its tests. **Consumes:** exact configs, row/summary
validators, existing sampler and provenance helpers. **Produces:** `preflight`,
`validate_release`, `generate`, `run`; only fixture mode is exercised here.

- [ ] Write red tests for metadata-only preflight, missing/foreign release,
  dirty/uncommitted code/config/lockfile, mismatched runtime/BLAS/code/logical
  identity, duplicate historical domains, changed review bytes, stale source
  hashes, fixture escalation, and scientific test-mode rejection. Replace the
  coordinate helper, sampler, RNG, and contraction with raising spies during
  metadata-only tests; assert none is called.
- [ ] Freeze a transitive source inventory including all three new modules,
  the new CLI, `planar_shot_data.py`, `planar_shot_accuracy.py`,
  `tensor_network.py`, `adaptive_intervention_pilot.py`,
  `adaptive_diagnostics.py`, `metrics/paired.py`, `artifacts.py`, and any further
  project imports they use scientifically. Bind `pyproject.toml`, `uv.lock`,
  full config bytes, contract bytes, clean commit, runtime, matching binary,
  code/stabilizer/logical bytes, historical inventory, and receipt. Reject a
  present but unlisted transitive scientific dependency in review.
- [ ] Implement domain inventory recursively over config keys ending in
  `seed_domain`, add explicit retired pilot v1/v2 and fixture identities, exclude
  only this contract's exact scientific config declarations, and bind file bytes.
  `preflight` compares domain strings/campaign identities only. Production
  `generate` validates the release first, derives all coordinate lists, checks
  internal and historical planar-seed collisions, then calls the existing
  generator against the scientific config in private staging. Fixtures bypass
  release only with exact fixture config and cannot obtain scientific status.
  Produce the generation manifest described above; the old sampler's source
  closure remains valid separately from the wrapper's larger closure.
- [ ] Write failure-injection tests for serialization, schema check, file hash,
  provenance recheck, rename, unexpected solver exception, and concurrent output
  creation. Test existing directory, file, broken symlink, and two publishers;
  no earlier output bytes may change and no partial result may appear.
- [ ] Implement an exclusive destination lock with `os.open(lock, O_CREAT |
  O_EXCL | O_WRONLY, 0o600)` and refuse preexisting/stale locks instead of
  removing another process's lock. Recheck destination/symlink under the lock,
  write a unique `tempfile.TemporaryDirectory(dir=out.parent)` staging tree,
  serialize all complete files, flush/fsync files and staging directory, validate
  all hashes/schemas, recheck source/input provenance, and atomically rename.
  Use `finally` to release only the lock inode owned by this process. A stale
  lock is an engineering review condition, not permission to overwrite output.
- [ ] Preserve an external failure receipt with exception type, approved
  source/input hashes, intended destination, and stage; omit partial scientific
  verdicts. No automatic retry, output overwrite, or resume by omission.
  A same-source restart uses a new destination and unchanged immutable shots.
- [ ] Run IO tests plus existing sampler/pilot regressions, Ruff/diff; commit
  `feat: enforce confirmation release and atomic provenance-bound outputs`.

## Task 6: Implement CLI and independent replay

**Files:** create CLI/integration tests; modify IO module/tests.
**Consumes:** Tasks 1–5. **Produces:** four CLI commands and `verify`, which writes
an immutable receipt and never modifies raw or timing artifacts.

- [ ] Add failing integration tests using only fixture config/two shots per rate:
  two runs produce byte-identical raw/compact scientific files and different
  timing sidecars; fixture final statuses always remain `fixture_only`; bad
  joined errors or missing rows exit nonzero without published result; changed
  clock values cannot alter any scientific byte or policy decision.
- [ ] Add tamper tests before replay implementation: mutate and rehash error,
  syndrome, recovery, logical signature, action parameter/order, reference
  certificate, failed-work record, summary gate, bootstrap metadata, source
  inventory, or undeclared file. Each must fail despite recomputed outer hashes.
  Fixture replay validates all science except production-size inference.
- [ ] Implement CLI argument parsing without `--shots-per-rate`, threshold,
  action, seed, rate, or work-test overrides. `--non-scientific-fixture` is
  explicit and demands the fixture shot config. Accept release receipt for
  production generate/run and refuse it as authority inside test mode. Default
  all scientific command paths to the frozen filenames only after exact loading.
- [ ] Implement `verify` in this order: validate exact file closure/outer schemas
  and committed identities; replay joined shots; independently execute every
  contraction under the approved runtime; independently reconstruct physical
  scoring and work; derive summary and bootstrap; compare scientific bytes;
  validate timing identity/coverage/finite durations and policy/action nesting;
  publish final verification receipt atomically. Do not count replay execution
  time as the original policy benchmark or mix its work with the original study.
  Reconstruct raw bytes using the authenticated original producer metadata;
  put new verifier identities only in the separate receipt, so a different replay
  destination does not change scientific provenance bytes.
- [ ] Test receipt status promotion only when all checks pass; final status still
  respects failed primary gates, reference invalidity, and secondary work guard.
  A fixture receipt cannot promote itself by editing its Boolean eligibility.
- [ ] Run:

  ```bash
  uv run pytest -q tests/decision/test_simple_confirmation.py tests/decision/test_simple_confirmation_runner.py tests/decision/test_simple_confirmation_io.py tests/integration/test_simple_confirmation_cli.py
  uv run ruff check .
  git diff --check
  ```

  Expect all focused tests and lint to pass. Commit
  `feat: add immutable confirmation CLI and independent replay`.

## Task 7: Independent preregistration release checkpoint

**Files reviewed:** all three contract/design/plan documents, new configs and
scientific modules/CLI, sampler guard, tests, full transitive source inventory.
**Produces:** the release receipt only after accepted reviews. No scientific
coordinate derivation occurs during this task.

- [ ] Obtain independent scientific review of target certification, logical
  scoring, exact event definitions, eight-bound Bonferroni family, 0/1/2/3-event
  values, 2,048-shot precision design, invalid handling, paired work bootstrap,
  status boundaries, comparator reporting, seed firewall, and cheapest
  disconfirming check. Explicitly check mean paired ratios versus ratio of totals.
- [ ] Obtain independent implementation review of information flow, strict
  schemas, complete work charging, internal retries, independent fallback,
  counterbalancing, provenance, transactionality, replay, and failure injection.
  Resolve every blocker/major with a red regression and atomic fix; re-review
  changed scientific decisions before any release.
- [ ] Run the cheapest disconfirming fixture case: accepted shared wrong class,
  invalid fallback, uncertified reference, and a three-event summary cell.
  Verify it prevents the corresponding positive status, retains denominators,
  and charges failed attempts. This is not a scientific-domain smoke run.
- [ ] Run the entire suite `uv run pytest -q`, `uv run ruff check .`, and
  `git diff --check`; inspect all results and require a clean committed tree.
  Audit subprocess fixture configs and seed guards, not only the new test files.
- [ ] Execute metadata-only preflight and independently confirm domain unopened
  history. Record accepted review hashes, clean approved commit, runtime/config
  and source hashes, exact contract digest, and full 4,096-shot scope in the
  release receipt. If exposure is uncertain, retire v1 and refreeze a new domain
  before generation. An approval receipt is not inferred from passing tests.

## Task 8: Generate and run the fixed confirmation after release

**These commands are to be implemented by Tasks 1–6. They do not exist as a
working confirmation workflow at contract-writing time and must not run now.**
The production destinations below are immutable; choose reviewed fresh sibling
destinations for identical-source replay instead of deleting or overwriting them.

- [ ] From the approved clean source/runtime, run the metadata-only check:

  ```bash
  uv run python experiments/36_run_simple_planar_confirmation.py preflight \
    --config configs/simple_planar_confirmation.json \
    --shot-config configs/simple_planar_confirmation_shots.json
  ```

  Expected: exact reviewed identities and `ready_for_release`; no shot seeds,
  RNG construction, errors, syndromes, contractions, or scientific output.
- [ ] After validating `artifacts/simple-planar-confirmation-v1/release.json`,
  generate the complete immutable physical-shot artifact:

  ```bash
  uv run python experiments/36_run_simple_planar_confirmation.py generate \
    --config configs/simple_planar_confirmation.json \
    --shot-config configs/simple_planar_confirmation_shots.json \
    --release artifacts/simple-planar-confirmation-v1/release.json \
    --out artifacts/simple-planar-confirmation-v1/shots
  ```

  Expected: 4,096 joined rows, exactly 2,048 per rate, full provenance and seed
  disjointness checks. The generated filename is `planar_shots.json`.
- [ ] Run all three policy arms and both references on every shot:

  ```bash
  uv run python experiments/36_run_simple_planar_confirmation.py run \
    --config configs/simple_planar_confirmation.json \
    --shot-config configs/simple_planar_confirmation_shots.json \
    --release artifacts/simple-planar-confirmation-v1/release.json \
    --shots artifacts/simple-planar-confirmation-v1/shots/planar_shots.json \
    --out artifacts/simple-planar-confirmation-v1/result
  ```

  Expected: atomic `confirmation.json`, `summary.json`, `timing.json` with
  `complete_pending_replay` status. There are 24,576 base contractions plus
  one independent margin fallback for each escalated shot. Decomposition retry
  counts are additional work, not additional statistical observations.
- [ ] Complete all 4,096 shots regardless of early event counts. No threshold,
  fallback, event definition, sample-size, runtime, or numerical-code changes
  after inspecting outcomes. An interruption permits only an identical-source
  fresh-destination rerun; a scientific repair requires a new domain and review.
- [ ] Run full independent replay into a new receipt directory:

  ```bash
  uv run python experiments/36_run_simple_planar_confirmation.py verify \
    --result artifacts/simple-planar-confirmation-v1/result \
    --out artifacts/simple-planar-confirmation-v1/verification
  ```

  Expected: every joined shot and contraction replays; deterministic raw-derived
  summary is byte-identical; exact primary/secondary results reproduce. Timing
  is integrity-checked but not compared for byte identity. Final receipt permits
  only the corresponding frozen statuses; no positive result is presumed.

## Task 9: Independent artifact review and bounded reporting

**Future outputs:**
`evidence/adaptive-computation/simple-planar-confirmation-v1/summary.json`,
`evidence/adaptive-computation/simple-planar-confirmation-v1/manifest.json`, and
`docs/adaptive-computation-simple-confirmation-results.md`. None is created by
the current contract-writing task.

- [ ] Obtain independent artifact review of every binding, raw availability,
  replay receipt, primary counts/denominators/bounds, invalid reference/recovery
  breakdown, baseline paired tables, accepted-set denominators, per-shot/total
  work, negative savings, bootstrap assumptions, and nondeterministic timing.
- [ ] Copy the compact summary byte-for-byte and construct a manifest binding
  source/config/input/raw/compact/timing/review/replay sizes and hashes plus exact
  commands. Validate declared-file closure and reproduce all compact bytes from
  raw rows before committing evidence. Do not claim ignored raw files are public.
- [ ] Report all eight bounds even when a policy fails; identify individual
  four-gate passes and the joint eight-gate status. Report baseline BLER and
  paired transitions descriptively. A failed reference/provenance/replay guard
  blocks positive prose. Distinguish arithmetic saving from host timing and
  distinguish the bootstrap work statement from the exact primary family.
- [ ] State that the pilot's `rows_tol003` selection was post hoc and this fresh
  domain alone supports confirmation. Preserve stopped learned-controller work.
  Exclude zero-risk, equivalence, superiority, device-safety, qLDPC transfer,
  repeated-round, hardware-latency, and learned-model claims.
- [ ] Check local links, manifest reconstruction, `uv run ruff check .`, and
  `git diff --check`. Commit only the reviewed evidence/results/link changes.
  Publishing or pushing follows the session's actual authorization; this plan
  does not itself authorize external publication.

## Coverage and handoff checklist

| Contract requirement | Implementation/review task |
| --- | --- |
| New disjoint domain, exact count/config, no test exposure | 1, 5, 7 |
| Eight exact bounds, invalid-event handling, event/sample budget | 2, 4, 7 |
| Fixed policies, certified target, logical scoring, independent fallback | 3, 7 |
| Comparator, coverage, paired physical tables, claim boundaries | 4, 9 |
| Attempted-work charging and secondary paired stratified bootstrap | 2, 3, 4 |
| Counterbalanced host timing with no scientific influence | 3, 6 |
| Strict schema, clean provenance, transactional failure behavior | 1, 4, 5 |
| Full joined-shot/contraction/statistical replay and immutable receipt | 6, 8, 9 |
| Cheapest disconfirmation and no optional success stopping | 7, 8 |

Do not execute a scientific command while implementing or testing this plan.
Complete the documented review gates and release before any physical coordinate
from the frozen confirmation domain is derived.
