# Frozen simple planar confirmation contract

**Contract:** `simple_planar_confirmation_v1`, frozen on 2026-09-22.
**State:** documentation only; implementation and independent preregistration
review are required before opening the confirmation domain. No confirmation
shots or outcomes were generated to write this contract. Hashing the three
domain strings below does not derive a physical-shot seed.

This experiment asks whether either of two selected simple policies stays within
a per-shot discrepancy budget of 0.005 relative to a certified unrestricted
tensor reference, separately at `p=0.10` and `p=0.15`, for distance-five qecsim
planar code under iid depolarizing code-capacity noise with perfect syndrome
measurement. It is not a qLDPC experiment or a device-safety assessment.

The [development pilot](adaptive-computation-intervention-pilot.md) selected
`rows_tol003` **post hoc**. Its 128 shots, every older screen and calibration
sample, and the retired pilot v1/v2 identities remain development data forever.
Only the new domain can support the predeclared confirmation statements below.
Learned-controller work under the pilot's TV objective is stopped: no controller,
TV-gain oracle, FNO, HiPPO, learned action selection, or timing-dependent decision
is part of this contract. The [design](superpowers/specs/2026-09-22-simple-planar-confirmation-design.md)
and [implementation plan](superpowers/plans/2026-09-22-simple-planar-confirmation.md)
specify the engineering handoff. This document is the normative scientific
contract; disagreement between these documents blocks preregistration release.

## Frozen sample and domain firewall

| Role | Domain | SHA-256 first-eight-byte unsigned big-endian integer |
| --- | --- | ---: |
| Confirmation physical shots | `qldpc-fno/simple-planar-confirmation/v1` | 2409828766515432030 |
| Tests only | `qldpc-fno/simple-planar-confirmation/test-fixture/v1` | 3697382327853009454 |
| Analysis resampling only | `qldpc-fno/simple-planar-confirmation/bootstrap/v1` | 2713269656809941213 |

There are exactly **2,048 joined physical shots per rate**, indices `0..2047`,
4,096 in total. One physical shot is the independent unit; actions and endpoints
on that shot are paired repeated measurements, not additional samples. Report
both strata and the equal-rate mean; pooling never substitutes for a per-rate
primary gate. No sample-size extension, optional stopping for success, rate
substitution, resampling an inconvenient error, or outcome-driven policy change
is allowed.

After release only, the existing planar sampler derives a shot seed as the first
eight SHA-256 bytes, unsigned big-endian, of the UTF-8 string
`{domain}|d5|p{rate:.6f}|i{index:06d}`. It samples one physical error with
`DepolarizingErrorModel.generate(PlanarCode(5, 5), rate,
numpy.random.default_rng(seed))` and derives its syndrome by binary symplectic
commutation with the stabilizers. Preserve the 82-bit physical error, 40-bit
syndrome, seed, rate, index, and shot ID together. The campaign integer identifies
the domain; it is not substituted for the per-shot seed.

Preflight binds every recursively declared `*seed_domain` in `configs/**/*.json`
and its config hash, plus retired pilot v1/v2 and both fixture domains. The new
physical domain must differ from every historical domain and from the bootstrap
domain. After release, before sampling, check the new seed list for duplicates
and against historical planar-shot seed lists, including the retired 64-per-rate
pilot identities and the reserved fixture coordinates. A collision aborts and
retires this version; never adjust one seed. Domain naming provides separation,
not a mathematical proof that truncated hashes cannot collide.

Tests may parse scientific constants and hash a domain string, but must never
derive a scientific per-shot seed, instantiate its RNG, generate/replay its
errors, or invoke a scientific generation command. An autouse seed guard must
raise before any scientific coordinate derivation in all tests. Actual fixture
generation uses only the test domain and cannot produce a positive scientific
status. Run tests, profilers, and smoke checks before release on fixtures only;
do not copy production configs and merely lower their shot count. Any uncertain
pre-release exposure retires this version, following the pilot v1/v2 precedent.

## Policies, target, and physical scoring

Every joined shot receives these three policy arms and both reference views:

| Arm | Frozen computation |
| --- | --- |
| `fixed_rows_tol003` | One independent rows contraction, `chi=None`, `tol=0.003`. |
| `margin_columns_chi8` | Independent columns then rows probes, each `chi=None`, `tol=0.01`; accept their common class only when both are valid and their minimum winning-class margin is strictly greater than `0.30710401263493464`; otherwise independently run columns `chi=8`, `tol=None`. |
| `fixed_columns_chi8` | One independent columns contraction, `chi=8`, `tol=None`, on every shot. |
| Reference | Independent columns and rows contractions, both `chi=None`, `tol=None`, on every shot. |

Classes use qecsim order `I, X, Y, Z` (indices `0,1,2,3`). The margin is the
largest normalized probability minus the second largest. Equality at the gate
threshold, probe disagreement, or either invalid probe escalates. The fallback
is never selected using the reference, sampled error, measured time, or posterior
TV. An invalid fallback yields an invalid policy recovery; no rescue action is
allowed. Both probes are always attempted, even if the first is invalid.

The margin arm's fallback and the fixed-column arm are **separate executions**,
with separate work and timing records. Neither reuses the other's result or
intermediate tensor state. On gate acceptance, the margin arm performs exactly
two contractions; on escalation it performs three. Each shot therefore has six
contractions plus one if escalated, including the two reference views. Internal
decomposition retries are additional charged attempts within a contraction.

Use `planar_mps_coset_masses(trace_work=True)` and the existing
`certify_reference` numerical contract. Each reference mass vector must have
shape `(4,)`, finite strictly positive entries, and finite positive sum. Normalize
each separately to `c` and `r`. Define

```text
delta_probability = max_j abs(c_j - r_j)
delta_log_ratio = max_j(log(c_j)-log(r_j)) - min_j(log(c_j)-log(r_j))
margin(v) = largest(v) - second_largest(v)
```

Certification requires the same winning class, `delta_probability <= 1e-10`,
`delta_log_ratio <= 1e-8`, and **each** margin strictly greater than
`2 * delta_probability`. Thus an ambiguous exact tie fails certification even
when the two vectors agree. The target posterior is
`normalize((normalize(c) + normalize(r)) / 2)` and its class is the common
certified winner. This is a numerical two-view certificate for the unrestricted
tensor reference, not a mathematical proof of exact arithmetic.

All normalized reference probabilities and computed certificate quantities must
also be finite, with strictly positive probabilities; normalization underflow
or nonfinite log differences fail certification. Wrap the existing certificate
with these checks instead of relying on comparisons with NaN. The recorded
`reference.certified` flag additionally requires a valid reference recovery;
a numerical certificate can be retained diagnostically when recovery is invalid.

Construct each recovery with `logical_class_recovery(code, syndrome, class)`.
For sampled physical error `e` and recovery `a`, form `e XOR a`; verify that
`a` has the observed syndrome and score the residual's commutation with
`code.logicals.T` using `score_recovery`. Physical failure is syndrome invalidity
or any nonzero residual logical bit. Stabilizer-equivalent recoveries succeed;
raw recovery-string inequality is not failure. Store the full residual logical
signature as well as the failure Boolean. A different wrong logical signature
can have the same physical-failure Boolean.

## Primary estimands and exact eight-gate family

For each selected policy `a` in `{fixed_rows_tol003, margin_columns_chi8}` and
rate `r`, let `C` mean reference certification and valid reference recovery, and
`V_a` mean a valid final policy recovery. Define binary per-shot events:

```text
M_a = 1 if not C or not V_a; otherwise 1[class_a != class_reference]
O_a = 1 if not C or not V_a; otherwise 1[failure_a != failure_reference]
theta_M(a,r) = Pr_r(M_a=1)
theta_O(a,r) = Pr_r(O_a=1)
```

The second endpoint is **physical-failure Boolean discordance**, as in the pilot;
it is not logical-signature inequality, marginal BLER difference, or only added
failures. Both harm and repair transitions count as discordance. Neither
endpoint can be inferred from the other, and a policy can match a reference
that itself fails physically.

There are `2 policies × 2 rates × 2 endpoints = 8` primary upper bounds.
Bonferroni assigns `alpha = 0.05 / 8 = 0.00625` to each. For `k` events in
`n=2048` shots, the one-sided Clopper-Pearson upper bound is

```text
U(k,n) = BetaInverseCDF(1 - 0.00625; k+1, n-k), 0 <= k < n
U(n,n) = 1
U(0,n) = 1 - 0.00625**(1/n)
```

Equivalently, for `k<n`, `U` solves
`sum_{j=0}^k choose(n,j) U^j (1-U)^(n-j) = 0.00625`.
Use full floating-point precision, not displayed rounding, to gate `U <= 0.005`.

| Events k | Exact-method upper bound at n=2048 | Gate |
| ---: | ---: | --- |
| 0 | 0.0024750442291897544 | pass |
| 1 | 0.003498836715834498 | pass |
| 2 | 0.004385297715096555 | pass |
| 3 | 0.005205093018026918 | fail |

The zero-event minimum sample size is
`ceil(log(0.00625)/log(0.995)) = 1013`. The chosen 2,048 shots additionally
allow two observed events per gate. This is a precision/event-budget design,
not a power claim at an assumed true risk. These numbers were independently
checked by binomial-CDF inversion and SciPy's beta quantile without sampling.

All eight bounds have simultaneous coverage at least 95% under the stated iid
within-rate sampling model; Bonferroni does not require endpoint or policy
independence. An individual policy earns `passed_discrepancy_budget` only if its
four gates pass and all execution/reference/replay eligibility checks pass.
`both_policies_passed` requires all eight gates. A predeclared individual policy
can pass when the other does not; do not select a new policy from these outcomes.
Passing does not establish zero risk, equivalence, superiority, or device safety.

## Invalid results, missing work, and execution failures

Numerical contraction invalidity is an observed result: retain exception type,
partial work, every attempted QR/SVD (including failed calls and retries), and
all planned independent actions. Catch only the solver's declared numerical
invalidity classes, including `InvalidCosetMassError`/`InvalidContractionError`;
unexpected programming/configuration errors abort the publication.

An invalid or uncertified reference counts as an event for **both endpoints of
both selected policies** and the descriptive comparator on that shot. It is
also reported separately and blocks a positive confirmation status, even if the
numerical bounds alone pass. This conservative reference-eligibility guard
keeps a failed benchmark from certifying a policy. Its target posterior, class,
and target-relative continuous metrics are null, not fabricated.

An invalid final policy recovery counts as an event for both primary endpoints
and as a policy system/physical block failure. A valid fallback after an invalid
probe is scored normally against the reference; record the invalid probe and
its work separately. Invalid-probe counts are not silently relabeled as invalid
final recoveries. Invalid recoveries may consume the primary event budget;
there is no additional zero-invalid-policy rule for the exact bounds.

No invalid, missing, or inconvenient shot is dropped or replaced. Every primary
denominator remains 2,048 on a complete run. An incomplete run, corrupt joined
shot, missing action row, unmetered attempted work, changed source/dependency,
or failed replay cannot receive a positive scientific status. Unknown work is
null with a reason, never zero. Completed measured invalid actions can have zero
work only when the trace proves failure before any metered operation.

## Comparator, coverage, and work

`fixed_columns_chi8` is the predeclared strong comparator. Report its same event
counts/rates, numerical invalidity, syndrome validity, and physical BLER. It does
not add primary selection tests. Report BLER and the full paired 2×2 physical
failure table for all three policy pairs and each policy versus the reference,
separately by rate; use all shots with invalid final recovery counted as failure.
For uncertified references, report separately the all-shot reference
system-failure rate (uncertified is failure) and the certified-only physical
BLER with its denominator. The reference paired table includes a separate
uncertified cell; do not pretend an unknown reference outcome was observed.

Report gate accept/escalate counts over 2,048, and accepted-set mismatch and
outcome-discordance counts over the accepted set. An accepted shot with failed
reference certification counts as an accepted-set event. If coverage is zero,
conditional rates are null with denominator zero. These selected-set quantities
are descriptive, not extra confirmation tests. Paired failure transitions and
BLER differences are descriptive; no superiority test is frozen here.

Work is the existing stable arithmetic model: einsum estimated FLOPs plus
estimated dense-decomposition FLOPs. For a matrix of shape `(m,n)` and
`q=min(m,n)`, SVD charges `int(4*m*n*q + 8*q**3)` and QR charges
`int(2*m*n*q - 2*q**3/3)` before the attempted call; each retry pays again.
Retain shapes, success flags, exception classes, counters, and trace entries.
The einsum estimate is the instrumented NumPy path's parsed naive FLOP count.
This is a frozen computational proxy, not measured hardware operations.

For shot `i`, record all action work and

```text
W_rows[i] = work of its rows_tol003 action
W_margin[i] = work(columns_tol01) + work(rows_tol01)
              + 1[escalated] * work(independent margin fallback columns_chi8)
W_column[i] = work(independent fixed-column columns_chi8)
W_reference[i] = work(exact_columns) + work(exact_rows)
```

Report per-shot paired values, per-rate means and totals, equal-rate means,
policy/comparator ratios of total work, and total study work including references.
There is no resumption or shared-work discount. `fixed_rows_tol003` work ratios
are descriptive; its pilot value is not a pass threshold.

The single secondary work claim is that the margin policy saves **more than
50% of the paired per-shot arithmetic work on the equal-rate mixture**. Define
`S[r,i] = 1 - W_margin[r,i]/W_column[r,i]` and
`mu_S = (E_0.10 S + E_0.15 S)/2`. This mean paired relative saving is distinct
from `1 - sum(W_margin)/sum(W_column)` when comparator costs differ. Report both
and never exchange their meanings. Negative savings remain negative.

Freeze a paired, within-rate percentile bootstrap: `B=10000`, NumPy
`Generator(PCG64(2713269656809941213))`, rows ordered by increasing index,
rates ordered `[0.10,0.15]`. For each replicate, in rate order, draw 2,048 indices
uniformly with replacement and carry each selected shot's entire paired vector
of arm costs and outcomes. Compute the mean `S` in each resampled rate and their
equal-weight mean. Use `numpy.quantile(replicates, 0.05, method="linear")` as the
one-sided 95% lower percentile bound `L`. Require **`L > 0.50`**, the margin
policy's four primary gates, and scientific eligibility for the saving claim.
Any invalid action contributing to margin/comparator work, nonpositive comparator
cost, or unmetered work suppresses this inferential claim; preserve the measured
costs and descriptive totals. Do not exclude such shots and bootstrap a subset.

Bootstrap uncertainty is approximate and assumes independent physical shots and
representative within-rate empirical distributions; it is not an exact
finite-sample guarantee. It is a secondary analysis outside the eight-bound
family. The 95% simultaneous primary statement does not cover this work claim.
Bootstrap draws are analysis-only and never supply physical errors or policies.

## Host timing and deterministic ordering

For each rate, rotate through the six lexicographic permutations of arm indices
`0=fixed_rows_tol003`, `1=margin_columns_chi8`, `2=fixed_columns_chi8`, using
`shot_index % 6`. Every order appears 341 or 342 times per rate. Within the margin
arm, always run columns probe, rows probe, then conditional fallback. Run both
references after all policy arms; reference order is columns/rows for even
indices and rows/columns for odd indices. References never feed an arm.

Measure monotonic host wall time for every action attempt, including failed
attempts, and for each whole policy call including routing and recovery
construction but excluding target/physical-error scoring. Also record total run
time, reference times, host, process/thread settings, dependency/BLAS identity,
and the exact order. Action-time sums and whole-policy times are separate.
There is no outcome-driven warmup or repeated best-of timing. Timing stays in a
separately hashed engineering sidecar and never enters deterministic results,
bootstrap selection, primary gates, or deployment-latency claims.

## Exact artifact schema and provenance

Use strict JSON: reject duplicate/extra keys, NaN/Infinity, Boolean-as-integer
values, nonfinite numerical fields, unexpected enum values, wrong vector lengths,
and noncanonical shot membership. Canonical bytes use UTF-8,
`json.dumps(..., indent=2, sort_keys=True, allow_nan=False) + "\n"`.
Schema version is `1`; extensions require a new schema and pre-opening review.

The implementation must define exact field sets for these records, with nullable
fields present even when unavailable:

| Record | Required exact keys and content |
| --- | --- |
| Shot artifact | Existing planar-shot `{config, provenance, shots}`; each shot has `shot_id, error_rate, shot_index, sampler_seed, error_bsf, syndrome`. |
| Generation manifest | `schema_version, contract_id, producer_commit, config_bindings, source_sha256, runtime, code_identity, historical_domains, release_binding, shots_binding`; binds the confirmation wrapper's full provenance alongside the unchanged sampler artifact. |
| Confirmation raw | `schema_version, contract_id, status, scientific_eligible, config, provenance, shot_provenance, shots, summary`. |
| Evaluation shot | `shot_id, error_rate, shot_index, sampler_seed, error_bsf, syndrome, arm_order, reference_order, actions, policies, reference`. |
| Action | `invocation_id, action_id, mode, chi, tol, valid, exception_type, exception_message, masses, probabilities, selected_class, work, estimated_arithmetic_flops`. |
| Policy | `policy_id, action_invocation_ids, gate_accepted, selected_class, recovery_bsf, syndrome_valid, logical_signature, physical_failure, valid_recovery, class_mismatch, outcome_discordance, estimated_arithmetic_flops`. |
| Reference | `action_invocation_ids, certified, certificate, exception_type, exception_message, probabilities, selected_class, recovery_bsf, syndrome_valid, logical_signature, physical_failure, estimated_arithmetic_flops`. |
| Compact | `schema_version, contract_id, status, scientific_eligible, provenance, raw_sha256, raw_size_bytes, summary`. |
| Summary | `sample_counts, invalid_counts, primary, policies, comparator, coverage, paired_physical, work, decision`. |
| Primary row | `policy_id, error_rate, endpoint, events, shots, alpha, upper_bound, threshold, gate_passed`; exactly eight ordered rows. |
| Timing | `schema_version, status, measurement, host, provenance, raw_sha256, summary_sha256, shots, total_host_wall_seconds`; per-shot timing binds shot ID, arm/reference orders, each invocation and whole-policy duration. |
| Manifest | `schema_version, contract_id, producer_commit, artifacts, configs, sources, commands, verification`; file entries have `path, size_bytes, sha256`. |

For `gate_accepted`, use a Boolean only for the margin arm and null for the other
arms. For an uncertified reference, `physical_failure` is null. Invalid final
policy recovery has `physical_failure=true`, nullable class/recovery/signature,
and both primary events true. Deterministic invocation IDs are
`{policy_id}/{action_id}` and `reference/exact_{mode}`; references to IDs must
resolve once and only once. Action arrays follow actual execution order; summary
rows follow policy order `[fixed_rows_tol003, margin_columns_chi8]`, rate order
`[0.10,0.15]`, endpoint order `[class_mismatch,outcome_discordance]`. Sort raw shots
by rate then index regardless of scheduling.

Nested certificate and work records preserve the existing certificate and stable
work schema, including all decomposition attempts; exact field validation must
be pinned in code and tested before release. The config fixes every constant in
this contract, including action table, counts, domains, thresholds, ordering,
bootstrap algorithm, and invalidity/status rules. No runtime scientific overrides
are permitted. The plan specifies exact nested summary/provenance field sets.

The immutable run status is `complete_pending_replay` (or
`reduced_non_scientific` for fixtures). Before independent replay, a numerical
policy pass is labeled `gates_passed_pending_replay`, never a final confirmation
pass. A separate immutable verification receipt promotes eligible passes to
`passed_discrepancy_budget` after every required replay check. Failed and blocked
statuses remain visible; raw/compact results are never edited to insert a later
verification claim. `scientific_eligible` in raw output records execution
eligibility only and cannot substitute for that receipt.

Provenance must bind clean producer commit, committed config content and hashes,
all transitive scientific source files (including CLI, sampler, scoring,
contraction, summary, and serialization), `pyproject.toml`, `uv.lock`, installed
Python/NumPy/SciPy/qecsim/networkx versions, matching backend and optional binary
hash, code/stabilizer/logical identity, historical-domain inventory, review
release receipt, and input shot size/hash. Recheck these before publication.
Config or dependency mismatch cannot be repaired by editing an artifact hash.

## Transactionality, review, replay, and decisions

Before generation, two independent reviews must accept (1) this scientific
contract and plan, and (2) committed implementation/configs plus fixture-only
red/green tests, without unresolved blocker/major findings. A release receipt
binds their reports, exact source/config/dependency identities, contract digest,
unopened domain, and full `4096`-shot scope. Preflight reads metadata only and
must not derive a scientific coordinate seed. Passing tests is not release.

Generation and evaluation are separate immutable directory publications. Write
all outputs to unique sibling staging directories, flush/validate their complete
schemas/hashes, recheck provenance, and publish by one atomic rename guarded
against existing destinations, symlinks, and concurrent publishers. Never
overwrite an output. An exception leaves no published partial directory;
preserve an out-of-band failure receipt and retire abandoned staging as
non-scientific. An output with incomplete membership cannot masquerade as a
completed study. No resume-by-skipping or post-outcome source repair is allowed.
An engineering interruption can be replayed from the identical immutable input
under unchanged approved source; any scientific change needs a new reviewed
version/domain. Publish no positive verdict until the full run and replay finish.

The shot directory contains `planar_shots.json` and `generation_manifest.json`.
The first preserves the sampler's existing producer schema; the second binds
the confirmation wrapper, release, and all additional scientific dependencies.
The run must validate and bind both before accepting any shot.

Replay must regenerate every joined shot from its recorded identity; independently
recompute syndrome/recovery/logical outcomes, certification, action membership,
charged attempt totals, all eight CP bounds, bootstrap replicates/bound, coverage,
paired tables, and decisions from raw rows. A full independent contraction rerun
under the bound source/runtime must reproduce deterministic scientific bytes.
Validate manifest file closure, sizes/hashes, source/config bindings, and
byte-identical raw-derived compact summary. Timing bytes are never expected to
repeat; validate their finite nonnegative values, identity, coverage, and nesting.
Compact publication without the raw artifact is not a complete replay record;
report raw availability honestly.

The cheapest useful pre-release disconfirmation is a fixture-only injected
counterexample: one accepted shared wrong class, one invalid fallback, one
uncertified reference, and a primary table with three events. It must defeat
the corresponding status and retain denominators/work. No fresh-domain preview
is permitted. After release, the first scientific execution is the full frozen
two-rate run; it already excludes the pilot's 14-action grid and costs only the
three selected arms plus references. At any intermediate point a third event
makes that final gate mathematically unable to pass, but **does not stop or alter
the fixed run** and is not a sequential confidence statement.

After full execution: publish all eight bounds and all comparator/descriptive
outcomes. A failed primary gate stops that policy's advancement. Failed reference
certification, provenance, completeness, or replay blocks positive scientific
status. A passed primary policy with a failed secondary work gate receives only
the discrepancy-budget statement. Both passing policies do not establish that
one is superior. New tuning or controller work requires a separate development
question and a newly frozen confirmation domain, not continued sampling here.

Claims apply only to these two simple policies, distance five, the two specified
iid code-capacity rates, and the certified reference under the recorded numerical
environment. They establish no threshold, circuit-level or repeated-round
performance, qLDPC/code-family transfer, physical-device safety, learned-model
benefit, or deployment/FPGA latency. The broader
[experiment methodology](experiment-methodology.md) supplies the existing
accuracy-before-speed and joined-shot principles; its LP-code and temporal
campaigns are separate experiments and are not evidence for this confirmation.
