# Simple Planar Confirmation Design

**Status:** frozen contract-writing deliverable, 2026-09-22. This design does not
implement or open the confirmation domain. The normative reader-facing
[contract](../../adaptive-computation-simple-confirmation-contract.md) and this
design must pass preregistration review before any scientific coordinate seed is
derived. The [implementation plan](../plans/2026-09-22-simple-planar-confirmation.md)
is a future TDD handoff, not authorization to run scientific data now.

## Decision and scope

The [128-shot pilot](../../adaptive-computation-intervention-pilot.md) found
posterior-TV opportunities without incremental logical-decision value, a
post-hoc inexpensive `rows_tol003` comparator, and host timing that contradicted
arithmetic savings. Stop learned-controller work under that objective. Freeze a
comparison of two selected simple policies, with fixed columns chi8 as the
strong comparator, on fresh distance-five planar iid depolarizing code-capacity
shots at `p=.10,.15`. No TV oracle, FNO, HiPPO, state reuse, or timing-dependent
decision enters the experiment.

`fixed_rows_tol003` is selected post hoc, so the pilot cannot confirm it. The
unchanged `margin_columns_chi8` first runs columns/rows `tol=.01`, accepts only
valid agreement with minimum winning-class margin strictly greater than
`0.30710401263493464`, and otherwise runs independent columns `chi=8`. Fixed
rows uses one rows `tol=.003`; the comparator uses independent columns `chi=8`
on every shot. Two unrestricted `chi=None,tol=None` views certify the target.
No other approximate action is run.

## Data, references, and estimands

Freeze 2,048 physical errors at each rate, 4,096 paired shots total, from domain
`qldpc-fno/simple-planar-confirmation/v1` (campaign identity
`2409828766515432030`). Tests use only
`qldpc-fno/simple-planar-confirmation/test-fixture/v1`
(`3697382327853009454`); bootstrap analysis uses a third separate domain
`qldpc-fno/simple-planar-confirmation/bootstrap/v1`
(`2713269656809941213`). These integers hash the domain strings only. No physical
coordinate seed is derived during contract writing or tests. Keep seed, sampled
82-bit error, derived 40-bit syndrome, rate, index, and shot ID joined.

The preflight inventories every historical config domain and retired pilot
v1/v2. The implementation checks seed collisions only after release and before
generation. Uncertain pre-release exposure retires this domain. The central
test fixture guard must fail before physical coordinate derivation for any
scientific confirmation domain; lowering a production count is prohibited.

For reference probability vectors `c,r`, certification requires finite positive
four-class masses, a common winner, `max(abs(c-r)) <= 1e-10`, log-ratio spread
`max(log(c)-log(r))-min(log(c)-log(r)) <= 1e-8`, and each winning margin strictly
greater than twice the maximum probability discrepancy. The target is the
renormalized mean of separately normalized views. Use the existing audited
`certify_reference`; preserve its ambiguity guard. This is numerical
certification, not an exact-arithmetic proof.
Explicitly reject normalized-probability underflow and nonfinite certificate
quantities before accepting its result, and require a syndrome-valid reference
recovery before recording the final reference as certified.

Recoveries use qecsim `I,X,Y,Z` classes. Score `sampled_error XOR recovery` by
stabilizer and logical commutation using `logical_class_recovery` and
`score_recovery`. Degenerate equivalent recoveries can succeed. Record logical
signature and physical failure separately; Boolean outcome discordance does not
mean raw-recovery or logical-signature inequality.

For each selected policy and rate, the two primary probabilities are (1)
reference-class mismatch and (2) inequality of physical-failure Booleans. If the
reference is invalid/uncertified or the policy's final recovery is invalid, both
events are one. All complete denominators remain 2,048. A failed probe followed
by a valid fallback is scored by its final recovery, while retaining the failed
probe and work. Uncertified references also block positive scientific status.

Eight one-sided Clopper-Pearson upper bounds use Bonferroni
`alpha=.05/8=.00625`; each passes only at `U<=.005`. For `k<n`,
`U=BetaInverseCDF(.99375;k+1,n-k)`; for `k=n`, `U=1`. At `n=2048`:

| Events | Upper bound | Result |
| ---: | ---: | --- |
| 0 | 0.0024750442291897544 | pass |
| 1 | 0.003498836715834498 | pass |
| 2 | 0.004385297715096555 | pass |
| 3 | 0.005205093018026918 | fail |

`ceil(log(.00625)/log(.995))=1013` suffices only for a zero-event design;
2,048 additionally tolerates two events. This is not a power claim. A policy
passes only its four gates plus eligibility; all eight are required to claim
both policies pass. The exact 95% simultaneous coverage concerns the eight
primary probabilities only and assumes iid shots within each stratum.

## Secondary work, baselines, and timing

Report all arms' BLER, invalidity, descriptive paired physical-failure tables,
gate coverage, accepted-set mismatch/discordance with denominators, and
fully charged work. The strong comparator is not a ninth primary test. No
superiority, noninferiority-in-BLER, or equivalence test is frozen. Reference
uncertainty is a separate table cell and never an invented physical outcome.

Use existing stable arithmetic traces, including failures and retries. Each
SVD on `(m,n)` charges `int(4*m*n*q+8*q**3)` and QR charges
`int(2*m*n*q-2*q**3/3)`, `q=min(m,n)`, before its call; add the frozen einsum
estimate. Margin cost always includes both probes and, when escalated, a complete
independent fallback. Reference work belongs in total study cost. Unknown work
is null and blocks scientific eligibility, never a silent zero.

Freeze the margin-only secondary saving estimand
`mu=(E_.10[1-W_margin/W_column]+E_.15[1-W_margin/W_column])/2`.
Resample complete paired physical-shot vectors independently within each rate
2,048 times per stratum for each of 10,000 bootstrap replicates, using
`Generator(PCG64(2713269656809941213))`; draw rate .10 then .15 in each replicate.
The one-sided 95% lower percentile bound is the linear `.05` quantile of those
replicate equal-rate means. Claim more than 50% arithmetic saving only if this
bound is **strictly greater than .50**, the margin primary gates pass, and all
contributing actions have valid, fully metered costs and positive comparator
cost. Preserve negative savings; never bootstrap a complete-case subset.
This is approximate uncertainty, outside the exact primary multiplicity family.
Fixed rows work and aggregate cost ratios are descriptive; neither uses the
pilot's development cost as a pass threshold.

Counterbalance three policy blocks with six lexicographic permutations indexed
by `shot_index%6`, independently within rate. Each occurs 341 or 342 times.
The margin block retains fixed probe/fallback order; the fixed and fallback
column actions execute independently. Alternate reference order by index parity
after the three arms. Six plus an optional seventh contraction occur per shot.
Measure actions and entire policy calls separately, including failed attempts,
and record total run time and host/thread/BLAS metadata. Policy timing includes
decision/recovery construction but not scoring against physical truth.
Nondeterministic timing lives outside scientific bytes and never selects an
action, gates correctness, or becomes deployment latency.

## Architecture and implementation boundaries

Keep the pilot and its evidence unchanged. Reuse the existing sampler, tensor
instrumentation, reference certificate, physical scoring, and canonical JSON
semantics through a new confirmation path with these responsibilities:

| File to create | Responsibility |
| --- | --- |
| `src/qldpc_fno/decision/simple_confirmation.py` | Frozen config/constants, strict schemas, pure endpoint/CP/bootstrap summaries and decisions. |
| `src/qldpc_fno/decision/simple_confirmation_runner.py` | Three independent arm executions, reference certification, recovery scoring, complete action/work records, isolated timing. |
| `src/qldpc_fno/decision/simple_confirmation_io.py` | Metadata-only preflight, release receipt validation, generation wrapper, provenance, transactional publication, independent replay. |
| `experiments/36_run_simple_planar_confirmation.py` | `preflight`, `generate`, `run`, `verify` CLI subcommands; no scientific overrides. |
| `configs/simple_planar_confirmation.json` | Complete frozen analysis/action/domain configuration. |
| `configs/simple_planar_confirmation_shots.json` | Existing seven-field sampler config with 2,048 shots per rate. |
| `configs/simple_planar_confirmation_fixture_shots.json` | Reserved test-domain sampler config with two shots per rate. |

Tests are split into pure statistics/schema, execution/scoring, IO/firewall,
and CLI integration modules, with a global test-only coordinate-seed guard.
Fixture mode is explicit, uses its own config, and hard-disables every positive
scientific decision. Production config parsing is permitted in tests, physical
production generation is not. No generic sampler weakening is permitted.

The runner accepts only syndrome/rate and frozen action parameters for policy
execution; exact reference and physical error enter a separate scoring stage.
Pure summaries consume recorded rows, not solver callbacks. Replay independently
rebuilds scientific rows and compares canonical bytes, so a tampered summary
cannot authenticate itself. All transitive source/dependency/config identities
and code/logical-basis identities are bound at generation, execution, and replay.

## Publication and failure semantics

The contract's exact record schemas cover joined shots, action invocations,
policy outcomes, reference certificate, eight primary rows, compact summary,
timing, and manifest. The plan fixes nested summaries and release receipt keys.
Reject unknown/duplicate keys, wrong types or lengths, NaN/Infinity, unresolved
action IDs, omitted required nulls, duplicate or missing shots, altered physical
errors, corrupt work, and source/config/runtime drift.

Numerical invalidity is recorded with full attempted work and does not erase a
shot. Unexpected software errors, missing traces, corruption, incomplete sample,
or identity changes abort without a positive publication. Both reference views
and all policy arms are attempted unless an unexpected fatal error prevents it.
Never replace invalid actions or extend the sample to improve the result.

Generate and evaluate into independent immutable directories. Stage complete
outputs beside their destination, verify schemas/hashes and unchanged provenance,
then atomically rename while refusing any existing destination or symlink.
Serialize competing publishers with exclusive ownership of the destination;
an unchecked rename is insufficient. On failure leave no published partial
output, preserve an external failure receipt, and never resume by skipping
already observed troublesome shots. Identical-source restart from immutable
inputs is allowed; a scientific change after opening requires a new contract
version and domain.

Replay every sampler seed/error/syndrome, every contraction and recovery,
certification, work accounting, endpoint, bootstrap statistic, and summary.
The deterministic raw and compact bytes must match under the pinned runtime;
timing need only pass identity/finite-value/coverage checks. Bind all outputs
and commands in a manifest and disclose where full raw artifacts are available.

## Review gates and stop/go rules

1. Independent scientific review must accept the contract, this design, sample
   calculation, and plan before implementation release can be considered.
2. Complete fixture-only TDD, regression checks, source/config commits, and
   independent implementation/preregistration review. No unresolved blocker or
   major finding is allowed. A metadata-only receipt binds exact reviewed
   identities and certifies the domain as unopened.
3. Only after that release may the production wrapper derive coordinates and
   generate the complete 4,096-shot artifact. Execute the full frozen run;
   stop neither for apparent success nor on a third event.
4. Require independent artifact/replay review before publishing positive
   confirmation prose. Numerical gate failures remain reportable failures.

The cheapest pre-release disconfirming run uses synthetic fixture rows: accepted
shared error, invalid fallback, uncertified reference, and a three-event primary
cell must defeat the relevant gates without losing denominators/work. The first
fresh-domain run is the full minimal-arm confirmation, not a pilot preview.
Three events already imply that a final 2,048-shot gate cannot pass; this fact
does not authorize optional stopping or sequential inference.

A failed selected-policy gate stops that policy's advancement. A passed policy
with no secondary work pass earns only the bounded discrepancy statement.
Reference, provenance, completeness, or replay failures block all positive
confirmation statuses. No outcome reopens learned-controller work. Claims do not
extend to zero risk, device safety, superiority, equivalence, circuit noise,
repeated rounds, code thresholds, qLDPC transfer, FNO/HiPPO benefit, or hardware
latency.
