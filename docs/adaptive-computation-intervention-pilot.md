# Adaptive intervention pilot: posterior opportunity does not establish controller value

This 128-shot development pilot found heterogeneous improvements in posterior
total variation (TV), but no evidence that a learned controller improves decoding.
The unchanged margin gate accepted 111/128 shots; adding the fixed column
`chi=8` fallback produced no reference-class mismatches in these development
shots. That observation is not a safety result. A post-hoc fixed
`rows_tol003` comparator used less estimated work than the composite policy,
and measured host action times reversed the apparent advantage over fixed
column `chi=8`. The next experiment should compare those simple policies on
fresh, independently frozen data. Learned-controller work under the current TV
objective does not advance.

The [compact evidence](../evidence/adaptive-computation/intervention-pilot-v3/summary.json)
is copied byte-for-byte from the frozen run. Its
[manifest](../evidence/adaptive-computation/intervention-pilot-v3/manifest.json)
binds the joined shots, full raw result, source summary, timing sidecar, tracked
summary, source commit, configs, and execution commands. The full raw files
remain outside Git at the recorded local paths; no public upload or downloadable
raw archive is claimed.

## Why this experiment follows the failed consensus screen

A tensor decoder estimates the posterior mass of each logical class and chooses
a class from those masses. Agreement between two approximations can conceal a
shared mistake. The correctly paired
[physical-shot screen](planar-shot-accuracy-results.md) exposed eight such
accepted agreements that changed the realized logical outcome. A subsequent
[diagnostic audit](adaptive-computation-diagnostic-audit.md) selected minimum
probe margin as a useful development diagnostic, without confirming its safety.

This pilot asks a narrower question: on shots that the frozen margin gate would
escalate, do different contraction actions improve the reference posterior on
different shots after charging every independent run? The
[frozen design](superpowers/specs/2026-09-22-adaptive-intervention-pilot-design.md)
and [implementation plan](superpowers/plans/2026-09-22-adaptive-intervention-pilot.md)
predeclare the sample, action order, reference, gate, labels, and opportunity
rule. The pilot measures a mechanism; all 128 shots remain development data
forever, including for any policy chosen after seeing this report.

## Methods and denominators

The sample is qecsim distance-five planar code under iid depolarizing
code-capacity noise: 64 independently seeded physical errors at each of
`p=0.10` and `p=0.15`. The sampler seed, error, and syndrome remain joined.
The domain is `qldpc-fno/adaptive-intervention-pilot/v3`, with campaign seed
`10044421296420932682`. Earlier v1 and v2 identities were retired before this
run because their unopened status could not be established; tests use a
separate fixture domain. No post-outcome action/config/sample change was made.

Each shot receives 14 approximate actions: columns and rows with
`chi=2,4,8,16`, then columns and rows with `tol=0.03,0.01,0.003`, in that
literal interleaved order. The IDs `tol03`, `tol01`, and `tol003` mean
those three tolerances respectively. Two additional unrestricted contractions
provide reference views. Both must be valid, choose the same logical class, and
meet probability tolerance `1e-10` and log-ratio tolerance `1e-8`. The target
is their separately normalized probability vectors' mean, normalized again.
Physical success is scored by residual logical commutation after combining the
recovery with the actual sampled error; raw recovery-string identity is not
the criterion.

The initial state runs `columns_tol01` and `rows_tol01`. The gate accepts
only if both are valid, agree on class, and their minimum winning-class margin
is strictly greater than `0.30710401263493464`. Equality or disagreement
escalates. A composite candidate pays for both probes plus its complete
independent run, without a resumption discount. The offline gate/fallback policy
uses that accepted class or a separate `columns_chi8` result on escalation.

Candidate TV gain is baseline TV minus candidate TV; the baseline is the
better-TV probe selected with hindsight using the reference. This baseline
selection is a label, not an inference-time rule. The 12 candidates exclude
the two probes. The accuracy oracle maximizes TV gain; the efficiency oracle
maximizes gain divided by fully charged composite FLOPs. Both require gain
greater than `1e-6`; negative gains remain recorded. Numerical ties use
relative tolerance `1e-12`, zero absolute tolerance, then lower work and
literal action order. Unique accuracy separation requires a gain gap greater
than `1e-6`; unique efficiency separation requires more than 1% over the
eligible positive runner-up (or no other positive candidate).

One physical shot is the independent statistical unit. The 14 action rows on
it are paired repeated measurements. Every endpoint is reported separately by
rate and as the equal-weight mean of the two rate-level estimates. For
unconditional endpoints, equal counts of 64 mean the pooled 128-shot rate is
also that equal-rate mean. Escalated samples have sizes 4 and 13, so their
equal-rate conditional means must not be replaced with a pooled 17-shot mean.
This development report includes no confidence intervals, bootstrap,
significance test, or population-risk bound.

All 128/128 joined shots replayed exactly and 128/128 references certified.
All 2,048/2,048 contractions were valid: 1,792 approximate actions and 256
reference views, with zero invalid actions or references at either rate.
All 352,256 recorded decomposition attempts reconciled; invalid charged work
is zero. Every action therefore has 64/64 valid attempts per rate and 128/128
overall. Posterior and reference-discordance denominators are also 64 and 128.
Invalid actions would count as system failures and retain their partial work;
uncertified references would remain in total denominators and block advancement.

## Gate and fixed-policy comparisons

These are retrospective evaluations on the development shots, not a new
confirmation test. Both the gate/fallback and fixed column `chi=8` match the
reference class and realized outcome on every shot; they still fail physically
on the reference's 11 sampled logical failures.

| Endpoint | p=.10 | p=.15 | Equal-rate / total |
| --- | --- | --- | --- |
| Gate accepts | 60/64 (93.75%) | 51/64 (79.6875%) | 111/128 (86.71875%) |
| Escalates | 4/64 | 13/64 | 17/128 |
| Accepted reference mismatches | 0/60 | 0/51 | 0/111 |
| Gate + column chi8 reference mismatches | 0/64 | 0/64 | 0/128 |
| Gate + column chi8 outcome discordances | 0/64 | 0/64 | 0/128 |
| Gate + column chi8 physical failures | 3/64 | 8/64 | 11/128 |
| Fixed column chi8 physical failures | 3/64 | 8/64 | 11/128 |
| Gate + column chi8 estimated FLOPs | 39,155,459 | 96,652,593 | 135,808,052 total |
| Fixed column chi8 estimated FLOPs | 330,277,632 | 330,277,632 | 660,555,264 total |
| Composite / fixed work | 0.1185531662 | 0.2926404444 | 0.2055968053 |
| Fixed rows_tol003 reference mismatches | 0/64 | 0/64 | 0/128 |
| Fixed rows_tol003 outcome discordances | 0/64 | 0/64 | 0/128 |
| Fixed rows_tol003 physical failures | 3/64 | 8/64 | 11/128 |
| Fixed rows_tol003 estimated FLOPs | 12,717,781 | 27,895,189 | 40,612,970 total |

The stronger simple comparator is `rows_tol003`, identified post hoc from
this grid. It used 29.9047% of the composite policy's total estimated FLOPs
while matching its observed class and physical outcomes. It is not a confirmed
winner: selecting it on these labels makes a new frozen, fresh-data comparison
necessary. Zero development mismatches cannot establish a safety certificate.

## Heterogeneous TV opportunity and its counterexamples

The efficiency oracle has a uniquely separated positive winner on all 17/17
escalated shots, spanning six action IDs. The minimum relative efficiency
separation is 3.011588635%, above the frozen 1% criterion. The accuracy oracle
chooses `columns_chi16` on 15 shots and `rows_chi16` on two, mostly reflecting
near-reference posterior fidelity. These are hindsight opportunities, not
executable controllers.

| Candidate | Accuracy wins p=.10 | p=.15 | Total | Efficiency wins p=.10 | p=.15 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| `columns_chi2` | 0/4 | 0/13 | 0/17 | 1/4 | 0/13 | 1/17 |
| `rows_chi2` | 0/4 | 0/13 | 0/17 | 1/4 | 0/13 | 1/17 |
| `columns_chi4` | 0/4 | 0/13 | 0/17 | 1/4 | 0/13 | 1/17 |
| `rows_chi4` | 0/4 | 0/13 | 0/17 | 1/4 | 4/13 | 5/17 |
| `columns_chi8` | 0/4 | 0/13 | 0/17 | 0/4 | 0/13 | 0/17 |
| `rows_chi8` | 0/4 | 0/13 | 0/17 | 0/4 | 0/13 | 0/17 |
| `columns_chi16` | 4/4 | 11/13 | 15/17 | 0/4 | 0/13 | 0/17 |
| `rows_chi16` | 0/4 | 2/13 | 2/17 | 0/4 | 0/13 | 0/17 |
| `columns_tol03` | 0/4 | 0/13 | 0/17 | 0/4 | 0/13 | 0/17 |
| `rows_tol03` | 0/4 | 0/13 | 0/17 | 0/4 | 0/13 | 0/17 |
| `columns_tol003` | 0/4 | 0/13 | 0/17 | 0/4 | 3/13 | 3/17 |
| `rows_tol003` | 0/4 | 0/13 | 0/17 | 0/4 | 6/13 | 6/17 |

The count table conditions on escalation. For equal-rate unconditional win
rates, the exact values are in `summary.equal_rate.actions`; e.g.
`rows_tol003` wins on 0/64 and 6/64, giving 3/64 = 0.046875. Each count is
one shot's paired comparison, not an independent action observation.

The following descriptive grouping uses only frozen pre-action features. It
does not fit a decision rule, search feature subsets, or establish that a
deployable controller can recognize the winners. Margins above the threshold
can occur among escalations when probe classes disagree.

| Efficiency winner | Wins | Probe-class agreements | Syndrome weight range | Minimum probe margin range |
| --- | --- | --- | --- | --- |
| columns_chi2 | 1 | 1/1 | 9 | 0.293631 |
| rows_chi2 | 1 | 0/1 | 8 | 0.517103 |
| columns_chi4 | 1 | 1/1 | 11 | 0.152954 |
| rows_chi4 | 5 | 4/5 | 6–18 | 0.013297–0.766879 |
| columns_tol003 | 3 | 3/3 | 7–14 | 0.025282–0.089665 |
| rows_tol003 | 6 | 3/6 | 12–17 | 0.033193–0.229128 |

The frozen feature set also includes rate, each probe's selected class and
margin, cross-probe TV/JS, stable work, and compact spectral summaries.
Those per-shot features remain in the raw artifact, separated from labels;
reference posteriors and physical outcomes are never inference-visible inputs.

There are 48 candidate rows on the four escalated p=.10 shots and 156 on the
13 escalated p=.15 shots. Respectively 33 and 91 gains exceed `1e-6`; 15
and 65 gains are negative. Across all 204 candidate rows there are **zero
reference-class repairs and 18 mismatch introductions** (3 and 15 by rate);
all 18 also change the realized physical outcome relative to the baseline.
The hindsight better-TV probe already has the correct reference class on
17/17 escalated shots. Thus variation in posterior approximation efficiency
does not establish incremental logical-decision value.

The strongest beneficial TV example is `d5/p0.100000/i000017`:
`columns_chi16` reduces TV by `0.2192573557773075`, while the efficient
`rows_chi2` winner gains `0.213274387263907` at 410,759 composite FLOPs.
The baseline already chooses the correct class. On the very same shot,
`columns_chi2` gains `0.15307654402077892` and `columns_tol003` gains
`0.08386137408278535`, yet both turn that correct baseline into the wrong
logical class and a physical failure. A smaller distance between probability
vectors can change the winning class in the harmful direction.

The strongest harmful escalated example is
`d5/p0.150000/i000026`: `columns_tol03` has gain
`-0.7221565719124328` and introduces a class mismatch and physical failure.
Among escalated p=.10 shots, the largest TV harm is `rows_chi2` on
`d5/p0.100000/i000027`, gain `-0.39298193769461964`, although its class
remains matched. Posterior fidelity, reference-class agreement, and physical
success are distinct endpoints and must remain distinct.

Mean gain and fully charged independent-run composite work below condition on
escalation (4 and 13 observations); equal-rate means weight the two strata
equally. The compact summary preserves every paired gain/work value, including
negative gains.

| Candidate | Mean TV gain p=.10 | p=.15 | Equal-rate | Mean composite FLOPs p=.10 | p=.15 | Equal-rate |
| --- | --- | --- | --- | --- | --- | --- |
| `columns_chi2` | -0.00437430414 | -0.165850506 | -0.085112405 | 468257.750 | 802428.154 | 635342.952 |
| `rows_chi2` | -0.0511757482 | -0.0740863093 | -0.0626310288 | 468257.750 | 802428.154 | 635342.952 |
| `columns_chi4` | 0.0591505949 | 0.0107416903 | 0.0349461426 | 1282277.750 | 1616448.154 | 1449362.952 |
| `rows_chi4` | 0.05896183 | 0.0147198307 | 0.0368408304 | 1282277.750 | 1616448.154 | 1449362.952 |
| `columns_chi8` | 0.0604816867 | 0.0201787459 | 0.0403302163 | 5452831.750 | 5787002.154 | 5619916.952 |
| `rows_chi8` | 0.0604805723 | 0.0203339527 | 0.0404072625 | 5452831.750 | 5787002.154 | 5619916.952 |
| `columns_chi16` | 0.0604817827 | 0.0203910382 | 0.0404364105 | 15589287.750 | 15923458.154 | 15756372.952 |
| `rows_chi16` | 0.0604817827 | 0.0203910382 | 0.0404364105 | 15589287.750 | 15923458.154 | 15756372.952 |
| `columns_tol03` | -0.158114338 | -0.147749528 | -0.152931933 | 373716.250 | 834771.769 | 604244.010 |
| `rows_tol03` | -0.0158563799 | -0.0474904008 | -0.0316733903 | 407304.000 | 828588.462 | 617946.231 |
| `columns_tol003` | 0.0151065832 | 0.00752250754 | 0.0113145453 | 451887.500 | 1186735.385 | 819311.442 |
| `rows_tol003` | -0.00173555552 | 0.0078354127 | 0.00304992859 | 539254.250 | 1287556.077 | 913405.163 |

## Complete action endpoints

For these observations, each action's reference-class mismatch indicators equal
its physical-outcome-discordance indicators; the first three columns therefore
report both endpoints, not just one. Physical failure is a separate measure:
an approximation can disagree with the reference and occasionally succeed on
a sampled error that the reference fails. That does not demonstrate lower
population risk. Exact counts below are never extrapolated to safety.

| Action | Mismatch/discordance p=.10 | p=.15 | Equal-rate | Physical failures p=.10 | p=.15 | Equal-rate |
| --- | --- | --- | --- | --- | --- | --- |
| `columns_chi2` | 1/64 | 5/64 | 6/128 | 4/64 | 11/64 | 15/128 |
| `rows_chi2` | 1/64 | 6/64 | 7/128 | 4/64 | 8/64 | 12/128 |
| `columns_chi4` | 0/64 | 1/64 | 1/128 | 3/64 | 7/64 | 10/128 |
| `rows_chi4` | 0/64 | 1/64 | 1/128 | 3/64 | 7/64 | 10/128 |
| `columns_chi8` | 0/64 | 0/64 | 0/128 | 3/64 | 8/64 | 11/128 |
| `rows_chi8` | 0/64 | 0/64 | 0/128 | 3/64 | 8/64 | 11/128 |
| `columns_chi16` | 0/64 | 0/64 | 0/128 | 3/64 | 8/64 | 11/128 |
| `rows_chi16` | 0/64 | 0/64 | 0/128 | 3/64 | 8/64 | 11/128 |
| `columns_tol03` | 1/64 | 4/64 | 5/128 | 4/64 | 10/64 | 14/128 |
| `rows_tol03` | 0/64 | 2/64 | 2/128 | 3/64 | 10/64 | 13/128 |
| `columns_tol01` | 1/64 | 3/64 | 4/128 | 4/64 | 9/64 | 13/128 |
| `rows_tol01` | 0/64 | 1/64 | 1/128 | 3/64 | 9/64 | 12/128 |
| `columns_tol003` | 1/64 | 0/64 | 1/128 | 4/64 | 8/64 | 12/128 |
| `rows_tol003` | 0/64 | 0/64 | 0/128 | 3/64 | 8/64 | 11/128 |

The next table gives the remaining frozen posterior endpoints. TV is half the
L1 distance; Jensen–Shannon (JS) divergence uses natural logarithms, in nats.
Reference-conditional excess risk is the reference winning-class mass minus the
mass of the action's selected class. All per-rate means use 64 certified shots;
the equal-rate row averages those means. Values are rounded for readability;
the tracked JSON retains full precision and per-shot distributions in ascending
shot-index order within each rate, preserving their pairing.

| Action | Rate | Mean TV | Mean JS (nats) | Mean excess risk |
| --- | --- | --- | --- | --- |
| `columns_chi2` | 0.10 | 0.00682202341 | 0.000664924381 | 0.00127181983 |
| `columns_chi2` | 0.15 | 0.0712947039 | 0.0194303152 | 0.0245547879 |
| `columns_chi2` | equal | 0.0390583637 | 0.0100476198 | 0.0129133038 |
| `rows_chi2` | 0.10 | 0.030590317 | 0.012865719 | 0.0133772597 |
| `rows_chi2` | 0.15 | 0.0564658836 | 0.0116736415 | 0.0213351563 |
| `rows_chi2` | equal | 0.0435281003 | 0.0122696802 | 0.017356208 |
| `columns_chi4` | 0.10 | 0.000328462864 | 6.2881825e-06 | 0 |
| `columns_chi4` | 0.15 | 0.00489905386 | 0.000193827475 | 0.00021166165 |
| `columns_chi4` | equal | 0.00261375836 | 0.000100057829 | 0.000105830825 |
| `rows_chi4` | 0.10 | 0.00078924971 | 3.94392407e-05 | 0 |
| `rows_chi4` | 0.15 | 0.00346317135 | 0.00013115105 | 0.00021166165 |
| `rows_chi4` | equal | 0.00212621053 | 8.52951452e-05 | 0.000105830825 |
| `columns_chi8` | 0.10 | 1.10021005e-07 | 2.93499079e-12 | 0 |
| `columns_chi8` | 0.15 | 6.9403301e-05 | 2.20770456e-07 | 0 |
| `columns_chi8` | equal | 3.4756661e-05 | 1.10386696e-07 | 0 |
| `rows_chi8` | 0.10 | 4.25708959e-07 | 1.21857007e-11 | 0 |
| `rows_chi8` | 0.15 | 1.73157222e-05 | 3.72860131e-09 | 0 |
| `rows_chi8` | equal | 8.87071557e-06 | 1.8703935e-09 | 0 |
| `columns_chi16` | 0.10 | 1.09810422e-15 | 1.16430574e-17 | 0 |
| `columns_chi16` | 0.15 | 4.71958856e-15 | 1.56385104e-17 | 0 |
| `columns_chi16` | equal | 2.90884639e-15 | 1.36407839e-17 | 0 |
| `rows_chi16` | 0.10 | 4.32816568e-15 | 7.67096929e-18 | 0 |
| `rows_chi16` | 0.15 | 4.96354873e-15 | 2.94896385e-17 | 0 |
| `rows_chi16` | equal | 4.6458572e-15 | 1.85803039e-17 | 0 |
| `columns_tol03` | 0.10 | 0.0230240862 | 0.00478892027 | 0.00127181983 |
| `columns_tol03` | 0.15 | 0.051467996 | 0.0125008016 | 0.0223233593 |
| `columns_tol03` | equal | 0.0372460411 | 0.00864486096 | 0.0117975896 |
| `rows_tol03` | 0.10 | 0.0165396425 | 0.00324424113 | 0 |
| `rows_tol03` | 0.15 | 0.0356128997 | 0.00569007315 | 0.00353123644 |
| `rows_tol03` | equal | 0.0260762711 | 0.00446715714 | 0.00176561822 |
| `columns_tol01` | 0.10 | 0.0122987581 | 0.00349877296 | 0.00127181983 |
| `columns_tol01` | 0.15 | 0.0388453268 | 0.00964901268 | 0.0191492899 |
| `columns_tol01` | equal | 0.0255720424 | 0.00657389282 | 0.0102105549 |
| `rows_tol01` | 0.10 | 0.00895345426 | 0.00103551193 | 0 |
| `rows_tol01` | 0.15 | 0.0198300062 | 0.00230021882 | 0.00317406941 |
| `rows_tol01` | equal | 0.0143917302 | 0.00166786537 | 0.0015870347 |
| `columns_tol003` | 0.10 | 0.00546979794 | 0.000572627639 | 0.00127181983 |
| `columns_tol003` | 0.15 | 0.00866381347 | 0.000665101801 | 0 |
| `columns_tol003` | equal | 0.0070668057 | 0.00061886472 | 0.000635909916 |
| `rows_tol003` | 0.10 | 0.00583908708 | 0.000568175633 | 0 |
| `rows_tol003` | 0.15 | 0.00708977372 | 0.000506147515 | 0 |
| `rows_tol003` | equal | 0.0064644304 | 0.000537161574 | 0 |

There are 116 tiny negative JS labels among 1,792 approximate-action labels,
with minimum `-1.215000596863831e-16`. They are floating-point roundoff at
numerical zero. Raw values and the byte-identical summary are preserved without
clipping; these values are not interpreted as negative divergence.

## Work and host timing

The full grid costs 19,486,595,542 estimated arithmetic FLOPs:
9,719,050,613 at p=.10 and 9,767,544,929 at p=.15. Both unrestricted
reference views are included. Every attempted QR/SVD is charged before the
numerical call using the frozen dense-decomposition formula; retries would
pay again. The tables do not equate this arithmetic model with measured runtime.

The entire instrumented pilot took 74.67079299967736 host seconds; the sum of
all recorded action times was 71.32059535291046 seconds. The remainder includes
runner overhead. Measurements are research-host engineering timing on the
manifest's macOS arm64 host, with Python 3.14.6, NumPy 2.4.1, SciPy 1.17.1,
qecsim 1.0b9, and networkx 3.6.1. These are instrumented sequential observations,
without a deployment benchmark or repeated timing uncertainty study.

| Action | Rate | Total estimated FLOPs | Host action seconds |
| --- | --- | --- | --- |
| `columns_chi2` | 0.10 | 11,264,896 | 2.170728 |
| `columns_chi2` | 0.15 | 11,264,896 | 2.234490 |
| `rows_chi2` | 0.10 | 11,264,896 | 2.182258 |
| `rows_chi2` | 0.15 | 11,264,896 | 2.233086 |
| `columns_chi4` | 0.10 | 63,362,176 | 2.016752 |
| `columns_chi4` | 0.15 | 63,362,176 | 2.042556 |
| `rows_chi4` | 0.10 | 63,362,176 | 2.027580 |
| `rows_chi4` | 0.15 | 63,362,176 | 2.043683 |
| `columns_chi8` | 0.10 | 330,277,632 | 1.958194 |
| `columns_chi8` | 0.15 | 330,277,632 | 1.974444 |
| `rows_chi8` | 0.10 | 330,277,632 | 1.901078 |
| `rows_chi8` | 0.15 | 330,277,632 | 1.921454 |
| `columns_chi16` | 0.10 | 979,010,816 | 1.956520 |
| `columns_chi16` | 0.15 | 979,010,816 | 1.967160 |
| `rows_chi16` | 0.10 | 979,010,816 | 1.892409 |
| `rows_chi16` | 0.15 | 979,010,816 | 1.896920 |
| `columns_tol03` | 0.10 | 6,163,062 | 2.180336 |
| `columns_tol03` | 0.15 | 10,356,693 | 2.197383 |
| `rows_tol03` | 0.10 | 6,689,098 | 2.175013 |
| `rows_tol03` | 0.15 | 9,684,074 | 2.207056 |
| `columns_tol01` | 0.10 | 8,651,747 | 2.211059 |
| `columns_tol01` | 0.15 | 14,741,463 | 2.336911 |
| `rows_tol01` | 0.10 | 9,861,360 | 2.235511 |
| `rows_tol01` | 0.15 | 14,823,486 | 2.203093 |
| `columns_tol003` | 0.10 | 11,029,517 | 2.227910 |
| `columns_tol003` | 0.15 | 26,105,976 | 2.214985 |
| `rows_tol003` | 0.10 | 12,717,781 | 2.195490 |
| `rows_tol003` | 0.15 | 27,895,189 | 2.242514 |
| `exact_columns` | 0.10 | 3,448,053,504 | 3.273553 |
| `exact_columns` | 0.15 | 3,448,053,504 | 3.315021 |
| `exact_rows` | 0.10 | 3,448,053,504 | 2.806961 |
| `exact_rows` | 0.15 | 3,448,053,504 | 2.878489 |

Each work row is the sum over 64 attempts; divide by 64 for its per-shot mean.
The equal-rate per-shot work/time mean is the two row totals' sum divided by
128; action totals over the full pilot are their sum. Exact work means and
distributions are also in the compact summary; exact per-shot timing remains
in the separately hashed sidecar.

The offline composite action-time sum adds both probes on every shot and
column chi8 only on the escalated shots. It is **slower** than running fixed
column chi8 on every shot:

| Action-time sum (seconds) | p=.10 | p=.15 | Total |
| --- | --- | --- | --- |
| Margin gate + column chi8 | 4.56884 | 4.94793 | 9.51677 |
| Fixed column chi8 | 1.95819 | 1.97444 | 3.93264 |
| Fixed rows_tol003 | 2.19549 | 2.24251 | 4.43800 |

These retrospective sums exclude a separately implemented controller's overhead.
Even the post-hoc cheapest arithmetic comparator is slower on this host than
fixed column chi8. Arithmetic saving is not runtime saving; this pilot supports
no decoder-latency, FPGA-throughput, or hardware claim. Timing never affects
the gate, labels, oracle winners, or advancement calculation.

## Decision and claim boundary

Both frozen advancement clauses are true: all references certify and the gate
accepts more than half the shots at each rate; heterogeneous uniquely separated
positive efficiency winners also exist. That satisfies the design's bounded
opportunity rule for considering a new frozen contract. It does not establish
controller value or authorize a confirmation claim.

The additional decision-level evidence argues against advancing a learned
controller with the current TV objective: there are no baseline class errors
to repair on the escalated shots, TV improvement can introduce a logical
failure, a simple fixed action undercuts the composite arithmetic cost, and host
timing reverses the apparent saving. Retain the unchanged margin gate with fixed
column chi8 fallback as a candidate, not as an accepted safe policy.

The next contract should freeze three comparators before any fresh samples:
fixed `rows_tol003`, the unchanged margin-plus-column-chi8 policy, and fixed
column chi8. It must prioritize class/outcome safety, then measure fully charged
work and host timing, with independent seed domains, estimands, event/sample
bounds, familywise treatment, and failure criteria fixed in advance. The
post-hoc rows_tol003 choice is development evidence only. No controller is
trained or evaluated here.

This small planar code-capacity pilot establishes no safety, superiority,
population-risk, threshold, circuit-level, repeated-round, qLDPC, FNO, HiPPO,
recurrent-controller, code-family-transfer, latency, or hardware result.

## Reproduction and integrity

The scientific source is commit
`a2ffdc5da79a64da6837f54230283e4735aaf674`, recorded clean in the raw artifact.
An independent artifact/replay review approved development-only publication,
reconstructed the compact summary from raw rows with byte identity, replayed
all joined physical shots, and reconciled valid contractions and charged work.
The tracked summary SHA-256 is
`5ff350877d909f8207974dd332d212037e8016dfe2041ace265f300fb44402b3`.

To rerun, use a fresh clean checkout at that source commit with the locked
dependencies. Existing output directories are deliberately refused. For example,
from the repository root (choose a new sibling path if the example exists):

```bash
git worktree add --detach ../qldpc-fno-pilot-replay a2ffdc5da79a64da6837f54230283e4735aaf674
cd ../qldpc-fno-pilot-replay
uv sync --frozen
uv run python experiments/31_generate_planar_shots.py \
  --config configs/adaptive_intervention_pilot_shots.json \
  --out artifacts/adaptive-intervention-pilot-v3/shots
uv run python experiments/35_run_adaptive_intervention_pilot.py \
  --config configs/adaptive_intervention_pilot.json \
  --shots artifacts/adaptive-intervention-pilot-v3/shots/planar_shots.json \
  --out artifacts/adaptive-intervention-pilot-v3/result
```

A fresh clone contains the compact evidence, not the ignored raw/timing files.
Rerunning can reconstruct the deterministic artifacts in the bound numerical
environment; changed dependencies or numerical implementations may change
floating-point bytes. The timing sidecar is intentionally nondeterministic and
is never required to reproduce byte-for-byte.

With the original raw artifacts available at the manifest's repository-relative
paths, verify every size/hash and the source/tracked summary identity:

```bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path(".")
manifest = json.loads((root / "evidence/adaptive-computation/intervention-pilot-v3/manifest.json").read_text())
for entry in [*manifest["artifacts"].values(), *manifest["configs"].values()]:
    content = (root / entry["path"]).read_bytes()
    assert len(content) == entry["size_bytes"]
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
source = root / manifest["artifacts"]["source_summary"]["path"]
tracked = root / manifest["artifacts"]["tracked_summary"]["path"]
assert source.read_bytes() == tracked.read_bytes()
print("All artifact/config identities and summary bytes match.")
PY
```

To reconstruct the compact summary from the raw rows without rerunning
contractions or editing either artifact:

```bash
uv run python - <<'PY'
import hashlib
import json
from pathlib import Path
from qldpc_fno.decision.adaptive_intervention_pilot import (
    load_pilot_config, summarize_evaluations,
)

base = Path("artifacts/adaptive-intervention-pilot-v3/result")
raw_bytes = (base / "intervention_pilot.json").read_bytes()
raw = json.loads(raw_bytes)
summary = summarize_evaluations(
    raw["shots"], load_pilot_config(Path("configs/adaptive_intervention_pilot.json"))
)
assert summary == raw["summary"]
compact = {key: raw[key] for key in (
    "schema_version", "status", "scientific_eligible", "provenance"
)}
compact.update(raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
               raw_size_bytes=len(raw_bytes), summary=summary)
rebuilt = (json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
assert rebuilt == (base / "summary.json").read_bytes()
assert rebuilt == Path("evidence/adaptive-computation/intervention-pilot-v3/summary.json").read_bytes()
print("Raw-row reconstruction is byte-identical to both summaries.")
PY
```

The summary's `scientific_eligible=true` means the full frozen run passed
its execution provenance checks. Its status is `development_pilot`; that
eligibility flag does not turn this development sample into confirmation.
