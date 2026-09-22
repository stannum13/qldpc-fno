# Adaptive computation diagnostic audit

## Result

The minimum winning-class margin across the two cheap tolerance contractions is the strongest single development signal for the eight common-mode reference mismatches in the frozen 4,096-shot planar result. Its exploratory ROC AUC is `0.963706` pooled, `0.965483` at `p=0.10`, and `0.963850` at `p=0.15`, where a smaller margin is scored as more suspicious.

This is a hypothesis-discovery result. The threshold and outcomes were derived from an already revealed physical-shot screen. No threshold has been evaluated on untouched shots, and the eight mismatches are too few to establish calibrated risk.

The result narrows the next question. The observed failures do not currently support an elaborate spectral controller. They support testing whether a simple margin gate can retain useful cheap coverage on fresh data, followed by targeted refinement experiments for cases the gate escalates.

## Source and reconstruction

The audit is derived from the immutable physical-shot artifact reported in [Planar physical-shot accuracy results](planar-shot-accuracy-results.md). Before analysis, the CLI verifies the raw artifact against its publication manifest:

- Source shots: `4,096`, with `2,048` at each of `p=0.10` and `p=0.15`.
- Source SHA-256: `b56fecd64497fb4aa1204027820543411bcab4dc4f2979deb2db52d00ff49769`.
- Source size: `780,670,635` bytes.
- Reconstructed mismatches: `2` at `p=0.10` and `6` at `p=0.15`.
- Reconstructed outcome discordances: `2` and `6`.
- Reconstructed original fallbacks: `47` and `101`.
- Reconstructed work totals: `2,371,095,456` policy FLOPs, `21,137,768,448` fixed-chi8 FLOPs, and `441,350,848,512` exact-reference FLOPs under the source estimator.

All reconstructed counts and work totals match the frozen artifact. The derived audit stores compact diagnostics and labels, not raw physical errors, syndromes, or recovery strings. It is available at [diagnostic_audit.json](../evidence/adaptive-computation/diagnostic-audit-v1/diagnostic_audit.json).

The tracked derived artifact is `11,249,028` bytes with SHA-256 `98636ae05b3195a3e10a9ffd77483295b09382e84e645cea8f7b03a0f054835e`. It records the publication-manifest hash as well as the raw source identity.

## Which signals are actually available?

The five proposed signals do not arrive at the same time.

| Signal | Earliest availability | Cost and status |
|---|---|---|
| Cross-view disagreement | After both tolerance contractions | Deployable, but both contractions must be charged. |
| Decision margin | After each tolerance posterior | Deployable; the two-view minimum is known after both cheap actions. |
| Cross-scale stability | After fixed-chi8 refinement | Useful offline diagnostic, not a free early-exit feature. |
| Discarded representation | During tolerance contractions | Deployable spectral summaries from already charged truncations. |
| Calibrated OOD score | After fitting a separate calibrator | Absent from this artifact and not evaluated here. |

Exact-reference distances and physical-shot success are labels. They never enter the deployable feature namespace. The compact schema preserves this boundary structurally.

## Exploratory discrimination

The original two-view policy accepted `3,948` shots because the two tolerance views returned the same class. All eight reference mismatches occurred in that accepted set. The table evaluates single scores only within those accepted agreements. An AUC of `0.5` is chance ordering; values below `0.5` mean the declared “larger is more suspicious” direction is reversed on this sample.

| Score | Pooled AUC | p=0.10 AUC | p=0.15 AUC | Timing |
|---|---:|---:|---:|---|
| Negative minimum cheap-view margin | 0.963706 | 0.965483 | 0.963850 | Cheap |
| Column tolerance-to-chi8 drift | 0.939086 | 0.990495 | 0.886485 | Paid refinement |
| Cross-view total variation | 0.778236 | 0.847424 | 0.692770 | Cheap |
| Mean spectral entropy | 0.712214 | 0.535768 | 0.727031 | Cheap |
| Cross-view Jensen–Shannon divergence | 0.654315 | 0.571786 | 0.592049 | Cheap |
| Maximum discarded squared-weight fraction | 0.557614 | 0.473112 | 0.497810 | Cheap |
| Cross-view log-ratio discrepancy | 0.276110 | 0.236368 | 0.339516 | Cheap |

The rate-stratified margin AUCs remain close to the pooled value, so the observed ranking is not merely the result of pooling the two physical error rates. That does not make it out-of-sample evidence.

The eight mismatch margins range from `0.002079` through `0.307104`, with median `0.094930`. Across all accepted agreements, the median minimum margin is `0.977151`; the fifth percentile is `0.158407`. The mismatches occupy the low-margin tail, although the tail also contains many correct cheap decisions.

The paid column-to-chi8 drift is also discriminative, but it is available only after performing the refinement whose cost an early policy is meant to avoid. Its proper role is a transition target or a second-stage signal, not a cheap stopping certificate.

## What the other diagnostics do and do not explain

The cross-view total-variation score has useful pooled ranking, but the individual failures are heterogeneous. Its mismatch values span `0.003851` to `0.361866`. A threshold low enough to catch every revealed mismatch using total variation alone would cheaply accept only `1,918` of the `3,948` originally accepted agreements.

Log-ratio discrepancy behaves differently because it can be dominated by changes among very small logical-class probabilities. The exact decision depends first on the leading classes. On these data, larger log-ratio disagreement is not a useful direction for identifying the reference mismatches.

Maximum discarded squared-weight fraction is close to chance. This supports a distinction central to the next experiments: local spectral loss is not automatically logical relevance. The surrounding contraction determines whether a discarded component changes the final class probabilities.

To reduce the obvious margin confounding, each of the eight mismatch shots was paired with a unique, accepted, reference-matching shot at the same physical error rate and nearest minimum margin. The pairs have absolute margin differences between `0.000255` and `0.002360`.

| Paired diagnostic: mismatch minus margin-matched control | Positive pairs | Negative pairs | Median difference |
|---|---:|---:|---:|
| Cross-view total variation | 3 | 5 | -0.0130943 |
| Jensen–Shannon divergence | 4 | 4 | 0.0000782 |
| Log-ratio discrepancy | 6 | 2 | 0.4844351 |
| Maximum discarded squared-weight fraction | 5 | 3 | 0.00000280 |
| Mean spectral entropy | 3 | 5 | -0.0542411 |

These eight matched examples are descriptive, not an adjusted population estimate. They show why view disagreement and truncation magnitude should not be presented as established explanations of the common-mode failures.

## A retrospective margin gate

The largest minimum cheap-view margin among the eight revealed mismatches is:

```text
0.30710401263493464
```

The development rule

```text
accept the cheap agreed class only when minimum margin > 0.30710401263493464;
otherwise run fixed column chi=8
```

has the following retrospective behaviour:

| Quantity | Development-open value |
|---|---:|
| All physical shots | 4,096 |
| Cheap accepts | 3,581 (87.427%) |
| Escalations | 515 (12.573%) |
| Cheap accepts at p=0.10 | 1,912/2,048 (93.359%) |
| Cheap accepts at p=0.15 | 1,669/2,048 (81.494%) |
| Accepted reference mismatches | 0 |
| Chi8 fallback reference mismatches | 0 |
| Projected total work | 4,265,031,252 estimated FLOPs |
| Fixed-chi8 work | 21,137,768,448 estimated FLOPs |
| Projected work/fixed-chi8 | 0.201773 |
| Projected arithmetic saving | 79.823% |

This apparent zero-mismatch result is selected on the same eight labels it explains. It has no valid confirmation interval and does not repair the falsified frozen policy. It is the threshold to challenge, not a reported safe decoder.

The work projection charges both tolerance contractions on every shot and fixed chi8 on every escalation. It assumes the fixed-chi8 actions already recorded in the artifact. It is an arithmetic estimate, not incremental reuse, runtime, energy, or FPGA latency.

Relative to the original agreement-only policy, the retrospective rule escalates an additional `367` accepted agreements. That increases estimated work while addressing exactly the common-mode failure mode the original gate could not see.

## Strongest counterexamples

The hardest revealed case for a simple margin gate is `d5/p0.100000/i001641`. Both tolerance views choose class `1`, their total-variation distance is only `0.003851`, and their minimum margin is `0.307104`; the exact reference chooses class `0` with margin `0.018677`. Cross-view agreement is close, reasonably confident, and wrong relative to the reference.

The converse problem is abundant: many low-margin cheap agreements are reference-correct. A conservative margin rule spends chi8 work on these benign cases. The fresh experiment must therefore measure both accepted-set risk and coverage/work, not celebrate one without the other.

## Consequence for the architecture

The evidence currently favours a staged controller:

1. Run the two tolerance views and compute their margins and probability disagreement.
2. Accept only high-margin agreements under a threshold frozen before confirmation.
3. Escalate the low-margin tail to a stronger action.
4. Within the escalated tail, measure which refinement family improves the posterior per unit of additional work.

This is where the broader multiscale program becomes testable. A learned recurrent controller is justified only if the escalated cases contain reproducible subfamilies for which different actions help. The present audit has identified the tail; it has not yet shown those action-dependent gains.

## Next experiment

The threshold above should be frozen unchanged and evaluated on a new deterministic seed domain with correctly paired physical errors and syndromes. The primary outcomes remain reference-class mismatch, physical-outcome discordance, valid recovery, cheap coverage, and fully charged work. Fixed chi8 and the unrestricted row/column reference remain comparators.

In parallel, a small development pilot can evaluate direction, chi, and tolerance actions on fresh but non-confirmatory shots. Its purpose is to define the refinement state/action table and estimate runtime. It cannot retune the frozen margin confirmation after confirmation results are opened.

The confirmation contract must declare its event-rate bound and sample size. “Zero observed mismatches” is not proof of zero population risk. Any claim about a finite mismatch budget requires the corresponding one-sided interval and multiplicity treatment.

## Reproduce

The source raw JSON is intentionally Git-ignored because it is approximately 745 MiB. Download and verify it using the commands in the [physical-shot result](planar-shot-accuracy-results.md), then run:

```bash
uv run python experiments/34_audit_planar_diagnostics.py \
  --artifact evidence/planar-shot-accuracy/planar_shot_accuracy.json \
  --manifest evidence/planar-shot-accuracy/manifest.json \
  --out evidence/adaptive-computation/diagnostic-audit-v1-replay
```

The CLI refuses to overwrite an existing output. Byte identity of the derived JSON also depends on the audited source version; the tracked artifact binds the source SHA-256 and reconstructs its declared counts and work totals before reporting the diagnostic audit.

## Claim boundary

This development audit establishes what was computed on the revealed 4,096-shot artifact and identifies a concrete hypothesis for fresh testing. It does not establish calibrated safety, accuracy-preserving work reduction, a deployable stopping rule, generalization to another distance, circuit-level noise, repeated measurement, qLDPC codes, learned-controller value, recurrent-memory value, latency, or hardware performance.
