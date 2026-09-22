# Adaptive Intervention Pilot Design

**Status:** Approved as item 4 of the first executable tranche in
`docs/adaptive-computation-reader-plan.md`. This document makes that bounded pilot
executable; it does not authorize a confirmation claim.

## Question

Among distance-five planar-code shots that a frozen two-view margin gate would
escalate, do different contraction interventions improve the unrestricted
reference posterior on different shots, after charging the complete arithmetic
work of every independent run?

The pilot is a mechanism and engineering study. It estimates runtime, removes
redundant actions, and identifies candidate action-dependent gains. Its 128 shots
are development data forever and cannot confirm a policy selected from them.

## Immutable data identity

- Seed domain: `qldpc-fno/adaptive-intervention-pilot/v1`
- Campaign seed: `16980117767564665917`, the unsigned big-endian integer formed
  from the first eight bytes of SHA-256 over the seed domain.
- Code: qecsim distance-five planar code.
- Noise: qecsim iid depolarizing code-capacity noise.
- Physical error rates: `0.10` and `0.15`.
- Shots: 64 independently seeded physical errors per rate.

The existing planar-shot generator creates each error and derives its syndrome
from that same error. Shot identity, sampler seed, physical error, and syndrome
remain joined. This pilot domain must be disjoint from every calibration, screen,
and future confirmation domain.

## Frozen action table

Every shot receives the same actions:

| Family | Modes | Parameters |
|---|---|---|
| Fixed bond dimension | columns, rows | `chi = 2, 4, 8, 16` |
| Local spectral tolerance | columns, rows | `tol = 0.03, 0.01, 0.003` |
| Unrestricted reference | columns, rows | `chi = None`, `tol = None` |

This is 14 approximate actions and two reference views per shot. The two
`tol=0.01` actions are the initial cheap state used by the retrospective gate.
They are computed once and reused in offline composite-policy accounting.

The pilot does not include average-mode contraction, local bond allocation,
resumption, learned policies, or hardware timing. Those would change the
scientific question or require solver support that is not yet audited.

## Reference and scoring

The two unrestricted contractions must be valid, select the same logical class,
and agree within the existing probability and log-ratio certificate tolerances.
An invalid or uncertified reference invalidates the shot for posterior-target
analysis but remains recorded and charged.

For a certified shot, the target posterior is the arithmetic mean of the two
separately normalized unrestricted row and column probability vectors, normalized
once more after averaging. This symmetric definition is fixed before sampling;
the certificate ensures the two inputs already agree within the declared bounds.

For every valid approximate action, store normalized class probabilities,
selected logical class, stable work counters, and spectral summaries. Score its
class recovery against the joined physical error by residual logical
commutation. Degenerate but logically correct recoveries are successes.

Compare each action with the certified reference using:

- logical-class mismatch;
- physical logical-outcome discordance;
- total-variation distance;
- Jensen-Shannon divergence;
- reference-conditional excess decision risk; and
- fully charged estimated arithmetic FLOPs.

Jensen-Shannon divergence uses natural logarithms and is reported in nats, with
the standard convention that a zero-probability numerator contributes zero.

Exact posteriors and physical outcomes are labels only. They may define offline
utility and oracle opportunity, never an inference-time feature.

## Frozen gate and candidate utility

The pre-action state runs column and row `tol=0.01`. It accepts only when both
views are valid, select the same class, and their minimum winning-class margin is
strictly greater than `0.30710401263493464`. Equality, disagreement, or invalidity
escalates.

For an escalated shot, a candidate action's independent-run composite cost is
the sum of both initial probes and the candidate action. No state-reuse discount
is claimed. Candidate benefit is the reduction in total-variation distance to
the certified reference relative to the better of the two initial posteriors.
Negative benefit is retained. Also report whether the candidate repairs or
introduces a logical-class mismatch and whether it changes the realized physical
outcome.

The candidate set excludes the two probes and therefore contains the remaining
12 approximate actions. A gain is positive only when it exceeds `1e-6`; this
prevents numerical noise from becoming action-selection opportunity.

The per-shot offline oracle selects the valid candidate with the greatest
positive TV reduction. Gains within `1e-12` are tied and are broken by lower
composite work, then frozen action order. If no candidate gain exceeds `1e-6`,
the winner ID is null and the shot has no measured action opportunity. A winner
is unique for the advance rule only when its gain exceeds the runner-up by more
than `1e-6`. The oracle is an opportunity bound, not a deployable policy.

The inference-visible pre-action feature set is frozen to physical error rate,
syndrome Hamming weight, both probe selected classes, class agreement, both
winning-class margins and their minimum, cross-probe TV and Jensen-Shannon
divergence, each probe's stable arithmetic work, and each probe's compact
spectral summaries. Invalidity flags replace unavailable values. The pilot may
show descriptive winner counts or plots over these features, but it may not
search feature subsets or fit and evaluate a controller on the same 128 shots.

## Outputs and decision rule

The canonical pilot artifact contains provenance, all shot/action rows, invalid
work, reference certificates, gate states, and compact summaries. A compact
tracked summary and a human-readable result note report:

- runtime and arithmetic cost by action;
- escalated-shot count by rate;
- paired error distributions by action;
- how often each action is the oracle winner;
- descriptive winner counts across the frozen inference-visible features, with
  no fitted controller or significance claim; and
- the strongest beneficial and harmful counterexamples.

All 128 shots remain in the total and invalid-reference denominators. Posterior
metrics use the certified-reference denominator, which is reported separately.
Any invalid or uncertified unrestricted reference makes both advancement clauses
false; it cannot be hidden by complete-case analysis.

Advance to a frozen risk-ladder contract only if at least one of these is true:

1. all 128 references certify and the unchanged margin gate accepts at least 50%
   of shots separately at each physical error rate; or
2. all 128 references certify and at least two distinct candidate action IDs are
   uniquely best on different escalated shots with TV reduction greater than
   `1e-6` and winner-to-runner-up separation greater than `1e-6`, establishing
   bounded action-selection opportunity.

The pilot cannot establish safety or superiority. If no heterogeneous positive
opportunity appears, the next contract tests only the simple margin gate with a
fixed fallback. A learned controller is not justified.

## Failure handling and reproducibility

Each invalid contraction records its exception class and stable partial work.
All attempted work is charged. The runner refuses an existing output directory,
validates exact config keys and the derived campaign seed, and binds config,
source, dependency, Git, and input-artifact hashes. Development runs may use a
dirty tree while testing; the scientific artifact requires a committed clean
tree and is regenerated from that commit.

Unit tests cover config rejection, seed separation, action ordering, reference
certification, recovery scoring, invalid-work charging, gate equality, oracle
tie-breaking, and canonical replay. A reduced integration fixture exercises the
CLI without representing scientific evidence.

## Scope boundary

This pilot concerns a small planar code under iid code-capacity noise. It does
not support claims about circuit-level decoding, repeated rounds, qLDPC codes,
FNOs, HiPPO, recurrent controllers, local tensor allocation, latency, FPGA
throughput, thresholds, or code-family transfer.
