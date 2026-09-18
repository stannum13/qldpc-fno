# Planar physical-shot accuracy results

**FALSIFIED: row/column tolerance agreement is not a safe certificate of
exact-reference outcome preservation in this frozen distance-5 screen.** The
canonical status is `falsified_exact_outcome_preservation`. The policy disagreed
with the certified unrestricted tensor class on 2 of 2,048 shots at p=0.10 and
6 of 2,048 at p=0.15. All eight were accepted agreements between the two
tolerance views, and all eight changed the physical-shot logical outcome.
Neither stratum passes the frozen gate.

This result uses correctly paired physical errors and symplectic syndromes,
as required by the [contract](planar-shot-accuracy-contract.md). It addresses the
[symplectic syndrome erratum](symplectic-syndrome-erratum.md); earlier planar
syndrome-marginal agreement results did not establish shot-level BLER. The
contract's “not run” status and absent-artifact wording are the preserved
pre-execution freeze, not the current campaign status.

## Primary endpoint and validity

The model is unrotated qecsim `PlanarCode(5, 5)`, i.i.d. depolarizing
code-capacity noise. Each independent unit is one physical shot. The fixed
policy evaluates row and column contractions at tolerance 0.01, accepts their
agreed class, and otherwise falls back to a column chi=8 contraction. The
reference certifies agreement of unrestricted row/column contractions, including
probability and log-ratio tolerances and a unique winning margin.

| Endpoint | p=0.10 | p=0.15 |
|---|---:|---:|
| Held-out physical shots | 2,048 | 2,048 |
| Class mismatches | 2 | 6 |
| Logical-failure outcome discordances | 2 | 6 |
| Adjusted one-sided Wilson upper bound | 0.0035538194 | 0.0063772332 |
| Required upper bound | ≤0.005 | ≤0.005 |
| Required class mismatches and outcome discordances | 0 and 0 | 0 and 0 |
| Gate passed | No | No |
| Fallbacks | 47 | 101 |
| Invalid exact references | 0 | 0 |
| Invalid final recoveries, each of four arms | 0 | 0 |

Wilson bounds use alpha=0.025 per rate, splitting familywise alpha=0.05 over
the two strata. Even though the p=0.10 upper bound is below 0.005, its nonzero
mismatch and discordance counts fail the zero-event requirements. The p=0.15
bound also exceeds 0.005. Aggregate failure counts cannot replace this
per-shot preservation test: pooled exact and policy failures both happen to be
432, while their outcomes differ on eight shots.

Every final recovery reproduced its syndrome. There were no decoder exceptions
in any arm. Two tolerance-column actions and two tolerance-row actions were
invalid; these intermediate invalidities were handled by the frozen fallback
path. There were no invalid final policy recoveries. The producer's complete
independent replay passed, and calibration/screen seeds were disjoint.

## Logical failures and paired comparisons

CMWPM calibration used 512 separate shots per rate and all 48 declared parameter
tuples. The frozen ordering selected `(factor=2, max_iterations=4,
box_shape="t", distance_algorithm=4)`, with 35 and 106 calibration failures
(141 pooled). All candidates completed with zero invalid recoveries or decode
exceptions. This is **calibrated CMWPM within the declared grid**. Ordinary
MWPM is the named factorized-reference arm.

The following are physical-shot BLERs with two-sided 95% Wilson intervals:

| Rate | Decoder | Failures | BLER | Wilson 95% interval |
|---|---|---:|---:|---|
| 0.10 | Unrestricted tensor | 109/2048 | 5.3223% | [4.4310%, 6.3808%] |
| 0.10 | Tolerance policy | 107/2048 | 5.2246% | [4.3420%, 6.2749%] |
| 0.10 | Calibrated CMWPM | 148/2048 | 7.2266% | [6.1834%, 8.4299%] |
| 0.10 | Ordinary MWPM | 224/2048 | 10.9375% | [9.6582%, 12.3631%] |
| 0.15 | Unrestricted tensor | 323/2048 | 15.7715% | [14.2572%, 17.4139%] |
| 0.15 | Tolerance policy | 325/2048 | 15.8691% | [14.3508%, 17.5153%] |
| 0.15 | Calibrated CMWPM | 416/2048 | 20.3125% | [18.6264%, 22.1098%] |
| 0.15 | Ordinary MWPM | 537/2048 | 26.2207% | [24.3616%, 28.1689%] |

Exact tensor had fewer observed failures than this calibrated CMWPM
configuration on both held-out strata. The paired evidence below applies to
that configuration and sample; it does not establish a globally optimal
matching decoder comparison. These secondary comparisons are descriptive and
do not control or rescue the primary preservation gate.

In each table row, “first only” means only the first decoder failed and
“second only” means only the second decoder failed. P-values are two-sided
exact McNemar values; all original one-sided values and paired intervals are
retained in the [machine-readable summary](../evidence/planar-shot-accuracy/summary.json).
No familywise correction is asserted for these secondary comparisons.

| Rate | First vs second | Both succeed | Both fail | First only | Second only | Exact two-sided p |
|---|---|---:|---:|---:|---:|---:|
| 0.10 | exact vs cmwpm | 1870 | 79 | 30 | 69 | 0.000110940 |
| 0.10 | exact vs mwpm | 1790 | 75 | 34 | 149 | 2.26508e-18 |
| 0.10 | exact vs policy | 1939 | 107 | 2 | 0 | 0.500000 |
| 0.10 | policy vs cmwpm | 1871 | 78 | 29 | 70 | 0.0000460624 |
| 0.15 | exact vs cmwpm | 1569 | 260 | 63 | 156 | 2.75291e-10 |
| 0.15 | exact vs mwpm | 1437 | 249 | 74 | 288 | 6.24501e-31 |
| 0.15 | exact vs policy | 1721 | 321 | 2 | 4 | 0.687500 |
| 0.15 | policy vs cmwpm | 1569 | 262 | 63 | 154 | 5.62334e-10 |

The exact/policy comparisons have only 2 and 6 discordant pairs. Their direction
does not support calling the policy better or worse in population BLER, or
equivalent to the exact decoder. At p=0.10 both changes helped on the realized
physical shots; at p=0.15 two helped and four harmed. Either direction violates
the predeclared exact-outcome preservation requirement.

## All eight accepted disagreements with the reference

Classes below are the integer class indices stored by the frozen evaluator.
“Failure direction” names the only failing decoder in each discordant pair.
The exact margin is the column reference's largest minus second-largest
normalized class probability. The tolerance discrepancy is the maximum
absolute difference between row and column normalized class probabilities.

| Shot ID | Exact → policy class | Failure direction | Exact margin | Tolerance row/column discrepancy |
|---|---|---|---:|---:|
| `d5/p0.100000/i000463` | 3 → 0 | Exact only | 0.02659040 | 0.14519694 |
| `d5/p0.100000/i001641` | 0 → 1 | Exact only | 0.01867658 | 0.00385130 |
| `d5/p0.150000/i000220` | 3 → 2 | Policy only | 0.03040237 | 0.35999517 |
| `d5/p0.150000/i000641` | 0 → 2 | Policy only | 0.04761608 | 0.08522995 |
| `d5/p0.150000/i000740` | 0 → 3 | Policy only | 0.08438622 | 0.03457542 |
| `d5/p0.150000/i001176` | 0 → 1 | Policy only | 0.01925885 | 0.00823636 |
| `d5/p0.150000/i001202` | 3 → 0 | Exact only | 0.02590582 | 0.00304395 |
| `d5/p0.150000/i001907` | 0 → 3 | Exact only | 0.06230619 | 0.04966500 |

Every row has `accepted=true` and `used_fallback=false`. The
[summary](../evidence/planar-shot-accuracy/summary.json) retains both exact margins,
both tolerance margins, all four class-probability vectors, exact-reference
certificates, tolerance log-ratio discrepancies, tolerance-versus-exact
discrepancies, sampler seeds, and outcome directions. Agreement of truncated
views alone was insufficient in these cases. These observations do not validate
a replacement acceptance threshold or a retuned policy.

## Estimated arithmetic work

Work charges both tolerance views on every shot and chi=8 work whenever fallback
runs. Fixed column chi=8 and the two unrestricted views are descriptive work
comparators. Ratios below are policy estimated arithmetic work divided by
comparator work; intervals use 10,000 frozen-seed paired-bootstrap replicates,
resampling physical shots within rate.

| Stratum | Comparator | Policy/comparator ratio | Two-sided 95% bootstrap interval |
|---|---|---:|---|
| 0.10 | Column chi=8 | 0.080859 | [0.074174, 0.087834] |
| 0.10 | Two-view unrestricted | 0.003873 | [0.003552, 0.004207] |
| 0.15 | Column chi=8 | 0.143488 | [0.133942, 0.153614] |
| 0.15 | Two-view unrestricted | 0.006872 | [0.006415, 0.007357] |
| Equal-rate pooled | Column chi=8 | 0.112173 | [0.106324, 0.118246] |
| Equal-rate pooled | Two-view unrestricted | 0.005372 | [0.005092, 0.005663] |

Totals are 2,371,095,456 estimated arithmetic FLOPs for the policy,
21,137,768,448 for fixed chi=8, and 441,350,848,512 for the two-view unrestricted
reference. The arithmetic reductions are **not accuracy-preserving savings**:
the exact-outcome gate failed. These estimates are not measured speed,
latency, throughput, energy, or hardware resource use.

## Evidence, publication, and provenance

The [manifest](../evidence/planar-shot-accuracy/manifest.json) binds the
[derived summary](../evidence/planar-shot-accuracy/summary.json) to the complete
[public gzip artifact](https://storage.googleapis.com/project-1178f0de-10fb-4e7e-8e4-qldpc-fno-artifacts/planar-shot-accuracy/v1/b56fecd64497fb4aa1204027820543411bcab4dc4f2979deb2db52d00ff49769/planar_shot_accuracy.json.gz?generation=1789705238581848).
The raw JSON is 780,670,635 bytes, so the full evidence is published as a
deterministic 44,451,709-byte gzip object rather than embedded as a Git blob.
Decompressing it restores the original JSON byte for byte. The local raw and
gzip paths alone are Git-ignored; the manifest and summary are tracked.

- Raw SHA-256: `b56fecd64497fb4aa1204027820543411bcab4dc4f2979deb2db52d00ff49769`.
- Gzip SHA-256: `d6268f928eef4dd2c5e17013b5ce6ecd67f0e94f6752de9f52594416d2d3c7b3`.
- GCS object generation: `1789705238581848`; bucket versioning is enabled.
- Gzip object CRC32C: `SbnIbw==`; MD5: `Dw4H11znjR6DMv3sj9Offw==`.
- The public-download round trip passed the raw SHA-256 check.

This storage publication changes the fourth barrier's packaging, not the
immutable scored result or scientific inputs. The exact GCS URI, HTTPS URL,
generation-pinned URL, sizes, and checksums are recorded in the manifest.

| Freeze or barrier | Commit |
|---|---|
| Source/config/policy freeze | `8ecda1f3d4a70e0ae3678a2d5eaf44384739813e` |
| Calibration shots committed and pushed | `e21a7623058d9b5869ed69a692a187ac03c61c28` |
| CMWPM selection committed and pushed | `dbaec2297eea30079300680c83d28d7b5afcdc00` |
| Screen shots committed and pushed; evaluator launch HEAD | `19af0a3e4381f0f4414ec2ca2af1692d3f3424c9` |

Each producer started from a clean tree after its input barrier was pushed.
Calibration generation, selection, screen generation, and held-out evaluation
were each invoked once. The evaluator ran on 2026-09-18 from 03:32:21Z until
the artifact write at 03:59:44.827308Z.

The raw artifact's `provenance.git_commit` is
`dbaec2297eea30079300680c83d28d7b5afcdc00`, because the frozen implementation
binds that field to the **screen producer's** commit. It is not the evaluator
launch HEAD, which was `19af0a3e4381f0f4414ec2ca2af1692d3f3424c9`. This
ambiguity is disclosed here and in the manifest; the original evidence bytes
were preserved. All source and input hashes remain available in the raw
artifact and summary.

The frozen matching backend was NetworkX 3.6.1, with Blossom5 unavailable.
Recorded versions were Python 3.14.6, qecsim 1.0b9, NumPy 2.4.1 and SciPy
1.17.1. These identities and the lockfile digest were checked across stages.

## Claim boundary

The evidence supports the reported distance-5, two-rate, i.i.d.-depolarizing
code-capacity observations and falsifies the frozen policy's exact-outcome
preservation gate. It does not establish posterior calibration, globally optimal
matching, circuit-level performance, thresholds, other distances or rates,
qLDPC transfer, FPGA fit, latency, throughput, backlog stability, learned-control
value, FNO/HiPPO/world-model value, or state-of-the-art decoding. No new policy
was tuned on these held-out results.
