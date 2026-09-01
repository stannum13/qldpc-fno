# Calibration throughput benchmark

This benchmark is a capacity measurement, not scientific decoder evidence. It was
run before changing the calibration policy, at Git commit
`51bad3727eb2c5eee2096f6cc5b6d195e42598dc`.

## Method

- Hardware: Apple M2 Max, macOS arm64.
- Software: the committed `uv.lock`, Python 3.14 environment, PyTorch with eight
  CPU threads.
- Artifact: the completed reduced campaign at
  `artifacts/accuracy-final-local-20260901`.
- Code and data: the real canonical `lp(3,7)_16`, `ell=45` matrices and eight
  independently sampled calibration-role shots from the campaign artifact.
- Candidate: epoch 1, `alpha=0.25`, `beta=0`, `temperature=0.5`.
- Timed region: `_score_candidate`, including calibrated probabilities, soft-prior
  BP-LSD, residual-repair BP-LSD, validity checks, observables, and NLL. FNO
  inference and artifact loading were outside the timed region.
- Procedure: one untimed warm-up followed by five trials over the same verified
  shots. Reuse makes this a throughput measurement, not a statistical result.

The five trial times were `1.8622`, `1.8740`, `1.9001`, `1.9235`, and `1.8972`
seconds. The median was `1.8972` seconds for eight shots, or 4.22 candidate-shots
per second across both hybrid methods.

## Capacity conclusion

The former policy hybrid-decoded every combination of 60 checkpoints, 48
parameter tuples, and 10,000 calibration shots: 28.8 million candidate-shots.
At the measured median that timed region alone would take about 79 days. This
excludes training, sampling, FNO inference, persistence, held-out evaluation,
and cloud/local hardware differences. It therefore cannot support an eight-hour
single-execution claim.

The revised policy uses the measurement only as a conservative lower-bound
planning input:

- all checkpoint/parameter combinations receive cheap calibration-only proxy
  scores;
- four candidates are shortlisted independently for soft-prior and residual
  decoding;
- the union (at most eight candidates) is hybrid-decoded on a deterministic,
  rate-stratified 512-shot calibration subset;
- at a deliberately slower planning rate of 3 candidate-shots/s, the expensive
  second stage is bounded near 23 minutes.

No claim is made about the duration of screening, training, held-out evaluation,
or an entire canonical campaign. Until those stages have representative capacity
benchmarks, canonical cloud runs are declared multi-execution and must use strict
resume. Each Cloud Run execution retains an eight-hour outer limit but begins
finalization no later than 7h15m.
