# Calibration throughput benchmark

This is a release-hardening capacity audit, not scientific decoder evidence. It
targets a pre-policy campaign artifact whose recorded Git commit is
`51bad3727eb2c5eee2096f6cc5b6d195e42598dc`.

## Method

- Hardware: Apple M2 Max, macOS arm64.
- Software: the committed `uv.lock`, Python 3.14 environment, PyTorch with eight
  CPU threads.
- Artifact: the completed reduced campaign at
  `artifacts/accuracy-final-local-20260901`.
- Immutable artifact identities (SHA-256): effective config
  `b1751f4373475192c58dd0226dcc9372eea9944da890c763ee5cc0617d38ba11`,
  code manifest
  `83a3f6d0dd5229cb686ca9642f251e2ed424bc271ee8c3a2d48b36b4abe4d277`,
  calibration manifest
  `8c931a791204d5c521a00d2814e6ded4cd88feb4c9564420d9c70d0074590534`,
  model manifest
  `ffe8840fc18529a1a565e6a1beab1a6bacfe397a7a1c4e0f271229f8f142ebcf`,
  and epoch-1 checkpoint
  `5417579795b3117fc2728e849e2736f1887e915ec626caf218c4325665750682`.
- Code and data: the real canonical `lp(3,7)_16`, `ell=45` matrices and eight
  independently sampled calibration-role shots from the campaign artifact.
- Candidate: epoch 1, `alpha=0.25`, `beta=0`, `temperature=0.5`.
- Timed region: `_score_candidate`, including calibrated probabilities, soft-prior
  BP-LSD, residual-repair BP-LSD, validity checks, observables, and NLL. FNO
  inference and artifact loading were outside the timed region.
- Audit procedure: reconstruct logits, then attempt one trial over all eight
  verified shots with no warm-up and a 645-second whole-process diagnostic ceiling.

The checked-in reproduction harness verifies those hashes, reconstructs the
canonical-code logits outside the timed region, and invokes the same private
`_score_candidate` path. The exact bounded reproduction command is:

```bash
PYTHONHASHSEED=0 OMP_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 \
uv run python - <<'PY'
import subprocess
import sys

command = [
    sys.executable,
    "scripts/benchmark_calibration_candidate.py",
    "artifacts/accuracy-final-local-20260901",
    "--threads", "8",
    "--warmups", "0",
    "--trials", "1",
]
try:
    subprocess.run(command, check=True, timeout=645)
except subprocess.TimeoutExpired:
    print("diagnostic_timeout_seconds=645")
PY
```

The artifact is intentionally ignored and is not distributed by the repository;
the hashes above are required when reproducing the recorded diagnostic.

An earlier ad hoc measurement reported five roughly 1.9-second samples. The exact
hash-bound harness did not reproduce them, so those samples and the derived 4.22
candidate-shots/s rate are withdrawn. The audit trial did not finish within 645
seconds. A process sample after six minutes placed the active thread inside
`ldpc::lsd::LsdDecoder::lsd_decode`, confirming that real hybrid decode work—not
only artifact setup—remained active. Because the timed region did not complete,
this audit intentionally reports no throughput estimate.

## Capacity conclusion

The former policy hybrid-decoded every combination of 60 checkpoints, 48
parameter tuples, and 10,000 calibration shots: 28.8 million candidate-shots.
The audit supplies no defensible rate for that work or for the revised hybrid
stage. Training, sampling, FNO inference, persistence, held-out evaluation, and
cloud/local hardware differences add further uncertainty.

The revised two-stage policy still reduces the number of hybrid candidates:

- all checkpoint/parameter combinations receive calibration-only proxy scores
  without hybrid decoder calls;
- four candidates are shortlisted independently for soft-prior and residual
  decoding;
- the union (at most eight candidates) is hybrid-decoded on a deterministic,
  rate-stratified 512-shot calibration subset;
- no runtime is assigned to the expensive second stage without a completed,
  representative worst-case measurement.

Multi-execution resume cannot make an individual unbounded native decoder call
safe. Canonical Cloud creation and execution are therefore fail-closed before
resource mutation. Reopening the gate requires killable per-decoder work units,
explicit timeout semantics in scientific provenance, and a representative
worst-case 8-vCPU benchmark with a conservative safety margin. The reduced Cloud
path remains an execution check only and cannot support scientific claims.
