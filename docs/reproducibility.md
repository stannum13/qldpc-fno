# Reproducibility and artifacts

This guide describes how runs are identified, published, checked, and resumed.
For the scientific definitions behind the artifacts, see
[Experiment methodology](experiment-methodology.md).

## Environment and source identity

Run commands from the repository root. The environment contract is:

- Python `>=3.14,<3.15`;
- the committed `uv.lock`;
- runtime versions pinned in `pyproject.toml`; and
- source references that can be captured by the smoke orchestrator or the
  optional manual `experiments/00_lock_sources.py` step.

Create the environment and verify the repository with:

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

Keep the Git commit, config file, and all JSON manifests with reported results.
Also keep `source-lock.json` when the smoke or optional manual source-lock step
was run. The accuracy runner does not publish or verify that file. Smoke manifests
and campaign shard manifests do not themselves record Git identity, so retain it
alongside those artifacts. Shards bind payload hashes to the config and code,
role, and error-rate/seed coordinates; each role completion manifest hashes every
shard manifest. Campaign training introduces Git-commit binding in addition to
hashes of the code, config, train-role completion manifest, and every train shard
manifest.

## Determinism and its limits

Sampling seeds are stored in manifests. Campaign shard seeds are deterministically
derived from the campaign seed, rate index, role, and shard index. Training records
its deterministic stratified split, NumPy state, PyTorch state, optimizer state,
and epoch. PyTorch deterministic algorithms are enabled by the training entry
point.

These controls make same-environment restarts auditable. They do not promise
bit-identical behavior across different CPU architectures, PyTorch builds, BLAS
implementations, or future dependency versions. Latency is especially
environment-specific and should always be reported with hardware, process
placement, batch size, and software versions.

## Immutability contract

Most stage entry points refuse an existing output path. This prevents an old and
new run from being silently mixed. Use a fresh directory for a new smoke run:

```bash
SMOKE_OUTPUT=artifacts/smoke-2026-09-01 bash scripts/run_smoke.sh
```

Do not manually replace data underneath a manifest. Downstream readers verify
hashes for code matrices, DEMs, packed samples, corrections, tensors, checkpoints,
and their source manifests. Campaign shard loading additionally verifies config
and code provenance, role, payload schema, rate coordinates, and deterministically
derived seeds. Git identity is first enforced by training and is rechecked when
the trained model is calibrated; smoke and shard publication do not enforce it.

Selection and shard-role publishers write into private staging directories and
expose the final directory only after `manifest.json` is complete. A role
directory with a valid completion manifest is an immutable publication.
Fixed selections are verified as empty, observation-free publications whose
rates exactly equal the committed grid. Pilot selections additionally verify
every pilot shard and configured shot/rate coordinate, replay baseline failures
from those samples, and recompute the deterministic selected noise points. The
evaluator records that semantic verification in a receipt whose hash is bound
into progress and every immutable evaluation batch. Later batch processes still
rehash the selection, pilot shard manifests, and payloads, but do not repeat the
BP-LSD replay once a verified batch anchors the receipt.

## Smoke artifact tree

For the default command, `artifacts/smoke/` contains:

| Path | Meaning |
| --- | --- |
| `source-lock.json` | Exact paper, Stim, `ldpc`, and Willow references |
| `code/` | `Hx`, `Hz`, code metadata, and validation checks |
| `dem/` | Independent-Z Stim DEM, logical-X basis, and hashes |
| `samples/` | Packed errors, detections, observables, seed, and dimensions |
| `bplsd/` | Teacher corrections, predicted observables, decoder metrics, and timing |
| `tensors/` | Ring-field arrays and contiguous 75%/25% train/test split |
| `fno/` | Frozen model and smoke training/gate metrics |
| `evaluation/metrics.json` | Held-out standalone FNO metrics and batch timing |

Packed `.b8` files use little-endian bit order. Array dimensions, shot counts, and
hashes live in adjacent JSON rather than being inferred from file size alone.

## Runner-store campaign artifact tree

`scripts/run_accuracy_campaign.sh` and the Cloud Run job use the same artifact
store contract. A local store rooted at `artifacts/accuracy-campaign/` (or the
configured Cloud Storage prefix) publishes:

| Path | Meaning |
| --- | --- |
| `inputs/config.json` | Immutable effective campaign configuration |
| `inputs/run-mode.json` | Git commit, config/code hashes, exact controls, execution identity, mode, and permitted claims |
| `inputs/code/` | Canonical `Hx`/`Hz`, source metadata, validation, and hashes |
| `pilot/selection.json` | Pilot-derived or predeclared fixed noise-point selection |
| `pilot/manifest.json` | Selection completion and optional pilot shard hashes |
| `shards/{train,calibration,test}/manifest.json` | Immutable role completion manifest |
| `shards/{role}/rate-*/shard-*/samples.json` | Per-shard identity, seed, rate, dimensions, and file hashes |
| `training/resume.json` | Current teacher/training restart state |
| `training/teacher_progress.json` | Teacher-chunk completion and source identity |
| `training/teacher_chunks/` | Verified, bounded BP-LSD teacher chunks |
| `training/teacher_corrections.b8` | Assembled packed BP-LSD teacher corrections |
| `training/teacher.json` | Assembled teacher cache identity and statistics |
| `training/epoch-*.pt` | Atomic epoch checkpoint candidates |
| `training/model.pt`, `training/model.json` | Completed frozen FNO and full provenance |
| `calibration/progress.json` | Strict two-stage screen/decode restart state and provenance |
| `calibration/grid.json` | Proxy results, method-specific shortlists, and decoded candidates |
| `calibration/selected.json` | Independently frozen soft-prior and residual selections |
| `evaluation/rate-*/batch-*/outcomes.npz` | Immutable paired per-shot outcomes and latency diagnostics |
| `evaluation/rate-*/batch-*/manifest.json` | Batch coordinates, counts, hashes, and provenance |
| `evaluation/selection-verification.json` | Selection semantic-verification identity bound into evaluation provenance |
| `evaluation/rate-*/summary.json` | Per-rate statistics, intervals, validity, and stopping reason |
| `evaluation/progress.json` | Resumable batch-manifest index |
| `evaluation/manifest.json` | Final complete or deadline-partial publication |
| `summary/results.{json,md}` | Final verified scientific summary |
| `.partial-summaries/<sequence>/` | Small, immutable deadline summaries published first |
| `.checkpoints/<stage>/<sequence>/` | Optional verified stage snapshots published after a partial summary |

Evaluation verifies the selection, test, model, teacher, checkpoint, and
calibration chain before decoding. Existing batches are semantically revalidated
on `--resume` without loading all prior outcome arrays at once. A verified
deadline-partial manifest and its derived rate summaries are retired before new
shots are collected: the manifest is removed first, progress is atomically reset
to `in_progress`, and summaries are removed from the highest rate downward so
every interruption state is resumable and checkpointable; immutable batches
remain. Manifest-less summaries from an
interrupted finalization are reconstructed and verified before that same
retirement, and tampered summaries are preserved and rejected.

Final summary generation derives campaign mode and claim permission from the
verified `inputs/run-mode.json`. Canonical permission is granted only after the
manifest is reconstructed from an allowlisted committed canonical config, the
effective config, code and Git hashes, exact overrides, controls, and execution
identity; caller-supplied mode labels cannot grant it. It rechecks every evaluation batch and outcome
hash, deterministic test-shot coordinate, stopping decision, marginal decoder
metric, paired statistic, and comparison status. Rehashing an edited rate summary
inside the mutable evaluation manifest therefore cannot make it reportable.
Verification hashes the exact bytes it parses and requires the terminal progress
index to match every batch-manifest digest exactly.

The store prefix `training/` materializes as the runner's local `model/` working
directory. This is intentional; do not construct manual stage commands from the
remote/store path names.

## Manual stage artifact tree

The individual commands in the README use a different local layout:

| Path | Meaning |
| --- | --- |
| `source-lock.json` | Optional bibliography/software source lock |
| `code/`, `pilot/` | Validated code and noise-point selection |
| `{train,calibration,test}/` | Role shards at the campaign root |
| `model/` | Teacher cache, checkpoints, and frozen model |
| `calibration/{progress,grid,selected}.json` | Calibration restart, grid, and selections beside calibration-role shards |
| `evaluation/` | Resumable paired held-out batches and summaries |

These paths are for direct CLI use only. The orchestrated runner owns the
runner-store mapping above.

## Resuming the campaign runner

The local runner resumes the exact existing store and refuses implicit reuse:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-campaign \
bash scripts/run_accuracy_campaign.sh --resume
```

## Fixed-shot disconfirming run

The predeclared one-rate disconfirming profile fixes selection at `p=0.0375` and
uses fixed-shot evaluation. Start it with:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm
```

Resume the same immutable store with:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm --resume
```

A detected harm result falsifies this candidate at `p=0.0375`. An inconclusive or
benefit result only permits the next experiment and is not evidence of
equivalence, noninferiority, or a positive accuracy result. Baseline tuning, at
least three training seeds, and a larger confirmatory test remain prerequisites
for a positive accuracy claim.

For Cloud Run, repeat the original clean commit, active project, region, and
campaign ID. The launcher verifies the immutable input publication and the
existing job's digest-pinned image, literal environment, command/arguments,
service account, CPU/memory, tasks/parallelism, retry policy, timeout, and
generation-2 Jobs contract. It does not recreate resources:

```bash
CAMPAIGN_ID=accuracy-20260901-a1b2 \
CLOUD_REGION=us-central1 \
bash scripts/launch_cloud_campaign.sh --execute --resume
```

For new reduced jobs, Cloud Build's effective default service account is resolved
and verified before mutation. Its Artifact Registry writer role is scoped only to
the campaign's unique repository and is granted before the build; the runtime
service account receives no registry writer role. Repository deletion removes the
repository-local binding without a separate project-IAM cleanup operation.

The canonical Cloud execution gate is closed because a representative real
candidate did not complete inside its 10m45s diagnostic ceiling. The command above
verifies old resources and provenance but refuses to submit an execution. A future
reviewed change needs killable decoder-unit timeouts and a conservative 8-vCPU
benchmark before reopening the 8-hour / 7h15m work-cutoff policy. Historical
partial status points through this verified launcher and never tells an operator
to bypass provenance checks with a direct execution command.

## Resuming campaign training

Training never resumes implicitly. A first invocation creates the model directory;
later invocations require `--resume` and revalidate the entire identity chain.

To initialize and record identity without generating teacher corrections:

```bash
uv run python experiments/15_train_conditional_fno.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --train artifacts/accuracy-campaign/train \
  --initialize-only \
  --out artifacts/accuracy-campaign/model
```

To bound one worker allocation to a single new teacher chunk:

```bash
uv run python experiments/15_train_conditional_fno.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --train artifacts/accuracy-campaign/train \
  --resume \
  --max-teacher-chunks-this-run 1 \
  --out artifacts/accuracy-campaign/model
```

To resume and run at most one additional epoch:

```bash
uv run python experiments/15_train_conditional_fno.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --train artifacts/accuracy-campaign/train \
  --resume \
  --max-epochs-this-run 1 \
  --out artifacts/accuracy-campaign/model
```

Omit both `--max-*` flags to continue through all remaining teacher chunks and
epochs. A completed model can be revalidated with the same `--resume` command; the
entry point returns without retraining when every recorded artifact is intact.

Resume is intentionally rejected when the config, Git commit, code manifest,
train completion manifest, shard manifests, split, teacher cache, or checkpoint
does not match. Preserve the exact checkout used to start a long run.

## Resuming held-out evaluation

The first evaluator invocation creates the output publication. To bound a worker
allocation, stop after a fixed number of new batches:

`--run-mode` must name the immutable `inputs/run-mode.json` published for the
exact config, code manifest, Git commit, campaign mode, and execution controls.
Verification anchors its claimed canonical config to the committed allowlist and
reconstructs the entire expected manifest rather than trusting its embedded
hashes or claim flag.
The manual preparation commands do not create this file; use one materialized
from the matching campaign-runner input publication.

```bash
uv run python experiments/17_evaluate_hybrid_decoders.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --run-mode /path/to/verified-runner-workspace/inputs/run-mode.json \
  --selection artifacts/accuracy-campaign/pilot/selection.json \
  --test artifacts/accuracy-campaign/test \
  --model artifacts/accuracy-campaign/model \
  --calibration artifacts/accuracy-campaign/calibration \
  --campaign-mode canonical \
  --max-batches-this-run 1 \
  --out artifacts/accuracy-campaign/evaluation
```

Continue the same publication by repeating the command with `--resume`. The
evaluator verifies the frozen inputs, every existing batch manifest and outcome,
and the source/rate/batch-hash coordinates in `progress.json` before decoding
another batch. The first invocation semantically verifies pilot selection; once
an immutable batch binds the resulting receipt, later bounded invocations use
hash/schema verification without replaying all pilot BP-LSD decodes. When a final or deadline-partial manifest exists, it also verifies
the per-rate summaries. Omit `--max-batches-this-run` when the process should
continue until scientific stopping or an externally supplied deadline.

## Interrupted and failed stages

- **Smoke:** there is no stage-level resume. Keep the failed directory for
  diagnosis and choose a fresh output root for another run.
- **Selection or role generation:** incomplete staging is not a published role.
  Rerun with the intended final role path after diagnosing the failure.
- **Training:** inspect `model/resume.json`; restart only with `--resume`. Existing
  teacher chunks and the checkpoint referenced by `resume.json` are verified
  before reuse.
- **Calibration:** each invocation persists one bounded checkpoint screen or
  shortlisted hybrid-decode unit in `progress.json`. Resume through the campaign
  runner or repeat the manual command with `--resume`; the config, calibration
  shards, model, policy, subset, shortlists, and recorded work order are verified
  before reuse. `selected.json` marks completion and is never overwritten. Use a
  fresh campaign tree for a different policy.
- **Evaluation:** restart an incomplete or deadline-partial publication with
  `--resume`. Existing batch hashes, coordinates, outcome semantics, summaries,
  and source provenance are verified before additional batches are decoded.

Do not repair a hash mismatch by editing the manifest. Regenerate the affected
stage from its verified parents so provenance remains meaningful.

## Reporting checklist

For each published result, record:

- repository commit and clean/dirty state;
- config hash, plus the source-lock hash only when the optional manual/smoke step
  was used;
- role completion and model/calibration manifests;
- shot count and physical error rate per point;
- syndrome-invalid and logical block-error counts, not rates alone;
- confidence interval method where an interval is reported;
- hardware, threads, batch size, and timed region for latency; and
- whether the result came from smoke, calibration, or untouched test data.

This checklist prevents diagnostic smoke or calibration metrics from being
mistaken for final comparative evidence.
