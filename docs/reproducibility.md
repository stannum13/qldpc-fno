# Reproducibility and artifacts

This guide describes how runs are identified, published, checked, and resumed.
For the scientific definitions behind the artifacts, see
[Experiment methodology](experiment-methodology.md).

## Environment and source identity

Run commands from the repository root. The environment contract is:

- Python `>=3.14,<3.15`;
- the committed `uv.lock`;
- runtime versions pinned in `pyproject.toml`; and
- source references captured by `experiments/00_lock_sources.py`.

Create the environment and verify the repository with:

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

Keep the Git commit, config file, `source-lock.json`, and all JSON manifests with
reported results. Smoke manifests and campaign shard manifests do not record Git
identity, so retain it alongside those artifacts. Shards bind payload hashes to
the config and code, role, and error-rate/seed coordinates; each role completion
manifest hashes every shard manifest. Campaign training introduces Git-commit
binding in addition to hashes of the code, config, train-role completion manifest,
and every train shard manifest.

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

Pilot and shard-role publishers write into private staging directories and expose
the final directory only after `manifest.json` is complete. A role directory with
a valid completion manifest is an immutable publication.

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

## Campaign artifact tree

A campaign rooted at `artifacts/accuracy-campaign/` publishes:

| Path | Meaning |
| --- | --- |
| `source-lock.json` | Exact paper, Stim, `ldpc`, and Willow references |
| `code/` | Canonical `Hx`/`Hz`, source metadata, validation, and hashes |
| `pilot/selection.json` | Pilot rows and deterministic noise-point selection |
| `pilot/manifest.json` | Pilot completion and shard hashes |
| `{train,calibration,test}/manifest.json` | Immutable role completion manifest |
| `{role}/rate-*/shard-*/samples.json` | Per-shard identity, dimensions, seed, rate, and file hashes |
| `model/resume.json` | Current teacher/training restart state |
| `model/teacher_progress.json` | Teacher-chunk completion and source identity |
| `model/teacher_chunks/` | Verified, bounded BP-LSD teacher chunks |
| `model/teacher_corrections.b8` | Assembled packed BP-LSD teacher corrections |
| `model/teacher.json` | Assembled teacher cache identity and statistics |
| `model/epoch-*.pt` | Atomic epoch checkpoint candidates |
| `model/model.pt` | Completed frozen FNO state dictionary |
| `model/model.json` | Completed model publication and full provenance |
| `calibration/grid.json` | All evaluated checkpoint/parameter candidates |
| `calibration/selected.json` | Frozen soft-prior and residual selections |

Test shards can be published now, but no final campaign evaluator consumes them
in the current repository.

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

## Interrupted and failed stages

- **Smoke:** there is no stage-level resume. Keep the failed directory for
  diagnosis and choose a fresh output root for another run.
- **Pilot or role generation:** incomplete staging is not a published role. Rerun
  with the intended final role path after diagnosing the failure.
- **Training:** inspect `model/resume.json`; restart only with `--resume`. Existing
  teacher chunks and the checkpoint referenced by `resume.json` are verified
  before reuse.
- **Calibration:** `selected.json` marks completion and is never overwritten.
  Use a fresh campaign tree for a different calibration run.

Do not repair a hash mismatch by editing the manifest. Regenerate the affected
stage from its verified parents so provenance remains meaningful.

## Reporting checklist

For each published result, record:

- repository commit and clean/dirty state;
- config and source-lock hashes;
- role completion and model/calibration manifests;
- shot count and physical error rate per point;
- syndrome-invalid and logical block-error counts, not rates alone;
- confidence interval method where an interval is reported;
- hardware, threads, batch size, and timed region for latency; and
- whether the result came from smoke, calibration, or untouched test data.

This checklist prevents diagnostic smoke or calibration metrics from being
mistaken for final comparative evidence.
