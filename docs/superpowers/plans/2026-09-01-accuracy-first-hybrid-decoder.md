# Accuracy-First Hybrid Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible overnight campaign comparing uniform BP-LSD, FNO-conditioned BP-LSD, and residual BP-LSD repair on identical held-out `lp(3,7)_16` code-capacity samples.

**Architecture:** Scientific logic stays in small local Python modules with deterministic packed artifacts. A single conditional FNO supplies calibrated per-bit probabilities to two algebraically checked hybrid decoders. A resumable campaign runner executes locally or through repeated bounded executions of one immutable Cloud Run Job and publishes completion manifests to Cloud Storage.

**Tech Stack:** Python 3.14, NumPy 2.4.1, SciPy 1.17.1, PyTorch 2.11.0, Stim 1.16.0, ldpc 2.4.1, google-cloud-storage 3.13.1, pytest 9.1.1, Ruff 0.16.5, Bash, Docker, gcloud.

## Global Constraints

- The only code under test is the paper's `lp(3,7)_16` seed at `ell=45`.
- The noise model is independent physical Z error with perfect syndrome measurement.
- Accuracy is primary; runtime and BP iterations are diagnostic until accuracy parity is established.
- Every final correction must satisfy `Hx @ correction mod 2 == syndrome`; invalid output is a block failure.
- Training, calibration, and test seeds are disjoint and deterministically derived.
- Test decoders consume identical shots and retain paired outcomes.
- The canonical cloud job is CPU-only, 8 vCPUs, 32 GiB, and has an 8-hour task timeout.
- Each cloud execution stops new work by 7h15m and reserves at least 45 minutes for persistence.
- Canonical cloud creation is asynchronous and explicitly acknowledges multi-execution resume.
- No stage overwrites a completed campaign prefix or trusts an unverified artifact.
- Cloud commands are dry-run by default and require `--execute` for billable mutation.
- Commit messages are concise technical descriptions of repository changes or measured results.

---

## File map

```text
configs/accuracy_campaign.json              canonical campaign policy
src/qldpc_fno/campaign/config.py            validated campaign configuration
src/qldpc_fno/campaign/seeds.py             stable role-separated seeds
src/qldpc_fno/campaign/storage.py           local/GCS completion manifests
src/qldpc_fno/campaign/runner.py            resumable stage state machine
src/qldpc_fno/decoders/hybrid.py            soft-prior and residual decoders
src/qldpc_fno/data/conditional_fields.py    syndrome plus logit(p) tensors
src/qldpc_fno/training/conditional.py       conditional FNO training
src/qldpc_fno/training/calibration.py       deterministic proxy screening and shortlists
src/qldpc_fno/metrics/paired.py              paired decoder statistics
experiments/13_pilot_noise_grid.py          select informative noise points
experiments/14_generate_campaign_shards.py  immutable role-separated shards
experiments/15_train_conditional_fno.py      train and checkpoint one model
experiments/16_calibrate_hybrid_priors.py    select calibration parameters
experiments/17_evaluate_hybrid_decoders.py   adaptive paired evaluation
experiments/18_summarize_accuracy_campaign.py final machine-readable summary
scripts/run_accuracy_campaign.sh            one-command local campaign
scripts/launch_cloud_campaign.sh             dry-run/create/verified-resume cloud launcher
Dockerfile                                   Cloud Run image
.dockerignore                                bounded build context
tests/campaign, tests/decoders, tests/integration focused and end-to-end coverage
README.md                                    progressive public explanation
```

---

### Task 1: Campaign configuration and deterministic seeds

**Files:**
- Create: `configs/accuracy_campaign.json`
- Create: `src/qldpc_fno/campaign/__init__.py`
- Create: `src/qldpc_fno/campaign/config.py`
- Create: `src/qldpc_fno/campaign/seeds.py`
- Create: `tests/campaign/test_config.py`
- Create: `tests/campaign/test_seeds.py`

**Interfaces:**
- Produces: `CampaignConfig.from_json(path: Path) -> CampaignConfig`
- Produces: `derive_seed(campaign_seed: int, *, p_index: int, role: str, shard_index: int) -> int`
- Consumes: no campaign artifacts.

- [ ] **Step 1: Write configuration and seed tests**

```python
def test_canonical_config_has_disjoint_roles_and_bounded_cloud_job() -> None:
    config = CampaignConfig.from_json(Path("configs/accuracy_campaign.json"))
    assert config.noise_grid == (0.003, 0.005, 0.008, 0.012, 0.018, 0.025)
    assert config.cloud_cpu == 8
    assert config.cloud_memory == "32Gi"
    assert config.cloud_timeout_seconds == 8 * 60 * 60
    assert config.max_test_shots_per_point == 200_000
    assert config.target_failures == 200


def test_seed_derivation_is_stable_and_role_separated() -> None:
    train = derive_seed(20260901, p_index=2, role="train", shard_index=7)
    assert train == derive_seed(20260901, p_index=2, role="train", shard_index=7)
    assert train != derive_seed(20260901, p_index=2, role="calibration", shard_index=7)
    assert 0 <= train < 2**63
```

- [ ] **Step 2: Verify the tests fail because campaign modules are absent**

Run: `uv run pytest tests/campaign/test_config.py tests/campaign/test_seeds.py -q`

Expected: import errors for `qldpc_fno.campaign`.

- [ ] **Step 3: Add the canonical JSON policy**

```json
{
  "campaign_seed": 20260901,
  "noise_grid": [0.003, 0.005, 0.008, 0.012, 0.018, 0.025],
  "pilot_shots_per_point": 256,
  "train_shots_cap": 50000,
  "calibration_shots_cap": 10000,
  "calibration_decode_shots_cap": 512,
  "calibration_shortlist_per_method": 4,
  "test_batch_shots": 2048,
  "max_test_shots_per_point": 200000,
  "target_failures": 200,
  "training_epochs": 60,
  "training_batch_size": 128,
  "training_learning_rate": 0.001,
  "training_seed": 1701,
  "checkpoint_every_epochs": 1,
  "cloud_cpu": 8,
  "cloud_memory": "32Gi",
  "cloud_timeout_seconds": 28800,
  "checkpoint_grace_seconds": 2700
}
```

- [ ] **Step 4: Implement strict frozen configuration parsing**

Use a frozen dataclass, reject unknown/missing fields, require increasing probabilities in `(0, 0.5)`, positive caps, `calibration_decode_shots_cap <= calibration_shots_cap`, shortlist size no larger than the fixed 48-tuple grid, `target_failures <= max_test_shots_per_point`, and `2700 <= checkpoint_grace_seconds < cloud_timeout_seconds`.

- [ ] **Step 5: Implement SHA-256 seed derivation**

```python
payload = f"{campaign_seed}:{p_index}:{role}:{shard_index}".encode()
return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
```

Allow only `pilot`, `train`, `calibration`, and `test` roles.

- [ ] **Step 6: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/campaign/test_config.py tests/campaign/test_seeds.py -q`

```bash
git add configs/accuracy_campaign.json src/qldpc_fno/campaign tests/campaign
git commit -m "feat: add deterministic accuracy campaign configuration"
```

---

### Task 2: Calibrated probability transform

**Files:**
- Create: `src/qldpc_fno/training/calibration.py`
- Create: `tests/training/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationParameters(alpha: float, beta: float, temperature: float)`
- Produces: `calibrated_probabilities(logits, error_rates, parameters) -> np.ndarray`
- Produces: `select_calibration(scores: Sequence[CalibrationScore]) -> CalibrationScore`

- [ ] **Step 1: Write formula, clipping, and deterministic selection tests**

```python
def test_calibrated_probabilities_apply_noise_condition_and_clip() -> None:
    logits = np.array([[[-100.0, 0.0, 100.0]]])
    rates = np.array([0.01])
    params = CalibrationParameters(alpha=1.0, beta=1.0, temperature=2.0)
    result = calibrated_probabilities(logits, rates, params)
    assert result.shape == logits.shape
    assert np.all(result >= 1e-5)
    assert np.all(result <= 1 - 1e-5)
    assert result[0, 0, 0] < result[0, 0, 1] < result[0, 0, 2]


def test_selection_prioritizes_validity_then_block_errors_then_nll() -> None:
    valid = CalibrationScore(
        parameters=CalibrationParameters(1.0, 1.0, 1.0),
        invalid_count=0,
        block_errors=2,
        nll=0.5,
    )
    invalid = CalibrationScore(
        parameters=CalibrationParameters(0.5, 0.0, 2.0),
        invalid_count=1,
        block_errors=0,
        nll=0.1,
    )
    assert select_calibration([invalid, valid]) == valid
```

- [ ] **Step 2: Run tests and observe missing module failure**

Run: `uv run pytest tests/training/test_calibration.py -q`

- [ ] **Step 3: Implement the transform**

Broadcast `logit(p)=log(p)-log1p(-p)` over correction coordinates, compute
`sigmoid(alpha * logits / temperature + beta * logit(p))`, and clip.

- [ ] **Step 4: Implement the fixed calibration grid**

Use:

```python
alpha in (0.25, 0.5, 1.0, 2.0)
beta in (0.0, 0.5, 1.0)
temperature in (0.5, 1.0, 2.0, 4.0)
```

Sort candidates by `(invalid_count, block_errors, nll, alpha, beta, temperature)`.
Persist every candidate score; never tune from the test split.

- [ ] **Step 5: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/training/test_calibration.py -q`

```bash
git add src/qldpc_fno/training/calibration.py tests/training/test_calibration.py
git commit -m "feat: calibrate fno correction probabilities"
```

---

### Task 3: Soft-prior and residual hybrid decoders

**Files:**
- Create: `src/qldpc_fno/decoders/hybrid.py`
- Modify: `src/qldpc_fno/decoders/__init__.py`
- Create: `tests/decoders/test_hybrid.py`

**Interfaces:**
- Produces: `HybridDecodeResult`
- Produces: `decode_soft_prior_batch(hx, syndromes, logical_x, probabilities) -> HybridDecodeResult`
- Produces: `decode_residual_batch(hx, syndromes, logical_x, probabilities) -> HybridDecodeResult`

- [ ] **Step 1: Write exact-validity tests on a small parity check matrix**

```python
def test_soft_prior_decoder_updates_the_channel_per_shot() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    probabilities = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])
    result = decode_soft_prior_batch(hx, syndromes, logical_x, probabilities)
    assert np.all(result.syndrome_valid)
    assert result.iterations.shape == (2,)


def test_residual_repair_satisfies_affine_syndrome_identity() -> None:
    hx = sparse.csr_matrix([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    logical_x = sparse.csr_matrix([[1, 1, 1]], dtype=np.uint8)
    syndromes = np.array([[1, 0]], dtype=np.uint8)
    probabilities = np.array([[0.1, 0.1, 0.1]])
    proposal = probabilities >= 0.5
    result = decode_residual_batch(hx, syndromes, logical_x, probabilities)
    assert np.array_equal((result.corrections @ hx.T.toarray()) % 2, syndromes)
    assert np.array_equal(result.residual_before, syndromes ^ ((proposal @ hx.T) % 2))
```

- [ ] **Step 2: Run tests and observe missing hybrid decoder**

Run: `uv run pytest tests/decoders/test_hybrid.py -q`

- [ ] **Step 3: Factor the pinned BP-LSD constructor**

Move the common configuration from `bplsd.py` into a private `_new_decoder(hx,
error_channel)` helper without changing baseline behavior.

- [ ] **Step 4: Implement soft-prior decoding**

Construct one decoder, call `update_channel_probs(probabilities[shot])`, decode,
then record correction, logical prediction, validity, convergence, `decoder.iter`,
and latency.

- [ ] **Step 5: Implement residual repair**

For each shot:

```python
proposal = probabilities[shot] >= 0.5
residual = syndrome ^ (hx @ proposal % 2)
uncertainty = np.minimum(probabilities[shot], 1 - probabilities[shot])
decoder.update_channel_probs(np.clip(uncertainty, 1e-5, 1 - 1e-5))
delta = decoder.decode(residual)
correction = proposal ^ delta
```

Verify the final syndrome independently. Record proposal weight, residual syndrome
weight, delta weight, iterations, and split latency.

- [ ] **Step 6: Run baseline and hybrid regression tests**

Run: `uv run ruff check . && uv run pytest tests/decoders -q`

- [ ] **Step 7: Commit**

```bash
git add src/qldpc_fno/decoders tests/decoders
git commit -m "feat: add syndrome-valid hybrid decoders"
```

---

### Task 4: Noise-conditioned ring fields and training

**Files:**
- Create: `src/qldpc_fno/data/conditional_fields.py`
- Create: `src/qldpc_fno/training/conditional.py`
- Create: `tests/data/test_conditional_fields.py`
- Create: `tests/training/test_conditional.py`

**Interfaces:**
- Produces: `add_noise_channel(syndromes, error_rates) -> np.ndarray`
- Produces: `train_conditional_fno(inputs, targets, split, config, checkpoint_dir) -> TrainingResult`

- [ ] **Step 1: Test the broadcast `logit(p)` channel**

```python
def test_noise_channel_is_broadcast_without_changing_syndrome_fields() -> None:
    syndromes = np.zeros((2, 21, 45), dtype=np.float32)
    result = add_noise_channel(syndromes, np.array([0.01, 0.02]))
    assert result.shape == (2, 22, 45)
    assert np.array_equal(result[:, :21], syndromes)
    assert np.all(result[0, 21] == np.log(0.01 / 0.99))
```

- [ ] **Step 2: Test deterministic checkpoint resumption**

Train two epochs uninterrupted and one epoch plus resume; assert equal final state
dicts, optimizer state, epoch, and loss history.

- [ ] **Step 3: Implement conditional field construction**

Validate binary syndrome fields and one rate per shot. Return owned C-contiguous
`float32` storage.

- [ ] **Step 4: Implement mini-batch conditional training**

Use `RingFNO(in_channels=22, out_channels=58, width=32, modes=12, depth=2)`, Adam,
weighted BCE, deterministic shuffled indices from the training seed, validation NLL,
and one checkpoint per epoch. Store model, optimizer, RNG state, epoch, and hashes.

- [ ] **Step 5: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/data/test_conditional_fields.py tests/training/test_conditional.py -q`

```bash
git add src/qldpc_fno/data/conditional_fields.py src/qldpc_fno/training/conditional.py tests/data tests/training
git commit -m "feat: train noise-conditioned ring fno"
```

---

### Task 5: Pilot grid and role-separated sample shards

**Files:**
- Create: `experiments/13_pilot_noise_grid.py`
- Create: `experiments/14_generate_campaign_shards.py`
- Create: `src/qldpc_fno/campaign/shards.py`
- Create: `tests/campaign/test_shards.py`
- Create: `tests/integration/test_campaign_data_clis.py`

**Interfaces:**
- Produces: `select_noise_points(pilot_rows) -> Sequence[float]`
- Produces immutable `pilot/`, `train/`, `calibration/`, and `test/` manifests.

- [ ] **Step 1: Test pilot selection rules**

Cover zero-failure extension, retention of low-noise controls, insertion of a
midpoint before baseline failure exceeds 50%, and deterministic ordering.

- [ ] **Step 2: Test seed-role disjointness in emitted manifests**

Generate two small rates with eight shots per role; assert no seed is reused and
every `b8` hash replays.

- [ ] **Step 3: Implement the pilot CLI**

For each configured point, build the exact DEM, sample 256 shots, run baseline
BP-LSD, and write one row containing errors, Wilson interval, validity, convergence,
and latency. Apply the selection rules and write `pilot/selection.json`.

- [ ] **Step 4: Implement shard generation**

Write shards of at most 2,048 shots. Each manifest includes role, rate, rate index,
shard index, seed, dimensions, source DEM/code hashes, and packed file hashes. Refuse
cross-role output paths and existing completion manifests.

- [ ] **Step 5: Run reduced CLIs and commit**

Run: `uv run pytest tests/campaign/test_shards.py tests/integration/test_campaign_data_clis.py -q`

```bash
git add experiments/13_pilot_noise_grid.py experiments/14_generate_campaign_shards.py src/qldpc_fno/campaign/shards.py tests/campaign tests/integration/test_campaign_data_clis.py
git commit -m "feat: generate role-separated decoder campaign shards"
```

---

### Task 6: Conditional model and calibration CLIs

**Files:**
- Create: `experiments/15_train_conditional_fno.py`
- Create: `experiments/16_calibrate_hybrid_priors.py`
- Create: `tests/integration/test_campaign_training_clis.py`

**Interfaces:**
- Consumes verified train/calibration shard manifests.
- Produces `model/model.pt`, `model/model.json`, `calibration/progress.json`, `calibration/grid.json`, and `calibration/selected.json`.

- [ ] **Step 1: Write a reduced subprocess test**

Use 16 training shots, eight calibration shots, two epochs, and a two-candidate
calibration grid. Assert the model source hashes reference only training manifests,
the two-stage progress is a strict resumable prefix, the method shortlists are
independent, and selected calibration references only calibration manifests.

- [ ] **Step 2: Implement streaming shard loading**

Unpack only the requested batch from `b8`; do not materialize the 50,000-shot target
tensor as float32. Convert each batch into `(B,22,45)` inputs and `(B,58,45)` labels.

- [ ] **Step 3: Implement training CLI with resumption**

Resume only when configuration, Git commit, train-manifest hash, and checkpoint hash
match. Publish `model.json` after the final model hash verifies.

- [ ] **Step 4: Implement calibration CLI**

Freeze the model. Screen every checkpoint/parameter tuple using calibration-only
NLL, threshold-proposal validity, and residual-syndrome-weight proxies. Independently
shortlist soft-prior and residual candidates, decode their union on the declared
deterministic calibration subset, and select each method only from its shortlist.

- [ ] **Step 5: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/integration/test_campaign_training_clis.py -q`

```bash
git add experiments/15_train_conditional_fno.py experiments/16_calibrate_hybrid_priors.py tests/integration/test_campaign_training_clis.py
git commit -m "feat: train and calibrate hybrid decoder priors"
```

---

### Task 7: Paired metrics and adaptive evaluation

**Files:**
- Create: `src/qldpc_fno/metrics/paired.py`
- Create: `experiments/17_evaluate_hybrid_decoders.py`
- Create: `tests/metrics/test_paired.py`
- Create: `tests/integration/test_hybrid_evaluation_cli.py`

**Interfaces:**
- Produces: `paired_decoder_summary(outcomes, *, bootstrap_seed, samples=10_000)`
- Produces append-only per-batch outcomes and per-rate summaries.

- [ ] **Step 1: Test paired disagreement and bootstrap reproducibility**

```python
def test_paired_summary_counts_disagreements_and_is_reproducible() -> None:
    baseline = np.array([0, 0, 1, 1], dtype=bool)
    hybrid = np.array([0, 1, 0, 1], dtype=bool)
    first = paired_decoder_summary(baseline, hybrid, bootstrap_seed=9, samples=1000)
    second = paired_decoder_summary(baseline, hybrid, bootstrap_seed=9, samples=1000)
    assert first == second
    assert first["baseline_only_failure"] == 1
    assert first["hybrid_only_failure"] == 1
```

- [ ] **Step 2: Implement paired statistics**

Report the 2x2 disagreement table, mean paired block-error delta, percentile 95%
bootstrap interval, and Wilson intervals for each decoder. Bootstrap shot indices,
not individual logical observables.

- [ ] **Step 3: Implement adaptive evaluation**

For every selected rate, consume deterministic test batches. Run baseline, soft
hybrid, and residual hybrid on identical shots. After each batch, checkpoint raw
boolean outcomes and metrics. Stop only when all three reach the target failure
count, the shot cap is reached, or the campaign deadline requests finalization.

- [ ] **Step 4: Enforce accuracy gates**

Any syndrome-invalid output is a failure. Mark a hybrid `accuracy_compatible` only
when validity is 100% and the paired block-error delta interval includes zero or is
strictly below zero. Do not use latency in this decision.

- [ ] **Step 5: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/metrics/test_paired.py tests/integration/test_hybrid_evaluation_cli.py -q`

```bash
git add src/qldpc_fno/metrics/paired.py experiments/17_evaluate_hybrid_decoders.py tests/metrics tests/integration/test_hybrid_evaluation_cli.py
git commit -m "feat: evaluate paired hybrid decoder accuracy"
```

---

### Task 8: Resumable storage and campaign state machine

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/qldpc_fno/campaign/storage.py`
- Create: `src/qldpc_fno/campaign/runner.py`
- Create: `experiments/18_summarize_accuracy_campaign.py`
- Create: `tests/campaign/test_storage.py`
- Create: `tests/campaign/test_runner.py`

**Interfaces:**
- Produces: `ArtifactStore` protocol with local and `gs://` implementations.
- Produces: `CampaignRunner.run(deadline_monotonic: float | None) -> CampaignStatus`.

- [ ] **Step 1: Pin the storage client**

Run: `uv add 'google-cloud-storage==3.13.1'`

- [ ] **Step 2: Test completion-manifest-last semantics**

Use a fake store. Assert partial files never cause a stage skip, digest mismatch
forces rerun, and a valid completion manifest makes rerun idempotent.

- [ ] **Step 3: Implement local and GCS stores**

Methods: `exists`, `download`, `upload`, `read_json`, `publish_directory`, and
`verify_completion`. Upload to `<prefix>/.partial/<uuid>/`, verify server-visible
sizes/digests, copy into the immutable stage prefix, then write `_COMPLETE.json`.

- [ ] **Step 4: Implement the stage state machine**

Stages are `pilot`, `shards`, `training`, `calibration`, `evaluation`, and `summary`.
Before every bounded work unit, compare monotonic time to `deadline - grace`. On
deadline, publish a small `status="partial_deadline"` summary first, then attempt an
optional changed-stage snapshot while time remains. Propagate the absolute deadline
through every store materialization/publication and exit zero when finalization is
bounded by a slow or failing store.

- [ ] **Step 5: Implement final summary**

Aggregate verified manifests into `summary/results.json` and `summary/results.md`.
Include code/noise scope, Git commit, selected rates, model/calibration hashes,
decoder metrics, paired intervals, validity, timing diagnostics, completion state,
and reasons for every early stop.

- [ ] **Step 6: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/campaign/test_storage.py tests/campaign/test_runner.py -q`

```bash
git add pyproject.toml uv.lock src/qldpc_fno/campaign experiments/18_summarize_accuracy_campaign.py tests/campaign
git commit -m "feat: add resumable decoder campaign runner"
```

---

### Task 9: One-command reduced and canonical local campaign

**Files:**
- Create: `scripts/run_accuracy_campaign.sh`
- Create: `tests/integration/test_accuracy_campaign_cli.py`

**Interfaces:**
- Consumes canonical config plus `CAMPAIGN_OUTPUT` and documented reduced overrides.
- Produces complete local campaign artifact tree.

- [ ] **Step 1: Write the reduced end-to-end test**

Run the shell script with eight pilot shots, 24 train, eight calibration, eight test,
one epoch, one calibration candidate, and a fresh temporary output. Assert every
stage completion manifest, three decoder summaries, paired table, and README-ready
summary exist.

- [ ] **Step 2: Implement fail-fast orchestration**

Use `set -euo pipefail`, refuse existing output, call numbered CLIs in order, and
support `CAMPAIGN_REDUCED=1` only for tests. Canonical mode reads all limits from
the committed config and enforces accuracy gates.

- [ ] **Step 3: Run reduced campaign twice**

The first run completes. The second run using the same output must refuse overwrite.
A resume invocation with `--resume` must verify and skip completed stages.

- [ ] **Step 4: Run and commit**

Run: `uv run pytest tests/integration/test_accuracy_campaign_cli.py -q`

```bash
git add scripts/run_accuracy_campaign.sh tests/integration/test_accuracy_campaign_cli.py
git commit -m "feat: add one-command hybrid accuracy campaign"
```

---

### Task 10: Cloud Run container and safe launcher

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `scripts/launch_cloud_campaign.sh`
- Create: `tests/integration/test_cloud_launcher.py`

**Interfaces:**
- Dry-run prints project, region, repository, image, bucket, prefix, CPU, memory,
  timeout, Git commit, and exact gcloud mutations.
- `--execute --multi-execution` creates only uniquely named canonical resources and
  launches the first execution asynchronously; `--execute --resume` verifies and
  executes only the exact existing commit-bound job.

- [ ] **Step 1: Write dry-run tests with a fake `gcloud` executable**

Assert no mutating fake command executes without `--execute`. Assert canonical
creation refuses without `--multi-execution`, uses 8 CPU, 32Gi, 8h, no GPU flag, a
unique campaign prefix, and the current commit. Assert verified resume never
recreates resources and refuses any job identity or image mismatch.

- [ ] **Step 2: Add a pinned CPU image**

```dockerfile
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY experiments/01_build_lp_codes.py experiments/02_validate_lp_codes.py experiments/
COPY experiments/13_pilot_noise_grid.py experiments/14_generate_campaign_shards.py experiments/
COPY experiments/15_train_conditional_fno.py experiments/16_calibrate_hybrid_priors.py experiments/
COPY experiments/17_evaluate_hybrid_decoders.py experiments/
COPY configs/accuracy_campaign.json configs/accuracy_campaign_cloud_reduced.json configs/
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "qldpc_fno.campaign.runner"]
```

Before implementation, verify the uv image tag exists; if not, use the installed
`uv --version` as the exact tag and update this plan line in the implementation
commit.

- [ ] **Step 3: Implement the launcher**

Default region `us-central1`. Confirm active project is non-empty and require a
clean exact commit. Submit a Git archive containing only the declared tracked
runtime paths. On creation, require all uniquely named resources to be absent;
never update or bypass a collision. On resume, require those exact resources,
identity label, and commit-tagged image to match before executing only the job.

Use:

```text
--cpu=8
--memory=32Gi
--task-timeout=8h
--max-retries=0
--tasks=1
```

Pass bucket/prefix/config/commit as environment variables. Print cleanup commands
but do not run them automatically.

- [ ] **Step 4: Build and run the container locally in reduced mode**

Run: `docker build -t qldpc-fno:accuracy .`

Run the image with a mounted temporary artifact directory and reduced configuration.
Expected: verified final summary and exit zero.

- [ ] **Step 5: Run and commit**

Run: `uv run ruff check . && uv run pytest tests/integration/test_cloud_launcher.py -q`

```bash
git add Dockerfile .dockerignore scripts/launch_cloud_campaign.sh tests/integration/test_cloud_launcher.py
git commit -m "feat: add bounded cloud accuracy campaign"
```

---

### Task 11: Progressive public README

**Files:**
- Modify: `README.md`
- Create: `docs/RESULTS.md`
- Create: `tests/test_readme_contract.py`

**Interfaces:**
- README links exact local/cloud commands and immutable results.
- `docs/RESULTS.md` is generated from `summary/results.json`, then reviewed for claim scope.

- [ ] **Step 1: Test required audience and reproducibility sections**

Assert README contains headings for `Two-minute explanation`, `For engineers`, `For
scientists`, `Run locally`, `Run on Google Cloud`, `Results`, `Limitations`, and
`Roadmap`, plus the phrases `code-capacity`, `distance upper bound`, and
`syndrome-valid`.

- [ ] **Step 2: Rewrite the README progressively**

Open with a plain-language explanation of a syndrome as a consistency alarm and a
decoder as a proposed repair. Explain why one wrong bit can invalidate an otherwise
99%-accurate vector. Follow with engineer commands/artifacts, then scientific code,
noise, decoder, split, calibration, paired statistics, and limitations.

- [ ] **Step 3: Add results without premature claims**

Preserve the first smoke result as a negative finding. Add campaign results only
from verified `results.json`. If the overnight campaign is partial, label it
partial and show achieved shots/failures. Never claim speedup in this phase.

- [ ] **Step 4: Run and commit**

Run: `uv run pytest tests/test_readme_contract.py -q`

```bash
git add README.md docs/RESULTS.md tests/test_readme_contract.py
git commit -m "docs: explain hybrid decoder accuracy campaign"
```

---

### Task 12: Full verification, cloud pilot, and overnight launch

**Files:**
- Modify only if verification exposes a tested defect.

**Interfaces:**
- Produces a pushed branch and a launched, bounded Cloud Run execution.

- [ ] **Step 1: Run complete local verification**

```bash
uv sync --all-groups
uv run ruff check .
uv run pytest -q
CAMPAIGN_REDUCED=1 CAMPAIGN_OUTPUT=artifacts/accuracy-final-local \
  bash scripts/run_accuracy_campaign.sh
git status --short
```

Expected: all checks pass, reduced campaign completes, and generated artifacts are
ignored.

- [ ] **Step 2: Request independent code review**

Review the branch against the design and this plan. Resolve every Critical and
Important correctness, provenance, cloud-safety, or statistical finding.

- [ ] **Step 3: Push the reviewed branch**

```bash
git push -u origin experiment/hybrid-decoder-campaign
```

- [ ] **Step 4: Perform a cloud dry run**

```bash
bash scripts/launch_cloud_campaign.sh
```

Inspect project, region, image, bucket, prefix, resource bounds, commit, and cleanup
commands. Expected: no resources created.

- [ ] **Step 5: Launch a small cloud pilot**

```bash
bash scripts/launch_cloud_campaign.sh --execute --reduced
```

Wait for completion, download the summary, and verify all hashes locally. If this
fails, diagnose before launching canonical work.

- [ ] **Step 6: Launch the overnight campaign**

```bash
bash scripts/launch_cloud_campaign.sh --execute --multi-execution
```

Record the Cloud Run execution name and GCS summary prefix. Do not wait locally;
the cloud job owns its deadline and persistence.

Continue a partial canonical campaign from the identical clean commit with:

```bash
CAMPAIGN_ID=<original-id> CLOUD_REGION=<original-region> \
  bash scripts/launch_cloud_campaign.sh --execute --resume
```

- [ ] **Step 7: Commit any machine-generated result pointers only after completion**

Do not commit raw shards or model binaries. Commit a small verified results JSON,
`docs/RESULTS.md`, and the GCS campaign URI only when the execution reaches complete
or partial-deadline state.

---

### Task 13: Release-hardening findings

**Files:**
- Modify: `.dockerignore`
- Modify: `Dockerfile`
- Modify: `configs/accuracy_campaign.json`
- Modify: `configs/accuracy_campaign_cloud_reduced.json`
- Modify: `experiments/16_calibrate_hybrid_priors.py`
- Modify: `scripts/launch_cloud_campaign.sh`
- Modify: `scripts/run_accuracy_campaign.sh`
- Modify: `src/qldpc_fno/campaign/config.py`
- Modify: `src/qldpc_fno/campaign/runner.py`
- Modify: `src/qldpc_fno/campaign/storage.py`
- Modify: focused campaign, calibration, and launcher tests
- Modify: `README.md`, methodology and reproducibility documentation

**Interfaces:**
- Produces: deterministic calibration screening, method-specific shortlists, and
  a rate-stratified decode subset whose complete policy is hash-bound on resume.
- Produces: artifact-store operations with optional absolute monotonic deadlines.
- Produces: canonical cloud `--multi-execution` creation and `--resume` execution
  modes against one immutable job/store identity.

- [ ] **Step 1: Preserve the pre-change benchmark**

Run the existing `_score_candidate` path over verified calibration-role shots
from the current-HEAD reduced artifact, with one warm-up and five timed trials.
Record hardware, threads, artifact identity, timed region, raw samples, median,
and scope limitations in `docs/calibration-throughput-benchmark.md`.

- [ ] **Step 2: Write failing configuration and calibration tests**

Assert a 512-shot deterministic stratified subset, four candidates per method,
independent shortlist membership/selection, strict progress provenance, no test
input, and a 45-minute finalization reserve. Run focused tests and confirm failure
for missing fields and APIs.

- [ ] **Step 3: Implement deterministic two-stage calibration**

Screen all checkpoint/parameter pairs using calibration-only NLL and proposal
validity/residual-weight proxies. Persist one bounded checkpoint screen or hybrid
decode work unit per runner invocation. Decode the union of independent shortlists
on the configured subset, then select each hybrid only from its own shortlist.

- [ ] **Step 4: Write failing cloud-context and resume tests**

Place an ignored sentinel under a normally copied directory. Assert cloud submit
receives a temporary archive created by `git archive` from the exact clean commit,
the sentinel is absent, `.dockerignore` starts with `**`, canonical execute refuses
without `--multi-execution`, and `--resume` executes only an existing job.

- [ ] **Step 5: Implement exact build context and cloud resume**

Archive only Dockerfile, lock/package metadata, the two campaign configs, required
numbered entry points, and `src/qldpc_fno` from the clean commit. Keep the eight-hour
outer job limit, require explicit multi-execution acknowledgement for canonical
creation, and execute the existing job for resume without recreating resources.

- [ ] **Step 6: Write failing deadline/store tests**

Assert the small partial summary is attempted before a changed-stage snapshot,
all publication/materialization operations receive the absolute deadline, and
delayed/failing stores return a bounded partial-deadline result without publishing
an unverified completion.

- [ ] **Step 7: Implement bounded finalization**

Set `checkpoint_grace_seconds=2700`. Thread the absolute monotonic deadline through
runner publications and storage APIs, convert it to bounded GCS request timeouts,
publish a partial summary first, then attempt optional snapshots while time remains.

- [ ] **Step 8: Correct operator documentation**

Document prerequisites, local canonical/reduced/resume commands, cloud dry-run,
multi-execution creation/resume, cleanup, billing, and exact CPU/memory/time bounds.
Separate runner-store and manual layouts. Remove campaign runner source-lock claims
and update the repository map.

- [ ] **Step 9: Verify before release**

Run focused red/green tests, the full suite, Ruff, Bash syntax checks, lock checks,
Docker/build-context inspection, and a fresh reduced local campaign plus resume.
Review `git diff --check` and request independent final re-review.

---

## Completion audit

- [ ] Uniform, soft-prior, and residual decoders see identical held-out shots.
- [ ] All final corrections are independently syndrome-checked.
- [ ] Train, calibration, and test seeds are disjoint.
- [ ] Calibration never reads test outcomes.
- [ ] Paired statistical intervals are deterministic and shot-level.
- [ ] Local artifacts and GCS stages reject corruption and cross-run mixing.
- [ ] Reduced local and container campaigns complete from clean state.
- [ ] Cloud dry run creates nothing; execution requires `--execute`.
- [ ] Cloud job has 8 CPU, 32Gi, no GPU, zero retries, and an 8-hour timeout.
- [ ] README serves lay readers, engineers, and scientists without overstating scope.
- [ ] No speed claim is made unless a later experiment establishes it.
