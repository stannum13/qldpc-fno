# qLDPC-FNO

Quantum information is fragile: unwanted interactions can change a physical
qubit, while directly inspecting an unknown quantum state would destroy the
information one is trying to protect. Quantum error correction addresses this by
spreading logical information across many physical qubits and measuring
parity-like constraints instead of the logical state itself.

Those measurements produce a **syndrome**—a compact fingerprint of which
constraints were disturbed. Decoding is the classical task of turning that
fingerprint into a correction. The correction must reproduce the measured
syndrome and preserve the encoded logical information; merely resembling another
decoder's bit string is not enough.

Surface-code architectures are the standard local-check reference point, but
their low encoding rate can make physical-qubit overhead important when many
logical qubits are needed. qLDPC codes explore a different tradeoff: sparse checks
with potentially higher encoding rate, usually at the cost of less local
connectivity. This project studies the cyclic lifted-product code `lp(3,7)_16`
from the [source-locked qLDPC paper](https://arxiv.org/abs/2603.28627v1), without
claiming that one code settles the broader architecture question.

The code's repeating length-45 structure makes a Fourier neural operator (FNO) a
plausible way to learn structured decoder priors. BP-LSD remains the conventional
decoder: belief propagation uses the sparse check graph, and localized-statistics
decoding supplies a fallback when message passing is insufficient. The
accuracy-first hypothesis is not that an FNO should replace those constraints,
but that it may help BP-LSD by supplying better soft priors or a proposal that
BP-LSD repairs.

The implemented campaign therefore compares:

1. BP-LSD with a uniform physical-error prior;
2. BP-LSD with per-qubit soft priors from a noise-conditioned Fourier neural
   operator (FNO); and
3. a thresholded FNO proposal followed by BP-LSD repair of the residual syndrome.

Development smoke runs exposed the central failure mode: high agreement with
BP-LSD's correction bits can coexist with failed syndrome checks. Those run
artifacts are not committed, so a fresh clone cannot audit a particular numeric
result. The durable lesson is encoded in the pipeline: teacher-bit accuracy is a
training diagnostic, syndrome validity is reported explicitly, and the held-out
comparison counts invalid corrections as block failures. Speed comes
only after accuracy and validity.

> **Status:** the smoke pipeline and the accuracy-campaign implementation are
> complete through held-out paired evaluation. No canonical campaign result is
> reported yet, so this repository does not claim that either learned method
> improves accuracy or latency over uniform BP-LSD.

## Why this pairing?

Lifted-product checks repeat under cyclic shifts. The ring FNO applies the same
learned rule at each cyclic position and mixes long-range information through a
small set of Fourier modes. That structural match makes the model a reasonable
prior generator, but it does not make the model a decoder by itself: an
unconstrained thresholded output need not satisfy the syndrome.

BP-LSD supplies the missing algebraic discipline. It is a strong baseline, but it
performs iterative graph decoding and may invoke a localized fallback for each
shot. Whether an FNO prior improves accuracy, reduces decoder work, or only adds
overhead is an empirical question. This repository measures correctness before
making any timing comparison.

Read [Research background](docs/background.md) for the full motivation,
[Concepts](docs/concepts.md) for the vocabulary, and
[Experiment methodology](docs/experiment-methodology.md) for the exact implemented
study.

## Experiment flow

```mermaid
flowchart LR
    A[Published lifted-product seed] --> B[Validated Hx and Hz]
    B --> C[Independent-Z Stim DEM]
    C --> D[Pilot noise grid]
    D --> E[Selected noise points]
    E --> F[Train / calibration / test shards]
    F --> G[Uniform-prior BP-LSD teacher]
    G --> H[Noise-conditioned ring FNO]
    H --> I[Calibration]
    I --> J[Soft-prior BP-LSD]
    I --> K[Hard proposal + residual BP-LSD]
    E -. baseline .-> L[Uniform-prior BP-LSD]
    J --> M[Paired held-out evaluation]
    K --> M
    L --> M
    M --> N[Accuracy gate before speed]
```

The code-capacity model applies independent physical Z errors with perfect
syndrome measurements. `Hx` maps a Z-error vector to its syndrome. The code's
`ell = 45` cyclic coordinate reshapes 945 syndrome bits into 21 channels and
2,610 correction bits into 58 channels. Campaign training adds the physical
error rate's log-odds, broadcast over the ring, as a 22nd input channel.

## Quickstart

### 1. Install and check the environment

The project requires Python `>=3.14,<3.15` and uses a committed `uv.lock`.
Install [`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

### 2. Exercise the complete smoke pipeline

The canonical smoke run uses 512 shots and 600 optimization steps:

```bash
bash scripts/run_smoke.sh
```

For a quick CLI execution check, use a fresh output directory and explicitly
reduce the work:

```bash
SMOKE_OUTPUT=artifacts/quick \
SMOKE_SHOTS=16 \
SMOKE_STEPS=2 \
bash scripts/run_smoke.sh
```

Setting `SMOKE_STEPS` disables the canonical overfit gates. The reduced command
checks that every stage executes; its metrics are not scientific decoder results.
Without that override, the script enforces all training gates and stops before
held-out evaluation if any gate fails. The smoke orchestrator refuses to overwrite
an existing output directory.

### 3. Run the accuracy campaign locally

The campaign runner pins canonical work to the committed configuration and a
clean Git checkout. It refuses an existing output unless `--resume` is explicit.
Allow substantial local disk and CPU time; the canonical caps are 50,000 training
shots, 10,000 calibration shots, and 200,000 test shots per selected rate, with
60 training epochs. A single invocation is not expected to finish that work.

Preview the committed policy in `configs/accuracy_campaign.json`, then start a
fresh canonical campaign:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-canonical \
bash scripts/run_accuracy_campaign.sh
```

When the runner reaches its work cutoff, it publishes a small partial summary
before attempting an optional verified stage snapshot. Continue the exact same
campaign and Git commit with:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-canonical \
bash scripts/run_accuracy_campaign.sh --resume
```

For a fast end-to-end execution check, opt into reduced mode and declare every
reduction. Its outputs are labelled `reduced_non_scientific` and must not be
reported as decoder evidence:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-reduced \
CAMPAIGN_REDUCED=1 \
CAMPAIGN_PILOT_SHOTS=8 \
CAMPAIGN_TRAIN_SHOTS=24 \
CAMPAIGN_CALIBRATION_SHOTS=8 \
CAMPAIGN_TEST_SHOTS=8 \
CAMPAIGN_EPOCHS=1 \
CAMPAIGN_CALIBRATION_CANDIDATES=1 \
bash scripts/run_accuracy_campaign.sh
```

### 4. Run the fixed-shot disconfirming experiment

The predeclared disconfirming profile evaluates its one selected rate, `p=0.0375`,
with fixed selection and a fixed 2,048-shot test cap. Start it with:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm
```

Resume the exact existing campaign with:

```bash
CAMPAIGN_OUTPUT=artifacts/accuracy-disconfirm-p0375 \
bash scripts/run_accuracy_campaign.sh --disconfirm --resume
```

A detected harm result falsifies this candidate at `p=0.0375`. An inconclusive or
benefit result only permits the next experiment; it is not a positive accuracy
claim. Baseline tuning, at least three training seeds, and a larger confirmatory
test remain required before any positive accuracy claim.

### 5. Run the accuracy campaign on Google Cloud

Cloud execution requires `gcloud`, an authenticated account, an active project
with billing, and permission to use Cloud Build, Artifact Registry, Cloud Run,
Cloud Storage, IAM service accounts, and IAM policy bindings. Enable the required
APIs before launching. Start from the exact clean commit that should identify the
campaign and choose a stable, unique lowercase `CAMPAIGN_ID` and region.

Before any mutation, the launcher resolves and verifies the project's effective
default Cloud Build service account and prints it as `build_service_account`.
After creating the unique Artifact Registry repository, it grants that principal
`roles/artifactregistry.writer` on that repository only, before submitting the
build. It never grants writer at project scope or to the campaign runtime service
account. Deleting the unique repository also removes its repository-local IAM
binding.

The launcher is dry-run by default. It prints every resource, mutation, resume
command, and cleanup command without changing Google Cloud:

```bash
CAMPAIGN_ID=accuracy-20260901-a1b2 \
CLOUD_REGION=us-central1 \
bash scripts/launch_cloud_campaign.sh
```

Run the small, synchronous execution check with:

```bash
CAMPAIGN_ID=accuracy-check-20260901 \
CLOUD_REGION=us-central1 \
bash scripts/launch_cloud_campaign.sh --execute --reduced
```

Canonical Cloud execution is currently fail-closed. A hash-verified diagnostic
did not finish one real eight-shot candidate within 10 minutes 45 seconds, so no
safe bound exists for a 512-shot candidate work unit. The launcher refuses before
creating resources even when multi-execution is acknowledged:

```bash
CAMPAIGN_ID=accuracy-20260901-a1b2 \
CLOUD_REGION=us-central1 \
bash scripts/launch_cloud_campaign.sh --execute --multi-execution
```

Reopen this gate only in a reviewed code change after representative worst-case
hybrid decoding has a killable per-unit timeout policy and a conservative 8-vCPU
benchmark. Do not use the reduced check for scientific results.

Each execution uses one task, 8 vCPUs, 32 GiB memory, no GPU, no retries, and an
8-hour Cloud Run timeout. New work stops by 7 hours 15 minutes, leaving at least
45 minutes to publish a small partial summary and optional verified snapshots.
For a canonical job created by an older revision, the resume command still checks
the identical clean commit, project, region, campaign ID, immutable inputs, and
full job contract, but this revision refuses to submit another execution:

```bash
CAMPAIGN_ID=accuracy-20260901-a1b2 \
CLOUD_REGION=us-central1 \
bash scripts/launch_cloud_campaign.sh --execute --resume
```

Resume verification covers the existing repository, bucket, service account,
immutable input publication, digest-pinned image, command/arguments, literal
environment, CPU, memory, tasks, parallelism, retry policy, timeout, and the
generation-2 Cloud Run Jobs execution contract. It neither recreates resources
nor bypasses name collisions, and then fails closed at the benchmark gate.
Historical partial status may print the exact
`launch_cloud_campaign.sh --execute --resume` verification command; that command
does not bypass the gate.

Cloud Build, Artifact Registry, Cloud Run, and Cloud Storage can incur charges;
stored artifacts continue to incur charges after a job stops. The launcher only
prints cleanup commands. Inspect the campaign bucket and retain required results
before manually running them: object deletion is recursive and irreversible.
Delete the job, bucket objects, bucket, repository, and campaign service account
when they are no longer needed.

### 5. Run individual stages manually

The campaign is also exposed as individual, potentially resource-intensive
stages. The manual layout below differs deliberately from the campaign runner's
store layout described in [Reproducibility](docs/reproducibility.md). The optional
source-lock step records bibliography/software references for manual reporting;
the runner itself does not publish or verify `source-lock.json`.

```bash
uv run python experiments/00_lock_sources.py \
  --out artifacts/accuracy-campaign/source-lock.json
uv run python experiments/01_build_lp_codes.py \
  --out artifacts/accuracy-campaign/code
uv run python experiments/02_validate_lp_codes.py \
  --code artifacts/accuracy-campaign/code

uv run python experiments/13_pilot_noise_grid.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --out artifacts/accuracy-campaign/pilot

uv run python experiments/14_generate_campaign_shards.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --selection artifacts/accuracy-campaign/pilot/selection.json \
  --role train \
  --out artifacts/accuracy-campaign/train

uv run python experiments/14_generate_campaign_shards.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --selection artifacts/accuracy-campaign/pilot/selection.json \
  --role calibration \
  --out artifacts/accuracy-campaign/calibration

uv run python experiments/15_train_conditional_fno.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --train artifacts/accuracy-campaign/train \
  --out artifacts/accuracy-campaign/model

uv run python experiments/16_calibrate_hybrid_priors.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --calibration artifacts/accuracy-campaign/calibration \
  --model artifacts/accuracy-campaign/model \
  --campaign-mode canonical \
  --out artifacts/accuracy-campaign/calibration

uv run python experiments/14_generate_campaign_shards.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --selection artifacts/accuracy-campaign/pilot/selection.json \
  --role test \
  --out artifacts/accuracy-campaign/test

uv run python experiments/17_evaluate_hybrid_decoders.py \
  --config configs/accuracy_campaign.json \
  --code artifacts/accuracy-campaign/code \
  --selection artifacts/accuracy-campaign/pilot/selection.json \
  --test artifacts/accuracy-campaign/test \
  --model artifacts/accuracy-campaign/model \
  --calibration artifacts/accuracy-campaign/calibration \
  --campaign-mode canonical \
  --out artifacts/accuracy-campaign/evaluation
```

The evaluator checkpoints immutable per-batch outcomes and can continue with
`--resume`; `--max-batches-this-run` provides a bounded execution slice. Review
[Reproducibility](docs/reproducibility.md) before starting a long run.

## Methods and success criteria

The campaign keeps the code, noise model, data selection, BP-LSD configuration,
and split provenance fixed while changing how the decoder receives its prior:

| Method | Prior or proposal | Syndrome enforcement |
| --- | --- | --- |
| Uniform BP-LSD | One physical Z-error rate for every qubit | BP-LSD |
| Soft-prior BP-LSD | Calibrated per-qubit FNO probabilities | BP-LSD |
| Proposal + residual BP-LSD | Thresholded FNO correction; uncertainty prior on the residual problem | BP-LSD repair |

The accuracy hierarchy used by learned smoke evaluation, hybrid calibration, and
held-out paired evaluation is:

1. **Syndrome validity:** does `Hx @ correction mod 2` equal the observed syndrome?
2. **Logical block-error rate:** does any predicted logical observable differ from
   the sampled one? Learned smoke evaluation and hybrid calibration include
   syndrome-invalid outputs among block failures; held-out evaluation applies
   that rule to every method.
3. **Uncertainty:** report error counts and a 95% Wilson interval where implemented.
4. **Paired inference:** exact discordant-pair statistics for held-out comparisons;
   adaptive or incomplete evaluations use them only as diagnostics.
5. **Diagnostics:** convergence, teacher-bit accuracy, negative log-likelihood,
   correction weights, and related intermediate measures.
6. **Timing:** report the measured batch and decoder components only after the
   accuracy comparison is valid; do not generalize one machine's timing.

The pilot is a preselection stage, not the final comparison. It reports syndrome
validity and counts either an invalid correction or an observable mismatch as a
block failure. Pilot rows should still be used only to select a noise range, not
as final comparative evidence.

See [Experiment methodology](docs/experiment-methodology.md) for the exact model,
decoder settings, data roles, and calibration rule.

## Scientific boundary

This repository studies one CSS sector of `lp(3,7)_16` under independent Z errors
and perfect syndrome measurement. It is a code-capacity study, not a reproduction
of the source paper's circuit-level neutral-atom noise, repeated measurement
rounds, leakage, surgery gadgets, or five-candidate decoder ensemble.

The suffix `16` is the source paper's **distance upper bound**. The repository
does not claim that the code's exact distance is 16. It also does not claim that
matching a BP-LSD teacher is equivalent to decoding successfully, that the FNO
generalizes outside the selected noise range, or that either hybrid wins before a
complete predeclared fixed-shot held-out campaign is run and its artifacts are
reported. An inconclusive result is neither equivalence nor noninferiority.

The Google Quantum AI Willow dataset is source-locked for future work but is not
mixed into this qLDPC experiment. It is surface-code hardware data and belongs in
a separate temporal-drift/generalization study with its own adapter, split policy,
and provenance chain.

## Repository map

| Path | Purpose |
| --- | --- |
| `configs/` | Pinned smoke and accuracy-campaign policies |
| `experiments/` | Numbered, single-stage command-line entry points |
| `scripts/run_smoke.sh` | End-to-end smoke orchestrator |
| `scripts/run_accuracy_campaign.sh` | Resumable local campaign runner |
| `scripts/launch_cloud_campaign.sh` | Dry-run-first Cloud Run launcher and verified resume |
| `scripts/benchmark_calibration_candidate.py` | Reproduction harness for the pre-policy capacity benchmark |
| `Dockerfile`, `.dockerignore` | Exact, default-deny cloud build context |
| `src/qldpc_fno/codes/` | Lifted-product construction and GF(2) operations |
| `src/qldpc_fno/stim/` | Detector-error-model and packed-sample utilities |
| `src/qldpc_fno/decoders/` | Uniform and hybrid BP-LSD implementations |
| `src/qldpc_fno/models/` | One-dimensional ring FNO |
| `src/qldpc_fno/campaign/` | Configuration, seeds, shards, and provenance checks |
| `src/qldpc_fno/training/` | Smoke and conditional training/calibration logic |
| `src/qldpc_fno/metrics/` | Syndrome and logical block-level scoring |
| `tests/` | Unit and integration tests, including corruption/restart checks |
| `docs/` | Concepts, methodology, and reproducibility guides |

## Artifacts, provenance, and restart behavior

Every substantial array or binary artifact is accompanied by canonical JSON
metadata and SHA-256 hashes. Smoke stages verify their content and source hashes,
but their manifests do not record a Git commit. Campaign shard provenance binds
payload hashes to the config and code, role, and rate/seed coordinates; the role
completion manifest hashes every shard manifest. Shard publication does not record
Git identity.

Smoke stages are immutable and have no resume mode: choose a new output path.
Campaign training introduces Git-commit binding in addition to config, code, and
train-shard hashes. It generates BP-LSD teacher corrections in verified chunks,
writes atomic epoch checkpoints, and resumes only when `--resume` is explicit and
that training identity still matches. See
[Reproducibility](docs/reproducibility.md) for safe restart commands and the
artifact contract.

## Roadmap

- Run and report the canonical held-out campaign for uniform BP-LSD, soft-prior
  BP-LSD, and proposal + residual BP-LSD on identical test shards.
- Tune the baseline, train at least three independent seeds, and run a larger
  confirmatory test before any positive accuracy claim.
- Stress-test any candidate that survives the disconfirming run across additional
  noise ranges and code sizes before treating transfer as established.
- Report comparable timing components only for accuracy-eligible methods.
- Treat Willow/temporal work as a separate study: add a hardware-data adapter,
  temporal splits, drift tests, and independent provenance without combining it
  with the qLDPC code-capacity campaign.

## Sources

The smoke orchestrator and optional manual `experiments/00_lock_sources.py` step
write these exact references to `source-lock.json`:

- [Primary qLDPC paper, arXiv:2603.28627v1](https://arxiv.org/abs/2603.28627v1)
- [Stim 1.16.0](https://github.com/quantumlib/Stim/tree/v1.16.0)
- [`ldpc` 2.4.1](https://pypi.org/project/ldpc/2.4.1/)
- [Google Quantum AI Willow dataset, Zenodo 10.5281/zenodo.13273331](https://zenodo.org/records/13273331)

When reporting results, cite the underlying paper and software or dataset sources
appropriate to the experiment. Retain a generated source lock when that optional
manual/smoke step was used; runner campaigns instead bind the exact clean Git
commit, effective configuration, code, stage manifests, and artifact hashes.
