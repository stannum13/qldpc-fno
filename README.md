# qLDPC-FNO experiment loop

This repository is a small, reproducible experiment asking whether a Fourier neural
operator can imitate a BP-LSD decoder on the cyclic coordinates of the
`lp(3,7)_16` quantum LDPC code. It reconstructs the published code, builds an exact
Stim detector error model, samples replayable data, generates BP-LSD teacher
corrections, trains a tiny 1D FNO, and evaluates it on a held-out split.

## Run it

Install [`uv`](https://docs.astral.sh/uv/), then create the pinned environment and
run the checks:

```bash
uv sync --all-groups
uv run pytest -q
uv run ruff check .
```

Run the canonical 512-shot, 600-step experiment:

```bash
bash scripts/run_smoke.sh
```

The orchestrator refuses to overwrite an existing output. Choose a fresh location
for another run, or use reduced settings for a quick execution check:

```bash
SMOKE_OUTPUT=artifacts/quick \
SMOKE_SHOTS=16 \
SMOKE_STEPS=2 \
bash scripts/run_smoke.sh
```

The canonical run enforces all overfit gates and stops before held-out evaluation
if any gate fails. Supplying `SMOKE_STEPS` explicitly selects a non-gating execution
check so a tiny CI run can still exercise every CLI; its metrics are not scientific
decoder results.

## Artifact contract

The output directory contains immutable stages:

- `source-lock.json`: primary paper, software, and Willow dataset references
- `code/`: sparse `Hx`/`Hz`, paper metadata, and algebraic validation
- `dem/`: exact independent-Z Stim detector error model and logical-X basis
- `samples/`: deterministic packed `b8` errors, detections, and observables
- `bplsd/`: teacher corrections, convergence, latency, and logical metrics
- `tensors/`: exact `(shots, channels, ell)` ring fields and contiguous split
- `fno/`: frozen PyTorch state dictionary and training gates
- `evaluation/metrics.json`: held-out bit accuracy, syndrome validity, logical
  block-error rate, Wilson interval, and batched CPU inference time

Binary and array artifacts are hashed in adjacent canonical JSON manifests. The
next stage verifies those hashes before consumption and rejects corrupted or
cross-run inputs. The default split is order-preserving: the first 75% of shots
train and the last 25% are held out.

## Scientific boundary

This smoke experiment uses independent physical Z errors with perfect syndrome
measurement. It is a code-capacity experiment, not a reproduction of the paper's
circuit-level neutral-atom noise, repeated rounds, leakage, surgery gadgets, or
five-candidate decoder ensemble. The suffix `16` in `lp(3,7)_16` is the paper's
distance **upper bound**; this repository does not claim an exact distance of 16.

The Google Quantum AI Willow release (Zenodo
[`10.5281/zenodo.13273331`](https://doi.org/10.5281/zenodo.13273331)) is source-locked
but is not mixed into this qLDPC smoke shard. It is surface-code hardware data and
belongs in a later temporal-drift/generalization experiment with a separate data
adapter and provenance manifest.

Most importantly, teacher-bit accuracy is not treated as decoder success. Every
learned correction must reproduce the observed syndrome, and any invalid
correction counts as a block failure even if its predicted logical bits happen to
match.
