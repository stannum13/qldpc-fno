# qLDPC-FNO Experiment Loop Design

Date: 2026-08-29

## 1. Purpose

Build a repeatable, Codex-friendly research loop for testing high-impact ML-for-science ideas with small, falsifiable experiments. The first instance tests whether a spectral neural decoder can transfer across related quantum LDPC code sizes.

The loop must produce evidence, not a demo. Each experiment is one script with declared inputs, immutable outputs, a statistical check, and a decision gate that selects the next experiment.

The first implementation is deliberately split into three levels:

1. exact code-capacity experiments using Stim detector error models;
2. circuit-level qLDPC experiments using custom Stim circuits;
3. real-device validation using Google Quantum AI's Willow surface-code data.

Only level 1 is required for the first complete loop. Levels 2 and 3 start after level 1 produces a technically sound result.

The level-1 vertical slice decodes the Z-error sector of the CSS code. A Bernoulli Z error is detected by `Hx` and may flip logical X observables, matching the X-basis memory experiment described in the paper. Mirroring the result for X errors is useful confirmation but is not required for the first loop.

## 2. Central research question

Can a decoder trained on a small quasi-cyclic lifted-product code produce valid, logically successful corrections on a larger code without retraining, and does a Fourier representation improve transfer over non-spectral controls?

The result is positive only if the model:

- produces syndrome-valid corrections;
- improves on matched non-spectral controls;
- remains competitive with a classical reference decoder under the same samples;
- transfers under a predeclared evaluation protocol rather than a post-hoc favorable split.

A negative result is still complete if it identifies whether failure comes from representation, code-family mismatch, optimization, or decoder accuracy.

## 3. Critical correction to the original hypothesis

The paper's three `lp(3,7)` memory codes use the same seed dimensions but are not simple refinements of a single fixed operator. Their independently optimized seed matrices use ring orders 45, 75, and 91. FNO-style zero-shot super-resolution is therefore not automatically justified: the underlying discrete operators change with code size.

The experiment must distinguish three claims:

1. **Controlled discretization transfer:** transfer across a synthetic family generated from one shared normalized seed operator.
2. **Naive paper-family transfer:** transfer across the independently optimized paper instances without operator conditioning.
3. **Operator-conditioned transfer:** transfer when the model receives the target seed or Tanner operator as an input.

This distinction prevents accidental cross-code generalization from being mislabeled as discretization invariance.

The suffixes 16, 20, and 24 in the paper's code names are numerical distance upper bounds, not established exact distances. Reports must use `d <= 16`, `d <= 20`, and `d <= 24`.

## 4. Verified source facts

The code definitions come from [arXiv:2603.28627v1, Appendix A](https://arxiv.org/html/2603.28627v1#A1). The three memory instances are:

| Name | Ring order | Reported parameters | Check weight |
|---|---:|---:|---:|
| `lp_3_7_16` | 45 | `[[2610, 744, <=16]]` | 10 |
| `lp_3_7_20` | 75 | `[[4350, 1224, <=20]]` | 10 |
| `lp_3_7_24` | 91 | `[[5278, 1480, <=24]]` | 10 |

The paper supplies seed matrices, but no executable circuits, parity-check files, logical bases, simulation scripts, or pinned dependency versions. Static parity checks must therefore be reconstructed from the cited lifted-product convention and validated independently.

The paper's circuit-level simulations use Stim with an edge-colored CNOT schedule and a custom five-member BP-LSD ensemble. That exact reproduction is a separate level because Stim does not natively generate lifted-product circuits.

Stim v1.16.0 is the initial pinned target. Exact replay uses stored `dets.b8`, `obs_actual.b8`, and sidecar manifests; Sinter is reserved for aggregate Monte Carlo because its normal collection path is not shot-seeded.

The real-device source is [Google Quantum AI's Willow dataset](https://zenodo.org/records/13273331), licensed CC BY 4.0. Its 105-qubit archive is approximately 5.7 GB and contains distance-3, distance-5, and distance-7 XZZX surface-code cases with:

- ideal and SI1000 Stim circuits;
- raw measurements and sweep bits;
- derived detector events and actual logical flips;
- metadata with distance, rounds, shots, and qubit coordinates;
- official decoder predictions and detector error models.

The Willow data is not qLDPC data. It tests real-hardware and distance-transfer behavior, not the lifted-product claim.

## 5. Experiment unit contract

Every numbered experiment is an independently runnable Python module under `experiments/`. It receives only configuration and artifact paths and writes to a new run directory. It never silently mutates an upstream artifact.

Each run directory contains:

- `manifest.json`: command, arguments, Git commit, package versions, platform, seed derivation, input hashes, and output hashes;
- `metrics.json`: scalar results and uncertainty intervals;
- `checks.json`: invariants and pass/fail status;
- small diagnostic plots when the experiment is analytical;
- raw replay artifacts when they are required to reproduce a decoder comparison.

Stim bit files always have an adjacent manifest declaring shot count, bits per shot, packing format, bit order, and whether logical bits are actual or predicted.

Random data is split by immutable shard identifier before any model sees it. Training, validation, and test shards are never re-shuffled between models.

## 6. Level 1: exact code-capacity loop

Level 1 is the first publishable experimental unit. It uses Stim as an exact detector-error-model sampler, not as a substitute for a missing qLDPC circuit generator.

### `00_lock_sources.py`

**Question:** What exact external objects define this experiment?

**Inputs:** source registry containing paper version, Stim version, `ldpc` version, and Willow DOI.

**Outputs:** `source_lock.json` with URLs, versions, checksums, retrieval date, and licenses.

**Gate:** fail if a source is unversioned or a checksum is missing for a downloaded artifact.

### `01_build_lp_codes.py`

**Question:** Can the paper's seed matrices be reconstructed into canonical binary CSS checks?

**Inputs:** checked-in seed specifications for ring orders 45, 75, and 91.

**Outputs:** sparse `hx.npz`, `hz.npz`, coordinate maps, and `code.json` for each instance.

**Gate:** shapes must match 945x2610, 1575x4350, and 1911x5278 for each X and Z check matrix.

### `02_validate_lp_codes.py`

**Question:** Are the reconstructed objects internally consistent with the paper?

**Inputs:** level-1 code artifacts.

**Outputs:** matrix ranks, inferred `n` and `k`, row/column weights, CSS commutation result, and cyclic-equivariance checks.

**Gate:** require `Hx @ Hz.T == 0 mod 2`, the reported `n` and `k`, check weight 10, exact equivariance under one ring shift, and a canonical logical basis with the required stabilizer commutation and logical anticommutation relations. Do not proceed on a convention mismatch.

### `03_audit_family_spectra.py`

**Question:** Are the three paper instances plausibly discretizations of one shared spectral operator?

**Inputs:** the three seed operators and coordinate maps.

**Outputs:** normalized exponent plots, singular spectra by Fourier mode, cross-size spectral alignment scores, and permutation/shift-alignment diagnostics.

**Gate:** predeclare an alignment threshold from controlled same-operator pairs. If paper-family alignment is below the threshold, naive transfer remains a falsification control rather than the main claimed mechanism.

### `04_build_controlled_family.py`

**Question:** What does genuine discretization transfer look like when the operator is actually shared?

**Inputs:** one normalized 3x7 monomial seed template and ring orders 45, 75, and 91.

**Outputs:** a controlled CSS family with the same artifact schema as the paper family.

**Gate:** require CSS validity and ring-shift equivariance at every size. Distance is characterized but is not required to match the paper.

### `05_build_code_capacity_dem.py`

**Question:** Can each static code be expressed as an exact replayable decoding distribution?

**Inputs:** one CSS code, an independent Bernoulli Z-error channel, logical basis, and physical error rate.

**Outputs:** a Stim `.dem` in which each physical error mechanism toggles its syndrome detectors and logical observables, plus a DEM manifest.

**Gate:** compare DEM symptoms against direct GF(2) multiplication on an exhaustive tiny-code fixture and randomized errors on each real code.

### `06_sample_code_capacity.py`

**Question:** Can we generate deterministic training and test shards with exact labels?

**Inputs:** `.dem`, fixed chunk seed, shots, and one fixed sampling call per chunk.

**Outputs:** `dets.b8`, `obs_actual.b8`, optional `errors.b8` for debugging, and `samples.json`.

**Gate:** independently recompute syndromes and logical flips from sampled error mechanisms on a validation subset. Store raw files because Stim seeds are only partially reproducible across versions and platforms.

Initial probabilities are `0.002`, `0.005`, `0.01`, and `0.02`. Smoke shards use 10,000 shots; pilot shards use 100,000. Larger runs are selected by observed failure counts, not assumed in advance.

### `07_decode_bplsd.py`

**Question:** What correction quality is achievable with the classical teacher/reference?

**Inputs:** parity checks, detector shards, channel probabilities, and an explicit BP-LSD configuration.

**Outputs:** teacher correction vectors, predicted logical flips, latency, convergence status, and syndrome-validity flags.

**Gate:** every accepted correction must reproduce the observed syndrome. The initial code-capacity reference may use one pinned BP-LSD configuration; the paper's five-member circuit-level ensemble is reserved for level 2.

### `08_tensorize_ring_fields.py`

**Question:** Can syndromes and corrections be represented with size-independent channel semantics?

**Inputs:** code coordinates, detector bits, and teacher corrections.

**Outputs:** memory-mapped tensors shaped as fixed semantic channels by ring coordinate, with masks and inverse mappings.

**Gate:** tensor-to-bit round trips must be exact, and a cyclic shift in the code must become a cyclic shift in the tensor.

For every 3x7 instance, `Hx` has `21 * ell` syndrome bits and the code has `58 * ell` physical qubits. The neural interface is therefore a 21-channel syndrome field over the ring mapped to a 58-channel Z-correction field over the same ring. Logical predictions are computed from the proposed correction and the target code's logical basis; they are not a size-dependent neural output. This is what makes the source and target interfaces compatible even though `k` changes.

### `09_probe_fourier_structure.py`

**Question:** Is there predictive information in shared low Fourier modes before fitting a neural network?

**Inputs:** training tensors from all declared families.

**Outputs:** mode-energy curves, syndrome-correction coherence, low-mode linear-probe accuracy, and cross-size stability plots.

**Gate:** if low modes are neither predictive nor stable, the next neural experiment still runs as a negative control, but no Fourier advantage is claimed.

### `10_overfit_tiny_models.py`

**Question:** Are the loss, correction mapping, and logical scorer implemented correctly?

**Inputs:** 128 examples from the smallest controlled code.

**Outputs:** learning curves for a tiny FNO, circular CNN, and linear spectral baseline.

**Gate:** each model must overfit the teacher correction targets and achieve near-perfect syndrome validity on the 128 examples. Failure is an implementation bug, not a research result.

### `11_train_decoders.py`

**Question:** What does each representation learn under equal budgets?

**Inputs:** fixed train/validation shards and model configs.

**Outputs:** checkpoints for FNO, parameter-matched circular CNN, linear spectral model, and a size-specific MLP sanity control.

**Gate:** equalize train examples, optimizer budget, early-stopping rule, and approximate parameter count. Select checkpoints only on validation data.

The supervised target is the BP-LSD canonical correction, not the sampled physical error. This avoids treating one arbitrary representative of a degenerate quantum error class as the unique correct label.

### `12_evaluate_in_size.py`

**Question:** Can the learned model decode the size on which it was trained?

**Inputs:** frozen checkpoints and held-out smallest-code shards.

**Outputs:** syndrome-validity rate, block logical error rate, excess failures relative to BP-LSD, calibration, and latency distribution.

**Gate:** do not interpret transfer if the FNO cannot learn the source-size task or if its syndrome-validity rate is materially below the circular CNN.

### `13_transfer_controlled_family.py`

**Question:** Does the FNO transfer when the larger codes genuinely discretize a shared operator?

**Inputs:** the smallest-size checkpoint and held-out controlled-family shards at ring orders 75 and 91.

**Outputs:** zero-shot metrics, in-size retrained oracle metrics, and transfer degradation ratios.

**Gate:** FNO transfer must beat the matched circular CNN under confidence intervals to support a Fourier-specific transfer claim.

### `14_transfer_paper_family.py`

**Question:** Does naive transfer survive the independently optimized paper seeds?

**Inputs:** the ring-order-45 checkpoint and paper-family shards at 75 and 91.

**Outputs:** the same metrics as experiment 13, with spectral-alignment diagnostics joined by run ID.

**Gate:** report this as cross-code transfer, not super-resolution. A failure is expected evidence if experiment 03 showed operator mismatch.

### `15_train_operator_conditioned.py`

**Question:** Can explicit seed/Tanner conditioning recover transfer across different operators?

**Inputs:** controlled and paper-family training shards plus target operator descriptors.

**Outputs:** conditioned-FNO checkpoint and zero-shot metrics on held-out operator/size combinations.

**Gate:** compare against an equally conditioned non-spectral model. Improvement must come from more than simply revealing the target graph.

### `16_decide_level1.py`

**Question:** What did the first loop establish, and what single next experiment is justified?

**Inputs:** immutable metrics from experiments 02 through 15.

**Outputs:** `decision.md` and `decision.json` containing one of:

- `spectral_transfer_supported`;
- `conditioning_required`;
- `representation_not_predictive`;
- `source_decoder_not_learned`;
- `inconclusive_more_samples_required`.

**Gate:** exactly one next branch is selected. HiPPO, attention, information-bottleneck regularization, and FPGA emulation cannot be selected without evidence matching their intended failure mode.

## 7. Level 2: circuit-level Stim loop

Level 2 reproduces the paper's physical setting after level 1 validates the representation and correction pipeline.

### `20_build_memory_circuit.py`

Construct X- and Z-check ancillas, edge-color Tanner-graph CNOT schedules, repeated rounds, boundary detectors, logical observables, and the paper's explicit circuit-level depolarizing channels.

### `21_validate_memory_circuit.py`

Require noiseless detector determinism, deterministic logical observables, expected detector counts, no qubit collision within a CNOT layer, and agreement with direct stabilizer propagation on tiny fixtures.

### `22_derive_explicit_dem.py`

Create the decoder DEM with every decomposition and approximation flag recorded. Automatic Sinter DEM derivation is not allowed in controlled runs.

### `23_sample_circuit.py`

Write replayable detector and actual-observable files with fixed chunks. Circuit sampling and decoder configuration remain separate artifacts.

### `24_reproduce_paper_decoder.py`

Implement and verify the paper's five BP-LSD candidates: nominal, randomized schedule, 0.8p, 1.2p, and independently perturbed priors, selecting the syndrome-valid minimum weighted-cost candidate.

### `25_train_temporal_decoder.py`

Extend the level-1 model to space-time detector tensors. HiPPO is introduced only here, after the spatial representation is validated.

### `26_collect_sinter.py`

Wrap frozen decoders as `sinter.Decoder` implementations and collect aggregate logical-error curves with explicit circuit/DEM pairs and append-only CSV output.

### `27_decide_level2.py`

Select between temporal modeling, quantization/FPGA work, or decoder-quality work based on measured accuracy and latency.

## 8. Level 3: Willow real-device loop

The Willow loop is an independent reality check. Its single logical observable avoids the changing-logical-basis problem of high-rate qLDPC codes.

### `30_index_willow.py`

Read Zenodo metadata and the remote ZIP central directory without downloading the full archive. Record archive size, MD5, license, case paths, and compressed entry ranges.

### `31_fetch_willow_cases.py`

Range-fetch only predeclared cases sharing basis and round count across distance 3, 5, and 7. Start with the 5.7 GB 105-qubit archive; do not fetch the 65 GB repetition-code archive in this loop.

### `32_validate_willow_conversion.py`

Use the supplied ideal Stim circuit and sweep bits to recompute detector events and actual observable flips from raw measurements, then require byte-for-byte equality with the released derived files.

### `33_score_willow_references.py`

Score the released Harmony, correlated-matching, and Libra prediction files against `obs_flips_actual.b8`. This validates parsing and establishes real-hardware reference error rates without reimplementing proprietary decoder details.

### `34_tensorize_willow.py`

Map detector coordinates from the ideal circuit into a space-time grid with check-type channels and masks. Round trips back to detector ordering must be exact.

### `35_transfer_willow_distance.py`

Train on distance 3 and evaluate zero-shot on matched distance-5 and distance-7 cases. Compare FNO with a parameter-matched CNN and an in-distance retrained oracle.

### `36_test_sim_to_real.py`

Train on samples from the released SI1000 Stim circuit or supplied DEM, evaluate on device data, and measure the value of limited real-data fine-tuning.

### `37_decide_willow.py`

Classify the outcome as distance transfer, simulation-to-real domain gap, hardware-correlation failure, or insufficient data alignment.

The reported 369 +/- 6 microsecond transient belongs to the 72-qubit repetition-code study. It is reserved for a later HiPPO-specific loop because the relevant archive is approximately 65 GB and answers a different temporal question.

## 9. Metrics and statistical policy

Primary metrics:

- block logical error rate: any logical observable predicted incorrectly;
- syndrome-validity rate of proposed corrections;
- excess logical failures relative to the BP-LSD reference on paired shots;
- zero-shot transfer degradation relative to an in-size retrained oracle;
- decoding latency per shot and throughput per batch.

Every reported proportion includes a 95% interval. Paired decoders are compared on identical test shots using paired error indicators. Runs continue until either the predeclared error-count target is met or the shot cap is reached; a zero-failure run is reported as an upper bound, never as zero error.

Model comparisons use at least three training seeds after the smoke stage. Hyperparameters are selected on source validation data only. Target-size labels are unavailable during zero-shot selection.

## 10. Data and compute policy

Local execution is the default for experiments 00 through 12. Google Cloud is used only when a measured local bottleneck justifies it or when persistent artifact storage is needed.

Cloud use must be explicit in a run manifest. No billable resource is created implicitly. Large artifacts use content-addressed paths and are never committed to Git.

Expected first-loop storage is dominated by packed bit shards and teacher corrections. Data is streamed in chunks; dense unpacked arrays are not persisted unless a profile proves them small enough.

## 11. Initial repository boundaries

Planned source boundaries:

```text
configs/                 declarative code, noise, model, and run configs
experiments/             numbered atomic experiment entry points
src/qldpc_fno/codes/     lifted-product construction and GF(2) validation
src/qldpc_fno/stim/      DEM, circuit, packed-bit, and Sinter adapters
src/qldpc_fno/decoders/  BP-LSD and learned-decoder interfaces
src/qldpc_fno/models/    FNO and matched controls
src/qldpc_fno/metrics/   logical scoring, intervals, paired comparisons
src/qldpc_fno/data/      immutable manifests, shards, and tensor mappings
tests/                   tiny exact fixtures and contract tests
artifacts/               ignored generated outputs
```

Core units communicate through documented artifacts rather than importing experiment scripts from one another.

## 12. Completion definition for the first loop

The first loop is complete when experiments 00 through 16 run from a clean environment and produce:

1. validated paper and controlled-family parity checks;
2. replayable Stim code-capacity shards;
3. a verified BP-LSD reference;
4. source-size results for FNO and matched controls;
5. controlled-family and paper-family zero-shot results;
6. one evidence-backed decision selecting or rejecting operator conditioning as the next step.

Circuit-level reproduction, Willow, HiPPO, attention, information bottlenecks, symbolic extraction, and FPGA emulation are explicitly outside this first completion boundary.
