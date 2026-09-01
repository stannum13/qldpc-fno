# Concepts

This guide builds the minimum vocabulary needed to read the experiment. The
[Research background](background.md) explains why the pieces are combined, the
[README](../README.md) gives the project overview, and
[Experiment methodology](experiment-methodology.md) defines the implemented study
precisely.

## From physical errors to a decoding problem

A logical qubit is encoded across many physical qubits. The redundancy lets a
quantum error-correcting code reveal information about errors without directly
measuring the protected logical state.

A **stabilizer** is a parity-like constraint that valid code states satisfy.
Measuring a family of checks produces a binary **syndrome**. A `1` marks a violated
check. The syndrome identifies constraints on the error, but usually not one
unique error pattern.

For the Z-error sector in this repository, let:

- `e` be the physical Z-error bit vector;
- `Hx` be the sparse X-check matrix; and
- `s = Hx @ e mod 2` be the measured syndrome.

A decoder returns a correction `c`. It is **syndrome-valid** only if
`Hx @ c mod 2 = s`. This is necessary but not sufficient: `e` and `c` may differ
by an operation that changes encoded information. The logical observable basis is
used to decide whether that difference is a logical block failure.

This distinction explains why bitwise imitation is weak evidence. Quantum codes
are degenerate: multiple corrections can be equally successful. Conversely, a
prediction can match almost every bit of a teacher correction while its few wrong
bits violate the syndrome.

## What makes a code qLDPC?

**LDPC** means low-density parity-check. The check matrices are sparse, so each
check touches relatively few qubits and each qubit participates in relatively few
checks. A **qLDPC** code applies this idea to quantum stabilizer codes.

The repository constructs the lifted-product CSS code identified as
`lp(3,7)_16` by its source. Its validated metadata is:

- 2,610 physical qubits (`n`);
- 744 logical qubits (`k`);
- 945 rows in each of `Hx` and `Hz`;
- check-row weight 10; and
- cyclic lift length `ell = 45`.

The `16` label is a published distance upper bound, not an exact-distance claim.

## Code capacity

A **code-capacity** experiment assumes data qubits experience errors but syndrome
measurements are perfect. Here, every physical qubit independently receives a Z
error with probability `p`. Stim represents the exact mapping from each possible
Z error to detectors and logical observables as a detector error model (DEM).

This controlled setting isolates decoder behavior. It does not model measurement
faults, repeated rounds, gates, leakage, or a complete hardware circuit.

## BP-LSD

**Belief propagation (BP)** passes probabilistic messages along the sparse check
graph. Short cycles and degeneracy can prevent BP alone from finding an adequate
solution. **Localized statistics decoding (LSD)** supplies a fallback around the
unresolved region.

The repository uses the pinned `ldpc` implementation with minimum-sum BP, serial
scheduling, at most 100 iterations, and order-5 `LSD_E`. In the baseline, every
qubit receives the same prior error probability `p`.

## Fourier neural operator

A Fourier neural operator learns interactions in frequency space. The ring FNO in
this repository treats the lifted-product code's length-45 cyclic coordinate as a
periodic spatial axis. Spectral convolutions mix the first 12 Fourier modes, while
pointwise convolutions mix channels at each ring position.

The smoke model maps 21 syndrome channels to 58 correction channels. The campaign
adds one constant channel containing the physical error rate's log-odds, allowing
one model to condition on several noise levels.

An unconstrained FNO does not automatically satisfy `Hx @ c = s`. This is why the
campaign does not use thresholded FNO output as a standalone decoder.

## The three decoder paths

### Uniform-prior BP-LSD

BP-LSD receives the same physical error rate for every qubit. This is the
conventional baseline and the teacher used to generate training corrections.

### Soft-prior BP-LSD

The FNO produces one logit per correction bit. Calibration turns logits, together
with the physical noise rate, into bounded per-qubit probabilities. BP-LSD uses
those probabilities as its channel prior and remains responsible for decoding the
original syndrome.

### Hard proposal plus residual BP-LSD

The calibrated probabilities are thresholded at `0.5` to form a proposal. The
proposal's syndrome is compared with the measured syndrome, leaving a residual.
BP-LSD decodes that residual using proposal uncertainty as its prior, and its
delta is XORed with the proposal. The result is checked against the original
syndrome.

The two hybrid paths ask different questions: whether the model is better as a
soft belief about each qubit, or as a concrete starting point that a conventional
decoder repairs.

## Reading the metrics

- **Teacher-bit accuracy** measures imitation of one BP-LSD correction. It is a
  useful optimization diagnostic, not a decoder-level outcome.
- **Syndrome-valid rate** measures whether proposed corrections satisfy the
  measured parity constraints.
- **Logical block-error rate** counts a shot as failed when any logical observable
  is wrong. Learned standalone evaluation and hybrid calibration also count every
  syndrome-invalid shot as failed. The current pilot preselection score reports
  syndrome validity separately from its observable-mismatch block errors; the
  planned final comparison will use one invalid-as-failure rule for every method.
- **Exact observable match rate** ignores how many logical bits differ and asks
  whether all of them match for each shot.
- **Negative log-likelihood (NLL)** evaluates probability calibration against
  sampled physical errors; it is used only after invalidity and block errors in
  the calibration selection rule.
- **Latency** is an environment-dependent measurement. It is meaningful only with
  the batch size, hardware, software versions, and timed components recorded.

The detailed priority and data-splitting rules are in
[Experiment methodology](experiment-methodology.md).
