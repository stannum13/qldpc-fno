# Erratum: physical-error and syndrome pairing in planar tensor studies

The planar tensor-network data generator used ordinary binary matrix multiplication
to obtain a syndrome from a sampled Pauli error. qecsim instead uses a *symplectic*
binary product. The two operations generally disagree. This affects the historical
tensor-network study, its derivative local-spectrum discovery sweep, the
tensor-policy design table, and the two locked consensus confirmations. Their
evidence files and source-locked commits are preserved.

Write a Pauli error as two bit vectors, `e = (x | z)`, and a stabilizer as
`h = (hX | hZ)`. The old generator computed `x·hX + z·hZ` modulo 2. The correct
syndrome is `z·hX + x·hZ` modulo 2. Equivalently, the stored syndrome is the
correct syndrome of the error after exchanging its X and Z halves.

Under the *i.i.d. depolarizing* model used in these studies, exchanging X and Z
preserves the error distribution: X and Z have the same probability. Therefore
the stored syndrome has the intended **marginal distribution**, even though it
is not generally paired with the original seeded physical error. The locked
logical-*class agreement*, fallback counts, paired estimated-work comparisons,
and their independent-syndrome confidence intervals remain interpretable for
that symmetric channel. They compare solver decisions conditional on each
syndrome, not a decoder's correction against the sampled physical error.

The old evidence **does not** establish an end-to-end logical error rate, a
physical-error-to-syndrome replay, or a result under X/Z-asymmetric noise. In
independent seed replay, the old syndrome disagreed with the original physical
error on 687 of 960 tolerance-confirmation draws, 680 of 960 earlier-consensus
draws, and 1,237 of 1,728 design draws. In every replayed draw, it matched the
correct syndrome of the X/Z-swapped error. Any BLER diagnostic using that swapped
error is post-hoc and is not a replacement for fresh confirmation.

Future generation now uses `qecsim.paulitools.bsp(error, code.stabilizers.T)`;
regression tests replay the sampled error from its seed and require the recorded
syndrome to match. The historical locked evaluators reproduce the old source
convention only at their corresponding source commits. Changing source code
changes their source hashes, so rerunning old artifacts against current HEAD is
not a canonical re-verification. Current evaluators also refuse to certify new
data under those revealed v1 seed domains. A shot-level accuracy claim requires a new seed
domain frozen before sampling, a correctly paired sampled error, a comparable
strong decoder baseline, and logical failure counted modulo the stabilizer group.
