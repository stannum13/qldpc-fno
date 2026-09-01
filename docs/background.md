# Research background

This guide explains the chain of ideas behind qLDPC-FNO. It motivates the
experiment; [Concepts](concepts.md) defines its vocabulary and
[Experiment methodology](experiment-methodology.md) states exactly what the code
currently implements.

## From fragile states to classical decoding

Noise can change a physical qubit through interactions with its environment or
imperfect control. Quantum information cannot be protected by periodically
reading and copying an unknown state. A quantum error-correcting code instead
stores a smaller logical state across a larger physical system.

The code measures stabilizer checks: parity-like questions whose answers reveal
whether constraints have changed without revealing the logical state itself. The
binary collection of changed-check outcomes is the **syndrome**. A classical
decoder receives that syndrome and proposes a physical correction.

A small classical analogy helps. Suppose one error bit participates in two parity
checks. Flipping that bit changes both check outcomes, producing syndrome `11`.
On a long block, predicting no error would agree with the true error vector at
almost every position, yet it would produce syndrome `00`. High per-bit agreement
has therefore missed the only event that matters to the checks. Quantum codes add
another complication: several different corrections may share a syndrome, and
some differences between them can change logical information.

That gives decoding two non-negotiable tests:

1. the correction must reproduce the observed syndrome; and
2. after correction, the encoded logical information must be preserved.

Bitwise similarity to one reference correction is useful for training, but it is
not either of those tests.

## Why study qLDPC codes?

Surface codes are the standard reference architecture for fault-tolerant quantum
computing because their checks can be arranged locally. Their low encoding rate,
however, makes the physical-qubit cost of protecting many logical qubits an
important systems concern.

Quantum low-density parity-check codes explore another point in the design space.
Their sparse checks touch a bounded number of qubits, while finite-rate families
can encode more logical information per physical qubit. The tradeoff is that their
connectivity, operations, and decoding can be less local or less mature. A decoder
study on one qLDPC code is therefore evidence about that controlled setting, not a
general resource comparison with surface codes.

This repository reconstructs `lp(3,7)_16` from the
[source-locked qLDPC paper](https://arxiv.org/abs/2603.28627v1). Its validated
instance has 2,610 physical qubits, 744 logical qubits, and sparse X- and Z-check
matrices. The suffix `16` is the paper's distance upper bound; it is not treated as
an exact-distance result here.

## Why the cyclic lift matters

The lifted-product construction starts from small matrices over a cyclic ring and
expands their entries into circulant blocks. For this code, the cyclic coordinate
has length `ell = 45`. The 945 syndrome bits can therefore be viewed as 21 fields
around a length-45 ring, and the 2,610 correction bits as 58 ring fields.

This is more than a convenient reshape. Shifting the cyclic coordinate shifts the
same local pattern throughout the check matrices. A model that applies the same
rule at each ring position can respect that repeated organization instead of
learning every coordinate independently.

## Why BP-LSD is the baseline

Belief propagation (BP) passes probability messages between error variables and
sparse parity checks. It is naturally matched to an LDPC graph and can use a known
physical error rate as its channel prior. Quantum degeneracy and short graph
cycles can still leave BP uncertain or unconverged.

Localized statistics decoding (LSD) supplies a targeted fallback around the
unresolved part of the problem. The combined BP-LSD decoder is a strong reference
because it uses the check matrix directly, and every returned correction can be
checked against the observed syndrome.

That strength has a possible systems cost. Each shot may require iterative message
passing and localized post-processing. The implementation permits up to 100 BP
iterations and uses order-5 `LSD_E`. This does not establish that BP-LSD is too
slow, nor that a learned component is faster. It identifies latency and decoder
work as measurements to make only after accuracy is established. The exact
implementation is the source-locked [`ldpc` 2.4.1](https://pypi.org/project/ldpc/2.4.1/).

## Why a Fourier operator is plausible

A one-dimensional FNO alternates two kinds of mixing:

- pointwise channel mixing combines information at each cyclic position; and
- spectral convolution mixes selected Fourier modes around the whole ring.

The result is translation-equivariant along the cyclic coordinate and can model
long-range patterns without assigning unrelated parameters to all 45 positions.
Those properties fit the lifted-product representation. They do not enforce the
parity equation `Hx @ correction mod 2 = syndrome`.

The campaign therefore gives the FNO a narrower role. It sees 21 syndrome fields
plus one broadcast log-odds noise channel and predicts 58 correction-probability
fields. BP-LSD then uses that information in one of two ways:

- **soft prior:** calibrated FNO probabilities replace the uniform per-qubit
  channel prior; or
- **proposal and repair:** a thresholded FNO proposal is formed, BP-LSD decodes
  its residual syndrome using proposal uncertainty, and the two corrections are
  combined.

Both paths preserve a conventional decoder in the loop and verify the final
syndrome.

## The smoke lesson and the hybrid hypothesis

During repository development, smoke runs showed that a model can agree with most
BP-LSD teacher bits while failing syndrome checks. Those artifacts are not
committed, so this is a design lesson rather than a reproducible numeric result
from a fresh clone.

The pipeline responds in three ways:

1. teacher-bit accuracy remains a diagnostic, not a success criterion;
2. syndrome validity is measured directly; and
3. learned information is evaluated through BP-LSD hybrids rather than assumed to
   be a valid standalone correction.

The research hypothesis is deliberately conditional: a structural prior may help
BP-LSD choose better corrections or do less repair work. The experiment must also
allow the opposite result—that the prior gives no accuracy benefit or adds more
cost than it removes. This is why the held-out comparison ranks validity and
logical block errors before timing.

The implementation includes pilot selection, role-separated sampling, resumable
teacher generation and FNO training, hybrid calibration, and paired held-out
evaluation. Calibration output alone is not evidence that either hybrid beats
uniform BP-LSD, and no canonical held-out result is claimed until a campaign is
run and its artifacts are reported.

## A separate future question: temporal hardware data

The source lock also records the
[Google Quantum AI Willow dataset](https://zenodo.org/records/13273331). That
dataset contains surface-code hardware observations, not samples from the qLDPC
code-capacity model used here. It is not mixed into this campaign.

A future temporal study would need its own data adapter, chronological split,
drift questions, metrics, and provenance chain. It could ask whether a learned
model remains calibrated as hardware behavior changes over time. That is a
separate research direction, not an extension of the present qLDPC result.

For the precise noise model, pilot scoring caveat, calibration rule, and current
implementation boundary, continue to
[Experiment methodology](experiment-methodology.md).
