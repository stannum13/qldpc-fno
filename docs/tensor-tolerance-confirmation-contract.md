# Two-view tolerance confirmation contract

The development sweep found that local-spectrum truncation can preserve the
winning logical class even when it distorts posterior mass ratios. This contract
tests only that decision-specific property. It does not test whether the output
is a calibrated posterior.

The frozen policy is
[`configs/tensor_tolerance_consensus_policy.json`](../configs/tensor_tolerance_consensus_policy.json).
For every planar-code syndrome at distance 3 or 5, it runs column and row MPS
contractions at fixed tolerance `0.01`. It accepts the column result only if
both views return valid positive coset masses and select the same logical class.
Otherwise it runs a column contraction at fixed bond dimension `chi=8` and
returns that result. The fixed comparator runs column `chi=8` on every case.

The untouched data contract is
[`configs/tensor_tolerance_confirmation_data.json`](../configs/tensor_tolerance_confirmation_data.json).
It defines 160 independent depolarizing-channel draws in each of six strata:
distance 3 or 5 crossed with `p=0.05`, `0.10`, or `0.15`. The seed domain is
`qldpc-fno/tensor-tolerance-consensus-confirmation/v1`. Each original syndrome
is paired with its transpose for an exact symmetry check; the transpose is not
another independent statistical unit. The data generator verifies unrestricted
row/column references and exhaustively enumerates stabilizer cosets at distance
3.

The primary safety endpoint is whether the selected class matches that exact
reference. Policy and fixed comparator are each evaluated in all six strata.
Familywise alpha `0.05` is split over these 12 predeclared comparisons, and
each one-sided Wilson upper bound must be at most `0.05`. At 160 draws per
stratum, zero observed failures passes; one observed failure does not. A weak
fixed comparator cannot establish a passing policy gate.

The work endpoint is the same deterministic arithmetic estimate used in the
earlier tensor studies: NumPy einsum-path FLOPs plus stated dense QR/SVD
estimates. The policy is charged for both tolerance views on every case and for
the full `chi=8` fallback when triggered. A paired, within-stratum bootstrap
with 10,000 replicates and frozen seed computes a one-sided 95% lower confidence
bound on equal-stratum mixture savings relative to fixed `chi=8`; that bound
must be at least `0.05`. Host wall time, FPGA cycles, throughput, and backlog
are outside this gate.

Canonical status additionally requires the exact policy and data config,
deterministic seed/syndrome replay, complete original/transpose pairs, action
metadata, reconcilable primitive and total work counters, matching generation
source hashes, independent replay of reference and action probabilities/work,
no invalid references, and zero transpose class/work mismatches.
The policy and evaluator must be committed before any samples from the locked
domain are generated. If a methodological defect is found after sampling, the
artifact is invalidated and a new seed domain is required; this one is never
retuned against its own result.
