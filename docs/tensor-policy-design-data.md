# Tensor policy design data

This artifact is the development table used to define the first adaptive
tensor-contraction rule. It is not confirmation evidence.

The table contains 1,728 independent physical-channel draws across planar-code
distances 3 and 5 and error rates 0.05, 0.10, and 0.15. Each draw is paired with
its transposed syndrome in the same split. Transposes are symmetry augmentation,
not additional independent statistical units. Nine actions are evaluated for
every context: row/column contraction at `chi=1,2,4,8` and averaged row/column
contraction at `chi=4`.

Reference validity is checked three ways:

- unrestricted row and column contractions must select the same class;
- their maximum probability difference must be at most `1e-10`;
- their maximum pairwise log-mass-ratio difference must be at most `1e-8`.

Distance-3 contexts are additionally checked by exhaustive stabilizer-coset
enumeration under both metrics. The completed table has zero invalid reference
contexts. Its maximum row/column log-ratio discrepancy is `8.88e-15`, and its
maximum enumeration discrepancy is `2.85e-13`.

The split named `confirmation` in this design artifact was inspected while the
consensus rule was being constructed. It is therefore retired from confirmation
use despite its original name. The frozen rule points to a separate seed domain,
`qldpc-fno/tensor-consensus-confirmation/v1`, whose samples are drawn only after
the policy and statistical gate are committed.

The complete table is
[`evidence/tensor-policy-data/tensor_policy_data.json`](../evidence/tensor-policy-data/tensor_policy_data.json).
