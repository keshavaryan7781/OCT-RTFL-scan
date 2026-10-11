# Local verification

Verified on synthetic data, not the remote HPC scans:

- Non-identity grayscale BMP palette decoded correctly; original/native identity checked for 20 cases
- Pack contains exact original dimensions and blank unreviewed templates; no invented labels
- No automatic subject identities or test split
- Existing output refused and preserved
- Source hash mismatch rejected
- Duplicate cases rejected
- Native PNG intensity mismatch rejected
- Non-grayscale source rejected
- Notebook code cells parse; no stored outputs

The generated 24 x 16 fixtures used the same 20 case IDs solely to test the pack contract. Actual HPC dimensions and hashes are checked when the user runs the notebook. No model accuracy test has been performed.
