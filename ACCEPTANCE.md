# Acceptance checks

Release `20260821-v1` passes the following public checks.

## Solver structure

- All Python files compile.
- The package-level solver and precision APIs import successfully.
- No private absolute paths, alternative completion routine, compatibility
  alias, development issue tag, or non-English code comment remains.
- Unit tests cover parameter identities, exact-prior moments, cutoff residuals,
  bracket validation, mixture normalization, and the selected public API.

## Figure 2

| Case | Required relationship |
|---|---|
| B0 | `d_PR > d_TR` |
| A1 | `d_PR < d_TR` |
| A1 | `U_R_TR > U_R_PR` |

Each reported benchmark scalar must match its reference value within
`2e-10`, and each pure-TR global deviation gap must be at most `2e-10`.
Figure 2(b) must display the component weights `pi_L = 3/4` and
`pi_H = 1/4` beside their corresponding component means.

## Figure 3

| Case | TR acceptance target | PR acceptance target | `U_R_PR - U_R_TR` target |
|---|---:|---:|---:|
| B1 | 0.572920 | 0.640661 | +0.038145 |
| B2 | 0.603288 | 0.632086 | -0.011999 |

Displayed targets use a tolerance of `1e-6`. PR first-order-condition and
cutoff residuals must be at most `2e-10`. Mixed-TR tie, balance, cutoff, payoff,
and global-support residuals must be at most `2e-9`; all named equilibrium
conditions must pass.
All numerical annotations in Figure 3 must use three significant digits; the
B2 low-design value is displayed as `0.0000356`.

## Output checks

- Each PDF contains one readable page.
- Each PNG is readable and meets the publication-resolution requirement.
- Each CSV is bound to the corresponding JSON registry.
- Each caption contains the notation and numerical comparisons used in the
  figure.
- `MANIFEST.sha256` matches all distributed files except the manifest itself.
