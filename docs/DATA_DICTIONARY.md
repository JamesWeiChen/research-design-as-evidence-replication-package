# Inputs and output dictionary

These are deterministic, dimensionless model calculations, not observational
records. Each CSV has one row per case; all numerical fields are unrounded
floating-point results. All reported scalar values must be finite.

## Input manifest

| Input | Source | Values / role | Access |
|---|---|---|---|
| Common primitives | `COMMON` in both computation scripts | alpha=beta=V=1, c_a=200, c_d=4/25, r=15/7 | Included source, MIT |
| Figure 2 cases | `CASES` in `figures/figure2/make_figure2.py` | c_p=4, delta0=15; B0 standard Normal; A1 mixture below | Included source, MIT |
| A1 prior | `CASES['A1']['components']` | weights (3/4,1/4), means (-3/14,9/14), standard deviations (15/14,1/14); caption uses variances | Included source, MIT |
| Figure 3 cases | `CASES` in `precision/compute_figure3.py` | standard Normal, c_p=5/6; delta0=15 for B1 and 20000 for B2 | Included source, MIT |
| Solver initial coordinates | Figure 2 case seeds; Figure 3 `SEEDS` | Localize numerical branches before root/FOC refinement; not RNG seeds | Included source, MIT |
| Benchmark acceptance targets | Figure 2 `REFERENCE_VALUES`; Figure 3 `CASES` and verifier targets | Check computed results; do not replace the solver | Included source, MIT |

No external raw or analysis dataset, data access application, data access payment,
or manually generated intermediate is required.

## CSV fields

| Field | Meaning | Range / cases |
|---|---|---|
| case | Named numerical example | B0/A1 in Figure 2; B1/B2 in Figure 3 |
| prior | Human-readable prior specification | Normal or Normal mixture |
| c_p, delta0 | Technology primitives | Positive in these cases |
| TR_type | TR candidate type | `mixed` in Figure 3 |
| d_TR, d_PR | Pure-TR or PR design rigour | Nonnegative; d_TR only in Figure 2 |
| c_TR, c_PR | Equilibrium acceptance cutoff | Finite real; c_TR only in Figure 2 |
| U_S_TR, U_S_PR | Scholar payoff | Finite real; Figure 2 only |
| U_R_TR, U_R_PR | Reviewer payoff | Finite real; Figure 3 TR is the mixture payoff |
| TR_acceptance, PR_acceptance | Population acceptance probability | [0,1] |
| U_R_PR_minus_TR | PR reviewer payoff minus TR reviewer payoff | Signed real |

Figure 2 schema has 15 columns, two rows. Figure 3 has 12 columns, two rows.
CSV equality to JSON uses absolute tolerance 1e-12, relative tolerance zero;
strings, column sets and case order are checked exactly.

## JSON registry fields

- `release_version`, `release_date`: package release identity; not the date of
  every rerun. `status`: scope/qualification of the computed results.
- `normalization`, `parameters`: common and case-specific primitives listed
  above. `case_order` controls figure columns and CSV rows.
- `cases`: mapping from case labels to their inputs and computed TR/PR results.
- Figure 2 `prior_components`: component `weight`, `mean`, `sd`; weights sum to 1.
- `TR.d`, `PR.d`, `TR.c`, `PR.c`, `U_S`, `U_R`, `acceptance`: design, cutoff,
  payoffs and probability as defined above.
- `foc_residual`, `cutoff_residual`: first-order-condition and reviewer cutoff
  equation errors; use absolute magnitude when applying acceptance tolerances.
- Figure 2 `TR.cutoff_slope`: derivative of the fixed-design cutoff at TR;
  `global_deviation_gap_16001`: refined global scholar deviation diagnostic
  using a 16001-point design grid on the documented domain.
- Figure 2 `comparison`: PR minus TR for design, cutoff, scholar payoff,
  reviewer payoff and acceptance.
- Figure 3 `environment`: actual Python, NumPy and SciPy versions used to
  recompute the registry. Rendering versions are documented separately.
- Figure 3 `TR.d_L`, `d_H`: low/high support designs, ordered and nonnegative;
  `w_L`, `w_H`: positive population probabilities summing to 1;
  `c`: common cutoff; `U_R_mix`: mixture reviewer payoff.
- `acceptance_L`, `acceptance_H`: conditional acceptance probabilities;
  `acceptance` is their mixture-weighted average.
- `Y_d_L`, `Y_d_H`, `Y_d_cp`, `Y_d_pp`, `PR.Y`: positive adjustment reaches.
- `d_cp`, `d_pp`: cutoff-preserving and reviewer-payoff-preserving comparison
  designs; `c_star_dcp`, `c_star_dpp`: their endogenous cutoffs;
  `U_R_dcp`: reviewer payoff at the cutoff-preserving comparison design.
- `tie_residual`, `balance_residual`, `foc_L`, `foc_H`, `G_residual`,
  `UR_residual`: signed equation errors for mixture indifference, pool balance,
  support first-order conditions and the two comparison-design equations.
- `diagnostics.status`: `approximate_candidate` for the supplied Figure 3 cases.
  `equilibrium_conditions` contains boolean sampled/refined diagnostics:
  on-path rejection, support global optimality, cutoff-pool acceptability,
  common-gap rejection and lower-boundary rejection.
- `diagnostics.max_support_payoff_gap`: support deviation gap;
  `balance_residual`: pool balance diagnostic; `common_gap_max_payoff` and
  `low_boundary_payoff`: reviewer payoff diagnostics, nonpositive when the
  corresponding rejection conditions hold within the solver tolerance.

See `ACCEPTANCE.md` for residual/target tolerances and `docs/SOLVER_GUIDE.md`
for model equations, integration conventions and numerical qualifications.
The original solver's sampled conditions are not analytic proofs of equilibrium
existence, uniqueness or unrestricted global optimality.
