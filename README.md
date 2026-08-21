# NHB solver and Figure 2--3 replication package

Release: `20260821-v1`

This package contains the selected numerical solver used in the paper and the
complete programs needed to reproduce Figures 2 and 3. The code uses the
paper's current notation, contains English documentation throughout, and has
no dependency on private paths or unpublished source trees.

## Quick start

Python 3.10 or later is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` executes unit tests, recomputes both numerical registries, renders
PDF and PNG figures, validates captions and CSV files, and checks the release
manifest.

Figure annotations display numerical values to three significant digits.
Figure 2(b) reports both component means and mixture weights; Figure 3 reports
the population weights on the two mixed-TR support designs. Machine-readable
registries and CSV files retain unrounded values.

## Package contents

```text
solver/                 Main fixed-design and mixed-design solver
precision/              Root and first-order-condition polishing tools
figures/figure2/        Figure 2 renderer, verifier, and outputs
figures/figure3/        Figure 3 renderer, verifier, and outputs
results/                Machine-readable numerical registries
tests/                  Public API and numerical smoke tests
docs/SOLVER_GUIDE.md    Equations, control flow, and numerical qualifications
```

The selected solver is the only solver implementation in this release.

## Reproduce individual outputs

Run commands from the package root.

```bash
python figures/figure2/make_figure2.py
python figures/figure2/verify_figure2.py

python precision/compute_figure3.py
python figures/figure3/make_figure3.py
python figures/figure3/verify_figure3.py
```

The generated files are:

- `results/figure2_results.json`
- `figures/figure2/output/figure2.pdf`
- `figures/figure2/output/figure2.png`
- `figures/figure2/output/figure2_values.csv`
- `figures/figure2/output/figure2_caption.md`
- `results/figure3_results.json`
- `figures/figure3/output/figure3.pdf`
- `figures/figure3/output/figure3.png`
- `figures/figure3/output/figure3_values.csv`
- `figures/figure3/output/figure3_caption.md`

## Solver API

The package-level imports cover the main public interface:

```python
from solver import Model, NormalPrior, Params

prior = NormalPrior(0.0, 1.0)
params = Params(
    alpha=1.0,
    beta=1.0,
    c_a=200.0,
    c_p=4.0,
    c_d=4.0 / 25.0,
    delta0=15.0,
    V=1.0,
    r=15.0 / 7.0,
)
model = Model(prior, params)

cutoff = model.c_star(d=2.0)
pr_solution = model.solve_PR(n=2001, d_max=4.0)
tr_scan = model.scan_TR(d_max=4.0, n=401, br_grid=2001)
```

For a finite-support common-cutoff mixed-TR candidate:

```python
from solver import DesignMixture, MixedTRSolver

mixture = DesignMixture(support=(0.05, 2.37), weights=(0.4, 0.6))
diagnostic = MixedTRSolver(model).evaluate_candidate(mixture, cutoff=2.33)
```

The diagnostic is tolerance-qualified. It should not be interpreted as an
analytic proof of equilibrium existence, uniqueness, or global optimality.

## Notation

The code follows the manuscript notation directly:

- `d`: design rigour
- `c`: manuscript acceptance cutoff
- `theta`: scholar type
- `alpha`, `beta`, `c_a`, `c_p`, `c_d`, `delta0`, `V`, `r`: primitives
- `omega(d)`: signal weight
- `Y(d)`: maximum profitable adjustment reach
- `G(c, d)`: reviewer payoff at a marginal manuscript
- `U_S(d, c)` and `U_R(d, c)`: scholar and reviewer payoffs
- `c_star(d)`: fixed-design acceptance cutoff
- `TR` and `PR`: traditional review and preregistration review

See [docs/SOLVER_GUIDE.md](docs/SOLVER_GUIDE.md) for the numerical sequence
and the role of each module.

## Validation and metadata

`ACCEPTANCE.md` records the numerical checks. `MANIFEST.sha256` records file
hashes, excluding the manifest itself. Numerical registries contain only the
public release identifier, release date, parameters, results, solver status,
and relevant package versions.

## Distribution note

Project-specific license and citation terms were not supplied and are not
inferred here. Add the intended license and citation metadata before posting
the package to a public repository.
