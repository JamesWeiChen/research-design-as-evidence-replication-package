# Research Design as Evidence — replication package

Release: `20260905-v2`

This package contains the selected numerical solver used in the paper and the
complete programs needed to reproduce Figures 2 and 3. 

## Quick start

Python 3.13 is the tested version. With an existing Conda `data_analysis`
environment:

```bash
conda run -n data_analysis python -m pip install -r requirements-tested.txt
conda run -n data_analysis sh run_all.sh
```

Alternatively, install into a virtual environment (no Conda required):

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
sh run_all.sh
```

`requirements-tested.txt` records exact versions of project dependencies and their
runtime dependencies. `requirements.txt` lists broader lower bounds for users
maintaining their own environment; those bounds are not a tested compatibility
matrix. Installation needs network access; the computation itself runs offline.

The master script checks source integrity, runs unit tests, recomputes both
numerical registries, renders PDF/PNG figures, and validates numerical targets,
all CSV cells, captions, and output structure. It can be run repeatedly. It never
rewrites the release manifest. Local environments, Git internals and caches are
excluded from release checks.

To check an untouched download against all published hashes **before** running:

```bash
python verify_release.py --archive
```

The default `python verify_release.py` checks the file set, source hashes and
release metadata. Generated outputs are compared numerically by the figure
validators during `run_all.sh`; their bytes can vary across software versions.

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

## Working paper and citation

Wei-Cheng Chen, Wei James Chen, and Greg Chih-Hsin Sheen (2026).
*Research Design as Evidence: Information and Incentives in Peer Review*.
SSRN working paper. [DOI: 10.2139/ssrn.7353099](https://doi.org/10.2139/ssrn.7353099).
See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.
The working paper was written 2026-08-26, posted 2026-08-28, and revised
2026-08-29 (53 pages). Title, author order and these dates follow the SSRN
landing-page metadata supplied by the author. ORCIDs and DOI were checked
against [Crossref](https://api.crossref.org/works/10.2139/ssrn.7353099) on
2026-09-05; Crossref shortens the second author to Wei Chen.
The package version identifies these numerical benchmarks; the SSRN manuscript
may be revised independently. This release covers Figures 2 and 3 and the
selected solver, not every statement or proof in the paper.

## Data and code availability

All inputs for these figures are theoretical model parameters and analytic
prior distributions defined in the supplied Python source. No external dataset,
restricted data, paid data access, participant data, survey instrument or manual
preparation is required. There are no omitted input data for these computations.
`PR` denotes the modeled preregistration-review regime; it is not a claim that
this numerical exercise was preregistered. No random sampling is used; the
`SEEDS` constants are deterministic starting coordinates for numerical solvers.

| Exhibit | Input definitions | Computation and rendering | Registry and outputs |
|---|---|---|---|
| Figure 2(a) B0, 2(b) A1 | `COMMON`, `CASES` in `figures/figure2/make_figure2.py` | `python figures/figure2/make_figure2.py` | `results/figure2_results.json`; `figures/figure2/output/` |
| Figure 3(a) B1, 3(b) B2 | `COMMON`, `CASES`, `SEEDS` in `precision/compute_figure3.py` | `python precision/compute_figure3.py`, then `python figures/figure3/make_figure3.py` | `results/figure3_results.json`; `figures/figure3/output/` |

Each output directory contains one PDF, one PNG, one numerical CSV and one
Markdown caption. JSON files retain parameters and numerical diagnostics;
CSV files provide selected unrounded scalar results. See
[the data dictionary](docs/DATA_DICTIONARY.md) for field meanings and ranges.

## Computational requirements and verification

The full calculation takes about one minute on the tested Intel Core i7-1068NG7
2.30GHz machine with 32 GiB RAM and macOS 15.7.7 x86_64. The code and reference
outputs occupy approximately 1.3 MB; allow additional disk space for Python,
installed packages and temporary rendered files. This is an observed machine,
not a measured minimum RAM specification. No GPU, R, Stata or LaTeX is required.

The tested environment is Python 3.13.12, NumPy 2.5.2, SciPy 1.18.1,
Matplotlib 3.11.1, Pillow 12.3.0 and pypdf 6.17.0. Exact project dependency
versions are in `requirements-tested.txt`. A clean file copy was rerun in the
existing Conda environment; a fresh dependency installation and other operating
systems have not been verified. See [ACCEPTANCE.md](ACCEPTANCE.md) for the
acceptance criteria and [the reproducibility record](docs/REPRODUCIBILITY.md)
for measured checks, limitations and release maintenance. The
[DCAS checklist](DCAS_checklist.md) records the scope of local verification.

## License and repository

This package's original source code, documentation and generated numerical and
figure outputs are distributed under the [MIT License](LICENSE). Third-party
Python dependencies retain their own licenses. The separately hosted working
paper is not relicensed by this package.

The code repository is
[research-design-as-evidence-replication-package](https://github.com/JamesWeiChen/research-design-as-evidence-replication-package).
A journal-specific archival deposit or software DOI has not been registered;
the DOI above identifies the working paper, not this software release.
