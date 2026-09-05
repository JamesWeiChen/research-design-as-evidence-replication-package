# Data and code availability checklist

Release `20260905-v2`. Scope: the selected solver and Figures 2–3.
Benchmark: [DCAS 1.0](https://datacodestandard.org/) and the
[Social Science Data Editors README template](https://social-science-data-editors.github.io/template_README/).
This is a local, scoped verification record, not approval by a journal or archive.

| Requirement | Status | Evidence |
|---|---|---|
| Data availability | PASS | README explicitly states all model inputs are included; no restricted or external datasets |
| Raw data | PASS / not applicable | Deterministic theoretical examples; no observed raw dataset |
| Analysis data and format | PASS | Recomputed JSON/CSV supplied in open formats |
| Input and output metadata | PASS | docs/DATA_DICTIONARY.md and docs/SOLVER_GUIDE.md |
| Data citations | PASS / not applicable | No external datasets; the related paper is identified in CITATION.cff |
| Transformation and analysis code | PASS for scoped exhibits | Model solution, registry production and rendering code included |
| Source format | PASS | Python and shell source |
| Instruments, ethics, preregistration | PASS / not applicable to these computations | No participant study; PR denotes a modeled review regime |
| Documentation | PASS | README commands, dependency versions, hardware, runtime, exhibit map and known limitations |
| License | PASS | MIT for original package contents; dependencies and paper remain separately licensed |
| Omissions | PASS for stated scope | No omitted inputs for Figures 2–3; full-manuscript proofs and other exhibits outside scope |
| Archival location | INCOMPLETE for a journal-specific deposit | GitHub repository identified; target-journal archive acceptance/software DOI not established |

Computational checks and measured results: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
Fresh dependency restore and cross-platform execution are not claimed. Use the
working-paper DOI to cite the paper; it does not identify this software archive.
