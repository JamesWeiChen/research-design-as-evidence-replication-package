#!/usr/bin/env python3
"""Validate Figure 3 numerics, caption binding, and output structure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figures.figure_style import format_significant
from figures.validation import validate_csv, require_finite


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
OUT = HERE / "output"
VERSION = (PACKAGE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
REGISTRY = PACKAGE_ROOT / "results" / "figure3_results.json"
STEM = "figure3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    expected_formats = {
        3.55943e-5: "0.0000356",
        2.400315: "2.40",
        0.038145: "0.0381",
        -0.011999: "-0.0120",
    }
    for value, expected_text in expected_formats.items():
        require(
            format_significant(value) == expected_text,
            f"significant-digit formatting for {value}",
        )

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    require_finite(registry)
    require(registry["release_version"] == VERSION, "wrong registry version")
    require(registry["case_order"] == ["B1", "B2"], "wrong case order")
    require(set(registry["cases"]) == {"B1", "B2"}, "registry must contain only B1/B2")
    require(registry["normalization"] == {
        "alpha": 1.0,
        "beta": 1.0,
        "c_a": 200.0,
        "c_d": 4.0 / 25.0,
        "V": 1.0,
        "r": 15.0 / 7.0,
    }, "common normalization mismatch")

    expected = {
        "B1": (15.0, 0.572920, 0.640661, 0.038145),
        "B2": (20000.0, 0.603288, 0.632086, -0.011999),
    }
    for label, (delta0, tr_acceptance, pr_acceptance, ur_diff) in expected.items():
        case = registry["cases"][label]
        tr, pr = case["TR"], case["PR"]
        require(case["prior"] == "N(0,1)", f"{label} prior")
        require(abs(case["parameters"]["c_p"] - 5.0 / 6.0) <= 1e-15,
                f"{label} c_p")
        require(case["parameters"]["delta0"] == delta0, f"{label} delta0")
        require(tr["type"] == "mixed", f"{label} TR type")
        require(abs(tr["acceptance"] - tr_acceptance) <= 1e-6,
                f"{label} TR acceptance")
        require(abs(pr["acceptance"] - pr_acceptance) <= 1e-6,
                f"{label} PR acceptance")
        require(abs(case["U_R_PR_minus_TR"] - ur_diff) <= 1e-6,
                f"{label} reviewer-payoff difference")
        require(max(abs(pr["foc_residual"]), abs(pr["cutoff_residual"])) <= 2e-10,
                f"{label} PR residuals")
        require(tr["diagnostics"]["status"] == "approximate_candidate",
                f"{label} mixed status")
        conditions = tr["diagnostics"]["equilibrium_conditions"]
        require(set(conditions) == {
            "on_path_rejection", "support_global_optimality",
            "cutoff_pool_acceptable", "common_gap_rejected",
            "lower_boundary_rejected",
        }, f"{label} missing equilibrium condition")
        require(
            all(value is True for value in conditions.values()),
            f"{label} equilibrium conditions",
        )
        require(max(
            abs(tr["tie_residual"]),
            abs(tr["balance_residual"]),
            abs(tr["G_residual"]),
            abs(tr["UR_residual"]),
            tr["diagnostics"]["max_support_payoff_gap"],
        ) <= 2e-9, f"{label} mixed residuals")

    b1, b2 = registry["cases"]["B1"], registry["cases"]["B2"]
    require(b1["PR"]["d"] > b1["TR"]["d_pp"] > b1["TR"]["d_cp"], "B1 design ordering")
    require(b2["TR"]["d_pp"] > b2["PR"]["d"] > b2["TR"]["d_cp"], "B2 design ordering")

    csv_path = OUT / "figure3_values.csv"
    validate_csv(csv_path, registry, figure=3)

    source = (HERE / "make_figure3.py").read_text(encoding="utf-8")
    require(r"Pre-adjustment value, $z=\alpha d+\theta$" in source,
            "pre-adjustment z definition missing")
    require(r"$f_Z(z\mid d)$" in source,
            "pre-adjustment density notation missing")
    require(r"Submitted manuscript, $m$" not in source,
            "incorrect smooth-m-density label remains")
    require(r"mapped to an atom at $m=c$" in source,
            "cutoff-atom mapping missing")
    require(r"d_{{cp}}^{{TR}}" in source and r"d_{{pp}}^{{TR}}" in source,
            "benchmark notation missing")
    require('"height_ratios": [1.10, 1.32, 0.80, 0.50]' in source,
            "approved row-height allocation missing")
    require("format_significant" in source,
            "three-significant-digit formatter missing")

    caption = (OUT / "figure3_caption.md").read_text(
        encoding="utf-8"
    )
    require(r"\(f_Z(z\mid d)\)" in caption,
            "caption density definition missing")
    require(r"\(z=\alpha d+\theta\)" in caption,
            "caption pre-adjustment variable definition missing")
    require("becomes an atom at \\(m=c\\)" in caption,
            "caption cutoff-atom explanation missing")
    require("Curve height does not encode" in caption,
            "caption mixed-weight qualification missing")
    require(r"\(\pi_L=\lambda\)" in caption and r"\(\pi_H=1-\lambda\)" in caption,
            "population-weight definitions missing")
    for token in ("| B1 | mixed | 0.573 | 0.641 | +0.0381 |",
                  "| B2 | mixed | 0.603 | 0.632 | -0.0120 |"):
        require(token in caption,
                f"three-significant-digit caption value missing: {token}")

    pdf = OUT / f"{STEM}.pdf"
    png = OUT / f"{STEM}.png"
    try:
        reader = PdfReader(pdf)
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid PDF structure: {exc}") from exc
    require(len(reader.pages) == 1, "PDF must contain exactly one page")
    try:
        with Image.open(png) as image:
            width, height = image.size
            require(image.format == "PNG", "invalid PNG format")
            image.verify()
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid PNG structure: {exc}") from exc
    require(height >= 2400 and width >= 2400, "PNG resolution too low")
    print("PASS: Figure 3 contains only B1 and B2 in the requested order.")
    print("PASS: parameters, targets, mixed-TR/PR residuals, and orderings hold.")
    print("PASS: notation, caption, CSV binding, PDF, and PNG outputs are valid.")


if __name__ == "__main__":
    main()
