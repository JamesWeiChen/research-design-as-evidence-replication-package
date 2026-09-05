#!/usr/bin/env python3
"""Validate Figure 2 numerics, caption binding, and output structure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from figures.validation import validate_csv, require_finite
from figures.figure2.make_figure2 import validate_case


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
OUT = HERE / "output"
REGISTRY = ROOT / "results" / "figure2_results.json"


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    require_finite(registry)
    if (
        registry["release_version"] != VERSION
        or registry["case_order"] != ["B0", "A1"]
    ):
        raise RuntimeError("wrong Figure 2 registry version or case order")
    if set(registry["cases"]) != {"B0", "A1"}:
        raise RuntimeError("Figure 2 registry must contain only B0/A1")
    for label in ("B0", "A1"):
        validate_case(label, registry["cases"][label])
    b0 = registry["cases"]["B0"]
    a1 = registry["cases"]["A1"]
    if not b0["PR"]["d"] > b0["TR"]["d"]:
        raise RuntimeError("B0 design ordering failed")
    if not a1["PR"]["d"] < a1["TR"]["d"]:
        raise RuntimeError("A1 design ordering failed")
    if not a1["PR"]["U_R"] < a1["TR"]["U_R"]:
        raise RuntimeError("A1 reviewer-payoff ordering failed")
    if max(
        b0["TR"]["global_deviation_gap_16001"],
        a1["TR"]["global_deviation_gap_16001"],
    ) > 2.0e-10:
        raise RuntimeError("pure-TR global-best-response gate failed")

    csv_path = OUT / "figure2_values.csv"
    validate_csv(csv_path, registry, figure=2)

    caption = (OUT / "figure2_caption.md").read_text(encoding="utf-8")
    for token in (
        "$c^*(d)$",
        "$d^{PR}=2.38>d^{TR}=2.33$",
        "$d^{PR}=1.62<d^{TR}=1.65$",
        "$U_R^{TR}=0.158>U_R^{PR}=0.144$",
        r"$\theta\sim\frac34\mathcal{N}(-3/14,225/196)",
    ):
        if token not in caption:
            raise RuntimeError(f"Figure 2 caption is missing {token}")

    source = (HERE / "make_figure2.py").read_text(encoding="utf-8")
    for token in (r"$\pi_L=3/4$", r"$\pi_H=1/4$"):
        if token not in source:
            raise RuntimeError(f"Figure 2 prior annotation is missing {token}")

    pdf_path = OUT / "figure2.pdf"
    png_path = OUT / "figure2.png"
    if len(PdfReader(str(pdf_path)).pages) != 1:
        raise RuntimeError("Figure 2 PDF must contain one page")
    with Image.open(png_path) as image:
        image.verify()
    with Image.open(png_path) as image:
        width, height = image.size
    if width < 2000 or height < 2200:
        raise RuntimeError("Figure 2 PNG resolution is below the publication gate")

    private_fragments = ("/" + "Users/", "Drop" + "box")
    if any(fragment in source for fragment in private_fragments):
        raise RuntimeError("Figure 2 renderer still contains an external absolute path")

    print("PASS: Figure 2 contains B0 and A1 in the requested order.")
    print("PASS: reference values, design orderings, and payoff reversal hold.")
    print("PASS: caption, CSV, PDF, PNG, and self-contained paths are valid.")


if __name__ == "__main__":
    main()
