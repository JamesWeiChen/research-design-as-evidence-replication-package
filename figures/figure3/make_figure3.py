#!/usr/bin/env python3
"""Render the normalized B1-versus-B2 comparison used in Figure 3."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from figures.figure_style import (  # noqa: E402
    CUTOFF_REFERENCE,
    DARK_GRAY,
    INK,
    LIGHT_GRAY,
    MID_GRAY,
    PALE_GRAY,
    apply_nhb_style,
    finish_axes,
    format_significant,
    save_pdf_png,
)


OUT = HERE / "output"
VERSION = (PACKAGE_ROOT / "VERSION").read_text(encoding="utf-8").strip()
REGISTRY = PACKAGE_ROOT / "results" / "figure3_results.json"
STEM = "figure3"
XLIM = (-3.8, 5.4)
YLIM = (-0.025, 0.365)
SIGMA = 1.0
RIDGE_HEIGHT = 0.130
BASE_LOW = 0.018
BASE_HIGH = 0.205
BASE_SINGLE = 0.095


def normal_pdf(x: np.ndarray, mean: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / SIGMA) ** 2) / (
        SIGMA * math.sqrt(2.0 * math.pi)
    )


def setup_axes(ax: plt.Axes, *, show_xticklabels: bool) -> None:
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks([-3, 0, 3])
    ax.set_yticks([])
    ax.tick_params(
        axis="x", bottom=show_xticklabels, labelbottom=show_xticklabels,
    )
    finish_axes(ax)


def draw_regions(
    ax: plt.Axes,
    x: np.ndarray,
    ridge: np.ndarray,
    base: float,
    cutoff: float,
    reach: float,
) -> None:
    strategic = (x > cutoff - reach) & (x <= cutoff)
    natural = x > cutoff
    ax.fill_between(
        x[natural], base, ridge[natural],
        facecolor=PALE_GRAY, edgecolor="none", alpha=0.98, zorder=1,
    )
    ax.fill_between(
        x[strategic], base, ridge[strategic],
        facecolor=MID_GRAY, edgecolor="none", alpha=0.72, zorder=2,
    )
    boundary = cutoff - reach
    boundary_height = float(np.interp(boundary, x, ridge))
    ax.vlines(
        boundary, base, boundary_height,
        color=MID_GRAY, linewidth=0.65, linestyle=(0, (2, 2)), zorder=5,
    )
    ax.scatter(
        [boundary], [base], marker="D", s=15, facecolor="white",
        edgecolor=INK, linewidth=0.75, zorder=7, clip_on=False,
    )


def draw_ridge(
    ax: plt.Axes,
    x: np.ndarray,
    *,
    d: float,
    cutoff: float,
    reach: float,
    base: float,
    color: str,
    linestyle: str | tuple = "-",
    label: str,
    label_y: float,
) -> None:
    raw = normal_pdf(x, d)
    ridge = base + RIDGE_HEIGHT * raw / raw.max()
    draw_regions(ax, x, ridge, base, cutoff, reach)
    ax.hlines(base, XLIM[0], XLIM[1], color=INK, linewidth=0.65, zorder=3)
    ax.plot(x, ridge, color=color, linewidth=1.55, linestyle=linestyle, zorder=4)
    ax.text(
        0.025, label_y, label,
        transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        zorder=8,
    )


def actual_tr_panel(ax: plt.Axes, case: dict, x: np.ndarray) -> None:
    tr = case["TR"]
    draw_ridge(
        ax, x, d=tr["d_L"], cutoff=tr["c"], reach=tr["Y_d_L"],
        base=BASE_HIGH, color=DARK_GRAY, linestyle=(0, (5, 3)),
        label=(rf"$d_L^{{TR}}={format_significant(tr['d_L'])}$" + "\n" +
               rf"$\pi_L={format_significant(tr['w_L'])}$"),
        label_y=0.94,
    )
    draw_ridge(
        ax, x, d=tr["d_H"], cutoff=tr["c"], reach=tr["Y_d_H"],
        base=BASE_LOW, color=INK,
        label=(rf"$d_H^{{TR}}={format_significant(tr['d_H'])}$" + "\n" +
               rf"$\pi_H={format_significant(tr['w_H'])}$"),
        label_y=0.44,
    )
    ax.axvline(tr["c"], **CUTOFF_REFERENCE, zorder=6)
    ax.text(
        tr["c"], 0.349, rf"$c^{{TR}}={format_significant(tr['c'])}$",
        ha="center", va="bottom", fontsize=6.8, zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.5},
    )
    setup_axes(ax, show_xticklabels=False)


def benchmark_panel(ax: plt.Axes, case: dict, x: np.ndarray) -> None:
    tr = case["TR"]
    draw_ridge(
        ax, x, d=tr["d_cp"], cutoff=tr["c_star_dcp"], reach=tr["Y_d_cp"],
        base=BASE_HIGH, color=DARK_GRAY, linestyle=(0, (5, 3)),
        label=rf"$d_{{cp}}^{{TR}}={format_significant(tr['d_cp'])}$",
        label_y=0.90,
    )
    draw_ridge(
        ax, x, d=tr["d_pp"], cutoff=tr["c_star_dpp"], reach=tr["Y_d_pp"],
        base=BASE_LOW, color=INK,
        label=rf"$d_{{pp}}^{{TR}}={format_significant(tr['d_pp'])}$",
        label_y=0.40,
    )
    ax.vlines(
        tr["c_star_dcp"], BASE_HIGH, BASE_HIGH + RIDGE_HEIGHT + 0.008,
        **CUTOFF_REFERENCE, zorder=6,
    )
    ax.vlines(
        tr["c_star_dpp"], BASE_LOW, BASE_LOW + RIDGE_HEIGHT + 0.008,
        **CUTOFF_REFERENCE, zorder=6,
    )
    ax.text(
        tr["c_star_dcp"], 0.349,
        rf"$c^*(d_{{cp}}^{{TR}})={format_significant(tr['c_star_dcp'])}$",
        ha="center", va="bottom", fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.4},
        zorder=8,
    )
    ax.text(
        tr["c_star_dpp"], BASE_LOW + RIDGE_HEIGHT + 0.015,
        rf"$c^*(d_{{pp}}^{{TR}})={format_significant(tr['c_star_dpp'])}$",
        ha="center", va="bottom", fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.4},
        zorder=8,
    )
    setup_axes(ax, show_xticklabels=False)


def pr_panel(ax: plt.Axes, case: dict, x: np.ndarray) -> None:
    pr = case["PR"]
    draw_ridge(
        ax, x, d=pr["d"], cutoff=pr["c"], reach=pr["Y"],
        base=BASE_SINGLE, color=INK,
        label=rf"$d^{{PR}}={format_significant(pr['d'])}$",
        label_y=0.86,
    )
    ax.axvline(pr["c"], **CUTOFF_REFERENCE, zorder=6)
    ax.text(
        pr["c"], 0.349, rf"$c^{{PR}}={format_significant(pr['c'])}$",
        ha="center", va="bottom", fontsize=6.8, zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 0.5},
    )
    setup_axes(ax, show_xticklabels=True)


def comparison_panel(ax: plt.Axes, case: dict) -> None:
    tr, pr = case["TR"], case["PR"]
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    if pr["d"] > tr["d_pp"] > tr["d_cp"]:
        ranking = r"$d^{PR}>d_{pp}^{TR}>d_{cp}^{TR}$"
    elif tr["d_pp"] > pr["d"] > tr["d_cp"]:
        ranking = r"$d_{pp}^{TR}>d^{PR}>d_{cp}^{TR}$"
    else:
        raise RuntimeError(f"unexpected design-rigor ordering in {case['case']}")
    ax.text(
        0.50, 0.78, ranking,
        transform=ax.transAxes, ha="center", va="center", fontsize=9.0,
    )
    ax.text(
        0.50, 0.20,
        (rf"$\Pr(TR\ accept)={format_significant(tr['acceptance'])}$" + "\n" +
         rf"$\Pr(PR\ accept)={format_significant(pr['acceptance'])}$" + "\n" +
         rf"$U_R^{{PR}}-U_R^{{TR,mix}}="
         rf"{format_significant(case['U_R_PR_minus_TR'], signed=True)}$"),
        transform=ax.transAxes, ha="center", va="center", fontsize=7.1,
    )
    ax.axis("off")


def write_companion_files(registry: dict) -> None:
    csv_path = OUT / "figure3_values.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case", "prior", "c_p", "delta0", "TR_type", "TR_acceptance",
            "PR_acceptance", "U_R_PR", "U_R_TR", "U_R_PR_minus_TR", "d_PR", "c_PR",
        ])
        for label in registry["case_order"]:
            case = registry["cases"][label]
            tr, pr = case["TR"], case["PR"]
            writer.writerow([
                label, case["prior"], case["parameters"]["c_p"],
                case["parameters"]["delta0"], tr["type"], tr["acceptance"],
                pr["acceptance"], pr["U_R"], tr["U_R_mix"], case["U_R_PR_minus_TR"],
                pr["d"], pr["c"],
            ])

    b1 = registry["cases"]["B1"]
    b2 = registry["cases"]["B2"]
    caption = r"""# Figure 3 caption and reading guide

## Caption

**Figure 3 | Hidden design mixing and reviewer payoff.** Both columns use a
standard Normal prior. B1 sets
\(c_p=5/6\) and \(\delta_0=15\); B2 sets \(c_p=5/6\) and
\(\delta_0=20{,}000\). The first row shows the two support designs in each
tolerance-qualified common-cutoff candidate under mixed traditional review
(TR).
The labels \(\pi_L=\lambda\) and \(\pi_H=1-\lambda\) give the population
probabilities of the low and high designs. The second row shows the
cutoff-preserving and reviewer-payoff-preserving pure-TR benchmarks. The third
row shows preregistration review (PR). The final row reports the design
ranking, TR and PR acceptance probabilities, and the reviewer-payoff
difference.

Each smooth curve is the conditional pre-adjustment density
\(f_Z(z\mid d)\), where \(z=\alpha d+\theta\). Curve height does not encode
the mixed-TR probability; the corresponding mixture weight \(\pi\) is printed
beside each mixed-TR curve. The diamond marks the reach boundary
\(c-Y(d)\). The dark shaded baseline mass satisfies
\(z\in(c-Y(d),c]\) and is mapped by strategic adjustment to the cutoff, so it
becomes an atom at \(m=c\) in the submitted-manuscript distribution. The pale
region satisfies \(z>c\), passes naturally, and retains \(m=z\).

Across the two cases, PR raises reviewer payoff in B1 and lowers it in B2. The
design-rigour comparison is \(d^{PR}>d_{pp}^{TR}>d_{cp}^{TR}\) in B1 and
\(d_{pp}^{TR}>d^{PR}>d_{cp}^{TR}\) in B2.

## Common normalization

\[
\alpha=\beta=V=1,\qquad c_a=200,\qquad c_d=\frac{4}{25},
\qquad r=\frac{15}{7}.
\]

## Displayed numerical comparisons

| Case | TR type | TR acceptance | PR acceptance | \(U_R^{PR}-U_R^{TR,mix}\) |
|---|---|---:|---:|---:|
""" + (
        f"| B1 | mixed | {format_significant(b1['TR']['acceptance'])} | "
        f"{format_significant(b1['PR']['acceptance'])} | "
        f"{format_significant(b1['U_R_PR_minus_TR'], signed=True)} |\n"
        f"| B2 | mixed | {format_significant(b2['TR']['acceptance'])} | "
        f"{format_significant(b2['PR']['acceptance'])} | "
        f"{format_significant(b2['U_R_PR_minus_TR'], signed=True)} |\n"
    )
    (OUT / "figure3_caption.md").write_text(
        caption, encoding="utf-8"
    )


def make_figure(registry: dict) -> plt.Figure:
    apply_nhb_style()
    fig, axes = plt.subplots(
        4, 2, figsize=(7.35, 7.25),
        gridspec_kw={
            "height_ratios": [1.10, 1.32, 0.80, 0.50],
            "hspace": 0.48,
            "wspace": 0.28,
        },
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.145, top=0.915)
    for ax in axes[3, :]:
        position = ax.get_position()
        ax.set_position([
            position.x0, position.y0 - 0.035,
            position.width, position.height,
        ])
    x = np.linspace(XLIM[0], XLIM[1], 2600)
    cases = [registry["cases"][label] for label in registry["case_order"]]
    headers = (
        r"(a) B1: $c_p=5/6$, $\delta_0=15$",
        r"(b) B2: $c_p=5/6$, $\delta_0=20{,}000$",
    )
    for col, (case, header) in enumerate(zip(cases, headers)):
        position = axes[0, col].get_position()
        fig.text(
            0.5 * (position.x0 + position.x1), 0.975,
            header, ha="center", va="top", fontsize=9.0, fontweight="bold",
        )
        actual_tr_panel(axes[0, col], case, x)
        benchmark_panel(axes[1, col], case, x)
        pr_panel(axes[2, col], case, x)
        comparison_panel(axes[3, col], case)

    row_labels = (
        "Mixed TR",
        "TR benchmarks",
        "PR",
        "Comparison",
    )
    for row, label in enumerate(row_labels):
        position = axes[row, 0].get_position()
        fig.text(
            position.x0, position.y1 + 0.009, label,
            ha="left", va="bottom", fontsize=7.4, fontweight="bold",
        )
    for row in range(3):
        axes[row, 0].text(
            -0.105, 0.50, r"$f_Z(z\mid d)$",
            transform=axes[row, 0].transAxes, rotation=90,
            ha="center", va="center", fontsize=7.2,
        )

    left = axes[2, 0].get_position().x0
    right = axes[2, 1].get_position().x1
    x_label_y = axes[2, 0].get_position().y0 - 0.047
    fig.text(
        0.5 * (left + right), x_label_y,
        r"Pre-adjustment value, $z=\alpha d+\theta$",
        ha="center", va="top", fontsize=8.2,
    )

    handles = [
        Line2D(
            [0], [0], marker="D", color="none", markerfacecolor="white",
            markeredgecolor=INK, markersize=5.2,
            label=r"Reach boundary $c-Y(d)$",
        ),
        Patch(
            facecolor=MID_GRAY, edgecolor="none",
            label="Adjustment reach",
        ),
        Patch(
            facecolor=PALE_GRAY, edgecolor=LIGHT_GRAY,
            label=r"Natural pass $z>c$",
        ),
    ]
    fig.legend(
        handles=handles, frameon=False, loc="lower center",
        bbox_to_anchor=(0.51, 0.043), ncol=3, columnspacing=1.5,
        handlelength=2.2, handletextpad=0.6,
    )
    fig.text(
        0.545, 0.014,
        (r"Dark shaded baseline mass is mapped to an atom at $m=c$; "
         r"for $z>c$, $m=z$."),
        ha="center", va="bottom", fontsize=6.5,
    )
    return fig


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["release_version"] != VERSION:
        raise RuntimeError("wrong Figure 3 registry version")
    if registry["case_order"] != ["B1", "B2"]:
        raise RuntimeError("Figure 3 must compare B1 then B2")
    OUT.mkdir(parents=True, exist_ok=True)
    write_companion_files(registry)
    fig = make_figure(registry)
    save_pdf_png(fig, OUT / STEM)
    plt.close(fig)
    print(f"Wrote {OUT / (STEM + '.pdf')}")
    print(f"Wrote {OUT / (STEM + '.png')}")


if __name__ == "__main__":
    main()
