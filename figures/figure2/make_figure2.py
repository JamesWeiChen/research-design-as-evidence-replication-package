#!/usr/bin/env python3
"""Recompute and render the normalized B0/A1 comparison used in Figure 2."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from solver import MixturePrior, Model, NormalPrior, Params  # noqa: E402
from precision.polish_analytic import (  # noqa: E402
    c_star_prime,
    polish_PR_analytic,
    polish_pure_TR_analytic,
)

from figures.figure_style import (  # noqa: E402
    CUTOFF_REFERENCE,
    INK,
    PR_LINE,
    PR_REFERENCE,
    TR_LINE,
    TR_REFERENCE,
    apply_nhb_style,
    finish_axes,
    format_significant,
    save_pdf_png,
)


VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
OUT = HERE / "output"
REGISTRY = ROOT / "results" / "figure2_results.json"

COMMON = {
    "alpha": 1.0,
    "beta": 1.0,
    "c_a": 200.0,
    "c_d": 4.0 / 25.0,
    "V": 1.0,
    "r": 15.0 / 7.0,
    "c_p": 4.0,
    "delta0": 15.0,
}

CASES = {
    "B0": {
        "label": r"$\mathbf{B}_{\mathbf{0}}$",
        "prior_label": "N(0,1)",
        "components": ((1.0, 0.0, 1.0),),
        "tr_seed": 2.332131949897777,
        "pr_seed": 2.384301789996887,
        "d_xlim": (2.25, 2.46),
        "d_ticks": (2.25, 2.30, 2.35, 2.40, 2.45),
        "payoff_fmt": "%.4f",
    },
    "A1": {
        "label": r"$\mathbf{A}_{\mathbf{1}}$",
        "prior_label": "3/4 N(-3/14,225/196) + 1/4 N(9/14,1/196)",
        "components": (
            (0.75, -3.0 / 14.0, 15.0 / 14.0),
            (0.25, 9.0 / 14.0, 1.0 / 14.0),
        ),
        "tr_seed": 1.648372528269393,
        "pr_seed": 1.6163388508954157,
        "d_xlim": (1.565, 1.700),
        "d_ticks": (1.57, 1.60, 1.63, 1.66, 1.69),
        "payoff_fmt": "%.3f",
    },
}

# Published benchmark values are acceptance targets only; all plotted values
# are recomputed from the public solver.
REFERENCE_VALUES = {
    "B0": {
        "d_tr": 2.332131949897777,
        "c_tr": 2.206399751699776,
        "u_s_tr": 0.16255162065448675,
        "u_r_tr": 0.4999078052903801,
        "acceptance_tr": 0.6211127924670967,
        "d_pr": 2.384301789996887,
        "c_pr": 2.2045819783775706,
        "u_s_pr": 0.16291483842550641,
        "u_r_pr": 0.5304919071426948,
        "acceptance_pr": 0.6404748653803654,
    },
    "A1": {
        "d_tr": 1.648372528269393,
        "c_tr": 2.213846773819967,
        "u_s_tr": 0.24151267737049534,
        "u_r_tr": 0.15782143854596103,
        "acceptance_tr": 0.4788948045918685,
        "d_pr": 1.6163388508954157,
        "c_pr": 2.2083990802014704,
        "u_s_pr": 0.24251744466472394,
        "u_r_pr": 0.14371355210361825,
        "acceptance_pr": 0.4732454644261148,
    },
}


def prior_for(case: dict):
    components = [NormalPrior(mean, sd) for _, mean, sd in case["components"]]
    weights = [weight for weight, _, _ in case["components"]]
    if len(components) == 1:
        return components[0]
    return MixturePrior(components, weights)


def normal_pdf(x: np.ndarray, mean: float, sd: float) -> np.ndarray:
    z = (x - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def acceptance_probability(model: Model, d: float, cutoff: float) -> float:
    threshold = cutoff - model.P.alpha * d - model.P.Y(d)
    return float(model.F.mass(threshold, math.inf))


def solve_case(label: str, spec: dict) -> tuple[Model, dict]:
    model = Model(prior_for(spec), Params(**COMMON))
    tr = polish_pure_TR_analytic(model, spec["tr_seed"])
    pr = polish_PR_analytic(model, spec["pr_seed"])
    tr["acceptance"] = acceptance_probability(model, tr["d"], tr["c"])
    pr["acceptance"] = acceptance_probability(model, pr["d"], pr["c"])
    tr["cutoff_slope"] = c_star_prime(model, tr["d"], tr["c"])
    tr["global_deviation_gap_16001"] = model.deviation_gap(
        tr["d"], d_max=4.0, br_grid=16001
    )

    result = {
        "case": label,
        "prior": spec["prior_label"],
        "prior_components": [
            {"weight": w, "mean": mean, "sd": sd}
            for w, mean, sd in spec["components"]
        ],
        "parameters": COMMON,
        "TR": tr,
        "PR": pr,
        "comparison": {
            "d_PR_minus_TR": pr["d"] - tr["d"],
            "cutoff_PR_minus_TR": pr["c"] - tr["c"],
            "U_S_PR_minus_TR": pr["U_S"] - tr["U_S"],
            "U_R_PR_minus_TR": pr["U_R"] - tr["U_R"],
            "acceptance_PR_minus_TR": pr["acceptance"] - tr["acceptance"],
        },
    }
    validate_case(label, result)
    return model, result


def validate_case(label: str, result: dict) -> None:
    target = REFERENCE_VALUES[label]
    observed = {
        "d_tr": result["TR"]["d"],
        "c_tr": result["TR"]["c"],
        "u_s_tr": result["TR"]["U_S"],
        "u_r_tr": result["TR"]["U_R"],
        "acceptance_tr": result["TR"]["acceptance"],
        "d_pr": result["PR"]["d"],
        "c_pr": result["PR"]["c"],
        "u_s_pr": result["PR"]["U_S"],
        "u_r_pr": result["PR"]["U_R"],
        "acceptance_pr": result["PR"]["acceptance"],
    }
    largest = max(abs(observed[key] - target[key]) for key in target)
    if largest > 2.0e-10:
        raise RuntimeError(f"{label} differs from the Figure 2 reference values")
    if result["TR"]["global_deviation_gap_16001"] > 2.0e-10:
        raise RuntimeError(f"{label} pure-TR point is not a global best response")
    if label == "B0" and not result["PR"]["d"] > result["TR"]["d"]:
        raise RuntimeError("B0 design ordering failed")
    if label == "A1" and not result["PR"]["d"] < result["TR"]["d"]:
        raise RuntimeError("A1 design ordering failed")
    if label == "A1" and not result["PR"]["U_R"] < result["TR"]["U_R"]:
        raise RuntimeError("A1 reviewer-payoff reversal failed")


def mark_designs(
    ax: plt.Axes,
    d_tr: float,
    d_pr: float,
    y_tr: float,
    y_pr: float,
) -> None:
    ax.axvline(d_tr, **TR_REFERENCE)
    ax.axvline(d_pr, **PR_REFERENCE)
    ax.scatter(
        [d_tr], [y_tr], s=24, facecolors="white", edgecolors=INK,
        linewidths=1.0, zorder=4,
    )
    ax.scatter(
        [d_pr], [y_pr], s=24, facecolors=INK, edgecolors=INK,
        linewidths=1.0, zorder=4,
    )


def render(models: dict[str, Model], results: dict[str, dict]) -> None:
    apply_nhb_style()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(7.2, 7.25),
        gridspec_kw={
            "height_ratios": [0.92, 1.0, 1.08],
            "hspace": 0.74,
            "wspace": 0.34,
        },
    )
    fig.subplots_adjust(left=0.09, right=0.982, bottom=0.11, top=0.915)
    prior_x = np.linspace(-3.5, 3.5, 1401)

    for col, label in enumerate(("B0", "A1")):
        spec = CASES[label]
        model = models[label]
        values = results[label]
        d_tr = values["TR"]["d"]
        d_pr = values["PR"]["d"]
        c_tr = values["TR"]["c"]
        panel = "a" if col == 0 else "b"
        position = axes[0, col].get_position()
        fig.text(
            0.5 * (position.x0 + position.x1),
            0.982,
            f"({panel}) {spec['label']}",
            ha="center",
            va="top",
            fontsize=9.8,
            fontweight="bold",
        )

        ax = axes[0, col]
        density = np.zeros_like(prior_x)
        for weight, mean, sd in spec["components"]:
            density += weight * normal_pdf(prior_x, mean, sd)
            ax.axvline(mean, **CUTOFF_REFERENCE)
        ax.plot(prior_x, density, linewidth=1.8, color=INK)
        ax.set_xlim(-3.5, 3.5)
        ax.set_xticks((-3, -2, -1, 0, 1, 2, 3))
        ax.set_xlabel(r"State, $\theta$")
        ax.set_ylabel("Probability density")
        ax.set_title("Prior distribution", fontweight="bold", pad=5)
        finish_axes(ax)
        prior_box = {
            "facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.4
        }
        if label == "B0":
            ax.text(
                0.0, 0.90, r"$\mu=0$", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7.4, bbox=prior_box,
            )
        else:
            ax.text(
                -3.0 / 14.0, 0.92,
                r"$\mu_L=-3/14$" + "\n" + r"$\pi_L=3/4$",
                transform=ax.get_xaxis_transform(), ha="right", va="top",
                fontsize=7.4, bbox=prior_box,
            )
            ax.text(
                9.0 / 14.0, 0.92,
                r"$\mu_H=9/14$" + "\n" + r"$\pi_H=1/4$",
                transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=7.4, bbox=prior_box,
            )

        d_grid = np.linspace(spec["d_xlim"][0], spec["d_xlim"][1], 500)
        cutoff_grid = np.asarray([model.c_star(float(d)) for d in d_grid])

        ax = axes[1, col]
        ax.plot(d_grid, cutoff_grid, linewidth=1.8, color=INK)
        ax.axhline(c_tr, **CUTOFF_REFERENCE)
        mark_designs(
            ax, d_tr, d_pr, values["TR"]["c"], values["PR"]["c"]
        )
        ax.set_xlim(spec["d_xlim"])
        ax.set_xticks(spec["d_ticks"])
        ax.set_xlabel(r"Design rigour, $d$")
        ax.set_ylabel(r"Acceptance cutoff, $c^*(d)$")
        ax.set_title(r"Cutoff schedule $c^*(d)$", fontweight="bold", pad=5)
        finish_axes(ax)

        ax = axes[2, col]
        u_tr = np.asarray([model.U_S(float(d), c_tr) for d in d_grid])
        u_pr = np.asarray([
            model.U_S(float(d), model.c_star(float(d))) for d in d_grid
        ])
        ax.plot(d_grid, u_tr, label=r"$U_S(d\mid c^{TR})$", **TR_LINE)
        ax.plot(d_grid, u_pr, label=r"$U_S(d\mid c^*(d))$", **PR_LINE)
        mark_designs(
            ax, d_tr, d_pr, values["TR"]["U_S"], values["PR"]["U_S"]
        )
        ax.set_xlim(spec["d_xlim"])
        ax.set_xticks(spec["d_ticks"])
        ax.set_xlabel(r"Design rigour, $d$")
        ax.set_ylabel("Scholar payoff")
        ax.set_title("Scholar payoff", fontweight="bold", pad=5)
        ax.yaxis.set_major_formatter(FormatStrFormatter(spec["payoff_fmt"]))
        finish_axes(ax)

    handles, labels = axes[2, 0].get_legend_handles_labels()
    handles.extend([
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0,
            label=r"$d^{TR}$",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", markersize=5.5,
            markerfacecolor=INK, markeredgecolor=INK, markeredgewidth=1.0,
            label=r"$d^{PR}$",
        ),
    ])
    labels.extend([r"$d^{TR}$", r"$d^{PR}$"])
    fig.legend(
        handles, labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 0.012), ncol=4, handlelength=3.2,
        columnspacing=1.6,
    )
    save_pdf_png(fig, OUT / "figure2")
    plt.close(fig)


def write_records(results: dict[str, dict]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    registry = {
        "release_version": VERSION,
        "release_date": "2026-08-21",
        "status": "self-contained numerical reproduction",
        "normalization": COMMON,
        "case_order": ["B0", "A1"],
        "cases": results,
    }
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    with (OUT / "figure2_values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "case", "prior", "c_p", "delta0", "d_TR", "c_TR", "U_S_TR",
            "U_R_TR", "TR_acceptance", "d_PR", "c_PR", "U_S_PR", "U_R_PR",
            "PR_acceptance", "U_R_PR_minus_TR",
        ])
        for label in ("B0", "A1"):
            case = results[label]
            tr, pr = case["TR"], case["PR"]
            writer.writerow([
                label, case["prior"], COMMON["c_p"], COMMON["delta0"],
                tr["d"], tr["c"], tr["U_S"], tr["U_R"], tr["acceptance"],
                pr["d"], pr["c"], pr["U_S"], pr["U_R"], pr["acceptance"],
                case["comparison"]["U_R_PR_minus_TR"],
            ])

    b0, a1 = results["B0"], results["A1"]
    caption = rf"""# Figure 2 caption

**Figure 2 | Prior shape, endogenous acceptance cutoffs, and design responses
under TR and PR.** Panel (a), benchmark B0, uses
$\theta\sim\mathcal{{N}}(0,1)$. Panel (b), benchmark A1, uses
$\theta\sim\frac34\mathcal{{N}}(-3/14,225/196)
+\frac14\mathcal{{N}}(9/14,1/196)$. Both cases set $c_p=4$ and
$\delta_0=15$; the manuscript reports the common normalization and remaining
primitives. The top row plots the prior density. Dashed vertical lines mark
component means, and panel (b) prints the mixture weights beside them. The
middle row plots the endogenous cutoff $c^*(d)$; the horizontal dashed line
marks the TR cutoff. The bottom row compares the scholar's payoff at the fixed
TR cutoff, $U_S(d\mid c^{{TR}})$ (grey dashed), with the payoff under the
design-specific PR cutoff, $U_S(d\mid c^*(d))$ (black solid). Open circles mark
the TR choice, and filled circles mark the PR choice. In panel (a), the cutoff
falls locally with design and
$d^{{PR}}={format_significant(b0['PR']['d'])}>d^{{TR}}={format_significant(b0['TR']['d'])}$.
In panel (b), the cutoff rises locally with design and
$d^{{PR}}={format_significant(a1['PR']['d'])}<d^{{TR}}={format_significant(a1['TR']['d'])}$;
the reviewer-payoff comparison is
$U_R^{{TR}}={format_significant(a1['TR']['U_R'])}>U_R^{{PR}}={format_significant(a1['PR']['U_R'])}$.
Within each column, the middle and bottom rows use the same design-rigour range.
"""
    (OUT / "figure2_caption.md").write_text(caption, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    models: dict[str, Model] = {}
    results: dict[str, dict] = {}
    for label in ("B0", "A1"):
        models[label], results[label] = solve_case(label, CASES[label])
        print(
            f"PASS {label}: d_TR={results[label]['TR']['d']:.9f}; "
            f"d_PR={results[label]['PR']['d']:.9f}"
        )
    write_records(results)
    render(models, results)
    print(f"Wrote {OUT / 'figure2.pdf'}")
    print(f"Wrote {OUT / 'figure2.png'}")


if __name__ == "__main__":
    main()
