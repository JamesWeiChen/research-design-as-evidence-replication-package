#!/usr/bin/env python3
"""Recompute the normalized B1/B2 comparison used in Figure 3."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from solver import DesignMixture, MixedTRSolver, Model, NormalPrior, Params  # noqa: E402
from precision.polish import polish_dcp_dpp  # noqa: E402
from precision.polish_analytic import (  # noqa: E402
    polish_PR_analytic,
    polish_mixed_analytic,
)


VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
OUT = ROOT / "results" / "figure3_results.json"

COMMON = {
    "alpha": 1.0,
    "beta": 1.0,
    "c_a": 200.0,
    "c_d": 4.0 / 25.0,
    "V": 1.0,
    "r": 15.0 / 7.0,
}

CASES = {
    "B1": {
        "c_p": 5.0 / 6.0,
        "delta0": 15.0,
        "acceptance_target": 0.572920,
        "ur_diff_target": 0.038145,
    },
    "B2": {
        "c_p": 5.0 / 6.0,
        "delta0": 20000.0,
        "acceptance_target": 0.603288,
        "ur_diff_target": -0.011999,
    },
}

# Starting values localize the relevant branches before precision polishing.
SEEDS = {
    "B1": {
        "d_L": 0.04827666998039215,
        "d_H": 2.3692690443451996,
        "c": 2.3252489732149657,
        "d_PR": 2.383985581642213,
    },
    "B2": {
        "d_L": 3.559441572786274e-05,
        "d_H": 2.4189874469394517,
        "c": 2.240825708567627,
        "d_PR": 2.3802396730874475,
    },
}


def model_for(label: str) -> Model:
    case = CASES[label]
    return Model(
        NormalPrior(0.0, 1.0),
        Params(
            **COMMON,
            c_p=case["c_p"],
            delta0=case["delta0"],
        ),
    )


def acceptance_probability(model: Model, d: float, cutoff: float) -> float:
    threshold = cutoff - model.P.alpha * d - model.P.Y(d)
    return float(model.F.mass(threshold, math.inf))


def run_case(label: str) -> dict:
    model = model_for(label)
    seed = SEEDS[label]
    pr = polish_PR_analytic(model, seed["d_PR"])
    mixed = polish_mixed_analytic(
        model, seed["d_L"], seed["d_H"], seed["c"]
    )
    mixture = DesignMixture(
        support=(mixed["d_L"], mixed["d_H"]),
        weights=(mixed["w_L"], mixed["w_H"]),
    )
    diag = MixedTRSolver(model).evaluate_candidate(
        mixture,
        mixed["c"],
        gap_points=2401,
        best_response_points=8001,
        d_max=4.0,
        condition_tolerance=2.0e-7,
    )
    inverse = polish_dcp_dpp(
        model,
        mixed["c"],
        mixed["U_R_mix"],
        mixed["d_L"],
        mixed["d_H"],
    )
    accept_l = acceptance_probability(model, mixed["d_L"], mixed["c"])
    accept_h = acceptance_probability(model, mixed["d_H"], mixed["c"])
    acceptance = mixed["w_L"] * accept_l + mixed["w_H"] * accept_h
    ur_diff = pr["U_R"] - mixed["U_R_mix"]

    case = {
        "case": label,
        "prior": "N(0,1)",
        "parameters": {**COMMON, **{
            "c_p": CASES[label]["c_p"],
            "delta0": CASES[label]["delta0"],
        }},
        "TR": {
            "type": "mixed",
            **mixed,
            "Y_d_L": model.P.Y(mixed["d_L"]),
            "Y_d_H": model.P.Y(mixed["d_H"]),
            "acceptance_L": accept_l,
            "acceptance_H": accept_h,
            "acceptance": acceptance,
            **inverse,
            "Y_d_cp": model.P.Y(inverse["d_cp"]),
            "Y_d_pp": model.P.Y(inverse["d_pp"]),
            "diagnostics": {
                "status": diag.status,
                "equilibrium_conditions": {
                    "on_path_rejection": diag.on_path_rejection,
                    "support_global_optimality": diag.support_global_optimality,
                    "cutoff_pool_acceptable": diag.cutoff_pool_acceptable,
                    "common_gap_rejected": diag.common_gap_rejected,
                    "lower_boundary_rejected": diag.lower_boundary_rejected,
                },
                "max_support_payoff_gap": diag.max_support_payoff_gap,
                "balance_residual": diag.balance_residual,
                "common_gap_max_payoff": diag.common_gap_max_payoff,
                "low_boundary_payoff": diag.low_boundary_payoff,
            },
        },
        "PR": {
            **pr,
            "Y": model.P.Y(pr["d"]),
            "acceptance": acceptance_probability(model, pr["d"], pr["c"]),
        },
        "U_R_PR_minus_TR": ur_diff,
    }
    gate_case(label, case)
    return case


def gate_case(label: str, case: dict) -> None:
    tr, pr = case["TR"], case["PR"]
    target = CASES[label]
    if abs(tr["acceptance"] - target["acceptance_target"]) > 1.0e-6:
        raise RuntimeError(f"{label} TR acceptance mismatch")
    if abs(case["U_R_PR_minus_TR"] - target["ur_diff_target"]) > 1.0e-6:
        raise RuntimeError(f"{label} reviewer-payoff difference mismatch")
    if max(pr["foc_residual"], pr["cutoff_residual"]) > 2.0e-10:
        raise RuntimeError(f"{label} PR residual gate failed")
    diag = tr["diagnostics"]
    if diag["status"] != "approximate_candidate" or not all(
        diag["equilibrium_conditions"].values()
    ):
        raise RuntimeError(f"{label} mixed-TR diagnostics failed")
    if max(
        abs(tr["tie_residual"]),
        abs(tr["balance_residual"]),
        abs(tr["G_residual"]),
        abs(tr["UR_residual"]),
        diag["max_support_payoff_gap"],
    ) > 2.0e-9:
        raise RuntimeError(f"{label} mixed-TR residual gate failed")


def main() -> None:
    records = {}
    for label in ("B1", "B2"):
        records[label] = run_case(label)
        case = records[label]
        print(
            f"PASS {label}: TR acceptance={case['TR']['acceptance']:.9f}; "
            f"U_R(PR)-U_R(TR)={case['U_R_PR_minus_TR']:+.9f}"
        )

    registry = {
        "release_version": VERSION,
        "release_date": "2026-08-21",
        "status": "tolerance-qualified numerical reproduction",
        "normalization": COMMON,
        "case_order": ["B1", "B2"],
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "cases": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
