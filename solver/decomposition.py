"""Reviewer-payoff decomposition for finite-support mixed-TR candidates.

For a common-cutoff candidate ``(mu, c)``, the exact identity is

    U_R^PR - U_R^(TR,mix) = S + C + P,

with screening ``S`` (rejecting negative-value cutoff pools), continuation
``C`` (moving each fixed design from the common cutoff to its own pure
cutoff), and design reallocation ``P`` (moving the design distribution to
the PR optimum).  The residual of the identity is reported and must be
numerical zero; every term is a numerical mesh quantity, not a certified
bound.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .general_solver import GRID_DEFAULT, Model, PRSolution

__all__ = [
    "AlignmentGrid",
    "DecompositionDesignRow",
    "ReviewerDecomposition",
    "alignment_grid",
    "reviewer_payoff_decomposition",
]


@dataclass(frozen=True)
class DecompositionDesignRow:
    """Protocol-specific terms entering the decomposition."""

    d: float
    weight: float
    pool_mass: float
    pool_numerator: float
    natural_value: float
    common_cutoff_reviewer_value: float
    fixed_pr_reviewer_value: float
    common_cutoff_scholar_value: float
    fixed_pr_scholar_value: float
    registration_bonus: float


@dataclass(frozen=True)
class ReviewerDecomposition:
    """Exact S + C + P decomposition of ``U_R^PR - U_R^(TR,mix)``."""

    screening: float
    continuation: float
    design_reallocation: float
    direct_pr_minus_tr: float
    decomposition_residual: float
    pr_solution: PRSolution
    tr_reviewer_payoff: float
    fixed_behavior_disclosure_payoff: float
    design_rows: Tuple[DecompositionDesignRow, ...]
    pr_points: int


@dataclass(frozen=True)
class AlignmentGrid:
    """Objects used to search for branch-local incentive alignment."""

    d: np.ndarray
    common_cutoff_scholar_value: np.ndarray
    fixed_pr_scholar_value: np.ndarray
    registration_bonus: np.ndarray
    fixed_pr_reviewer_value: np.ndarray


def reviewer_payoff_decomposition(
    model: Model,
    mixture,
    cutoff: float,
    *,
    pr_points: int = 4001,
    d_max: Optional[float] = None,
) -> ReviewerDecomposition:
    """Compute the exact S + C + P decomposition for one candidate.

    ``mixture`` is any object exposing sorted ``support`` and normalized
    ``weights`` tuples (``mixed_tr_solver.DesignMixture`` in practice).
    """

    if not math.isfinite(float(cutoff)):
        raise ValueError("cutoff must be finite")
    if pr_points < 3:
        raise ValueError("pr_points must be at least three")
    upper = model.d_bar() if d_max is None else float(d_max)
    if not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("d_max must be finite and positive")

    rows: List[DecompositionDesignRow] = []
    for d, weight in zip(mixture.support, mixture.weights):
        d0 = float(d)
        mass, numerator = model.pool_terms(cutoff, d0)
        natural = model.natural_value(cutoff, d0)
        common_scholar = model.U_S(d0, cutoff)
        fixed_pr_scholar = model.pr_objective(d0)
        fixed_pr_reviewer = model.U_R(d0, model.c_star(d0))
        rows.append(
            DecompositionDesignRow(
                d=d0,
                weight=float(weight),
                pool_mass=float(mass),
                pool_numerator=float(numerator),
                natural_value=float(natural),
                common_cutoff_reviewer_value=float(natural + numerator),
                fixed_pr_reviewer_value=float(fixed_pr_reviewer),
                common_cutoff_scholar_value=float(common_scholar),
                fixed_pr_scholar_value=float(fixed_pr_scholar),
                registration_bonus=float(
                    fixed_pr_scholar - common_scholar
                ),
            )
        )

    weights = np.asarray([row.weight for row in rows], dtype=float)
    numerators = np.asarray(
        [row.pool_numerator for row in rows], dtype=float
    )
    naturals = np.asarray(
        [row.natural_value for row in rows], dtype=float
    )
    fixed_reviewer = np.asarray(
        [row.fixed_pr_reviewer_value for row in rows], dtype=float
    )

    pr = model.solve_PR(n=pr_points, d_max=upper)
    tr_payoff = float(weights @ (naturals + numerators))
    disclosed = float(
        weights @ (naturals + np.maximum(numerators, 0.0))
    )
    screening = disclosed - tr_payoff
    continuation = float(
        weights
        @ (fixed_reviewer - naturals - np.maximum(numerators, 0.0))
    )
    reallocation = float(pr.U_R - weights @ fixed_reviewer)
    direct_difference = float(pr.U_R - tr_payoff)
    residual = float(
        direct_difference - screening - continuation - reallocation
    )
    return ReviewerDecomposition(
        screening=float(screening),
        continuation=continuation,
        design_reallocation=reallocation,
        direct_pr_minus_tr=direct_difference,
        decomposition_residual=residual,
        pr_solution=pr,
        tr_reviewer_payoff=tr_payoff,
        fixed_behavior_disclosure_payoff=disclosed,
        design_rows=tuple(rows),
        pr_points=pr_points,
    )


def alignment_grid(
    model: Model,
    cutoff: float,
    *,
    n: int = GRID_DEFAULT,
    d_max: Optional[float] = None,
) -> AlignmentGrid:
    """Evaluate registration bonuses and reviewer values on a design grid."""

    if n < 3:
        raise ValueError("n must be at least three")
    upper = model.d_bar() if d_max is None else float(d_max)
    d = np.linspace(0.0, upper, n)
    common = model.U_S_vec(d, cutoff)
    fixed_scholar = np.asarray(
        [model.pr_objective(float(value)) for value in d], dtype=float
    )
    fixed_reviewer = np.asarray(
        [
            model.U_R(float(value), model.c_star(float(value)))
            for value in d
        ],
        dtype=float,
    )
    return AlignmentGrid(
        d=d,
        common_cutoff_scholar_value=common,
        fixed_pr_scholar_value=fixed_scholar,
        registration_bonus=fixed_scholar - common,
        fixed_pr_reviewer_value=fixed_reviewer,
    )
