"""Finite-support common-cutoff mixed-TR search over a general type prior.

This module is an outer equilibrium-search layer over :mod:`general_solver`.
The fixed-design model remains the only owner of prior integration, reviewer
terms, scholar payoffs, and global best responses.

The search class is deliberately narrower than a general mixed PBE:

* the scholar's design distribution has finite support;
* the reviewer uses one common manuscript cutoff;
* two-point candidates are constructed at separated global-best-response ties;
* equilibrium conditions are evaluated numerically under the model's
  ``Conventions``; the null-window completion, tie rule, and mass threshold
  all come from ``model.conventions``.

Passing diagnostics identify a tolerance-qualified candidate.  They are not an
analytic or interval-certified existence result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .conventions import TieRule
from .decomposition import ReviewerDecomposition, reviewer_payoff_decomposition
from .general_solver import (
    GRID_DEFAULT,
    Model,
    comparison_tolerance,
    payoff_scale,
    refine_grid_maxima,
    refine_grid_minima,
)

__all__ = [
    "AggregateCutoffResult",
    "BranchTie",
    "MixedTRDiagnostic",
    "MixedTRSearchResult",
    "MixedTRSolver",
    "DesignDiagnostic",
    "DesignMixture",
    "RejectionReason",
]


@dataclass(frozen=True)
class AggregateCutoffResult:
    """Diagnostic cutoff of the aggregate pooling-payoff calculation."""

    value: float
    bracket_low: float
    bracket_high: float
    payoff_low: float
    payoff_high: float
    pool_mass: float
    pool_numerator: float
    completion_used: bool
    upper_set_on_mesh: bool
    scan_low: float
    scan_high: float
    scan_points: int
    status: str


@dataclass(frozen=True)
class DesignMixture:
    """A normalized finite-support probability measure over designs."""

    support: Tuple[float, ...]
    weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        support = np.asarray(self.support, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if support.ndim != 1 or support.size == 0:
            raise ValueError("support must be a nonempty one-dimensional vector")
        if weights.shape != support.shape:
            raise ValueError("support and weights must have the same shape")
        if np.any(~np.isfinite(support)) or np.any(support < 0.0):
            raise ValueError("design support must be finite and nonnegative")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("mixture weights must be finite and strictly positive")
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("mixture weights must have positive finite mass")
        order = np.argsort(support)
        support = support[order]
        weights = weights[order] / total
        if support.size > 1 and np.any(np.diff(support) <= 0.0):
            raise ValueError("design support points must be distinct")
        object.__setattr__(self, "support", tuple(float(x) for x in support))
        object.__setattr__(self, "weights", tuple(float(x) for x in weights))


@dataclass(frozen=True)
class RejectionReason:
    """Structured record of one rejected search interval.

    ``code`` is one of ``"invalid_split"``, ``"no_tie_bracket"``,
    ``"balance_failed"``, ``"evaluation_error"``, or
    ``"below_separation_floor"``. The last code identifies a hull movement
    larger than two design-grid steps but below the branch-separation floor.
    """

    interval: Tuple[float, float]
    code: str
    message: str


@dataclass(frozen=True)
class DesignDiagnostic:
    d: float
    weight: float
    pool_mass: float
    pool_numerator: float
    scholar_payoff: float
    global_best_response_gap: float
    pure_cutoff: float
    on_path_ok: str


@dataclass(frozen=True)
class BranchTie:
    cutoff: float
    d_left: float
    d_right: float
    value_left: float
    value_right: float
    split: float
    manuscript_bracket: Tuple[float, float]


@dataclass(frozen=True)
class MixedTRDiagnostic:
    """Numerical equilibrium checks for one common-cutoff candidate."""

    mixture: DesignMixture
    cutoff: float
    y_min: float
    cutoff_pool_mass: float
    cutoff_pool_payoff: float
    balance_residual: float
    common_gap_max_score: float
    common_gap_argmax: float
    common_gap_max_payoff: float
    common_gap_payoff_argmax: float
    gap_endpoint_value: float
    low_boundary_payoff: float
    cutoff_face_value_payoff: float
    best_response_value: float
    max_support_payoff_gap: float
    design_diagnostics: Tuple[DesignDiagnostic, ...]
    on_path_rejection: bool
    support_global_optimality: bool
    cutoff_pool_acceptable: bool
    common_gap_rejected: bool
    lower_boundary_rejected: bool
    monotone_reviewer_utility_ok: bool
    upper_set_on_mesh: bool
    reviewer_cutoff_response_on_mesh: bool
    tie_points_below_cutoff: int
    theorem_scope_ok: bool
    completion_used_in_gap: bool
    gap_points: int
    best_response_points: int
    condition_tolerance: float
    status: str
    decomposition: Optional[ReviewerDecomposition] = None

    @property
    def approximate_candidate(self) -> bool:
        return self.status in {
            "approximate_candidate",
            "approximate_candidate_outside_proved_subclass",
        }

@dataclass
class MixedTRSearchResult:
    candidates: List[MixedTRDiagnostic]
    ties: List[BranchTie]
    manuscript_grid: np.ndarray
    selected_design_grid: np.ndarray
    pure_cutoff_range: Tuple[float, float]
    rejected_intervals: List[RejectionReason]
    manuscript_points: int
    best_response_points: int
    branch_points: int
    separation: float
    status: str


class MixedTRSolver:
    """Search for two-point common-cutoff mixed-TR candidates."""

    def __init__(self, model: Model) -> None:
        self.model = model

    def _u_scalar(self, q: float) -> float:
        if self.model.P.u is None:
            return float(q - self.model.P.r)
        return float(
            np.asarray(
                self.model.P.u(np.asarray([q], dtype=float)),
                dtype=float,
            )[0]
        )

    def _mass_tol(self, mass_tolerance: Optional[float]) -> float:
        """Resolve the null-window threshold from the model conventions."""

        if mass_tolerance is None:
            return float(self.model.conventions.mass_tol)
        return float(mass_tolerance)

    def null_value(self, c: float, mixture: DesignMixture) -> float:
        """Aggregated completion value at a null-window manuscript.

        Mixes the per-design ``Model.null_window_value`` with the mixture
        weights; under the face-value completion this equals ``u(c)``.
        """

        return float(
            sum(
                weight * self.model.null_window_value(c, d)
                for d, weight in zip(mixture.support, mixture.weights)
            )
        )

    def aggregate_terms(
        self, c: float, mixture: DesignMixture
    ) -> Tuple[float, float]:
        """Return aggregate pooling mass and reviewer-payoff numerator."""

        mass = 0.0
        numerator = 0.0
        for d, weight in zip(mixture.support, mixture.weights):
            component_mass, component_numerator = self.model.pool_terms(c, d)
            mass += weight * component_mass
            numerator += weight * component_numerator
        return float(mass), float(numerator)

    def reviewer_value(
        self,
        c: float,
        mixture: DesignMixture,
        *,
        mass_tolerance: Optional[float] = None,
    ) -> Tuple[float, bool]:
        """Return the aggregate conditional payoff and completion-use flag.

        At a null posterior window (aggregate mass at or below the
        conventions' ``mass_tol``) the value is the weight-aggregated
        completion value ``sum_j w_j Model.null_window_value(c, d_j)``.
        ``mass_tolerance`` defaults to ``model.conventions.mass_tol``.
        """

        mass, numerator = self.aggregate_terms(c, mixture)
        if mass > self._mass_tol(mass_tolerance):
            return float(numerator / mass), False
        return self.null_value(c, mixture), True

    def aggregate_cutoff_result(
        self,
        mixture: DesignMixture,
        *,
        scan_points: int = 4001,
        mass_tolerance: Optional[float] = None,
        condition_tolerance: float = 1e-9,
    ) -> AggregateCutoffResult:
        """Locate the first aggregate pooling-payoff acceptance boundary.

        For a general prior, the calculation applies the selected face-value
        completion whenever the aggregate posterior window is null.

        The method scans before bisecting and reports whether the sampled
        accepted set is an upper set.  It is a diagnostic calculation, not a
        substitute for the full common-cutoff equilibrium checks.
        """

        if scan_points < 3:
            raise ValueError("scan_points must be at least three")
        component_cutoffs = np.asarray(
            [self.model.c_star(d) for d in mixture.support],
            dtype=float,
        )
        finite = component_cutoffs[np.isfinite(component_cutoffs)]
        y_max = max(self.model.P.Y(d) for d in mixture.support)
        if finite.size:
            lo = float(finite.min() - y_max - 5.0)
            hi = float(finite.max() + y_max + 5.0)
        else:
            lo = float(self.model.P.r - y_max - 5.0)
            hi = float(self.model.P.r + y_max + 5.0)
        step = max(1.0, hi - lo)

        for _ in range(60):
            value, _ = self.reviewer_value(
                lo, mixture, mass_tolerance=mass_tolerance
            )
            if value < 0.0:
                break
            lo -= step
            step *= 1.5
        else:
            return AggregateCutoffResult(
                value=-math.inf,
                bracket_low=-math.inf,
                bracket_high=-math.inf,
                payoff_low=float(value),
                payoff_high=float(value),
                pool_mass=0.0,
                pool_numerator=0.0,
                completion_used=False,
                upper_set_on_mesh=True,
                scan_low=-math.inf,
                scan_high=hi,
                scan_points=scan_points,
                status="all_accept",
            )

        step = max(1.0, hi - lo)
        for _ in range(60):
            value, _ = self.reviewer_value(
                hi, mixture, mass_tolerance=mass_tolerance
            )
            if value >= 0.0:
                break
            hi += step
            step *= 1.5
        else:
            return AggregateCutoffResult(
                value=math.inf,
                bracket_low=math.inf,
                bracket_high=math.inf,
                payoff_low=float(value),
                payoff_high=float(value),
                pool_mass=0.0,
                pool_numerator=0.0,
                completion_used=False,
                upper_set_on_mesh=True,
                scan_low=lo,
                scan_high=math.inf,
                scan_points=scan_points,
                status="all_reject",
            )

        grid = np.linspace(lo, hi, scan_points)
        values = np.asarray(
            [
                self.reviewer_value(
                    float(c),
                    mixture,
                    mass_tolerance=mass_tolerance,
                )[0]
                for c in grid
            ],
            dtype=float,
        )
        accepted = values >= 0.0
        accepted_index = np.flatnonzero(accepted)
        if accepted_index.size == 0:
            raise RuntimeError("aggregate cutoff scan found no accepted manuscript")
        first = int(accepted_index[0])
        if first == 0:
            raise RuntimeError(
                "aggregate cutoff scan began inside the accepted set"
            )
        a = float(grid[first - 1])
        b = float(grid[first])
        for _ in range(200):
            tolerance = 1e-12 + 1e-12 * max(1.0, abs(a), abs(b))
            if b - a <= tolerance:
                break
            midpoint = 0.5 * (a + b)
            midpoint_value = self.reviewer_value(
                midpoint,
                mixture,
                mass_tolerance=mass_tolerance,
            )[0]
            if midpoint_value >= 0.0:
                b = midpoint
            else:
                a = midpoint

        cutoff = float(b)
        mass, numerator = self.aggregate_terms(cutoff, mixture)
        cutoff_payoff, completion_used = self.reviewer_value(
            cutoff,
            mixture,
            mass_tolerance=mass_tolerance,
        )
        payoff_low = self.reviewer_value(
            a,
            mixture,
            mass_tolerance=mass_tolerance,
        )[0]
        above_grid = np.linspace(cutoff, hi, scan_points)
        above_values = np.asarray(
            [
                self.reviewer_value(
                    float(c),
                    mixture,
                    mass_tolerance=mass_tolerance,
                )[0]
                for c in above_grid
            ],
            dtype=float,
        )
        upper_set = bool(np.all(above_values >= -condition_tolerance))
        status = (
            "finite_upper_set_on_mesh"
            if upper_set
            else "finite_non_upper_set_on_mesh"
        )
        return AggregateCutoffResult(
            value=cutoff,
            bracket_low=float(a),
            bracket_high=float(b),
            payoff_low=float(payoff_low),
            payoff_high=float(cutoff_payoff),
            pool_mass=float(mass),
            pool_numerator=float(numerator),
            completion_used=bool(completion_used),
            upper_set_on_mesh=upper_set,
            scan_low=float(lo),
            scan_high=float(hi),
            scan_points=scan_points,
            status=status,
        )

    def balance_weight(
        self, d_first: float, d_second: float, cutoff: float
    ) -> float:
        """Return the weight on ``d_first`` implied by aggregate balance.

        With ``n_j = n_c(d_j)``, the solution to

        ``w*n_first + (1-w)*n_second = 0``

        is ``w = -n_second / (n_first - n_second)``.  The design ordering
        carries no sign assumption, so the formula covers both the normal and
        binary controls.
        """

        if not 0.0 <= d_first < d_second:
            raise ValueError("require 0 <= d_first < d_second")
        n_first = self.model.pool_terms(cutoff, d_first)[1]
        n_second = self.model.pool_terms(cutoff, d_second)[1]
        denominator = n_first - n_second
        if abs(denominator) <= 1e-15:
            raise ValueError("pool numerators do not identify a mixing weight")
        weight = -n_second / denominator
        if not 0.0 < weight < 1.0:
            raise ValueError(
                "aggregate balance requires opposite-sign component numerators; "
                f"got {n_first:.12g} and {n_second:.12g}"
            )
        return float(weight)

    def evaluate_candidate(
        self,
        mixture: DesignMixture,
        cutoff: float,
        *,
        gap_points: int = 2001,
        best_response_points: int = 8001,
        d_max: Optional[float] = None,
        condition_tolerance: float = 1e-7,
        mass_tolerance: Optional[float] = None,
        assume_monotone_u: bool = False,
        assume_regular_custom_primitives: bool = False,
        decompose: bool = False,
    ) -> MixedTRDiagnostic:
        """Evaluate a proposed finite-support common-cutoff equilibrium.

        Gap rejection is an open-interval condition. It is evaluated on a
        band-qualified mesh that stays one mesh step inside the lower
        boundary and an explicit band ``eta`` inside the cutoff end, with
        the value at ``cutoff - eta`` reported as ``gap_endpoint_value``
        (a trend indicator).  ``mass_tolerance`` defaults to
        ``model.conventions.mass_tol``; the tie rule comes from
        ``model.conventions.tie_rule``.  With ``decompose=True`` the exact
        S + C + P reviewer-payoff decomposition is attached (slow: it
        solves the PR benchmark).

        Units of ``condition_tolerance`` by check:

        ============================  =================================
        balance residual              aggregate-numerator units
        gap/boundary/reviewer checks  payoff units (numerator / mass)
        ``common_gap_max_score``      numerator units (auxiliary only)
        ============================  =================================

        ``low_boundary_payoff`` and ``cutoff_face_value_payoff`` are
        Bayes-forced on-path face-value objects. They are ``u(c)`` at
        ``cutoff - y_min`` and at ``cutoff``.
        """

        if not math.isfinite(float(cutoff)):
            raise ValueError("cutoff must be finite")
        if gap_points < 3 or best_response_points < 3:
            raise ValueError("diagnostic meshes must contain at least three points")
        mass_tol = self._mass_tol(mass_tolerance)
        if condition_tolerance <= 0.0 or mass_tol < 0.0:
            raise ValueError("diagnostic tolerances must be nonnegative")

        upper = self.model.d_bar() if d_max is None else float(d_max)
        response = self.model.best_response_details(
            cutoff, d_max=upper, n=best_response_points
        )
        payoff_tolerance = max(
            condition_tolerance,
            response.comparison_tolerance,
            comparison_tolerance(
                response.value,
                variation=payoff_scale(self.model),
            ),
        )

        y_values = np.asarray(
            [self.model.P.Y(d) for d in mixture.support],
            dtype=float,
        )
        y_min = float(y_values.min())
        if not y_min > 0.0:
            raise ValueError("all support designs must have positive Y(d)")

        # Under the submit-natural convention, every charged on-path
        # rejected natural manuscript must be weakly unacceptable at face value.
        # The simple sufficient test compares the sup of charged naturals at
        # or below cutoff - Y(d) against r.  Unknown support fails closed to
        # the full-support bound.
        try:
            support_fn = getattr(
                self.model.F, "theta_support_intervals", None
            )
            support_intervals = (
                support_fn() if callable(support_fn) else None
            )
        except NotImplementedError:
            support_intervals = None
        if support_intervals is None:
            support_intervals = [(-math.inf, math.inf)]

        def on_path_verdict(d: float, y_of_d: float) -> str:
            bound = cutoff - y_of_d
            charged_sup = -math.inf
            for theta_lo, theta_hi in support_intervals:
                natural_lo = self.model.P.alpha * d + float(theta_lo)
                natural_hi = self.model.P.alpha * d + float(theta_hi)
                if natural_lo > bound:
                    continue
                charged_sup = max(charged_sup, min(natural_hi, bound))
            if charged_sup == -math.inf:
                return "vacuous_null"
            # Rejecting the highest charged natural manuscript is optimal
            # iff u(charged_sup) <= 0. Evaluate u directly so a custom
            # increasing payoff need not have its zero at r.
            if self._u_scalar(charged_sup) <= condition_tolerance:
                return "satisfied_charged"
            return "violated"

        component_rows: List[DesignDiagnostic] = []
        component_masses: List[float] = []
        component_numerators: List[float] = []
        support_gaps: List[float] = []
        for d, weight, y_of_d in zip(
            mixture.support, mixture.weights, y_values
        ):
            mass, numerator = self.model.pool_terms(cutoff, d)
            payoff = self.model.U_S(d, cutoff)
            gap = max(0.0, response.value - payoff)
            component_masses.append(mass)
            component_numerators.append(numerator)
            support_gaps.append(gap)
            component_rows.append(
                DesignDiagnostic(
                    d=float(d),
                    weight=float(weight),
                    pool_mass=float(mass),
                    pool_numerator=float(numerator),
                    scholar_payoff=float(payoff),
                    global_best_response_gap=float(gap),
                    pure_cutoff=float(self.model.c_star(d)),
                    on_path_ok=on_path_verdict(float(d), float(y_of_d)),
                )
            )
        on_path_rejection = bool(
            all(row.on_path_ok != "violated" for row in component_rows)
        )

        weights = np.asarray(mixture.weights, dtype=float)
        masses = np.asarray(component_masses, dtype=float)
        numerators = np.asarray(component_numerators, dtype=float)
        cutoff_mass = float(weights @ masses)
        balance = float(weights @ numerators)
        cutoff_pool_payoff, cutoff_completion_used = self.reviewer_value(
            cutoff,
            mixture,
            mass_tolerance=mass_tol,
        )

        completion_used = False

        def gap_score(z: float) -> float:
            nonlocal completion_used
            mass, numerator = self.aggregate_terms(z, mixture)
            if mass > mass_tol:
                return float(numerator)
            completion_used = True
            return self.null_value(z, mixture)

        def gap_payoff(z: float) -> float:
            nonlocal completion_used
            value, used = self.reviewer_value(
                z,
                mixture,
                mass_tolerance=mass_tol,
            )
            completion_used = bool(completion_used or used)
            return float(value)

        # Band-qualified open-interval mesh: one mesh step inside the lower
        # boundary, an explicit band eta inside the cutoff end.
        mesh_step = y_min / (gap_points + 1)
        eta = max(2.0 * mesh_step, 1e-9)
        gap_grid = np.linspace(
            cutoff - y_min + mesh_step,
            cutoff - eta,
            gap_points,
            dtype=float,
        )
        gap_values = np.asarray(
            [gap_score(float(z)) for z in gap_grid],
            dtype=float,
        )
        gap_candidates = refine_grid_maxima(
            gap_score,
            gap_grid,
            gap_values,
        )
        gap_argmax, gap_max = max(
            gap_candidates, key=lambda item: item[1]
        )
        gap_payoff_values = np.asarray(
            [gap_payoff(float(z)) for z in gap_grid],
            dtype=float,
        )
        gap_payoff_candidates = refine_grid_maxima(
            gap_payoff,
            gap_grid,
            gap_payoff_values,
        )
        gap_payoff_argmax, gap_payoff_max = max(
            gap_payoff_candidates, key=lambda item: item[1]
        )
        gap_endpoint_value = float(gap_values[-1])

        # These are Bayes-forced on-path face-value objects under the
        # submit-natural convention, so do not route them through completion.
        low_payoff = self._u_scalar(cutoff - y_min)
        cutoff_face_value = self._u_scalar(cutoff)
        max_support_gap = float(max(support_gaps))
        monotone_u_ok = bool(self.model.P.u is None or assume_monotone_u)

        # The reviewer gate includes an above-cutoff acceptance check.
        # Monotonicity of the aggregate G is not guaranteed for
        # mixtures even with linear u, so the mesh always runs.
        y_max = float(y_values.max())
        upper_grid = np.linspace(
            cutoff, cutoff + 2.0 * y_max + 5.0, 400
        )

        def upper_value(z: float) -> float:
            return float(
                self.reviewer_value(
                    z, mixture, mass_tolerance=mass_tol
                )[0]
            )

        upper_values = np.asarray(
            [upper_value(float(z)) for z in upper_grid], dtype=float
        )
        refined_minima = refine_grid_minima(
            upper_value, upper_grid, upper_values
        )
        upper_set_ok = bool(
            np.all(upper_values >= -condition_tolerance)
            and all(
                value >= -condition_tolerance
                for _, value in refined_minima
            )
        )

        # Under ALL_ACCEPT, indifference in the gap implies acceptance, so
        # the gap conditions become strict and any detected tie fails closed.
        strict_ties = (
            self.model.conventions.tie_rule is TieRule.ALL_ACCEPT
        )
        rejection_bound = (
            -condition_tolerance if strict_ties else condition_tolerance
        )
        if strict_ties:
            # Count near-zero values and strict sign changes. Each crossing
            # pair contains an off-mesh zero that a pointwise scan would miss.
            near_zero = (
                np.abs(gap_payoff_values)
                <= self.model.conventions.tie_tol
            )
            crossings = (
                gap_payoff_values[:-1] * gap_payoff_values[1:] < 0.0
            )
            tie_points = int(np.sum(near_zero) + np.sum(crossings))
        else:
            tie_points = 0

        support_global_optimality = bool(max_support_gap <= payoff_tolerance)
        cutoff_pool_acceptable = bool(
            cutoff_mass > mass_tol
            and abs(balance) <= condition_tolerance
            and cutoff_face_value >= -condition_tolerance
        )
        # Decide gap rejection on the payoff scale. The numerator-scale score
        # remains available as an auxiliary diagnostic.
        common_gap_rejected = bool(gap_payoff_max <= rejection_bound)
        lower_boundary_rejected = bool(low_payoff <= rejection_bound)
        reviewer_cutoff_response = bool(
            cutoff_mass > mass_tol
            and not cutoff_completion_used
            and cutoff_pool_payoff >= -condition_tolerance
            and gap_payoff_max <= rejection_bound
            and low_payoff <= rejection_bound
            and cutoff_face_value >= -condition_tolerance
            and upper_set_ok
            and monotone_u_ok
        )

        capabilities = self.model.F.capabilities
        regular_technology_ok = bool(
            (
                self.model.P.delta is None
                and self.model.P.C0 is None
            )
            or assume_regular_custom_primitives
        )
        theorem_scope_ok = bool(
            capabilities.full_support
            and not capabilities.has_atoms
            and capabilities.continuous_positive_density
            and monotone_u_ok
            and regular_technology_ok
        )
        all_conditions = bool(
            all(weight > 0.0 for weight in mixture.weights)
            and on_path_rejection
            and support_global_optimality
            and cutoff_pool_acceptable
            and common_gap_rejected
            and lower_boundary_rejected
            and reviewer_cutoff_response
            and tie_points == 0
        )
        if not all_conditions:
            status = "failed_diagnostics"
        elif theorem_scope_ok:
            status = "approximate_candidate"
        else:
            status = "approximate_candidate_outside_proved_subclass"

        decomposition = (
            reviewer_payoff_decomposition(
                self.model, mixture, cutoff, d_max=upper
            )
            if decompose
            else None
        )

        return MixedTRDiagnostic(
            mixture=mixture,
            cutoff=float(cutoff),
            y_min=y_min,
            cutoff_pool_mass=cutoff_mass,
            cutoff_pool_payoff=float(cutoff_pool_payoff),
            balance_residual=balance,
            common_gap_max_score=float(gap_max),
            common_gap_argmax=float(gap_argmax),
            common_gap_max_payoff=float(gap_payoff_max),
            common_gap_payoff_argmax=float(gap_payoff_argmax),
            gap_endpoint_value=gap_endpoint_value,
            low_boundary_payoff=float(low_payoff),
            cutoff_face_value_payoff=float(cutoff_face_value),
            best_response_value=float(response.value),
            max_support_payoff_gap=max_support_gap,
            design_diagnostics=tuple(component_rows),
            on_path_rejection=on_path_rejection,
            support_global_optimality=support_global_optimality,
            cutoff_pool_acceptable=cutoff_pool_acceptable,
            common_gap_rejected=common_gap_rejected,
            lower_boundary_rejected=lower_boundary_rejected,
            monotone_reviewer_utility_ok=monotone_u_ok,
            upper_set_on_mesh=upper_set_ok,
            reviewer_cutoff_response_on_mesh=reviewer_cutoff_response,
            tie_points_below_cutoff=tie_points,
            theorem_scope_ok=theorem_scope_ok,
            completion_used_in_gap=completion_used,
            gap_points=gap_points,
            best_response_points=best_response_points,
            condition_tolerance=float(condition_tolerance),
            status=status,
            decomposition=decomposition,
        )

    def evaluate_two_point_candidate(
        self,
        d_first: float,
        d_second: float,
        cutoff: float,
        **kwargs,
    ) -> MixedTRDiagnostic:
        weight_first = self.balance_weight(d_first, d_second, cutoff)
        mixture = DesignMixture(
            support=(d_first, d_second),
            weights=(weight_first, 1.0 - weight_first),
        )
        return self.evaluate_candidate(mixture, cutoff, **kwargs)

    def _branch_maximum(
        self,
        c: float,
        lo: float,
        hi: float,
        points: int,
    ) -> Tuple[float, float]:
        if points < 3 or not 0.0 <= lo < hi:
            raise ValueError("branch search requires 0 <= lo < hi")
        grid = np.linspace(lo, hi, points)
        values = self.model.U_S_vec(grid, c)
        candidates = refine_grid_maxima(
            lambda d: self.model.U_S(d, c),
            grid,
            values,
        )
        return max(candidates, key=lambda item: item[1])

    @staticmethod
    def _bisect_tie(
        gap: Callable[[float], float],
        lo: float,
        hi: float,
        iterations: int = 70,
    ) -> float:
        f_lo = float(gap(lo))
        f_hi = float(gap(hi))
        if f_lo == 0.0:
            return float(lo)
        if f_hi == 0.0:
            return float(hi)
        if (f_lo > 0.0) == (f_hi > 0.0):
            raise ValueError("branch-value difference does not bracket a tie")
        for _ in range(iterations):
            midpoint = 0.5 * (lo + hi)
            f_mid = float(gap(midpoint))
            if f_mid == 0.0:
                return float(midpoint)
            if (f_mid > 0.0) == (f_lo > 0.0):
                lo, f_lo = midpoint, f_mid
            else:
                hi, f_hi = midpoint, f_mid
        return float(0.5 * (lo + hi))

    def search_two_point(
        self,
        *,
        d_max: Optional[float] = None,
        pure_cutoff_points: int = 81,
        manuscript_points: int = 161,
        best_response_points: int = GRID_DEFAULT,
        branch_points: int = 1201,
        gap_points: int = 1201,
        condition_tolerance: float = 2e-6,
        branch_separation: Optional[float] = None,
    ) -> MixedTRSearchResult:
        """Locate and evaluate two-point candidates at global branch ties.

        The manuscript grid scans only the finite pure-cutoff range
        ``[c_min, c_max]``; ties outside that range are not searched.
        Branch jumps are detected on the argmax hull pair
        against the separation floor
        ``max(10 * d_max / (best_response_points - 1),
        0.02 * max(1, d_max))`` unless ``branch_separation`` overrides it.
        Hull movements above two design-grid steps but below
        the floor are recorded as ``below_separation_floor`` rejections
        rather than silently dropped.
        """

        if min(
            pure_cutoff_points,
            manuscript_points,
            best_response_points,
            branch_points,
            gap_points,
        ) < 3:
            raise ValueError("all search meshes must contain at least three points")
        upper = self.model.d_bar() if d_max is None else float(d_max)
        if not math.isfinite(upper) or upper <= 0.0:
            raise ValueError("d_max must be finite and positive")
        design_spacing = upper / (best_response_points - 1)
        separation = (
            max(10.0 * design_spacing, 0.02 * max(1.0, upper))
            if branch_separation is None
            else float(branch_separation)
        )

        design_grid = np.linspace(0.0, upper, pure_cutoff_points)
        pure_cutoffs = np.asarray(
            [self.model.c_star(float(d)) for d in design_grid],
            dtype=float,
        )
        finite = pure_cutoffs[np.isfinite(pure_cutoffs)]
        if finite.size == 0:
            return MixedTRSearchResult(
                candidates=[],
                ties=[],
                manuscript_grid=np.empty(0),
                selected_design_grid=np.empty(0),
                pure_cutoff_range=(math.nan, math.nan),
                rejected_intervals=[
                    RejectionReason(
                        interval=(math.nan, math.nan),
                        code="evaluation_error",
                        message="no finite pure-design cutoff",
                    )
                ],
                manuscript_points=manuscript_points,
                best_response_points=best_response_points,
                branch_points=branch_points,
                separation=separation,
                status="no_finite_cutoff_range",
            )
        c_min = float(finite.min())
        c_max = float(finite.max())
        if c_max <= c_min:
            padding = max(1.0, abs(c_min) * 0.1)
            c_min -= padding
            c_max += padding
        manuscript_grid = np.linspace(c_min, c_max, manuscript_points)

        responses = [
            self.model.best_response_details(
                float(c), d_max=upper, n=best_response_points
            )
            for c in manuscript_grid
        ]
        # Detect jumps on the argmax hull pair, not the hull midpoint;
        # the midpoint series is kept for reporting and plotting only.
        d_min_series = np.asarray(
            [response.d_min for response in responses], dtype=float
        )
        d_max_series = np.asarray(
            [response.d_max for response in responses], dtype=float
        )
        selected = 0.5 * (d_min_series + d_max_series)
        jump_magnitude = np.maximum(
            np.abs(np.diff(d_min_series)),
            np.abs(np.diff(d_max_series)),
        )
        jump_index = np.flatnonzero(jump_magnitude > separation)

        ties: List[BranchTie] = []
        candidates: List[MixedTRDiagnostic] = []
        rejected: List[RejectionReason] = []
        # Record near-threshold hull movements so a sub-floor
        # branch tie is distinguishable from no tie at all.
        for index in np.flatnonzero(
            (jump_magnitude > 2.0 * design_spacing)
            & (jump_magnitude < separation)
        ):
            rejected.append(
                RejectionReason(
                    interval=(
                        float(manuscript_grid[index]),
                        float(manuscript_grid[index + 1]),
                    ),
                    code="below_separation_floor",
                    message=(
                        f"hull jump {float(jump_magnitude[index]):.8g} "
                        f"below separation floor {separation:.8g}"
                    ),
                )
            )
        for index in jump_index:
            c_lo = float(manuscript_grid[index])
            c_hi = float(manuscript_grid[index + 1])
            # Union-hull midpoint over the interval.  With one branch per
            # endpoint this equals 0.5*(d_min_before + d_max_after); when a
            # mesh manuscript lands on the tie itself (wide hull at one
            # endpoint) it still separates the two branches, which the
            # literal before/after pair does not.
            split = 0.5 * (
                min(
                    float(d_min_series[index]),
                    float(d_min_series[index + 1]),
                )
                + max(
                    float(d_max_series[index]),
                    float(d_max_series[index + 1]),
                )
            )
            if not 0.0 < split < upper:
                rejected.append(
                    RejectionReason(
                        interval=(c_lo, c_hi),
                        code="invalid_split",
                        message=f"invalid branch split {split:.8g}",
                    )
                )
                continue

            def branch_gap(c: float) -> float:
                left = self._branch_maximum(
                    c, 0.0, split, branch_points
                )[1]
                right = self._branch_maximum(
                    c, split, upper, branch_points
                )[1]
                return float(right - left)

            try:
                cutoff = self._bisect_tie(branch_gap, c_lo, c_hi)
            except ValueError:
                rejected.append(
                    RejectionReason(
                        interval=(c_lo, c_hi),
                        code="no_tie_bracket",
                        message="branch values do not bracket a tie",
                    )
                )
                continue

            d_left, value_left = self._branch_maximum(
                cutoff, 0.0, split, branch_points
            )
            d_right, value_right = self._branch_maximum(
                cutoff, split, upper, branch_points
            )
            if d_right < d_left:
                d_left, d_right = d_right, d_left
                value_left, value_right = value_right, value_left

            tie = BranchTie(
                cutoff=float(cutoff),
                d_left=float(d_left),
                d_right=float(d_right),
                value_left=float(value_left),
                value_right=float(value_right),
                split=float(split),
                manuscript_bracket=(c_lo, c_hi),
            )
            if any(
                abs(tie.cutoff - prior.cutoff)
                <= 1e-7 * max(1.0, abs(tie.cutoff))
                for prior in ties
            ):
                continue
            ties.append(tie)

            try:
                weight_left = self.balance_weight(
                    tie.d_left, tie.d_right, tie.cutoff
                )
            except ValueError as exc:
                rejected.append(
                    RejectionReason(
                        interval=(c_lo, c_hi),
                        code="balance_failed",
                        message=f"c={tie.cutoff:.8g}: {exc}",
                    )
                )
                continue
            try:
                diagnostic = self.evaluate_candidate(
                    DesignMixture(
                        support=(tie.d_left, tie.d_right),
                        weights=(weight_left, 1.0 - weight_left),
                    ),
                    tie.cutoff,
                    gap_points=gap_points,
                    best_response_points=max(
                        best_response_points, branch_points
                    ),
                    d_max=upper,
                    condition_tolerance=condition_tolerance,
                )
            except ValueError as exc:
                rejected.append(
                    RejectionReason(
                        interval=(c_lo, c_hi),
                        code="evaluation_error",
                        message=f"c={tie.cutoff:.8g}: {exc}",
                    )
                )
                continue
            candidates.append(diagnostic)

        if any(candidate.approximate_candidate for candidate in candidates):
            status = "approximate_candidates_found"
        elif candidates:
            status = "candidates_failed_diagnostics"
        elif ties:
            status = "ties_without_balance_candidate"
        else:
            status = "no_branch_tie_on_mesh"

        return MixedTRSearchResult(
            candidates=candidates,
            ties=ties,
            manuscript_grid=manuscript_grid,
            selected_design_grid=selected,
            pure_cutoff_range=(c_min, c_max),
            rejected_intervals=rejected,
            manuscript_points=manuscript_points,
            best_response_points=best_response_points,
            branch_points=branch_points,
            separation=separation,
            status=status,
        )
