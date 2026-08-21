"""General-distribution solver for the design-cutoff peer-review model.

This module contains numerical economics only and has no plotting dependency.

The implementation is built around the fixed-cutoff problem.  A prior enters
only through probabilities, moments, and integrals over half-open windows
``(lo, hi]``.  The bundled :class:`Prior` represents finite atoms plus a
piecewise-linear density.  :class:`OraclePrior` admits any probability measure
for which the required window operations are supplied.

The default off-path completion is face value (zero required adjustment), and
the default reviewer threshold is the global cutoff

    inf {c : G(c; d) >= 0}.

All equilibrium outputs are numerical diagnostics.  Grid refinement does not
constitute an existence, non-existence, uniqueness, or global-optimality proof.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .conventions import (
    Conventions,
    TieRule,
)

__all__ = [
    "ARGMAX_XTOL",
    "CUTOFF_RTOL",
    "CUTOFF_XTOL",
    "GRID_DEFAULT",
    "PAYOFF_TIE_TOL",
    "ArgmaxComponent",
    "BestResponseResult",
    "Conventions",
    "CutoffResult",
    "FixedPointCandidate",
    "Model",
    "OraclePrior",
    "PRSolution",
    "Params",
    "Prior",
    "PriorCapabilities",
    "PriorMeasure",
    "SignChange",
    "TRScan",
    "TieRule",
]


CUTOFF_XTOL = 1e-12
CUTOFF_RTOL = 1e-12
ARGMAX_XTOL = 1e-10
# Relative tolerance for scholar-payoff and argmax comparisons. This is
# distinct from ``Conventions.tie_tol``, which detects reviewer indifference.
PAYOFF_TIE_TOL = 1e-9
GRID_DEFAULT = 2001


@dataclass(frozen=True)
class PriorCapabilities:
    """Metadata used to separate computation from theorem scope."""

    representation: str
    has_atoms: bool
    full_support: bool
    continuous_positive_density: bool
    exact_window_moments: bool
    error_bound_available: bool


@runtime_checkable
class PriorMeasure(Protocol):
    """Operations required from a probability measure on scholar types."""

    full_support: bool
    capabilities: PriorCapabilities

    def mass(self, lo: float, hi: float) -> float:
        """Return ``F((lo, hi])``."""

    def moment(self, lo: float, hi: float) -> float:
        """Return ``integral theta dF(theta)`` over ``(lo, hi]``."""

    def cond_mean(self, lo: float, hi: float) -> Optional[float]:
        """Return the conditional mean, or ``None`` when the window is null."""

    def window_moments(
        self, lo: np.ndarray, hi: np.ndarray, center=None
    ) -> np.ndarray:
        """Return mass and first/second window moments about ``center``.

        ``center`` is a scalar or per-window array; ``None`` means raw
        moments about zero.
        """

    def integrate(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        lo: float,
        hi: float,
    ) -> float:
        """Return ``integral fn(theta) dF(theta)`` over ``(lo, hi]``."""


class Prior:
    """Finite atoms plus one piecewise-linear continuous density.

    All interval queries use ``(lo, hi]``.  This convention implements the
    model's tie-break: a type requiring exactly ``Y(d)`` exits, while a type
    requiring zero adjustment remains in the cutoff pool.
    """

    def __init__(
        self,
        atoms: Sequence[Tuple[float, float]] = (),
        grid: Optional[Sequence[float]] = None,
        density: Optional[Sequence[float]] = None,
        full_support: bool = False,
        continuous_positive_density: bool = False,
    ) -> None:
        self.full_support = bool(full_support)
        self.atom_theta = np.asarray([a[0] for a in atoms], dtype=float)
        self.atom_w = np.asarray([a[1] for a in atoms], dtype=float)
        if np.any(~np.isfinite(self.atom_theta)):
            raise ValueError("atom locations must be finite")
        if np.any(~np.isfinite(self.atom_w)):
            raise ValueError("atom weights must be finite")
        if np.any(self.atom_w < 0.0):
            raise ValueError("atom weights must be nonnegative")

        if (grid is None) != (density is None):
            raise ValueError("grid and density must be supplied together")
        if grid is None:
            self.grid = np.empty(0)
            self.dens = np.empty(0)
            self._dens_slope = np.empty(0)
            self._ref = self._support_midpoint()
            self._cmass = np.empty(0)
            self._cmom = np.empty(0)
            self._cmom2 = np.empty(0)
        else:
            self.grid = np.asarray(grid, dtype=float)
            self.dens = np.asarray(density, dtype=float)
            if self.grid.ndim != 1 or self.grid.shape != self.dens.shape:
                raise ValueError("grid and density must be matching 1-D arrays")
            if self.grid.size < 2:
                raise ValueError("continuous grid must contain at least two points")
            if np.any(~np.isfinite(self.grid)) or np.any(~np.isfinite(self.dens)):
                raise ValueError("grid and density must be finite")
            if np.any(np.diff(self.grid) <= 0.0):
                raise ValueError("grid must be strictly increasing")
            if np.any(self.dens < 0.0):
                raise ValueError("density must be nonnegative")

            dx = np.diff(self.grid)
            self._dens_slope = np.diff(self.dens) / dx
            self._ref = self._support_midpoint()
            cell = [
                self._cell_integral(np.arange(dx.size), dx, order)
                for order in range(3)
            ]
            self._cmass = np.concatenate([[0.0], np.cumsum(cell[0])])
            self._cmom = np.concatenate([[0.0], np.cumsum(cell[1])])
            self._cmom2 = np.concatenate([[0.0], np.cumsum(cell[2])])

        continuous_mass = self._cmass[-1] if self._cmass.size else 0.0
        self.total = float(self.atom_w.sum() + continuous_mass)
        if not self.total > 0.0:
            raise ValueError("prior has no positive mass")

        positive_density = bool(
            self.grid.size
            and np.all(self.dens > 0.0)
            and continuous_positive_density
        )
        self.capabilities = PriorCapabilities(
            representation="finite-atoms-plus-piecewise-linear-density",
            has_atoms=bool(self.atom_theta.size),
            full_support=self.full_support,
            continuous_positive_density=positive_density,
            exact_window_moments=True,
            error_bound_available=False,
        )

    @classmethod
    def binary(cls, theta_L: float, theta_H: float, pi_H: float) -> "Prior":
        if not 0.0 <= pi_H <= 1.0:
            raise ValueError("pi_H must lie in [0, 1]")
        return cls(atoms=[(theta_L, 1.0 - pi_H), (theta_H, pi_H)])

    @classmethod
    def from_callable(
        cls,
        f: Callable[[np.ndarray], np.ndarray],
        lo: float,
        hi: float,
        n: int = 200001,
        *,
        full_support: bool = False,
        continuous_positive_density: bool = False,
    ) -> "Prior":
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            raise ValueError("require finite lo < hi")
        if n < 2:
            raise ValueError("n must be at least two")
        grid = np.linspace(lo, hi, n)
        density = np.asarray(f(grid), dtype=float)
        return cls(
            grid=grid,
            density=density,
            full_support=full_support,
            continuous_positive_density=continuous_positive_density,
        )

    @classmethod
    def normal(
        cls,
        mu: float = 0.0,
        sd: float = 1.0,
        half_width: float = 12.0,
        n: int = 200001,
    ) -> "Prior":
        if not math.isfinite(mu) or not math.isfinite(sd) or sd <= 0.0:
            raise ValueError("normal prior requires finite mu and sd > 0")
        if half_width <= 0.0:
            raise ValueError("half_width must be positive")
        grid = np.linspace(mu - half_width * sd, mu + half_width * sd, n)
        density = np.exp(-0.5 * ((grid - mu) / sd) ** 2) / (
            sd * math.sqrt(2.0 * math.pi)
        )
        return cls(
            grid=grid,
            density=density,
            full_support=True,
            continuous_positive_density=True,
        )

    def _support_midpoint(self) -> float:
        """Reference point for internally centered continuous moments.

        Centering the cumulative moment arrays about the support midpoint
        removes the catastrophic cancellation raw moments suffer when the
        support sits far from zero.
        """

        points: List[float] = []
        if self.atom_theta.size:
            points.extend(
                (float(self.atom_theta.min()), float(self.atom_theta.max()))
            )
        if self.grid.size:
            points.extend((float(self.grid[0]), float(self.grid[-1])))
        if not points:
            return 0.0
        return 0.5 * (min(points) + max(points))

    def _cell_integral(self, j, t, order: int):
        x0 = self.grid[j] - self._ref
        f0 = self.dens[j]
        slope = self._dens_slope[j]
        if order == 0:
            return f0 * t + 0.5 * slope * t * t
        if order == 1:
            return (
                x0 * f0 * t
                + 0.5 * (x0 * slope + f0) * t * t
                + (slope / 3.0) * t**3
            )
        if order == 2:
            return (
                x0 * x0 * f0 * t
                + 0.5 * (x0 * x0 * slope + 2.0 * x0 * f0) * t * t
                + ((2.0 * x0 * slope + f0) / 3.0) * t**3
                + 0.25 * slope * t**4
            )
        raise ValueError("only moment orders zero, one, and two are supported")

    def _interp(self, cumulative: np.ndarray, order: int, x: float) -> float:
        if self.grid.size == 0:
            return 0.0
        if x <= self.grid[0]:
            return 0.0
        if x >= self.grid[-1]:
            return float(cumulative[-1])
        j = int(np.searchsorted(self.grid, x, side="right") - 1)
        return float(
            cumulative[j] + self._cell_integral(j, x - self.grid[j], order)
        )

    def _interp_many(
        self,
        cumulative: np.ndarray,
        order: int,
        x: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.grid.size == 0:
            return np.zeros_like(x)
        clipped = np.clip(x, self.grid[0], self.grid[-1])
        j = np.searchsorted(self.grid, clipped, side="right") - 1
        j = np.clip(j, 0, self.grid.size - 2)
        out = cumulative[j] + self._cell_integral(
            j, clipped - self.grid[j], order
        )
        out = np.where(x <= self.grid[0], 0.0, out)
        out = np.where(x >= self.grid[-1], cumulative[-1], out)
        return out

    def mass(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        value = self._interp(self._cmass, 0, hi) - self._interp(
            self._cmass, 0, lo
        )
        if self.atom_theta.size:
            selected = (self.atom_theta > lo) & (self.atom_theta <= hi)
            value += float(self.atom_w[selected].sum())
        return value / self.total

    def moment(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        value = self._interp(self._cmom, 1, hi) - self._interp(
            self._cmom, 1, lo
        )
        if self.grid.size:
            # The cumulative array is centered about _ref; shift back to raw.
            value += self._ref * (
                self._interp(self._cmass, 0, hi)
                - self._interp(self._cmass, 0, lo)
            )
        if self.atom_theta.size:
            selected = (self.atom_theta > lo) & (self.atom_theta <= hi)
            value += float(
                (self.atom_theta[selected] * self.atom_w[selected]).sum()
            )
        return value / self.total

    def cond_mean(self, lo: float, hi: float) -> Optional[float]:
        probability = self.mass(lo, hi)
        if probability <= 0.0:
            return None
        return self.moment(lo, hi) / probability

    def window_moments(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        center=None,
    ) -> np.ndarray:
        """Return window moments about ``center`` (``None`` means raw).

        Both the continuous and the atomic parts accumulate about the
        internal reference ``_ref`` and are shifted to ``center`` with one
        binomial shift, so the two parts stay mutually consistent.
        """

        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have matching shapes")
        flat_lo = lo.ravel()
        flat_hi = hi.ravel()
        out = np.zeros((3, flat_lo.size))
        if self.grid.size:
            for order, cumulative in enumerate(
                (self._cmass, self._cmom, self._cmom2)
            ):
                out[order] = self._interp_many(
                    cumulative, order, flat_hi
                ) - self._interp_many(cumulative, order, flat_lo)
        if self.atom_theta.size:
            theta = self.atom_theta[:, None] - self._ref
            weights = self.atom_w[:, None]
            selected = (self.atom_theta[:, None] > flat_lo[None, :]) & (
                self.atom_theta[:, None] <= flat_hi[None, :]
            )
            out[0] += (weights * selected).sum(axis=0)
            out[1] += (theta * weights * selected).sum(axis=0)
            out[2] += (theta * theta * weights * selected).sum(axis=0)
        out[:, flat_hi <= flat_lo] = 0.0
        if center is None:
            delta = -self._ref
        else:
            delta = np.asarray(center, dtype=float).ravel() - self._ref
        m0, m1, m2 = out
        out = np.stack(
            (
                m0,
                m1 - delta * m0,
                m2 - 2.0 * delta * m1 + delta * delta * m0,
            )
        )
        return out / self.total

    def integrate(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        lo: float,
        hi: float,
    ) -> float:
        total = 0.0
        if self.grid.size and hi > lo:
            left = max(float(lo), float(self.grid[0]))
            right = min(float(hi), float(self.grid[-1]))
            if right > left:
                interior = self.grid[
                    (self.grid > left) & (self.grid < right)
                ]
                edges = np.concatenate([[left], interior, [right]])
                a, b = edges[:-1], edges[1:]
                midpoint = 0.5 * (a + b)
                half = 0.5 * (b - a)
                z = math.sqrt(3.0 / 5.0)
                nodes = np.stack(
                    (midpoint - z * half, midpoint, midpoint + z * half)
                )
                weights = np.asarray(
                    [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]
                )[:, None]
                density = np.interp(nodes, self.grid, self.dens)
                fn_value = np.asarray(fn(nodes), dtype=float)
                total += float(
                    np.sum(
                        half
                        * np.sum(weights * fn_value * density, axis=0)
                    )
                )
        if self.atom_theta.size:
            selected = (self.atom_theta > lo) & (self.atom_theta <= hi)
            if selected.any():
                total += float(
                    (
                        np.asarray(fn(self.atom_theta[selected]), dtype=float)
                        * self.atom_w[selected]
                    ).sum()
                )
        return total / self.total

    def theta_support_intervals(self) -> List[Tuple[float, float]]:
        """Return charged type-support components for optional diagnostics."""

        if self.full_support:
            return [(-math.inf, math.inf)]
        intervals: List[Tuple[float, float]] = [
            (float(theta), float(theta))
            for theta, weight in zip(self.atom_theta, self.atom_w)
            if weight > 0.0
        ]
        if self.grid.size:
            charged = (self.dens[:-1] + self.dens[1:]) > 0.0
            index = np.flatnonzero(charged)
            if index.size:
                runs = np.split(
                    index, np.flatnonzero(np.diff(index) > 1) + 1
                )
                for run in runs:
                    intervals.append(
                        (
                            float(self.grid[run[0]]),
                            float(self.grid[run[-1] + 1]),
                        )
                    )
        return intervals


class OraclePrior:
    """Adapter for a probability measure supplied through numerical oracles.

    The caller is responsible for normalization, endpoint conventions, and the
    accuracy of the supplied operations.  This class makes arbitrary-measure
    computation possible; it does not manufacture error bounds or theorem
    regularity.
    """

    def __init__(
        self,
        *,
        mass: Callable[[float, float], float],
        moment: Callable[[float, float], float],
        integrate: Callable[
            [Callable[[np.ndarray], np.ndarray], float, float], float
        ],
        window_moments: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        full_support: bool = False,
        has_atoms: bool = False,
        continuous_positive_density: bool = False,
        exact_window_moments: bool = False,
        error_bound_available: bool = False,
        support_intervals: Optional[
            Callable[[], Sequence[Tuple[float, float]]]
        ] = None,
        description: str = "measure-oracle",
    ) -> None:
        self._mass_fn = mass
        self._moment_fn = moment
        self._integrate_fn = integrate
        self._window_moments_fn = window_moments
        self._support_intervals_fn = support_intervals
        # Emit at most one cancellation warning for each oracle instance.
        self._raw_shift_warned = False
        self.full_support = bool(full_support)
        self.capabilities = PriorCapabilities(
            representation=description,
            has_atoms=bool(has_atoms),
            full_support=self.full_support,
            continuous_positive_density=bool(
                continuous_positive_density
            ),
            exact_window_moments=bool(exact_window_moments),
            error_bound_available=bool(error_bound_available),
        )

    def mass(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        value = float(self._mass_fn(lo, hi))
        if not math.isfinite(value) or value < -1e-14 or value > 1.0 + 1e-12:
            raise ArithmeticError("mass oracle returned an invalid probability")
        return min(1.0, max(0.0, value))

    def moment(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        value = float(self._moment_fn(lo, hi))
        if not math.isfinite(value):
            raise ArithmeticError("moment oracle returned a non-finite value")
        return value

    def cond_mean(self, lo: float, hi: float) -> Optional[float]:
        probability = self.mass(lo, hi)
        if probability <= 0.0:
            return None
        return self.moment(lo, hi) / probability

    def integrate(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        lo: float,
        hi: float,
    ) -> float:
        if hi <= lo:
            return 0.0
        value = float(self._integrate_fn(fn, lo, hi))
        if not math.isfinite(value):
            raise ArithmeticError("integration oracle returned a non-finite value")
        return value

    def window_moments(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        center=None,
    ) -> np.ndarray:
        """Return window moments about ``center`` (``None`` means raw).

        When the wrapped oracle supplies a vector ``window_moments`` without
        ``center`` support, the centered moments are recovered by a binomial
        shift of the oracle's raw moments.  That expanded shift retains the
        catastrophic cancellation the centered path is designed to remove:
        the roundoff loss grows as ``eps * (|m2_raw| + 2 |delta| |m1_raw|
        + delta^2 m0)`` with ``eps ~ 2.2e-16``; that is, as ``eps * delta**2``
        for far centers. When the estimated loss exceeds ``1e-9``
        of the shifted second moment a ``RuntimeWarning`` is issued once
        per instance; supply a center-aware oracle to avoid the loss.
        """

        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have matching shapes")
        if self._window_moments_fn is not None:
            if center is None:
                raw = np.asarray(
                    self._window_moments_fn(lo, hi), dtype=float
                )
            else:
                try:
                    result = np.asarray(
                        self._window_moments_fn(lo, hi, center),
                        dtype=float,
                    )
                except TypeError:
                    raw = np.asarray(
                        self._window_moments_fn(lo, hi), dtype=float
                    )
                else:
                    if result.shape != (3, lo.size):
                        raise ValueError(
                            "window_moments oracle must return shape (3, lo.size)"
                        )
                    return result
            if raw.shape != (3, lo.size):
                raise ValueError(
                    "window_moments oracle must return shape (3, lo.size)"
                )
            if center is None:
                return raw
            delta = np.asarray(center, dtype=float).ravel()
            m0, m1, m2 = raw
            shifted_m2 = m2 - 2.0 * delta * m1 + delta * delta * m0
            # Estimate roundoff loss from shifting raw oracle moments.
            eps = 2.2e-16
            loss = eps * (
                np.abs(m2)
                + 2.0 * np.abs(delta) * np.abs(m1)
                + delta * delta * m0
            )
            if not self._raw_shift_warned and np.any(
                loss > 1e-9 * np.maximum(np.abs(shifted_m2), 1e-300)
            ):
                warnings.warn(
                    "OraclePrior: centered window moments recovered by "
                    "shifting raw oracle moments lose an estimated "
                    f"{float(np.max(loss)):.3e} to cancellation; supply a "
                    "center-aware window_moments oracle",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._raw_shift_warned = True
            return np.stack(
                (
                    m0,
                    m1 - delta * m0,
                    shifted_m2,
                )
            )
        flat_lo = lo.ravel()
        flat_hi = hi.ravel()
        if center is None:
            centers = np.zeros(flat_lo.size)
        else:
            centers = np.broadcast_to(
                np.asarray(center, dtype=float).ravel(), flat_lo.shape
            )
        out = np.zeros((3, flat_lo.size))
        for i, (left, right) in enumerate(zip(flat_lo, flat_hi)):
            shift = float(centers[i])
            out[0, i] = self.mass(float(left), float(right))
            if shift == 0.0:
                out[1, i] = self.moment(float(left), float(right))
            else:
                out[1, i] = self.integrate(
                    lambda theta: np.asarray(theta, dtype=float) - shift,
                    float(left),
                    float(right),
                )
            out[2, i] = self.integrate(
                lambda theta: (np.asarray(theta, dtype=float) - shift)
                ** 2,
                float(left),
                float(right),
            )
        return out

    def theta_support_intervals(self) -> List[Tuple[float, float]]:
        if self.full_support:
            return [(-math.inf, math.inf)]
        if self._support_intervals_fn is None:
            raise NotImplementedError(
                "support-restricted diagnostics require a support oracle"
            )
        return [
            (float(lo), float(hi))
            for lo, hi in self._support_intervals_fn()
        ]


@dataclass
class Params:
    """Model primitives using the paper's notation."""

    alpha: float = 1.5
    beta: float = 1.5
    c_a: float = 1.0
    c_p: float = 1.0
    c_d: float = 0.5
    delta0: float = 0.05
    V: float = 10.0
    r: float = 1.0

    delta: Optional[Callable[[float], float]] = None
    C0: Optional[Callable[[float], float]] = None
    u: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __post_init__(self) -> None:
        scalars = (
            self.alpha,
            self.beta,
            self.c_a,
            self.c_p,
            self.c_d,
            self.delta0,
            self.V,
            self.r,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ValueError("scalar parameters must be finite")
        if (
            self.alpha < 0.0
            or self.beta <= 0.0
            or self.c_a <= 0.0
            or self.c_p <= 0.0
            or self.c_d < 0.0
            or self.delta0 < 0.0
            or self.V <= 0.0
        ):
            raise ValueError(
                "require alpha>=0, beta,c_a,c_p,V>0, and c_d,delta0>=0"
            )

    def delta_of(self, d: float) -> float:
        value = (
            self.delta0 * d * d
            if self.delta is None
            else float(self.delta(d))
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("delta(d) must be finite and nonnegative")
        return value

    def C0_of(self, d: float) -> float:
        value = (
            0.5 * self.c_d * d * d
            if self.C0 is None
            else float(self.C0(d))
        )
        if not math.isfinite(value):
            raise ValueError("C0(d) must be finite")
        return value

    def omega(self, d: float) -> float:
        delta = self.delta_of(d)
        return (self.c_p + delta) / (
            self.c_p + delta + self.c_a * self.beta**2
        )

    def Y(self, d: float) -> float:
        return math.sqrt(2.0 * self.V / (self.c_a * self.omega(d)))

    def C_min(self, d: float, R: float) -> float:
        return (
            0.5
            * self.c_a
            * self.omega(d)
            * R
            * R
        )


@dataclass(frozen=True)
class CutoffResult:
    value: float
    bracket_low: float
    bracket_high: float
    payoff_low: float
    payoff_high: float
    iterations: int
    status: str
    window_mass: float
    completion_used: bool


@dataclass(frozen=True)
class ArgmaxComponent:
    d: float
    value: float
    bracket: Tuple[float, float]
    grid_index: int
    is_boundary: bool
    global_gap: float
    is_global: bool


@dataclass(frozen=True)
class BestResponseResult:
    components: Tuple[ArgmaxComponent, ...]
    global_components: Tuple[ArgmaxComponent, ...]
    value: float
    d_min: float
    d_max: float
    grid_points: int
    comparison_tolerance: float


@dataclass(frozen=True)
class PRSolution:
    d: float
    c: float
    U_S: float
    U_R: float
    grid_points: int
    comparison_tolerance: float


@dataclass(frozen=True)
class SignChange:
    d_lo: float
    d_hi: float
    h_lo: float
    h_hi: float
    kind: str
    br_at_boundary: Tuple[float, float]


@dataclass(frozen=True)
class FixedPointCandidate:
    d: float
    deviation_gap: float
    comparison_tolerance: float
    scan_points: int
    best_response_points: int


@dataclass
class TRScan:
    fixed_points: List[float]
    sign_changes: List[SignChange]
    d_grid: np.ndarray
    h_grid: np.ndarray
    gap_grid: np.ndarray
    min_gap_d: float
    min_gap: float
    status: str
    finite_cutoff_share: float
    candidates: List[FixedPointCandidate]

    @property
    def exists(self) -> bool:
        return bool(self.fixed_points)


class Model:
    """Numerical solver bound to one prior and one parameter set."""

    def __init__(
        self,
        prior: PriorMeasure,
        params: Params,
        conventions: Optional[Conventions] = None,
    ) -> None:
        """Bind the solver to one prior, one parameter set, and conventions.

        A non-face-value completion is incompatible with a custom ``u``
        because the completion stores only a mean adjustment. That mean is
        sufficient for the built-in affine reviewer payoff, but not for an
        arbitrary nonlinear payoff.
        """

        required = ("mass", "moment", "window_moments", "integrate")
        if not all(callable(getattr(prior, name, None)) for name in required):
            raise TypeError("prior does not implement the PriorMeasure operations")
        self.F = prior
        self.P = params
        self.conventions = conventions or Conventions()
        if params.u is not None and not self.conventions.completion.is_face_value:
            raise ValueError(
                "a non-degenerate completion carries only a mean adjustment "
                "and pins the reviewer payoff only for affine u"
            )

    def null_window_value(self, c: float, d: float) -> float:
        """Reviewer payoff at a null-window manuscript under the completion.

        Under the face-value completion this equals ``u(c)`` exactly.
        """

        params = self.P
        omega = params.omega(d)
        Y = params.Y(d)
        x = c - params.alpha * d
        mean = self.conventions.completion.cond_mean(x, Y)
        quality = omega * c + (1.0 - omega) * (params.alpha * d + mean)
        if params.u is None:
            return float(quality - params.r)
        return float(
            np.asarray(params.u(np.asarray([quality])), dtype=float)[0]
        )

    def G(self, c: float, d: float) -> float:
        """Reviewer payoff at a candidate marginal manuscript ``c``."""

        params = self.P
        omega = params.omega(d)
        Y = params.Y(d)
        x = c - params.alpha * d
        mass = self.F.mass(x - Y, x)
        if mass <= self.conventions.mass_tol:
            return self.null_window_value(c, d)
        if params.u is None:
            mean = self.F.moment(x - Y, x) / mass
            return (
                omega * c
                + (1.0 - omega) * (params.alpha * d + mean)
                - params.r
            )
        quality = lambda theta: (
            omega * c
            + (1.0 - omega)
            * (params.alpha * d + np.asarray(theta, dtype=float))
        )
        return self.F.integrate(
            lambda theta: np.asarray(params.u(quality(theta)), dtype=float),
            x - Y,
            x,
        ) / mass

    def _first_sign_change(
        self,
        lo: float,
        hi: float,
        d: float,
        points: int = 513,
    ) -> Optional[Tuple[float, float]]:
        """Return the first interior [G<0, G>=0] mesh bracket, if any."""

        mesh = np.linspace(lo, hi, points)
        values = np.asarray(
            [self.G(float(c), d) for c in mesh], dtype=float
        )
        negative = values < 0.0
        index = np.flatnonzero(negative[:-1] & ~negative[1:])
        if index.size == 0:
            return None
        j = int(index[0])
        return float(mesh[j]), float(mesh[j + 1])

    def cutoff_result(
        self,
        d: float,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> CutoffResult:
        """Find the global cutoff by adaptive bracketing and bisection.

        Bracket endpoints are auto-derived only where the caller passed
        ``None``; explicitly passed endpoints are honoured as given.

        The construction targets weakly increasing ``G`` (linear ``u``
        under the face-value completion), where the boundary equals
        ``inf{c : G(c; d) >= 0}`` and the two ``TieRule`` selections
        coincide at that infimum. For non-monotone
        ``G`` (custom ``u`` or a non-degenerate completion) the result is
        a sign-change point of ``G``, not necessarily the infimum; before
        conceding an extended all-accept/all-reject status the original
        bracket interior is scanned for a hidden crossing.
        """

        # Preserve endpoint provenance for informative validation errors.
        lo_supplied = lo is not None
        hi_supplied = hi is not None
        if self.P.u is None:
            if lo is None:
                lo = self.P.r - max(1.0, self.P.Y(d))
                if lo == self.P.r:
                    raise ArithmeticError(
                        "cutoff scale exceeds floating-point resolution"
                    )
            if hi is None:
                hi = self.P.r + self.P.Y(d)
                if hi == self.P.r:
                    raise ArithmeticError(
                        "cutoff scale exceeds floating-point resolution"
                    )
        else:
            if lo is None:
                lo = -1e3
            if hi is None:
                hi = 1e3
        if hi <= lo:
            raise ValueError(
                "cutoff bracket must satisfy lo < hi; got lo="
                f"{lo!r} "
                f"({'caller-supplied' if lo_supplied else 'auto-derived'})"
                ", hi="
                f"{hi!r} "
                f"({'caller-supplied' if hi_supplied else 'auto-derived'})"
            )

        # With custom u or a non-face-value completion, G may be
        # non-monotone and an acceptance band can hide between expansion
        # probes. Scan the initial bracket before returning an extended
        # all-accept or all-reject status in that case.
        non_monotone_risk = (
            self.P.u is not None
            or not self.conventions.completion.is_face_value
        )
        g_lo = self.G(lo, d)
        g_hi = self.G(hi, d)
        step = max(1.0, hi - lo)

        if g_lo >= 0.0:
            b = lo
            for expansion in range(80):
                a = b - step
                g_a = self.G(a, d)
                if g_a < 0.0:
                    break
                b = a
                step *= 2.0
            else:
                # Scan for a hidden rejection region before returning.
                interior = (
                    self._first_sign_change(lo, hi, d)
                    if non_monotone_risk
                    else None
                )
                if interior is None:
                    # This status describes the sampled sign of G, not the
                    # reviewer tie rule.
                    return CutoffResult(
                        value=-math.inf,
                        bracket_low=-math.inf,
                        bracket_high=-math.inf,
                        payoff_low=g_lo,
                        payoff_high=g_lo,
                        iterations=80,
                        status="all_accept",
                        window_mass=0.0,
                        completion_used=False,
                    )
                a, b = interior
        elif g_hi < 0.0:
            a = hi
            for expansion in range(80):
                b = a + step
                g_b = self.G(b, d)
                if g_b >= 0.0:
                    break
                a = b
                step *= 2.0
            else:
                # Scan for a hidden acceptance region before returning.
                interior = (
                    self._first_sign_change(lo, hi, d)
                    if non_monotone_risk
                    else None
                )
                if interior is None:
                    # This status describes the sampled sign of G, not the
                    # reviewer tie rule.
                    return CutoffResult(
                        value=math.inf,
                        bracket_low=math.inf,
                        bracket_high=math.inf,
                        payoff_low=g_hi,
                        payoff_high=g_hi,
                        iterations=80,
                        status="all_reject",
                        window_mass=0.0,
                        completion_used=False,
                    )
                a, b = interior
        else:
            a, b = lo, hi

        iterations = 0
        for iterations in range(1, 257):
            tolerance = CUTOFF_XTOL + CUTOFF_RTOL * max(
                1.0, b - a, self.P.Y(d)
            )
            if b - a <= tolerance:
                break
            midpoint = 0.5 * (a + b)
            if midpoint == a or midpoint == b:
                break
            if self.G(midpoint, d) >= 0.0:
                b = midpoint
            else:
                a = midpoint
        else:
            raise RuntimeError("cutoff bisection did not converge")

        value = b
        x = value - self.P.alpha * d
        window_mass = self.F.mass(x - self.P.Y(d), x)
        return CutoffResult(
            value=float(value),
            bracket_low=float(a),
            bracket_high=float(b),
            payoff_low=float(self.G(a, d)),
            payoff_high=float(self.G(b, d)),
            iterations=iterations,
            status="finite_boundary",
            window_mass=float(window_mass),
            completion_used=bool(window_mass <= self.conventions.mass_tol),
        )

    def attainable_manuscript_intervals(
        self, d: float
    ) -> List[Tuple[float, float]]:
        """Return payoff-attainable manuscript intervals for diagnostics."""

        if self.F.full_support:
            return [(-math.inf, math.inf)]
        support_fn = getattr(self.F, "theta_support_intervals", None)
        if not callable(support_fn):
            raise NotImplementedError(
                "support-restricted diagnostics require support information"
            )
        Y = self.P.Y(d)
        intervals = [
            (
                self.P.alpha * d + float(lo),
                self.P.alpha * d + float(hi) + Y,
            )
            for lo, hi in support_fn()
        ]
        intervals.sort()
        merged: List[Tuple[float, float]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def c_star(
        self,
        d: float,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
        support_restricted: bool = False,
    ) -> float:
        """Return the global cutoff; support snapping is diagnostic-only.

        Cutoff-selection semantics assume weakly increasing ``G``, where
        the two ``TieRule`` selections coincide at the infimum;
        see ``cutoff_result`` for the non-monotone caveat.
        """

        cutoff = self.cutoff_result(d, lo=lo, hi=hi).value
        if not support_restricted or not math.isfinite(cutoff):
            return cutoff
        next_attainable = math.inf
        for left, right in self.attainable_manuscript_intervals(d):
            if left <= cutoff < right:
                return cutoff
            if left > cutoff:
                next_attainable = min(next_attainable, left)
        return next_attainable

    def pool_terms(self, c: float, d: float) -> Tuple[float, float]:
        """Return pooling mass and reviewer-payoff numerator.

        The linear-``u`` branch is written in moments centered at
        ``r - alpha d``, so far-from-zero supports do not cancel
        catastrophically.
        """

        params = self.P
        omega = params.omega(d)
        Y = params.Y(d)
        x = c - params.alpha * d
        if params.u is None:
            center = params.r - params.alpha * d
            mass, centered_moment, _ = self.F.window_moments(
                np.asarray([x - Y]),
                np.asarray([x]),
                center=center,
            )[:, 0]
            numerator = (
                omega * (c - params.r) * mass
                + (1.0 - omega) * centered_moment
            )
            return float(mass), float(numerator)
        mass = self.F.mass(x - Y, x)
        quality = lambda theta: (
            omega * c
            + (1.0 - omega)
            * (params.alpha * d + np.asarray(theta, dtype=float))
        )
        numerator = self.F.integrate(
            lambda theta: np.asarray(
                params.u(quality(theta)), dtype=float
            ),
            x - Y,
            x,
        )
        return float(mass), float(numerator)

    def natural_value(self, c: float, d: float) -> float:
        """Reviewer contribution from types that pass without adjustment."""

        params = self.P
        x = c - params.alpha * d
        if params.u is None:
            # E[(theta - (r - alpha d)) 1{theta > x}] equals the natural
            # integrand exactly, in centered (cancellation-free) form.
            return float(
                self.F.window_moments(
                    np.asarray([x]),
                    np.asarray([math.inf]),
                    center=params.r - params.alpha * d,
                )[1, 0]
            )
        return float(
            self.F.integrate(
                lambda theta: np.asarray(
                    params.u(params.alpha * d + theta), dtype=float
                ),
                x,
                math.inf,
            )
        )

    def U_R(self, d: float, c: Optional[float] = None) -> float:
        """Reviewer's expected equilibrium payoff."""

        if c is None:
            c = self.c_star(d)
        if c == math.inf:
            return 0.0
        if c == -math.inf:
            if self.P.u is None:
                return float(
                    self.F.window_moments(
                        np.asarray([-math.inf]),
                        np.asarray([math.inf]),
                        center=self.P.r - self.P.alpha * d,
                    )[1, 0]
                )
            return self.F.integrate(
                lambda theta: np.asarray(
                    self.P.u(self.P.alpha * d + theta), dtype=float
                ),
                -math.inf,
                math.inf,
            )
        _, numerator = self.pool_terms(c, d)
        return self.natural_value(c, d) + numerator

    def _omega_vec(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if self.P.delta is None:
            delta = self.P.delta0 * d * d
        else:
            delta = np.asarray(
                [self.P.delta_of(float(value)) for value in d],
                dtype=float,
            )
        return (self.P.c_p + delta) / (
            self.P.c_p + delta + self.P.c_a * self.P.beta**2
        )

    def _C0_vec(self, d: np.ndarray) -> np.ndarray:
        d = np.asarray(d, dtype=float)
        if self.P.C0 is None:
            return 0.5 * self.P.c_d * d * d
        return np.asarray(
            [self.P.C0_of(float(value)) for value in d], dtype=float
        )

    def U_S_vec(self, d: np.ndarray, c: float) -> np.ndarray:
        """Scholar payoff at many designs against one fixed cutoff."""

        d = np.asarray(d, dtype=float)
        if d.ndim != 1:
            raise ValueError("d must be a one-dimensional array")
        if np.any(d < 0.0) or np.any(~np.isfinite(d)):
            raise ValueError("design values must be finite and nonnegative")
        if c == -math.inf:
            return self.P.V - self._C0_vec(d)
        if c == math.inf:
            return -self._C0_vec(d)

        omega = self._omega_vec(d)
        Y = np.sqrt(2.0 * self.P.V / (self.P.c_a * omega))
        x = c - self.P.alpha * d
        natural_mass = self.F.window_moments(
            x, np.full_like(x, math.inf)
        )[0]
        # E[(x - theta)^2] over the pool equals the second window moment
        # centered at x; centering avoids the raw-moment cancellation.
        m0, _, m2 = self.F.window_moments(x - Y, x, center=x)
        k = 0.5 * self.P.c_a * omega
        adjustment_cost = k * m2
        return (
            self.P.V * (natural_mass + m0)
            - adjustment_cost
            - self._C0_vec(d)
        )

    def U_S(self, d_own: float, c: float) -> float:
        return float(
            self.U_S_vec(np.asarray([d_own], dtype=float), c)[0]
        )

    def d_bar(self) -> float:
        """Compact upper bound for design optimization."""

        target = self.P.V + self.P.C0_of(0.0)
        if self.P.C0_of(1e9) <= target:
            raise ValueError(
                "design search cannot be compactified: C0(d) never "
                "exceeds V + C0(0); pass an explicit d_max (this happens "
                "e.g. when c_d = 0)"
            )
        d = 1.0
        for _ in range(200):
            if self.P.C0_of(d) > target:
                return d
            d *= 1.6
        raise RuntimeError(
            "failed to compactify design search: "
            "C0(d) did not exceed V + C0(0)"
        )

    def best_response_details(
        self,
        c: float,
        d_max: Optional[float] = None,
        n: int = GRID_DEFAULT,
    ) -> BestResponseResult:
        """Return all sampled/refined peak candidates and global winners."""

        if n < 3:
            raise ValueError("n must be at least three")
        upper = self.d_bar() if d_max is None else float(d_max)
        if not math.isfinite(upper) or upper <= 0.0:
            raise ValueError("d_max must be finite and positive")
        grid = np.linspace(0.0, upper, n)
        values = self.U_S_vec(grid, c)
        spacing = upper / (n - 1)
        raw = refine_grid_maxima(
            lambda value: self.U_S(value, c), grid, values
        )
        candidates = _dedupe_candidate_pairs(
            raw, radius=max(1e-8, 0.25 * spacing)
        )
        best_value = max(value for _, value in candidates)
        tolerance = comparison_tolerance(
            best_value, variation=payoff_scale(self)
        )

        components: List[ArgmaxComponent] = []
        for d_value, payoff in candidates:
            index = int(np.clip(round(d_value / spacing), 0, n - 1))
            bracket = (
                float(grid[max(index - 1, 0)]),
                float(grid[min(index + 1, n - 1)]),
            )
            gap = max(0.0, best_value - payoff)
            components.append(
                ArgmaxComponent(
                    d=float(d_value),
                    value=float(payoff),
                    bracket=bracket,
                    grid_index=index,
                    is_boundary=bool(
                        index == 0
                        or index == n - 1
                        or d_value <= ARGMAX_XTOL
                        or upper - d_value <= ARGMAX_XTOL
                    ),
                    global_gap=float(gap),
                    is_global=bool(gap <= tolerance),
                )
            )
        global_components = tuple(
            component
            for component in components
            if component.is_global
        )
        winners = [component.d for component in global_components]
        return BestResponseResult(
            components=tuple(sorted(components, key=lambda item: item.d)),
            global_components=tuple(
                sorted(global_components, key=lambda item: item.d)
            ),
            value=float(best_value),
            d_min=float(min(winners)),
            d_max=float(max(winners)),
            grid_points=n,
            comparison_tolerance=float(tolerance),
        )

    def best_response(
        self,
        c: float,
        d_max: Optional[float] = None,
        n: int = GRID_DEFAULT,
    ) -> Tuple[float, float, float]:
        details = self.best_response_details(c, d_max=d_max, n=n)
        return details.d_min, details.d_max, details.value

    def pr_objective(self, d: float) -> float:
        cutoff = self.c_star(d)
        if cutoff == -math.inf:
            return self.P.V - self.P.C0_of(d)
        if cutoff == math.inf:
            return -self.P.C0_of(d)
        return self.U_S(d, cutoff)

    def solve_PR(
        self,
        n: int = GRID_DEFAULT,
        d_max: Optional[float] = None,
    ) -> PRSolution:
        if n < 3:
            raise ValueError("n must be at least three")
        upper = self.d_bar() if d_max is None else float(d_max)
        if not math.isfinite(upper) or upper <= 0.0:
            raise ValueError("d_max must be finite and positive")
        grid = np.linspace(0.0, upper, n)
        values = np.asarray(
            [self.pr_objective(float(d)) for d in grid], dtype=float
        )
        candidates = refine_grid_maxima(
            self.pr_objective, grid, values
        )
        best_value = max(value for _, value in candidates)
        tolerance = comparison_tolerance(
            best_value, variation=payoff_scale(self)
        )
        tied_designs = [
            d for d, value in candidates
            if best_value - value <= tolerance
        ]
        # Part C defines the selected PR design as the largest maximizer.
        # Apply the same convention to numerically tied refined components.
        d_opt = max(tied_designs)
        value = self.pr_objective(d_opt)
        cutoff = self.c_star(d_opt)
        return PRSolution(
            d=float(d_opt),
            c=float(cutoff),
            U_S=float(value),
            U_R=float(self.U_R(d_opt, cutoff)),
            grid_points=n,
            comparison_tolerance=float(tolerance),
        )

    def deviation_gap(
        self,
        d: float,
        d_max: Optional[float] = None,
        br_grid: int = GRID_DEFAULT,
    ) -> float:
        upper = self.d_bar() if d_max is None else float(d_max)
        cutoff = self.c_star(d)
        best = self.best_response_details(
            cutoff, d_max=upper, n=br_grid
        ).value
        return max(0.0, best - self.U_S(d, cutoff))

    def scan_TR(
        self,
        n: int = 801,
        br_grid: int = GRID_DEFAULT,
        d_max: Optional[float] = None,
    ) -> TRScan:
        """Locate tolerance-qualified pure-TR fixed-point candidates."""

        if n < 3 or br_grid < 3:
            raise ValueError("scan grids must contain at least three points")
        upper = self.d_bar() if d_max is None else float(d_max)
        if not math.isfinite(upper) or upper <= 0.0:
            raise ValueError("d_max must be finite and positive")

        grid = np.linspace(0.0, upper, n)
        low_h = np.empty(n)
        high_h = np.empty(n)
        gap = np.empty(n)
        finite_cutoff = np.empty(n, dtype=bool)
        for i, conjecture in enumerate(grid):
            cutoff = self.c_star(float(conjecture))
            finite_cutoff[i] = math.isfinite(cutoff)
            response = self.best_response_details(
                cutoff, d_max=upper, n=br_grid
            )
            low_h[i] = response.d_min - conjecture
            high_h[i] = response.d_max - conjecture
            gap[i] = max(
                0.0,
                response.value - self.U_S(float(conjecture), cutoff),
            )

        fixed: List[float] = []
        changes: List[SignChange] = []
        for i in range(n - 1):
            if (
                finite_cutoff[i]
                and finite_cutoff[i + 1]
                and low_h[i] > 0.0
                and high_h[i + 1] < 0.0
            ):
                boundary = _bisect_sign(
                    lambda d: self.best_response_details(
                        self.c_star(d), d_max=upper, n=br_grid
                    ).d_min
                    - d,
                    float(grid[i]),
                    float(grid[i + 1]),
                )
                epsilon = max(
                    1e-9, 1e-9 * max(1.0, abs(boundary))
                )
                hits = [
                    point
                    for point in (boundary, boundary + epsilon)
                    if _in_argmax(self, point, upper, br_grid)
                ]
                left_response = self.best_response_details(
                    self.c_star(boundary), d_max=upper, n=br_grid
                )
                right_response = self.best_response_details(
                    self.c_star(boundary + epsilon),
                    d_max=upper,
                    n=br_grid,
                )
                changes.append(
                    SignChange(
                        d_lo=float(grid[i]),
                        d_hi=float(grid[i + 1]),
                        h_lo=float(low_h[i]),
                        h_hi=float(high_h[i + 1]),
                        kind="crossing" if hits else "jump",
                        br_at_boundary=(
                            min(
                                left_response.d_min,
                                right_response.d_min,
                            ),
                            max(
                                left_response.d_max,
                                right_response.d_max,
                            ),
                        ),
                    )
                )
                if hits:
                    fixed.append(float(hits[0]))

        gap_candidates: List[Tuple[float, float]] = []
        finite_index = np.flatnonzero(finite_cutoff)
        if finite_index.size:
            runs = np.split(
                finite_index,
                np.flatnonzero(np.diff(finite_index) > 1) + 1,
            )
            for run in runs:
                if run.size >= 2:
                    selection = slice(run[0], run[-1] + 1)
                    gap_candidates.extend(
                        refine_grid_minima(
                            lambda d: self.deviation_gap(
                                d, upper, br_grid
                            ),
                            grid[selection],
                            gap[selection],
                        )
                    )
                elif run.size == 1:
                    gap_candidates.append(
                        (
                            float(grid[run[0]]),
                            float(gap[run[0]]),
                        )
                    )

        for candidate, _ in gap_candidates:
            if (
                math.isfinite(self.c_star(candidate))
                and _in_argmax(self, candidate, upper, br_grid)
            ):
                fixed.append(float(candidate))

        if gap_candidates:
            min_gap_d, min_gap = min(
                gap_candidates, key=lambda item: item[1]
            )
        else:
            min_gap_d, min_gap = math.nan, math.inf

        fixed = _dedupe_points(fixed)
        fixed_candidates: List[FixedPointCandidate] = []
        for candidate in fixed:
            cutoff = self.c_star(candidate)
            response = self.best_response_details(
                cutoff, d_max=upper, n=br_grid
            )
            own = self.U_S(candidate, cutoff)
            fixed_candidates.append(
                FixedPointCandidate(
                    d=float(candidate),
                    deviation_gap=float(
                        max(0.0, response.value - own)
                    ),
                    comparison_tolerance=float(
                        comparison_tolerance(
                            response.value,
                            own,
                            variation=payoff_scale(self),
                        )
                    ),
                    scan_points=n,
                    best_response_points=br_grid,
                )
            )

        if not finite_cutoff.any():
            status = "no_finite_cutoff"
        elif fixed:
            status = "approximate_candidates_found"
        else:
            status = "no_candidate_on_mesh"

        return TRScan(
            fixed_points=fixed,
            sign_changes=changes,
            d_grid=grid,
            h_grid=low_h,
            gap_grid=gap,
            min_gap_d=float(min_gap_d),
            min_gap=float(min_gap),
            status=status,
            finite_cutoff_share=float(np.mean(finite_cutoff)),
            candidates=fixed_candidates,
        )


_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def _golden(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    xtol: float = ARGMAX_XTOL,
) -> Tuple[float, float]:
    a, b = lo, hi
    c = b - _INV_PHI * (b - a)
    d = a + _INV_PHI * (b - a)
    f_c, f_d = f(c), f(d)
    while b - a > xtol:
        if f_c > f_d:
            b, d, f_d = d, c, f_c
            c = b - _INV_PHI * (b - a)
            f_c = f(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + _INV_PHI * (b - a)
            f_d = f(d)
    x = 0.5 * (a + b)
    return float(x), float(f(x))


def refine_grid_maxima(
    f: Callable[[float], float],
    grid: np.ndarray,
    values: np.ndarray,
) -> List[Tuple[float, float]]:
    """Refine every sampled local maximum while retaining grid candidates."""

    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid.ndim != 1 or values.shape != grid.shape or grid.size < 2:
        raise ValueError("grid and values must be matching 1-D arrays")
    index: List[int] = [0, grid.size - 1, int(np.argmax(values))]
    if grid.size > 2:
        local = (
            np.flatnonzero(
                (values[1:-1] >= values[:-2])
                & (values[1:-1] >= values[2:])
            )
            + 1
        )
        index.extend(int(i) for i in local)
    output: List[Tuple[float, float]] = []
    for i in sorted(set(index)):
        output.append((float(grid[i]), float(values[i])))
        lo = float(grid[max(i - 1, 0)])
        hi = float(grid[min(i + 1, grid.size - 1)])
        if hi > lo:
            output.append(_golden(f, lo, hi))
    return output


def refine_grid_minima(
    f: Callable[[float], float],
    grid: np.ndarray,
    values: np.ndarray,
) -> List[Tuple[float, float]]:
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if grid.ndim != 1 or values.shape != grid.shape or grid.size < 2:
        raise ValueError("grid and values must be matching 1-D arrays")
    mask = np.ones(grid.size, dtype=bool)
    mask[1:] &= values[1:] <= values[:-1]
    mask[:-1] &= values[:-1] <= values[1:]
    index = np.flatnonzero(mask)
    runs = np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)
    output: List[Tuple[float, float]] = []
    for run in runs:
        if run.size == 0:
            continue
        center = int(run[run.size // 2])
        candidates = [(float(grid[center]), float(values[center]))]
        lo = float(grid[max(int(run[0]) - 1, 0)])
        hi = float(grid[min(int(run[-1]) + 1, grid.size - 1)])
        if hi > lo:
            x, negative = _golden(lambda d: -f(d), lo, hi)
            candidates.append((x, -negative))
        output.append(min(candidates, key=lambda item: item[1]))
    return output


def _dedupe_candidate_pairs(
    candidates: Sequence[Tuple[float, float]],
    radius: float = 1e-8,
) -> List[Tuple[float, float]]:
    """Merge candidates within an ABSOLUTE spacing-derived radius."""

    output: List[Tuple[float, float]] = []
    for d, value in sorted(
        ((float(d), float(value)) for d, value in candidates),
        key=lambda item: item[0],
    ):
        if output and abs(d - output[-1][0]) <= radius:
            if value > output[-1][1]:
                output[-1] = (d, value)
        else:
            output.append((d, value))
    return output


def comparison_tolerance(
    *values: float,
    variation: Optional[float] = None,
) -> float:
    level = max((abs(float(value)) for value in values), default=0.0)
    spread = abs(float(variation)) if variation is not None else level
    spread = max(spread, np.finfo(float).tiny)
    roundoff_scale = max(level, spread, np.finfo(float).tiny)
    return (
        PAYOFF_TIE_TOL * spread
        + 64.0 * np.finfo(float).eps * roundoff_scale
    )


def payoff_scale(model: Model) -> float:
    return max(
        float(model.P.V + abs(model.P.C0_of(0.0))),
        np.finfo(float).tiny,
    )


def _dedupe_points(
    points: Sequence[float], atol: float = 1e-7
) -> List[float]:
    output: List[float] = []
    for value in sorted(float(point) for point in points):
        if not output or abs(value - output[-1]) > atol * max(
            1.0, abs(value), abs(output[-1])
        ):
            output.append(value)
    return output


def _bisect_sign(
    h: Callable[[float], float],
    lo: float,
    hi: float,
    iterations: int = 60,
) -> float:
    for _ in range(iterations):
        midpoint = 0.5 * (lo + hi)
        if h(midpoint) > 0.0:
            lo = midpoint
        else:
            hi = midpoint
    return float(lo)


def _in_argmax(
    model: Model,
    d: float,
    d_max: float,
    n: int,
) -> bool:
    cutoff = model.c_star(d)
    response = model.best_response_details(
        cutoff, d_max=d_max, n=n
    )
    own = model.U_S(d, cutoff)
    return own >= response.value - comparison_tolerance(
        own, response.value, variation=payoff_scale(model)
    )
