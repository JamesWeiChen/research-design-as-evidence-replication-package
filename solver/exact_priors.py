"""Exact closed-form priors and finite mixtures of prior measures.

Both classes implement the ``PriorMeasure`` protocol of ``general_solver``
with machine-precision window moments, no density tabulation, and genuine
full support. Closed forms avoid both tail truncation and density underflow
while preserving the ``continuous_positive_density`` capability.

Moment identities (window ``(lo, hi]``, standardised bounds
``a = (lo-mu)/sd``, ``b = (hi-mu)/sd``, ``phi``/``Phi`` the standard
normal pdf/cdf):

    Normal:  m0 = Phi(b) - Phi(a)
             m1 = mu*m0 + sd*(phi(a) - phi(b))
             m2 = (mu^2 + sd^2)*m0 + sd*((lo + mu)*phi(a) - (hi + mu)*phi(b))

    Laplace(m, b), one-sided piece on the right tail with u = (theta-m)/b:
             int f          = (1/2)*(e^{-u1} - e^{-u2})
             int theta f    = (1/2)*(m*De + b*D[(u+1)e^{-u}])
             int theta^2 f  = (1/2)*(m^2*De + 2mb*D[(u+1)e^{-u}]
                                     + b^2*D[(u^2+2u+2)e^{-u}])
    with D[g] = g(u1) - g(u2); the left tail by symmetry; windows crossing
    the median split at ``m``.

Centred moments are computed by shifting the location parameter
(``E[(theta-c)^k] over the window`` equals the raw moments of the same
family with location ``mu-c`` over ``(lo-c, hi-c]``), so the cancellation
scale is ``|mu - c|``, bounded in every solver query.

``integrate`` (custom-``u`` path only) uses composite Gauss-Legendre
quadrature on the window, split at the Laplace kink; spectral accuracy for
smooth integrands.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .general_solver import PriorCapabilities

__all__ = ["MixturePrior", "NormalPrior", "LaplacePrior"]

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

# 32-node Gauss-Legendre rule on [-1, 1] for the integrate() fallback.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _std_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(z, float) ** 2) / _SQRT2PI


def _std_normal_mass(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Phi(b) - Phi(a), tail-stable via erfc on the dominant side."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    try:
        from scipy.special import erf, erfc
    except ImportError:                                    # pragma: no cover
        erf = np.vectorize(math.erf)
        erfc = np.vectorize(math.erfc)
    # For windows in the far right tail use sf differences, far left use
    # cdf differences, otherwise the plain erf difference is fine.
    plain = 0.5 * (erf(b / _SQRT2) - erf(a / _SQRT2))
    right = 0.5 * (erfc(a / _SQRT2) - erfc(b / _SQRT2))
    left = 0.5 * (erfc(-b / _SQRT2) - erfc(-a / _SQRT2))
    out = np.where(a > 0.0, right, np.where(b < 0.0, left, plain))
    return np.maximum(out, 0.0)


class _ClosedFormPrior:
    """Shared plumbing for the two closed-form families."""

    full_support = True

    def __init__(self) -> None:
        self.capabilities = PriorCapabilities(
            representation=self._representation,
            has_atoms=False,
            full_support=True,
            continuous_positive_density=True,
            exact_window_moments=True,
            error_bound_available=True,
        )

    # subclasses provide _raw_moments(lo, hi, shift) returning (3, N)

    def mass(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(self._raw_moments(
            np.asarray([lo]), np.asarray([hi]), 0.0)[0][0])

    def moment(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(self._raw_moments(
            np.asarray([lo]), np.asarray([hi]), 0.0)[1][0])

    def cond_mean(self, lo: float, hi: float) -> Optional[float]:
        w = self.mass(lo, hi)
        if w <= 0.0:
            return None
        return self.moment(lo, hi) / w

    def window_moments(self, lo, hi, center=None) -> np.ndarray:
        lo = np.atleast_1d(np.asarray(lo, float))
        hi = np.atleast_1d(np.asarray(hi, float))
        if center is None:
            out = self._raw_moments(lo, hi, 0.0)
        else:
            c = np.broadcast_to(np.asarray(center, float), lo.shape)
            # E[(theta-c)^k 1] = raw moments of the location-shifted family
            # on the shifted window; loop over distinct centers is avoided
            # by vectorising the shift inside _raw_moments.
            out = self._raw_moments(lo, hi, c)
        out[:, hi <= lo] = 0.0
        return out

    def integrate(self, fn: Callable[[np.ndarray], np.ndarray],
                  lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        lo_c, hi_c = self._clip_window(lo, hi)
        if hi_c <= lo_c:
            return 0.0
        pieces = self._quad_pieces(lo_c, hi_c)
        total = 0.0
        for a, b in pieces:
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            nodes = mid + half * _GL_NODES
            total += half * float(np.sum(
                _GL_WEIGHTS * np.asarray(fn(nodes), float)
                * self._pdf(nodes)))
        return total

    def theta_support_intervals(self) -> List[Tuple[float, float]]:
        return [(-math.inf, math.inf)]


class MixturePrior:
    """Finite probability mixture of objects implementing ``PriorMeasure``.

    Window probabilities, moments, and integrals are combined linearly.  This
    keeps mixtures of closed-form priors on their exact component backends;
    in particular, a Gaussian mixture does not require density tabulation or
    tail truncation.
    """

    def __init__(self, components: Sequence[object], weights: Sequence[float]):
        if not components:
            raise ValueError("components must be nonempty")
        if len(components) != len(weights):
            raise ValueError("components and weights must have the same length")
        required = ("mass", "moment", "window_moments", "integrate")
        for component in components:
            if not all(callable(getattr(component, name, None)) for name in required):
                raise TypeError("every component must implement PriorMeasure operations")

        raw_weights = np.asarray(weights, dtype=float)
        if raw_weights.ndim != 1 or np.any(~np.isfinite(raw_weights)):
            raise ValueError("weights must be a finite one-dimensional vector")
        if np.any(raw_weights <= 0.0):
            raise ValueError("weights must be strictly positive")
        total = math.fsum(float(weight) for weight in raw_weights)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("mixture weights must sum to one within 1e-12")

        self.components = tuple(components)
        self.weights = tuple(float(weight / total) for weight in raw_weights)
        capabilities = [component.capabilities for component in self.components]
        self.full_support = any(
            weight > 0.0 and bool(component.full_support)
            for component, weight in zip(self.components, self.weights)
        )
        self.capabilities = PriorCapabilities(
            representation="finite_mixture[" + ",".join(
                capability.representation for capability in capabilities
            ) + "]",
            has_atoms=any(capability.has_atoms for capability in capabilities),
            full_support=self.full_support,
            continuous_positive_density=any(
                capability.full_support and capability.continuous_positive_density
                for capability in capabilities
            ),
            exact_window_moments=all(
                capability.exact_window_moments for capability in capabilities
            ),
            error_bound_available=all(
                capability.error_bound_available for capability in capabilities
            ),
        )

    def mass(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return math.fsum(
            weight * component.mass(lo, hi)
            for component, weight in zip(self.components, self.weights)
        )

    def moment(self, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return math.fsum(
            weight * component.moment(lo, hi)
            for component, weight in zip(self.components, self.weights)
        )

    def cond_mean(self, lo: float, hi: float) -> Optional[float]:
        mass = self.mass(lo, hi)
        if mass <= 0.0:
            return None
        return self.moment(lo, hi) / mass

    def window_moments(self, lo, hi, center=None) -> np.ndarray:
        terms = [
            weight * component.window_moments(lo, hi, center=center)
            for component, weight in zip(self.components, self.weights)
        ]
        return np.add.reduce(terms)

    def integrate(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        lo: float,
        hi: float,
    ) -> float:
        if hi <= lo:
            return 0.0
        return math.fsum(
            weight * component.integrate(fn, lo, hi)
            for component, weight in zip(self.components, self.weights)
        )

    def theta_support_intervals(self) -> List[Tuple[float, float]]:
        if self.full_support:
            return [(-math.inf, math.inf)]
        intervals: List[Tuple[float, float]] = []
        for component in self.components:
            support = getattr(component, "theta_support_intervals", None)
            if not callable(support):
                raise NotImplementedError(
                    "every non-full-support component must report its support"
                )
            intervals.extend((float(lo), float(hi)) for lo, hi in support())
        intervals.sort()
        merged: List[Tuple[float, float]] = []
        for lo, hi in intervals:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged


class NormalPrior(_ClosedFormPrior):
    """Exact Normal(mu, sd) prior via erf/erfc closed forms."""

    _representation = "closed_form_normal"

    def __init__(self, mu: float = 0.0, sd: float = 1.0) -> None:
        if not (math.isfinite(mu) and math.isfinite(sd) and sd > 0.0):
            raise ValueError("require finite mu and sd > 0")
        self.mu = float(mu)
        self.sd = float(sd)
        super().__init__()

    def _pdf(self, theta: np.ndarray) -> np.ndarray:
        return _std_normal_pdf(
            (np.asarray(theta, float) - self.mu) / self.sd) / self.sd

    def _clip_window(self, lo: float, hi: float) -> Tuple[float, float]:
        span = 40.0 * self.sd
        return max(lo, self.mu - span), min(hi, self.mu + span)

    def _quad_pieces(self, lo: float, hi: float):
        edges = np.linspace(lo, hi, max(2, int((hi - lo) / self.sd) + 2))
        return list(zip(edges[:-1], edges[1:]))

    def _raw_moments(self, lo: np.ndarray, hi: np.ndarray, shift) -> np.ndarray:
        mu = self.mu - np.asarray(shift, float)     # location after shift
        lo_s = lo - shift
        hi_s = hi - shift
        a = (lo_s - mu) / self.sd
        b = (hi_s - mu) / self.sd
        m0 = _std_normal_mass(a, b)
        # Clamp the pdf-product bounds to +/-40 sd, where
        # the density is exactly 0.0 in double precision; otherwise a
        # semi-infinite window evaluates ``inf * 0 = nan`` in the m2 row
        # (and emits numpy invalid-multiply warnings) even though the
        # limit is 0.  m0 keeps the unclamped erf/erfc bounds.
        a_c = np.clip(a, -40.0, 40.0)
        b_c = np.clip(b, -40.0, 40.0)
        lo_c = mu + self.sd * a_c
        hi_c = mu + self.sd * b_c
        pa = _std_normal_pdf(a_c)
        pb = _std_normal_pdf(b_c)
        m1 = mu * m0 + self.sd * (pa - pb)
        m2 = (mu * mu + self.sd * self.sd) * m0 \
            + self.sd * ((lo_c + mu) * pa - (hi_c + mu) * pb)
        return np.stack([m0, m1, m2])


def _laplace_tail_terms(u1: np.ndarray, u2: np.ndarray):
    """D[g] terms for the right-tail piece, u2 >= u1 >= 0.

    At u = +inf every ``poly(u) * e^{-u}`` term is 0 in
    the limit, but naive evaluation gives ``inf * 0 = nan`` and poisons
    every semi-infinite window query (``natural_value`` uses ``hi = inf``).
    Clamping u to 750 keeps the limit exact in double precision
    (``e^{-750}`` underflows to 0) without branching.
    """
    u1 = np.minimum(np.asarray(u1, float), 750.0)
    u2 = np.minimum(np.asarray(u2, float), 750.0)
    e1 = np.exp(-u1)
    e2 = np.exp(-u2)
    d0 = e1 - e2
    d1 = (u1 + 1.0) * e1 - (u2 + 1.0) * e2
    d2 = (u1 * u1 + 2.0 * u1 + 2.0) * e1 - (u2 * u2 + 2.0 * u2 + 2.0) * e2
    return d0, d1, d2


class LaplacePrior(_ClosedFormPrior):
    """Exact Laplace(m, b) prior via piecewise-exponential closed forms."""

    _representation = "closed_form_laplace"

    def __init__(self, m: float = 0.0, b: float = 1.0) -> None:
        if not (math.isfinite(m) and math.isfinite(b) and b > 0.0):
            raise ValueError("require finite m and b > 0")
        self.m = float(m)
        self.b = float(b)
        super().__init__()

    def _pdf(self, theta: np.ndarray) -> np.ndarray:
        z = np.abs(np.asarray(theta, float) - self.m) / self.b
        return np.exp(-z) / (2.0 * self.b)

    def _clip_window(self, lo: float, hi: float) -> Tuple[float, float]:
        span = 80.0 * self.b
        return max(lo, self.m - span), min(hi, self.m + span)

    def _quad_pieces(self, lo: float, hi: float):
        pieces = []
        if lo < self.m < hi:
            cuts = [lo, self.m, hi]
        else:
            cuts = [lo, hi]
        for a0, b0 in zip(cuts[:-1], cuts[1:]):
            edges = np.linspace(a0, b0, max(2, int((b0 - a0) / self.b) + 2))
            pieces.extend(zip(edges[:-1], edges[1:]))
        return pieces

    def _one_side(self, lo, hi, loc):
        """Raw moments for a window on one side, vectorised.

        ``lo``, ``hi`` arrays with lo >= loc (right side) or hi <= loc
        (left side handled by mirroring in the caller).
        """
        u1 = (lo - loc) / self.b
        u2 = (hi - loc) / self.b
        d0, d1, d2 = _laplace_tail_terms(u1, u2)
        m0 = 0.5 * d0
        m1 = 0.5 * (loc * d0 + self.b * d1)
        m2 = 0.5 * (loc * loc * d0 + 2.0 * loc * self.b * d1
                    + self.b * self.b * d2)
        return m0, m1, m2

    def _raw_moments(self, lo: np.ndarray, hi: np.ndarray, shift) -> np.ndarray:
        loc = self.m - np.asarray(shift, float)
        lo_s = np.asarray(lo, float) - shift
        hi_s = np.asarray(hi, float) - shift
        loc_b = np.broadcast_to(loc, lo_s.shape).astype(float)
        m0 = np.zeros_like(lo_s)
        m1 = np.zeros_like(lo_s)
        m2 = np.zeros_like(lo_s)
        # right part: (max(lo, loc), hi]
        r_lo = np.maximum(lo_s, loc_b)
        r_ok = hi_s > r_lo
        if np.any(r_ok):
            a0, a1, a2 = self._one_side(r_lo[r_ok], hi_s[r_ok], loc_b[r_ok])
            m0[r_ok] += a0
            m1[r_ok] += a1
            m2[r_ok] += a2
        # left part: (lo, min(hi, loc)] mirrored through loc: theta ->
        # 2*loc - theta maps it to a right-side window; odd moments flip.
        l_hi = np.minimum(hi_s, loc_b)
        l_ok = l_hi > lo_s
        if np.any(l_ok):
            mlo = 2.0 * loc_b[l_ok] - l_hi[l_ok]
            mhi = 2.0 * loc_b[l_ok] - lo_s[l_ok]
            a0, a1, a2 = self._one_side(mlo, mhi, loc_b[l_ok])
            # E[theta^k] under mirror: theta = 2*loc - t
            m0[l_ok] += a0
            m1[l_ok] += 2.0 * loc_b[l_ok] * a0 - a1
            m2[l_ok] += (4.0 * loc_b[l_ok] ** 2 * a0
                         - 4.0 * loc_b[l_ok] * a1 + a2)
        return np.stack([m0, m1, m2])
