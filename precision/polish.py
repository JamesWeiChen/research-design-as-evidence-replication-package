"""Post-solution precision tools for the NHB solver.

The functions in this module improve localized roots and maximizers without
changing the model equations or equilibrium tests. The two basic tools are:

* root polish (``polish_root``): after a bracketing method has localized a
  root of a directly computable function (G in d, U_R(d) - target, the
  branch-tie function in c, G in c), a guarded secant/Newton iteration
  drives the RESIDUAL down to the function's floating-point evaluation
  floor (~1e-15 in source units), far below bisection's bracket-width
  floor (~1e-12).

* argmax polish (``polish_argmax``): flat-top maximizers (d_PR, d_TR, the
  two mixed branches) cannot be pinned by value comparison beyond
  ~sqrt(noise/curvature) ~ 1e-6.  Instead we solve f'(d) = 0, with f'
  computed by noise-optimal central differences; the derivative crosses
  zero with O(1) slope, so the crossing is pinned to ~1e-9 or better,
  which makes the 6th reported decimal of the coordinate and of any O(1)-
  slope functional of it (e.g. U_R at the argmax) reproducible.

Composite solvers:

* `polish_pure_TR`:   d = argmax U_S(. | c*(d))  (fixed point, secant on
                      the inner first-order condition composed with the
                      polished cutoff map)
* `polish_mixed`:     (d_L, d_H, c) with two branch first-order conditions
                      and the branch-value tie, by alternating argmax
                      polish with a secant root on the tie in c
* `polish_PR`:        d_PR = argmax U_S(. | c*(.)), then U_R at the point
* `polish_dcp_dpp`:   generalized inverses on the support hull

All outputs remain tolerance-qualified numerical diagnostics.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

from solver.general_solver import Model


# Generic one-dimensional tools.

def polish_root(f: Callable[[float], float], x0: float,
                scale: float = 1.0, max_iter: int = 60) -> Tuple[float, float]:
    """Guarded secant iteration started at a near-root point.

    Returns (x, |f(x)|) with the residual driven to the evaluation floor.
    `scale` sets the initial secant step (x-units).
    """
    x1 = x0
    f1 = f(x1)
    h = 1e-7 * max(1.0, abs(x0)) * scale
    x2 = x1 + h
    f2 = f(x2)
    best_x, best_f = (x1, abs(f1)) if abs(f1) <= abs(f2) else (x2, abs(f2))
    for _ in range(max_iter):
        denom = f2 - f1
        if denom == 0.0:
            break
        x3 = x2 - f2 * (x2 - x1) / denom
        if not math.isfinite(x3):
            break
        f3 = f(x3)
        if abs(f3) < best_f:
            best_x, best_f = x3, abs(f3)
        if f3 == 0.0 or abs(x3 - x2) < 1e-16 * max(1.0, abs(x3)):
            break
        x1, f1, x2, f2 = x2, f2, x3, f3
    return float(best_x), float(best_f)


def central_diff(f: Callable[[float], float], x: float,
                 h: Optional[float] = None) -> float:
    """Noise-optimal central difference (default step ~1e-4 relative)."""
    if h is None:
        h = 1e-4 * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2.0 * h)


def polish_argmax(f: Callable[[float], float], x0: float,
                  span: float = 5e-3, max_iter: int = 40,
                  h: Optional[float] = None) -> Tuple[float, float]:
    """Pin an interior maximizer by root-finding on the central-difference
    derivative.  Returns (x*, derivative residual at x*).

    Uses a guarded secant on g(x) = f'(x); the seed must come from a grid
    search so that x* is the nearest stationary point.
    """
    def g(x: float) -> float:
        return central_diff(f, x, h)

    # bracket the sign change of g around the seed when possible
    a, b = x0 - span, x0 + span
    ga, gb = g(a), g(b)
    if ga > 0.0 > gb:
        # bisection to shrink, then secant to finish
        for _ in range(40):
            m = 0.5 * (a + b)
            gm = g(m)
            if gm > 0.0:
                a, ga = m, gm
            else:
                b, gb = m, gm
        x, res = polish_root(g, 0.5 * (a + b), scale=1e-2, max_iter=20)
        return x, res
    # fall back to secant straight from the seed
    return polish_root(g, x0, scale=1e-2, max_iter=max_iter)


# Model-specific composite solvers.

def polished_cutoff(model: Model, d: float) -> Tuple[float, float]:
    """c*(d): bisection result + root polish on c -> G(c,d)."""
    seed = model.cutoff_result(d).value
    if not math.isfinite(seed):
        return seed, math.nan
    return polish_root(lambda c: model.G(c, d), seed, scale=1e-3)


def polish_PR(model: Model, d_seed: float,
              span: float = 5e-3) -> dict:
    """Polish d_PR as the stationary point of d -> U_S(d | c*(d))."""
    def U_hat(d: float) -> float:
        c, _ = polished_cutoff(model, d)
        return model.U_S(d, c)

    d_star, dres = polish_argmax(U_hat, d_seed, span=span)
    c_star, cres = polished_cutoff(model, d_star)
    return {"d": d_star, "c": c_star,
            "U_S": model.U_S(d_star, c_star),
            "U_R": model.U_R(d_star, c_star),
            "derivative_residual": dres, "cutoff_residual": cres}


def polish_pure_TR(model: Model, d_seed: float,
                   span: float = 5e-3) -> dict:
    """Polish the pure-TR fixed point: solve phi(d) = 0 where
    phi(d) = dU_S/d d~ evaluated at d~ = d against the cutoff c*(d)."""
    def phi(d: float) -> float:
        c, _ = polished_cutoff(model, d)
        return central_diff(lambda dd: model.U_S(dd, c), d)

    d_star, res = polish_root(phi, d_seed, scale=1e-2)
    c_star, cres = polished_cutoff(model, d_star)
    return {"d": d_star, "c": c_star,
            "U_S": model.U_S(d_star, c_star),
            "U_R": model.U_R(d_star, c_star),
            "foc_residual": res, "cutoff_residual": cres}


def polish_mixed(model: Model, dL0: float, dH0: float, c0: float,
                 rounds: int = 12) -> dict:
    """Polish (d_L, d_H, c): branch stationarity + branch-value tie."""
    dL, dH, c = dL0, dH0, c0
    for _ in range(rounds):
        dL, _ = polish_argmax(lambda d: model.U_S(d, c), dL, span=5e-3)
        dH, _ = polish_argmax(lambda d: model.U_S(d, c), dH, span=5e-3)

        def tie(cc: float, dl=dL, dh=dH) -> float:
            # re-polish branch maxima at the trial cutoff (envelope: cheap
            # one-step refinement is enough inside the tie root search)
            dl2, _ = polish_argmax(lambda d: model.U_S(d, cc), dl, span=2e-3)
            dh2, _ = polish_argmax(lambda d: model.U_S(d, cc), dh, span=2e-3)
            return model.U_S(dl2, cc) - model.U_S(dh2, cc)

        c_new, tie_res = polish_root(tie, c, scale=1e-3, max_iter=25)
        if abs(c_new - c) < 5e-14 * max(1.0, abs(c)):
            c = c_new
            break
        c = c_new
    dL, _ = polish_argmax(lambda d: model.U_S(d, c), dL, span=2e-3)
    dH, _ = polish_argmax(lambda d: model.U_S(d, c), dH, span=2e-3)
    nL = model.pool_terms(c, dL)[1]
    nH = model.pool_terms(c, dH)[1]
    w_L = nH / (nH - nL)
    tie_final = model.U_S(dL, c) - model.U_S(dH, c)
    balance = w_L * nL + (1.0 - w_L) * nH
    ur_mix = w_L * model.natural_value(c, dL) + (1.0 - w_L) * model.natural_value(c, dH)
    return {"d_L": dL, "d_H": dH, "c": c, "w_L": w_L, "w_H": 1.0 - w_L,
            "tie_residual": tie_final, "balance_residual": balance,
            "U_R_mix": ur_mix,
            "U_S_at_L": model.U_S(dL, c)}


def polish_dcp_dpp(model: Model, c: float, ur_mix: float,
                   dL: float, dH: float) -> dict:
    """Generalized inverses on the support hull, root-polished."""
    # seeds by coarse bisection
    def bisect(f, lo, hi, iters=60):
        flo = f(lo)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if (f(mid) >= 0.0) == (flo >= 0.0):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    d0_seed = bisect(lambda d: model.G(c, d), dL, dH)
    d0, g_res = polish_root(lambda d: model.G(c, d), d0_seed, scale=1e-2)
    d1_seed = bisect(lambda d: model.U_R(d) - ur_mix, dL, dH)
    d1, u_res = polish_root(lambda d: model.U_R(d) - ur_mix, d1_seed, scale=1e-2)
    return {"d_cp": d0, "G_residual": g_res,
            "c_star_dcp": polished_cutoff(model, d0)[0],
            "d_pp": d1, "UR_residual": u_res,
            "c_star_dpp": polished_cutoff(model, d1)[0],
            "U_R_dcp": model.U_R(d0)}
