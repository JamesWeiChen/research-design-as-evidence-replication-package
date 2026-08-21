r"""Analytic precision layer: pin flat-direction coordinates to ~1e-12.

Derivation (quadratic technology, any prior with exact window moments):

    U_S(d | c) = V[1-F(x)] + \int_{x-Y}^{x} [V - k(d)(x-t)^2] dF(t) - C0(d),
    x = c - alpha d,   k(d) = c_a omega(d)/2,   k(d) Y(d)^2 = V.

Because the integrand vanishes at the lower endpoint (k Y^2 = V) and the
two boundary terms at the upper endpoint cancel against the natural-mass
term, the derivatives are pure window-moment expressions:

    dU_S/dd = 2 alpha k(d) M1 - k'(d) M2 - C0'(d),
    dU_S/dc = -2 k(d) M1,

with M1 = \int (x-t) dF and M2 = \int (x-t)^2 dF over the pool window
(x-Y, x], both available EXACTLY from `prior.window_moments(center=x)`
(M1 = -m1_centered, M2 = m2_centered), and

    omega' = delta'(d) c_a beta^2 / (c_p + delta + c_a beta^2)^2,
    k'   = c_a omega'/2,          delta'(d) = 2 delta0 d (quadratic default),
    Y'   = -Y omega' / (2 omega).

For the PR objective U_hat(d) = U_S(d | c*(d)):

    dU_hat/dd = dU_S/dd + dU_S/dc * c*'(d),   c*'(d) = -G_d / G_c,

with G = omega (c-r) + (1-omega)(alpha d + m - r), m the pool conditional
mean; G_c and G_d use the prior pdf at the window endpoints (pdf obtained
by a tight central difference of the exact window mass, error ~1e-12).

"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

from solver.general_solver import Model

from .polish import central_diff, polish_root, polished_cutoff


# Exact building blocks.

def _window(model: Model, d: float, c: float):
    """Return (x, Y, m0, M1, M2) for the pool window (x-Y, x]."""
    P = model.P
    x = c - P.alpha * d
    Y = P.Y(d)
    m0, m1c, m2c = model.F.window_moments(
        np.asarray([x - Y]), np.asarray([x]), center=x)[:, 0]
    return x, Y, float(m0), -float(m1c), float(m2c)


def _k_and_prime(model: Model, d: float) -> Tuple[float, float, float, float]:
    """Return (omega, omega', k, k') for the quadratic technology."""
    P = model.P
    A = P.c_p + P.delta_of(d)
    B = P.c_a * P.beta ** 2
    omega = A / (A + B)
    ddelta = 2.0 * P.delta0 * d if P.delta is None else central_diff(P.delta_of, d)
    omega_p = ddelta * B / (A + B) ** 2
    return omega, omega_p, 0.5 * P.c_a * omega, 0.5 * P.c_a * omega_p


def dUS_dd(model: Model, d: float, c: float) -> float:
    """Exact partial derivative of the scholar payoff in the design."""
    P = model.P
    _, _, _, M1, M2 = _window(model, d, c)
    _, _, k, k_p = _k_and_prime(model, d)
    C0_p = P.c_d * d if P.C0 is None else central_diff(P.C0_of, d)
    return 2.0 * P.alpha * k * M1 - k_p * M2 - C0_p


def dUS_dc(model: Model, d: float, c: float) -> float:
    _, _, _, M1, _ = _window(model, d, c)
    _, _, k, _ = _k_and_prime(model, d)
    return -2.0 * k * M1


def _pdf(model: Model, x: float, h: float = 1e-5) -> float:
    return model.F.mass(x - h, x + h) / (2.0 * h)


def c_star_prime(model: Model, d: float, c: float) -> float:
    """c*'(d) = -G_d/G_c by the implicit function theorem (linear u)."""
    P = model.P
    x, Y, m0, M1, _ = _window(model, d, c)
    omega, omega_p, _, _ = _k_and_prime(model, d)
    m = x - M1 / m0                    # pool conditional mean of t
    fb = _pdf(model, x)
    fa = _pdf(model, x - Y)
    dm_db = (x - m) * fb / m0
    dm_da = (m - (x - Y)) * fa / m0
    Y_p = -0.5 * Y * omega_p / omega
    dm_dc = dm_db + dm_da              # both endpoints shift with c
    dm_dd = dm_db * (-P.alpha) + dm_da * (-P.alpha - Y_p)
    G_c = omega + (1.0 - omega) * dm_dc
    G_d = (omega_p * (c - P.r)
           - omega_p * (P.alpha * d + m - P.r)
           + (1.0 - omega) * (P.alpha + dm_dd))
    return -G_d / G_c


# Polished solvers using analytic first-order conditions.

def polish_PR_analytic(model: Model, d_seed: float) -> dict:
    def T(d: float) -> float:
        c, _ = polished_cutoff(model, d)
        return dUS_dd(model, d, c) + dUS_dc(model, d, c) * c_star_prime(model, d, c)

    d_star, res = polish_root(T, d_seed, scale=1e-2)
    c_star, cres = polished_cutoff(model, d_star)
    return {"d": d_star, "c": c_star,
            "U_S": model.U_S(d_star, c_star),
            "U_R": model.U_R(d_star, c_star),
            "foc_residual": res, "cutoff_residual": cres}


def polish_pure_TR_analytic(model: Model, d_seed: float) -> dict:
    def phi(d: float) -> float:
        c, _ = polished_cutoff(model, d)
        return dUS_dd(model, d, c)

    d_star, res = polish_root(phi, d_seed, scale=1e-2)
    c_star, cres = polished_cutoff(model, d_star)
    return {"d": d_star, "c": c_star,
            "U_S": model.U_S(d_star, c_star),
            "U_R": model.U_R(d_star, c_star),
            "foc_residual": res, "cutoff_residual": cres}


def polish_mixed_analytic(model: Model, dL0: float, dH0: float, c0: float,
                          rounds: int = 30) -> dict:
    dL, dH, c = dL0, dH0, c0
    for _ in range(rounds):
        dL, _ = polish_root(lambda d: dUS_dd(model, d, c), dL, scale=1e-2)
        dH, _ = polish_root(lambda d: dUS_dd(model, d, c), dH, scale=1e-2)
        tie = model.U_S(dL, c) - model.U_S(dH, c)
        # envelope theorem: at branch optima, d(tie)/dc = dUS_dc(L) - dUS_dc(H)
        slope = dUS_dc(model, dL, c) - dUS_dc(model, dH, c)
        step = -tie / slope
        c_new = c + step
        if abs(step) < 1e-15 * max(1.0, abs(c)):
            c = c_new
            break
        c = c_new
    dL, focL = polish_root(lambda d: dUS_dd(model, d, c), dL, scale=1e-2)
    dH, focH = polish_root(lambda d: dUS_dd(model, d, c), dH, scale=1e-2)
    nL = model.pool_terms(c, dL)[1]
    nH = model.pool_terms(c, dH)[1]
    w_L = nH / (nH - nL)
    # ulp nudge: among float neighbours of w_L, keep the one whose computed
    # balance residual is smallest (w_L has one-ulp freedom; the exact
    # balancing weight is generally not a float)
    candidates = [w_L]
    for _ in range(3):
        candidates.append(math.nextafter(candidates[-1], 0.0))
    up = w_L
    for _ in range(3):
        up = math.nextafter(up, 1.0)
        candidates.append(up)
    w_L = min(candidates, key=lambda w: abs(w * nL + (1.0 - w) * nH))
    ur_mix = w_L * model.natural_value(c, dL) + (1.0 - w_L) * model.natural_value(c, dH)
    return {"d_L": dL, "d_H": dH, "c": c, "w_L": w_L, "w_H": 1.0 - w_L,
            "tie_residual": model.U_S(dL, c) - model.U_S(dH, c),
            "balance_residual": w_L * nL + (1.0 - w_L) * nH,
            "foc_L": focL, "foc_H": focH,
            "U_R_mix": ur_mix}


# Derivative self-check.

def self_check(model: Model, points) -> float:
    """Return the worst |analytic - central| relative gap over the probes."""
    worst = 0.0
    for d, c in points:
        a1 = dUS_dd(model, d, c)
        n1 = central_diff(lambda dd: model.U_S(dd, c), d)
        a2 = dUS_dc(model, d, c)
        n2 = central_diff(lambda cc: model.U_S(d, cc), c)
        worst = max(worst,
                    abs(a1 - n1) / max(1.0, abs(a1)),
                    abs(a2 - n2) / max(1.0, abs(a2)))
        cs = c_star_prime(model, d, polished_cutoff(model, d)[0])
        h = 1e-5 * max(1.0, abs(d))
        num = (polished_cutoff(model, d + h)[0]
               - polished_cutoff(model, d - h)[0]) / (2.0 * h)
        worst = max(worst, abs(cs - num) / max(1.0, abs(cs)))
    return worst
