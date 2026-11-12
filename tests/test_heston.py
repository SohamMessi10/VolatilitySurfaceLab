import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from volsurface.black_scholes import normalized_otm, put_call_parity_gap
from volsurface.heston import (
    HestonError,
    HestonParams,
    _carr_madan,
    fft_log_strike_grid,
    heston_call_lewis,
    heston_cf,
    heston_otm_fft,
    heston_price,
    moment_explosion_time,
)

EQUITY = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)  # Feller ratio 0.56
POS_RHO = HestonParams(v0=0.09, kappa=1.0, theta=0.06, xi=1.0, rho=0.3)
LONG_TRAP = HestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=1.0, rho=-0.9)
EXTREME = HestonParams(v0=0.04, kappa=0.5, theta=0.04, xi=1.5, rho=-0.9)


def cf_ode(u: float, T: float, p: HestonParams) -> complex:
    """Independent reference: integrate the affine Riccati ODEs for the CF numerically."""

    def rhs(_t, y):
        D = y[1]
        dC = p.kappa * p.theta * D
        dD = 0.5 * p.xi**2 * D * D + (p.rho * p.xi * 1j * u - p.kappa) * D - 0.5 * (u * u + 1j * u)
        return [dC, dD]

    sol = solve_ivp(rhs, (0.0, T), [0j, 0j], method="DOP853", rtol=1e-12, atol=1e-14)
    C, D = sol.y[:, -1]
    return complex(np.exp(C + D * p.v0))


def cf_original_heston(u: float, T: float, p: HestonParams) -> complex:
    """Heston's (1993) original form with g = (beta + d) / (beta - d) and exp(+dT)."""
    iu = 1j * u
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + p.xi**2 * (iu + u * u))
    g = (beta + d) / (beta - d)
    e = np.exp(d * T)
    C = p.kappa * p.theta / p.xi**2 * ((beta + d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (beta + d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return complex(np.exp(C + D * p.v0))


def otm_lewis(k: np.ndarray, T: float, p: HestonParams) -> np.ndarray:
    c = heston_call_lewis(k, T, p)
    return np.where(k >= 0, c, c + np.expm1(k))


# --- characteristic function -------------------------------------------------------------


@pytest.mark.parametrize("p", [EQUITY, POS_RHO, LONG_TRAP])
@pytest.mark.parametrize("T", [0.05, 1.0, 10.0])
def test_cf_normalization_and_martingale(p, T):
    assert heston_cf(0.0, T, p) == pytest.approx(1.0, abs=1e-15)
    assert heston_cf(-1j, T, p) == pytest.approx(1.0, abs=1e-12)  # E[F_T / F_0] = 1


@pytest.mark.parametrize(("p", "T"), [(EQUITY, 1.0), (LONG_TRAP, 10.0)])
def test_cf_matches_riccati_ode(p, T):
    for u in (0.3, 1.0, 2.5, 5.0):
        assert abs(heston_cf(u, T, p) - cf_ode(u, T, p)) < 1e-12


def test_little_trap_avoids_original_branch_cut():
    # At T = 10 the original formula's complex log jumps branches: it disagrees with the
    # ODE reference by O(1), while the little-trap form agrees to ~1e-15.
    T, u = 10.0, 1.0
    ref = cf_ode(u, T, LONG_TRAP)
    assert abs(heston_cf(u, T, LONG_TRAP) - ref) < 1e-12
    assert abs(cf_original_heston(u, T, LONG_TRAP) - ref) > 0.1


# --- FFT vs direct integration -------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("p", [EQUITY, POS_RHO])
def test_fft_agrees_with_direct_integration(p):
    k = np.linspace(-1.5, 1.0, 26)
    maturities = (0.05, 0.25, 1.0, 3.0) + ((10.0,) if p is EQUITY else ())
    for T in maturities:
        err = np.max(np.abs(heston_otm_fft(k, T, p) - otm_lewis(k, T, p)))
        assert err < 5e-11, (T, err)  # normalized by the forward


def test_fft_rejects_unusable_damping_and_lewis_still_works():
    # Moments of order < -0.12 explode before T = 10: no safe damping for the put side.
    with pytest.raises(HestonError, match="lewis"):
        heston_otm_fft(np.array([-0.5, 0.5]), 10.0, EXTREME)
    prices = heston_price(100.0, [60.0, 100.0, 150.0], 10.0, EXTREME, is_call=False, method="lewis")
    assert np.all(np.isfinite(prices))
    assert np.all(np.diff(prices) > 0)


def test_raw_transform_refuses_exploded_moment():
    # Put damping -4 needs E[(F_T/F_0)^-3], which explodes at T* = 1.99 for EQUITY.
    t_star = moment_explosion_time(-3.0, EQUITY)
    assert t_star == pytest.approx(1.9923, abs=1e-4)
    with pytest.raises(HestonError, match="moment explosion"):
        _carr_madan(3.0, EQUITY, -4.0, 2**12, 0.25)


@pytest.mark.parametrize("p", [EQUITY, POS_RHO, EXTREME])
@pytest.mark.parametrize("omega", [-3.0, -1.5, -0.5, 2.5, 4.0, 10.0])
def test_moment_explosion_time_matches_riccati_ode(p, omega):
    # Real Riccati ODE for the moment generating function blows up at T*.
    def rhs(_t, y):
        B = y[0]
        return [
            0.5 * p.xi**2 * B * B
            + (p.rho * p.xi * omega - p.kappa) * B
            + 0.5 * (omega * omega - omega)
        ]

    def blowup(_t, y):
        return y[0] - 1e7

    blowup.terminal = True
    sol = solve_ivp(rhs, (0.0, 50.0), [0.0], events=blowup, rtol=1e-11, atol=1e-12)
    ode_t = sol.t_events[0][0] if len(sol.t_events[0]) else math.inf
    t_star = moment_explosion_time(omega, p)
    if math.isinf(ode_t):
        assert t_star > 50.0
    else:
        assert t_star == pytest.approx(ode_t, rel=1e-5)


# --- limits and no-arbitrage properties ----------------------------------------------------


@pytest.mark.parametrize("T", [0.1, 1.0, 5.0])
def test_black_scholes_limit(T):
    # xi -> 0 with v0 = theta: variance stays at theta, so Heston = BS at vol sqrt(theta).
    k = np.linspace(-1.0, 1.0, 21)
    bs = normalized_otm(k, math.sqrt(0.04 * T))
    exact = HestonParams(v0=0.04, kappa=1.5, theta=0.04, xi=0.0, rho=-0.7)
    assert np.max(np.abs(heston_otm_fft(k, T, exact) - bs)) < 2e-11
    assert np.max(np.abs(otm_lewis(k, T, exact) - bs)) < 1e-14
    # Convergence as xi -> 0: first order in xi when rho != 0 (skew), second order when rho = 0.
    for rho, order in ((-0.7, 1), (0.0, 2)):
        errs = [
            np.max(np.abs(heston_otm_fft(k, T, HestonParams(0.04, 1.5, 0.04, xi, rho)) - bs))
            for xi in (1e-2, 1e-3)
        ]
        assert errs[1] < errs[0] * 10.0 ** (-order) * 1.5
        assert errs[1] > errs[0] * 10.0 ** (-order) / 1.5


@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
def test_put_call_parity_between_independent_transforms(T):
    # The call side (damping +1.5) and put side (damping -2.5) are separate FFTs; parity
    # c - p = 1 - e^k must hold between them across the whole strike range.
    n, eta = 2**15, 0.1
    grid = fft_log_strike_grid(n, eta)
    calls = _carr_madan(T, EQUITY, 1.5, n, eta)
    puts = _carr_madan(T, EQUITY, -2.5, n, eta)
    sel = np.abs(grid) < 1.5
    gap = put_call_parity_gap(calls[sel], puts[sel], 1.0, np.exp(grid[sel]), 1.0)
    assert np.max(np.abs(gap)) < 1e-11
    # ...and between direct integration (calls) and the FFT (puts).
    k = np.array([-1.0, -0.3, 0.0, 0.4])
    c_lewis = heston_call_lewis(k, T, EQUITY)
    p_fft = np.where(k < 0, heston_otm_fft(k, T, EQUITY), np.nan)
    np.testing.assert_allclose(c_lewis[:2] - p_fft[:2], -np.expm1(k[:2]), atol=5e-11)


@pytest.mark.parametrize("p", [EQUITY, POS_RHO])
@pytest.mark.parametrize("T", [0.05, 0.5, 2.0])
def test_prices_monotone_and_convex_in_strike(p, T):
    F, df = 100.0, 0.97
    strikes = np.linspace(50.0, 200.0, 301)
    calls = heston_price(F, strikes, T, p, df, True)
    puts = heston_price(F, strikes, T, p, df, False)
    # Strictly monotone wherever the price is above the FFT accuracy floor (5e-11 * df * F);
    # below it (deep-OTM short-dated wings, true prices ~1e-30) only up to that floor.
    floor = 5e-11 * df * F
    assert np.all(np.diff(calls)[calls[1:] > 100 * floor] < 0)
    assert np.all(np.diff(puts)[puts[:-1] > 100 * floor] > 0)
    assert np.min(-np.diff(calls)) > -floor
    assert np.min(np.diff(puts)) > -floor
    # Butterfly spreads are worth >= 0 (allow FFT noise far below a cent).
    assert np.min(np.diff(calls, 2)) > -1e-9
    # Slopes bounded by the discount factor: -df <= dC/dK <= 0.
    assert np.all(np.diff(calls) / np.diff(strikes) >= -df - 1e-9)


def test_heston_price_methods_agree_in_price_units():
    F, df, T = 105.0, 0.95, 0.75
    strikes = np.array([70.0, 95.0, 105.0, 120.0, 150.0])
    fft = heston_price(F, strikes, T, EQUITY, df, [False, False, True, True, True])
    lewis = heston_price(
        F, strikes, T, EQUITY, df, [False, False, True, True, True], method="lewis"
    )
    np.testing.assert_allclose(fft, lewis, atol=1e-8)
    with pytest.raises(ValueError, match="unknown method"):
        heston_price(F, strikes, T, EQUITY, method="mc")


# --- parameters ---------------------------------------------------------------------------


def test_params_validation_and_serialization():
    assert EQUITY.feller_ratio == pytest.approx(2 * 2.0 * 0.05 / 0.36)
    assert HestonParams.from_dict(EQUITY.to_dict()) == EQUITY
    for bad in (
        {"v0": -0.1},
        {"kappa": 0.0},
        {"xi": -1.0},
        {"rho": 1.5},
        {"theta": float("nan")},
    ):
        with pytest.raises(HestonError):
            HestonParams(**{**EQUITY.to_dict(), **bad})
