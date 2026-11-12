"""Heston (1993) stochastic-volatility model: characteristic function and pricers.

Dynamics under the pricing measure (forward ``F_t`` of the spot)::

    dF_t / F_t = sqrt(v_t) dW_t
    dv_t       = kappa (theta - v_t) dt + xi sqrt(v_t) dZ_t,   d<W, Z> = rho dt

Characteristic function
-----------------------
We use the "little Heston trap" form of Albrecher, Mayer, Schoutens & Tistaert
(2007), which writes the solution with ``exp(-d T)`` and ``g = (beta - d) / (beta + d)``
so that ``|g exp(-d T)| < 1`` and the complex logarithm does not jump branches for
long maturities (the original Heston form does). On top of that, every ``1/xi^2``
is removed algebraically using ``beta - d = -xi^2 (iu + u^2) / (beta + d)``, so the
formula has no catastrophic cancellation as the vol-of-vol ``xi -> 0`` and
reduces exactly to Black–Scholes at ``xi = 0``.

Pricers
-------
* :func:`heston_otm_fft` – Carr & Madan (1999) FFT with damping ``alpha`` and Simpson
  weights. Calls (damping ``+alpha``) are used for ``k >= 0`` and puts (mirrored
  damping ``-1 - alpha``, Lee 2004) for ``k < 0``, so the factor ``exp(-alpha k)`` is
  always <= 1 and OTM time values keep their absolute accuracy. The damping is capped
  per expiry at half the admissible range given by the critical moments (Andersen &
  Piterbarg 2007), so aliasing stays negligible; the raw transform refuses exploded
  moments rather than silently mispricing.
* :func:`heston_call_lewis` – direct adaptive quadrature (``scipy.integrate.quad``) of
  Lewis' (2001) single-integral formula; an independent check on the FFT.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .black_scholes import log_moneyness

__all__ = [
    "DampingError",
    "HestonError",
    "HestonParams",
    "critical_moments",
    "fft_log_strike_grid",
    "heston_call_lewis",
    "heston_cf",
    "heston_otm_fft",
    "heston_price",
    "moment_explosion_time",
]


class HestonError(ValueError):
    """Invalid Heston parameters or a failed transform."""


class DampingError(HestonError):
    """No Carr–Madan damping is usable at this expiry (the needed moments explode)."""


@dataclass(frozen=True)
class HestonParams:
    """Heston parameters.

    v0: initial variance; kappa: mean-reversion speed; theta: long-run variance;
    xi: volatility of variance ("vol-of-vol"); rho: spot/variance correlation.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise HestonError(f"Heston parameters must be finite: {self}")
        if self.v0 < 0 or self.theta < 0:
            raise HestonError("v0 and theta must be >= 0")
        if self.kappa <= 0:
            raise HestonError("kappa must be > 0")
        if self.xi < 0:
            raise HestonError("xi (vol-of-vol) must be >= 0")
        if not -1.0 <= self.rho <= 1.0:
            raise HestonError("rho must be in [-1, 1]")

    @property
    def feller_ratio(self) -> float:
        """``2 kappa theta / xi^2``; >= 1 means the variance process never hits zero."""
        return math.inf if self.xi == 0 else 2.0 * self.kappa * self.theta / self.xi**2

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> HestonParams:
        return cls(**{name: float(d[name]) for name in ("v0", "kappa", "theta", "xi", "rho")})


def _log1p_over_z(z: NDArray) -> NDArray:
    """``log(1 + z) / z`` for complex ``z``, accurate for tiny ``|z|``.

    ``numpy.log1p`` on complex input is not accurate for small arguments, so we use
    Kahan's trick ``log(w) / (w - 1)`` with ``w = fl(1 + z)``: the rounding error in
    ``w`` cancels between numerator and denominator.
    """
    w = 1.0 + z
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(w) / (w - 1.0)
    return np.where(w == 1.0, 1.0 - 0.5 * z, r)


def heston_cf(u: ArrayLike, T: float, p: HestonParams) -> NDArray:
    """Characteristic function ``E[exp(i u X_T)]`` of ``X_T = ln(F_T / F_0)``.

    ``u`` may be complex (the FFT evaluates it at ``v - (alpha + 1) i``).
    """
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    q = iu + u * u  # iu + u^2
    beta = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(beta * beta + p.xi * p.xi * q)  # principal branch: Re(d) >= 0
    bpd = beta + d
    gm = -q / bpd  # (beta - d) / xi^2, computed without cancellation
    g = p.xi * p.xi * gm / bpd  # (beta - d) / (beta + d), the "little trap" g
    e = np.exp(-d * T)
    one_minus_ge = 1.0 - g * e
    D = gm * (1.0 - e) / one_minus_ge
    # log((1 - g e) / (1 - g)) / xi^2 = (z / xi^2) * log1p(z) / z with z = g (1 - e) / (1 - g)
    z_over_xi2 = (gm / bpd) * (1.0 - e) / (1.0 - g)
    z = p.xi * p.xi * z_over_xi2
    C = p.kappa * p.theta * (gm * T - 2.0 * z_over_xi2 * _log1p_over_z(z))
    return np.exp(C + D * p.v0)


def moment_explosion_time(omega: float, p: HestonParams) -> float:
    """Time ``T*`` after which ``E[(F_T / F_0)^omega]`` is infinite (Andersen & Piterbarg 2007).

    Moments with ``0 <= omega <= 1`` never explode. Carr–Madan damping ``alpha`` needs
    the moment of order ``omega = alpha + 1`` to be finite at the option's expiry;
    past ``T*`` the characteristic function formula still returns finite numbers, but
    they belong to an analytic continuation and the prices are wrong.
    """
    if 0.0 <= omega <= 1.0:
        return math.inf
    k = p.rho * p.xi * omega - p.kappa
    disc = k * k - p.xi * p.xi * (omega * omega - omega)
    if disc >= 0.0:
        if k < 0.0:
            return math.inf
        gamma = math.sqrt(disc)
        if gamma == 0.0:
            return 2.0 / k
        return math.log((k + gamma) / (k - gamma)) / gamma
    beta = math.sqrt(-disc)
    angle = math.pi / 2.0 if k == 0.0 else math.atan(beta / k) + (math.pi if k < 0.0 else 0.0)
    return 2.0 / beta * angle


def fft_log_strike_grid(n: int = 2**15, eta: float = 0.1) -> NDArray:
    """Log-strike grid ``k_u = -b + lambda u`` used by :func:`heston_otm_fft`."""
    lam = 2.0 * np.pi / (n * eta)
    return -0.5 * n * lam + lam * np.arange(n)


def _carr_madan(T: float, p: HestonParams, alpha: float, n: int, eta: float) -> NDArray:
    """Normalized undiscounted call (alpha > 0) or put (alpha < -1) prices on the FFT grid.

    Lee (2004): the same transform with ``alpha < -1`` yields put prices.
    """
    t_star = moment_explosion_time(alpha + 1.0, p)
    if t_star <= T:
        raise HestonError(
            f"moment E[(F_T/F_0)^{alpha + 1.0:g}] needed by damping alpha={alpha:g} is infinite "
            f"for T={T:g} >= T*={t_star:.4g} (moment explosion); use a smaller |alpha|"
        )
    j = np.arange(n)
    v = eta * j
    lam = 2.0 * np.pi / (n * eta)
    b = 0.5 * n * lam
    psi = heston_cf(v - (alpha + 1.0) * 1j, T, p) / (
        alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    )
    if not np.all(np.isfinite(psi)):
        raise HestonError(
            f"characteristic function not finite at damping alpha={alpha}; try a smaller |alpha|"
        )
    simpson = (3.0 + (-1.0) ** (j + 1)) / 3.0
    simpson[0] = 1.0 / 3.0
    x = np.exp(1j * b * v) * psi * eta * simpson
    k = -b + lam * j
    return np.exp(-alpha * k) / np.pi * np.fft.fft(x).real


MIN_DAMPING = 0.25  # below this the integrand's poles sit too close to the real axis


def critical_moments(T: float, p: HestonParams, *, cap: float = 64.0) -> tuple[float, float]:
    """Moment orders ``(omega_minus, omega_plus)`` where ``E[(F_T/F_0)^omega]`` becomes infinite.

    Moments are finite for ``omega_minus < omega < omega_plus``; infinite bounds are
    reported as ``-inf`` / ``inf`` when nothing explodes within ``|omega| <= cap``.
    """

    def boundary(lo: float, hi: float) -> float:
        # T*(omega) decreases as omega moves away from [0, 1]; ``lo`` is finite, ``hi`` exploded.
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if moment_explosion_time(mid, p) > T:
                lo = mid
            else:
                hi = mid
        return lo

    plus = math.inf if moment_explosion_time(1.0 + cap, p) > T else boundary(1.0, 1.0 + cap)
    minus = -math.inf if moment_explosion_time(-cap, p) > T else boundary(0.0, -cap)
    return minus, plus


def _damping(alpha: float, side: str, T: float, p: HestonParams) -> float:
    """Damping for one side: ``+a`` for calls, the mirror ``-1 - a`` for puts.

    The Carr–Madan integrand has poles at distances ``|a|`` and ``|a + 1|`` from the
    real axis, so the mirror gives the put side exactly the call side's regularity.
    The damped price decays like ``exp(-(omega_crit - omega) |k|)``; to keep FFT
    aliasing negligible, ``a`` is capped at half of the largest admissible value
    (Carr & Madan suggest a fraction of the admissible range).
    """
    minus, plus = critical_moments(T, p)
    # call: omega = a + 1 < plus  -> a_max = plus - 1;  put: omega = -a > minus -> a_max = -minus
    a_max = plus - 1.0 if side == "call" else -minus
    a = min(alpha, 0.5 * a_max)
    if a < MIN_DAMPING:
        raise DampingError(
            f"Carr-Madan damping for the {side} side at T={T:g} would be {a:.3g} "
            f"(moments explode beyond order {plus if side == 'call' else minus:.3g}); "
            "the FFT is unreliable here - use method='lewis'"
        )
    return a if side == "call" else -1.0 - a


def heston_otm_fft(
    k: ArrayLike,
    T: float,
    p: HestonParams,
    *,
    alpha: float = 1.5,
    n: int = 2**15,
    eta: float = 0.1,
) -> NDArray:
    """Normalized OTM option prices (time values) at log-moneyness ``k`` by Carr–Madan FFT.

    Returns ``E[(e^X - e^k)^+]`` for ``k >= 0`` and ``E[(e^k - e^X)^+]`` for ``k < 0``.
    Values between FFT nodes (spacing ``2 pi / (n eta)``) use a cubic spline.
    """
    if not alpha > 0.0:
        raise HestonError("alpha must be > 0 (calls use +alpha, puts use -1 - alpha)")
    k = np.asarray(k, dtype=float)
    grid = fft_log_strike_grid(n, eta)
    if k.size and (k.min() < grid[1] or k.max() > grid[-2]):
        raise HestonError(f"log-moneyness outside the FFT grid [{grid[1]:.2f}, {grid[-2]:.2f}]")
    out = np.empty_like(k)
    pos = k >= 0.0
    for mask, side in ((pos, "call"), (~pos, "put")):
        if np.any(mask):
            a = _damping(alpha, side, T, p)
            # Both sides have their nearest pole at distance a_mag; keep eta / a_mag fixed
            # (Simpson error ~ (eta / distance)^4) and n * eta fixed (same strike spacing).
            a_mag = a if side == "call" else -1.0 - a
            scale = max(1.0, alpha / a_mag)
            n_side = n * (1 << math.ceil(math.log2(scale))) if scale > 1.0 else n
            eta_side = n * eta / n_side
            grid = fft_log_strike_grid(n_side, eta_side)
            prices = _carr_madan(T, p, a, n_side, eta_side)
            sel = (grid > k[mask].min() - 0.5) & (grid < k[mask].max() + 0.5)
            out[mask] = CubicSpline(grid[sel], prices[sel])(k[mask])
    return np.maximum(out, 0.0)


def heston_call_lewis(
    k: ArrayLike, T: float, p: HestonParams, *, epsabs: float = 1e-14, limit: int = 2000
) -> NDArray:
    """Normalized undiscounted call ``E[(e^X - e^k)^+]`` by direct numerical integration.

    Lewis (2001), integrated with ``scipy.integrate.quad`` on ``[0, inf)``::

        c(k) = 1 - e^{k/2} / pi * int_0^inf Re[e^{-iuk} phi(u - i/2)] / (u^2 + 1/4) du
    """
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    out = np.empty_like(k_arr)
    for i, kk in enumerate(k_arr):

        def integrand(u: float, kk: float = kk) -> float:
            val = np.exp(-1j * u * kk) * heston_cf(u - 0.5j, T, p) / (u * u + 0.25)
            return float(val.real)

        integral, _ = quad(integrand, 0.0, np.inf, epsabs=epsabs, epsrel=1e-12, limit=limit)
        out[i] = 1.0 - math.exp(0.5 * kk) / math.pi * integral
    return out.reshape(np.shape(k))


def heston_price(
    F: float,
    K: ArrayLike,
    T: float,
    p: HestonParams,
    df: float = 1.0,
    is_call: ArrayLike = True,
    *,
    method: str = "fft",
    **kwargs,
) -> NDArray:
    """Discounted Heston prices of European options on forward ``F`` (one expiry).

    ``method="fft"`` (Carr–Madan), ``"lewis"`` (direct integration) or ``"auto"``: FFT,
    falling back to Lewis when no FFT damping is usable at this expiry. Lewis only needs
    the moment of order 1/2, which never explodes in Heston, so it is always valid.
    """
    if method == "auto":
        try:
            return heston_price(F, K, T, p, df, is_call, method="fft", **kwargs)
        except DampingError:
            return heston_price(F, K, T, p, df, is_call, method="lewis")
    k = log_moneyness(K, F)
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), k.shape)
    if method == "fft":
        otm = heston_otm_fft(k, T, p, **kwargs)
        call = otm + np.maximum(-np.expm1(k), 0.0)
    elif method == "lewis":
        call = heston_call_lewis(k, T, p, **kwargs)
    else:
        raise ValueError(f"unknown method {method!r}; use 'fft', 'lewis' or 'auto'")
    put = call + np.expm1(k)  # normalized parity: p = c - (1 - e^k)
    return df * F * np.where(is_call, call, put)
