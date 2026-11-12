"""Black–Scholes (Black-76) pricing in forward / log-moneyness coordinates.

Conventions used throughout the package
---------------------------------------
* ``F``  – forward price of the underlying for the option's expiry.
* ``K``  – strike.
* ``T``  – time to expiry in years.
* ``df`` – discount factor to expiry, ``exp(-r T)``.
* ``k = ln(K / F)`` – log-moneyness (``k < 0``: strike below the forward).
* ``s = sigma * sqrt(T)`` – total standard deviation ("total vol").

Working with the forward removes rates and dividends from the problem: the
*normalized* undiscounted price ``c = C / (df * F)`` depends only on ``k`` and
``s``. The out-of-the-money (OTM) option is computed directly and the
in-the-money one is recovered from put–call parity, so the time value of deep
OTM options is never lost to cancellation against the intrinsic value.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

__all__ = [
    "bs_price",
    "bs_vega",
    "forward_price",
    "log_moneyness",
    "normalized_call",
    "normalized_otm",
    "normalized_put",
    "normalized_vega",
    "price_bounds",
    "put_call_parity_gap",
]

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def forward_price(spot: ArrayLike, T: ArrayLike, r: float = 0.0, q: float = 0.0) -> NDArray:
    """Forward ``F = S exp((r - q) T)`` with continuous rate ``r`` and dividend yield ``q``."""
    return np.asarray(spot, dtype=float) * np.exp((r - q) * np.asarray(T, dtype=float))


def log_moneyness(K: ArrayLike, F: ArrayLike) -> NDArray:
    """``k = ln(K / F)``."""
    return np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float))


def _d1_d2(k: NDArray, s: NDArray) -> tuple[NDArray, NDArray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = -k / s + 0.5 * s
    return d1, d1 - s


def normalized_otm(k: ArrayLike, s: ArrayLike) -> NDArray:
    """Normalized price of the OTM option: the call for ``k >= 0``, the put for ``k < 0``.

    This equals the time value of either option at that strike (in units of
    ``df * F``). For ``s == 0`` the time value is zero.
    """
    k, s = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(s, dtype=float))
    d1, d2 = _d1_d2(k, s)
    call = ndtr(d1) - np.exp(k) * ndtr(d2)
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
    out = np.where(k >= 0.0, call, put)
    out = np.where(s > 0.0, out, 0.0)
    # Rounding can produce values a few ulps below zero far in the wings.
    return np.maximum(out, 0.0)


def normalized_call(k: ArrayLike, s: ArrayLike) -> NDArray:
    """Undiscounted call price divided by the forward, ``E[(e^X - e^k)^+]``."""
    k = np.asarray(k, dtype=float)
    intrinsic = np.maximum(-np.expm1(k), 0.0)  # max(1 - e^k, 0) without cancellation
    return normalized_otm(k, s) + intrinsic


def normalized_put(k: ArrayLike, s: ArrayLike) -> NDArray:
    """Undiscounted put price divided by the forward, ``E[(e^k - e^X)^+]``."""
    k = np.asarray(k, dtype=float)
    intrinsic = np.maximum(np.expm1(k), 0.0)  # max(e^k - 1, 0)
    return normalized_otm(k, s) + intrinsic


def normalized_vega(k: ArrayLike, s: ArrayLike) -> NDArray:
    """Derivative of the normalized price with respect to total vol ``s``: ``phi(d1)``.

    Identical for calls and puts. Multiply by ``df * F * sqrt(T)`` for the usual
    vega ``dC/dsigma``.
    """
    k, s = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(s, dtype=float))
    d1, _ = _d1_d2(k, s)
    with np.errstate(over="ignore", invalid="ignore"):
        v = _INV_SQRT_2PI * np.exp(-0.5 * d1 * d1)
    return np.where(np.isfinite(v), v, 0.0)


def bs_price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    df: ArrayLike = 1.0,
    is_call: ArrayLike = True,
) -> NDArray:
    """Discounted Black-76 price of a European call or put on the forward ``F``."""
    F = np.asarray(F, dtype=float)
    k = log_moneyness(K, F)
    s = np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(T, dtype=float))
    norm = np.where(np.asarray(is_call, dtype=bool), normalized_call(k, s), normalized_put(k, s))
    return np.asarray(df, dtype=float) * F * norm


def bs_vega(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> NDArray:
    """Vega ``dC/dsigma = df * F * phi(d1) * sqrt(T)`` (same for calls and puts)."""
    F = np.asarray(F, dtype=float)
    sqrt_t = np.sqrt(np.asarray(T, dtype=float))
    k = log_moneyness(K, F)
    s = np.asarray(sigma, dtype=float) * sqrt_t
    return np.asarray(df, dtype=float) * F * normalized_vega(k, s) * sqrt_t


def put_call_parity_gap(
    call: ArrayLike, put: ArrayLike, F: ArrayLike, K: ArrayLike, df: ArrayLike = 1.0
) -> NDArray:
    """Residual of put–call parity, ``C - P - df (F - K)``; zero for consistent prices."""
    return (
        np.asarray(call, dtype=float)
        - np.asarray(put, dtype=float)
        - np.asarray(df, dtype=float) * (np.asarray(F, dtype=float) - np.asarray(K, dtype=float))
    )


def price_bounds(
    F: ArrayLike, K: ArrayLike, df: ArrayLike = 1.0, is_call: ArrayLike = True
) -> tuple[NDArray, NDArray]:
    """Model-free no-arbitrage bounds on a European option price.

    Calls: ``df * max(F - K, 0) <= C <= df * F``.
    Puts:  ``df * max(K - F, 0) <= P <= df * K``.

    Any price strictly inside these bounds corresponds to exactly one
    Black–Scholes volatility in ``(0, inf)``.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    df = np.asarray(df, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)
    lower = np.where(is_call, df * np.maximum(F - K, 0.0), df * np.maximum(K - F, 0.0))
    upper = np.where(is_call, df * F, df * K)
    return lower, upper
