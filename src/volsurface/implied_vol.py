"""Implied-volatility inversion: safeguarded Newton on vega with a Brent fallback.

The price is first mapped to the normalized time value of the OTM option
(``c = C / (df F)`` minus intrinsic), which is a strictly increasing function of
total vol ``s = sigma sqrt(T)`` on ``(0, inf)`` with range ``(0, upper)``. Newton's
method is started at the inflection point ``s* = sqrt(2 |k|)``: the price is
convex in ``s`` below ``s*`` and concave above it, so Newton iterates from ``s*``
move monotonically towards the root. When vega is tiny (deep OTM / very short
expiry) Newton steps become unreliable, so the solver falls back to
``scipy.optimize.brentq`` on a bracket that Newton has already established.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

__all__ = [
    "ImpliedVolError",
    "ImpliedVolResult",
    "PriceOutOfBoundsError",
    "implied_total_vol",
    "implied_vol",
    "implied_vol_array",
]

# Below this normalized vega a Newton step is considered unreliable.
VEGA_FLOOR = 1e-8
MAX_NEWTON_ITERATIONS = 40
S_MAX = 50.0  # total vol upper bracket; normalized OTM price is within 1e-16 of its cap here


class ImpliedVolError(ValueError):
    """Raised when an implied volatility cannot be computed."""


class PriceOutOfBoundsError(ImpliedVolError):
    """Raised when a price violates the no-arbitrage bounds (no implied vol exists)."""


@dataclass(frozen=True)
class ImpliedVolResult:
    """Implied volatility plus solver diagnostics."""

    sigma: float
    total_vol: float
    method: Literal["newton", "brent"]
    iterations: int


_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    # erfc keeps full relative accuracy in the lower tail (unlike 1 + erf).
    return 0.5 * math.erfc(-x / _SQRT2)


def _otm_scalar(k: float, s: float) -> float:
    """Scalar fast path of :func:`normalized_otm` (same formula, ``math`` instead of numpy)."""
    if s <= 0.0:
        return 0.0
    d1 = -k / s + 0.5 * s
    d2 = d1 - s
    if k >= 0.0:
        v = _ncdf(d1) - math.exp(k) * _ncdf(d2)
    else:
        v = math.exp(k) * _ncdf(-d2) - _ncdf(-d1)
    return max(v, 0.0)


def _vega_scalar(k: float, s: float) -> float:
    """Scalar fast path of :func:`normalized_vega`."""
    d1 = -k / s + 0.5 * s
    return _INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)


def implied_total_vol(target: float, k: float) -> tuple[float, str, int]:
    """Invert the normalized OTM time value ``target`` at log-moneyness ``k``.

    Returns ``(s, method, iterations)`` where ``s = sigma sqrt(T)``.
    ``target`` must lie strictly inside ``(0, cap)`` with ``cap = 1`` for
    ``k >= 0`` (OTM call) and ``cap = e^k`` for ``k < 0`` (OTM put).
    """
    cap = 1.0 if k >= 0.0 else math.exp(k)
    if not (0.0 < target < cap):
        raise PriceOutOfBoundsError(
            f"normalized OTM price {target!r} is outside the open interval (0, {cap!r})"
        )

    lo, hi = 0.0, math.inf  # bracket: f(lo) < 0 < f(hi)
    s = math.sqrt(2.0 * abs(k)) if k != 0.0 else math.sqrt(2.0 * math.pi) * target
    iterations = 0
    for iterations in range(1, MAX_NEWTON_ITERATIONS + 1):
        f = _otm_scalar(k, s) - target
        if f == 0.0:
            return s, "newton", iterations
        if f > 0.0:
            hi = min(hi, s)
        else:
            lo = max(lo, s)
        vega = _vega_scalar(k, s)
        if vega < VEGA_FLOOR:
            break  # tiny vega: hand over to Brent
        step = f / vega
        s_new = s - step
        if not (lo < s_new < hi):
            break  # step left the bracket: hand over to Brent
        if abs(step) <= 4.0 * np.finfo(float).eps * s_new:
            return s_new, "newton", iterations
        s = s_new

    # Brent fallback on an established bracket.
    if not math.isfinite(hi):
        hi = max(2.0 * s, 1.0)
        while _otm_scalar(k, hi) - target <= 0.0:
            if hi >= S_MAX:
                raise ImpliedVolError(
                    f"normalized price {target!r} is too close to its upper bound {cap!r} "
                    "to resolve an implied volatility in double precision"
                )
            hi = min(2.0 * hi, S_MAX)
    lo = max(lo, 0.0)
    root, info = brentq(
        lambda x: _otm_scalar(k, x) - target,
        lo,
        hi,
        xtol=1e-300,
        rtol=4.0 * np.finfo(float).eps,
        maxiter=500,
        full_output=True,
    )
    if not info.converged:  # pragma: no cover - brentq raises before this in practice
        raise ImpliedVolError(f"Brent failed to converge: {info.flag}")
    return float(root), "brent", iterations + info.iterations


def implied_vol(
    price: float,
    F: float,
    K: float,
    T: float,
    df: float = 1.0,
    is_call: bool = True,
    *,
    details: bool = False,
) -> float | ImpliedVolResult:
    """Black-76 implied volatility of a single European option price.

    Raises :class:`PriceOutOfBoundsError` if ``price`` is not strictly inside the
    no-arbitrage bounds (see :func:`volsurface.black_scholes.price_bounds`).
    """
    price, F, K, T, df = (float(x) for x in (price, F, K, T, df))
    if not (F > 0.0 and K > 0.0 and T > 0.0 and df > 0.0):
        raise ImpliedVolError(f"need F, K, T, df > 0; got F={F}, K={K}, T={T}, df={df}")
    if not math.isfinite(price):
        raise PriceOutOfBoundsError(f"price must be finite, got {price!r}")

    if is_call:
        lower, upper = df * max(F - K, 0.0), df * F
    else:
        lower, upper = df * max(K - F, 0.0), df * K
    kind = "call" if is_call else "put"
    if not (lower < price < upper):
        raise PriceOutOfBoundsError(
            f"{kind} price {price!r} violates the no-arbitrage bounds "
            f"({lower!r}, {upper!r}) for F={F}, K={K}, T={T}, df={df}; "
            "no implied volatility exists"
        )

    k = math.log(K / F)
    norm = price / (df * F)
    # Convert to the OTM option via put–call parity: c - p = 1 - e^k (normalized).
    if is_call and k < 0.0:
        target = norm + math.expm1(k)  # p = c - (1 - e^k)
    elif not is_call and k >= 0.0:
        target = norm - math.expm1(k)  # c = p + (1 - e^k)
    else:
        target = norm
    if target <= 0.0:
        raise PriceOutOfBoundsError(
            f"{kind} price {price!r} has no resolvable time value over intrinsic "
            f"({lower!r}) in double precision; no implied volatility exists"
        )

    s, method, iterations = implied_total_vol(target, k)
    sigma = s / math.sqrt(T)
    if details:
        return ImpliedVolResult(sigma, s, method, iterations)  # type: ignore[arg-type]
    return sigma


def implied_vol_array(
    prices: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    df: ArrayLike = 1.0,
    is_call: ArrayLike = True,
    *,
    errors: Literal["raise", "nan"] = "raise",
) -> NDArray:
    """Vectorized :func:`implied_vol` over broadcastable inputs.

    With ``errors="nan"`` prices without an implied vol yield ``nan`` instead of raising.
    """
    arrays = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (prices, F, K, T, df)),
        np.asarray(is_call, dtype=bool),
    )
    out = np.empty(arrays[0].shape, dtype=float)
    for idx in np.ndindex(out.shape):
        p, f, k, t, d, c = (a[idx] for a in arrays)
        try:
            out[idx] = implied_vol(p, f, k, t, d, bool(c))
        except ImpliedVolError:
            if errors == "raise":
                raise
            out[idx] = np.nan
    return out
