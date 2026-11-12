"""Raw SVI parametrization of total implied variance and slice/surface fitting.

Raw SVI (Gatheral 2004) models the total implied variance of one expiry as

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),

where k = ln(K / F) is log-moneyness and w(k) = sigma_iv(k)^2 * T.

Why total variance: w is the natural object for no-arbitrage conditions. The
calendar condition is simply "w non-decreasing in T at fixed k", the Durrleman
butterfly condition is written in w and its k-derivatives, and the wings of w
are asymptotically linear in k (Lee's moment formula), which SVI captures
exactly. Fitting in w also avoids dividing by T and keeps residuals comparable
across expiries.

Constraint handling. The minimum of the smile is a + b * sigma * sqrt(1 - rho^2);
requiring it to be non-negative (w >= 0 everywhere) is the only hard constraint
besides b >= 0, |rho| < 1 and sigma > 0. We optimise over x = (c, b, rho, m, sigma)
with box bounds and set a = c - b * sigma * sqrt(1 - rho^2). Then the minimum
variance equals c >= 0 for every point in the box, so the constraints hold
exactly (no penalties, no projection) and the optimiser never sees an invalid
parameter set.

Wing bound. Roger Lee's moment formula caps the asymptotic slope of total
variance at 2, i.e. b * (1 + |rho|) <= 2 for SVI. With only a handful of quotes
per short expiry an unconstrained fit can extrapolate to steeper wings, which
produces butterfly and calendar arbitrage outside the quoted range. The fitter
therefore optimises beta in [0, 1] with b = 2 * beta / (1 + |rho|), enforcing the
Lee bound exactly by the same reparametrization trick.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

__all__ = [
    "SVIParams",
    "SliceFit",
    "fit_svi_slice",
    "fit_svi_surface",
    "svi_derivatives",
    "svi_implied_vol",
    "svi_total_variance",
]

_MIN_VAR_TOL = 1e-12
_RHO_MAX = 0.999
_SIGMA_MIN = 1e-4
_SIGMA_MAX = 5.0


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters (a, b, rho, m, sigma) for one expiry."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        vals = (self.a, self.b, self.rho, self.m, self.sigma)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError(f"SVI parameters must be finite, got {vals}")
        if self.b < 0:
            raise ValueError(f"b must be >= 0, got {self.b}")
        if not abs(self.rho) < 1:
            raise ValueError(f"|rho| must be < 1, got {self.rho}")
        if not self.sigma > 0:
            raise ValueError(f"sigma must be > 0, got {self.sigma}")
        min_var = self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho**2)
        if min_var < -_MIN_VAR_TOL:
            raise ValueError(f"minimum total variance must be >= 0, got {min_var}")

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable dict."""
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SVIParams:
        """Build from a dict produced by :meth:`to_dict`."""
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            rho=float(d["rho"]),
            m=float(d["m"]),
            sigma=float(d["sigma"]),
        )


def svi_total_variance(k: ArrayLike, p: SVIParams) -> NDArray[np.float64]:
    """Total implied variance w(k) for raw SVI."""
    x = np.asarray(k, dtype=float) - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma**2))


def svi_derivatives(
    k: ArrayLike, p: SVIParams
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (w, w', w'') with analytic derivatives in k."""
    x = np.asarray(k, dtype=float) - p.m
    s = np.sqrt(x * x + p.sigma**2)
    w = p.a + p.b * (p.rho * x + s)
    w1 = p.b * (p.rho + x / s)
    w2 = p.b * p.sigma**2 / s**3
    return w, w1, w2


def svi_implied_vol(k: ArrayLike, T: float, p: SVIParams) -> NDArray[np.float64]:
    """Black-Scholes implied vol sqrt(w(k) / T)."""
    if not T > 0:
        raise ValueError(f"T must be > 0, got {T}")
    w = np.maximum(svi_total_variance(k, p), 0.0)
    return np.sqrt(w / T)


@dataclass(frozen=True)
class SliceFit:
    """Result of fitting one expiry slice."""

    T: float
    params: SVIParams
    n_points: int
    k_min: float
    k_max: float
    rmse_vol_points: float
    max_abs_vol_points: float
    cost: float
    success: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "T": float(self.T),
            "params": self.params.to_dict(),
            "n_points": int(self.n_points),
            "k_min": float(self.k_min),
            "k_max": float(self.k_max),
            "rmse_vol_points": float(self.rmse_vol_points),
            "max_abs_vol_points": float(self.max_abs_vol_points),
            "cost": float(self.cost),
            "success": bool(self.success),
            "message": str(self.message),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SliceFit:
        """Build from a dict produced by :meth:`to_dict`."""
        return cls(
            T=float(d["T"]),
            params=SVIParams.from_dict(d["params"]),
            n_points=int(d["n_points"]),
            k_min=float(d["k_min"]),
            k_max=float(d["k_max"]),
            rmse_vol_points=float(d["rmse_vol_points"]),
            max_abs_vol_points=float(d["max_abs_vol_points"]),
            cost=float(d["cost"]),
            success=bool(d["success"]),
            message=str(d["message"]),
        )


def _unpack(x: NDArray[np.float64]) -> tuple[float, float, float, float, float]:
    """Optimiser coordinates (c, beta, rho, m, sigma) -> raw SVI (a, b, rho, m, sigma)."""
    c, beta, rho, m, sigma = (float(v) for v in x)
    b = 2.0 * beta / (1.0 + abs(rho))  # Lee bound b (1 + |rho|) <= 2 for beta <= 1
    a = c - b * sigma * math.sqrt(1.0 - rho * rho)  # min variance = c >= 0
    return a, b, rho, m, sigma


def _params_from_x(x: NDArray[np.float64]) -> SVIParams:
    return SVIParams(*_unpack(x))


def _model_w(x: NDArray[np.float64], k: NDArray[np.float64]) -> NDArray[np.float64]:
    a, b, rho, m, sigma = _unpack(x)
    u = k - m
    return a + b * (rho * u + np.sqrt(u * u + sigma * sigma))


def _model_jac(x: NDArray[np.float64], k: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytic Jacobian of :func:`_model_w` w.r.t. (c, beta, rho, m, sigma).

    Writing w = c + b G with G = rho u + s - sigma sqrt(1 - rho^2), u = k - m,
    s = sqrt(u^2 + sigma^2) and b = 2 beta / (1 + |rho|).
    """
    _c, beta, rho, m, sigma = (float(v) for v in x)
    abs_rho = abs(rho)
    b = 2.0 * beta / (1.0 + abs_rho)
    root = math.sqrt(1.0 - rho * rho)
    u = k - m
    s = np.sqrt(u * u + sigma * sigma)
    G = rho * u + s - sigma * root
    jac = np.empty((k.size, 5))
    jac[:, 0] = 1.0
    jac[:, 1] = 2.0 / (1.0 + abs_rho) * G
    db_drho = -b * math.copysign(1.0, rho) / (1.0 + abs_rho) if rho != 0.0 else 0.0
    jac[:, 2] = db_drho * G + b * (u + sigma * rho / root)
    jac[:, 3] = -b * (rho + u / s)
    jac[:, 4] = b * (sigma / s - root)
    return jac


def fit_svi_slice(
    k: ArrayLike,
    iv: ArrayLike,
    T: float,
    *,
    weights: ArrayLike | None = None,
) -> SliceFit:
    """Fit raw SVI to one expiry by least squares on total implied variance.

    Parameters
    ----------
    k : log-moneyness ln(K/F), at least 5 points.
    iv : market Black-Scholes implied vols at ``k``.
    T : expiry in years.
    weights : optional per-point multipliers applied to the residuals.
    """
    k_arr = np.asarray(k, dtype=float).ravel()
    iv_arr = np.asarray(iv, dtype=float).ravel()
    if k_arr.shape != iv_arr.shape:
        raise ValueError("k and iv must have the same length")
    if k_arr.size < 5:
        raise ValueError(f"need at least 5 points to fit SVI, got {k_arr.size}")
    if not T > 0:
        raise ValueError(f"T must be > 0, got {T}")
    if not (np.all(np.isfinite(k_arr)) and np.all(np.isfinite(iv_arr))):
        raise ValueError("k and iv must be finite")
    if weights is None:
        wts = np.ones_like(k_arr)
    else:
        wts = np.asarray(weights, dtype=float).ravel()
        if wts.shape != k_arr.shape:
            raise ValueError("weights must match k in length")

    order = np.argsort(k_arr, kind="stable")
    k_arr, iv_arr, wts = k_arr[order], iv_arr[order], wts[order]
    w_mkt = iv_arr**2 * T

    k_min, k_max = float(k_arr[0]), float(k_arr[-1])
    lower = np.array([0.0, 0.0, -_RHO_MAX, k_min - 1.0, _SIGMA_MIN])
    upper = np.array([np.inf, 1.0, _RHO_MAX, k_max + 1.0, _SIGMA_MAX])

    def resid(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (_model_w(x, k_arr) - w_mkt) * wts

    def jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _model_jac(x, k_arr) * wts[:, None]

    k_atm_min = float(k_arr[int(np.argmin(w_mkt))])
    w_min = max(float(w_mkt.min()), 1e-8)
    # Wing slopes of total variance give a starting scale for b.
    slope_l = (w_mkt[1] - w_mkt[0]) / max(k_arr[1] - k_arr[0], 1e-8)
    slope_r = (w_mkt[-1] - w_mkt[-2]) / max(k_arr[-1] - k_arr[-2], 1e-8)
    b0 = float(np.clip(0.5 * (abs(slope_l) + abs(slope_r)), 0.01, 1.0))
    beta0 = min(b0 / 2.0, 0.5)  # b = 2 beta / (1 + |rho|)

    best = None
    for rho0, m0, sig0 in itertools.product((-0.6, 0.0, 0.6), (k_atm_min, 0.0), (0.05, 0.2, 0.5)):
        x0 = np.array([w_min, beta0 * (1.0 + abs(rho0)), rho0, m0, sig0])
        x0 = np.clip(x0, lower, np.where(np.isfinite(upper), upper, x0))
        res = least_squares(
            resid,
            x0,
            jac=jac,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
            max_nfev=400,
        )
        if best is None or res.cost < best.cost:
            best = res
    assert best is not None

    params = _params_from_x(best.x)
    iv_fit = svi_implied_vol(k_arr, T, params)
    err = 100.0 * (iv_fit - iv_arr)
    return SliceFit(
        T=float(T),
        params=params,
        n_points=int(k_arr.size),
        k_min=k_min,
        k_max=k_max,
        rmse_vol_points=float(np.sqrt(np.mean(err**2))),
        max_abs_vol_points=float(np.max(np.abs(err))),
        cost=float(best.cost),
        success=bool(best.success),
        message=str(best.message),
    )


def fit_svi_surface(
    slices: Iterable[tuple[float, ArrayLike, ArrayLike]],
) -> list[SliceFit]:
    """Fit each ``(T, k, iv)`` slice independently; output is sorted by T."""
    ordered = sorted(slices, key=lambda s: float(s[0]))
    return [fit_svi_slice(k, iv, T) for T, k, iv in ordered]
