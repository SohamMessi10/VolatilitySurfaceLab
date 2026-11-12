"""Static-arbitrage diagnostics for SVI total-variance surfaces.

Butterfly (within one expiry). With total variance w(k), the Durrleman function

    g(k) = (1 - k w' / (2 w))^2 - (w'^2 / 4) (1/w + 1/4) + w'' / 2

is proportional to the risk-neutral density of log-moneyness (Gatheral & Jacquier
2014): density(k) = g(k) / sqrt(2 pi w) * exp(-d_-^2 / 2). Hence g >= 0 (with
w > 0) is exactly the absence of butterfly arbitrage in the slice; a negative g
means a negative density, i.e. a butterfly spread with negative price.

Calendar (across expiries). At fixed forward moneyness k, total variance must be
non-decreasing in T: w(k, T2) >= w(k, T1) for T2 > T1. Otherwise a calendar
spread has negative value (Gatheral & Jacquier 2014). For SVI slices this is
checked on a k grid for each consecutive pair of expiries.

Violations are reported as contiguous runs of grid points where the condition
fails; the interval endpoints are therefore grid points (resolution = grid step).
Each violation records whether it overlaps the quoted strike range of the slice(s)
(``in_quoted_range``) or lies purely in the extrapolated wings, because the checks
run on a fixed grid that is usually wider than the quotes.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .svi import SliceFit, SVIParams, svi_derivatives, svi_total_variance

__all__ = [
    "ArbitrageReport",
    "Violation",
    "check_butterfly",
    "check_calendar",
    "check_surface",
    "durrleman_g",
]


def durrleman_g(k: ArrayLike, p: SVIParams) -> NDArray[np.float64]:
    """Durrleman's g(k); non-negative iff the slice is free of butterfly arbitrage.

    Where w <= 0 the formula is undefined and NaN is returned.
    """
    k_arr = np.asarray(k, dtype=float)
    w, w1, w2 = svi_derivatives(k_arr, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k_arr * w1 / (2.0 * w)) ** 2 - (w1**2 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    return np.where(w > 0, g, np.nan)


@dataclass(frozen=True)
class Violation:
    """A contiguous grid interval where a no-arbitrage condition fails."""

    kind: str
    T: float
    T_prev: float | None
    k_start: float
    k_end: float
    k_worst: float
    worst_value: float
    in_quoted_range: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "kind": self.kind,
            "T": float(self.T),
            "T_prev": None if self.T_prev is None else float(self.T_prev),
            "k_start": float(self.k_start),
            "k_end": float(self.k_end),
            "k_worst": float(self.k_worst),
            "worst_value": float(self.worst_value),
            "in_quoted_range": self.in_quoted_range,
        }


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of contiguous True runs."""
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _overlaps(lo: float, hi: float, quoted: tuple[float, float] | None) -> bool | None:
    return None if quoted is None else bool(lo <= quoted[1] and hi >= quoted[0])


def check_butterfly(
    p: SVIParams,
    T: float,
    k_grid: ArrayLike,
    tol: float = 1e-10,
    quoted: tuple[float, float] | None = None,
) -> list[Violation]:
    """Runs on ``k_grid`` where g < -tol or w <= 0. worst_value is the minimum g
    (or w when w <= 0 inside the run and g is undefined). ``quoted`` is the slice's
    quoted ``(k_min, k_max)``, used to set ``in_quoted_range``."""
    k = np.asarray(k_grid, dtype=float)
    w = svi_total_variance(k, p)
    g = durrleman_g(k, p)
    bad = (w <= 0) | (g < -tol)
    out: list[Violation] = []
    for i, j in _runs(bad):
        seg_g = g[i : j + 1]
        if np.all(np.isnan(seg_g)):
            seg_w = w[i : j + 1]
            idx = int(np.argmin(seg_w))
            val = float(seg_w[idx])
        else:
            idx = int(np.nanargmin(seg_g))
            val = float(seg_g[idx])
        out.append(
            Violation(
                "butterfly",
                float(T),
                None,
                float(k[i]),
                float(k[j]),
                float(k[i + idx]),
                val,
                _overlaps(float(k[i]), float(k[j]), quoted),
            )
        )
    return out


FitLike = SliceFit | tuple[float, SVIParams]


Quoted = tuple[float, float] | None


def _as_pairs(fits: Sequence[FitLike]) -> list[tuple[float, SVIParams, Quoted]]:
    triples: list[tuple[float, SVIParams, Quoted]] = []
    for f in fits:
        if isinstance(f, SliceFit):
            triples.append((f.T, f.params, (f.k_min, f.k_max)))
        else:
            triples.append((float(f[0]), f[1], None))
    return sorted(triples, key=lambda x: x[0])


def check_calendar(
    fits: Sequence[FitLike],
    k_grid: ArrayLike,
    tol: float = 1e-10,
) -> list[Violation]:
    """Runs where w(k, T_next) < w(k, T_prev) - tol for consecutive expiries.

    ``in_quoted_range`` refers to the intersection of the two slices' quoted ranges.
    """
    k = np.asarray(k_grid, dtype=float)
    pairs = _as_pairs(fits)
    out: list[Violation] = []
    for (t0, p0, q0), (t1, p1, q1) in itertools.pairwise(pairs):
        quoted = None if q0 is None or q1 is None else (max(q0[0], q1[0]), min(q0[1], q1[1]))
        diff = svi_total_variance(k, p1) - svi_total_variance(k, p0)
        for i, j in _runs(diff < -tol):
            idx = int(np.argmin(diff[i : j + 1]))
            out.append(
                Violation(
                    "calendar",
                    float(t1),
                    float(t0),
                    float(k[i]),
                    float(k[j]),
                    float(k[i + idx]),
                    float(diff[i + idx]),
                    _overlaps(float(k[i]), float(k[j]), quoted),
                )
            )
    return out


@dataclass
class ArbitrageReport:
    """Aggregate result of :func:`check_surface`."""

    butterfly: list[Violation] = field(default_factory=list)
    calendar: list[Violation] = field(default_factory=list)
    k_grid_min: float = -1.5
    k_grid_max: float = 1.5
    n_grid: int = 3001

    @property
    def passed(self) -> bool:
        """True when no butterfly or calendar violations were found anywhere on the grid."""
        return not self.butterfly and not self.calendar

    @property
    def passed_in_quoted_range(self) -> bool:
        """True when every violation (if any) lies purely in extrapolated wings."""
        return not any(v.in_quoted_range for v in self.butterfly + self.calendar)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "passed": self.passed,
            "passed_in_quoted_range": self.passed_in_quoted_range,
            "butterfly": [v.to_dict() for v in self.butterfly],
            "calendar": [v.to_dict() for v in self.calendar],
            "k_grid_min": float(self.k_grid_min),
            "k_grid_max": float(self.k_grid_max),
            "n_grid": int(self.n_grid),
        }


def check_surface(
    fits: Sequence[FitLike],
    k_grid: ArrayLike | None = None,
    tol: float = 1e-10,
) -> ArbitrageReport:
    """Run butterfly checks on every slice and calendar checks on consecutive pairs."""
    k = np.linspace(-1.5, 1.5, 3001) if k_grid is None else np.asarray(k_grid, dtype=float)
    pairs = _as_pairs(fits)
    butterfly: list[Violation] = []
    for T, p, quoted in pairs:
        butterfly.extend(check_butterfly(p, T, k, tol, quoted))
    calendar = check_calendar(fits, k, tol)
    return ArbitrageReport(
        butterfly=butterfly,
        calendar=calendar,
        k_grid_min=float(k.min()),
        k_grid_max=float(k.max()),
        n_grid=int(k.size),
    )
