"""Synthetic option market generated from a Heston model.

Pipeline: for every expiry and strike of a :class:`MarketSpec` the out-of-the-money option
(put if ``K < F``, call otherwise) is priced with the Heston FFT pricer, quotes whose true
price is below ``spec.min_price`` are dropped, and a bid/ask quote is built around the mid.

Noise model (deliberately simple, *not* a microstructure model): the half-spread is
``h = max(tick, rel_half_spread * true_price)``; the mid is the true price plus
``noise * h * U(-1, 1)`` with independent uniform draws; bid is ``mid - h`` rounded down to the
tick and ask is ``mid + h`` rounded up (bid floored at 0). There is no autocorrelation across
strikes or expiries, no stale quotes, no inventory effects and no arbitrage-consistent noise:
a noisy mid is only clipped strictly inside the model-free price bounds so that an implied
vol exists. With ``noise=0`` the mid equals the true price exactly and the market is
deterministic. Implied vols for bid/ask are ``nan`` when the quote is outside the bounds.

Conventions: ``k = ln(K/F)``, ``F = S exp((r - q) T)``, ``df = exp(-r T)``.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .black_scholes import forward_price, log_moneyness, price_bounds
from .heston import HestonParams, heston_price
from .implied_vol import implied_vol_array

__all__ = ["DEFAULT_HESTON", "MarketData", "MarketSpec", "Quote", "generate_market", "load_market"]

DEFAULT_HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)

_DEFAULT_MONEYNESS = tuple(round(0.60 + 0.05 * i, 10) for i in range(19))
_BOUND_EPS = 1e-8


def _check_increasing(name: str, values: tuple[float, ...]) -> None:
    if len(values) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{name} must be positive and finite")
    if any(b <= a for a, b in itertools.pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class MarketSpec:
    """Grid and quoting conventions of the synthetic market."""

    spot: float = 100.0
    r: float = 0.03
    q: float = 0.01
    expiries: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
    strike_moneyness: tuple[float, ...] = _DEFAULT_MONEYNESS
    tick: float = 0.01
    rel_half_spread: float = 0.02
    min_price: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "expiries", tuple(float(x) for x in self.expiries))
        object.__setattr__(self, "strike_moneyness", tuple(float(x) for x in self.strike_moneyness))
        if not (math.isfinite(self.spot) and self.spot > 0):
            raise ValueError("spot must be positive")
        if not (math.isfinite(self.r) and math.isfinite(self.q)):
            raise ValueError("r and q must be finite")
        _check_increasing("expiries", self.expiries)
        _check_increasing("strike_moneyness", self.strike_moneyness)
        if not self.tick > 0:
            raise ValueError("tick must be > 0")
        if not self.rel_half_spread >= 0:
            raise ValueError("rel_half_spread must be >= 0")
        if not self.min_price >= 0:
            raise ValueError("min_price must be >= 0")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expiries"] = list(self.expiries)
        d["strike_moneyness"] = list(self.strike_moneyness)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MarketSpec:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class Quote:
    """One option quote (the OTM option at a given expiry and strike)."""

    expiry: float
    strike: float
    forward: float
    discount_factor: float
    log_moneyness: float
    option_type: str
    true_price: float
    bid: float
    ask: float
    mid: float
    iv_true: float
    iv_bid: float
    iv_ask: float
    iv_mid: float


_QUOTE_FIELDS = tuple(f.name for f in fields(Quote))
_FLOAT_FIELDS = tuple(n for n in _QUOTE_FIELDS if n != "option_type")


@dataclass
class MarketData:
    """A set of quotes plus the metadata that generated them."""

    spec: MarketSpec
    heston: HestonParams | None
    noise: float
    seed: int | None
    n_dropped: int
    rows: list[Quote] = field(default_factory=list)

    def expiries(self) -> list[float]:
        return sorted({q.expiry for q in self.rows})

    def slice(self, T: float) -> list[Quote]:
        return sorted((q for q in self.rows if q.expiry == T), key=lambda q: q.strike)

    def slices(self) -> list[tuple[float, NDArray, NDArray]]:
        """``(T, k, iv_mid)`` per expiry, sorted by T and k, skipping nan ``iv_mid``."""
        out = []
        for T in self.expiries():
            qs = [q for q in self.slice(T) if not math.isnan(q.iv_mid)]
            if qs:
                out.append(
                    (
                        T,
                        np.array([q.log_moneyness for q in qs]),
                        np.array([q.iv_mid for q in qs]),
                    )
                )
        return out

    def to_json(self, path: str | Path) -> None:
        meta = {
            "spec": self.spec.to_dict(),
            "heston": None if self.heston is None else self.heston.to_dict(),
            "noise": None if math.isnan(self.noise) else self.noise,
            "seed": self.seed,
            "n_dropped": self.n_dropped,
        }
        rows = [
            {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in asdict(q).items()
            }
            for q in self.rows
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"metadata": meta, "rows": rows}, fh, allow_nan=False, indent=1)

    def to_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_QUOTE_FIELDS)
            for q in self.rows:
                w.writerow([repr(v) if isinstance(v, float) else v for v in _astuple(q)])


def _astuple(q: Quote) -> tuple:
    return tuple(getattr(q, n) for n in _QUOTE_FIELDS)


def _round_tick(x: NDArray, tick: float, *, up: bool) -> NDArray:
    n = x / tick
    n = np.ceil(n - 1e-9) if up else np.floor(n + 1e-9)
    return np.round(n * tick, 12)


def generate_market(
    params: HestonParams = DEFAULT_HESTON,
    spec: MarketSpec | None = None,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> MarketData:
    """Generate OTM quotes from a Heston model; see the module docstring for the noise model."""
    spec = MarketSpec() if spec is None else spec
    if noise < 0:
        raise ValueError("noise must be >= 0")
    rng = np.random.default_rng(seed)
    K_all = spec.spot * np.asarray(spec.strike_moneyness)
    rows: list[Quote] = []
    dropped = 0
    for T in spec.expiries:
        F = float(forward_price(spec.spot, T, spec.r, spec.q))
        df = math.exp(-spec.r * T)
        is_call = K_all >= F
        # FFT normally; direct Lewis integration for expiries where no FFT damping is usable.
        true = heston_price(F, K_all, T, params, df, is_call, method="auto")
        keep = true >= spec.min_price
        dropped += int((~keep).sum())
        # draw for all strikes so that the noise stream does not depend on dropping
        u = rng.uniform(-1.0, 1.0, size=K_all.shape)
        K, is_call, true, u = K_all[keep], is_call[keep], true[keep], u[keep]
        if K.size == 0:
            continue
        h = np.maximum(spec.tick, spec.rel_half_spread * true)
        mid = true + noise * h * u if noise > 0 else true.copy()
        lo, hi = price_bounds(F, K, df, is_call)
        if noise > 0:
            mid = np.clip(mid, lo + _BOUND_EPS, hi - _BOUND_EPS)
        bid = np.maximum(_round_tick(mid - h, spec.tick, up=False), 0.0)
        ask = _round_tick(mid + h, spec.tick, up=True)
        ivs = [
            implied_vol_array(p, F, K, T, df, is_call, errors="nan") for p in (true, mid, bid, ask)
        ]
        k = log_moneyness(K, F)
        for i in range(K.size):
            rows.append(
                Quote(
                    expiry=T,
                    strike=float(K[i]),
                    forward=F,
                    discount_factor=df,
                    log_moneyness=float(k[i]),
                    option_type="call" if is_call[i] else "put",
                    true_price=float(true[i]),
                    bid=float(bid[i]),
                    ask=float(ask[i]),
                    mid=float(mid[i]),
                    iv_true=float(ivs[0][i]),
                    iv_bid=float(ivs[2][i]),
                    iv_ask=float(ivs[3][i]),
                    iv_mid=float(ivs[1][i]),
                )
            )
    return MarketData(spec, params, float(noise), seed, dropped, rows)


def _quote_from_mapping(d: dict) -> Quote:
    kw = {}
    for n in _QUOTE_FIELDS:
        v = d[n]
        if n == "option_type":
            kw[n] = str(v)
        else:
            kw[n] = math.nan if v is None or v == "" else float(v)
    return Quote(**kw)


def load_market(path: str | Path) -> MarketData:
    """Load a market from ``.json`` (full metadata) or ``.csv`` (rows only)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        meta = blob["metadata"]
        return MarketData(
            spec=MarketSpec.from_dict(meta["spec"]),
            heston=None if meta["heston"] is None else HestonParams.from_dict(meta["heston"]),
            noise=math.nan if meta["noise"] is None else float(meta["noise"]),
            seed=meta["seed"],
            n_dropped=int(meta["n_dropped"]),
            rows=[_quote_from_mapping(r) for r in blob["rows"]],
        )
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [_quote_from_mapping(r) for r in csv.DictReader(fh)]
        expiries = tuple(sorted({q.expiry for q in rows}))
        spec = MarketSpec(expiries=expiries) if expiries else MarketSpec()
        return MarketData(spec, None, math.nan, None, 0, rows)
    raise ValueError(f"unsupported market file extension {suffix!r}; use .json or .csv")
