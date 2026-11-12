"""Volatility Surface Lab: Heston synthetic market -> implied vols -> SVI -> arbitrage checks."""

from .black_scholes import (
    bs_price,
    bs_vega,
    forward_price,
    log_moneyness,
    normalized_call,
    normalized_otm,
    normalized_put,
    normalized_vega,
    price_bounds,
    put_call_parity_gap,
)
from .implied_vol import (
    ImpliedVolError,
    ImpliedVolResult,
    PriceOutOfBoundsError,
    implied_vol,
    implied_vol_array,
)

__version__ = "0.1.0"

__all__ = [
    "ImpliedVolError",
    "ImpliedVolResult",
    "PriceOutOfBoundsError",
    "__version__",
    "bs_price",
    "bs_vega",
    "forward_price",
    "implied_vol",
    "implied_vol_array",
    "log_moneyness",
    "normalized_call",
    "normalized_otm",
    "normalized_put",
    "normalized_vega",
    "price_bounds",
    "put_call_parity_gap",
]
