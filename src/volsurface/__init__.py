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
from .heston import (
    HestonError,
    HestonParams,
    critical_moments,
    heston_call_lewis,
    heston_cf,
    heston_otm_fft,
    heston_price,
    moment_explosion_time,
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
    "HestonError",
    "HestonParams",
    "ImpliedVolError",
    "ImpliedVolResult",
    "PriceOutOfBoundsError",
    "__version__",
    "bs_price",
    "bs_vega",
    "critical_moments",
    "forward_price",
    "heston_call_lewis",
    "heston_cf",
    "heston_otm_fft",
    "heston_price",
    "implied_vol",
    "implied_vol_array",
    "log_moneyness",
    "moment_explosion_time",
    "normalized_call",
    "normalized_otm",
    "normalized_put",
    "normalized_vega",
    "price_bounds",
    "put_call_parity_gap",
]
