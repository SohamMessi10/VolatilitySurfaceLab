"""Volatility Surface Lab: Heston synthetic market -> implied vols -> SVI -> arbitrage checks."""

from .arbitrage import ArbitrageReport, Violation, check_surface, durrleman_g
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
from .market import DEFAULT_HESTON, MarketData, MarketSpec, Quote, generate_market, load_market
from .svi import (
    SliceFit,
    SVIParams,
    fit_svi_slice,
    fit_svi_surface,
    svi_implied_vol,
    svi_total_variance,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_HESTON",
    "ArbitrageReport",
    "HestonError",
    "HestonParams",
    "ImpliedVolError",
    "ImpliedVolResult",
    "MarketData",
    "MarketSpec",
    "PriceOutOfBoundsError",
    "Quote",
    "SVIParams",
    "SliceFit",
    "Violation",
    "__version__",
    "bs_price",
    "bs_vega",
    "check_surface",
    "critical_moments",
    "durrleman_g",
    "fit_svi_slice",
    "fit_svi_surface",
    "forward_price",
    "generate_market",
    "heston_call_lewis",
    "heston_cf",
    "heston_otm_fft",
    "heston_price",
    "implied_vol",
    "implied_vol_array",
    "load_market",
    "log_moneyness",
    "moment_explosion_time",
    "normalized_call",
    "normalized_otm",
    "normalized_put",
    "normalized_vega",
    "price_bounds",
    "put_call_parity_gap",
    "svi_implied_vol",
    "svi_total_variance",
]
