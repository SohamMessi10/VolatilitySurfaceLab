import math

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from volsurface.black_scholes import bs_price, normalized_otm, normalized_vega
from volsurface.implied_vol import (
    ImpliedVolError,
    PriceOutOfBoundsError,
    _otm_scalar,
    _vega_scalar,
    implied_vol,
    implied_vol_array,
)


def test_recovers_textbook_vol():
    F, df = 100.0 * math.exp(0.05), math.exp(-0.05)
    assert implied_vol(10.450583572185565, F, 100.0, 1.0, df, True) == pytest.approx(0.2, abs=1e-13)
    assert implied_vol(5.573526022256971, F, 100.0, 1.0, df, False) == pytest.approx(0.2, abs=1e-13)


def test_near_atm_uses_newton_with_few_iterations():
    price = float(bs_price(100.0, 105.0, 0.5, 0.25))
    res = implied_vol(price, 100.0, 105.0, 0.5, details=True)
    assert res.method == "newton"
    assert res.iterations <= 8
    assert res.sigma == pytest.approx(0.25, abs=1e-14)


def test_deep_otm_tiny_vega_falls_back_to_brent():
    # k = ln(2) with 10% vol for 1 month: vega at the root is ~1e-30 of the forward.
    F, K, T, sig = 100.0, 200.0, 1.0 / 12.0, 0.10
    price = float(bs_price(F, K, T, sig))
    assert 0.0 < price < 1e-20
    res = implied_vol(price, F, K, T, details=True)
    assert res.method == "brent"
    assert res.sigma == pytest.approx(sig, rel=1e-9)


@pytest.mark.parametrize(
    ("price", "is_call", "reason"),
    [
        (4.99, True, "below intrinsic"),  # F=105, K=100 -> intrinsic 5
        (5.0, True, "at intrinsic (sigma = 0)"),
        (105.0, True, "at the call upper bound F (sigma = inf)"),
        (106.0, True, "above the call upper bound"),
        (-1.0, False, "negative put"),
        (100.0, False, "at the put upper bound K"),
        (float("nan"), True, "nan"),
    ],
)
def test_rejects_prices_outside_bounds(price, is_call, reason):
    with pytest.raises(PriceOutOfBoundsError, match=r"no implied volatility exists|finite"):
        implied_vol(price, 105.0, 100.0, 1.0, 1.0, is_call)


def test_error_message_reports_bounds():
    with pytest.raises(PriceOutOfBoundsError) as exc:
        implied_vol(3.0, 105.0, 100.0, 1.0, 0.9, True)
    msg = str(exc.value)
    assert "call price 3.0" in msg
    assert "4.5" in msg  # lower bound df*(F-K) = 0.9*5
    assert "94.5" in msg  # upper bound df*F


def test_rejects_bad_inputs():
    with pytest.raises(ImpliedVolError):
        implied_vol(1.0, 100.0, 100.0, 0.0)
    with pytest.raises(ImpliedVolError):
        implied_vol(1.0, -100.0, 100.0, 1.0)


def test_array_version_with_nan_errors():
    F, T = 100.0, 0.75
    strikes = np.array([70.0, 100.0, 140.0])
    prices = bs_price(F, strikes, T, np.array([0.3, 0.2, 0.25]))
    prices_bad = prices.copy()
    prices_bad[1] = -1.0
    out = implied_vol_array(prices_bad, F, strikes, T, errors="nan")
    np.testing.assert_allclose(out[[0, 2]], [0.3, 0.25], rtol=1e-12)
    assert np.isnan(out[1])
    with pytest.raises(PriceOutOfBoundsError):
        implied_vol_array(prices_bad, F, strikes, T)


@settings(max_examples=2000, deadline=None)
@given(
    k=st.floats(-3.0, 3.0),
    sigma=st.floats(0.01, 3.0),
    T=st.floats(1.0 / 365.0, 10.0),
    F=st.floats(1.0, 1000.0),
    df=st.floats(0.5, 1.0),
    is_call=st.booleans(),
)
def test_round_trip_price_iv_price(k, sigma, T, F, df, is_call):
    K = F * math.exp(k)
    price = float(bs_price(F, K, T, sigma, df, is_call))
    s = sigma * math.sqrt(T)
    # Time value of the option relative to the forward; needs to be representable.
    time_value = float(normalized_otm(math.log(K / F), s))
    assume(time_value > 1e-250)
    try:
        iv = implied_vol(price, F, K, T, df, is_call)
    except PriceOutOfBoundsError:
        # Only legitimate when the time value is lost below the rounding of an ITM price.
        assert time_value < 1e-12 * price / (df * F)
        return
    repriced = float(bs_price(F, K, T, iv, df, is_call))
    # 1) price round trip: within a few ulps of the price (plus rounding of intrinsic).
    assert abs(repriced - price) <= 1e-12 * price + 1e-14 * df * F
    # 2) vol round trip wherever the price actually pins the vol down in double precision.
    vega_rel = float(normalized_vega(math.log(K / F), s)) * s / max(price / (df * F), 1e-300)
    if time_value > 1e-8 * price / (df * F) and vega_rel > 1e-3:
        assert iv == pytest.approx(sigma, rel=1e-8)


@given(k=st.floats(-4.0, 4.0), s=st.floats(1e-4, 8.0))
def test_scalar_fast_path_matches_vectorized(k, s):
    # Both paths evaluate N(d1) - e^k N(d2) (or the put analogue). Far in the wings with
    # small s that difference cancels: rounding of d1 perturbs N(d1) by ~d1^2 eps, and the
    # cancellation amplifies it by ~|d1|/s. The tolerance follows that condition number.
    d1 = -k / s + 0.5 * s
    cond = 1.0 + d1 * d1 + abs(d1) ** 3 / s
    tol = 1e-14 * cond
    assert _otm_scalar(k, s) == pytest.approx(float(normalized_otm(k, s)), rel=tol, abs=1e-300)
    assert _vega_scalar(k, s) == pytest.approx(float(normalized_vega(k, s)), rel=tol, abs=1e-300)
