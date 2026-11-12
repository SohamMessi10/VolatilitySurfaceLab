import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from volsurface.black_scholes import (
    bs_price,
    bs_vega,
    forward_price,
    normalized_call,
    normalized_put,
    price_bounds,
    put_call_parity_gap,
)

# Hull's textbook example: S=100, K=100, T=1, r=5%, q=0, sigma=20%.
S, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
F = forward_price(S, T, R)
DF = math.exp(-R * T)


def test_reference_values():
    # Closed-form reference values (independently known: C = 10.4506, P = 5.5735).
    assert bs_price(F, K, T, SIGMA, DF, True) == pytest.approx(10.450583572185565, abs=1e-12)
    assert bs_price(F, K, T, SIGMA, DF, False) == pytest.approx(5.573526022256971, abs=1e-12)


def test_vega_matches_finite_difference():
    h = 1e-6
    fd = (bs_price(F, K, T, SIGMA + h, DF) - bs_price(F, K, T, SIGMA - h, DF)) / (2 * h)
    assert bs_vega(F, K, T, SIGMA, DF) == pytest.approx(fd, rel=1e-8)


def test_put_call_parity_on_grid():
    strikes = np.linspace(40.0, 250.0, 51)
    for t in (0.01, 0.5, 3.0):
        for sig in (0.05, 0.3, 1.5):
            c = bs_price(F, strikes, t, sig, DF, True)
            p = bs_price(F, strikes, t, sig, DF, False)
            gap = put_call_parity_gap(c, p, F, strikes, DF)
            assert np.max(np.abs(gap)) < 1e-11


def test_prices_inside_no_arbitrage_bounds():
    strikes = np.linspace(20.0, 400.0, 101)
    for is_call in (True, False):
        lo, hi = price_bounds(F, strikes, DF, is_call)
        for sig in (1e-4, 0.2, 5.0):
            p = bs_price(F, strikes, 2.0, sig, DF, is_call)
            assert np.all(p >= lo - 1e-12)
            assert np.all(p <= hi + 1e-12)


def test_volatility_limits():
    strikes = np.array([50.0, 100.0, 150.0])
    lo, hi = price_bounds(F, strikes, DF, True)
    np.testing.assert_allclose(bs_price(F, strikes, T, 1e-9, DF), lo, atol=1e-10)
    np.testing.assert_allclose(bs_price(F, strikes, T, 0.0, DF), lo, atol=0.0)
    np.testing.assert_allclose(bs_price(F, strikes, T, 60.0, DF), hi, rtol=1e-12)


def test_call_decreasing_and_convex_in_strike():
    strikes = np.linspace(50.0, 200.0, 301)
    c = bs_price(F, strikes, T, SIGMA, DF, True)
    assert np.all(np.diff(c) < 0)
    assert np.all(np.diff(c, 2) > 0)


def test_deep_otm_time_value_is_positive_and_accurate():
    # A deep OTM call at k = 1.5 with s = 0.2 is ~1e-14 of the forward. Computing it
    # directly (not via parity from the ITM put) keeps it positive and accurate.
    c = float(normalized_call(1.5, 0.2))
    assert 0.0 < c < 1e-10
    # Same number from the symmetry c(k, s) = e^k p(-k, s).
    assert c == pytest.approx(math.exp(1.5) * float(normalized_put(-1.5, 0.2)), rel=1e-12)


@given(
    k=st.floats(-3.0, 3.0),
    s=st.floats(1e-3, 5.0),
)
def test_put_call_symmetry(k, s):
    # Black-76 symmetry in forward coordinates: c(k, s) = e^k p(-k, s).
    c = float(normalized_call(k, s))
    p = float(normalized_put(-k, s))
    assert c == pytest.approx(math.exp(k) * p, rel=1e-9, abs=1e-15)
