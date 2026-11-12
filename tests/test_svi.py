"""Tests for raw SVI evaluation and fitting."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from volsurface.svi import (
    SliceFit,
    SVIParams,
    fit_svi_slice,
    fit_svi_surface,
    svi_derivatives,
    svi_implied_vol,
    svi_total_variance,
)

# Worst recovery error measured over 300 hypothesis examples: 5e-14 vol points.
TOL = 1e-8

P = SVIParams(a=0.04, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
K = np.linspace(-1.0, 1.0, 25)


def _min_var(p: SVIParams) -> float:
    return p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2)


def test_derivatives_match_finite_differences():
    k = np.linspace(-1.5, 1.5, 31)
    h = 1e-4
    w, w1, w2 = svi_derivatives(k, P)
    wp, wm = svi_total_variance(k + h, P), svi_total_variance(k - h, P)
    np.testing.assert_allclose(w, svi_total_variance(k, P))
    np.testing.assert_allclose(w1, (wp - wm) / (2 * h), atol=1e-8)
    np.testing.assert_allclose(w2, (wp - 2 * w + wm) / h**2, atol=1e-5)


def test_implied_vol_is_sqrt_w_over_T():
    iv = svi_implied_vol(K, 2.0, P)
    np.testing.assert_allclose(iv**2 * 2.0, svi_total_variance(K, P))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"b": -0.1},
        {"rho": 1.0},
        {"sigma": 0.0},
        {"a": -1.0},
        {"a": float("nan")},
    ],
)
def test_params_validation(kwargs):
    base = {"a": 0.04, "b": 0.1, "rho": -0.5, "m": 0.0, "sigma": 0.2}
    with pytest.raises(ValueError, match=r"must be|finite"):
        SVIParams(**{**base, **kwargs})


def test_params_dict_round_trip():
    assert SVIParams.from_dict(P.to_dict()) == P


def test_exact_round_trip():
    iv = svi_implied_vol(K, 1.0, P)
    fit = fit_svi_slice(K, iv, 1.0)
    grid = np.linspace(-1, 1, 201)
    err = 100 * np.abs(svi_implied_vol(grid, 1.0, fit.params) - svi_implied_vol(grid, 1.0, P))
    assert err.max() < 1e-6
    assert fit.max_abs_vol_points < 1e-6
    assert fit.success
    assert fit.n_points == 25


def test_fit_is_deterministic():
    iv = svi_implied_vol(K, 1.0, P)
    assert fit_svi_slice(K, iv, 1.0) == fit_svi_slice(K, iv, 1.0)


def test_weights_are_accepted():
    iv = svi_implied_vol(K, 1.0, P)
    fit = fit_svi_slice(K, iv, 1.0, weights=np.linspace(1, 2, K.size))
    assert fit.max_abs_vol_points < 1e-6


def _assert_constraints(p: SVIParams, fit: SliceFit) -> None:
    assert fit.params == p
    assert p.b >= 0
    assert abs(p.rho) < 1
    assert p.sigma > 0
    assert _min_var(p) >= -1e-12


def test_constraints_hold_for_noisy_fits():
    rng = np.random.default_rng(42)
    for seed_scale in (0.001, 0.01, 0.05):
        iv = svi_implied_vol(K, 1.0, P) + rng.normal(0, seed_scale, K.size)
        iv = np.abs(iv) + 0.01
        fit = fit_svi_slice(K, iv, 1.0)
        _assert_constraints(fit.params, fit)
        assert np.all(svi_total_variance(np.linspace(-3, 3, 601), fit.params) >= -1e-12)
        assert fit.rmse_vol_points <= fit.max_abs_vol_points


def test_flat_smile():
    iv = np.full(K.size, 0.2)
    fit = fit_svi_slice(K, iv, 1.0)
    _assert_constraints(fit.params, fit)
    assert fit.max_abs_vol_points < 1e-3


def test_requires_five_points():
    with pytest.raises(ValueError, match="at least 5"):
        fit_svi_slice([-0.1, 0.0, 0.1, 0.2], [0.2, 0.19, 0.2, 0.21], 1.0)


def test_slicefit_dict_round_trip():
    fit = fit_svi_slice(K, svi_implied_vol(K, 1.0, P), 1.0)
    assert SliceFit.from_dict(fit.to_dict()) == fit


def test_fit_surface_sorted_by_T():
    slices = [(T, K, svi_implied_vol(K, T, P)) for T in (1.0, 0.25, 0.5)]
    fits = fit_svi_surface(slices)
    assert [f.T for f in fits] == [0.25, 0.5, 1.0]


@st.composite
def _svi_params(draw):
    b = draw(st.floats(0.02, 0.5))
    rho = draw(st.floats(-0.8, 0.8))
    m = draw(st.floats(-0.3, 0.3))
    sigma = draw(st.floats(0.05, 0.6))
    c = draw(st.floats(0.01, 0.1))  # minimum total variance
    return SVIParams(c - b * sigma * np.sqrt(1 - rho**2), b, rho, m, sigma)


@settings(max_examples=50, deadline=None)
@given(_svi_params())
def test_recovers_random_valid_params(p):
    iv = svi_implied_vol(K, 1.0, p)
    fit = fit_svi_slice(K, iv, 1.0)
    _assert_constraints(fit.params, fit)
    grid = np.linspace(-1, 1, 101)
    err = 100 * np.abs(svi_implied_vol(grid, 1.0, fit.params) - svi_implied_vol(grid, 1.0, p))
    assert err.max() < TOL
