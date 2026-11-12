"""End-to-end: Heston market -> implied vols -> SVI fits -> static-arbitrage checks."""

import dataclasses

import numpy as np
import pytest

from volsurface.arbitrage import check_surface, durrleman_g
from volsurface.market import generate_market
from volsurface.svi import SVIParams, fit_svi_surface, svi_implied_vol, svi_total_variance

VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)  # T = 1


@pytest.fixture(scope="module")
def clean():
    market = generate_market(noise=0.0)
    return market, fit_svi_surface(market.slices())


def test_clean_heston_surface_passes(clean):
    market, fits = clean
    report = check_surface(fits)
    assert report.passed, report.to_dict()
    # SVI approximates Heston; it does not reproduce it. Measured worst error vs the true
    # Heston implied vol on the default market is 0.082 vol points (T = 0.5).
    for fit in fits:
        qs = market.slice(fit.T)
        k = np.array([q.log_moneyness for q in qs])
        iv_true = np.array([q.iv_true for q in qs])
        err = 100.0 * np.abs(svi_implied_vol(k, fit.T, fit.params) - iv_true)
        assert err.max() < 0.1, (fit.T, err.max())
        assert fit.params.b * (1 + abs(fit.params.rho)) <= 2.0 + 1e-12  # Lee bound


def test_injected_calendar_arbitrage_is_located(clean):
    market, _ = clean
    slices = market.slices()
    # Replace the T = 1.5 quotes by vols whose total variance is 2% *below* T = 1.0's.
    i_prev = next(i for i, s in enumerate(slices) if s[0] == 1.0)
    T0, k0, iv0 = slices[i_prev]
    T1 = slices[i_prev + 1][0]
    assert T1 == 1.5
    slices[i_prev + 1] = (T1, k0, iv0 * np.sqrt(0.98 * T0 / T1))
    report = check_surface(fit_svi_surface(slices))
    assert not report.passed
    hits = [v for v in report.calendar if v.T == 1.5 and v.T_prev == 1.0]
    assert len(hits) == 1
    v = hits[0]
    assert v.in_quoted_range
    # The whole quoted range is inverted, so the violating run must cover it.
    assert v.k_start <= k0.min()
    assert v.k_end >= k0.max()
    assert v.worst_value < 0
    # Nothing else is flagged inside the quoted ranges.
    others = [x for x in report.butterfly + report.calendar if x is not v and x.in_quoted_range]
    assert others == []


def test_injected_butterfly_arbitrage_is_located(clean):
    market, _ = clean
    slices = market.slices()
    # Quotes at T = 1 taken from Axel Vogt's smile, whose density is negative for
    # k in ~[0.64, 1.26] (Gatheral & Jacquier 2014). The fit must reproduce the arbitrage
    # and the checker must locate it.
    k = np.linspace(-0.4, 1.5, 25)
    iv = np.sqrt(svi_total_variance(k, VOGT) / 1.0)
    slices = [s for s in slices if s[0] != 1.0] + [(1.0, k, iv)]
    fits = fit_svi_surface(slices)
    report = check_surface(fits)
    hits = [v for v in report.butterfly if v.T == 1.0]
    assert len(hits) == 1
    v = hits[0]
    assert v.in_quoted_range
    assert v.k_start == pytest.approx(0.643, abs=0.02)
    assert v.k_end == pytest.approx(1.256, abs=0.02)
    fit_1y = next(f for f in fits if f.T == 1.0)
    assert durrleman_g(v.k_worst, fit_1y.params) < 0


def test_noisy_market_has_no_arbitrage_inside_quoted_range():
    # With seeded bid/ask-style noise, independent SVI slice fits stay arbitrage-free where
    # there are quotes, but may violate in the extrapolated wings (reported, not hidden).
    market = generate_market(noise=1.0, seed=3)
    report = check_surface(fit_svi_surface(market.slices()))
    assert report.passed_in_quoted_range
    for v in report.butterfly + report.calendar:
        assert v.in_quoted_range is False


def test_report_serializes(clean):
    _, fits = clean
    d = check_surface(fits).to_dict()
    assert d["passed"] is True
    assert d["passed_in_quoted_range"] is True
    assert dataclasses.is_dataclass(fits[0])
