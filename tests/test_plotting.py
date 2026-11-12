"""Tests for the plotly figures and the HTML report."""

from __future__ import annotations

import numpy as np
import pytest

from volsurface.arbitrage import check_surface
from volsurface.market import generate_market
from volsurface.plotting import (
    build_smile_figure,
    build_surface_figure,
    fit_error_vs_true,
    interpolate_total_variance,
    write_report_html,
)
from volsurface.svi import SliceFit, SVIParams, fit_svi_surface


@pytest.fixture(scope="module")
def market():
    return generate_market()


@pytest.fixture(scope="module")
def fits(market):
    return fit_svi_surface(market.slices())


def _flatten_min(fit: SliceFit) -> SliceFit:
    """Valid SVI whose minimum total variance is 0 (far below the previous expiry)."""
    p = fit.params
    new = SVIParams(-p.b * p.sigma * np.sqrt(1 - p.rho**2), p.b, p.rho, p.m, p.sigma)
    return SliceFit(**{**fit.__dict__, "params": new})


@pytest.fixture(scope="module")
def bad_fits(fits):
    out = list(fits)
    out[3] = _flatten_min(out[3])  # variance of T=1 drops below T=0.5
    return out


def test_surface_figure_traces(fits, market):
    fig = build_surface_figure(fits, market, check_surface(fits))
    assert [type(t).__name__ for t in fig.data] == ["Surface", "Scatter3d"]
    assert fig.data[1].name == "Market IV (mid)"
    assert len(fig.data[1].x) == len([q for q in market.rows if np.isfinite(q.iv_mid)])


def test_surface_without_market_has_only_surface(fits):
    assert len(build_surface_figure(fits).data) == 1


def test_surface_grid_includes_expiries_and_does_not_extrapolate(fits):
    fig = build_surface_figure(fits, None)
    y = np.asarray(fig.data[0].y)
    for f in fits:
        assert np.any(np.isclose(y, f.T))
    assert y.min() == pytest.approx(fits[0].T)
    assert y.max() == pytest.approx(fits[-1].T)
    assert len(y) > len(fits)
    k = np.asarray(fig.data[0].x)
    assert k.min() == pytest.approx(min(f.k_min for f in fits) - 0.05)
    assert k.max() == pytest.approx(max(f.k_max for f in fits) + 0.05)


def test_interpolated_total_variance_monotone_in_T(fits):
    assert check_surface(fits).calendar == []
    k = np.linspace(-0.8, 0.6, 50)
    T, W = interpolate_total_variance(fits, k)
    assert np.all(np.diff(T) > 0)
    assert np.all(np.diff(W, axis=0) >= -1e-12)


def test_no_violation_markers_when_clean(fits, market):
    fig = build_surface_figure(fits, market, check_surface(fits))
    assert not any("Arbitrage" in (t.name or "") for t in fig.data)


def test_violation_markers_when_arbitrage(bad_fits, market):
    report = check_surface(bad_fits)
    assert report.calendar
    fig = build_surface_figure(bad_fits, market, report)
    flags = [t for t in fig.data if "Arbitrage" in (t.name or "")]
    assert flags
    assert all(type(t).__name__ == "Scatter3d" for t in flags)
    # calendar markers sit at the later expiry
    ys = {y for t in flags for y in t.y}
    assert any(np.isclose(y, 1.0) for y in ys)


def test_smile_figure(fits, market):
    fig = build_smile_figure(fits, market, check_surface(fits))
    names = [t.name for t in fig.data]
    assert names.count("SVI fit") == len(fits)
    assert sum(t.mode == "markers" for t in fig.data) == len(fits)
    dashed = [t for t in fig.data if t.line is not None and t.line.dash == "dash"]
    assert len(dashed) == len(fits)
    assert len(fig.layout.shapes) == 0  # no butterfly violation, no shading


def test_smile_shading_on_butterfly(market):
    from volsurface.arbitrage import ArbitrageReport, Violation

    fits = fit_svi_surface(market.slices())
    v = Violation("butterfly", fits[0].T, None, -0.1, 0.0, -0.05, -0.1, True)
    fig = build_smile_figure(fits, market, ArbitrageReport(butterfly=[v]))
    assert len(fig.layout.shapes) == 1


def test_fit_error_vs_true(fits, market):
    err = fit_error_vs_true(fits, market)
    assert err is not None
    assert len(err) == len(fits)
    assert all(e["rmse_vol_points"] < 0.5 for e in err)
    assert fit_error_vs_true(fits, None) is None


def test_report_html_cdn_and_escape(fits, market, tmp_path):
    out = write_report_html(
        tmp_path / "r.html",
        fits,
        market,
        check_surface(fits),
        include_plotlyjs="cdn",
        title="<script>alert(1)</script>",
    )
    text = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "cdn.plot.ly" in text
    assert "viewport" in text
    assert "approximates the Heston" in text
    assert out.stat().st_size < 200_000


def test_report_html_lists_violations(bad_fits, market, tmp_path):
    out = write_report_html(
        tmp_path / "r.html", bad_fits, market, check_surface(bad_fits), include_plotlyjs="cdn"
    )
    text = out.read_text(encoding="utf-8")
    assert "FAIL" in text
    assert "calendar" in text
    assert "inside quoted range" in text


def test_invalid_plotlyjs(fits, tmp_path):
    with pytest.raises(ValueError, match="include_plotlyjs"):
        write_report_html(tmp_path / "x.html", fits, include_plotlyjs="nope")
