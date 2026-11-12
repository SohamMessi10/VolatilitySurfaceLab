"""Tests for the optional fetch module using a fake ticker (no network, no yfinance)."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from volsurface.black_scholes import bs_price
from volsurface.fetch import fetch_market, fetch_market_with_info
from volsurface.market import load_market

NOW = datetime(2026, 1, 1, 12, 0)
SPOT = 100.0
STRIKES = np.arange(80.0, 120.1, 2.5)
HALF = 0.01
# days -> (r, q) used to build the model forward and discount factor
RATES = {
    3: (0.04, 0.01),
    30: (0.04, 0.01),
    90: (0.045, 0.012),
    400: (0.05, 0.015),
    800: (0.05, 0.0),
}


def model_vol(K, F):
    k = np.log(K / F)
    return 0.2 - 0.1 * k + 0.3 * k**2


def model_fd(days):
    r, q = RATES[days]
    T = days / 365.0
    return SPOT * math.exp((r - q) * T), math.exp(-r * T), T


def make_frames(days, *, mutate=None):
    F, df, T = model_fd(days)
    sig = model_vol(STRIKES, F)
    out = {}
    for name, is_call in (("calls", True), ("puts", False)):
        p = bs_price(F, STRIKES, T, sig, df, is_call)
        out[name] = {
            "strike": list(STRIKES),
            "bid": list(np.maximum(p - HALF, 0.0)),
            "ask": list(p + HALF),
            "lastPrice": list(p),
        }
    if mutate:
        mutate(out)
    return out


def make_ticker(days_list=(3, 30, 90, 400, 800), mutate=None):
    dates = {(NOW + timedelta(days=d)).strftime("%Y-%m-%d"): d for d in days_list}

    def option_chain(date):
        fr = make_frames(dates[date], mutate=mutate.get(dates[date]) if mutate else None)
        return SimpleNamespace(calls=fr["calls"], puts=fr["puts"])

    return SimpleNamespace(
        options=tuple(dates), option_chain=option_chain, fast_info={"lastPrice": SPOT}
    )


def run(**kw):
    t = kw.pop("ticker", None) or make_ticker()
    return fetch_market_with_info("TEST", now=NOW, ticker_factory=lambda s: t, **kw)


def test_expiry_filtering_and_order():
    market, info = run()
    days = sorted(round(v["T"] * 365) for v in info.values())
    assert days == [30, 90, 400]  # 3 < min_days, 800 > max_days
    assert market.spec.expiries == tuple(d / 365 for d in days)
    _, info2 = run(max_expiries=2)
    assert sorted(round(v["T"] * 365) for v in info2.values()) == [30, 90]
    _, info3 = run(min_days=0, max_days=100)
    assert sorted(round(v["T"] * 365) for v in info3.values()) == [3, 30, 90]


def test_parity_recovers_forward_and_discount_factor():
    _, info = run()
    errs_F, errs_df = [], []
    for d in info.values():
        days = round(d["T"] * 365)
        F, df, _ = model_fd(days)
        assert d["method"] == "parity"
        errs_F.append(abs(d["F"] / F - 1))
        errs_df.append(abs(d["df"] - df))
    assert max(errs_F) < 2e-4
    assert max(errs_df) < 2e-3


def test_iv_mid_matches_model_and_otm_selection():
    market, info = run()
    assert market.heston is None
    assert math.isnan(market.noise)
    assert market.seed is None
    for q in market.rows:
        days = round(q.expiry * 365)
        F, _, _ = model_fd(days)
        assert q.option_type == ("put" if q.strike < q.forward else "call")
        assert q.forward == pytest.approx(info[_date(days)]["F"])
        assert math.isnan(q.true_price)
        assert math.isnan(q.iv_true)
        assert q.log_moneyness == pytest.approx(math.log(q.strike / q.forward))
        assert q.iv_bid < q.iv_mid < q.iv_ask
        # mids are model prices at the true F, df; implied F/df errors perturb the vol slightly
        assert abs(q.iv_mid - model_vol(q.strike, F)) < 5e-3
    assert market.spec.spot == SPOT
    assert list(market.spec.strike_moneyness) == sorted(set(market.spec.strike_moneyness))


def _date(days):
    return (NOW + timedelta(days=days)).strftime("%Y-%m-%d")


def test_bad_quotes_dropped():
    def mutate(fr):
        # strike 105 call: bid 0; strike 90 put: crossed; strike 85 put: nan
        i105, i90, i85 = (list(STRIKES).index(k) for k in (105.0, 90.0, 85.0))
        fr["calls"]["bid"][i105] = 0.0
        fr["puts"]["bid"][i90], fr["puts"]["ask"][i90] = 5.0, 4.0
        fr["puts"]["ask"][i85] = float("nan")

    clean, _ = run()
    market, _ = run(ticker=make_ticker((30,), {30: mutate}))
    strikes = {q.strike for q in market.rows}
    assert not strikes & {105.0, 90.0, 85.0}
    assert market.n_dropped >= 3
    assert strikes <= {q.strike for q in clean.rows}
    assert len(strikes) >= 8


def test_fallback_when_too_few_two_sided_strikes():
    def mutate(fr):
        # keep only two strikes quoted on both sides inside +-10%: zero the puts elsewhere
        keep = {95.0, 105.0}
        for i, k in enumerate(STRIKES):
            if k not in keep:
                fr["puts"]["bid"][i] = 0.0

    market, info = run(ticker=make_ticker((30,), {30: mutate}), fallback_rate=0.03)
    (d,) = info.values()
    T = 30 / 365
    assert d["method"] == "fallback"
    assert d["F"] == pytest.approx(SPOT * math.exp(0.03 * T))
    assert d["df"] == pytest.approx(math.exp(-0.03 * T))
    assert d["n_quotes"] == len(market.rows) > 0


def test_no_quotes_raises():
    t = make_ticker((3,))
    with pytest.raises(ValueError, match="no usable"):
        run(ticker=t)


def test_fetch_market_returns_market():
    m = fetch_market("TEST", now=NOW, ticker_factory=lambda s: make_ticker())
    assert len(m.rows) > 0


def test_import_error_without_yfinance(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(ImportError, match=r"volsurface\[fetch\]"):
        fetch_market("SPY")


def test_json_roundtrip(tmp_path):
    market, _ = run()
    path = tmp_path / "m.json"
    market.to_json(path)
    back = load_market(path)
    assert back.spec == market.spec
    assert len(back.rows) == len(market.rows)
    assert back.rows[0].strike == market.rows[0].strike
    assert back.rows[0].iv_mid == pytest.approx(market.rows[0].iv_mid)
    assert math.isnan(back.rows[0].true_price)
    assert math.isnan(back.noise)
