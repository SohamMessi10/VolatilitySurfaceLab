"""Tests for the synthetic market generator."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict

import numpy as np
import pytest

from volsurface.black_scholes import price_bounds
from volsurface.heston import heston_price
from volsurface.implied_vol import implied_vol_array
from volsurface.market import (
    DEFAULT_HESTON,
    MarketSpec,
    generate_market,
    load_market,
)

SMALL = MarketSpec(expiries=(0.25, 1.0), strike_moneyness=(0.7, 0.9, 1.0, 1.1, 1.3))


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def _rows_equal(r1, r2) -> bool:
    for x, y in zip(r1, r2, strict=True):
        for k, v in asdict(x).items():
            w = asdict(y)[k]
            if isinstance(v, float):
                if not _same(v, w):
                    return False
            elif v != w:
                return False
    return len(r1) == len(r2)


@pytest.fixture(scope="module")
def default_market():
    return generate_market()


def test_default_counts(default_market):
    spec = default_market.spec
    total = len(spec.expiries) * len(spec.strike_moneyness)
    assert len(default_market.rows) + default_market.n_dropped == total
    assert all(q.true_price >= spec.min_price for q in default_market.rows)
    assert default_market.n_dropped > 0  # deep OTM short-dated wings


def test_noise_zero_deterministic_and_consistent(default_market):
    for q in default_market.rows:
        assert q.mid == q.true_price
        assert abs(q.iv_mid - q.iv_true) <= 1e-12
    again = generate_market()
    assert _rows_equal(default_market.rows, again.rows)


def test_matches_lewis(default_market):
    p = DEFAULT_HESTON
    for i in (0, 12, 40, len(default_market.rows) - 1):
        q = default_market.rows[i]
        is_call = q.option_type == "call"
        lewis = heston_price(
            q.forward, [q.strike], q.expiry, p, q.discount_factor, is_call, method="lewis"
        )
        iv = implied_vol_array(lewis, q.forward, q.strike, q.expiry, q.discount_factor, is_call)
        assert abs(iv[0] - q.iv_true) < 1e-7


def test_seed_reproducibility():
    a = generate_market(spec=SMALL, noise=0.5, seed=3)
    b = generate_market(spec=SMALL, noise=0.5, seed=3)
    c = generate_market(spec=SMALL, noise=0.5, seed=4)
    assert _rows_equal(a.rows, b.rows)
    assert [q.mid for q in a.rows] != [q.mid for q in c.rows]


def test_noisy_ordering():
    m = generate_market(spec=SMALL, noise=1.0, seed=1)
    for q in m.rows:
        assert 0.0 <= q.bid <= q.mid <= q.ask
        for lo, hi in ((q.iv_bid, q.iv_mid), (q.iv_mid, q.iv_ask)):
            if not (math.isnan(lo) or math.isnan(hi)):
                assert lo <= hi
        assert not math.isnan(q.iv_mid)
        lo_b, hi_b = price_bounds(q.forward, q.strike, q.discount_factor, q.option_type == "call")
        assert lo_b < q.mid < hi_b


def test_otm_selection_and_forwards(default_market):
    s = default_market.spec
    for q in default_market.rows:
        F = s.spot * math.exp((s.r - s.q) * q.expiry)
        assert q.forward == pytest.approx(F, rel=1e-14)
        assert q.discount_factor == pytest.approx(math.exp(-s.r * q.expiry), rel=1e-14)
        assert q.log_moneyness == pytest.approx(math.log(q.strike / F), abs=1e-14)
        assert (q.option_type == "put") == (q.strike < F)


def test_min_price_drop_count():
    spec = MarketSpec(expiries=(0.1,), strike_moneyness=(0.6, 1.0, 1.5), min_price=0.05)
    m = generate_market(spec=spec)
    p = heston_price(
        100 * math.exp(0.02 * 0.1),
        [60.0, 100.0, 150.0],
        0.1,
        DEFAULT_HESTON,
        math.exp(-0.03 * 0.1),
        [False, True, True],
    )
    assert m.n_dropped == int((p < 0.05).sum())
    assert len(m.rows) == 3 - m.n_dropped
    high = generate_market(spec=MarketSpec(expiries=(0.1,), min_price=1e9))
    assert high.rows == []
    assert high.n_dropped == 19


def test_roundtrip_json_csv(tmp_path):
    m = generate_market(spec=SMALL, noise=1.0, seed=7)
    # force some NaN fields
    m.rows[0] = type(m.rows[0])(**{**asdict(m.rows[0]), "iv_bid": math.nan, "true_price": math.nan})
    jp, cp = tmp_path / "m.json", tmp_path / "m.csv"
    m.to_json(jp)
    m.to_csv(cp)
    assert "NaN" not in jp.read_text()
    json.loads(jp.read_text())
    lj = load_market(jp)
    assert lj.spec == m.spec
    assert lj.heston == m.heston
    assert (lj.noise, lj.seed, lj.n_dropped) == (m.noise, m.seed, m.n_dropped)
    assert _rows_equal(lj.rows, m.rows)
    lc = load_market(cp)
    assert lc.heston is None
    assert lc.spec.expiries == SMALL.expiries
    assert _rows_equal(lc.rows, m.rows)
    with pytest.raises(ValueError, match="unsupported"):
        load_market(tmp_path / "m.txt")


def test_slices(default_market):
    sl = default_market.slices()
    assert [t for t, _, _ in sl] == list(default_market.spec.expiries)
    for _, k, iv in sl:
        assert k.shape == iv.shape
        assert np.all(np.diff(k) > 0)
        assert not np.isnan(iv).any()
    assert [q.strike for q in default_market.slice(1.0)] == sorted(
        q.strike for q in default_market.slice(1.0)
    )


def test_skew(default_market):
    for T in (0.25, 1.0):
        qs = [q for q in default_market.slice(T) if abs(q.log_moneyness) < 0.15]
        ivs = [q.iv_true for q in qs]
        assert len(ivs) >= 3
        assert all(a > b for a, b in itertools.pairwise(ivs))


def test_spec_validation_and_dict():
    s = MarketSpec()
    assert len(s.strike_moneyness) == 19
    assert isinstance(s.strike_moneyness, tuple)
    assert MarketSpec.from_dict(s.to_dict()) == s
    with pytest.raises(ValueError, match="increasing"):
        MarketSpec(expiries=(1.0, 0.5))
    with pytest.raises(ValueError, match="positive"):
        MarketSpec(expiries=(-1.0,))
    with pytest.raises(ValueError, match="tick"):
        MarketSpec(tick=0.0)
