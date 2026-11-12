"""Tests for butterfly and calendar arbitrage diagnostics."""

from __future__ import annotations

import json

import numpy as np

from volsurface.arbitrage import (
    ArbitrageReport,
    check_butterfly,
    check_calendar,
    check_surface,
    durrleman_g,
)
from volsurface.svi import SliceFit, SVIParams, fit_svi_slice, svi_implied_vol, svi_total_variance

GRID = np.linspace(-1.5, 1.5, 3001)
STEP = GRID[1] - GRID[0]
CLEAN = SVIParams(0.04, 0.1, -0.5, 0.0, 0.2)
# Axel Vogt example (Gatheral & Jacquier 2014): butterfly arbitrage.
VOGT = SVIParams(-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)


def test_clean_slice_has_no_butterfly():
    assert np.nanmin(durrleman_g(GRID, CLEAN)) > 0.3
    assert check_butterfly(CLEAN, 1.0, GRID) == []


def test_vogt_butterfly_detected():
    g = durrleman_g(GRID, VOGT)
    neg = g < -1e-10
    assert neg.any()
    v = check_butterfly(VOGT, 1.0, GRID)
    assert len(v) == 1
    x = v[0]
    assert x.kind == "butterfly"
    assert x.T == 1.0
    assert x.T_prev is None
    # Measured: g < 0 on k in [0.643, 1.256], minimum -0.0329 at k = 0.879.
    assert abs(x.k_start - 0.643) < 0.01
    assert abs(x.k_end - 1.256) < 0.01
    assert abs(x.k_worst - 0.879) < 0.01
    assert x.k_start <= x.k_worst <= x.k_end
    assert x.worst_value < 0
    # The interval agrees with a direct computation of where g < 0.
    assert x.k_start == GRID[neg][0]
    assert x.k_end == GRID[neg][-1]
    assert durrleman_g(x.k_worst, VOGT) < 0
    assert abs(x.worst_value - g.min()) < 1e-12
    # Just outside the interval g is non-negative.
    assert durrleman_g(x.k_start - STEP, VOGT) >= -1e-10
    assert durrleman_g(x.k_end + STEP, VOGT) >= -1e-10


def test_calendar_clean_pair_passes():
    later = SVIParams(0.06, 0.12, -0.5, 0.0, 0.2)
    assert check_calendar([(0.5, CLEAN), (1.0, later)], GRID) == []


def test_calendar_violation_detected():
    # Later slice has lower variance on the left wing (smiles cross).
    later = SVIParams(0.03, 0.1, 0.5, 0.0, 0.2)
    fits = [(1.0, later), (0.5, CLEAN)]  # unsorted on purpose
    diff = svi_total_variance(GRID, later) - svi_total_variance(GRID, CLEAN)
    bad = diff < -1e-10
    assert bad.any()
    v = check_calendar(fits, GRID)
    assert len(v) == 1
    x = v[0]
    assert x.kind == "calendar"
    assert x.T == 1.0
    assert x.T_prev == 0.5
    assert x.k_start == GRID[bad][0]
    assert x.k_end == GRID[bad][-1]
    assert x.worst_value == diff.min() < 0
    assert x.k_worst == GRID[np.argmin(diff)]
    assert x.k_start < 0  # lies on the left wing


def test_check_surface_passed_and_json():
    later = SVIParams(0.06, 0.12, -0.5, 0.0, 0.2)
    rep = check_surface([(0.5, CLEAN), (1.0, later)])
    assert isinstance(rep, ArbitrageReport)
    assert rep.passed
    assert rep.n_grid == 3001
    assert (rep.k_grid_min, rep.k_grid_max) == (-1.5, 1.5)
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["passed"] is True
    assert d["butterfly"] == []
    assert d["calendar"] == []


def test_check_surface_reports_violations_and_json():
    crossing = SVIParams(0.03, 0.1, 0.5, 0.0, 0.2)
    rep = check_surface([(0.5, CLEAN), (1.0, crossing), (2.0, VOGT)])
    assert not rep.passed
    assert len(rep.butterfly) == 1
    assert rep.butterfly[0].T == 2.0
    assert rep.calendar
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["passed"] is False
    assert d["calendar"][0]["T_prev"] == 0.5
    assert d["butterfly"][0]["T_prev"] is None


def test_check_surface_accepts_slice_fits():
    k = np.linspace(-1, 1, 25)
    fits = [
        fit_svi_slice(k, svi_implied_vol(k, T, p), T)
        for T, p in [(0.5, CLEAN), (1.0, SVIParams(0.06, 0.12, -0.5, 0.0, 0.2))]
    ]
    assert all(isinstance(f, SliceFit) for f in fits)
    assert check_surface(fits).passed
