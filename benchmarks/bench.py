"""Reproducible accuracy / speed measurements quoted in PROGRESS.md and README.md.

Run:  python benchmarks/bench.py [--quick] [--json out.json]

Every run prints the machine, Python and library versions and the exact settings, so
the numbers can be compared across machines. Timings are wall-clock medians.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time

import numpy as np
import scipy

import volsurface
from volsurface.arbitrage import check_surface
from volsurface.black_scholes import bs_price, normalized_otm
from volsurface.heston import heston_call_lewis, heston_otm_fft
from volsurface.implied_vol import PriceOutOfBoundsError, implied_vol
from volsurface.market import DEFAULT_HESTON, generate_market
from volsurface.svi import fit_svi_surface, svi_implied_vol


def machine_info() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        if sys.platform == "darwin":
            import subprocess

            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "volsurface": volsurface.__version__,
    }


def timed(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_implied_vol(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        k = rng.uniform(-3, 3)
        sig = rng.uniform(0.01, 3.0)
        T = rng.uniform(1 / 365, 10.0)
        F = rng.uniform(1.0, 1000.0)
        df = rng.uniform(0.5, 1.0)
        is_call = bool(rng.integers(2))
        K = F * math.exp(k)
        price = float(bs_price(F, K, T, sig, df, is_call))
        tv = float(normalized_otm(math.log(K / F), sig * math.sqrt(T)))
        cases.append((price, F, K, T, df, is_call, sig, tv))

    methods = {"newton": 0, "brent": 0}
    rejected = rejected_unresolvable = 0
    errors = []
    t0 = time.perf_counter()
    for price, F, K, T, df, is_call, sig, tv in cases:
        try:
            res = implied_vol(price, F, K, T, df, is_call, details=True)
        except PriceOutOfBoundsError:
            rejected += 1
            rejected_unresolvable += tv < 1e-12 * price / (df * F) or tv < 1e-250
            continue
        methods[res.method] += 1
        if tv > 1e-8 * price / (df * F):
            errors.append(abs(res.sigma - sig) / sig)
    elapsed = time.perf_counter() - t0
    solved = methods["newton"] + methods["brent"]
    return {
        "settings": {
            "n": n,
            "seed": seed,
            "k": [-3, 3],
            "sigma": [0.01, 3.0],
            "T_years": [1 / 365, 10.0],
            "F": [1, 1000],
            "df": [0.5, 1.0],
        },
        "solved": solved,
        "newton_share": methods["newton"] / solved,
        "brent_share": methods["brent"] / solved,
        "rejected": rejected,
        "rejected_with_unresolvable_time_value": rejected_unresolvable,
        "max_rel_vol_error_resolvable": max(errors),
        "median_rel_vol_error_resolvable": statistics.median(errors),
        "us_per_inversion": elapsed / n * 1e6,
    }


def bench_heston(repeats: int) -> dict:
    p = DEFAULT_HESTON
    k = np.linspace(-1.5, 1.0, 26)
    maturities = (0.05, 0.25, 1.0, 3.0, 10.0)
    max_err = {}
    for T in maturities:
        c = heston_call_lewis(k, T, p)
        ref = np.where(k >= 0, c, c + np.expm1(k))
        max_err[str(T)] = float(np.max(np.abs(heston_otm_fft(k, T, p) - ref)))
    fft_ms = timed(lambda: heston_otm_fft(k, 1.0, p), repeats) * 1e3
    lewis_ms = timed(lambda: heston_call_lewis(k[:5], 1.0, p), max(1, repeats // 5)) * 1e3 / 5
    return {
        "settings": {
            "params": p.to_dict(),
            "k": "linspace(-1.5, 1.0, 26)",
            "fft": {"n": 2**15, "eta": 0.1, "alpha": 1.5},
            "reference": "Lewis (2001) integral, scipy.integrate.quad on [0, inf)",
        },
        "max_abs_error_vs_lewis_normalized_by_forward": max_err,
        "fft_ms_per_expiry_26_strikes": fft_ms,
        "lewis_ms_per_strike": lewis_ms,
    }


def _error_vs_true(market, fits) -> list[dict]:
    rows = []
    for f in fits:
        qs = [q for q in market.slice(f.T) if math.isfinite(q.iv_mid)]
        k = np.array([q.log_moneyness for q in qs])
        iv_true = np.array([q.iv_true for q in qs])
        err = 100.0 * (svi_implied_vol(k, f.T, f.params) - iv_true)
        rows.append(
            {
                "T": f.T,
                "n": f.n_points,
                "rmse_vs_market_vp": f.rmse_vol_points,
                "max_vs_market_vp": f.max_abs_vol_points,
                "rmse_vs_true_vp": float(np.sqrt(np.mean(err**2))),
                "max_vs_true_vp": float(np.max(np.abs(err))),
            }
        )
    return rows


def bench_pipeline(n_seeds: int) -> dict:
    out = {}
    t0 = time.perf_counter()
    market = generate_market(noise=0.0)
    t1 = time.perf_counter()
    fits = fit_svi_surface(market.slices())
    t2 = time.perf_counter()
    report = check_surface(fits)
    t3 = time.perf_counter()
    out["clean"] = {
        "quotes": len(market.rows),
        "dropped": market.n_dropped,
        "generate_s": t1 - t0,
        "calibrate_s": t2 - t1,
        "check_s": t3 - t2,
        "arbitrage_free_full_grid": report.passed,
        "slices": _error_vs_true(market, fits),
    }
    passed_full = passed_quoted = 0
    worst_true = 0.0
    calib_times = []
    for seed in range(n_seeds):
        m = generate_market(noise=1.0, seed=seed)
        s0 = time.perf_counter()
        f = fit_svi_surface(m.slices())
        calib_times.append(time.perf_counter() - s0)
        r = check_surface(f)
        passed_full += r.passed
        passed_quoted += r.passed_in_quoted_range
        worst_true = max(worst_true, max(row["max_vs_true_vp"] for row in _error_vs_true(m, f)))
    out["noisy"] = {
        "settings": {"noise": 1.0, "seeds": f"0..{n_seeds - 1}", "rel_half_spread": 0.02},
        "arbitrage_free_in_quoted_range": f"{passed_quoted}/{n_seeds}",
        "arbitrage_free_full_grid_k_-1.5_1.5": f"{passed_full}/{n_seeds}",
        "worst_svi_error_vs_true_vp": worst_true,
        "median_calibrate_s": statistics.median(calib_times),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes")
    parser.add_argument("--json", help="also write results to this JSON file")
    args = parser.parse_args(argv)

    results = {"machine": machine_info()}
    results["implied_vol"] = bench_implied_vol(n=2000 if args.quick else 20000, seed=0)
    results["heston"] = bench_heston(repeats=5 if args.quick else 20)
    results["pipeline"] = bench_pipeline(n_seeds=3 if args.quick else 20)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
