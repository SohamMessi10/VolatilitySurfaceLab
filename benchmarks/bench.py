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
from volsurface.black_scholes import bs_price, normalized_otm
from volsurface.implied_vol import PriceOutOfBoundsError, implied_vol


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes")
    parser.add_argument("--json", help="also write results to this JSON file")
    args = parser.parse_args(argv)

    results = {"machine": machine_info()}
    results["implied_vol"] = bench_implied_vol(n=2000 if args.quick else 20000, seed=0)
    text = json.dumps(results, indent=2)
    print(text)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
