"""Command line interface ``vsl``.

Subcommands: ``generate`` (synthetic Heston market), ``calibrate`` (SVI fit per expiry),
``check`` (static-arbitrage report), ``plot`` (interactive HTML report) and ``run``
(all of the above into one directory).

Exit codes:

* 0 - success / no arbitrage detected
* 1 - arbitrage detected (``check``, ``run``)
* 2 - usage error (argparse)
* 3 - input error: missing/unreadable/malformed file, invalid parameters
* 4 - numerical failure: implied-vol inversion, Heston pricing or an SVI fit failed

Expected errors are printed as ``vsl: error: <message>`` on stderr without a traceback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from . import __version__
from .arbitrage import ArbitrageReport, check_surface
from .heston import HestonError, HestonParams
from .implied_vol import ImpliedVolError
from .market import DEFAULT_HESTON, MarketData, MarketSpec, generate_market, load_market
from .plotting import fit_error_vs_true, write_report_html
from .svi import SliceFit, fit_svi_surface

__all__ = ["main"]

EXIT_OK = 0
EXIT_ARBITRAGE = 1
EXIT_INPUT = 3
EXIT_NUMERICAL = 4

_EPILOG = """\
exit codes:
  0  success / no arbitrage detected
  1  arbitrage detected (check, run)
  2  usage error
  3  input error (missing/unreadable/malformed file, invalid parameters)
  4  numerical failure (implied vol inversion, Heston pricing, failed SVI fit)
"""


class CliError(Exception):
    """An expected failure with an exit code."""

    def __init__(self, message: str, code: int = EXIT_INPUT) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- generate


def _add_model_args(p: argparse.ArgumentParser) -> None:
    d, s = DEFAULT_HESTON, MarketSpec()
    p.add_argument("--noise", type=float, default=0.0, help="quote noise level (default 0)")
    p.add_argument("--seed", type=int, default=None, help="random seed for the noise")
    for name in ("v0", "kappa", "theta", "xi", "rho"):
        p.add_argument(f"--{name}", type=float, default=getattr(d, name), help="Heston %(dest)s")
    p.add_argument("--spot", type=float, default=s.spot, help="spot price")
    p.add_argument("--r", type=float, default=s.r, help="risk-free rate")
    p.add_argument("--q", type=float, default=s.q, help="dividend yield")
    p.add_argument(
        "--expiries",
        default=None,
        help=f"comma-separated expiries in years (default {','.join(map(str, s.expiries))})",
    )


def _build_market(args: argparse.Namespace) -> MarketData:
    try:
        params = HestonParams(args.v0, args.kappa, args.theta, args.xi, args.rho)
        kw: dict = {"spot": args.spot, "r": args.r, "q": args.q}
        if args.expiries is not None:
            kw["expiries"] = tuple(float(x) for x in args.expiries.split(",") if x.strip())
        spec = MarketSpec(**kw)
        if args.noise < 0:
            raise ValueError("noise must be >= 0")
    except ValueError as e:  # includes HestonError
        raise CliError(f"invalid parameters: {e}") from e
    try:
        return generate_market(params, spec, noise=args.noise, seed=args.seed)
    except (HestonError, ImpliedVolError) as e:
        raise CliError(f"market generation failed: {e}", EXIT_NUMERICAL) from e
    except ValueError as e:
        raise CliError(f"invalid parameters: {e}") from e


def _write_market(market: MarketData, json_path: Path, csv_path: Path | None) -> None:
    try:
        market.to_json(json_path)
        if csv_path is not None:
            market.to_csv(csv_path)
    except OSError as e:
        raise CliError(f"cannot write {e.filename or json_path}: {e.strerror}") from e


def _summary_line(market: MarketData, path: Path) -> str:
    exp = ", ".join(f"{t:g}" for t in market.spec.expiries)
    return (
        f"generated {len(market.rows)} quotes ({market.n_dropped} dropped) "
        f"for expiries [{exp}] -> {path}"
    )


def cmd_generate(args: argparse.Namespace) -> int:
    market = _build_market(args)
    out = Path(args.output)
    _write_market(market, out, Path(args.csv) if args.csv else None)
    print(_summary_line(market, out))
    return EXIT_OK


# --------------------------------------------------------------------------- calibrate


def _load_market(path: str) -> MarketData:
    try:
        return load_market(path)
    except FileNotFoundError as e:
        raise CliError(f"market file not found: {path}") from e
    except OSError as e:
        raise CliError(f"cannot read {path}: {e.strerror}") from e
    except (ValueError, KeyError, TypeError) as e:  # JSONDecodeError is a ValueError
        raise CliError(f"malformed market file {path}: {e!r}") from e


def _calibrate(market: MarketData, market_file: str) -> tuple[list[SliceFit], dict]:
    slices = market.slices()
    if not slices:
        raise CliError("market has no usable quotes (all implied vols are nan)")
    try:
        fits = fit_svi_surface(slices)
    except ValueError as e:
        raise CliError(f"cannot fit SVI: {e}") from e
    failed = [f for f in fits if not f.success or not math.isfinite(f.rmse_vol_points)]
    if failed:
        ts = ", ".join(f"{f.T:g}" for f in failed)
        raise CliError(f"SVI fit failed for expiries [{ts}]: {failed[0].message}", EXIT_NUMERICAL)
    blob = {
        "version": 1,
        "market_file": str(market_file),
        "slices": [f.to_dict() for f in fits],
        "error_vs_true": fit_error_vs_true(fits, market),
    }
    return fits, blob


def _print_fit_table(fits: list[SliceFit], blob: dict) -> None:
    tmap = {round(d["T"], 9): d for d in blob["error_vs_true"] or []}
    print(
        f"{'T':>7} {'n':>4} {'k range':>16} {'RMSE mkt':>9} {'max mkt':>8} "
        f"{'RMSE true':>10} {'max true':>9}   (vol points)"
    )
    for f in fits:
        t = tmap.get(round(f.T, 9))
        rt = f"{t['rmse_vol_points']:10.3f}" if t else f"{'n/a':>10}"
        mt = f"{t['max_abs_vol_points']:9.3f}" if t else f"{'n/a':>9}"
        rng = f"[{f.k_min:6.2f},{f.k_max:6.2f}]"
        print(
            f"{f.T:7.4g} {f.n_points:4d} {rng:>16} {f.rmse_vol_points:9.3f} "
            f"{f.max_abs_vol_points:8.3f} {rt} {mt}"
        )


def _write_json(path: Path, blob: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=1, allow_nan=False)
    except OSError as e:
        raise CliError(f"cannot write {path}: {e.strerror}") from e


def cmd_calibrate(args: argparse.Namespace) -> int:
    market = _load_market(args.market)
    fits, blob = _calibrate(market, args.market)
    _print_fit_table(fits, blob)
    _write_json(Path(args.output), blob)
    print(f"wrote {args.output}")
    return EXIT_OK


# --------------------------------------------------------------------------- check


def _load_fits(path: str) -> list[SliceFit]:
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        fits = [SliceFit.from_dict(d) for d in blob["slices"]]
    except FileNotFoundError as e:
        raise CliError(f"fit file not found: {path}") from e
    except OSError as e:
        raise CliError(f"cannot read {path}: {e.strerror}") from e
    except (ValueError, KeyError, TypeError) as e:
        raise CliError(f"malformed fit file {path}: {e!r}") from e
    if not fits:
        raise CliError(f"fit file {path} contains no slices")
    return sorted(fits, key=lambda f: f.T)


def _run_check(fits: list[SliceFit], k_min: float, k_max: float, n_grid: int) -> ArbitrageReport:
    import numpy as np

    if not (math.isfinite(k_min) and math.isfinite(k_max) and k_min < k_max):
        raise CliError("--k-min must be smaller than --k-max")
    if n_grid < 2:
        raise CliError("--n-grid must be at least 2")
    return check_surface(fits, np.linspace(k_min, k_max, n_grid))


def _print_report(report: ArbitrageReport, quoted_only: bool) -> bool:
    """Print violations and the verdict; returns True when the check passed."""
    vs = report.butterfly + report.calendar
    for v in vs:
        where = f"T={v.T:g}" if v.T_prev is None else f"T={v.T_prev:g} -> {v.T:g}"
        where = f"{where:<16}"
        place = "inside quotes" if v.in_quoted_range else "outside quotes"
        what = "min g" if v.kind == "butterfly" else "min dw"
        print(
            f"  {v.kind:<9} {where} k in [{v.k_start:+.3f}, {v.k_end:+.3f}]  "
            f"{what}={v.worst_value:.3e} at k={v.k_worst:+.3f}  ({place})"
        )
    n_b, n_c = len(report.butterfly), len(report.calendar)
    n_in = sum(1 for v in vs if v.in_quoted_range)
    passed = report.passed_in_quoted_range if quoted_only else report.passed
    detail = f"{n_b} butterfly, {n_c} calendar violation(s), {n_in} inside the quoted range"
    grid = f"k in [{report.k_grid_min:g}, {report.k_grid_max:g}], {report.n_grid} points"
    scope = " (quoted range only)" if quoted_only else ""
    print(f"{'PASS' if passed else 'FAIL'}{scope}: {detail}; grid {grid}")
    return passed


def cmd_check(args: argparse.Namespace) -> int:
    fits = _load_fits(args.fit)
    report = _run_check(fits, args.k_min, args.k_max, args.n_grid)
    passed = _print_report(report, args.quoted_only)
    if args.json:
        _write_json(Path(args.json), report.to_dict())
    return EXIT_OK if passed else EXIT_ARBITRAGE


# --------------------------------------------------------------------------- plot


def _plot(
    path: Path,
    fits: list[SliceFit],
    market: MarketData | None,
    report: ArbitrageReport | None,
    plotlyjs: str,
    title: str | None,
) -> None:
    kw = {} if title is None else {"title": title}
    try:
        write_report_html(path, fits, market, report, include_plotlyjs=plotlyjs, **kw)
    except OSError as e:
        raise CliError(f"cannot write {path}: {e.strerror}") from e


def cmd_plot(args: argparse.Namespace) -> int:
    fits = _load_fits(args.fit)
    market = _load_market(args.market) if args.market else None
    report = _run_check(fits, -1.5, 1.5, 3001)
    _plot(Path(args.output), fits, market, report, args.plotlyjs, args.title)
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.2f} MB)")
    return EXIT_OK


# --------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise CliError(f"cannot create {outdir}: {e.strerror}") from e
    market = _build_market(args)
    mjson, mcsv = outdir / "market.json", outdir / "market.csv"
    _write_market(market, mjson, mcsv)
    print(_summary_line(market, mjson))
    fits, blob = _calibrate(market, str(mjson))
    _print_fit_table(fits, blob)
    _write_json(outdir / "fit.json", blob)
    report = _run_check(fits, -1.5, 1.5, 3001)
    passed = _print_report(report, quoted_only=False)
    _write_json(outdir / "report.json", report.to_dict())
    _plot(outdir / "surface.html", fits, market, report, args.plotlyjs, args.title)
    print(f"wrote market.json, market.csv, fit.json, report.json, surface.html in {outdir}")
    return EXIT_OK if passed else EXIT_ARBITRAGE


# --------------------------------------------------------------------------- parser


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="vsl",
        description="Volatility surface lab: Heston market, SVI fit, arbitrage check, report.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"vsl {__version__}")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, func, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(
            name,
            help=help_,
            description=help_,
            epilog=_EPILOG,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        p.set_defaults(func=func)
        return p

    p = add("generate", cmd_generate, "generate a synthetic Heston option market")
    p.add_argument("-o", "--output", default="market.json", help="market JSON (market.json)")
    p.add_argument("--csv", default=None, help="also write the quotes as CSV")
    _add_model_args(p)

    p = add("calibrate", cmd_calibrate, "fit SVI to every expiry of a market")
    p.add_argument("market", help="market file (.json or .csv)")
    p.add_argument("-o", "--output", default="fit.json", help="fit JSON (fit.json)")

    p = add("check", cmd_check, "check fitted slices for butterfly/calendar arbitrage")
    p.add_argument("fit", help="fit JSON from 'vsl calibrate'")
    p.add_argument("--json", default=None, help="write the report as JSON")
    p.add_argument("--k-min", type=float, default=-1.5, help="grid start (-1.5)")
    p.add_argument("--k-max", type=float, default=1.5, help="grid end (1.5)")
    p.add_argument("--n-grid", type=int, default=3001, help="grid points (3001)")
    p.add_argument(
        "--quoted-only",
        action="store_true",
        help="exit 0 if all violations lie outside the quoted range",
    )

    p = add("plot", cmd_plot, "write the interactive HTML report")
    p.add_argument("fit", help="fit JSON from 'vsl calibrate'")
    p.add_argument("--market", default=None, help="market file to overlay (.json or .csv)")
    p.add_argument("-o", "--output", default="surface.html", help="HTML file (surface.html)")
    p.add_argument("--plotlyjs", choices=("inline", "cdn"), default="inline", help="plotly.js")
    p.add_argument("--title", default=None, help="report title")

    p = add("run", cmd_run, "generate, calibrate, check and plot into a directory")
    p.add_argument("--outdir", default="out", help="output directory (out)")
    p.add_argument("--plotlyjs", choices=("inline", "cdn"), default="inline", help="plotly.js")
    p.add_argument("--title", default=None, help="report title")
    _add_model_args(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code (see the module docstring)."""
    args = _build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"vsl: error: {e}", file=sys.stderr)
        return e.code
    except (HestonError, ImpliedVolError) as e:
        print(f"vsl: error: numerical failure: {e}", file=sys.stderr)
        return EXIT_NUMERICAL


if __name__ == "__main__":
    raise SystemExit(main())
