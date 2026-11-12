"""Interactive plotly report for a calibrated SVI surface.

Two figures and one HTML page:

* :func:`build_surface_figure` - 3D view (x = log-moneyness ``k``, y = expiry ``T`` in years,
  z = implied vol in %) with the fitted surface, the market mid vols and arbitrage flags.
* :func:`build_smile_figure` - one panel per expiry (market mid vol with bid/ask bars, SVI
  curve, true Heston vol, shaded butterfly-violation regions).
* :func:`write_report_html` - a single self-contained page with a plain-language summary
  and both figures; plotly.js is embedded once (or loaded from the CDN).

Honesty note repeated in the report: SVI is an approximation of the Heston smile, not the
model itself, so a small non-zero error against the true Heston vol is expected.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from .arbitrage import ArbitrageReport, Violation
from .market import MarketData, Quote
from .svi import SliceFit, svi_implied_vol, svi_total_variance

__all__ = [
    "build_smile_figure",
    "build_surface_figure",
    "fit_error_vs_true",
    "interpolate_total_variance",
    "write_report_html",
]

# Palette (reference categorical slots, light mode) - see the dataviz skill.
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_INK = "#0b0b0b"
_MUTED = "#52514e"
_CRIT = "#d03b3b"
_SERIOUS = "#ec835a"
_GRID = "#e4e3df"
_SURFACE_SCALE = [[0.0, "#cde2fb"], [0.5, "#5598e7"], [1.0, "#104281"]]

_K_PAD = 0.05
_N_K = 121
_N_BETWEEN = 4
_MATCH_TOL = 1e-9


# --------------------------------------------------------------------------- helpers


def _sorted_fits(fits: Sequence[SliceFit]) -> list[SliceFit]:
    if len(fits) == 0:
        raise ValueError("need at least one fitted slice")
    return sorted(fits, key=lambda f: f.T)


def _quotes_at(market: MarketData | None, T: float) -> list[Quote]:
    if market is None:
        return []
    return [q for q in market.rows if abs(q.expiry - T) <= _MATCH_TOL]


def _fit_at(fits: Sequence[SliceFit], T: float) -> SliceFit | None:
    for f in fits:
        if abs(f.T - T) <= _MATCH_TOL:
            return f
    return None


def interpolate_total_variance(
    fits: Sequence[SliceFit], k_grid: NDArray, n_between: int = _N_BETWEEN
) -> tuple[NDArray, NDArray]:
    """Total variance on a ``(T, k)`` grid, ``W[i, j] = w(k_j, T_i)``.

    Rows are every slice expiry plus ``n_between`` rows between consecutive expiries.
    Between slices the *total variance* is interpolated linearly in T at fixed k, which
    preserves calendar monotonicity (if both slices satisfy ``w(k, T1) <= w(k, T2)`` then so
    does every convex combination). No extrapolation before the first or after the last
    expiry. Implied vol is recovered as ``sqrt(w / T)``.
    """
    fs = _sorted_fits(fits)
    k = np.asarray(k_grid, dtype=float)
    w_slices = [np.maximum(svi_total_variance(k, f.params), 0.0) for f in fs]
    T_rows: list[float] = [fs[0].T]
    W_rows: list[NDArray] = [w_slices[0]]
    for i in range(len(fs) - 1):
        for j in range(1, n_between + 1):
            lam = j / (n_between + 1)
            T_rows.append((1 - lam) * fs[i].T + lam * fs[i + 1].T)
            W_rows.append((1 - lam) * w_slices[i] + lam * w_slices[i + 1])
        T_rows.append(fs[i + 1].T)
        W_rows.append(w_slices[i + 1])
    return np.array(T_rows), np.vstack(W_rows)


def fit_error_vs_true(fits: Sequence[SliceFit], market: MarketData | None) -> list[dict] | None:
    """Per-slice SVI error against the true Heston IV in vol points, or ``None``.

    Uses the quotes that were available to the fit (finite ``iv_mid``) and finite ``iv_true``.
    """
    if market is None:
        return None
    out = []
    for f in _sorted_fits(fits):
        qs = [
            q
            for q in _quotes_at(market, f.T)
            if math.isfinite(q.iv_mid) and math.isfinite(q.iv_true)
        ]
        if not qs:
            continue
        k = np.array([q.log_moneyness for q in qs])
        truth = np.array([q.iv_true for q in qs])
        err = 100.0 * (svi_implied_vol(k, f.T, f.params) - truth)
        out.append(
            {
                "T": float(f.T),
                "rmse_vol_points": float(np.sqrt(np.mean(err**2))),
                "max_abs_vol_points": float(np.max(np.abs(err))),
            }
        )
    return out or None


def _quoted_k_range(fits: Sequence[SliceFit]) -> tuple[float, float]:
    return min(f.k_min for f in fits), max(f.k_max for f in fits)


def _all_violations(report: ArbitrageReport | None) -> list[Violation]:
    return [] if report is None else list(report.butterfly) + list(report.calendar)


def _violation_text(v: Violation) -> str:
    where = f"T={v.T:.4g}" if v.T_prev is None else f"T={v.T_prev:.4g} → {v.T:.4g}"
    rng = "inside quoted range" if v.in_quoted_range else "outside quoted range (wings)"
    what = "min density g" if v.kind == "butterfly" else "variance gap Δw"
    return (
        f"{v.kind}, {where}, k ∈ [{v.k_start:.3f}, {v.k_end:.3f}], "
        f"worst {what} = {v.worst_value:.3g} at k = {v.k_worst:.3f} ({rng})"
    )


def _fmt(x: float, nd: int = 2) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- 3D figure


def build_surface_figure(
    fits: Sequence[SliceFit],
    market: MarketData | None = None,
    report: ArbitrageReport | None = None,
) -> go.Figure:
    """3D implied-vol surface: fitted SVI, market mids and arbitrage flags.

    Axes: x = log-moneyness ``k = ln(K/F)``, y = expiry ``T`` (years), z = implied vol (%).
    The fitted surface lives on a k grid spanning the union of the quoted ranges padded by
    0.05 and on a T grid containing every slice expiry plus interpolated rows. Between
    slices **total variance is interpolated linearly in T** at fixed k (this preserves
    calendar monotonicity of calendar-clean slices) and ``iv = sqrt(w / T)``; there is no
    extrapolation below the first or above the last expiry.
    """
    fs = _sorted_fits(fits)
    lo, hi = _quoted_k_range(fs)
    k_lo, k_hi = lo - _K_PAD, hi + _K_PAD
    k_grid = np.linspace(k_lo, k_hi, _N_K)
    T_grid, W = interpolate_total_variance(fs, k_grid)
    iv = 100.0 * np.sqrt(W / T_grid[:, None])

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=k_grid,
            y=T_grid,
            z=iv,
            colorscale=_SURFACE_SCALE,
            name="SVI surface",
            showscale=True,
            colorbar={"title": {"text": "IV (%)"}, "len": 0.6, "thickness": 14},
            opacity=0.92,
            hovertemplate="k=%{x:.3f}<br>T=%{y:.3f}<br>IV=%{z:.2f}%<extra>SVI fit</extra>",
        )
    )

    quotes = [q for q in market.rows if math.isfinite(q.iv_mid)] if market is not None else []
    if quotes:

        def f2(x: float) -> str:
            return f"{100 * x:.2f}%" if math.isfinite(x) else "n/a"

        fig.add_trace(
            go.Scatter3d(
                x=[q.log_moneyness for q in quotes],
                y=[q.expiry for q in quotes],
                z=[100 * q.iv_mid for q in quotes],
                mode="markers",
                name="Market IV (mid)",
                marker={"size": 3, "color": _INK, "line": {"width": 0}},
                text=[
                    f"T={q.expiry:.4g}  K={q.strike:.2f}  k={q.log_moneyness:.3f}<br>"
                    f"IV bid/mid/ask = {f2(q.iv_bid)} / {f2(q.iv_mid)} / {f2(q.iv_ask)}<br>"
                    f"IV true = {f2(q.iv_true)}"
                    for q in quotes
                ],
                hovertemplate="%{text}<extra>market</extra>",
            )
        )

    # Arbitrage flags: inside vs outside the quoted range, clipped to the plotted k range.
    groups: dict[bool, dict[str, list]] = {
        True: {"x": [], "y": [], "z": [], "text": []},
        False: {"x": [], "y": [], "z": [], "text": []},
    }
    for v in _all_violations(report):
        fit = _fit_at(fs, v.T)
        a, b = max(v.k_start, k_lo), min(v.k_end, k_hi)
        if fit is None or a > b:
            continue
        n = max(2, round((b - a) / (k_hi - k_lo) * (_N_K - 1)) + 1)
        ks = np.linspace(a, b, n)
        z = 100.0 * svi_implied_vol(ks, fit.T, fit.params)
        g = groups[bool(v.in_quoted_range)]
        label = _violation_text(v)
        g["x"].extend(ks.tolist())
        g["y"].extend([fit.T] * n)
        g["z"].extend(z.tolist())
        g["text"].extend([label] * n)
    styles = {
        True: ("Arbitrage (inside quoted range)", _CRIT, "diamond"),
        False: ("Arbitrage (outside quotes, wings)", _SERIOUS, "x"),
    }
    for inside, (name, color, symbol) in styles.items():
        g = groups[inside]
        if not g["x"]:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="markers",
                name=name,
                marker={"size": 4, "color": color, "symbol": symbol},
                text=g["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.02, "x": 0},
        scene={
            "xaxis": {"title": {"text": "Log-moneyness k = ln(K/F)"}},
            "yaxis": {"title": {"text": "Expiry T (years)"}},
            "zaxis": {"title": {"text": "Implied vol (%)"}},
            "camera": {"eye": {"x": 1.6, "y": -1.7, "z": 0.9}},
        },
    )
    return fig


# --------------------------------------------------------------------------- smiles


def build_smile_figure(
    fits: Sequence[SliceFit],
    market: MarketData | None = None,
    report: ArbitrageReport | None = None,
) -> go.Figure:
    """Small multiples, one panel per expiry, all on the same implied-vol axis (%).

    Each panel shows market mid vols with bid/ask as error bars (quotes lacking a finite
    bid or ask vol get no bar), the SVI curve over the slice's quoted range +/- 0.1, the true
    Heston vol as a dashed line when available, and shaded k-regions where the slice has
    butterfly violations.
    """
    fs = _sorted_fits(fits)
    n = len(fs)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"T = {f.T:.4g} y" for f in fs],
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=min(0.12, 0.35 / nrows),
    )
    seen: set[str] = set()

    def show(name: str) -> bool:
        if name in seen:
            return False
        seen.add(name)
        return True

    butterflies = report.butterfly if report is not None else []
    for idx, f in enumerate(fs):
        row, col = idx // ncols + 1, idx % ncols + 1
        ks = np.linspace(f.k_min - 0.1, f.k_max + 0.1, 200)
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=100 * svi_implied_vol(ks, f.T, f.params),
                mode="lines",
                name="SVI fit",
                legendgroup="svi",
                showlegend=show("SVI fit"),
                line={"color": _BLUE, "width": 2},
                hovertemplate="k=%{x:.3f}<br>SVI %{y:.2f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )
        qs = sorted(_quotes_at(market, f.T), key=lambda q: q.log_moneyness)
        mids = [q for q in qs if math.isfinite(q.iv_mid)]
        truth = [q for q in qs if math.isfinite(q.iv_true)]
        if truth:
            fig.add_trace(
                go.Scatter(
                    x=[q.log_moneyness for q in truth],
                    y=[100 * q.iv_true for q in truth],
                    mode="lines",
                    name="True Heston IV",
                    legendgroup="true",
                    showlegend=show("True Heston IV"),
                    line={"color": _ORANGE, "width": 2, "dash": "dash"},
                    hovertemplate="k=%{x:.3f}<br>true %{y:.2f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )
        if mids:
            up = [100 * (q.iv_ask - q.iv_mid) if math.isfinite(q.iv_ask) else 0.0 for q in mids]
            down = [100 * (q.iv_mid - q.iv_bid) if math.isfinite(q.iv_bid) else 0.0 for q in mids]
            fig.add_trace(
                go.Scatter(
                    x=[q.log_moneyness for q in mids],
                    y=[100 * q.iv_mid for q in mids],
                    mode="markers",
                    name="Market mid (bars: bid/ask)",
                    legendgroup="mkt",
                    showlegend=show("Market mid"),
                    marker={"size": 6, "color": _INK, "line": {"width": 1, "color": "#fcfcfb"}},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": up,
                        "arrayminus": down,
                        "thickness": 1,
                        "width": 2,
                        "color": _MUTED,
                    },
                    hovertemplate="k=%{x:.3f}<br>mid %{y:.2f}%<extra></extra>",
                ),
                row=row,
                col=col,
            )
        for v in butterflies:
            if abs(v.T - f.T) > _MATCH_TOL:
                continue
            x0, x1 = max(v.k_start, ks[0]), min(v.k_end, ks[-1])
            if x0 > x1:
                continue
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=_CRIT if v.in_quoted_range else _SERIOUS,
                opacity=0.18,
                line_width=0,
                row=row,
                col=col,
            )
        fig.update_xaxes(title_text="k" if row == nrows else None, row=row, col=col)
    fig.update_yaxes(title_text="Implied vol (%)", col=1)
    fig.update_xaxes(gridcolor=_GRID, zeroline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    if butterflies:
        fig.add_annotation(
            text="Shaded: butterfly violation (red = inside quotes, orange = wings)",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.62 / nrows - 0.06,
            showarrow=False,
            font={"size": 11, "color": _MUTED},
            xanchor="left",
        )
    fig.update_layout(
        template="plotly_white",
        height=270 * nrows + 90,
        margin={"l": 50, "r": 10, "t": 40, "b": 90},
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.45 / nrows - 0.04, "x": 0},
    )
    return fig


# --------------------------------------------------------------------------- HTML


_CSS = """
:root{--bg:#fcfcfb;--panel:#ffffff;--ink:#0b0b0b;--muted:#52514e;--line:#e4e3df;
--pass:#0b6b0b;--fail:#d03b3b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.4em 0 .2em}h2{font-size:1.15rem;margin:1.4em 0 .4em}
p,li{max-width:70ch}.muted{color:var(--muted)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px 16px;margin:12px 0}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;font-size:.9rem;width:100%}
th,td{padding:4px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}
.pass{color:var(--pass);font-weight:700}.fail{color:var(--fail);font-weight:700}
.plot{width:100%;overflow:hidden}
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        '<div class="tablewrap"><table>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _summary_html(
    fits: Sequence[SliceFit],
    market: MarketData | None,
    report: ArbitrageReport | None,
    omitted: list[Violation],
) -> str:
    parts: list[str] = []
    parts.append(
        "<p>Each expiry's smile is fitted with a raw SVI curve (a five-parameter formula for "
        "total implied variance). The fits are then checked for static arbitrage: "
        "<em>butterfly</em> (negative implied density inside one expiry) and "
        "<em>calendar</em> (total variance falling as maturity increases).</p>"
    )
    if market is not None:
        items = [f"{len(market.rows)} quotes kept, {market.n_dropped} dropped (too cheap)"]
        h = market.heston
        if h is not None:
            items.append(
                "Heston model: "
                f"v0={h.v0:g}, kappa={h.kappa:g}, theta={h.theta:g}, xi={h.xi:g}, rho={h.rho:g}"
            )
        s = market.spec
        items.append(f"spot={s.spot:g}, r={s.r:g}, q={s.q:g}")
        if isinstance(market.noise, float) and math.isfinite(market.noise):
            seed = "none" if market.seed is None else str(market.seed)
            items.append(f"noise level={market.noise:g}, seed={seed}")
        parts.append(
            '<div class="card"><strong>Market</strong><ul>'
            + "".join(f"<li>{html.escape(i)}</li>" for i in items)
            + "</ul></div>"
        )
    truth = fit_error_vs_true(fits, market)
    tmap = {round(d["T"], 9): d for d in truth or []}
    rows = []
    for f in _sorted_fits(fits):
        t = tmap.get(round(f.T, 9))
        rows.append(
            [
                f"{f.T:.4g}",
                str(f.n_points),
                f"[{f.k_min:.2f}, {f.k_max:.2f}]",
                _fmt(f.rmse_vol_points, 3),
                _fmt(f.max_abs_vol_points, 3),
                _fmt(t["rmse_vol_points"], 3) if t else "n/a",
                _fmt(t["max_abs_vol_points"], 3) if t else "n/a",
            ]
        )
    parts.append(
        '<div class="card"><strong>Fit error (vol points, 1 point = 1% IV)</strong>'
        + _table(
            ["T", "quotes", "k range", "RMSE vs mkt", "max vs mkt", "RMSE vs true", "max vs true"],
            rows,
        )
        + "</div>"
    )
    if report is None:
        parts.append('<div class="card"><strong>Arbitrage:</strong> not checked.</div>')
    else:
        vs = _all_violations(report)
        n_in = sum(1 for v in vs if v.in_quoted_range)
        if not vs:
            head = '<span class="pass">PASS</span> no butterfly or calendar arbitrage found'
        elif report.passed_in_quoted_range:
            head = (
                f'<span class="fail">FAIL</span> {len(vs)} violation(s), all in the extrapolated '
                "wings (none inside the quoted range)"
            )
        else:
            head = (
                f'<span class="fail">FAIL</span> {len(vs)} violation(s), {n_in} inside the '
                "quoted range"
            )
        grid = (
            f"checked on {report.n_grid} points over k ∈ "
            f"[{report.k_grid_min:g}, {report.k_grid_max:g}]"
        )
        body = (
            f"<strong>Arbitrage:</strong> {head} <span class='muted'>({html.escape(grid)})</span>"
        )
        if vs:
            body += (
                "<ul>"
                + "".join(f"<li>{html.escape(_violation_text(v))}</li>" for v in vs)
                + "</ul>"
            )
        if omitted:
            body += (
                "<p class='muted'>Not drawn in the 3D view (entirely outside the plotted k "
                f"range): {len(omitted)} violation(s), listed above.</p>"
            )
        parts.append(f'<div class="card">{body}</div>')
    parts.append(
        "<p class='muted'>Note: SVI approximates the Heston smile; it is not the Heston model. "
        "A small residual against the true Heston vol is expected, and the SVI wings beyond "
        "the quoted strikes are extrapolation.</p>"
    )
    return "\n".join(parts)


def write_report_html(
    path: str | Path,
    fits: Sequence[SliceFit],
    market: MarketData | None = None,
    report: ArbitrageReport | None = None,
    *,
    include_plotlyjs: str = "inline",
    title: str = "Volatility surface report",
) -> Path:
    """Write the one-page report; returns the path.

    ``include_plotlyjs="inline"`` embeds plotly.js (works offline, ~5 MB); ``"cdn"`` links it.
    plotly.js is emitted only with the first figure.
    """
    if include_plotlyjs not in ("inline", "cdn"):
        raise ValueError("include_plotlyjs must be 'inline' or 'cdn'")
    js: Any = True if include_plotlyjs == "inline" else "cdn"
    fs = _sorted_fits(fits)
    lo, hi = _quoted_k_range(fs)
    omitted = [
        v for v in _all_violations(report) if v.k_end < lo - _K_PAD or v.k_start > hi + _K_PAD
    ]
    cfg = {"responsive": True, "displaylogo": False}
    fig3d = build_surface_figure(fs, market, report)
    figs = build_smile_figure(fs, market, report)
    div3d = fig3d.to_html(full_html=False, include_plotlyjs=js, config=cfg)
    divs = figs.to_html(full_html=False, include_plotlyjs=False, config=cfg)
    t = html.escape(title)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title><style>{_CSS}</style></head>
<body><main>
<h1>{t}</h1>
{_summary_html(fs, market, report, omitted)}
<h2>Implied-vol surface (drag to rotate)</h2>
<div class="plot">{div3d}</div>
<h2>Smiles by expiry</h2>
<div class="plot">{divs}</div>
</main></body></html>
"""
    out = Path(path)
    out.write_text(page, encoding="utf-8")
    return out
