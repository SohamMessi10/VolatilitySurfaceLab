# Progress ledger

Status labels: **VERIFIED** (implemented and checked by a test or measurement recorded
here), **IMPLEMENTED** (code exists, not independently verified), **BLOCKED** (needs
something outside this environment), **FAILED** (attempted, does not work).

Environment for all measurements unless stated otherwise: cloud container,
Linux x86_64, Intel Xeon @ 2.30 GHz, 4 vCPU, Python 3.12.3, numpy 2.5.4, scipy 1.18.1.
(Not the author's Apple Silicon laptop; `benchmarks/bench.py` reproduces the numbers there.)

| # | Milestone | Status | Evidence |
|---|-----------|--------|----------|
| 0 | Scaffold: src layout, `pyproject.toml`, CI (ruff + pytest on 3.12/3.13), `.gitignore` | VERIFIED | CI run #1 on `main` green on 3.12 and 3.13 (`c93f3bc`) |
| 1 | Black–Scholes core in forward / log-moneyness coordinates | VERIFIED | `tests/test_black_scholes.py` |
| 2 | Implied-vol solver (Newton on vega + Brent fallback, bound checks) | VERIFIED | `tests/test_implied_vol.py`, 2,000-example hypothesis round trip |
| 3 | Heston pricer: little-trap CF, Carr–Madan FFT, Lewis direct integration | VERIFIED | `tests/test_heston.py` (48 tests) |
| 4 | Synthetic market: Heston grid → seeded bid/ask noise → IV table (JSON/CSV) | VERIFIED | `tests/test_market.py` |
| 5 | Raw SVI per-expiry calibration (constraints + Lee wing bound exact, 18-start least squares) | VERIFIED | `tests/test_svi.py`, `tests/test_pipeline.py` |
| 6 | Static-arbitrage checks: Durrleman butterfly, calendar total variance | VERIFIED | `tests/test_arbitrage.py`, `tests/test_pipeline.py` |
| 7 | CLI `vsl` (generate / calibrate / check / plot / run) + interactive HTML report | VERIFIED | `tests/test_cli.py`, `tests/test_plotting.py`, rendered in headless Chromium at 1280 px and 390 px |
| 7b | GitHub Pages demo from `/docs` | BLOCKED | Pages files committed; enabling Pages needs a repo-settings change this session cannot make (see README) |
| 8 | `vsl fetch` (real SPY chain via yfinance, optional extra) | IMPLEMENTED | Offline tests with a fake chain (`tests/test_fetch.py`); live call BLOCKED here: the environment's proxy refuses Yahoo hosts (HTTP 403) |

## Log

### M1 — Black–Scholes core
- Prices are computed as normalized OTM time value `c(k, s)` with `k = ln(K/F)`,
  `s = σ√T`; ITM prices come from parity, so deep-OTM time value is not lost to
  cancellation against intrinsic.
- Verified: Hull reference values (C = 10.450583572, P = 5.573526022), vega vs central
  finite difference (rel 1e-8), put–call parity residual < 1e-11 on 51×3×3 grid, prices
  inside model-free bounds, σ→0 / σ→∞ limits, strike monotonicity/convexity,
  `c(k,s) = e^k p(−k,s)` symmetry (hypothesis).

### M2 — Implied-vol solver
- Newton started at the inflection point `s* = √(2|k|)`; falls back to `brentq` when
  normalized vega < 1e-8 or a Newton step leaves the bracket. Prices on or outside the
  no-arbitrage bounds raise `PriceOutOfBoundsError` with the bounds in the message.
- Measured with `python benchmarks/bench.py` (20,000 random cases, seed 0: k ∈ [−3, 3],
  σ ∈ [1%, 300%], T ∈ [1 day, 10 y], F ∈ [1, 1000], df ∈ [0.5, 1], call/put):
  max relative vol error 7.3e-10 (median 3.9e-16) wherever the time value is resolvable
  (> 1e-8 of the price); 87.6% solved by Newton, 12.4% by the Brent fallback; 375 cases
  rejected, all with time value below double-precision resolution; 10.7 µs per inversion
  (scalar `math` fast path; ~237 µs per inversion with the numpy-scalar version it replaced).
- Known limitation: far in the wings with tiny total vol (|d1|/s ~ 1e5) the direct
  Black formula loses digits to cancellation (relative error ≈ ε·|d1|³/s). This only
  affects normalized prices around 1e-90 and below; the test tolerance follows that
  condition number instead of hiding it.

### M3 — Heston pricer
- CF: Albrecher et al. (2007) "little trap" form, with every `1/ξ²` removed algebraically
  (`β − d = −ξ²(iu+u²)/(β+d)`) and Kahan's `log(w)/(w−1)` for `log1p(z)/z`
  (numpy's complex `log1p` measured at 6e-8 relative error for |z| ~ 1e-9). Result: exact
  Black–Scholes at ξ = 0, no cancellation for small ξ.
- Verified against an independent reference, numerical integration of the Riccati ODEs
  (DOP853, rtol 1e-12): |Δφ| < 1e-12, including T = 10 where the original Heston (1993)
  formula is off by O(1) (branch-cut jump; asserted in `test_little_trap_avoids_original_branch_cut`).
- Carr–Madan FFT, N = 2^15, η = 0.1 (strike spacing 0.0019 in k), Simpson weights,
  cubic-spline interpolation between nodes. Calls use damping +α for k ≥ 0; puts use the
  mirrored −1−α for k < 0 (same pole distances ⇒ same Simpson accuracy; −α instead measured
  a 3e-8 bias). α is capped per expiry at half the admissible range from the critical
  moments (Andersen–Piterbarg explosion time, verified against the real Riccati ODE to
  rel 1e-5); η shrinks with α to keep η/pole-distance fixed. Unusable damping raises and
  points to `method="lewis"` instead of silently mispricing (found while testing: α = 3
  at T = 3 gave a 0.068 error with no NaN).
- FFT vs Lewis (`quad` on [0, ∞)) on k ∈ [−1.5, 1], 26 strikes: max |Δ| / F ≤ 1.5e-11 for
  the equity parameter set at T ∈ {0.05, 0.25, 1, 3, 10} and ≤ 9e-12 for a ρ > 0 set
  (T ≤ 3); test threshold 5e-11.
- Black–Scholes limit (v0 = θ): ξ = 0 matches BS to < 2e-11 (FFT) / < 1e-14 (Lewis); error
  shrinks O(ξ) for ρ = −0.7 and O(ξ²) for ρ = 0, as theory predicts.
- Put–call parity between the *independent* call- and put-side FFTs: |gap| < 1e-11 on
  |k| < 1.5. Prices strictly monotone in strike above the FFT accuracy floor
  (5e-11·df·F), butterflies ≥ −1e-9, slopes within [−df, 0].
- ~17 ms per expiry for both sides (N = 2^15).

### M4 — Synthetic market
- Default: S = 100, r = 3%, q = 1%, expiries {0.1, 0.25, 0.5, 1, 1.5, 2} y, strikes
  60–150% of spot (step 5%), OTM option per strike, Heston (v0 0.04, κ 2, θ 0.05, ξ 0.6,
  ρ −0.7). 114 grid points → 84 quotes kept, 30 dropped (true price < 0.05, deep wings of
  short expiries). Generation 0.13 s.
- Noise: half-spread max(tick, 2% of price), mid = true + noise·h·U(−1,1) (seeded), bid/ask
  rounded outward to the tick. A simple perturbation, not a microstructure model.
- Verified: noise = 0 ⇒ iv_mid = iv_true (1e-12); FFT-based iv_true vs Lewis-based IV max
  |Δ| = 3.3e-9 over all 84 quotes; seed reproducibility; bid ≤ mid ≤ ask; exact JSON/CSV
  round trips (NaN → null in JSON).

### M5 — SVI calibration
- `least_squares` (TRF) on total variance from 18 deterministic starts. Constraints are
  enforced *exactly* by reparametrization: a = c − bσ√(1−ρ²) with c ≥ 0 (min variance ≥ 0),
  and b = 2β/(1+|ρ|) with β ∈ [0, 1] (Lee's wing bound b(1+|ρ|) ≤ 2), plus box bounds
  on ρ and σ.
- Why the Lee bound was added: without it, a noisy 6-quote slice (T = 0.1) extrapolated to
  wings with g(k) = −7.8 and created butterfly and calendar arbitrage outside the quotes.
- Fit error on the clean default market (SVI vs market = vs true Heston IV, since
  noise = 0), max abs in vol points: T=0.1 0.047, 0.25 0.079, 0.5 0.082, 1.0 0.012,
  1.5 0.002, 2.0 0.002. SVI *approximates* Heston; it does not recover it exactly (the error
  is largest at short/intermediate maturities, consistent with SVI being the large-maturity
  limit of Heston, Gatheral–Jacquier 2011).
- Exact SVI round trip (data generated by SVI): ~5e-15 vol points; hypothesis recovery
  over random valid parameters < 1e-8 vol points.

### M6 — Arbitrage checks
- Butterfly: Durrleman g(k) ≥ 0 on a 3,001-point grid over k ∈ [−1.5, 1.5]; calendar:
  w(k, T_{i+1}) ≥ w(k, T_i) on the same grid. Violations report the expiry (pair), the
  contiguous k-interval, the worst point/value, and whether they overlap the quoted range.
- Verified: clean Heston surface passes everywhere; Vogt's smile detected on
  k ∈ [0.643, 1.256] (worst g = −0.0329 at k = 0.879) both as raw parameters and end-to-end
  from quotes; injected calendar inversion (T = 1.5 variance 2% below T = 1.0) detected and
  located over the whole quoted range.
- Measured on 20 noisy markets (noise = 1, seeds 0–19): 20/20 free of arbitrage inside the
  quoted ranges, but only 2/20 free on the full extrapolated grid. Independent per-slice SVI
  fits do not guarantee arbitrage-free extrapolation; a joint (SSVI-type) calibration is the
  standard fix and is listed as future work.

### M7 — CLI and report
- `vsl generate | calibrate | check | plot | run | fetch`; exit codes 0 ok, 1 arbitrage,
  2 usage, 3 input error, 4 numerical failure; expected errors print one line, no traceback.
- Report: summary, per-slice fit error (vs market and vs true Heston IV), arbitrage list with
  in/out-of-quoted-range flags, 3D SVI surface + market points + flags, smile panels with
  bid/ask IV error bars. Surface between expiries interpolates total variance linearly in T
  (keeps calendar monotonicity). Inline plotly.js 4.9 MB (offline); `--plotlyjs cdn` 120 kB.
- Rendered in headless Chromium (SwiftShader WebGL) at 1280 px and 390 px: no console errors,
  no horizontal page scroll (wide tables scroll in their own container).
- Found while building the demo: `vsl plot -o docs/x.html` failed when `docs/` did not exist;
  outputs now create their parent directories.
- Demo pages: `docs/index.html` (noise 1, seed 3: one calendar violation, purely in the
  extrapolated wings) and `docs/clean.html` (noise-free: passes). They load plotly.js 4.1.1
  from cdn.plot.ly (version confirmed on npm; the CDN itself is not reachable from this
  container, so the live page render must be checked once Pages is enabled).

### M8 — Optional live fetch
- Verified against the installed yfinance 1.8.0 source: `Ticker(sym).options`,
  `option_chain(date) -> (calls, puts, underlying)`, column names, `fast_info["lastPrice"]`.
- Forward and discount factor per expiry are implied by put–call parity (regression of
  C − P on K near the money), so no dividend/rate guess is needed; fallback to a flat rate
  is reported per expiry.
- Yahoo's terms: personal use only. Output defaults to `data/` (git-ignored); nothing fetched
  is committed. A blocked/offline network now maps to exit 3 with a one-line message (found
  by running it here: yfinance raised an internal `TypeError`).
