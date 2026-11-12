# Progress ledger

Status labels: **VERIFIED** (implemented and checked by a test or measurement recorded
here), **IMPLEMENTED** (code exists, not independently verified), **BLOCKED** (needs
something outside this environment), **FAILED** (attempted, does not work).

Environment for all measurements unless stated otherwise: cloud container,
Linux x86_64, Intel Xeon @ 2.30 GHz, 4 vCPU, Python 3.12.3, numpy 2.5.4, scipy 1.18.1.
(Not the author's Apple Silicon laptop; `benchmarks/bench.py` reproduces the numbers there.)

| # | Milestone | Status | Evidence |
|---|-----------|--------|----------|
| 0 | Scaffold: src layout, `pyproject.toml`, `vsl` entry point, CI (ruff + pytest on 3.12/3.13), `.gitignore` | IMPLEMENTED | CI result recorded below once the first run finishes |
| 1 | Black–Scholes core in forward / log-moneyness coordinates | VERIFIED | `tests/test_black_scholes.py` |
| 2 | Implied-vol solver (Newton on vega + Brent fallback, bound checks) | VERIFIED | `tests/test_implied_vol.py`, 2,000-example hypothesis round trip |

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
