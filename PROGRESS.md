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
