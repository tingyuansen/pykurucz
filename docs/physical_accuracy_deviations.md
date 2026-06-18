# Physical Accuracy Deviations from Fortran Parity

This document tracks known cases where the **Fortran-parity version** of pykurucz intentionally
reproduces a Fortran behavior that is physically incomplete or numerically accidental. Each entry
describes what Fortran does, why it is likely unintentional, and what a physically correct
implementation would do differently.

Once Fortran parity is confirmed across all validation cases, these items should be revisited in
priority order to produce a more physically accurate spectrum.

---

## 1. Opacity line far-wing cutoff (CUTOFF=NaN in tfort.93)

**Status**: Matched in Python (cutoff default changed from `1e-7` to `0.0`)

### What Fortran does
`synthe.for` reads `CUTOFF` from `tfort.93` at runtime:
```fortran
READ(93) NLINES,LENGTH,IFVAC,IFNLTE,N19,TURBV,DECKJ,IFPRED,
1        WLBEG,WLEND,RESOLU,RATIO,RATIOLG,CUTOFF,LINOUT
```
The specific binary `tfort.93` used in this pipeline stores `CUTOFF=NaN` (IEEE 754 Not-a-Number).

This causes two effects in Fortran's line-wing accumulation loop:
1. **Near-wing early-exit condition** (`IF(PROFILE(NSTEP).LT.KAPMIN) GO TO 323`):
   `NaN < KAPMIN` always evaluates `FALSE`, so the near-wing loop always runs to completion
   through `N10DOP` steps (10 Doppler widths). This part is **correct behavior**.
2. **Far-wing extent** (`MAXSTEP = SQRT(X/KAPMIN) + 1`):
   `X / NaN = NaN`, `SQRT(NaN) = NaN`, `NaN + 1 = NaN`. When `MAXSTEP=NaN` is used as a
   Fortran `DO` loop upper bound, it converts to a very large negative integer (or 0), so the
   far-wing `DO 322` loop **does not execute at all**. No far-wing opacity is accumulated
   beyond N10DOP steps.

### Why it is likely unintentional
`NaN` is not a valid physical cutoff. The `CUTOFF` parameter was designed (see `synthe.for`
comments) to be a small positive number (e.g., `1e-4`) controlling when line wings are
truncated for performance. A `NaN` in `tfort.93` is most likely a corruption or missing
initialization in the binary file generated for this pipeline, not a deliberate choice.

### Impact observed
For **cool stars** (Teff ≤ 3600 K) that have strong resonance lines (e.g., Co I at 312.23 nm,
V III at 312.22 nm), the Lorentzian far-wing tails carry measurable opacity beyond 10 Doppler
widths. When Python used `cutoff=1e-7` (computing far wings), it produced up to **36%** higher
normalized absorption at ~312 nm than Fortran. Warmer cases (Teff ≥ 4100 K) were unaffected
(max error stayed < 5%) because their strong lines are at different wavelengths or have different
damping constants.

### Python implementation (current — Fortran-parity)
```python
# synthe_py/cli.py
parser.add_argument(
    "--cutoff", type=float, default=0.0,
    help="cutoff=0 matches Fortran NaN behavior: near wings run to N10DOP, far wings skipped"
)
```
In `opacity.py`, the far-wing block checks `if x_far > 0 and kapmin > 0`. With `cutoff=0.0`,
`kapmin = 0.0 * continuum = 0.0`, so the condition is `False` and no far wings are computed.

### What a physically correct implementation would do
Use a small positive cutoff (e.g., `1e-4` to `1e-7` depending on desired accuracy) so that
far-wing Lorentzian contributions are included whenever they exceed the threshold relative to the
local continuum. This is more physically complete and is what the Fortran code was **designed**
to do when given a valid `CUTOFF`. The reference Fortran binary `tfort.93` should also be
regenerated with a real `CUTOFF` value.

**Suggested future value**: `cutoff=1e-4` (matches typical Kurucz pipeline usage) or `1e-7`
(maximally accurate, ~10× slower for cool stars with many strong lines).

**Files to change**:
- `synthe_py/cli.py`: change `default=0.0` back to `default=1e-4` (or chosen value)
- Optionally regenerate `tfort.93` with a valid `CUTOFF` for Fortran cross-checks

---

## 2. FREEFF concatenated-field parsing and ISIGN leak in `fortran_iter1.atm`

**Status**: Matched in Python (`_freeff_parse_float` with ISIGN leak + validation uses `fortran_iter1.atm`)

### What Fortran does

`xnfpelsyn.for` reads the `.atm` file with Fortran's `FREEFF` (free-field) parser
(atlas12.for line 2612).  When ATLAS12 writes `fortran_iter1.atm` via FORMAT 554
(`1PE15.8,0PF9.1,1P7E10.3`), a positive field (e.g. P in E10.3 = 9 visible chars)
immediately followed by a negative field (XNE with leading `-`, filling the full 10 chars)
produces a merged token with no separating space:

```
 4.875E-01-1.376E+06    ← true P=4.875e-1, true XNE=-1.376e6
```

FREEFF is a character-by-character state machine.  When it hits the separator `-` in state 400
(reading exponent digits of the first number), it jumps to label 999 which resets ANSWER,
ASIGN, NPT, IF0, and N — **but not ISIGN**.  ISIGN was set by the first number's exponent
sign (label 304 for `-`, label 303 for `+`).  The separator `-` is consumed; ASIGN resets to
+1.  The second number starts parsing from the digit after the separator.

When the second number's exponent sign goes through label 303 (`+`), ISIGN is not touched and
the leaked value from the first number persists.  This produces an **ISIGN-leak bug**:

| First exp sign | Second exp sign | Leaked ISIGN | Effect on result |
|:-:|:-:|:-:|:--|
| `E+` | `E+` | +1 (init) | Correct |
| `E+` | `E-` | −1 (set by 304) | Correct |
| **`E-`** | **`E+`** | **−1 (leaked!)** | **Exponent negated** |
| `E-` | `E-` | −1 (set by 304) | Correct |

Examples:

```
'1.933E+00-1.697E+06'  →  +1.697e+06  (E+00: ISIGN=+1, no leak)
'4.875E-01-1.376E+06'  →  +1.376e-06  (E-01: ISIGN=−1 leaks, E+06 → E−06)
'8.740E-01-1.021E+07'  →  +1.021e-07  (E-01: ISIGN=−1 leaks, E+07 → E−07)
```

The column shift also applies: FREEFF returns one value per call for the merged token, so the
*next* FREEFF call reads what was intended as ABROSS as XNE, and so on.

### Why it is likely unintentional

The ATLAS12 `.atm` writer was not designed for this collision — it simply does not enforce
a minimum field width between small positive values followed by large negative ones.  The
FREEFF parser then silently misreads the concatenated token.  The ISIGN leak is an
additional bug: the 999-reset path clears most state variables but forgets ISIGN, causing
the second number's exponent to be negated when the first had a negative exponent.

### Impact observed

**Without ISIGN leak replication**, `_freeff_parse_float` returned `+1.376e+06` instead of
`+1.376e-06` for `t04350` depth 0 — a factor of **10¹²** error.  This inflated XNTOT,
producing wildly wrong Na I / K I populations and `max_norm_abs ~ 0.94` at the Na D and K I
doublets.

Cases confirmed affected: `t04350_g+1.70_mh-0.30`, `t04650_g+2.60_mh+0.50` (first exponent
is `E-01`, triggering the leak).  Cases with `E+00` in the first number (e.g., `t04300`,
`t04600`) are unaffected by the leak because ISIGN stays at its initial +1.

### Python implementation (current — Fortran-parity)

1. **`_freeff_parse_float`** in `synthe_py/tools/convert_atm_to_npz.py`: detects merged
   tokens and replicates the ISIGN leak.  When the first number has a negative exponent
   (`E-NN`) and the second has a positive exponent (`E+MM`), the effective exponent sign is
   negated to match Fortran's FREEFF behavior.

2. **Validation uses `fortran_iter1.atm` for both pipelines** (`atlas_py/tools/validate_synthe_e2e.py`,
   `_resolve_case_atm_paths`): the Python SYNTHE run uses the same `fortran_iter1.atm` as
   the Fortran run so `convert_atm_to_npz` encounters the same concatenated tokens.

3. **No pressure unit-conversion heuristic** in `convert_atm_to_npz.py`: the Python code
   previously had a heuristic that compared P to G*RHOX and, when the ratio exceeded 1e5,
   multiplied P by 1e6 (bar→dyn/cm² conversion).  This Python-only logic has no Fortran
   counterpart (`xnfpelsyn.for` line 2092: `P(J)=FREEFF(CARD)` — raw value, no conversion).
   The heuristic silently corrupted the ISIGN-leaked tiny pressure values (e.g., 1.376e-6 →
   1.376) and was removed.

### What a physically correct implementation would do

Use the true physical gas pressure from `python_iter1.atm` (space-separated, correctly
parsed).  This requires simultaneously fixing NMOLEC so it converges to the correct electron
density when given a tiny surface-layer pressure.  Until that root-cause NMOLEC fix is in
place, using `fortran_iter1.atm` is the correct Fortran-parity approach.

**Files involved**:
- `synthe_py/tools/convert_atm_to_npz.py`: `_freeff_parse_float`, pressure assignment
- `atlas_py/tools/validate_synthe_e2e.py`: `_resolve_case_atm_paths`

---

## 3. NMOLEC electron-density seed: `xntot / 2.0`

**Status**: Matched in Python (seed changed from `0` to `xntot / 2.0`)

### What Fortran does

Fortran's `NMOLEC` subroutine initialises the electron density iterate `XN(1)` (the first
unknown in the Newton system) to approximately half the total particle number density.  This
puts the initial guess in the middle of the physically plausible range `[0, xntot]` and
ensures the Newton solver converges to the ionized branch for hot/intermediate layers.

### Why it matters

With a seed of `0` (or any value near zero), the Newton solver can converge to the neutral
branch for intermediate-temperature layers, yielding electron densities that are orders of
magnitude too low.  This in turn causes wildly wrong line populations (e.g., K I at 766–770 nm
was ~6–16 orders of magnitude too large or too small depending on the layer, resulting in the
K I doublet discrepancy).

### Impact observed

The K I doublet (~766.7 nm, ~770.1 nm) showed catastrophic errors (`max_norm_abs >> 0.1`)
before this fix.  After changing the seed to `xntot / 2.0` in `nmolec_exact.py` (line 1906),
the doublet error dropped to < 1% across all tested cases.

### Python implementation (current — Fortran-parity)

```python
# synthe_py/tools/nmolec_exact.py  line 1906
xne_j = xntot_j / 2.0   # seed: half total particle density, matching Fortran NMOLEC init
```

### What a physically correct implementation would do

A more robust seed is `xne_prev` (electron density from the previous depth layer), which is
what many equilibrium solvers use for a depth-by-depth sweep.  The `xntot / 2.0` seed is
faithful to Fortran and sufficient for current validation purposes.

**Files involved**:
- `synthe_py/tools/nmolec_exact.py`: line 1906

---

## Template for future entries

```
## N. Short description

**Status**: [Matched in Python | Not yet matched | Under investigation]

### What Fortran does
...

### Why it is likely unintentional / physically incomplete
...

### Impact observed
...

### Python implementation (current — Fortran-parity)
...

### What a physically correct implementation would do
...
```
