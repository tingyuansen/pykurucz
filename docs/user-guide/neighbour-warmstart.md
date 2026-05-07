# Neighbour-warmstart workflow (`--warmstart-atm`)

!!! note "When you need this"
    The kurucz-a1 emulator (the default `[1/3]` step) gives an atmospheric
    starting guess that is good enough for the great majority of the
    parameter space.  In a narrow corner — cool ($T_{\rm eff} \lesssim 4500$ K)
    low-gravity ($\log g = 0$) atmospheres with non-zero α-element
    perturbations and small $\Delta\mathrm{[C/H]}$ — the emulator's prior
    is far enough from the true converged solution that ATLAS iterates
    into all-NaN before recovering.  Symptom: the run fails with a
    `RuntimeError: ATLAS atmosphere degenerate ...` (the Fix 13 guard) or
    `ATLAS atmosphere went all-NaN at iteration N for K consecutive
    iterations` (Fix 14).  See [`PYKURUCZ_FIXES.md`][fixes] Issue 15 for
    the full diagnosis.

## What the flag does

`pykurucz.py --warmstart-atm /path/to/donor.atm` skips the kurucz-a1
emulator step entirely.  Instead it takes the supplied `.atm` file's
**layer structure** ($T(\tau)$, $P(\tau)$, $\rho_x(\tau)$, $X_{\rm Ne}(\tau)$,
$\kappa_{\rm Ross}(\tau)$, $a_{\rm rad}(\tau)$, $v_{\rm turb}$,
$F_{\rm cnv}$, $v_{\rm cnv}$ per layer) and rewrites the header
($T_{\rm eff}$, $\log g$, the full ABUNDANCE block) using the target's
`--teff / --logg / --mh / --am / --abund` arguments.

This means atlas_py iterates from a near-converged starting state for
the *target* cell, even though the donor was a different cell.  It
typically converges in 5–10 ATLAS iterations rather than 25–30.

## When to use it

- After getting the `Fix 13`/`Fix 14` failure modes from a from-emulator
  run.
- When the emulator's `[alpha/M]` warning shows the requested cell is
  outside the training range: `[alpha/M]=-0.30 outside training range
  [-0.2, +0.6]`.
- When you want fast turnaround on a cell whose neighbour you've
  already converged.

## Picking a donor

The donor `.atm` should be a converged cell at the closest available
$(T_{\rm eff}, \Delta\mathrm{[C/H]}, \Delta\mathrm{[N/H]}, [\alpha/{\rm Fe}])$
to your target.  Same galaxy/host metallicity is preferred but not
strictly required — the donor's layer structure is just an initial
guess; ATLAS will re-converge against the target's chemistry.

A simple L2-distance picker (treating $T_{\rm eff}$/100 as units that
balance the dex-scaled chemistry axes) works well in practice.

## Example

```bash
pykurucz.py \
    --teff 4300 --logg 0.0 --mh -0.5 --am -0.3 --vturb 2.0 \
    --abund C:-1.10 --abund N:-0.20 \
    --wl-start 600 --wl-end 900 --resolution 50000 \
    --warmstart-atm /path/to/LMC_t4300_dCm050_dNp030/atm/t04300g0.00_mh-0.50_am-0.30.atm \
    --output-dir /path/to/output
```

The output `[1/3]` line in stdout will read

```
[1/3] Using neighbour donor .atm for layer structure, target chemistry: /path/to/...
```

confirming the emulator was skipped and your supplied file was used.
The remaining stages (`[2/3]` atlas_py, `[3/3]` synthe_py) proceed
normally.

## Caveats

- `pykurucz.py` does **not** ship with a candidate-selector.  You
  provide the donor path; pykurucz uses it.  An external selector
  (looking up your local grid for the closest match) is a few-line
  Python script.
- The donor's `.atm` must be a real converged file (not a
  `*_warmstart.atm`).  pykurucz parses the `READ DECK6` block, so any
  Kurucz-format atmosphere with that structure works.
- This is a workflow escape hatch, not a numerical fix to the emulator
  or to ATLAS.  The right long-term solution is to broaden the
  emulator's training distribution (Issue 15).

[fixes]: https://github.com/...your-fork.../PYKURUCZ_FIXES.md
