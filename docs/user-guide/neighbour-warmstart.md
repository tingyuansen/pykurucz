# Neighbour-warmstart workflow (`--warmstart-atm`)

!!! note "What this is"
    A bridge between the two main workflows:
    [from stellar parameters](from-parameters.md) (which uses the
    kurucz-a1 emulator to seed ATLAS) and
    [from an existing atmosphere](from-atmosphere.md) (which skips
    ATLAS entirely and treats the supplied `.atm` as fixed).

    With `--warmstart-atm` you supply a pre-computed `.atm` file as
    pykurucz's *initial guess* for the target's chemistry, then let
    ATLAS iterate it to convergence against the requested `--mh`,
    `--am`, and `--abund` arguments.  The atmosphere is **not** frozen
    (unlike pure from-atmosphere mode), and the emulator is **not**
    consulted (unlike pure from-parameters mode).

## When you need this

The kurucz-a1 emulator covers the great majority of the parameter
space, but like any neural emulator its accuracy degrades near the
edges of its training distribution.  In those regions the prior can be
far enough from the true converged solution that ATLAS iterates into
all-NaN before recovering.

This is a **known coverage limitation** of the trained emulator, not a
bug in the solver.  The intended long-term fix is to broaden the
emulator's training distribution so the failing region is covered
natively; in the meantime, supplying a converged neighbour cell as the
warmstart bypasses the bad prior and lets you proceed.

The defensive guards on the ATLAS side detect the failure cleanly
(rather than silently writing a degenerate atmosphere), so you get a
clear error rather than a corrupt result.  Typical symptoms that mean
"try `--warmstart-atm`":

- ATLAS aborts with a NaN-streak or degenerate-atmosphere error on a
  from-parameters run.
- The emulator emits an out-of-range warning, e.g.
  `[alpha/M]=-0.30 outside training range [-0.2, +0.6]`.

It is also a useful speedup if you already have a converged neighbour
on disk — convergence typically takes 5–10 ATLAS iterations rather
than 25–30.

## What the flag does

`pykurucz.py --warmstart-atm /path/to/donor.atm` skips the kurucz-a1
emulator step entirely.  It takes the supplied `.atm` file's **layer
structure** ($T(\tau)$, $P(\tau)$, $\rho_x(\tau)$, $X_{\rm Ne}(\tau)$,
$\kappa_{\rm Ross}(\tau)$, $a_{\rm rad}(\tau)$, $v_{\rm turb}$,
$F_{\rm cnv}$, $v_{\rm cnv}$ per layer) and rewrites the header
($T_{\rm eff}$, $\log g$, the full ABUNDANCE block) using the target's
`--teff / --logg / --mh / --am / --abund` arguments.

ATLAS then iterates this rewritten state to convergence for the
*target* cell.

## Picking a donor

The donor `.atm` should be a converged cell at the closest available
$(T_{\rm eff}, \log g, [\mathrm{M/H}], [\alpha/\mathrm{M}],$ any other
abundance dimensions you vary$)$ to your target.  The donor's layer
structure is just an initial guess; ATLAS re-converges against the
target's chemistry, so an exact chemistry match is not required.

A simple L2-distance picker (treating $T_{\rm eff}$/100 as units that
balance the dex-scaled chemistry axes) works well in practice.

## Example

```bash
pykurucz.py \
    --teff 4300 --logg 0.0 --mh -0.5 --am -0.3 --vturb 2.0 \
    --abund C:-1.10 --abund N:-0.20 \
    --wl-start 600 --wl-end 900 --resolution 50000 \
    --warmstart-atm /path/to/donor_grid/t04300g0.00_mh-0.50_am-0.30.atm \
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
- This is a workflow escape hatch, not a numerical fix.  The proper
  long-term solution is to broaden the emulator's training
  distribution so the failing region is covered natively.
