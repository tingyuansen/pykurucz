---
title: "pykurucz — Synthetic Stellar Spectra"
description: "Pure-Python ATLAS12 & SYNTHE — generate synthetic stellar spectra from fundamental parameters with sub-0.1% Fortran-validated accuracy."
hide:
  - navigation
  - toc
---

<div class="pk-hero" markdown>
<div class="pk-hero__main" markdown>

# Synthesize stellar spectra<br/>with ATLAS12 &amp; SYNTHE,<br/>*in pure Python.*

<div class="pk-hero__sub" markdown>
A faithful, performance-tuned reimplementation of Robert Kurucz's
**ATLAS12** stellar atmosphere code and **SYNTHE** spectrum synthesis
code, in pure Python. Go from $T_{\rm eff}$, $\log g$, and either a
**bulk metallicity / α-enhancement** ($\rm[M/H]$, $\rm[\alpha/M]$) or
an **arbitrary per-element abundance pattern**, to a self-consistent
300–1800 nm spectrum — atmosphere included.
</div>

<div class="pk-hero__cta" markdown>
[Get started](getting-started/installation.md){ .md-button .md-button--primary }
[User guide](user-guide/overview.md){ .md-button }
[Python reference](api-reference/pykurucz.md){ .md-button }
[Paper](https://arxiv.org/abs/2603.11693){ .md-button }
</div>

<div class="pk-hero__meta" markdown>
<span><b>Python</b> 3.10+</span><span class="dot"></span>
<span><b>License</b> MIT</span><span class="dot"></span>
<span><b>Validated</b> sub-0.1% vs. Fortran</span><span class="dot"></span>
<span><b>Range</b> 300–1800 nm</span>
</div>

<div class="pk-byline" markdown>
By **Elliot M. Kim** (Cornell) &amp;
**[Yuan-Sen Ting](https://ysting.space/)** (OSU) ·
honoring the legacy of **Robert L. Kurucz** (1944–2025)
</div>

</div>

<aside class="pk-hero__aside" aria-label="Interactive ASCII portrait of Robert L. Kurucz">
  <canvas id="pk-bob-canvas" class="pk-bob" aria-label="Interactive ASCII portrait of Robert L. Kurucz; hover to perturb the characters"></canvas>
  <figcaption class="pk-bob__caption">
    Bob Kurucz · hover to interact
  </figcaption>
</aside>

<svg class="pk-spectrum" viewBox="0 0 800 80" preserveAspectRatio="none" aria-hidden="true">
  <defs>
    <linearGradient id="pk-fade" x1="0" x2="0" y1="0" y2="1">
      <stop offset="0%" stop-color="currentColor" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="currentColor" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <path
    d="M4,18 L8,18 L12,18 L16,17 L20,16 L24,18 L28,28 L32,40 L36,52 L40,52 L44,42 L48,30 L52,20 L56,17 L60,16 L64,17 L68,18 L72,18 L76,17 L80,16 L84,14 L88,13 L92,15 L96,22 L100,30 L104,38 L108,42 L112,40 L116,32 L120,22 L124,16 L128,14 L132,13 L136,14 L140,16 L144,18 L148,19 L152,18 L156,16 L160,14 L164,13 L168,15 L172,20 L176,28 L180,38 L184,46 L188,48 L192,46 L196,40 L200,30 L204,22 L208,17 L212,15 L216,14 L220,15 L224,16 L228,18 L232,20 L236,21 L240,22 L244,20 L248,18 L252,16 L256,14 L260,13 L264,12 L268,11 L272,12 L276,16 L280,22 L284,30 L288,36 L292,38 L296,36 L300,30 L304,22 L308,16 L312,13 L316,12 L320,13 L324,14 L328,15 L332,16 L336,16 L340,15 L344,14 L348,13 L352,13 L356,14 L360,16 L364,20 L368,26 L372,32 L376,36 L380,36 L384,32 L388,26 L392,20 L396,16 L400,13 L404,12 L408,12 L412,13 L416,15 L420,17 L424,20 L428,23 L432,26 L436,28 L440,28 L444,26 L448,22 L452,18 L456,15 L460,13 L464,12 L468,12 L472,13 L476,15 L480,18 L484,20 L488,21 L492,21 L496,20 L500,18 L504,16 L508,14 L512,13 L516,13 L520,14 L524,16 L528,20 L532,26 L536,32 L540,38 L544,40 L548,38 L552,32 L556,26 L560,20 L564,16 L568,14 L572,13 L576,13 L580,14 L584,16 L588,18 L592,20 L596,22 L600,23 L604,22 L608,20 L612,18 L616,16 L620,14 L624,13 L628,13 L632,14 L636,16 L640,18 L644,20 L648,21 L652,22 L656,21 L660,20 L664,18 L668,16 L672,15 L676,14 L680,14 L684,15 L688,17 L692,21 L696,26 L700,30 L704,32 L708,30 L712,26 L716,22 L720,18 L724,15 L728,14 L732,13 L736,14 L740,16 L744,18 L748,20 L752,21 L756,22 L760,22 L764,21 L768,20 L772,18 L776,17 L780,16 L784,16 L788,16 L792,17 L796,17 L796,76 L4,76 Z"
    fill="url(#pk-fade)" stroke="none"/>
  <path
    d="M4,18 L8,18 L12,18 L16,17 L20,16 L24,18 L28,28 L32,40 L36,52 L40,52 L44,42 L48,30 L52,20 L56,17 L60,16 L64,17 L68,18 L72,18 L76,17 L80,16 L84,14 L88,13 L92,15 L96,22 L100,30 L104,38 L108,42 L112,40 L116,32 L120,22 L124,16 L128,14 L132,13 L136,14 L140,16 L144,18 L148,19 L152,18 L156,16 L160,14 L164,13 L168,15 L172,20 L176,28 L180,38 L184,46 L188,48 L192,46 L196,40 L200,30 L204,22 L208,17 L212,15 L216,14 L220,15 L224,16 L228,18 L232,20 L236,21 L240,22 L244,20 L248,18 L252,16 L256,14 L260,13 L264,12 L268,11 L272,12 L276,16 L280,22 L284,30 L288,36 L292,38 L296,36 L300,30 L304,22 L308,16 L312,13 L316,12 L320,13 L324,14 L328,15 L332,16 L336,16 L340,15 L344,14 L348,13 L352,13 L356,14 L360,16 L364,20 L368,26 L372,32 L376,36 L380,36 L384,32 L388,26 L392,20 L396,16 L400,13 L404,12 L408,12 L412,13 L416,15 L420,17 L424,20 L428,23 L432,26 L436,28 L440,28 L444,26 L448,22 L452,18 L456,15 L460,13 L464,12 L468,12 L472,13 L476,15 L480,18 L484,20 L488,21 L492,21 L496,20 L500,18 L504,16 L508,14 L512,13 L516,13 L520,14 L524,16 L528,20 L532,26 L536,32 L540,38 L544,40 L548,38 L552,32 L556,26 L560,20 L564,16 L568,14 L572,13 L576,13 L580,14 L584,16 L588,18 L592,20 L596,22 L600,23 L604,22 L608,20 L612,18 L616,16 L620,14 L624,13 L628,13 L632,14 L636,16 L640,18 L644,20 L648,21 L652,22 L656,21 L660,20 L664,18 L668,16 L672,15 L676,14 L680,14 L684,15 L688,17 L692,21 L696,26 L700,30 L704,32 L708,30 L712,26 L716,22 L720,18 L724,15 L728,14 L732,13 L736,14 L740,16 L744,18 L748,20 L752,21 L756,22 L760,22 L764,21 L768,20 L772,18 L776,17 L780,16 L784,16 L788,16 L792,17 L796,17"
    fill="none" stroke="currentColor" stroke-width="1.25"
    stroke-linecap="round" stroke-linejoin="round"/>
</svg>
</div>

<div class="pk-section-head">
  <span class="pk-section-head__num">01</span>
  <span class="pk-section-head__title">Quick start</span>
</div>

Give pykurucz $T_{\rm eff}$, $\log g$, and an abundance specification — bulk, per-element, or both — and it returns a self-consistent synthetic spectrum. Two equivalent ways to drive it:

=== "Bulk metallicity (most common)"

    ```python
    import pykurucz

    # Metal-poor α-enhanced K giant (the standard workflow)
    spec_path = pykurucz.synthesize(
        teff=4500, logg=2.0,
        mh=-1.5, am=0.3,
        wl_start=500, wl_end=510,
        resolution=300_000,
    )
    ```

    ```bash
    python pykurucz.py \
        --teff 4500 --logg 2.0 \
        --mh -1.5 --am 0.3 \
        --wl-start 500 --wl-end 510
    ```

=== "Per-element abundances"

    ```python
    import pykurucz

    # CEMP-s star: Fe-poor, C and Ba enhanced
    spec_path = pykurucz.synthesize(
        teff=4800, logg=1.5,
        abundances={26: -2.5, 6: +1.2, 56: +1.0},  # Z → dex offset
        wl_start=400, wl_end=700,
    )
    ```

    ```bash
    python pykurucz.py \
        --teff 4800 --logg 1.5 \
        --abund Fe:-2.5 --abund C:+1.2 --abund Ba:+1.0 \
        --wl-start 400 --wl-end 700
    ```

=== "Plain solar (sanity check)"

    ```python
    import pykurucz
    spec_path = pykurucz.synthesize(
        teff=5770, logg=4.44, mh=0.0, am=0.0,
        wl_start=500, wl_end=510,
    )
    ```

    ```bash
    python pykurucz.py --teff 5770 --logg 4.44 \
        --wl-start 500 --wl-end 510
    ```

In every case the **atmosphere is rebuilt** from the requested abundances and the spectrum is synthesised on top of it — no scaled-solar template assumption. See [Stellar Parameters](user-guide/from-parameters.md#abundances) for the syntax in detail.


<div class="pk-section-head">
  <span class="pk-section-head__num">02</span>
  <span class="pk-section-head__title">Why pykurucz</span>
</div>

<div class="pk-features" markdown>

<div class="pk-feature" markdown>
<span class="pk-feature__tag">01 · DEPLOY</span>
### Pure Python
No Fortran compiler, no opaque binary blobs. NumPy, SciPy, and Numba — install with `pip` and run anywhere CPython runs, including air-gapped HPC nodes.
</div>

<div class="pk-feature" markdown>
<span class="pk-feature__tag">02 · CORRECTNESS</span>
### Fortran-validated
End-to-end parity with Kurucz's original ATLAS12 + SYNTHE pipeline. Sub-0.1% flux differences across 300–1800 nm on the validation grid — every kernel was tested against its Fortran counterpart before being trusted.
</div>

<div class="pk-feature" markdown>
<span class="pk-feature__tag">03 · WARM START</span>
### Neural starting atmosphere
A small PyTorch network predicts the starting atmosphere from
$T_{\rm eff}$, $\log g$, [M/H], [α/M]. The warm start lands inside ATLAS's
**convergence basin**, so `atlas_py` reaches the tolerance in ~10–15 iterations.
A generic grey-atmosphere cold start can stall in a local minimum and
never reach the threshold within the iteration budget.
</div>

<div class="pk-feature" markdown>
<span class="pk-feature__tag">04 · ABUNDANCES</span>
### Bulk and per-element, atmosphere included
Two common modes, one consistent pipeline. Set bulk **`--mh`** and
**`--am`** for the standard scaled-solar / α-enhanced cases, or
override **any individual element** (`--abund Fe:-1.0 --abund C:+0.4 …`)
when you need a peculiar pattern. Either way, the **atmosphere is
rebuilt from scratch** with the matching opacity, so line blanketing
reshapes the temperature structure self-consistently — not just the
spectrum on top of a generic template. Halo dwarfs, α-rich giants,
CEMP-s carbon-rich giants, peculiar Ap stars all work end-to-end from
one command.
</div>

</div>

<div class="pk-section-head">
  <span class="pk-section-head__num">03</span>
  <span class="pk-section-head__title">Documentation</span>
</div>

<div class="pk-docindex" markdown>

<div class="pk-docindex__cell" markdown>
<div class="pk-docindex__head">
  <span class="pk-docindex__num">01</span>
  <span class="pk-docindex__title">Getting started</span>
</div>

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)
- [Downloading data](getting-started/downloading-data.md)
- [Your first spectrum](getting-started/first-spectrum.md)

</div>

<div class="pk-docindex__cell" markdown>
<div class="pk-docindex__head">
  <span class="pk-docindex__num">02</span>
  <span class="pk-docindex__title">User guide</span>
</div>

- [From stellar parameters (with abundances)](user-guide/from-parameters.md)
- [From an existing `.atm` file](user-guide/from-atmosphere.md)
- [The atmosphere emulator](user-guide/emulator.md)
- [CLI reference](user-guide/cli-reference.md)

</div>

<div class="pk-docindex__cell" markdown>
<div class="pk-docindex__head">
  <span class="pk-docindex__num">03</span>
  <span class="pk-docindex__title">How it works</span>
</div>

- [Architecture overview](architecture/overview.md)
- [`atlas_py` — atmospheres](architecture/atlas-py.md)
- [`synthe_py` — synthesis](architecture/synthe-py.md)
- [Fortran parity tests](architecture/fortran-parity.md)

</div>

<div class="pk-docindex__cell" markdown>
<div class="pk-docindex__head">
  <span class="pk-docindex__num">04</span>
  <span class="pk-docindex__title">Physics reference</span>
</div>

- [Radiative transfer](physics/radiative-transfer.md)
- [Opacity](physics/opacity.md)
- [Molecular equilibrium](physics/molecular-equilibrium.md)
- [Line broadening](physics/line-broadening.md)

</div>

</div>

<div class="pk-section-head" id="pipeline-at-a-glance">
  <span class="pk-section-head__num">04</span>
  <span class="pk-section-head__title">Pipeline at a glance</span>
</div>

Two entry points share the same validated synthesis core. Pre-computed atmospheres skip the emulator and `atlas_py` iteration entirely.

<div class="pk-diagram">
<svg viewBox="0 0 960 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="pykurucz pipeline">
  <defs>
    <marker id="pk-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="var(--pk-muted)"/>
    </marker>
  </defs>

  <!-- Stage labels -->
  <g font-family="JetBrains Mono, ui-monospace, monospace" font-size="10" letter-spacing="0.1em" fill="var(--pk-muted)" text-anchor="middle">
    <text x="105" y="20">INPUTS</text>
    <text x="385" y="20">ATMOSPHERE</text>
    <text x="665" y="20">PREPROCESSING</text>
    <text x="860" y="20">SYNTHESIS</text>
  </g>

  <!-- Inputs (column 1) -->
  <g font-family="Inter, sans-serif" font-size="13.5" fill="var(--pk-ink)">
    <rect x="20" y="50"  width="170" height="46" rx="9" fill="var(--pk-surface)" stroke="var(--pk-rule-strong)" stroke-width="1"/>
    <text x="34" y="71" font-weight="600">Stellar parameters</text>
    <text x="34" y="87" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">Teff · logg · [M/H] · [α/M]</text>

    <rect x="20" y="140" width="170" height="46" rx="9" fill="var(--pk-surface)" stroke="var(--pk-rule-strong)" stroke-width="1"/>
    <text x="34" y="161" font-weight="600">Existing .atm file</text>
    <text x="34" y="177" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">ATLAS · MARCS · PHOENIX</text>
  </g>

  <!-- Atmosphere stage (columns 2 & 3) -->
  <g font-family="Inter, sans-serif" font-size="13.5" fill="var(--pk-ink)">
    <rect x="220" y="50" width="160" height="46" rx="9" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)" stroke-width="1"/>
    <text x="234" y="71" font-weight="600">atmosphere emulator</text>
    <text x="234" y="87" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">predict warm start</text>

    <rect x="410" y="50" width="140" height="46" rx="9" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)" stroke-width="1"/>
    <text x="424" y="71" font-weight="600">atlas_py</text>
    <text x="424" y="87" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">iterate to converge</text>
  </g>

  <!-- Preprocessing (column 4) -->
  <g font-family="Inter, sans-serif" font-size="13.5" fill="var(--pk-ink)">
    <rect x="580" y="95" width="170" height="46" rx="9" fill="var(--pk-surface)" stroke="var(--pk-rule-strong)" stroke-width="1"/>
    <text x="594" y="116" font-weight="600">preprocess</text>
    <text x="594" y="132" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">populations · opacities</text>
  </g>

  <!-- Synthesis (column 5) -->
  <g font-family="Inter, sans-serif" font-size="13.5" fill="var(--pk-ink)">
    <rect x="780" y="50" width="160" height="46" rx="9" fill="var(--pk-accent-soft)" stroke="var(--pk-accent-rule)" stroke-width="1"/>
    <text x="794" y="71" font-weight="600">synthe_py</text>
    <text x="794" y="87" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-muted)">line-by-line transfer</text>

    <rect x="780" y="140" width="160" height="46" rx="9" fill="var(--pk-ink)" stroke="var(--pk-ink)"/>
    <text x="794" y="161" font-weight="600" fill="var(--pk-bg)">.spec output</text>
    <text x="794" y="177" font-family="Inter, sans-serif" font-size="10.5" fill="var(--pk-bg)" opacity="0.7">flux · continuum</text>
  </g>

  <!-- Connectors -->
  <g stroke="var(--pk-muted)" stroke-width="1.25" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <!-- 1. Stellar params -> emulator -->
    <path d="M190,73 L216,73" marker-end="url(#pk-arr)"/>
    <!-- 2. Emulator -> atlas_py -->
    <path d="M380,73 L406,73" marker-end="url(#pk-arr)"/>
    <!-- 3. Atlas_py -> preprocess -->
    <path d="M550,73 L562,73 L562,118 L576,118" marker-end="url(#pk-arr)"/>
    <!-- 4. External .atm -> preprocess (bypass route) -->
    <path d="M190,163 L562,163 L562,118 L576,118" marker-end="url(#pk-arr)"/>
    <!-- 5. Preprocess -> synthe_py -->
    <path d="M750,118 L766,118 L766,73 L776,73" marker-end="url(#pk-arr)"/>
    <!-- 6. Synthe_py -> .spec output -->
    <path d="M860,96 L860,138" marker-end="url(#pk-arr)"/>
  </g>
</svg>
</div>

> *Dedicated to the memory of **Robert L. Kurucz** (1944–2025), whose ATLAS and SYNTHE codes laid the foundations for modern stellar spectroscopy.*
