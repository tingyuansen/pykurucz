# Citation

If you use pykurucz in your research, please cite the following papers. Accurate citation helps us secure continued funding and maintenance for the project.

## pykurucz Paper

```bibtex
@misc{kim2026pykuruczpurepythonreimplementation,
  title         = {pyKurucz: A Pure Python Reimplementation of Kurucz ATLAS12 and SYNTHE for Stellar Spectrum Synthesis},
  author        = {Elliot M. Kim and Yuan-Sen Ting},
  year          = {2026},
  eprint        = {2603.11693},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.SR},
  url           = {https://arxiv.org/abs/2603.11693}
}
```

## kurucz-a1 Emulator Paper

If you use the `kurucz-a1` atmosphere emulator (the warm-start stage of Stellar Parameters), please also cite:

```bibtex
@article{li2025kurucza1,
  title   = {Differentiable Stellar Atmospheres with Physics-Informed
             Neural Networks},
  author  = {Li, Jiadong and Jian, Mingjie and Ting, Yuan-Sen
             and Green, Gregory M.},
  journal = {arXiv e-prints},
  year    = {2025},
  eprint  = {2507.06357},
  url     = {https://arxiv.org/abs/2507.06357}
}
```

## Original Kurucz Codes

Because pykurucz is a direct reimplementation of Robert L. Kurucz's ATLAS and SYNTHE codes, we also ask that you acknowledge the original work:

> "This research made use of pykurucz (Kim & Ting 2026), a Python reimplementation of the ATLAS and SYNTHE codes originally developed by R. L. Kurucz."

Key historical references for the Fortran codes include:

- **ATLAS12**: Kurucz, R. L. 1993, *ATLAS9 Stellar Atmosphere Programs and 2 km/s grid*
- **SYNTHE**: Kurucz, R. L., & Avrett, E. H. 1981, *SAO Special Report 391*
- **GFALL line list**: Kurucz, R. L. 2011, *Canadian Journal of Physics*, 89, 417

## How to Acknowledge

A suitable acknowledgment paragraph for your paper might read:

> "The synthetic spectra in this work were computed with pykurucz (Kim & Ting 2026), a pure Python reimplementation of the ATLAS12 and SYNTHE codes originally developed by R. L. Kurucz. The atmosphere warm-start was provided by the kurucz-a1 emulator (Li et al. 2025)."

!!! tip "BibTeX entries are also in the README"
    The `README.md` file in the repository root contains the same BibTeX entries, ready to copy into your reference manager.

## Next Steps

- Read the [License](license.md) for terms of use.
- See [Acknowledgements](acknowledgements.md) for the broader community contributions that make pykurucz possible.
