For every ORCA/Gaussian output the parser returns the excited states grouped by multiplicity. Each state
has its energy $E_k$ (eV), oscillator strength $f_k$, rotatory strength $R_k$ (10⁻⁴⁰ cgs), ⟨S²⟩ and the list
of orbital excitations $i \to a$ with their weights. It also returns the HOMO index and, for SOC jobs, the
⟨S|H_SO|T⟩ matrix (cm⁻¹). Files ending in `.cube` skip the parsing and go straight to step 4.

1. **State table.** Only excitations with weight > `-t` (0.20) are listed. Orbitals are renamed relative
   to the HOMO (β HOMO for β excitations), unless `--raw-orbitals`.
2. **SOC matrix.** Rows S0, S1… and columns T1, T2… as parsed. `--soc-states`/`--soc-singlets`/
   `--soc-triplets` only select rows and columns.
3. **Spectra** (`--uv`, `--ecd`). The window `--x-start`…`--x-end` (2–8 eV) is sampled on 1000 points. Every
   state with a non-zero intensity becomes an area-normalised Gaussian $g$ with
   $\sigma = \mathrm{FWHM}/(2\sqrt{2\ln 2})$ (`--fwhm`, 0.3 eV), centred at $E_k + \Delta$ (`--shift`):

   $$
   \varepsilon(E) \propto \sum_k f_k\, g(E - E_k - \Delta)
   \qquad
   \Delta\varepsilon(E) \propto \sum_k (E_k + \Delta)\, R_k\, g(E - E_k - \Delta)
   $$

   All the curves of one plot are divided by the same factor, the largest |value| among them. Relative
   heights between files in a superimposed plot are therefore kept. The x axis is
   $\lambda\,(\mathrm{nm}) = 1239.84193 / E\,(\mathrm{eV})$, with the energy axis on top. `--save-spectra`
   writes the same normalised curves as CSV.
4. **Charge transfer** (`--dct`). The basis functions are evaluated on a box around the molecule
   (`--voxels` points per axis, `--padding` Å). For each selected state:

   $$
   \Delta\rho(\mathbf r) = \sum_{ia} w_{ia}\left[\varphi_a(\mathbf r)^2 - \varphi_i(\mathbf r)^2\right]
   $$

   Only the transitions printed in the output enter the sum (ORCA prints weights above 0.01, Gaussian
   coefficients above 0.1), so $\Sigma w = \sum_{ia} w_{ia}$ is a little below 1: Δρ is divided by
   $\Sigma w$, shown in the table. This is the orbital-diagonal form of the difference density: the cross
   terms $\varphi_a\varphi_b$ and $\varphi_i\varphi_j$ of the full unrelaxed Δρ are left out.
   The positive and negative voxels give the charge $q_\mathrm{CT}$ (e) and the barycentres
   $\mathbf R_+$, $\mathbf R_-$. Then $D_\mathrm{CT} = |\mathbf R_+ - \mathbf R_-|$ (Å) and the dipole is
   $q_\mathrm{CT} D_\mathrm{CT}$. `--dct-threshold` < 100 discards the voxels above that percentile of |Δρ|.
   The state is classified with `--le-threshold` and `--ct-threshold`.
5. **Orbital grid** (`--orbital-grid`). The orbitals are the $i$ and $a$ of every transition with
   $w_{ia}$ ≥ `--grid-weight` (0.10) in the chosen states (`--grid-states`, by default the 4 lowest in
   energy), plus HOMO−N … LUMO+N with `--grid-frontier`, or exactly the list of `--grid-orbitals`. The
   automatic choice needs a closed-shell calculation. The orbitals are evaluated on the same box as in step
   4 and written to one multicube, each with its largest lobe positive (as in `frog-density`).
   `<stem>_grid.py` holds a manifest (file names, labels, energies, isovalues `--grid-iso` and
   `--grid-dct-iso`, caption) followed by a viewer that runs inside ChimeraX. The viewer reads the cubes
   itself, because ChimeraX's reader misreads multicubes with 10 or more orbitals. It opens one copy of the
   geometry per panel, turned by 25° about the vertical axis, with virtual orbitals above occupied ones and
   the Δρ cubes of `--dct` at the bottom, 3 panels per row, blue for + and red for −.
6. **Ensemble** (`-c`). Each state's E, f, R and ⟨S²⟩ are averaged with the Boltzmann weights over the
   conformers that contain that state. The SOC matrix is averaged the same way. The spectrum is the
   weighted sum of the per-conformer curves.

The broadening and the ECD weighting are explained in [TD-DFT theory](../theory/tddft.md).
