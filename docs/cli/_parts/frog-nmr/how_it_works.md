From each ORCA output the parser takes the isotropic shielding $\sigma_\mathrm{iso}$ and the anisotropy of
every nucleus with shielding data, the J couplings (Hz) if they were computed, and the geometry.

1. **Shift.** Shieldings are converted with

   $$
   \delta = \frac{\sigma_\mathrm{ref} - \sigma_\mathrm{iso}}{1 - \sigma_\mathrm{ref}\cdot 10^{-6}}
   $$

   $\sigma_\mathrm{ref}$ comes from `--ref`: a number, `1H=…,13C=…`, or a reference output such as TMS,
   parsed once. Without `--ref` the per-nucleus default of the bundled `nuclei.json` is used.
2. **Tables.** `-n` keeps only the requested elements and `-i` keeps only the listed atoms. J and distance
   matrices are printed for ≤ 15 atoms or with `-i`. Otherwise J is shown as a list of pairs with
   |J| ≥ 0.5 Hz (40 rows unless `--couplings`). The NOE table lists every pair with $r \le$ `--r-cutoff`
   (5 Å), its relative NOE $(1.8\,\text{Å}/r)^6$ and its class.
3. **Ensemble** (`-c`). σ and the anisotropy are averaged with the Boltzmann weights over the conformers
   that have the atom, and J with the same weights. Distances are reported as ⟨r⟩ and as the NOE distance
   $\langle r^{-6}\rangle^{-1/6}$.
4. **Equivalence.** Before simulating, atoms are grouped as in `frog-nmr-equiv`. Each group takes the mean
   σ of its members. Between groups, J is the mean over member pairs and the distance is
   $\langle r^{-6}\rangle^{-1/6}$. `--no-equiv` keeps one group per atom.
5. **1D spectrum.** For a coupled nucleus (1H by default, all with `--coupled`) the spin Hamiltonian of each
   J-connected cluster is diagonalised exactly. Clusters are cut at `--j-cutoff`, 0.3 Hz. A cluster larger
   than 8 spins or than a 256-dimensional Hilbert space is treated to first order. The lines are Lorentzians
   of FWHM `--fwhm` (Hz) at the Larmor frequency, `--freq` × γ/γ(¹H). Decoupled nuclei get one line per
   group with height ∝ N.
6. **2D maps.** COSY uses the J matrix. NOESY uses the group distances up to `--r-cutoff`. HSQC/HMBC pairs
   are found from |J| and distance: a direct pair has |J| > 40 Hz or r < 1.45 Å. For ¹H–¹³C pairs the HSQC
   is multiplicity-edited.

The spin Hamiltonian, the cluster partitioning and the 2D models are in [NMR theory](../theory/nmr.md). The
[NMR user guide](../user-guide/nmr.md) covers references and ensembles.
