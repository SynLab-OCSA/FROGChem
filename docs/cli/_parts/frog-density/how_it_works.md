The output is parsed into the geometry, the basis set (exponents and contraction coefficients per element
and shell), the map from basis functions to atoms and shell types, the MO coefficients $C_{\mu i}$ and the
occupations. Open-shell runs have separate α and β sets. If the basis or the coefficients are missing, the
command stops before building any grid.

1. **Basis table.** Each contracted function is placed on its atom (coordinates in bohr) as
   $\chi_\mu(\mathbf r) = P_\mu(x,y,z)\sum_p c_p N_p\, e^{-\alpha_p r^2}$. $P_\mu$ is the real solid harmonic
   (pure `5D 7F`) or Cartesian monomial (`6D 10F`) of that shell.
2. **Grid.** The box is the atomic bounding box plus `--padding` (3.0 Å) on every side. With `--voxels N`
   (default 80) each axis has N points from end to end and the spacing differs per axis; `--voxels NX NY NZ`
   sets the three counts. With `--spacing` instead the points are spaced by that step, so the three axes
   have different numbers of points. The two options are alternatives: giving both is an error. All
   coordinates are converted to bohr.
3. **Quantity on the grid.** For every requested cube:

   $$
   \varphi_i(\mathbf r) = \sum_\mu C_{\mu i}\,\chi_\mu(\mathbf r)
   \qquad
   \rho(\mathbf r) = \sum_i n_i\,\varphi_i(\mathbf r)^2
   \qquad
   \rho_s = \rho_\alpha - \rho_\beta
   $$

   `--orbital N`, `--homo`/`--lumo` and `--frontier N` write $\varphi_i$ itself. The basis functions are
   evaluated once for all the requested orbitals, in blocks of points, so the full basis × grid matrix is
   never held in memory. The sign of each orbital is chosen so that $\sum_{\mathbf r}\varphi_i^3 > 0$ (its
   largest lobe positive). For open shells the α set is used, or the β set with `--beta`; `--beta` on a
   closed-shell run stops with an error. The HOMO and LUMO are the last occupied and first empty orbitals
   of that set, and are printed at every run. `--density`
   (also the default when no option is given) sums the occupied orbitals. In open-shell runs it sums the α and
   β orbitals separately. `--spin` subtracts the β density from the α density and is skipped with a warning
   for multiplicity 1.
4. **Writing.** Cubes are written in bohr, with values in atomic units (e/bohr³, or bohr⁻³ᐟ² for orbitals).
   Each run logs the grid size and the min/max value. Files are named `<stem>_<tag>.cube`, or follow
   `-o`/`--out-dir`. One orbital gives an ordinary cube. Two or more give one **multicube**,
   `<stem>_orbitals.cube`: the atom count is written negative and the atoms are followed by the record
   `N id₁ … id_N` (Gaussian `10I5`, ten fields per line, ids = 0-based index + 1); at every grid point the
   N values follow one another. `--split` writes one cube per orbital instead.

The grid is identical for every cube of one run, so the cubes can be combined with `frog-cube`. See also
the [density cubes guide](../user-guide/density_cubes.md).
