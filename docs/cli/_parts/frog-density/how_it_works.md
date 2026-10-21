The output is parsed into the geometry, the basis set (exponents and contraction coefficients per element
and shell), the map from basis functions to atoms and shell types, the MO coefficients $C_{\mu i}$ and the
occupations. Open-shell runs have separate α and β sets. If the basis or the coefficients are missing, the
command stops before building any grid.

1. **Basis table.** Each contracted function is placed on its atom (coordinates in bohr) as
   $\chi_\mu(\mathbf r) = P_\mu(x,y,z)\sum_p c_p N_p\, e^{-\alpha_p r^2}$. $P_\mu$ is the real solid harmonic
   (pure `5D 7F`) or Cartesian monomial (`6D 10F`) of that shell.
2. **Grid.** The box is the atomic bounding box plus `--padding` (3.0 Å) on every side. With `--spacing`
   (0.15 Å) the points are spaced by that step, so the three axes have different numbers of points. With
   `--voxels N` each axis has N points from end to end and the spacing differs per axis. All coordinates
   are converted to bohr.
3. **Quantity on the grid.** For every requested cube:

   $$
   \varphi_i(\mathbf r) = \sum_\mu C_{\mu i}\,\chi_\mu(\mathbf r)
   \qquad
   \rho(\mathbf r) = \sum_i n_i\,\varphi_i(\mathbf r)^2
   \qquad
   \rho_s = \rho_\alpha - \rho_\beta
   $$

   `--orbital N` and `--homo`/`--lumo` write $\varphi_i$ itself. For open shells the α set is used, or the β
   set with `--beta`. The HOMO and LUMO are the last occupied and first empty orbitals of that set. `--density`
   (also the default when no option is given) sums the occupied orbitals. In open-shell runs it sums the α and
   β orbitals separately. `--spin` subtracts the β density from the α density and is skipped with a warning
   for multiplicity 1.
4. **Writing.** Cubes are written in bohr, with values in atomic units (e/bohr³, or bohr⁻³ᐟ² for orbitals).
   Each run logs the grid size and the min/max value. Files are named `<stem>_<tag>.cube`, or follow
   `-o`/`--out-dir`.

The grid is identical for every cube of one run, so the cubes can be combined with `frog-cube`. See also
the [density cubes guide](../user-guide/density_cubes.md).
