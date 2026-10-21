The command integrates a volumetric quantity, the spin density $\rho_s = \rho_\alpha - \rho_\beta$, over
atomic regions defined by a Voronoi partition of space.

1. **Reading the cube.** The Gaussian-cube parser reads the atoms (atomic number and position), the origin,
   the three voxel vectors, the number of points per axis and the values. The usual unit convention of
   the cube header applies: a positive point count means Bohr, a negative one means Å.
2. **Voronoi assignment.** The Cartesian position of each grid point,
   $\mathbf{r}_{ijk} = \mathbf{o} + i\,\mathbf{v}_1 + j\,\mathbf{v}_2 + k\,\mathbf{v}_3$, is computed. The point
   is then assigned to the **nearest nucleus** (k-d tree search). There is no atomic-radius weighting, so a
   large atom next to a small one gets no extra share of the space.
3. **Integration.** For atom $A$ with region $V_A$:

$$
N_A = \Delta V \sum_{\mathbf{r} \in V_A} \rho_s(\mathbf{r}),
\qquad
\Delta V = |\mathbf{v}_1 \cdot (\mathbf{v}_2 \times \mathbf{v}_3)|
$$

   The *Total* row is $\sum_A N_A$. `% Total` = $100\,N_A / N_\mathrm{tot}$, left empty when
   $|N_\mathrm{tot}| < 10^{-5}$.
4. **Fragments** (`--fragment NAME:0-5,8`, repeatable; or `--interactive`). The 0-based indices are
   parsed with inclusive ranges, and the fragment value is $\sum_{A \in F} N_A$. Atoms not in any fragment are
   summed in *Rest of system*. An index outside the molecule stops the command with an error, whereas in the
   interactive mode it is dropped with a warning.

The accuracy depends on the grid. The box must contain the whole density (padding), and the spacing must be fine
enough near the nuclei. Because the program's Mulliken or Löwdin spin populations come from basis-function
partitions, they differ from these real-space numbers. The cube is usually written with
`frog-density --spin`, see [Density and orbital cubes](../user-guide/density_cubes.md).
