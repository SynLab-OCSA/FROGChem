Only element symbols and coordinates (Å) are read, from the geometry that `load_calculation` returns for
the file. For an ORCA/Gaussian output that is the last geometry. Each atom gets the radius $r_i$ = Bondi vdW radius + `--vdw-offset`. Elements
missing from the table get 1.5 × the covalent radius.

1. **Grid.** The box is the atomic bounding box, enlarged by the largest $r_i + r_p$ plus `--padding`
   (3.0 Å). $r_p$ is `--probe-radius` (1.4 Å). The step is `--spacing` (0.1 Å), or `--voxels` points per
   axis.
2. **SAS** — an analytical signed distance field:

   $$
   \phi_\mathrm{SAS}(\mathbf p) = \min_i \left( |\mathbf p - \mathbf c_i| - (r_i + r_p) \right)
   $$

3. **SES** — EDTSurf algorithm (Xu & Zhang, 2009) on the same grid:
   1. Voxels with $\phi_\mathrm{SAS} \le 0$ form the SAS solid.
   2. A Euclidean distance transform gives each inside voxel its distance to the SAS boundary.
   3. Voxels further than $r_p$ from the boundary form the SES solid.
   4. Its signed distance field is $d_\mathrm{out} - d_\mathrm{in}$, the two distance transforms of the solid
      and of its complement.

   Distances are measured to voxel centres, so the SES volume is slightly too large (the code notes about
   +1 % at 0.1 Å and +10 % at 0.2 Å spacing).
4. **Volume.** The number of voxels with $\phi < 0$ times the voxel volume (Å³) is printed for each surface.
5. **Cube.** The field is written in Å: negative inside, zero on the surface, positive outside. The origin,
   voxel vectors and atoms are converted to bohr, as the cube format requires. Files are
   `<stem>_sas.cube` and `<stem>_ses.cube` (`--sas-only`, `--ses-only`, `--out-dir`).

Draw the 0.0 isosurface. The cube values are distances, not densities, so they cannot be mixed with
`frog-density` cubes in `frog-cube`.
