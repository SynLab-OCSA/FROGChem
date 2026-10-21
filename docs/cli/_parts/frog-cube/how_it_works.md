Both cubes are read in full with `frogchem.io.parse_gaussian_cube`, which reads the title lines, origin,
number of points and voxel vectors along each axis, atoms and the 3D array of values. Then:

1. **Grid check** (`-a`, `-s`). The command stops unless the two cubes have the same number of points per
   axis. The origin and the three voxel vectors must also agree within $10^{-5}$ (in the cube's own units).
   The values are never interpolated or resampled.
2. **Arithmetic, voxel by voxel.** The order of `-s FILE1 FILE2` is reversed with respect to the command
   line:

   $$
   \texttt{-a}:\; V = V_2 + V_1
   \qquad
   \texttt{-s}:\; V = V_2 - V_1
   \qquad
   \texttt{-m}:\; V = c\,V
   $$

3. **Header.** The result is a copy of the first file (`FILE1`, or `FILE` for `-m`): atoms, origin, voxel
   vectors and units. Only the two comment lines (`Difference` / `Difference of grids`, `Addition` …,
   `Multiplication` / `Scaled by c`) and the data are replaced.
4. **Output name.** `-a`/`-s` write `<stem2>_plus|minus_<stem1>.cube` in the current folder. `-m` writes
   `<stem>_x<c>.cube` next to the input. `-o` overrides both. A path that is one of the inputs is refused.

The values keep the units of the inputs (e/bohr³ for densities). The command does not integrate the result
or check its sign. Building comparable cubes is covered in the
[density cubes guide](../user-guide/density_cubes.md).
