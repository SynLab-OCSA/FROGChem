The method follows the SambVca model: the fraction of a sphere around the metal occupied by the
van der Waals spheres of the other atoms (`frogchem.steric_maps`).

1. **Geometry.** `load_calculation` reads the structure (XYZ, or the geometry of an ORCA/Gaussian output).
2. **Orientation** (`orient_positions`). The centre (`--center`) is moved to the origin. With `-z`, the
   centroid of those atoms is rotated onto +Z (Rodrigues rotation; `--flip-z` uses −Z). With `-x`, the
   centroid of those atoms, projected onto the $xy$ plane, is rotated onto +X around Z (`--flip-x`:
   −X). Without `-z` the input orientation is kept.
3. **Radii.** Bondi radii × 1.17. Elements without a Bondi radius get 1.5 × covalent radius × 1.17.
   The centre and every `-E` atom are removed; hydrogens are kept.
4. **Buried volume** (`compute_buried_volume`). A cubic grid with points $k\,s$
   ($s$ = `-s`, 0.1 Å; $k=-m\dots m$, $m=\lfloor R/s\rfloor$) fills the sphere of radius $R$
   (`-r`, 3.5 Å). A point is buried if it lies inside any atomic sphere:

   $$
   \%V_\mathrm{bur} = 100\,\frac{N_\mathrm{buried}}{N_\mathrm{sphere}}.
   $$

   Quadrants (NE: $x>0,y>0$; NW: $x<0,y>0$; SW; SE) and octants (Front: $z>0$, Back: $z<0$) use the
   same ratio inside each region. Points that lie exactly on a dividing plane count ½ on each side,
   so no region is favoured.
5. **Steric map** (`-m`, `compute_steric_map`). On an $xy$ grid with spacing $s$ inside the circle of
   radius $R$, the height is the highest point of any atomic sphere, $z_i+\sqrt{r_i^2-\rho^2}$, clipped
   to the sphere surface, for spheres whose chord overlaps the sphere column. Points with no atom are
   left empty. The map is drawn as seen from +Z, with the molecule overlaid, by `stericrender` and
   `xyzrender` into `<stem>_steric_map.svg`. `-E` atoms are drawn again on top of the map unless
   `--no-excluded-on-top` is given. `--png` converts the SVG with `cairosvg` at 2× scale.
