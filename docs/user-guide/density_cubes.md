# Orbitals, Densities & Cube Files

A cube file (`.cube`, Gaussian cube format) stores a quantity on a 3D grid of points around the molecule:
an orbital, the electron density, a spin density, a density difference or a surface. Viewers (VMD,
ChimeraX, Avogadro, GaussView) draw it as an isosurface. FROG builds cubes directly from ORCA outputs,
combines them, integrates them on atoms and builds solvent surfaces. The examples use the sample files in
`files/`; run them from the repository root. Details of each command: [`frog-density`](../cli/frog-density.md),
[`frog-cube`](../cli/frog-cube.md), [`frog-solvent`](../cli/frog-solvent.md) and [`frog-spin`](../cli/frog-spin.md).

---

## 1. What you need

Orbitals and densities are computed from the basis set and the MO coefficients, which ORCA prints only on
request: add **`! PrintBasis PrintMOs`** to the input (a single point on the final geometry is enough).
Gaussian outputs are not supported for cubes. Solvent surfaces need only a geometry (`.xyz` or any output).

Units: densities in e/bohr³, orbitals in bohr⁻³ᐟ², solvent surfaces as a signed distance in Å.

---

## 2. Orbitals and densities (`frog-density`)

```bash
frog-density files/tddft_mos.out --homo --lumo --density --voxels 50 --out-dir cubes
```

writes `cubes/tddft_mos_homo.cube`, `..._lumo.cube` and `..._density.cube` (50×50×50 points, 1.75 MB each).
Other options: `--orbital 41 42` (0-based MO numbers), `--spin` (α − β spin density of an open-shell
calculation), `--beta` (β orbitals), `--spacing`/`--voxels`/`--padding` for the grid.

Typical isovalues: ±0.03–0.05 for orbitals, 0.02–0.05 e/bohr³ for the total density, ±0.002–0.005 for spin
densities and density differences.

From Python:

```python
import frogchem as frog
from frogchem.density_builder import build_density_cube, build_orbital_cube

calc = frog.read("files/tddft_mos.out")                     # water, ! PrintBasis PrintMOs
build_density_cube(calc, output="water_density.cube", n_voxels=60, padding=4.0)
build_orbital_cube(calc, calc.homo, output="water_homo.cube", n_voxels=60, padding=4.0)
```

---

## 3. Integrating a cube on atoms

`integrate_density_voronoi` assigns every grid point to the nearest atom and returns ∫ρ dV per atom (in
electrons for a density in e/bohr³). This is what `frog-spin` does with a spin-density cube.

```python
from frogchem.io import parse_gaussian_cube
from frogchem.grid import integrate_density_voronoi

cube = parse_gaussian_cube("water_density.cube")
per_atom = integrate_density_voronoi(cube)
print(cube['grid_size'], [round(float(v), 3) for v in per_atom], f"total {per_atom.sum():.3f}")
```

```text
(60, 60, 60) [6.577, 1.376, 1.384] total 9.337
```

Two things to notice:

* The total is 9.34 instead of 10 electrons: the core density at the nuclei is too sharp for a ~0.16 Å grid.
  Integrated **total** densities need fine grids; spin densities and density differences (no core
  contribution) are much less sensitive.
* "Nearest atom" ignores atomic size, so H atoms get more and O less than in Mulliken or NPA analyses.
  Use it for trends and for spin populations (`frog-spin`), not as atomic charges.

---

## 4. Cube arithmetic (`frog-cube`)

File names below are placeholders for your own cubes.

```bash
frog-cube -s A.cube B.cube           # B − A  -> B_minus_A.cube (current folder)
frog-cube -a A.cube B.cube           # B + A  -> B_plus_A.cube
frog-cube -m spin.cube -1            # ×(−1)  -> spin_x-1.cube (next to spin.cube)
```

The two cubes must share the grid (same number of points, origin and spacing): make them from the same
geometry with the same grid options, otherwise the command stops with "Grid sizes must match". A typical
use is a binding density difference ρ(AB) − ρ(A) − ρ(B), with the fragments computed at the geometry they
have in the complex:

```bash
frog-cube -a A.cube B.cube -o AplusB.cube
frog-cube -s AplusB.cube AB.cube -o delta_rho.cube    # AB − (A + B)
```

In Python, `frogchem.grid.add_grids(g1, g2)`, `subtract_grids(g1, g2)` (returns **g2 − g1**) and
`multiply_grid(g, c)` work on the dictionaries returned by `parse_gaussian_cube`, and
`frogchem.io.write_gaussian_cube(path, grid)` writes the result.

Excited-state difference densities are made by `frog-tddft --dct` (see [Excited states](spectroscopy.md)).

---

## 5. Solvent surfaces (`frog-solvent`)

```bash
frog-solvent files/water.xyz --out-dir surfaces
```

writes `surfaces/water_sas.cube` (solvent accessible surface, traced by the centre of a 1.4 Å probe) and
`surfaces/water_ses.cube` (solvent excluded surface, the boundary the solvent cannot cross), and reports
the enclosed volumes: 124.50 Å³ (SAS) and 19.85 Å³ (SES) for water. Radii are Bondi van der Waals radii
(`--vdw-offset` to enlarge them), the probe radius is `--probe-radius`. The cube values are a signed
distance in Å (negative inside): draw the isosurface at **0.0**. At the default 0.1 Å spacing each cube is
about 28 MB even for water: use `--spacing 0.2` for large molecules.
