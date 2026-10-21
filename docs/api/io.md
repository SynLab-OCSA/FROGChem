# I/O & Interoperability API Reference: `frogchem.io`

The `frogchem.io` module handles file input/output for quantum chemical coordinate formats, volumetric grids, and third-party chemistry library converters.

---

## Functions

### `read_xyz(filepath)`

Reads coordinates and atom labels from single-frame or multi-frame `.xyz` files.

**Parameters:**

* `filepath` (`str | Path`): Path to `.xyz` file.

**Returns:**

* `tuple[list[str], np.ndarray]`: Atom labels and Cartesian coordinates array $(N, 3)$ or $(F, N, 3)$ for trajectories.

**Esempio:**

```python
from frogchem.io import read_xyz

atoms, coords = read_xyz("water.xyz")
print("Atoms:", atoms)
print("Coordinates shape:", coords.shape)
```

**Output atteso:**
```text
Atoms: ['O', 'H', 'H']
Coordinates shape: (3, 3)
```

---

### `write_xyz(filepath, atoms, coordinates, comment="")`

Writes Cartesian coordinates to an `.xyz` format file.

**Parameters:**

* `filepath` (`str | Path`): Destination file path.
* `atoms` (`list[str]`): List of elemental symbols.
* `coordinates` (`np.ndarray`): Cartesian coordinates of shape $(N, 3)$ in $\text{\AA}$.
* `comment` (`str`, optional): Header comment line. Default: `""`.

**Esempio:**

```python
import numpy as np
from frogchem.io import write_xyz

atoms = ["C", "H", "H", "H", "H"]
coords = np.array([
    [0.0, 0.0, 0.0],
    [0.63, 0.63, 0.63],
    [-0.63, -0.63, 0.63],
    [-0.63, 0.63, -0.63],
    [0.63, -0.63, -0.63],
])

write_xyz("methane.xyz", atoms, coords, comment="Idealized methane")
print("Methane geometry written successfully.")
```

**Output atteso:**
```text
Methane geometry written successfully.
```

---

### `read_cube(filepath)`

Parses standard Gaussian volumetric CUBE files.

**Parameters:**

* `filepath` (`str | Path`): Path to `.cube` file.

**Returns:**

* `dict`: Dictionary containing `origin`, `voxel_vectors`, `atoms`, `coordinates`, and the 3D scalar grid data array.

---

### `write_cube(filepath, cube_data)`

Writes a 3D scalar grid to a Gaussian volumetric CUBE file.

**Parameters:**

* `filepath` (`str | Path`): Destination `.cube` path.
* `cube_data` (`dict`): CUBE data dictionary matching `read_cube` structure.
