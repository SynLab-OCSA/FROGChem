# Geometry & Symmetry API Reference

FROG includes mathematical and graph-theoretical routines for 3D coordinates, point group symmetries, and molecular topology.

---

## Symmetry: `frogchem.symmetry`

### `detect_point_group(coords, atoms, tol=0.1)`

Detects the Schoenflies point group symbol of a 3D molecular geometry.

**Parameters:**

* `coords` (`np.ndarray`): Cartesian coordinates of shape $(N, 3)$ in $\text{\AA}$.
* `atoms` (`list[str]`): List of elemental symbols.
* `tol` (`float`, optional): Geometric tolerance for symmetry elements. Default: `0.1`.

**Returns:**

* `str`: Schoenflies symbol (e.g. `"C2v"`, `"D3h"`, `"Cs"`, `"C1"`).

**Esempio:**

```python
import numpy as np
from frogchem.symmetry import detect_point_group

# Water molecule coordinates in yz plane
coords = np.array([
    [0.0,  0.000,  0.117],  # O
    [0.0,  0.757, -0.469],  # H
    [0.0, -0.757, -0.469],  # H
])
atoms = ["O", "H", "H"]

pg = detect_point_group(coords, atoms)
print("Point Group:", pg)
```

**Output atteso:**
```text
Point Group: C2v
```

---

## Math & Geometry: `frogchem.math_geometry`

### `calc_distance(coord1, coord2)`
Calculates Euclidean distance between two 3D points in $\text{\AA}$.

**Esempio:**
```python
import numpy as np
from frogchem.math_geometry import calc_distance

p1 = np.array([0.0, 0.0, 0.0])
p2 = np.array([1.0, 1.0, 1.0])
print(f"Distance: {calc_distance(p1, p2):.4f} Å")
```

**Output atteso:**
```text
Distance: 1.7321 Å
```

### `calc_angle(coord1, coord2, coord3)`
Calculates the valence bond angle between three points in degrees ($0^\circ \text{ to } 180^\circ$).

**Esempio:**
```python
import numpy as np
from frogchem.math_geometry import calc_angle

p1 = np.array([1.0, 0.0, 0.0])
p2 = np.array([0.0, 0.0, 0.0])
p3 = np.array([0.0, 1.0, 0.0])
print(f"Angle: {calc_angle(p1, p2, p3):.1f}°")
```

**Output atteso:**
```text
Angle: 90.0°
```

### `calc_dihedral(coord1, coord2, coord3, coord4)`
Calculates the proper dihedral (torsion) angle between four points in degrees ($-180^\circ \text{ to } +180^\circ$).

### `calc_rmsd_kabsch(coords_p, coords_q)`
Performs optimal superposition (Kabsch algorithm) and returns the minimum root-mean-square deviation (RMSD) in $\text{\AA}$.

---

## Graph Theory: `frogchem.graph_theory`

### `create_graph_matrix(coords, atoms, tolerance=1.3)`
Builds the molecular adjacency matrix based on covalent radii and Euclidean distances.

### `detect_fragments(adj_matrix)`
Finds connected components in the adjacency matrix, identifying disconnected fragments.

### `find_stereocenters(adj_matrix, atoms, coords)`
Identifies chiral carbons ($sp^3$ centers bonded to 4 distinct groups) and determines stereochemical descriptors.
