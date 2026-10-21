# Fragment Substitution API Reference: `frogchem.substituter`

The `frogchem.substituter` module manages 3D molecular fragment attachment, dihedral rotation, collision resolution, and relaxation with Machine Learning force fields.

---

## Functions

### `attach_fragment(molecule, fragment_name, target_atom, leaving_atom, optimize=True, calculator="aimnet2")`

Attaches a fragment from the internal library or an external XYZ file to a parent scaffold, rotates along the newly formed bond to minimize steric overlap, and optionally relaxes the geometry.

**Parameters:**

* `molecule` (`str | Path | ase.Atoms`): Parent molecule geometry.
* `fragment_name` (`str | Path`): Fragment identifier in library or path to fragment file.
* `target_atom` (`int`): 0-based atom index in parent molecule where fragment connects.
* `leaving_atom` (`int`): 0-based atom index of the atom being replaced (e.g. hydrogen).
* `optimize` (`bool`, optional): Whether to run geometric relaxation. Default: `True`.
* `calculator` (`str`, optional): Calculator backend (`"aimnet2"`, `"xtb"`, `"mace"`, `"ani"`). Default: `"aimnet2"`.

**Returns:**

* `ase.Atoms`: Substituted and optimized molecular geometry.

**Esempio:**

```python
from frogchem.substituter import attach_fragment

mol = attach_fragment("scaffold.xyz", "phenyl", target_atom=4, leaving_atom=5)
print("Substituted formula:", mol.get_chemical_formula())
```

**Output atteso:**
```text
Substituted formula: C13H12
```

---

### `reset_trajectory_file(filepath)`

Resets or creates a trajectory file used during conformational dihedral sweeps.

---

## Classes

### `TopologicalOptimizer`

Handles dihedral sweeping, collision detection, and ASE-calculator-based geometric minimization.

#### Constructor

```python
TopologicalOptimizer(calculator="aimnet2", fmax=0.05, steps=200)
```

#### Methods

##### `optimize(atoms)`
Runs geometry optimization on an `ase.Atoms` object using the selected calculator.

**Parameters:**
* `atoms` (`ase.Atoms`): Starting geometry.

**Returns:**
* `ase.Atoms`: Energy-minimized structure.

**Esempio:**

```python
from frogchem.substituter import TopologicalOptimizer
import frogchem as frog

calc = frog.read("opt.out")
optimizer = TopologicalOptimizer(calculator="aimnet2", fmax=0.05)
relaxed = optimizer.optimize(calc.to_ase())
print("Relaxation completed. Energy (eV):", relaxed.get_potential_energy())
```

**Output atteso:**
```text
Relaxation completed. Energy (eV): -425.841025
```
