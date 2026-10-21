# Core API Reference: Parsers & Calculations

The core parsing module provides the primary entry points for loading quantum chemistry outputs into memory.

---

## Top-Level Functions

### `frogchem.read(filename)`

Auto-detects output format (ORCA or Gaussian, plain text or gzip-compressed) and returns an initialized calculation instance.

**Parameters:**

* `filename` (`str | Path`): Path to output or log file (`.out`, `.log`, `.out.gz`, `.log.gz`).

**Returns:**

* `Calculation | GaussianCalculation`: Initialized calculation object with lazy property caching.

**Raises:**

* `FileNotFoundError`: If the target file does not exist on disk.

**Esempio:**

```python
import frogchem as frog

calc = frog.read("opt_freq.out.gz")
print(f"Software: {type(calc).__name__}")
print(f"Theory:   {calc.theory}")
print(f"Energy:   {calc.final_energy:.6f} Eh")
```

**Output atteso:**
```text
Software: Calculation
Theory:   wB97X-D4/def2-TZVP
Energy:   -382.451928 Eh
```

---

### `frogchem.read_orca(filename)`

Explicitly parses an ORCA output file.

**Parameters:**

* `filename` (`str | Path`): Path to ORCA output file.

**Returns:**

* `Calculation`: ORCA calculation parser instance.

**Esempio:**

```python
import frogchem as frog

orca_calc = frog.read_orca("opt.out")
print(f"ORCA energy: {orca_calc.final_energy:.6f} Hartree")
```

**Output atteso:**
```text
ORCA energy: -382.451928 Hartree
```

---

### `frogchem.read_gaussian(filename)`

Explicitly parses a Gaussian log/output file.

**Parameters:**

* `filename` (`str | Path`): Path to Gaussian log file.

**Returns:**

* `GaussianCalculation`: Gaussian calculation parser instance.

**Esempio:**

```python
import frogchem as frog

gau_calc = frog.read_gaussian("opt.log")
print(f"Gaussian energy: {gau_calc.final_energy:.6f} Hartree")
```

**Output atteso:**
```text
Gaussian energy: -382.451928 Hartree
```

---

## Classes

### `frogchem.orca_parser.calculation.Calculation`

Main ORCA parser class with lazy attribute evaluation.

#### Attributes & Properties

* `final_energy` (`float`): Final electronic single-point energy in Hartree.
* `scf_energy` (`float`): Converged SCF electronic energy in Hartree.
* `dispersion_correction` (`float`): Empirical dispersion correction (D3, D4) in Hartree.
* `coordinates` (`np.ndarray`): Cartesian coordinates of shape $(N, 3)$ in $\text{\AA}$.
* `initial_coordinates` (`np.ndarray`): Starting Cartesian coordinates before optimization.
* `atoms` (`list[str]`): List of elemental symbols (`['C', 'H', ...]`).
* `charge` (`int`): Net molecular charge.
* `multiplicity` (`int`): Spin multiplicity ($2S + 1$).
* `theory` (`str`): Functional/method and basis set description.
* `frequencies` (`np.ndarray`): Harmonic vibrational frequencies in $\text{cm}^{-1}$.
* `rotational_constants` (`np.ndarray`): Rotational constants $(B_x, B_y, B_z)$ in $\text{cm}^{-1}$.
* `molecular_weight` (`float`): Molecular mass in g/mol.
* `mulliken_charges` (`np.ndarray`): Mulliken partial atomic charges.
* `loewdin_charges` (`np.ndarray`): Löwdin partial atomic charges.
* `spin_densities` (`np.ndarray`): Localized atomic spin densities.
* `transitions` (`list[dict]`): TD-DFT electronic excitation states.
* `nmr_shieldings` (`np.ndarray`): Magnetic isotropic shieldings $\sigma_{\text{iso}}$ in ppm.
* `nmr_couplings` (`np.ndarray`): Pairwise scalar coupling matrix $J_{ij}$ in Hz.

#### Methods

##### `to_ase()`
Converts the calculation into an `ase.Atoms` object.

**Returns:**
* `ase.Atoms`: Atomic Simulation Environment atoms object with positions, symbols, and cell info.

**Esempio:**
```python
atoms = calc.to_ase()
print(f"Formula: {atoms.get_chemical_formula()}")
```

**Output atteso:**
```text
Formula: C7H8O
```

##### `to_rdkit(charge=None)`
Converts the calculation into an RDKit `Mol` object with a 3D conformer and perceived bond orders.

**Parameters:**
* `charge` (`int | None`): Net charge override (defaults to `calc.charge`).

**Returns:**
* `rdkit.Chem.Mol`: RDKit molecule instance.

**Esempio:**
```python
mol = calc.to_rdkit()
print(f"Num atoms in Mol: {mol.GetNumAtoms()}")
```

**Output atteso:**
```text
Num atoms in Mol: 16
```

---

### `frogchem.gaussian_parser.calculation.GaussianCalculation`

Main Gaussian 09/16 output parser class.

Provides the same core properties (`final_energy`, `coordinates`, `frequencies`, `atoms`, `to_ase`, `to_rdkit`) mirroring `Calculation` for cross-software compatibility.
