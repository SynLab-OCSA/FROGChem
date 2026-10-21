# NMR API Reference: `frogchem.nmr` & `frogchem.nmr_sim`

The `frogchem.nmr` and `frogchem.nmr_sim` modules handle NMR data calibration, equivalency detection, and quantum spin dynamics simulations.

---

## NMR Processing: `frogchem.nmr`

### `convert_shielding_to_shift(shieldings, nucleus="1H", ref_shielding=None, slope=-1.0, intercept=0.0)`

Converts magnetic isotropic shieldings $\sigma_{\text{iso}}$ into chemical shifts $\delta$ using standard or empirically fitted reference values.

$$\delta = \frac{\sigma_{\text{ref}} - \sigma_{\text{iso}}}{m} + c$$

**Parameters:**

* `shieldings` (`np.ndarray`): Array of isotropic magnetic shieldings in ppm.
* `nucleus` (`str`, optional): Target nucleus (`"1H"`, `"13C"`, `"19F"`, `"31P"`). Default: `"1H"`.
* `ref_shielding` (`float | None`, optional): Shielding of standard (e.g. TMS). If `None`, uses built-in database default.
* `slope` (`float`, optional): Linear calibration slope. Default: `-1.0`.
* `intercept` (`float`, optional): Calibration intercept in ppm. Default: `0.0`.

**Returns:**

* `np.ndarray`: Array of chemical shifts $\delta$ in ppm.

**Esempio:**

```python
import numpy as np
from frogchem.nmr import convert_shielding_to_shift

sigmas = np.array([24.58, 24.71, 29.50])
shifts = convert_shielding_to_shift(sigmas, nucleus="1H", ref_shielding=31.85)
print("1H shifts (ppm):", np.round(shifts, 2))
```

**Output atteso:**
```text
1H shifts (ppm): [7.27 7.14 2.35]
```

---

### `get_nmr_equivalencies(coords, atoms, target_nucleus="1H", sym_tol=0.1)`

Determines magnetically equivalent nuclei by merging 3D point group symmetry with local graph rotator topology.

**Parameters:**

* `coords` (`np.ndarray`): Cartesian coordinates $(N, 3)$ in $\text{\AA}$.
* `atoms` (`list[str]`): List of elemental symbols.
* `target_nucleus` (`str`, optional): Nucleus filter (`"1H"`, `"13C"`, or `"all"`). Default: `"1H"`.
* `sym_tol` (`float`, optional): Geometric tolerance for symmetry detection. Default: `0.1`.

**Returns:**

* `list[list[int]]`: List of equivalent atom index groups (0-based).

**Esempio:**

```python
import frogchem as frog
from frogchem.nmr import get_nmr_equivalencies

calc = frog.read("toluene.out")
equivs = get_nmr_equivalencies(calc.coordinates, calc.atoms, target_nucleus="1H")
print("Equivalent proton sets:", equivs)
```

**Output atteso:**
```text
Equivalent proton sets: [[1, 2, 3], [4, 5], [6, 7], [8]]
```

---

### `symmetrize_nmr_data(shifts, couplings, equivalencies)`

Averages chemical shifts and scalar coupling matrices over equivalent sets of nuclei.

**Parameters:**

* `shifts` (`np.ndarray`): Raw chemical shifts of shape $(N,)$.
* `couplings` (`np.ndarray | None`): Symmetric scalar coupling matrix $(N, N)$ in Hz.
* `equivalencies` (`list[list[int]]`): Equivalent atom groups from `get_nmr_equivalencies`.

**Returns:**

* `tuple[np.ndarray, np.ndarray | None]`: Symmetrized shifts and symmetrized coupling matrix.

---

## Quantum Spin Simulation: `frogchem.nmr_sim`

### `solve_spin_cluster(nu, J, spins=None, min_intensity=1e-5)`

Constructs and diagonalizes the exact isotropic spin Hamiltonian for a cluster of coupled spins using total magnetic quantum number $M$ block-partitioning.

**Parameters:**

* `nu` (`np.ndarray`): Array of resonance frequencies in Hz ($\nu_i = \delta_i \times \text{spec\_freq}$).
* `J` (`np.ndarray`): Symmetric scalar coupling matrix $(n, n)$ in Hz.
* `spins` (`float | Sequence[float] | None`, optional): Spin quantum numbers $S_k$. Default: `0.5` (spin-1/2).
* `min_intensity` (`float`, optional): Minimum transition probability cutoff. Default: `1e-5`.

**Returns:**

* `list[SpinTransition]`: Allowed magnetic dipole transitions with frequencies (Hz) and intensities.

**Esempio:**

```python
import numpy as np
from frogchem.nmr_sim import solve_spin_cluster

# Simple 2-spin system (AX system: delta_nu = 100 Hz, J = 10 Hz)
nu = np.array([500.0, 600.0])
J = np.array([[0.0, 10.0], [10.0, 0.0]])
transitions = solve_spin_cluster(nu, J)

for t in transitions:
    print(f"Freq: {t.freq_hz:.1f} Hz, Intensity: {t.intensity:.3f}")
```

**Output atteso:**
```text
Freq: 495.0 Hz, Intensity: 1.000
Freq: 505.0 Hz, Intensity: 1.000
Freq: 595.0 Hz, Intensity: 1.000
Freq: 605.0 Hz, Intensity: 1.000
```

---

### `simulate_1d_spectrum(transitions, spec_freq_mhz=500.0, ppm_min=None, ppm_max=None, points=4096, linewidth_hz=1.0)`

Broadens discrete quantum transitions into a continuous 1D NMR spectrum using Lorentzian line shapes.

**Parameters:**

* `transitions` (`list[SpinTransition]`): Quantum transitions from `solve_spin_cluster`.
* `spec_freq_mhz` (`float`, optional): Spectrometer base frequency in MHz. Default: `500.0`.
* `ppm_min`, `ppm_max` (`float | None`, optional): Spectral window limits in ppm.
* `points` (`int`, optional): Number of frequency points. Default: `4096`.
* `linewidth_hz` (`float`, optional): Full width at half maximum (FWHM) Lorentzian broadening in Hz. Default: `1.0`.

**Returns:**

* `tuple[np.ndarray, np.ndarray]`: `(ppm_axis, intensity_array)`.

---

### `simulate_cosy(shifts_ppm, j_matrix, spec_freq_mhz=500.0, ppm_min=None, ppm_max=None, points=512, linewidth_hz=2.0)`

Simulates a 2D COSY spectrum with diagonal peaks and scalar $J$-coupling cross-peaks.

**Parameters:**

* `shifts_ppm` (`np.ndarray`): Chemical shifts in ppm.
* `j_matrix` (`np.ndarray`): Scalar coupling matrix in Hz.
* `spec_freq_mhz` (`float`, optional): Spectrometer frequency in MHz. Default: `500.0`.
* `points` (`int`, optional): Grid resolution ($N \times N$). Default: `512`.
* `linewidth_hz` (`float`, optional): Peak width in Hz. Default: `2.0`.

**Returns:**

* `tuple[np.ndarray, np.ndarray, np.ndarray]`: `(f1_ppm, f2_ppm, 2d_intensity_matrix)`.

---

### `simulate_noesy(shifts_ppm, coordinates, distance_cutoff=5.0, spec_freq_mhz=500.0, points=512, linewidth_hz=2.0)`

Simulates a 2D NOESY spectrum with cross-relaxation cross-peaks proportional to $r_{ij}^{-6}$.

**Parameters:**

* `shifts_ppm` (`np.ndarray`): Chemical shifts in ppm.
* `coordinates` (`np.ndarray`): 3D Cartesian coordinates $(N, 3)$ in $\text{\AA}$.
* `distance_cutoff` (`float`, optional): Maximum distance for NOE cross-peak generation in $\text{\AA}$. Default: `5.0`.
* `spec_freq_mhz` (`float`, optional): Spectrometer frequency in MHz. Default: `500.0`.
* `points` (`int`, optional): Grid dimension. Default: `512`.

**Returns:**

* `tuple[np.ndarray, np.ndarray, np.ndarray]`: `(f1_ppm, f2_ppm, 2d_intensity_matrix)`.
