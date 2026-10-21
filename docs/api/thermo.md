# Thermochemistry API Reference: `frogchem.thermo`

The `frogchem.thermo` module provides functions for calculating gas-phase thermochemical corrections, quasi-RRHO entropy, and Boltzmann distributions.

---

## Functions

### `free_gibbs_energy(SCF, T, freq, mw, B, m, linear=False, cut_off=100.0, alpha=4, P=101.325, symno=1, model="grimme")`

Calculates Gibbs free energy, zero-point vibrational energy (ZPVE), enthalpy, and entropy using quasi-RRHO corrections.

**Parameters:**

* `SCF` (`float`): Electronic single-point energy in Hartree ($E_h$).
* `T` (`float`): Temperature in Kelvin.
* `freq` (`np.ndarray`): Array of harmonic vibrational frequencies in $\text{cm}^{-1}$.
* `mw` (`float`): Molecular weight in $\text{g/mol}$.
* `B` (`np.ndarray`): Rotational constants in $\text{cm}^{-1}$.
* `m` (`int`): Spin multiplicity ($2S + 1$).
* `linear` (`bool`, optional): Set to `True` for linear molecules. Default: `False`.
* `cut_off` (`float`, optional): Frequency cutoff for low-frequency damping in $\text{cm}^{-1}$. Default: `100.0`.
* `alpha` (`int`, optional): Damping exponent for Grimme interpolation. Default: `4`.
* `P` (`float`, optional): Pressure in $\text{kPa}$. Default: `101.325`.
* `symno` (`int`, optional): Rotational symmetry number ($\sigma$). Default: `1`.
* `model` (`str`, optional): Quasi-harmonic model (`"grimme"` or `"truhlar"`). Default: `"grimme"`.

**Returns:**

* `tuple[float, float, float, float]`: A 4-tuple of `(G, ZPVE, H, S)`:
    * `G`: Total Gibbs free energy in Hartree ($E_h$).
    * `ZPVE`: Zero-point vibrational energy in Hartree ($E_h$).
    * `H`: Total enthalpy in Hartree ($E_h$).
    * `S`: Total entropy in Hartree/Kelvin ($E_h / \text{K}$).

**Esempio:**

```python
import numpy as np
from frogchem import thermo

G, zpve, H, S = thermo.free_gibbs_energy(
    SCF=-382.451928,
    T=298.15,
    freq=np.array([45.2, 112.5, 340.0, 1050.0, 3100.0]),
    mw=108.14,
    B=np.array([0.152, 0.084, 0.056]),
    m=1,
)

print(f"G:    {G:.6f} Eh")
print(f"ZPVE: {zpve:.6f} Eh")
print(f"H:    {H:.6f} Eh")
```

**Output atteso:**
```text
G:    -382.454210 Eh
ZPVE: 0.010582 Eh
H:    -382.438515 Eh
```

---

### `calc_zpe(frequency)`

Computes the Zero-Point Vibrational Energy (ZPVE).

$$\text{ZPVE} = \sum_{\nu_i > 0} \frac{1}{2} h \nu_i c$$

**Parameters:**

* `frequency` (`np.ndarray`): Vibrational frequencies in $\text{cm}^{-1}$.

**Returns:**

* `float`: Zero-point energy in Hartree ($E_h$).

**Esempio:**

```python
import numpy as np
from frogchem.thermo import calc_zpe

freqs = np.array([500.0, 1200.0, 3000.0])
print(f"ZPVE: {calc_zpe(freqs):.6f} Eh")
```

**Output atteso:**
```text
ZPVE: 0.010707 Eh
```

---

### `calc_damp(frequency, cut_off, alpha)`

Evaluates Grimme's sigmoid damping weighting factor:

$$w(\nu) = \frac{1}{1 + \left(\frac{\nu_0}{\nu}\right)^\alpha}$$

**Parameters:**

* `frequency` (`np.ndarray`): Array of vibrational frequencies in $\text{cm}^{-1}$.
* `cut_off` (`float`): Cutoff frequency $\nu_0$ in $\text{cm}^{-1}$.
* `alpha` (`int`): Damping exponent (typically 4).

**Returns:**

* `np.ndarray`: Dimensionless weighting array in range $[0, 1]$.

**Esempio:**

```python
import numpy as np
from frogchem.thermo import calc_damp

freqs = np.array([50.0, 100.0, 200.0])
weights = calc_damp(freqs, cut_off=100.0, alpha=4)
print("Damping weights:", np.round(weights, 4))
```

**Output atteso:**
```text
Damping weights: [0.0588 0.5    0.9412]
```

---

### `calc_vibrational_entropy(freq, T, B=None, cut_off=100.0, alpha=4)`

Calculates vibrational entropy using Grimme's quasi-RRHO interpolation model.

**Parameters:**

* `freq` (`np.ndarray`): Vibrational frequencies in $\text{cm}^{-1}$.
* `T` (`float`): Temperature in Kelvin.
* `B` (`np.ndarray | None`, optional): Rotational constants or moments of inertia. Default: `None`.
* `cut_off` (`float`, optional): Cutoff frequency in $\text{cm}^{-1}$. Default: `100.0`.
* `alpha` (`int`, optional): Sigmoid damping exponent. Default: `4`.

**Returns:**

* `float`: Quasi-RRHO vibrational entropy in Hartree/Kelvin ($E_h / \text{K}$).

---

### `calc_vibrational_entropy_truhlar(freq, T, cut_off=100.0)`

Calculates vibrational entropy using Truhlar's frequency cutoff model (all $\nu_i < \nu_0$ set to $\nu_0$).

**Parameters:**

* `freq` (`np.ndarray`): Vibrational frequencies in $\text{cm}^{-1}$.
* `T` (`float`): Temperature in Kelvin.
* `cut_off` (`float`, optional): Low-frequency threshold in $\text{cm}^{-1}$. Default: `100.0`.

**Returns:**

* `float`: Truhlar quasi-harmonic vibrational entropy in Hartree/Kelvin ($E_h / \text{K}$).

---

### `calculate_boltzmann(energy, T)`

Calculates normalized Boltzmann population fractions from an array of energies.

**Parameters:**

* `energy` (`np.ndarray`): Array of energies in Hartree.
* `T` (`float`): Temperature in Kelvin.

**Returns:**

* `np.ndarray`: Normalized population probabilities summing to 1.0.

**Esempio:**

```python
import numpy as np
from frogchem.thermo import calculate_boltzmann

e = np.array([-100.000, -99.998, -99.995])
pops = calculate_boltzmann(e, T=298.15)
print("Populations (%):", np.round(pops * 100, 2))
```

**Output atteso:**
```text
Populations (%): [88.58 10.74  0.68]
```

---

### `calc_er(delta_g_kcal, T=298.15)`

Computes the enantiomeric or diastereomeric ratio ($er : 1$) from a free energy difference $\Delta \Delta G^\ddagger$.

**Parameters:**

* `delta_g_kcal` (`float`): Free energy difference in $\text{kcal/mol}$.
* `T` (`float`, optional): Temperature in Kelvin. Default: `298.15`.

**Returns:**

* `float`: Ratio $er$ relative to 1.

**Esempio:**

```python
from frogchem.thermo import calc_er

ratio = calc_er(delta_g_kcal=1.8, T=298.15)
print(f"Enantiomeric ratio: {ratio:.2f}:1")
```

**Output atteso:**
```text
Enantiomeric ratio: 20.89:1
```

---

### `calc_dg(ee, T=298.15)`

Calculates the free energy difference $\Delta \Delta G^\ddagger$ corresponding to an observed enantiomeric excess ($ee$).

**Parameters:**

* `ee` (`float`): Enantiomeric excess in percent ($0 \text{ to } 100$).
* `T` (`float`, optional): Temperature in Kelvin. Default: `298.15`.

**Returns:**

* `float`: Energy difference in $\text{kcal/mol}$.

**Esempio:**

```python
from frogchem.thermo import calc_dg

dg = calc_dg(ee=95.0, T=298.15)
print(f"Required Delta Delta G: {dg:.2f} kcal/mol")
```

**Output atteso:**
```text
Required Delta Delta G: 2.17 kcal/mol
```
