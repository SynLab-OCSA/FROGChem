# NMR Prediction & Spin Simulation

FROG reads NMR shieldings and J couplings from ORCA outputs, converts shieldings into
chemical shifts, detects chemically equivalent nuclei and simulates 1D and 2D spectra.

**What you need.** An ORCA NMR calculation on an optimised geometry: `! NMR` computes the
shieldings; for J couplings add an `%eprnmr` block, e.g.

```text
! B3LYP D4 def2-TZVPP CPCM(CHCl3) NMR
%eprnmr
  nuclei = all H {shift, ssall}
end
```

For accurate shifts compute TMS with exactly the same method, basis and solvent and use it as
the reference. The examples below use `files/NMR_ORCA.out` from the repository (39 atoms;
1H shieldings and H–H couplings) and are run from the repository root.

---

## 1. Shieldings and couplings from Python

```python
import frogchem as frog

calc = frog.read("files/NMR_ORCA.out")
nmr = calc.nmr                      # NMRAnalysis object

for idx in sorted(nmr.shifts)[:3]:  # keys are 0-based atom indices
    d = nmr.shifts[idx]
    print(f"{d['element']}{idx + 1} (index {idx}): sigma_iso = {d['iso']:.3f} ppm")

print(f"J(H30, H31) = {nmr.j_couplings[(29, 30)]:.2f} Hz")
```

```text
H24 (index 23): sigma_iso = 25.708 ppm
H25 (index 24): sigma_iso = 27.933 ppm
H26 (index 25): sigma_iso = 23.443 ppm
J(H30, H31) = -11.99 Hz
```

`nmr.shifts[idx]` also holds the anisotropy (`'aniso'`); `nmr.j_couplings` is keyed by
0-based atom pairs (both orders are present). Atom labels such as `H24` are 1-based, as in the
ORCA output.

## 2. From shielding to chemical shift

δ ≈ σ_ref − σ (exactly δ = (σ_ref − σ)/(1 − σ_ref·10⁻⁶)):

```python
from frogchem import nmr

print(f"{nmr.convert_shielding_to_shift(25.708, nucleus='1H', ref_sigma=31.8):.3f} ppm")
```

```text
6.092 ppm
```

`ref_sigma` can be a number, the path of a reference output (e.g. TMS), or omitted: the
built-in defaults are 31.8 ppm (1H), 185.0 ppm (13C), −135.0 ppm (15N), 188.0 ppm (19F) and
280.0 ppm (31P). The defaults are only approximate: a TMS calculation at your own level is
better.

## 3. Chemically equivalent nuclei

```python
from frogchem import nmr
from frogchem.filetypes import load_structures

atoms, coords = load_structures("files/tol.xyz")[0][:2]   # toluene
groups = nmr.get_nmr_equivalencies((atoms, coords))
print(groups["H"])
print(groups["C"])
```

```text
[[7, 8], [9, 11], [10], [12, 13, 14]]
[[0, 2], [1], [3, 5], [4], [6]]
```

Groups come from the point group of the heavy atoms and from the bond topology (freely rotating
methyl groups are merged): ortho, meta, para and methyl protons of toluene. The same analysis is
available as `frog-nmr-equiv files/tol.xyz`.

## 4. Spin simulation of a 1D spectrum

For coupled protons the spectrum is computed from the full spin Hamiltonian, which also handles
strongly coupled (second-order) systems:

```python
import numpy as np
from frogchem import nmr_sim

shifts_ppm = [7.28, 7.15, 6.90]
j_hz = np.array([[0.0, 8.2, 1.5],
                 [8.2, 0.0, 7.8],
                 [1.5, 7.8, 0.0]])

transitions = nmr_sim.simulate_quantum_transitions(shifts_ppm, j_hz, freq_mhz=400.0)
ppm, intensity = nmr_sim.simulate_1d_spectrum(transitions, freq_mhz=400.0, fwhm_hz=0.8)
print(len(transitions), "lines; strongest point at", f"{ppm[np.argmax(intensity)]:.3f} ppm")
```

```text
13 lines; strongest point at 7.150 ppm
```

`group_counts=[3, 2]` treats each entry as a group of equivalent spins (e.g. CH3 and CH2 of an
ethyl group), so the multiplets have the correct n+1 pattern and intensities.

---

## 5. The command line: `frog-nmr`

```bash
frog-nmr files/NMR_ORCA.out -n H
```

prints three tables for the 1H nuclei:

* **Chemical Shifts & Shieldings**: index (0-based), label, σ_iso, δ and anisotropy (ppm),
  e.g. `23 │ H24 │ H │ 25.708 │ 6.092 │ 4.965`.
* **J couplings**: a matrix for ≤ 15 atoms, otherwise the pairs with |J| ≥ 0.5 Hz, e.g.
  `H30 │ H31 │ -11.995 │ 1.778` (Hz, Å).
* **NOE Estimation**: pairs within 5 Å with relative NOE ∝ r⁻⁶ and a class
  (Strong < 2.5 Å, Medium < 3.5 Å, Weak beyond).

Main options:

* `-n` selects the nuclei. **Without `-n` the tables list every nucleus in the output** (only
  the simulations default to 1H).
* `--ref` sets the reference: a TMS output (`--ref tms.out`) or one value per nucleus
  (`--ref 1H=31.8,13C=185.0`). A single number (`--ref 31.8`) is applied to *every* nucleus
  shown, so use it together with `-n 1H`.
* `--shifts`, `--couplings`, `--distances` show one table only; `-i 23 24 25` restricts to those
  atoms (0-based).
* `--sim-1d --plot` simulates and saves the 1D spectrum (`NMR_ORCA_1d_1h.png` next to the input,
  or in `--out-dir`); `--cosy`, `--noesy`, `--hsqc`, `--hmbc` (with `-n 1H 13C` for the
  heteronuclear ones) add 2D maps; `--save-xy` and `--save-matrix` write the data.
* `-c checkpoint.json` Boltzmann-averages shifts, couplings and distances over a conformer
  ensemble (see [Ensemble analysis](ensemble.md)).

Example: `frog-nmr files/NMR_ORCA.out --sim-1d --plot --out-dir nmr_out` prints the
*Simulated 1H NMR Signals* table (the three methyl protons H37–H39 merged into one 3H signal at
4.248 ppm) and writes `nmr_out/NMR_ORCA_1d_1h.png`.

## 6. Other NMR commands

* `frog-nmr-equiv opt.xyz` — groups of equivalent atoms (Section 3).
* `frog-nics ring.xyz -i 0 1 2 3 4 5 -w` — adds a ghost atom at a ring centroid for a NICS
  aromaticity calculation; NICS(0) = −σ_iso of the ghost atom after an ORCA `! NMR` run.

Each command's page in the CLI reference (*NMR Spectroscopy*) has a step-by-step guide.
