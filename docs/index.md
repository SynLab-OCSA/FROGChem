# FROG — Find, Refine, Organize and Graph

**FROG** turns the output of quantum-chemistry programs (**ORCA** 4–6, **Gaussian** 09/16) into the numbers and
pictures you need for a paper or a thesis: free energies and selectivities, Boltzmann-averaged conformer
ensembles, NMR shifts and simulated spectra, UV-Vis/ECD spectra, orbitals and densities, symmetry, steric maps,
reaction profiles and 3D substituent replacement.

ORCA outputs are supported by every tool. For Gaussian, FROG reads geometries, energies, frequencies
(thermochemistry), TD-DFT states and NPA charges; orbital, density, Mulliken/Löwdin, NBO and Hessian analyses
need ORCA outputs.

The name says what it does: **F**ind the data in the output files, **R**efine them (quasi-RRHO
thermochemistry, Boltzmann averages, spectrum simulation), **O**rganize structures and ensembles, and
**G**raph the results. In Python the package is imported as `frogchem`.

It is both a Python library and a set of **command-line tools** (`frog-rrho`, `frog-nmr`, `frog-tddft`, …).
You do not need to write Python to use it: most tasks are one command on one or more output files.

## Where to start

* **New to the tools?** Read the [Quickstart](getting-started/quickstart.md), then open the page of the command you
  need in the *Command-line tools* section. Every command page explains, in plain language,
  **what the command is for**, **what you need before running it** (which calculation and keywords),
  **how to use it step by step** and **what to expect** in the output (units, typical values, warnings).
* **Not sure which command you need?** See the [command reference](cli/reference.md): one line per tool.
* **Want the equations?** The *Scientific Theory* pages give the formulas behind thermochemistry, TD-DFT
  spectra and NMR simulation.

Two conventions hold everywhere: **atom numbers are 0-based** (the first atom of the file is atom 0), and
TD-DFT excited states are labelled **S1, S2, …, T1, T2, …, D1, …** (numbered within each spin multiplicity, with
**S0** the ground state).

---

## Quick examples

### 1. Free energy from a frequency calculation (terminal)

```bash
frog-rrho opt.out
```

Prints one row per file with E, E+ZPVE, H, T·S, G and G−E (all in Hartree, quasi-RRHO, 298.15 K, 1 atm) and lists
imaginary frequencies, if any. For the ORCA r²SCAN-3c optimisation + frequency job in `files/opt.out`:

```text
│ File    │       E [Eh] │  E+ZPVE [Eh] │       H [Eh] │  TS [Eh] │       G [Eh] │ G-E [Eh] │ Imag Freq (cm⁻¹) │
│ opt.out │ -3265.187260 │ -3264.434095 │ -3264.382645 │ 0.126991 │ -3264.509635 │ 0.677625 │ -180.94          │
```

An entry under *Imag Freq* means the structure is **not** a minimum (one imaginary mode is expected only for a
transition state).

### 2. The same in Python

```python
import frogchem as frog
from frogchem import thermo

calc = frog.read("opt.out")          # ORCA or Gaussian, detected automatically
print("Level of theory:", calc.theory)
print(f"Final energy:    {calc.final_energy:.6f} Eh")
print("Formula:        ", calc.to_ase().get_chemical_formula())

G, zpve, H, S = thermo.free_gibbs_energy(
    SCF=calc.final_energy, T=298.15, freq=calc.freq,
    mw=calc.mw, B=calc.B, m=calc.mult,
)
print(f"ZPVE:            {zpve:.6f} Eh")
print(f"H(298.15 K):     {H:.6f} Eh")
print(f"G(298.15 K):     {G:.6f} Eh")
```

Output:

```text
Level of theory: r2scan-3c d4/def2-mTZVPP
Final energy:    -3265.187260 Eh
Formula:         C45H42F3IO2P2Ru
ZPVE:            0.753166 Eh
H(298.15 K):     -3264.382645 Eh
G(298.15 K):     -3264.509635 Eh
```

### 3. A strongly coupled NMR spin pair

```python
import numpy as np
from frogchem import nmr_sim

# Two coupled protons 0.10 ppm apart at 400 MHz, J = 8 Hz (an "AB" system)
nu = np.array([7.30, 7.20]) * 400.0          # shifts in Hz
J = np.array([[0.0, 8.0],
              [8.0, 0.0]])
for freq, intensity in sorted(nmr_sim.solve_spin_cluster(nu, J)):
    print(f"{freq / 400.0:7.4f} ppm   intensity {intensity:.3f}")
```

Output — the "roof effect" (inner lines taller than outer ones) of the exact quantum solution:

```text
 7.1890 ppm   intensity 0.804
 7.2090 ppm   intensity 1.196
 7.2910 ppm   intensity 1.196
 7.3110 ppm   intensity 0.804
```

---

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
getting-started/architecture
```

```{toctree}
:maxdepth: 2
:caption: User Guide

user-guide/parsing
user-guide/thermo
user-guide/ensemble
user-guide/nmr
user-guide/reaction_profiles
user-guide/spectroscopy
user-guide/geometry
user-guide/substituter
user-guide/density_cubes
```

```{toctree}
:maxdepth: 2
:caption: Scientific Theory

theory/thermo
theory/tddft
theory/nmr
```

```{toctree}
:maxdepth: 1
:caption: Command-line tools
:glob:

cli/reference
cli/conventions
cli/frog-*
```

```{toctree}
:maxdepth: 2
:caption: Python API Reference

api/core
api/thermo
api/ensemble
api/nmr
api/reaction_plotter
api/substituter
api/geometry
api/io
```
