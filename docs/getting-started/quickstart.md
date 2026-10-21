# Quickstart Guide

This page is a first session with FROG for people who run quantum-chemistry calculations only now and then.
You will check that a calculation finished as expected, get its free energy, read atomic charges and excited
states, and turn an energy difference into a selectivity. Every output below is real: it was produced from the
sample files in the `files/` folder of the repository.

## Before you start

* Install FROG ([Installation](installation.md)) and check that `frog` prints the list of tools.
* Have your **output files** at hand: ORCA `.out` or Gaussian `.log`/`.out` (gzipped files also work).
  frog never modifies them; new files are written next to them with a suffix (e.g. `opt_UV.png`).
* Every tool works the same way: `frog-<tool> FILE(S) [options]`. `frog-<tool> --help` lists the options,
  what the tool prints or writes, and examples. The page of each tool in the *Command-Line Interface* section
  explains in addition **what it is for, what you need, how to use it and what to expect**.
* Atom numbers are **0-based** everywhere: the first atom of the file is atom 0. This is the numbering ORCA
  prints (e.g. `0 C :` in the population analysis); Gaussian, GaussView and Avogadro start at 1, so subtract 1.

## Your first session

### 1. Did the optimisation reach a minimum, and what is its free energy?

You need an **optimisation + frequency** job (ORCA `! Opt Freq`, Gaussian `opt freq`).

```bash
frog-rrho opt.out
```

```text
│ File    │       E [Eh] │  E+ZPVE [Eh] │       H [Eh] │  TS [Eh] │       G [Eh] │ G-E [Eh] │ Imag Freq (cm⁻¹) │
│ opt.out │ -3265.187260 │ -3264.434095 │ -3264.382645 │ 0.126991 │ -3264.509635 │ 0.677625 │ -180.94          │
```

* All energies are in Hartree (1 Eh = 627.5 kcal/mol). **G** is the quasi-RRHO Gibbs free energy at 298.15 K and
  1 atm (change with `-T` and `-P`); *G−E* is the thermal correction you add to a better single-point energy.
* **Imag Freq** lists imaginary frequencies (printed as negative numbers). For a stable molecule this column must
  be empty; exactly one is expected only for a transition state. Here a −181 cm⁻¹ mode is present, so this
  structure is not a true minimum and should be re-optimised along that mode before its G is used.
* Pass several files (`frog-rrho conf_*/opt.out`) to get one row per file.

### 2. Atomic charges

```bash
frog-charges opt.out -i 0 1 2 --type all
```

```text
│ File    │ #0 El │ #0 Mulliken │ #0 Löwdin │ #1 El │ #1 Mulliken │ #1 Löwdin │ #2 El │ #2 Mulliken │ #2 Löwdin │
│ opt.out │ C     │      -0.448 │    -0.362 │ C     │      -0.407 │    -0.365 │ P     │       0.468 │     0.551 │
```

Charges are in units of the elementary charge and are taken from the **last** population analysis in the file
(the final geometry of an optimisation). Mulliken and Löwdin charges are different partitioning schemes:
compare trends between molecules computed the same way, not absolute values between schemes.

### 3. Excited states (TD-DFT)

You need a TD-DFT job (ORCA `%tddft nroots 10 end`; add `triplets true` for triplets and `dosoc true` for
spin–orbit coupling).

```bash
frog-tddft soc.out
```

```text
│ State │ Mult │ Energy (eV) │     f │ R (1e-40 cgs) │ <S**2> │ Population  │
│ S1    │    1 │       2.778 │ 0.080 │        20.696 │  0.000 │ H->L (0.99) │
│ T1    │    3 │       2.506 │ 0.000 │        -0.000 │  2.000 │ H->L (0.88) │

│ State │    T1 │
│ S0    │ 6.857 │
│ S1    │ 1.040 │
```

* States are labelled **S1, S2, … (singlets), T1, T2, … (triplets)**, numbered within each multiplicity;
  **S0** is the ground state. The same labels are used by `--dct-states` and `--soc-states`.
* *f* is the oscillator strength (how bright the absorption is; triplets are dark, f = 0), *R* the rotatory
  strength used for ECD, *Population* the dominant orbital excitation (H = HOMO, L = LUMO) with its weight.
* The second table is the spin–orbit coupling ⟨T|H_SO|S⟩ in cm⁻¹.
* Add `--uv` and/or `--ecd` to save the broadened spectra as images.

### 4. From an energy difference to a selectivity

```bash
frog-er er 1.5
```

```text
│ dG (kcal/mol) │  T (K) │ Major (%) │ Minor (%) │ ee (%) │
│          1.50 │ 298.15 │     92.63 │      7.37 │  85.27 │
```

A free-energy difference of 1.5 kcal/mol between the two competing transition states corresponds to a 92.6 : 7.4
ratio (85 % ee) at room temperature. `frog-er dg 95 5` does the reverse.

## Which tool do I need?

| I want to… | Tool |
|---|---|
| get G, H, S (and check for imaginary frequencies) | `frog-rrho` |
| turn ΔG into e.r./ee, or a measured ratio into ΔG | `frog-er` |
| draw a reaction energy profile | `frog-autoprofile` (from outputs), `frog-profile` (from a text/YAML file) |
| follow a NEB optimisation | `frog-plot-neb` |
| read atomic charges / spin populations | `frog-charges` / `frog-spin` |
| see which atoms an orbital sits on | `frog-mo` |
| make orbital, density or spin-density cubes | `frog-density` |
| add or subtract cubes (e.g. density differences) | `frog-cube` |
| UV-Vis / ECD spectra, excited states, charge transfer | `frog-tddft` |
| the colour of an emission/absorption spectrum | `frog-cie` |
| NBO analysis | `frog-nbo` |
| conceptual-DFT indices (μ, η, ω) | `frog-cdft` |
| NMR shifts and simulated spectra | `frog-nmr` (equivalent atoms: `frog-nmr-equiv`; NICS: `frog-nics`) |
| distances, angles, dihedrals | `frog-internals` |
| point group / symmetry | `frog-sym` |
| compare conformers (RMSD), sort them, find stereocentres | `frog-rmsd`, `frog-sort`, `frog-sort-ez`, `frog-stereocenters` |
| buried volume and steric maps of a ligand | `frog-steric` |
| solvent-accessible / excluded surfaces | `frog-solvent` |
| final geometries as XYZ, a Supporting-Information file | `frog-xyz`, `frog-si` |
| replace a substituent in 3D | `frog-substitute` (library: `frog-sub-db`) |

---

## Using FROG from Python

Everything the tools do is also available as a library, e.g. in a Jupyter notebook.
The examples below run on the sample files of the repository.

### Read an output and compute the free energy

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

```text
Level of theory: r2scan-3c d4/def2-mTZVPP
Final energy:    -3265.187260 Eh
Formula:         C45H42F3IO2P2Ru
ZPVE:            0.753166 Eh
H(298.15 K):     -3264.382645 Eh
G(298.15 K):     -3264.509635 Eh
```

The numbers are identical to those of `frog-rrho opt.out` above. `frog.read_orca()` and `frog.read_gaussian()`
force the format; `calc.to_ase()` and `calc.to_rdkit()` hand the structure to ASE or RDKit.
Parsing is lazy: the file is read the first time you access an attribute.

### Exact simulation of a coupled spin pair

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

```text
 7.1890 ppm   intensity 0.804
 7.2090 ppm   intensity 1.196
 7.2910 ppm   intensity 1.196
 7.3110 ppm   intensity 0.804
```

`solve_spin_cluster` returns `(frequency in Hz, intensity)` pairs. The inner lines are taller than the outer ones
(the "roof effect"), which a first-order n+1 rule cannot reproduce.

## Next steps

* The [User Guide](../user-guide/parsing.md) pages walk through complete workflows (thermochemistry,
  conformer ensembles, NMR, spectroscopy, geometry, substitution).
* The *Command-Line Interface* pages document every tool and every option.
