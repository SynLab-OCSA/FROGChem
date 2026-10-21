# Reading Quantum-Chemistry Outputs

Every FROG command starts by reading an ORCA or Gaussian output file. This page shows what is read,
which keywords the calculation needs, and how to use the same parser from Python. The examples use the
sample files in `files/` of the repository; run them from the repository root.

---

## 1. What can be read, and which keywords you need

| You want | ORCA input | Gaussian | Command |
|---|---|---|---|
| Final energy, geometry | any job | any job | `frog-xyz`, `frog-si` |
| Thermochemistry, imaginary frequencies | `! Opt Freq` (`! OptTS Freq` for a TS) | `opt freq` | `frog-si`, `frog-rrho` |
| Which atoms move in a vibration | `! Freq` | not supported | `frog-hessian` |
| HOMO/LUMO energies, CDFT indices | any SCF job | any SCF job | `frog-cdft` |
| Mulliken / Löwdin charges | default population analysis (not `! NoPop`) | not supported | `frog-charges` |
| MO composition, orbital/density cubes | `! PrintBasis PrintMOs` | not supported | `frog-mo`, `frog-density` |
| NBO: NPA charges, E(2), bonds | `! NBO` (+ `%nbo NBOKEYLIST = "$NBO BNDIDX $END" end` for Wiberg indices) | not supported | `frog-nbo` |
| Excited states, UV/ECD, SOC | `%tddft nroots 10 end` (`triplets true`, `dosoc true`) | `td=(nstates=10)` | `frog-tddft` |

The ORCA parser is the complete one; the Gaussian parser reads geometry, energy, frequencies, TD-DFT states
and NPA charges. Files can be gzipped (`job.out.gz`). Atom and MO numbers are **0-based** everywhere in
FROG (first atom = 0, as in ORCA); NBO labels and NBO numbers stay 1-based as printed by NBO.

Each command is described in detail (what it is for, what you need, what to expect) in
its own page under [Command-line tools](../cli/reference.md).

---

## 2. Quick start from the command line

```bash
frog-xyz files/opt.out --out-dir geoms         # final geometry -> geoms/opt_final.xyz
frog-cdft files/opt.out files/test.out          # HOMO, LUMO, gap, μ, η, ω for both files
frog-hessian files/opt.out                      # atoms moving in the imaginary mode of a TS
frog-si files/opt.out files/test.out -o si.md   # SI report with energies and coordinates
```

`frog-cdft` prints (r²SCAN-3c; the second file is an anion):

```text
│ File     │ HOMO (eV) │ LUMO (eV) │ Gap (eV) │ μ (eV) │ η (eV) │ ω (eV) │ S (1/eV) │
│ opt.out  │    -5.158 │    -3.065 │    2.093 │ -4.111 │  2.093 │  4.038 │    0.478 │
│ test.out │    -3.995 │    -0.725 │    3.270 │ -2.360 │  3.270 │  0.852 │    0.306 │
```

Commands that write files put them next to the input unless you give `--out-dir` or `-o`.

---

## 3. Reading a file from Python

`frogchem.read()` detects the program from the file header and returns a calculation object. Parsing is
lazy: the file is read the first time you access an attribute.

```python
import frogchem as frog

calc = frog.read("files/test.out")          # also .out.gz; frog.read_orca / frog.read_gaussian force the format
print(type(calc).__name__)
print(f"Theory:       {calc.theory}")
print(f"Charge/mult:  {calc.charge} / {calc.mult}")
print(f"Final energy: {calc.final_energy:.6f} Eh")
```

```text
Calculation
Theory:       r2scan-3c/def2-mTZVPP
Charge/mult:  -1 / 1
Final energy: -800.754157 Eh
```

### Geometry and frequencies

```python
print(len(calc.atoms), calc.atoms[:4], calc.geom.shape)   # element symbols; coordinates in Å
freq = calc.freq                                          # cm⁻¹, includes the 6 zeros of ORCA
vib = freq[freq != 0]
print(len(vib), vib[:3], "imaginary:", (vib < 0).sum())
print(calc.B, round(calc.mw, 2))                          # rotational constants (cm⁻¹), mass (g/mol)
```

```text
19 ['C' 'C' 'C' 'C'] (19, 3)
51 [ 92.7  104.85 127.99] imaginary: 0
[0.07914  0.022233 0.017414] 162.23
```

The geometry is the **last** one in the file (the optimised structure, if the optimisation converged).

### Charges and frontier orbitals (ORCA)

```python
for i in (0, 1):
    m, l = calc.mulliken[i], calc.loewdin[i]          # {index: {'atom': 'C', 'charge': ...}}
    print(f"{m['atom']}{i}: Mulliken {m['charge']:+.3f}  Löwdin {l['charge']:+.3f}")

HARTREE_TO_EV = 27.211386245988
h = calc.homo                                          # 0-based index of the HOMO
print(f"HOMO #{h}: {calc.orbital_energies[h] * HARTREE_TO_EV:.3f} eV   "
      f"LUMO #{h + 1}: {calc.orbital_energies[h + 1] * HARTREE_TO_EV:.3f} eV")
```

```text
C0: Mulliken -0.373  Löwdin -0.196
C1: Mulliken +0.307  Löwdin -0.050
HOMO #42: -3.995 eV   LUMO #43: -0.725 eV
```

`calc.orbital_energies` is in Hartree. Mulliken and Löwdin charges can disagree even in sign (atom 1
above): compare trends within one method, not absolute values.

### NBO results

```python
nbo = frog.read("files/nbo.out").nbo                    # needs `! NBO` in the ORCA input
print(nbo.npa_charges[["Atom", "Index", "Charge"]].head(3))
```

```text
  Atom  Index   Charge
0    C      0 -0.25182
1    C      1 -0.05221
2    C      2 -0.43784
```

`nbo.perturbation_theory` (E(2) table), `nbo.nbo_details` and `nbo.wiberg_df` are pandas DataFrames
as well; `frog-nbo` prints them with filters.

### XYZ files and other formats

`frogchem.filetypes.load_calculation()` is what the commands use: it also accepts `.xyz` files (first
frame), returning an object with `atoms` and `geom` only.

```python
from frogchem.filetypes import load_calculation
mol = load_calculation("files/water.xyz")
print(list(mol.atoms), mol.geom.shape)
```

```text
['O', 'H', 'H'] (3, 3)
```

### Converting to ASE and RDKit

```python
atoms = calc.to_ase()                       # needs ase
print(atoms.get_chemical_formula())

from rdkit.Chem import rdMolDescriptors      # needs rdkit; bonds are guessed from the geometry
mol = calc.to_rdkit(charge=calc.charge)
print(rdMolDescriptors.CalcMolFormula(mol))
```

```text
C9H8NS
C9H8NS-
```

---

## 4. Common problems

* **An attribute is missing** (`AttributeError: Calculation has no attribute 'nbo'`): the calculation did
  not print that section; add the keyword from the table in section 1 and rerun.
* **Numbers are from the wrong geometry**: FROG takes the last block of the file. For a crashed
  optimisation that is the last step reached, not a converged structure.
* **Gaussian files**: only geometry, energy, frequencies, TD-DFT states and NPA charges are read; the
  commands that need more (charges, normal modes, MO coefficients, NBO) work with ORCA outputs only.
