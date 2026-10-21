# Quasi-RRHO Thermochemistry

The free energy G printed by a frequency calculation is dominated, for flexible molecules, by
the **low-frequency vibrations** (below ~100 cm⁻¹: methyl rotations, ring flips, floppy side
chains). In the harmonic-oscillator model their entropy is far too large, so G is too low and
depends on tiny numerical details. `frogchem.thermo` recomputes H, S and G with two standard
*quasi-RRHO* corrections:

1. **Grimme's interpolation** (default): each low mode is blended into a free rotor
   (*Chem. Eur. J.* 2012, 18, 9955).
2. **Truhlar's cut-off**: every frequency below the cut-off is raised to it
   (*J. Phys. Chem. B* 2011, 115, 14556).

The formulas are in the [theory page](../theory/thermo.md). You need the output of a frequency
calculation (ORCA `! Opt Freq`, Gaussian `opt freq`). The examples below use `files/test.out`
from the repository (an ORCA r²SCAN-3c frequency job of a C1 molecule); run them from the
repository root.

---

## 1. From the command line

```bash
frog-rrho files/test.out
```

prints one row per file (energies in hartree, Eh; 1 Eh = 627.5 kcal/mol):

```text
│ File           │      E [Eh] │ E+ZPVE [Eh] │      H [Eh] │  TS [Eh] │      G [Eh] │ G-E [Eh] │ Imag Freq (cm⁻¹) │
│ files/test.out │ -800.754157 │ -800.608362 │ -800.598179 │ 0.044648 │ -800.642826 │ 0.111330 │ -                │
```

These are the same numbers ORCA prints (*Final Gibbs free energy* −800.64282616 Eh,
*G-E(el)* 0.11133044 Eh), because ORCA also uses Grimme's treatment with a 100 cm⁻¹ cut-off.
The differences appear when you change the conditions:

| Command | G [Eh] |
|---|---|
| `frog-rrho files/test.out` | −800.642826 |
| `frog-rrho files/test.out -T 273.15` | −800.639195 |
| `frog-rrho files/test.out -m truhlar` | −800.643094 |

Useful options (see `frog-rrho --help` and the CLI reference for all of them):

* `-T` temperature (K), `-P` pressure (kPa), `-s` rotational symmetry number σ (default 1;
  ORCA uses the σ of the point group it detects, so for a symmetric molecule give the same σ to
  get the same G).
* `-b` Boltzmann populations from G and `--sort` to order the files by G.
* `--sp-path SP_DIR`: for each `conf/opt/opt.out` the only `.out`/`.log` in `conf/SP_DIR/` is read
  as a single point; then E = E_SP and G = E_SP + (G − E)_freq.
* `-c checkpoint.json`: ensemble average with populations from a checkpoint
  (see [Ensemble analysis](ensemble.md)).

Two behaviours to know:

* **Linear molecules** are detected automatically (fewer than three non-zero rotational
  constants); `--linear` forces the linear treatment.
* If the rotational constants cannot be read, a `RuntimeWarning` says
  `frequencies given but no rotational constants (B)`: the rotational entropy is then zero and G is
  wrong by several kcal/mol (≈ 9 kcal/mol for `test.out`). Check that the output is complete.

---

## 2. From Python

```python
import frogchem as frog
from frogchem import thermo

calc = frog.read("files/test.out")

G, zpve, H, S = thermo.free_gibbs_energy(
    SCF=calc.final_energy,      # electronic energy [Eh]
    T=298.15,                   # K
    freq=calc.freq,             # cm-1; imaginary (negative) values are ignored
    mw=calc.mw,                 # molecular mass [g/mol]
    B=calc.B,                   # rotational constants [cm-1]
    m=int(calc.mult),           # spin multiplicity
    P=101.325,                  # kPa
    model="grimme",             # or "truhlar"
    cut_off=100,                # cm-1
    alpha=4,                    # damping exponent (Grimme)
)

print(f"ZPVE {zpve:.6f} Eh")
print(f"H    {H:.6f} Eh")
print(f"TS   {298.15 * S:.6f} Eh")
print(f"G    {G:.6f} Eh")
```

```text
ZPVE 0.145794 Eh
H    -800.598179 Eh
TS   0.044648 Eh
G    -800.642826 Eh
```

`S` is returned in Eh/K, so `T * S` is in hartree.

---

## 3. Boltzmann populations and selectivity

```python
import numpy as np
from frogchem import thermo

energies = np.array([-450.2104, -450.2089, -450.2075])   # G of three conformers [Eh]
populations = thermo.calculate_boltzmann(energies, T=298.15)
for i, pop in enumerate(populations, start=1):
    print(f"Conformer {i}: {pop * 100:.2f}%")
```

```text
Conformer 1: 79.96%
Conformer 2: 16.33%
Conformer 3: 3.71%
```

From a free-energy difference (kcal/mol) to a ratio, and back:

```python
from frogchem import thermo

major, minor, ee = thermo.calc_er(1.42, 298.15)
print(f"{major * 100:.2f} : {minor * 100:.2f}, ee {ee:.2f} %")

print(f"{thermo.calc_dg(95, 5, 298.15):.3f} kcal/mol")
```

```text
91.66 : 8.34, ee 83.31 %
1.745 kcal/mol
```

The same from the command line: `frog-er er 1.42` and `frog-er dg 95 5`
(add `-T 195` for a reaction at −78 °C).
