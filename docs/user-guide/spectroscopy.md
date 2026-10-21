# Excited States & Spectroscopy

A TD-DFT calculation gives the vertical excited states of a molecule. FROG turns them into a readable
table, UV-Vis and ECD spectra, spin-orbit couplings, a charge-transfer index, and CIE colour
coordinates. The examples use the sample files in `files/`; run them from the repository root.
Each command is described in detail in its own page under [Command-line tools](../cli/reference.md).

---

## 1. The calculation you need

ORCA, for absorption and ECD (rotatory strengths are printed automatically):

```text
! B3LYP def2-TZVP
%tddft
  nroots 20
end
```

* More roots = a wider spectrum: 10–30 roots cover the near-UV/visible of a typical organic molecule.
* `triplets true` adds the triplet states; `dosoc true` (inside `%tddft`) adds spin-orbit couplings.
* For the charge-transfer analysis (`--dct`) also add `! PrintBasis PrintMOs`.
* Gaussian: `td=(nstates=20)` is also read (energies, f, R and transitions).

---

## 2. Excited-state table and SOC

```bash
frog-tddft files/soc.out
```

```text
│ State │ Mult │ Energy (eV) │     f │ R (1e-40 cgs) │ <S**2> │ Population  │
│ S1    │    1 │       2.778 │ 0.080 │        20.696 │  0.000 │ H->L (0.99) │
│ T1    │    3 │       2.506 │ 0.000 │        -0.000 │  2.000 │ H->L (0.88) │

│ State │    T1 │
│ S0    │ 6.857 │
│ S1    │ 1.040 │
```

* States are labelled per multiplicity: **S1, S2… T1, T2… D1… Q1…**; S0 is the ground state. The same
  labels are used in the SOC matrix (cm⁻¹; rows singlets, columns triplets) and in `--dct-states`.
* **f** (oscillator strength): > 0.1 bright, < 0.01 dark. **R** (rotatory strength, 10⁻⁴⁰ cgs): its sign
  pattern is what you compare with an experimental ECD spectrum.
* *Population* lists the orbital transitions with weight above `-t` (default 0.20): `H->L (0.99)` = 99 %
  HOMO → LUMO. Use `frog-mo` to see what those orbitals are made of.

---

## 3. UV-Vis and ECD spectra

```bash
frog-tddft files/tddft_mos.out --uv --fwhm 0.3 --x-start 6 --x-end 12 --save-spectra --out-dir spectra
```

(`--ecd` works the same way; water is achiral, so its rotatory strengths are zero.) Each transition becomes a Gaussian band of full width at half maximum `--fwhm` (eV) and the curves are
plotted on a wavelength axis (energy on top), **normalised** to a maximum of 1. ECD bands are weighted by
the transition energy (Δε ∝ E·R). `--shift -0.2` moves all energies by −0.2 eV, a common empirical
correction (hybrid functionals typically overestimate excitation energies by 0.1–0.4 eV). With several
files you also get a superimposed plot; with `-c checkpoint.json` a Boltzmann-averaged spectrum.
`--save-spectra` writes the curves as CSV (`Energy_eV`, `Wavelength_nm`, one column per curve).

The same functions are available in Python:

```python
import numpy as np
import frogchem as frog
from frogchem.cli.tddft_parser import extract_peaks, gaussian_convolution, state_label

calc = frog.read("files/soc.out")
for mult, states in calc.tddft_states.items():          # {multiplicity: {root: {...}}}
    for n, st in states.items():
        print(f"{state_label(mult, n)}: {st['energy']:.3f} eV  {1239.84193 / st['energy']:.1f} nm  "
              f"f = {st['fosc']:.4f}  R = {st['R']:.2f}")

calc = frog.read("files/tddft_mos.out")                     # water, 10 singlets
x = np.linspace(6.0, 12.0, 601)                           # energy grid, eV
E, f = extract_peaks(calc.tddft_states, 'fosc', shift=0.0)
uv = gaussian_convolution(x, E, f, fwhm=0.3)
print(f"UV maximum at {x[np.argmax(uv)]:.2f} eV ({1239.84193 / x[np.argmax(uv)]:.1f} nm)")
```

```text
S1: 2.778 eV  446.3 nm  f = 0.0800  R = 20.70
T1: 2.506 eV  494.7 nm  f = 0.0000  R = -0.00
UV maximum at 9.70 eV (127.8 nm)
```

For ECD, multiply the rotatory strengths by their energies before broadening
(`frogchem.cli.tddft_parser.ecd_band_intensities(E, R)`).

---

## 4. Charge-transfer distance (D_CT)

For each state FROG builds the difference density Δρ = ρ(excited) − ρ(ground) on a grid, finds the
barycentres of the regions where density is gained and lost, and reports their distance Δr (the D_CT
index of Le Bahers, Adamo and Ciofini, *JCTC* 2011, 7, 2498):

```bash
frog-tddft files/tddft_mos.out --dct --dct-states S1 S2 --voxels 40 --chimerax --out-dir dct
```

```text
│ State │ Energy (eV) │ Wavelength (nm) │ Osc. Strength (f) │ Δr (Å)            │ Dipole Moment │
│ S1    │ 7.58        │ 163.6           │ 0.0337            │ 0.86 (Local Exc.) │ 0.7171        │
│ S2    │ 9.46        │ 131.1           │ 0.0000            │ 0.89 (Local Exc.) │ 0.7613        │
```

* Δr < 1.5 Å = local excitation, > 2 Å = charge transfer, in between = mixed (`--le-threshold`,
  `--ct-threshold`). *Dipole Moment* is q_CT·Δr in e·Å.
* One cube per state (`<stem>_S1_delta_rho.cube`) is written unless `--no-save-cube`; `--chimerax`
  writes a ChimeraX script with the two barycentres (blue +, red −).
* A difference-density cube can be analysed again directly:
  `frog-tddft files/1_p8_orca_S1_delta_rho.cube` gives Δr = 0.77 Å (Local Exc.).

---

## 5. Colour: CIE 1931 coordinates

`frog-cie` converts spectra in CSV files (the file names below are placeholders for your own spectra) (columns `wavelength` in nm and `intensity`, or names given
with `--wl`/`--int`) into CIE 1931 (x, y) chromaticity coordinates and draws them on the diagram. It needs
the optional package `colour-science`.

```bash
frog-cie green_emission.csv                                  # emission: shape of the spectrum only
frog-cie uv_absorbance.csv --spectrum-type absorption -o abs.png   # perceived colour of the sample (D65)
```

For a synthetic Gaussian emission band at 520 nm (σ = 20 nm) the result is x = 0.1468, y = 0.6902 (green);
for an absorbance of 1.0 at 450 nm (σ = 30 nm) with `--spectrum-type absorption` it is x = 0.4160,
y = 0.4748 (the sample looks yellow, the complement of the absorbed blue).
