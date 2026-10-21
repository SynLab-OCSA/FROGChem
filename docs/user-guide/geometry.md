# Geometry: conformers, ensembles, symmetry and sterics

This page walks through the geometry tools of FROG in the order you typically meet them in a project:
set up a conformer search, split its output into calculations, inspect and sort the ensemble, check
symmetry and stereochemistry, and quantify sterics around a metal. Each command has a full reference
(every flag, every output) in its own page under
[Command-line tools](../cli/reference.md); here the focus is on *when* to
use what and how to read the results.

All examples were run on files shipped in the repository folder `files/`: `ensemble.xyz` (18 CREST
conformers of a 168-atom Ag complex), `opt.out` (ORCA optimisation of a 96-atom Ru complex), `water.xyz`,
`methane.xyz`, `ethyne.xyz`, `tol.xyz` (toluene) and `nmr_iPr.xyz` (a chiral organic molecule). Numbers
quoted below come from those runs. Copy the files to a scratch folder before trying: the commands write
their output next to the input.

---

## 1. Conventions you need to know first

* **Atom numbers are 0-based** in every FROG command: the first atom of the file is atom 0. Viewers
  such as GaussView or Avogadro, and ORCA's own lists, usually count from 1 — subtract 1. Column headers
  repeat the element next to the index (e.g. `B Ag0-P50 (Å)`), so you can check that you picked the
  right atoms.
* **Conformer folders are 1-based**: `frog-split` writes `conf_1`, `conf_2`, ...; the ensemble tools
  read the conformer number from these folder names to match each file with its Boltzmann population.
* **Inputs**: every geometry command reads `.xyz` files (several structures per file allowed), ORCA
  outputs and Gaussian outputs. From ORCA/Gaussian outputs only the **final geometry** is used.
* **Outputs** are written next to the input file as `<stem>_<tag>.<ext>` (for example
  `ensemble_mirror.xyz`); `--out-dir DIR` puts them elsewhere. Tables go to the screen; `--format csv`
  or `--table-file results.csv` gives them in a spreadsheet-friendly form.

---

## 2. From a conformer search to individual calculations

### Constraining part of the molecule in CREST (`frog-crest`)

If part of the system must not move during the conformer search (a metal and its donor atoms, the core of
a transition state), write a CREST constraint file:

```bash
frog-crest ensemble.xyz -i 0 4 50 83 -o crest/constrain.inp
```

```text
$constrain
  atoms: 1 5 51 84
  force constant=0.5
  reference=ensemble_ref.xyz
$end
```

The atoms are converted to CREST's 1-based numbering (0 4 50 83 → 1 5 51 84). Because `ensemble.xyz`
holds 18 structures, the first one was written as the reference `crest/ensemble_ref.xyz` (with a warning).
Then run CREST from that folder (a sketch: add your usual method, solvent and thread options):

```bash
cd crest && crest ensemble_ref.xyz --cinp constrain.inp
```

### One folder per conformer (`frog-split`)

```bash
frog-split ensemble.xyz
```

creates `ensemble_split/conf_1/geom.xyz` ... `ensemble_split/conf_18/geom.xyz`, keeping the original
comment line (here the CREST energy) of each structure. Build your DFT inputs inside the `conf_<n>`
folders and keep their names: they carry the conformer number. Running the command again skips the
existing folders (*"Skipped 18 existing conf_<n> folders ... (use -f to overwrite)"*).

To look at the optimised structures together, merge them back:

```bash
frog-split --merge ensemble_split/conf_1/geom.xyz ensemble_split/conf_2/geom.xyz -o two.xyz
```

Note that a shell pattern such as `conf_*/geom.xyz` is expanded in alphabetical order
(`conf_1, conf_10, conf_11, ..., conf_2`); list the files explicitly when the order matters.

---

## 3. Inspecting an ensemble

### Is it one molecule or several? (`frog-fragments`)

```bash
frog-fragments ensemble.xyz
```

Every conformer gives `2 | 0:11 50:167 | 12:49`: the complex (atoms 0–11 and 50–167, ranges inclusive)
and a separate 38-atom molecule (atoms 12–49). Bonds are guessed from covalent radii plus a 0.3 Å buffer
(`-b`); `-w` writes each fragment to its own XYZ file. A conformer whose fragment count differs from the
others deserves a look.

### Measuring distances, angles and dihedrals (`frog-internals`)

```bash
frog-internals ensemble.xyz -i 0 50 -i 50 0 83 -i 50 0 4 3
```

Two indices give a distance (Å), three an angle, four a dihedral (degrees, −180° to +180°). For
conformer 0: Ag0–P50 = 2.619 Å, P50–Ag0–P83 = 89.575°, P50–Ag0–C4–N3 = −72.307°. Across the 18
conformers the dihedral goes from −72° to +80°, so it is a good handle to sort them.

### Sorting conformers by a dihedral (`frog-sort`, `frog-sort-ez`)

`frog-sort` uses the **sign** of one or more dihedrals (P = positive, N = negative):

```bash
frog-sort ensemble.xyz -i 50 0 4 3 -w
```

gives N = 10 and P = 8 conformers and, with `-w`, writes `ensemble_N.xyz` and `ensemble_P.xyz`. Choose a
dihedral whose values are clearly away from 0° and 180°: near those values the sign flips with tiny
changes.

`frog-sort-ez` asks instead whether a dihedral is near **0°** (Z, cis, s-cis) or near **180°** (E, trans,
s-trans):

```bash
frog-sort-ez ensemble.xyz -i 2 1 3 4
frog-sort-ez ensemble.xyz -i 2 1 3 4 --tolerance 60 --notation cis-trans -w
```

With the default 90° boundary this dihedral (−134° to −34°) gives Z = 8, E = 10; with a ±60° window
around 0° and 180°, cis = 5, trans = 9 and 4 conformers are flagged as *other*. The classification is
geometric: choose the two substituents with the higher CIP priority if you want "Z" to mean the true Z
descriptor.

### Stereocentres and enantiomers (`frog-stereocenters`, `frog-mirror`)

```bash
frog-stereocenters nmr_iPr.xyz
frog-mirror nmr_iPr.xyz
frog-stereocenters nmr_iPr_mirror.xyz
```

The first command finds C0 = S and C2 = R; the mirror image has C0 = R and C2 = S. Bonds are assigned
with RDKit, so give `--charge` for ions. **Metal complexes cannot be analysed**: on `ensemble.xyz` the
command stops with *"Atom 0 with atomic number 47 has no valences defined"*. `frog-mirror` itself works
on any structure, including the whole Ag ensemble (`ensemble_mirror.xyz`, 18 structures, comment lines
kept).

### How different are two structures? (`frog-rmsd`)

```bash
frog-rmsd ensemble.xyz
frog-rmsd ensemble.xyz --no-hydrogen -r xray.xyz     # sketch: compare with your own X-ray file
```

The λ-RMSD compares the eigenvalues of the interatomic-distance matrices, so it needs no alignment and no
common atom order. Relative to conformer 0 the values range from 0.504 Å (conformer 4) to 5.051 Å
(conformer 15). This is **not** the Kabsch RMSD printed by viewers or CREST (the scales differ), and
mirror images give exactly 0 (`nmr_iPr.xyz` vs `nmr_iPr_mirror.xyz`: 0.000 Å).

---

## 4. Boltzmann-weighted averages (`-c`)

`frog-internals` and `frog-rmsd` accept `-c CHECKPOINT` with Boltzmann populations: the
`checkpoint.json` of the ensemble workflow, a CSV, or a flat JSON mapping conformer number to population
(%). Conformers are matched by their `conf_<N>` folder, or — for structures in one multi-XYZ file — by
position: structure 0 ↔ conformer 1, structure 1 ↔ conformer 2, and so on.

```bash
echo '{"1": 40.0, "2": 25.0, "3": 15.0, "4": 10.0, "5": 10.0}' > pop.json
frog-internals ensemble.xyz -i 0 50 -i 50 0 4 3 -c pop.json
```

An *Ensemble Weights* table is printed first (here conformers 6–18 get 0 %: they are not in `pop.json`),
then the averages: ⟨Ag0–P50⟩ = 2.623 Å (the r⁻⁶ NOE average, 2.623 Å, is also given for distances) and
⟨P50–Ag0–C4–N3⟩ = −41.3° (dihedrals are averaged as angles). Add `--individual` to also see the
per-conformer table. `frog-rmsd ensemble.xyz -c pop.json` adds an *⟨Ensemble Average⟩* row (0.564 Å).
Always read the weights table: a conformer with an unexpected 0 % was not matched.

See [Conformer & Ensemble Analysis](ensemble.md) for the checkpoint formats.

---

## 5. Point group (`frog-sym`)

```bash
frog-sym water.xyz methane.xyz ethyne.xyz tol.xyz
```

| Molecule | 0.1 Å | 0.2 Å | 0.5 Å | 1.0 Å |
|---|---|---|---|---|
| water | C2v | C2v | D8h | Td |
| methane | Td | Td | Td | Td |
| ethyne | D_inf_h | D_inf_h | D_inf_h | D_inf_h |
| toluene | Cs | Cs | Cs | Cs |

* Read the group at the **small tolerances** (0.1–0.2 Å). Large tolerances accept big distortions and give
  nonsense (water as D8h or Td).
* The "RMSD (Å)" column is the largest displacement of any atom under the accepted operations
  (0.0000 for water, 0.0053 Å for toluene).
* All atoms, hydrogens included, are used by default: toluene is Cs because the methyl hydrogens break the
  ring's C2v symmetry. `--exclude-h` gives the heavy-atom framework (toluene: C2v). Do not use
  `--exclude-h` for molecules with a single heavy atom (water gives a meaningless Th).
* `-g C2v` checks one group by scanning tolerances 0.05–1.5 Å; only an **exact** match counts
  (`frog-sym tol.xyz -g C2v Cs --exclude-h`: C2v True at 0.05 Å, Cs False).
* `-w` writes the molecule in its principal-axes frame as `<stem>_std.xyz`.
* The *Rotor Properties* table gives the moments of inertia (amu·Å²): water 0.60 / 1.21 / 1.81 (asymmetric
  top), methane 3.19 × 3 (spherical top), ethyne 0.00 / 13.68 / 13.68 (linear).

---

## 6. Buried volume and steric maps (`frog-steric`)

The buried volume %V_bur is the percentage of a sphere (default radius 3.5 Å) around the metal that is
occupied by ligand atoms (Bondi radii × 1.17, hydrogens included: the SambVca model). Quadrants and
octants show where the bulk is.

```bash
frog-steric opt.out --center 36 -z 37 -x 3
```

```text
File     Center  Radius (Å)  %V_bur  NE (%)  NW (%)  SW (%)  SE (%)
opt.out  Ru36          3.50   96.03   98.17   96.37   92.05   97.51
```

With **all** ligands included the sphere is nearly full (96 % here). To describe a single ligand, exclude
every atom that does not belong to it with `-E` (indices one by one; `-E $(seq 12 49)` in a shell writes
a range); such values are much smaller, typically 25–50 %. For the Ag complex, excluding the separate
molecule found by `frog-fragments` lowers %V_bur from 93.67 % to 85.65 %:

```bash
frog-steric ensemble.xyz --center 0 -z 50 83 -E $(seq 12 49)
```

Only the first structure of a multi-XYZ file is analysed; use `frog-split` first to treat each conformer.

Steric maps (`-m`, plus `--png` for a PNG) need the `stericrender` and `xyzrender` packages (and the cairo
library for PNG). The map looks down the Z axis from +Z; to view a ligand from the metal side, put the
ligand on −Z: its donor atom(s) with `-z` and `--flip-z`. If `-m` fails, that file gets no %V_bur either —
rerun without `-m` to get the numbers.

---

## 7. Checking an Amber MD run (`frog-md`)

`frog-md` reads an Amber topology, trajectory and log and draws temperature, pressure, total energy and
the RMSD of a group of atoms in one figure (`analysis_summary.png`, plus `analysis_rmsd.csv` and
`analysis_thermo.csv`). It needs MDTraj. A sketch (no Amber files ship with the repository):

```bash
frog-md -p sys.prmtop -y prod.nc --mdout prod.out --mask "resname LIG"
```

Look for a temperature that fluctuates around the thermostat value, a flat total energy and an RMSD that
reaches a plateau; see `frog-md --help` for details.
