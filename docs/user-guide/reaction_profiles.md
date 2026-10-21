# Reaction Energy Profiles & NEB Pathways

Three commands draw reaction energy diagrams:

| Command | Input | Use it for |
|---|---|---|
| `frog-autoprofile` | QM outputs of the stationary points | a quick profile straight from the calculations |
| `frog-profile` | a small YAML/TXT/JSON file with relative energies | publication figures, several pathways, branches |
| `frog-plot-neb` | ORCA `.interp` file of an NEB run | checking a reaction-path search |

All energies must come from structures with the **same atoms** (same total composition and
charge) at the same level of theory; otherwise relative energies are meaningless.

---

## 1. Directly from the outputs: `frog-autoprofile`

```bash
frog-autoprofile R.out TS.out P.out -l R TS P -o profile.png
```

The files are given in reaction order. G (default; `--energy E`, `EZPVE` or `H` also possible) is
recomputed with Grimme's quasi-RRHO at `-T` (298.15 K), exactly as `frog-rrho` with default
settings, and converted to kcal/mol relative to the first file. The command prints a
*Reaction Profile* table (state, file, energy in Eh, relative energy in kcal/mol) and saves the
plot (`reaction_profile.png` by default). Files without frequencies fall back to E with a warning.
With `--steric-center 0` the buried volume %V_bur around atom 0 (3.5 Å sphere) is added to the
table and to the labels.

## 2. From an energy table: `frog-profile`

Write the relative energies (usually ΔG in kcal/mol) in a small input file. The three formats
below describe the same figure: a main path R → TS1 → INT → TS2 → P and a second path that
branches from INT. These inputs were run with `frog-profile`; each produces
`<stem>_profile.png` next to the input.

**YAML** (`pes.yaml`, recommended):

```yaml
title: "Model reaction"
units: "kcal/mol"
steps: [R, TS1, INT, TS2, P]
label_points: [TS1, TS2]
pathways:
  - name: "Path A"
    energies: [0.0, 18.4, -3.2, 12.1, -10.5]
  - name: "Path B"
    style: dashed
    branch_at: INT
    energies: [-3.2, 15.8, -12.0]
```

**TXT** (`pes.txt`): `;`-separated energies, one line per pathway; `@INT:` starts a branch at
the state named `INT` in the labels.

```text
--# PES
Path A: 0.0; 18.4; -3.2; 12.1; -10.5
Path B: @INT: -3.2; 15.8; -12.0

--# LABELS
R; TS1; INT; TS2; P

--# GRAPHS
title : Model reaction
labelpoint : TS1; TS2
```

**JSON** (`pes.json`, simplest form, one pathway, no labels):

```json
{"Path A": [0.0, 18.4, -3.2, 12.1, -10.5]}
```

Run:

```bash
frog-profile pes.yaml                                  # pes_profile.png next to pes.yaml
frog-profile pes.yaml -o figure1.svg --style presentation
frog-profile pes.yaml --out-dir figs --point-type bar --legend-barriers
```

`--style presentation` uses larger fonts and thicker lines, `--style straight` straight segments;
`--curviness 0` and `--point-type {hollow,bar,dot}` change the curve and the markers.
`frog-profile --interactive` builds the input file step by step in the terminal; `--gui` opens a
browser interface (needs the optional `streamlit` package).

Note: several inputs with the same stem (`pes.yaml`, `pes.txt`, `pes.json`) all write
`pes_profile.png`; the last one wins. Use different file names or `-o PREFIX`.

### From Python

```python
from frogchem.reaction_plotter import ReactionProfileAnalyzer

analyzer = ReactionProfileAnalyzer("pes.yaml", auto_plot=False)
print(list(analyzer.energy_data))
print(analyzer.energy_data["Path B"])

fig, ax = analyzer.plot(filename=None, style="default")
fig.savefig("pes_profile.png", dpi=300)
```

```text
['Path A', 'Path B']
[None, None, -3.2, 15.8, -12.0]
```

(`None` marks the states where a branch has no point.)

---

## 3. NEB paths: `frog-plot-neb`

An ORCA NEB job (`! NEB`, `! NEB-CI` or `! NEB-TS`) writes `<job>.interp`, with the energies of
the images at every iteration. Plot them with

```bash
frog-plot-neb myjob.interp -o neb.png
```

The first iteration is drawn in black, intermediate ones in light grey and the last one in red;
energies are in kcal/mol (converted from Eh). A converged NEB has the red images on a smooth
curve with one maximum, whose structure is the starting point for a TS optimisation. Several
`.interp` files can be given to compare runs on the same axes. Nothing is printed on the terminal.
