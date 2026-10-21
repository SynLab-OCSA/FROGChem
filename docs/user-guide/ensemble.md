# Conformer Ensembles & Boltzmann Averaging

A flexible molecule exists as a mixture of conformers. Any property you compare with experiment
(NMR shifts, charges, distances, spectra, thermochemistry) should therefore be an average over
the conformers, each weighted by its **Boltzmann population** (the fraction of molecules in that
conformer at temperature T, from the relative free energies: ~1.4 kcal/mol higher in G means
about 10 times less populated at room temperature).

FROG does this in two steps: the populations come from a **checkpoint** file, and every
ensemble-aware command applies them to its per-conformer results when given `-c`.

---

## 1. Populations from free energies

The simplest route is `frog-rrho` on the frequency outputs of all conformers:

```bash
frog-rrho conf_*/opt.out -b --sort
```

adds a *Boltz %* column computed from G (see [Thermochemistry](thermo.md)). From Python,
`frogchem.thermo.calculate_boltzmann(energies_in_Eh, T)` returns the same fractions.

## 2. Checkpoint files

`-c PATH` accepts three formats.

**CSV / TSV** — conformer number and population (%); a header line is skipped:

```text
conf,pop
1,75
2,25
```

**Flat JSON** — `{"1": 60.0, "2": 40.0}`.

**Hierarchical checkpoint JSON** (as written by the EnAn / ensemble_analyzer workflow), with
energies per conformer and per *protocol* (a level of theory):

```json
{
  "1": {"energies": {"1": {"E": -100.0000, "G": -99.9000},
                     "2": {"E": -100.0100, "G": -99.9100}}},
  "2": {"energies": {"1": {"E": -100.0010, "G": -99.9012},
                     "2": {"E": -100.0110, "G": -99.9095}}}
}
```

From this format the populations are recomputed on the fly: by default from G (`-e E`, `EZPVE`,
`H` to change), at `-T` (298.15 K), using the highest-numbered protocol unless
`--read-boltzmann PROTOCOL` selects another. If some conformers lack the requested energy they are
left out with a warning (energy terms are never mixed); if the records contain frequencies and
rotational constants (`Freq`, `B_vec`), H and G are recomputed with quasi-RRHO. Stored `Pop`
values are used only when no energies are available.

```python
from frogchem.ensemble import load_checkpoint_populations

def show(pops):
    print({conf: round(p, 2) for conf, p in pops.items()})

show(load_checkpoint_populations("pops.csv"))
show(load_checkpoint_populations("checkpoint.json"))               # G, latest protocol (2)
show(load_checkpoint_populations("checkpoint.json", protocol=1))   # G of protocol 1
show(load_checkpoint_populations("checkpoint.json", energy_term="E"))
```

```text
{1: 75.0, 2: 25.0}
{1: 62.94, 2: 37.06}
{1: 21.91, 2: 78.09}
{1: 25.75, 2: 74.25}
```

(`pops.csv` and `checkpoint.json` are the two examples above; populations in %).

## 3. Matching files to conformers

Each input file is matched to a conformer number taken from its path:

```python
from frogchem.ensemble import extract_conf_and_protocol

print(extract_conf_and_protocol("conf_1/protocol_2/opt.out"))
print(extract_conf_and_protocol("3_p2_orca.out"))
```

```text
(1, 2)
(3, 2)
```

Recognised patterns: folders `conf_<N>/protocol_<P>/`, file names `conf_<N>_p<P>`, `<N>_p<P>`,
`conf<N>`. If no number is found, files are numbered 1, 2, 3… in the order given. A file whose
conformer has no population gets weight 0: always check the *Ensemble Weights* table.

## 4. Ensemble-aware commands

`frog-rrho`, `frog-nmr`, `frog-charges`, `frog-nbo`, `frog-internals`, `frog-rmsd`
and `frog-tddft` accept the same options:

| Option | Meaning |
|---|---|
| `-c PATH` | checkpoint with the populations |
| `--read-boltzmann PROTOCOL` | protocol whose energies/populations are used |
| `-e {E,EZPVE,H,G}` | energy used for on-the-fly populations (default G) |
| `-T K` | temperature (default 298.15 K) |
| `--individual` | also show the per-conformer results |

For example, with the CSV above and two copies of `files/test.out` named `conf_1.out` and
`conf_2.out`:

```bash
frog-rrho conf_1.out conf_2.out -c pops.csv
```

```text
│ Conformer │ Protocol │ File       │ Boltzmann Pop (%) │ Normalized Weight (%) │
│ conf_1    │ N/A      │ conf_1.out │             75.00 │                 75.00 │
│ conf_2    │ N/A      │ conf_2.out │             25.00 │                 25.00 │
```

followed by the population-weighted E, E+ZPVE, H, TS, G and G−E. *Boltzmann Pop* is the
population read from the checkpoint; *Normalized Weight* is the same renormalised over the files
actually given (they differ when some conformers are missing).

## 5. Related geometry tools

Preparing an ensemble often also needs the geometry commands below; their options and examples
are in the CLI reference (*Conformer & Geometry*):

* `frog-rmsd` — compare structures with a reference (duplicates have RMSD ≈ 0).
* `frog-sort`, `frog-sort-ez` — sort conformers by dihedral-angle signs or into E/Z classes.
* `frog-sym` — point group and alignment to the principal axes.
* `frog-mirror` — mirror image (enantiomer) of every structure.
* `frog-split` — split a multi-XYZ file into one folder per structure, or merge files.
