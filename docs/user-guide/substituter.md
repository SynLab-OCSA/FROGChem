# Fragment Substitution

`frogchem.substituter` builds derivatives of a molecule automatically: it removes one atom (usually
an H), attaches a substituent in its place, rotates the substituent to avoid clashes and relaxes
**only the new fragment**, keeping the original scaffold frozen. Use it to prepare starting
geometries for a substituent screen; the results are force-field (or ML-potential) structures and
must still be optimised with your QM method.

## How it works

1. The **target site** is a pair of 0-based atom indices of the base molecule: the atom to
   replace and the anchor atom it is bonded to.
2. The **fragment** comes from the library (`frog-sub-db list`) or from your own XYZ file; it
   also has a pair of atoms: the one deleted on attachment and the one that bonds to the anchor.
3. The fragment is placed along the old bond, the bond length is set from atomic radii, and the
   fragment is rotated about the new bond (36 angles by default) to find a clash-free orientation.
4. The fragment is relaxed with the scaffold frozen: RDKit force field by default; AIMNet2
   (`--method aimnet`, installed with FROG, main-group elements only) or MACE (`--method mace`,
   needs `pip install mace-torch`) as alternatives. `--method xtb` is not implemented yet and falls
   back to RDKit with a warning.

---

## 1. Command line

Find the site in a viewer. In `files/tol.xyz` (toluene) the para H is atom 10, bonded to the
para carbon, atom 4:

```bash
frog-substitute --mol files/tol.xyz --mol-pairs 10 4 --db-frag phenyl --out-dir subs
```

```text
│ Molecule │ Fragment │ Status  │ Core RMSD (Å) │ Atoms │ Output File    │
│ tol.xyz  │ phenyl   │ SUCCESS │        0.0000 │    25 │ tol_phenyl.xyz │
```

The result is `subs/tol_phenyl.xyz` (without `--out-dir` it is written next to the input).
Core RMSD 0 Å means the original atoms did not move.

Screening several fragments:

```bash
frog-substitute --mol files/tol.xyz --mol-pairs 10 4 --scope halogens --out-dir subs --combined-xyz subs/all.xyz
```

writes `tol_fluoro.xyz`, `tol_chloro.xyz`, `tol_bromo.xyz`, `tol_iodo.xyz` (15 atoms each) and
all four in `subs/all.xyz`. Other options: `--db-frags methyl ethyl` (a list), several sites at
once (`--mol-pairs 10 4 7 0`), `--no-opt` (placement only), `--gui-pick` (choose the atoms in a
3D window). A custom fragment, `--frag OMe.xyz --frag-pair 4 0`, is **added to the library**
unless an identical one is already there.

## 2. The fragment library: `frog-sub-db`

```bash
frog-sub-db list --category halogens
frog-sub-db scopes
```

`list` shows name, category, delete/attach indices, tags and description (70 fragments in the
current library, e.g. `fluoro`: delete 0, attach 1); `scopes` shows the presets for `--scope`
(`halogens` 4, `alkyls` 12, `aryls` 17, `ewg` 14, ...). New fragments:

```bash
frog-sub-db add-smiles Bpin "*B1OC(C)(C)C(C)(C)O1" --category boron --tags ewg
frog-sub-db add tolyl tolyl.xyz --indices 7 0
```

`*` in the SMILES marks the attachment point. `add`, `add-smiles`, `edit` and `remove` change the
library **in place** inside the installed package (`frogchem/substituter/data`). Global options go
before the subcommand: `frog-sub-db -q list`.

## 3. Python

The command line uses these building blocks:

```python
from ase.io import read
from frogchem.substituter import FragmentDatabase, TopologicalOptimizer, attach_fragment

db = FragmentDatabase()
frag_path, frag_pair = db.get_fragment("phenyl")     # (xyz path, (delete, attach))

mol = read("files/tol.xyz")
new, added, bond = attach_fragment(mol, read(frag_path), (10, 4), frag_pair)

opt = TopologicalOptimizer(new, None)                # None = RDKit force field
opt.prepare(active_indices=added, new_bond=bond)     # scaffold frozen by default
converged = opt.run(optimizer_type="rdkit")

print(new.get_chemical_formula(), len(new), converged, f"{opt.get_core_rmsd():.4f}")
print(f"new C-C bond: {new.get_distance(*bond):.3f} Å")
new.write("tol_phenyl.xyz")
```

```text
C13H12 25 True 0.0000
new C-C bond: 1.475 Å
```

The new aryl–aryl bond of 1.475 Å is close to the 1.48–1.49 Å of biphenyl, as expected for a
force-field guess.
