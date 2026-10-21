The base molecule is read with ASE. The fragments come from the library in
`frogchem/substituter/data` (see [frog-sub-db](frog-sub-db.md)) or from `--frag`. Each fragment is an XYZ
file with a pair of 0-based indices $(j_\mathrm{del}, j_\mathrm{att})$. Each site on the molecule is a pair
$(i_\mathrm{rep}, i_\mathrm{anc})$. `--scope` is resolved in this order: `all`, preset name,
comma-separated list, category, tag. With `--frag`, the fragment is first compared with every library
entry: same formula and the same sorted eigenvalues of the bond-graph adjacency matrix (tolerance $10^{-4}$).
If no entry matches, the fragment is **written into the library**.

For every molecule × fragment, the sites are processed one after the other (`frogchem.substituter.attach_fragment`):

1. **Orientation.** The fragment is translated so that $j_\mathrm{att}$ is at the origin. It is then rotated
   so that the vector $j_\mathrm{att}\to j_\mathrm{del}$ is anti-parallel to $\hat u = (\mathbf r_\mathrm{rep} -
   \mathbf r_\mathrm{anc})/|\mathbf r_\mathrm{rep} - \mathbf r_\mathrm{anc}|$. This makes the fragment body point
   away from the scaffold.
2. **Deletion and placement.** Atoms $j_\mathrm{del}$ and $i_\mathrm{rep}$ are deleted. The attach atom is placed at
   $\mathbf r_\mathrm{anc} + d\,\hat u$. Here $d$ comes from a table of standard single-bond lengths (e.g. C–C 1.54 Å,
   C–F 1.35 Å). Pairs that are not in the table use the sum of the covalent radii.
3. **Dihedral scan.** The fragment is rotated about $\hat u$ in `--rotations` equal steps (36, i.e. 10°).
   Each pose gets a score, compared in this order: the number of contacts with
   $r_{ij} < c\,(R_i+R_j)$ (with $c$ = `--clash`, 0.8, and $R$ the covalent radii), then a soft repulsion
   $\sum \max\!\left(0, \tfrac{R_i+R_j}{\max(r_{ij},0.2)} - 0.75\right)^4$, then the shortest contact. The scan
   stops early at a pose with no contacts and repulsion below $10^{-4}$. If every pose has contacts, the best one is still used.
4. **Relaxation** (skipped with `--no-opt`). The fragment atoms are always pre-relaxed with RDKit
   (bonds from `DetermineBonds`, MMFF94 or UFF as a fallback, 500 iterations). All other atoms are held with
   `AddFixedPoint`. With `--method aimnet`/`mace`, ASE (`--optimizer`, default LBFGS) then continues for at most 100
   steps, until $f_\mathrm{max} = 0.05$ eV/Å. The scaffold is held by `FixAtoms`. A step that changes the bond
   graph (bonding distance $R_i+R_j$ + `--buffer`, 0.3 Å) is undone, and the run stops with
   *Optimization incomplete*.

The deleted atom $i_\mathrm{rep}$ is removed, so the atoms after it move down by one index. Fragment atoms are
appended at the end. Later sites are still found through their original indices. *Core RMSD* is the RMSD of the
frozen atoms, in Å, against their positions before the relaxation.
