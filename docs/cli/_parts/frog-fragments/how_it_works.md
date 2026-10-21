Fragments are the connected components of a distance-based bond graph
(`frogchem.graph_theory.detect_fragments_from_geom`).

1. **Read the structures.** `load_structures` returns every structure of an XYZ file (each one is
   analysed independently) or the geometry of an ORCA/Gaussian output.
2. **Radii.** Each element gets its covalent radius $r_i$ from the periodic table
   (`frogchem.chemistry.get_atomic_radius`).
3. **Bond graph.** All interatomic distances $d_{ij}$ are computed at once (`scipy` `cdist`), and atoms
   $i$ and $j$ are bonded when

   $$
   d_{ij} < r_i + r_j + b ,
   $$

   with the buffer $b$ = `-b` (default 0.3 Å). Bond orders and charges play no role.
4. **Connected components.** A depth-first search starting from atom 0, then from the lowest unvisited
   atom, collects the fragments. Fragment numbering therefore follows the lowest atom index of each
   fragment, and each fragment is a sorted list of 0-based indices.
5. **Table.** The indices are compacted into ranges `start:end` (both ends included), separated by
   spaces; fragments are separated by ` | `.
6. **Files** (`-w`). Each fragment is written to `<stem>_conf<i>_frag<j>.xyz` (next to the input or in
   `--out-dir`), with the atoms in their original order and the comment `CONF i - Fragment j`.

A larger `-b` adds bonds and can only merge fragments; a smaller `-b` can only split them.
