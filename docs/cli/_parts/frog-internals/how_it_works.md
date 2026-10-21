1. **Read the structures.** Every input is read with `load_structures` (all structures of an XYZ file,
   the geometry of an ORCA/Gaussian output). Each `-i` set must have 2, 3 or 4 indices, and all of
   them must be valid for the first structure of each file.
2. **Measure** every set on every structure (`frogchem.math_geometry`):
   * 2 atoms, `B`: distance $|\mathbf r_2-\mathbf r_1|$ in Å.
   * 3 atoms, `A`: angle at the middle atom, in degrees.
   * 4 atoms, `D`: dihedral from $\mathbf b_0=\mathbf r_1-\mathbf r_2$, $\mathbf b_1=\mathbf r_3-\mathbf r_2$
     (normalised), $\mathbf b_2=\mathbf r_4-\mathbf r_3$. With $\mathbf v$ and $\mathbf w$ the
     components of $\mathbf b_0$ and $\mathbf b_2$ perpendicular to $\mathbf b_1$,
     $\phi=\operatorname{atan2}\big((\mathbf b_1\times\mathbf v)\cdot\mathbf w,\ \mathbf v\cdot\mathbf w\big)$,
     in $(-180°, 180°]$.
3. **Table.** One row per structure (`<file> - Conf <n>`, n 0-based). The column label is the type
   letter, the atom chain (element + 0-based index) and the unit. Columns are sorted by label, so
   all angles (`A`) come first, then distances (`B`), then dihedrals (`D`), whatever the `-i` order.
4. **Ensemble average** (`-c checkpoint.json`). Boltzmann weights $w_k$ come from the checkpoint (see
   [Ensembles](../user-guide/ensemble.md)) and are renormalised over the structures that have a value.
   Distances and angles are averaged linearly, $\langle x\rangle=\sum_k w_k x_k$. Dihedrals use the
   circular mean

   $$
   \langle\phi\rangle=\operatorname{atan2}\Big(\sum_k w_k\sin\phi_k,\ \sum_k w_k\cos\phi_k\Big).
   $$

   For distances an NOE-type average $\langle r^{-6}\rangle^{-1/6}=\big(\sum_k w_k r_k^{-6}\big)^{-1/6}$
   is also given. The per-structure table is printed only with `--individual`.
