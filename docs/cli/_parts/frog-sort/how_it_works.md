1. **Read the frames.** `load_frames` returns every structure of the input with its comment line
   (XYZ files give all their structures, ORCA/Gaussian outputs their geometry). All `-i` indices
   must be valid for the first structure.
2. **Dihedrals.** Each `-i a b c d` defines a dihedral that is measured on every structure with
   `frogchem.math_geometry.calculate_dihedral`. The angle is
   $\phi=\operatorname{atan2}\big((\mathbf b_1\times\mathbf v)\cdot\mathbf w,\ \mathbf v\cdot\mathbf w\big)$,
   in degrees, in $(-180°, 180°]$, where $\mathbf v$ and $\mathbf w$ are the bonds a–b and c–d
   projected perpendicular to the central bond b–c.
3. **Code.** Each dihedral gives one letter: **P** if $\phi \ge 0$ (0° included), **N** if $\phi<0$.
   The letters are concatenated in the order of the `-i` options, so $k$ dihedrals give up to $2^k$
   codes (`NP` = first dihedral negative, second positive).
4. **Tables.** *Dihedral signs*: one row per structure (0-based, file order) with the angles (two
   decimals) and the code. *Categories*: the number of structures for each code found, codes sorted
   alphabetically. Codes with no structure are not listed.
5. **Files** (`-w`). One multi-structure XYZ per code found, `<stem>_<code>.xyz` (next to the input or
   in `--out-dir`), with the structures in the original order and the original comment lines.

No energies, symmetry or atom mapping are used: the sorting is purely geometric, and all files must
share the same atom numbering for the indices to mean the same atoms.
