The command applies one reflection to every structure and writes the result; nothing else changes.

1. **Read the frames.** `frogchem.filetypes.load_frames` returns, for every structure of the input, the
   element symbols, the Cartesian coordinates (Å) and the comment line. XYZ files give all their
   structures; ORCA and Gaussian outputs give their geometry.
2. **Reflect.** For every atom

   $$
   (x, y, z) \;\to\; (-x,\ y,\ z),
   $$

   i.e. a reflection through the $yz$ plane. Any reflection turns a chiral structure into its
   enantiomer; this one was chosen because it needs no reference frame. Bond lengths, angles and the
   absolute values of the dihedrals are unchanged, while every dihedral changes sign and every R/S
   label is inverted.
3. **Write.** All mirrored structures go into one file, `<stem>_mirror.xyz` (next to the input, or in
   `--out-dir`), in the input order and with the atoms in the input order. Each structure keeps its
   original comment line; an empty comment becomes `Conf <i> mirrored`.

Coordinates are written with six decimals, so an atom on the $yz$ plane may appear as `-0.000000`.
No energies are recomputed: the copied comment lines (e.g. CREST energies) are still correct, because
enantiomers have the same energy in an achiral environment.
