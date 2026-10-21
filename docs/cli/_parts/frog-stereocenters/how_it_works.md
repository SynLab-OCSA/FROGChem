The command converts each 3D structure into an RDKit molecule and lets RDKit assign the CIP labels.

1. **Read the structures.** `load_structures` returns every structure of an XYZ file or the geometry of
   an ORCA/Gaussian output. Each structure is processed separately.
2. **Build the molecule.** An RDKit molecule is created atom by atom (element symbols only) with the
   coordinates as a conformer.
3. **Perceive bonds** with `rdkit.Chem.rdDetermineBonds.DetermineBonds(mol, charge=--charge)`:
   connectivity from interatomic distances, then bond orders and formal charges that must add up
   to the given total charge (default 0). If no consistent assignment exists (wrong charge, distorted
   geometry, or an element without defined valences such as most metals), the command stops with
   *Bond perception failed* and the RDKit message.
4. **Chirality from the geometry.** `Chem.AssignAtomChiralTagsFromStructure` sets the chiral tag of each
   tetrahedral centre from the 3D arrangement of its neighbours.
5. **CIP labels.** `Chem.FindMolChiralCenters(mol, includeUnassigned=True)` returns every stereocentre
   with its label, **R** or **S**, or **?** when the centre is a stereocentre but no label could be
   assigned.
6. **Table.** One row per centre: 0-based atom index, element, label. Multi-structure files get an
   extra 0-based *Conf* column. A file with no stereocentre prints *No stereocenters found*.

Only point stereocentres (atoms with four different substituents) are reported. Double-bond (E/Z),
axial, helical and planar chirality are not. Labels depend on the perceived bond orders, so they are
only as good as step 3.
