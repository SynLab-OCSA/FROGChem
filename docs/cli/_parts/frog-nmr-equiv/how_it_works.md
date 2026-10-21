For each file the first structure is used (symbols and Cartesian coordinates in Å). The groups come from
`frogchem.nmr.get_nmr_equivalencies`, the same function `frog-nmr` uses before simulating spectra:

1. **Point group (printed).** `detect_point_group` runs with a 0.1 Å tolerance on the heavy atoms (all
   atoms with `--include-h-sym`). The result is only reported in the *Global Point Group* info line, together
   with the RMSD of the symmetrised structure.
2. **Symmetry-equivalent atoms.** Inside `get_nmr_equivalencies` the point group is always detected on the
   heavy atoms. Its operations are then applied to all atoms, H included. This means `--include-h-sym`
   changes the printed line but not the groups.
3. **Topological filter.** Bonds connect atoms closer than the sum of their covalent radii plus `-b`
   (0.3 Å). Each atom gets a colour from up to six rounds of Weisfeiler–Lehman refinement on this graph
   (element plus the sorted colours of its neighbours). Two atoms related by symmetry are merged only if
   they have the same colour. This rejects a heavy-atom symmetry that the H atoms break.
4. **Rotors.** Every non-H atom with three or more H neighbours (CH₃, NH₃⁺, …) puts those H atoms into one
   group, even in a C1 structure, because such groups rotate fast on the NMR time scale.
5. **Merging.** The pairs from steps 2–4 are joined with a disjoint-set union. The groups are sorted by
   element and by their first index.

The table notes are labels only: `Rotor / Sym` for ≥ 3 H, `CH2` for any 2 H, `Aromatic / Ring` for ≥ 6 C.
The method is described in [NMR theory, §2](../theory/nmr.md).
