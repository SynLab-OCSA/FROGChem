1. **Read the frames.** `load_frames` returns every structure with its comment line. The four `-i`
   indices must be valid for the first structure.
2. **Dihedral.** On every structure the dihedral $\phi$ (degrees, $(-180°,180°]$) is computed with
   `frogchem.math_geometry.calculate_dihedral`, the same function used by `frog-sort`. Only its
   absolute value $|\phi|$ is used for the classification.
3. **Classify.** Two modes:
   * **Threshold** (default, `-t`, 90°): $|\phi| < t$ → Z (cis), otherwise E (trans). Every
     structure gets one of the two labels. `-t` must be in $(0°, 180°]$.
   * **Tolerance** (`--tolerance τ`, overrides `-t`): $|\phi|\le\tau$ → Z, $|\phi|\ge 180°-\tau$ → E,
     anything in between → *other*. τ must be in $[0°, 90°)$.
4. **Labels.** `--notation` only renames the classes: `ez` (Z / E), `cis-trans` (cis / trans) or
   `scis-strans` (s-cis / s-trans), plus *other* in tolerance mode. Z, cis and s-cis are always the
   class near 0°.
5. **Tables.** *Dihedral classes*: one row per structure (0-based, file order) with $\phi$ and the
   class. *Class counts*: every class of the chosen scheme, also those with 0 structures.
6. **Files** (`-w`). One XYZ per non-empty class, `<stem>_<label>.xyz`, with the hyphen removed from
   the label (`s-cis` → `scis`), next to the input or in `--out-dir`. Structures keep their order and
   comment lines.

The label is geometric only: it is the cis/trans relation of the four chosen atoms, not a CIP E/Z
descriptor (which needs the priorities of the substituents).
