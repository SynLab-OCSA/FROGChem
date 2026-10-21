1. **Parsing.** Each input is opened with `frogchem.filetypes.load_calculation`. A `.xyz`/`.xyz.gz` file is
   read directly and its first frame is used. For any other file, the format (ORCA or Gaussian) is detected
   from the content, and gzipped files are read transparently. The parser keeps the element symbols and the
   **last** Cartesian geometry printed in the file (Å):
   * ORCA: the last `CARTESIAN COORDINATES (ANGSTROEM)` block;
   * Gaussian: the last `Input orientation` table, or `Standard orientation` when the input orientation
     is not printed.

   For an optimisation this is the converged structure. For an unfinished job it is the last step reached;
   convergence is not checked.
2. **Checks.** A file that cannot be loaded, or that has no atoms or no coordinates, is skipped with a
   warning. The remaining files are still processed. If anything was skipped, the command ends with an error
   that lists those files.
3. **Writing.** One file per input is written with `frogchem.io.write_xyz`: the atom count, the comment line
   `FROG formatted`, then one line per atom with the symbol and x, y, z with 6 decimals. Charge,
   multiplicity, energy and connectivity are not written.
4. **File names.** The output is `<stem>_final.xyz`, with compression suffixes ignored (`opt.out.gz` →
   `opt_final.xyz`). It goes in the input's folder, or in `--out-dir`, which is created if needed. The `_PAC`
   suffix keeps the input, and ORCA's own `<stem>.xyz`, from being overwritten. If two inputs in different
   folders have the same stem and `--out-dir` is used, the parent folder names are prepended until the names
   are unique. An existing `<stem>_final.xyz` from an earlier run is overwritten.

Geometry handling in other commands is described in [Geometries](../user-guide/geometry.md).
