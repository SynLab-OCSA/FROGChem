The command only writes text files; it never runs CREST or xtb.

1. **Read the geometry.** Every input is read with `frogchem.filetypes.load_structures`: all structures
   of an XYZ file, or the geometry of an ORCA/Gaussian output. Only the first structure is used.
2. **Check the indices.** Every `-i` index must satisfy $0 \le i < N_\mathrm{atoms}$, otherwise the
   command stops before writing anything.
3. **Choose the reference.**
   * Single-structure `.xyz`: the input itself is the reference. `reference=` is the path of the input
     relative to the folder of the constraint file (`os.path.relpath`).
   * ORCA/Gaussian output or multi-structure XYZ: the first structure is written to `<stem>_ref.xyz` in
     the folder of the constraint file (comment line `Reference from <file>`), and `reference=` is
     that file name.
4. **Write the block** (`frogchem.io.write_crest_constraints`). The indices are converted to 1-based
   ($i \to i+1$), sorted, de-duplicated and compacted into CREST list syntax: consecutive runs become
   `a-b`, items are separated by commas (0-based `2 3 36 37` → `3-4,37-38`). The force constant is
   fixed at 0.5 (it has no command-line option):

   ```text
   $constrain
     atoms: <list>
     force constant=0.5
     reference=<file>
   $end
   ```

5. **Output names.** With one input the file is `-o` (default `constrain.inp`, folders are created).
   With several inputs each one gets `<stem>_<tag>.inp`, where the tag is the stem of `-o`
   (`constrain`), next to the input or in `--out-dir`; `-o` must then be a bare file name.
   The command refuses to overwrite an input file.
