The command reads structures with `frogchem.filetypes.load_frames` (symbols, coordinates in Å and the
comment line of every structure; ORCA and Gaussian outputs give their geometry) and writes plain XYZ
files with `frogchem.io.write_xyz`. No geometry is changed.

**Split mode** (default)

1. **Destination.** `-o DIR` if given (only with one input), otherwise `<stem>_split` next to the
   input or in `--out-dir`. If the destination exists and is a file, the command stops.
2. **One folder per structure.** Structure $n$ (counted from **1**, in file order) is written to
   `<dest>/conf_<n>/geom.xyz`. The comment line is copied; an empty one becomes `conf_<n>`.
3. **Existing folders.** A `conf_<n>` folder that already exists is skipped (a warning gives the
   number of skipped folders), so a second run does not overwrite edited structures. `-f` writes them
   again.

The `conf_<n>/geom.xyz` layout and 1-based numbering are the ones expected by the ensemble tools: with
`-c`, commands match structures to the checkpoint by their `conf_<N>` folder (see
[Ensembles](../user-guide/ensemble.md)).

**Merge mode** (`--merge`)

1. All structures of all inputs are concatenated, in the order the files are given on the command
   line, each with its own comment line.
2. They are written to one multi-structure file, `-o` (default `merged.xyz`). The command refuses to
   overwrite an input file, and an existing output needs `-f`.

Atoms are not checked across files: merging structures with different atom counts or orders gives a
valid XYZ file, but not a valid ensemble for the other commands.
