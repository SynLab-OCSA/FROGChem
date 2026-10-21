The library is `frogchem.substituter.database_sub.FragmentDatabase`. It is a JSON index, `library.json`, plus one XYZ file
per fragment in `structures/`. Both live in the package directory `frogchem/substituter/data`. No option or
environment variable can point the command at another library. Each entry stores the file name, the pair
`frag_pair` = (DELETE, ATTACH) of 0-based atom indices inside that file, a description, a category, tags and
the source SMILES (if any).

1. **Opening the library.** The data folder is created if it is missing, and `library.json` is read. If the
   index has fewer entries than the built-in standard set (68 SMILES in `STANDARD_LIBRARY`), the missing ones are
   rebuilt from SMILES. Opening the library can therefore write files, even for `list`. If the JSON cannot be
   parsed, the index is treated as empty.
2. **`list`** prints every entry, or only those of one `--category` or one `--tag` (exact match). Looking up
   a standard fragment whose XYZ file has been deleted rebuilds the file.
3. **`scopes`** prints the fixed dictionary `SCOPE_PRESETS`. It is defined in the code, not in `library.json`.
4. **`add-smiles`.** RDKit reads the SMILES. The first `*` sets the indices: DELETE is the `*` and ATTACH is
   its first neighbour. The `*` is turned into an H, and hydrogens are added. One conformer is embedded with
   ETKDGv3 (random seed 42; plain ETKDG if that fails) and minimised with MMFF (UFF as a fallback). The result
   is written to `structures/<name>.xyz`. An entry with the same name is **overwritten**.
5. **`add`** copies the XYZ file into `structures/` with the given indices, or with the two atoms picked in
   `--gui`. If the name is already taken, `_1`, `_2`, … is appended, and `final_name` is the name actually stored.
   The indices are not checked against the file.
6. **`edit`** changes only the fields you give. `--rename` also renames the XYZ file and fails if the new name
   exists. **`remove`** deletes the entry and its XYZ file.

Every change rewrites `library.json` at once. There is no backup and no undo.
