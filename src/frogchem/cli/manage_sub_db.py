"""Manage the fragment library used by frog-substitute.

List, add (from XYZ or SMILES), edit and remove substituent and ligand fragments, and
show the scope presets for high-throughput screening. The library lives in the package
data directory (frogchem/substituter/data): every change edits it in place.

Andrea Pellegrini, 2026
"""
from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.printer import print_info

DB_LOCATION = 'frogchem/substituter/data'


class ManageSubDb(CLICommand, name='frog-sub-db'):
    """Manage the fragment library used by frog-substitute.

List, add (from XYZ or SMILES), edit and remove substituent and ligand fragments, and
show the scope presets for high-throughput screening. The library lives in the package
data directory (frogchem/substituter/data): every change edits it in place.

Global options (-q, --format, --table-file) go before the subcommand.
"""
    category = 'substituter'
    outputs = [
        Output('table', 'Fragment Database (<n> fragments)',
               'One row per fragment: category, delete/attach atom indices (0-based), tags, description',
               when='list'),
        Output('table', 'Scope Presets', 'One row per scope preset: number of fragments and their names',
               when='scopes'),
        Output('file', f'{DB_LOCATION}/library.json', 'Fragment library index, updated in place',
               when='add, add-smiles, edit, remove'),
        Output('file', f'{DB_LOCATION}/structures/<name>.xyz', 'Fragment geometry (deleted by remove)',
               when='add, add-smiles'),
    ]
    guide = Guide(
        why="""
            `frog-substitute` takes its substituents from a fragment library. This command shows what is
            in it (names to pass to `--db-frag`, and the *scope presets* used with `--scope`), and lets you
            add your own groups (from an XYZ file or a SMILES string), fix their attachment atoms, rename
            or delete them. Use it before a substitution screen to find the right fragment names, or to add
            a group that is missing (e.g. a special ligand arm).
            """,
        needs="""
            Nothing to list the library. To add a fragment: either an XYZ file of the substituent including
            one extra atom (usually an H) where it attaches, plus the 0-based numbers of that atom
            (DELETE) and of the atom that bonds to the molecule (ATTACH); or a SMILES string with `*` at
            the attachment point (e.g. `*c1ccccc1` for phenyl), from which a 3D fragment is generated.
            The library lives inside the installed package (`frogchem/substituter/data`): adding, editing
            and removing change it **in place** for every project that uses this installation.
            """,
        how=[
            "See all fragments: `frog-sub-db list`; filter with `--category aryls` or `--tag bulky`.",
            "See the screening presets: `frog-sub-db scopes` (e.g. `halogens`, `aryls`, `ewg`).",
            "Add from SMILES: `frog-sub-db add-smiles Bpin \"*B1OC(C)(C)C(C)(C)O1\" --category boron --tags ewg` "
            "(quote the SMILES so the shell does not interpret it).",
            "Add from XYZ: `frog-sub-db add tolyl tolyl.xyz --indices 7 0` (delete atom 7, bond through atom 0), "
            "or pick the two atoms in a 3D window with `--gui`.",
            "Correct or clean up: `frog-sub-db edit tolyl --indices 8 0 --rename p-tolyl2` and "
            "`frog-sub-db remove tolyl`. Global options such as `-q` or `--format csv` go **before** the "
            "subcommand (`frog-sub-db -q list`).",
        ],
        expect="""
            `list` prints *Fragment Database (n fragments)*: name, category, **Delete** and **Attach**
            indices (0-based, inside the fragment file), tags and description; the current library has
            70 entries, e.g. `fluoro` (halogens, delete 0, attach 1). `scopes` prints each preset with
            the number of fragments and their names (e.g. `halogens` 4: fluoro, chloro, bromo, iodo;
            `aryls` 17).

            * Entries with category `general` and description *Auto-registered via substitute_fragment* were
              added automatically when a custom `--frag` was used in `frog-substitute`; remove the ones you
              do not need.
            * The add/edit/remove commands report the files they wrote (`structures/<name>.xyz` and
              `library.json`). If FROG is reinstalled, your additions may be lost: keep a copy of the
              fragment XYZ files.
            * A wrong DELETE/ATTACH pair is the most common cause of strange substitutions (fragment attached
              through the wrong atom): check it with `list` and fix it with `edit --indices`.
            """,
    )
    examples = [
        ('frog-sub-db list --category aryls', 'List the aryl fragments'),
        ('frog-sub-db add-smiles Bpin "*B1OC(C)(C)C(C)(C)O1" --category boron --tags ewg',
         'Generate a fragment from SMILES; * marks the attachment point'),
        ('frog-sub-db add tolyl tolyl.xyz --indices 7 0', 'Add an XYZ fragment: delete atom 7, attach through atom 0'),
    ]

    def setup_args(self, parser):
        subparsers = parser.add_subparsers(dest="command", required=True, metavar='COMMAND')
        fmt = {'formatter_class': parser.formatter_class}

        list_parser = subparsers.add_parser("list", **fmt, help="List the fragments in the library")
        list_parser.add_argument("--category", type=str, help="Only fragments of this category, e.g. aryls")
        list_parser.add_argument("--tag", type=str, help="Only fragments with this tag, e.g. bulky")

        subparsers.add_parser("scopes", **fmt, help="List the scope presets for high-throughput screening")

        add_parser = subparsers.add_parser("add", **fmt, help="Add a fragment from an XYZ file")
        add_parser.add_argument("name", help="Unique fragment name, e.g. phenyl")
        add_parser.add_argument("xyz", help="XYZ file of the fragment")
        add_parser.add_argument("--indices", type=int, nargs=2, metavar=('DELETE', 'ATTACH'),
                                help="Atom deleted on attachment and atom bonded to the molecule, 0-based")
        add_parser.add_argument("--gui", action="store_true",
                                help="Pick the delete and attach atoms in a 3D viewer")
        add_parser.add_argument("--desc", type=str, default="", help="Description")
        add_parser.add_argument("--category", type=str, default="general", help="Category")
        add_parser.add_argument("--tags", type=str, nargs="+", default=[], help="Tags")

        smiles_parser = subparsers.add_parser("add-smiles", **fmt,
                                              help="Add a fragment from SMILES with '*' as attachment point")
        smiles_parser.add_argument("name", help="Unique fragment name, e.g. Bpin")
        smiles_parser.add_argument("smiles", help="SMILES with a '*' attachment point, e.g. '*c1ccccc1'")
        smiles_parser.add_argument("--desc", type=str, default="", help="Description")
        smiles_parser.add_argument("--category", type=str, default="general", help="Category")
        smiles_parser.add_argument("--tags", type=str, nargs="+", default=[], help="Tags")

        edit_parser = subparsers.add_parser("edit", **fmt, help="Edit a fragment")
        edit_parser.add_argument("name", help="Name of the fragment to edit")
        edit_parser.add_argument("--rename", help="New fragment name")
        edit_parser.add_argument("--indices", type=int, nargs=2, metavar=('DELETE', 'ATTACH'),
                                 help="New delete and attach atom indices, 0-based")
        edit_parser.add_argument("--gui", action="store_true",
                                 help="Re-pick the delete and attach atoms in a 3D viewer")
        edit_parser.add_argument("--desc", help="New description")
        edit_parser.add_argument("--category", help="New category")
        edit_parser.add_argument("--tags", nargs="+", help="New tags")

        rm_parser = subparsers.add_parser("remove", **fmt, help="Remove a fragment and its XYZ file")
        rm_parser.add_argument("name", help="Name of the fragment to remove")

    def run(self, args):
        from frogchem.substituter.database_sub import FragmentDatabase
        db = FragmentDatabase()

        if args.command == "list":
            rows = []
            for name in db.list_fragments(category=args.category, tag=args.tag):
                _, (delete_idx, attach_idx) = db.get_fragment(name)
                entry = db.data.get(name, {})
                rows.append([name, entry.get('category', 'general'), int(delete_idx), int(attach_idx),
                             ", ".join(entry.get('tags', [])), entry.get('description', '')])
            if rows:
                self.output(rows, headers=["Name", "Category", "Delete", "Attach", "Tags", "Description"],
                            title=f"Fragment Database ({len(rows)} fragments)")
            else:
                print_info("No fragments match the criteria")

        elif args.command == "scopes":
            rows = [[scope, len(frags), ", ".join(frags)] for scope, frags in db.get_scope_presets().items()]
            self.output(rows, headers=["Scope", "Count", "Fragments"], title="Scope Presets")

        elif args.command == "add-smiles":
            try:
                final_name = db.add_from_smiles(name=args.name, smiles=args.smiles, desc=args.desc,
                                                category=args.category, tags=args.tags)
            except Exception as e:
                raise InputError(f"Could not generate a fragment from SMILES {args.smiles!r}: {e}") from e
            self._report(db, final_name, f"Fragment '{final_name}' generated from SMILES")

        elif args.command == "add":
            self.require_files([args.xyz])
            if args.gui:
                delete_idx, attach_idx = self._pick(args.xyz)
            elif args.indices:
                delete_idx, attach_idx = args.indices
            else:
                raise InputError("Provide --indices DELETE ATTACH or use --gui")
            final_name = db.add_fragment(name=args.name, source_xyz=args.xyz, delete_idx=delete_idx,
                                         attach_idx=attach_idx, desc=args.desc,
                                         category=args.category, tags=args.tags)
            self._report(db, final_name, f"Added fragment '{final_name}'")

        elif args.command == "edit":
            current = db.get_fragment(args.name)
            if not current:
                raise InputError(f"Fragment '{args.name}' not found")
            new_indices = None
            if args.gui:
                new_indices = self._pick(current[0], required=False)
            elif args.indices:
                new_indices = tuple(args.indices)
            try:
                db.update_fragment(name=args.name, new_name=args.rename, new_pair=new_indices,
                                   new_desc=args.desc, new_category=args.category, new_tags=args.tags)
            except ValueError as e:
                raise InputError(str(e)) from e
            print_info(f"Updated fragment '{args.name}'")
            self.saved(db.db_file, 'library')

        elif args.command == "remove":
            if args.name not in db.data:
                raise InputError(f"Fragment '{args.name}' not found")
            db.delete_fragment(args.name)
            print_info(f"Removed fragment '{args.name}'")
            self.saved(db.db_file, 'library')

    def _report(self, db, name, message):
        print_info(message)
        self.saved(db.structures_dir / db.data[name]['filename'], 'fragment')
        self.saved(db.db_file, 'library')

    @staticmethod
    def _pick(xyz, required=True):
        """Pick (delete, attach) atoms in the GUI; None if skipped and not required."""
        from ase.io import read
        from frogchem.gui_utils import select_atoms_visually
        print_info("Opening the GUI: pick 1. the atom to delete, 2. the attachment atom")
        selection = select_atoms_visually(read(xyz), limit=2)
        if len(selection) == 2:
            print_info(f"Selected: delete={selection[0]}, attach={selection[1]}")
            return tuple(selection)
        if required:
            raise InputError("GUI selection cancelled or invalid")
        print_info("GUI selection skipped: indices unchanged")
        return None


def parse_args():
    return ManageSubDb().build_parser().parse_args()

def main():
    ManageSubDb().main()


if __name__ == '__main__':
    main()
