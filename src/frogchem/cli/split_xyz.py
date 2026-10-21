#!/usr/bin/env python3
"""Split a multi-XYZ file into one folder per structure, or merge several files into one.

Supports XYZ, ORCA and Gaussian input files.
"""

from pathlib import Path

from frogchem.exceptions import InputError
from frogchem.filetypes import load_frames
from frogchem.io import write_xyz
from frogchem.printer import print_info, print_warning

from frogchem.cli.base import CLICommand, Guide, Output


class SplitXyz(CLICommand, name='frog-split'):
    """Split a multi-XYZ file into one folder per structure, or merge several files into one.

    Split mode writes '<dir>/conf_<n>/geom.xyz' with n starting from 1 in file order;
    the original comment line of each structure is kept ('conf_<n>' when it is empty).
    Existing conf_<n> folders are skipped unless -f is given.
    With --merge, all structures of all inputs are concatenated into one multi-XYZ file.
    """
    category = 'geometry'
    has_tables = False
    outputs = [
        Output('file', '<stem>_split/conf_<n>/geom.xyz',
               'One structure per folder, n = 1, 2, ... (next to the input; -o DIR sets the folder)'),
        Output('file', 'merged.xyz', 'All structures of all inputs, comment lines kept (-o sets the path)',
               when='--merge'),
    ]
    guide = Guide(
        why="""
            Conformer searches give one big file with many structures, but quantum-chemistry programs need one
            input per structure. This command splits a multi-structure XYZ into one folder per conformer
            (`conf_1/geom.xyz`, `conf_2/geom.xyz`, ...) — the layout the FROG ensemble tools understand, so
            that later results are matched to the right conformer. With `--merge` it does the reverse:
            collects single structures (e.g. the optimised geometries of each conformer) into one file to view
            them together or to feed them to another command.
            """,
        needs="""
            * **Split**: a multi-structure `.xyz` (e.g. `crest_conformers.xyz`, an MD or scan trajectory).
              ORCA/Gaussian outputs are accepted but give only their **final** geometry (one folder), not every
              optimisation step.
            * **Merge**: several files (`.xyz`, ORCA or Gaussian outputs); all their structures are
              concatenated in the order you give the files. Structures taken from an ORCA/Gaussian output get
              the file name as comment line.
            """,
        how=[
            "Split: `frog-split crest_conformers.xyz` writes `crest_conformers_split/conf_1/geom.xyz`, "
            "`conf_2/geom.xyz`, ... next to the input; use `-o structures` to choose the folder.",
            "Build your calculation inputs inside each `conf_<n>` folder (keep the folder names: the conformer "
            "number is read from them by the ensemble commands).",
            "Re-running on the same folder skips existing `conf_<n>` folders; add `-f` to overwrite them.",
            "Merge: `frog-split --merge conf_*/geom.xyz -o ensemble.xyz` (without `-o` the result is "
            "`merged.xyz`; an existing output is overwritten only with `-f`).",
        ],
        expect="""
            No table, only files and messages. Example: `frog-split ensemble.xyz` on an 18-structure file
            created `ensemble_split/conf_1` ... `conf_18`, each with a `geom.xyz` of 168 atoms whose comment
            line is the original one (here the CREST energy, `-257.40634594`). Running it a second time printed
            *"Skipped 18 existing conf_<n> folders ... (use -f to overwrite)"* and changed nothing.

            * Numbering starts at **1** (`conf_1` is the first structure of the file), unlike atom indices,
              which are 0-based. CREST sorts its output by energy, so `conf_1` is the lowest CREST conformer.
            * The shell sorts `conf_*` alphabetically (`conf_1, conf_10, conf_11, ..., conf_2`): with `--merge`
              the merged order follows that order, not the numeric one. List the files explicitly if order
              matters.
            * With several inputs in split mode drop `-o`: each input goes to its own `<stem>_split` folder.
            """,
    )
    examples = [
        ('frog-split crest_conformers.xyz', 'Write crest_conformers_split/conf_1/geom.xyz, conf_2/, ...'),
        ('frog-split crest_conformers.xyz -o structures -f', 'Split into ./structures, overwriting'),
        ('frog-split --merge conf_*/geom.xyz -o ensemble.xyz', 'Merge single structures into one ensemble'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='Input XYZ, ORCA or Gaussian file(s)')
        parser.add_argument('--merge', action='store_true',
                            help='Merge all inputs into a single multi-XYZ file')
        self.add_output_arg(parser,
                            help='Split: destination folder, <stem>_split next to the input if omitted; '
                                 'merge: output file, merged.xyz if omitted')
        self.add_out_dir_arg(parser)
        self.add_force_arg(parser, help='Overwrite existing conf_<n> folders or merged file')

    def run(self, args):
        self.require_files(args.file)
        if args.merge:
            self._merge(args.file)
            return
        if args.output and len(args.file) > 1:
            raise InputError("-o/--output takes one input in split mode: drop -o (each input goes to "
                             "<stem>_split) or use --merge")
        for f in args.file:
            dest = Path(args.output) if args.output else self.derived_path(f, 'split', '')
            self._split(f, dest)

    def _split(self, file_path, dest: Path) -> None:
        frames = load_frames(file_path)
        if not frames:
            raise InputError(f"No structures found in {file_path}")
        if dest.exists() and not dest.is_dir():
            raise InputError(f"{dest} exists and is not a folder")
        dest.mkdir(parents=True, exist_ok=True)
        print_info(f"Splitting {len(frames)} structures from {file_path} into {dest}")

        skipped = 0
        for n, (atoms, positions, comment) in enumerate(frames, start=1):
            conf_dir = dest / f"conf_{n}"
            geom_path = conf_dir / "geom.xyz"
            if conf_dir.exists() and not self.args.force:
                skipped += 1
                continue
            conf_dir.mkdir(parents=True, exist_ok=True)
            write_xyz(str(geom_path), [atoms], [positions], [comment or f"conf_{n}"])
            self.saved(geom_path, 'structure')
        if skipped:
            print_warning(f"Skipped {skipped} existing conf_<n> folders in {dest} (use -f to overwrite)")

    def _merge(self, files) -> None:
        out = self.output_path('merged.xyz', '.xyz')
        inputs = {Path(f).resolve() for f in files}
        if out.resolve() in inputs:
            raise InputError(f"Refusing to overwrite the input file {out}")
        if out.exists() and not self.args.force:
            raise InputError(f"Output file {out} exists (use -f to overwrite)")

        all_atoms, all_geoms, all_comments = [], [], []
        for f in files:
            frames = load_frames(f)
            if not frames:
                raise InputError(f"No structures found in {f}")
            for atoms, geom, comment in frames:
                all_atoms.append(atoms)
                all_geoms.append(geom)
                all_comments.append(comment)

        print_info(f"Merging {len(all_atoms)} structures from {len(files)} files")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_xyz(str(out), all_atoms, all_geoms, all_comments)
        self.saved(out, f"merged ensemble ({len(all_atoms)} structure{'s' * (len(all_atoms) != 1)})")


# backward compat
def parse_args():
    return SplitXyz().build_parser().parse_args()


def main():
    SplitXyz().main()


if __name__ == '__main__':
    main()
