"""
Write the final geometry of ORCA/Gaussian calculations to XYZ files.
"""

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import ParseError
from frogchem.filetypes import load_calculation
from frogchem.io import write_xyz
from frogchem.printer import print_warning


class WriteXYZ(CLICommand, name='frog-xyz'):
    """Write the final geometry of ORCA/Gaussian calculations to XYZ files.

    One XYZ file is written per input, next to it (or in --out-dir). The input is never overwritten.
    """

    category = 'parsing'
    has_tables = False
    outputs = [
        Output('file', '<stem>_final.xyz', 'Final geometry of each input (Å)'),
    ]
    guide = Guide(
        why="""
            After an optimisation you usually need the final structure as a plain XYZ file: to open it in a
            viewer, to start the next calculation (frequencies, TD-DFT, a better single point), or to put it
            in a conformer workflow. ORCA writes `<job>.xyz` itself but Gaussian does not, and when you have
            many outputs (or only the output left on your laptop) extracting the last geometry by hand is
            tedious. This command does it for any number of files at once.
            """,
        needs="""
            ORCA (`.out`) or Gaussian (`.log`/`.out`) **output files**, also gzipped (`.out.gz`); any job type
            that prints coordinates. The **last** geometry in the file is taken, i.e. the optimised one if the
            optimisation finished, or the last step reached if it did not (check the output for
            convergence first: this command does not). An `.xyz` file is also accepted (its first frame is
            re-written).
            """,
        how=[
            "Run `frog-xyz opt.out`: writes `opt_final.xyz` next to `opt.out`.",
            "For many files use a wildcard and collect the results in one folder: "
            "`frog-xyz conf_*.out --out-dir geoms`.",
            "Open the XYZ in a viewer, or paste it into the next input (ORCA: `* xyzfile 0 1 opt_final.xyz`).",
        ],
        expect="""
            One `<stem>_final.xyz` per input (standard XYZ: number of atoms, a comment line
            `FROG formatted`, then element and x y z in Å). Charge and multiplicity are **not** stored
            in an XYZ file: remember them for the next input.

            * Files without a readable geometry are skipped with a warning and the command ends with an error
              listing them; the other files are still written.
            * The input file is never overwritten; an existing `<stem>_final.xyz` is.

            Example: `frog-xyz opt.out water.xyz --out-dir xyzout` writes `xyzout/opt_final.xyz` (96 atoms)
            and `xyzout/water_final.xyz`.
            """,
    )
    examples = [
        ('frog-xyz opt.out', 'Write opt_final.xyz next to opt.out'),
        ('frog-xyz conf_*.out --out-dir geoms', 'Write one XYZ per conformer into geoms/'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='ORCA/Gaussian output files (the last geometry of each is written)')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.file)
        failed = []
        for file in args.file:
            try:
                calc = load_calculation(file)
                atoms, geom = calc.atoms, calc.geom
                if atoms is None or geom is None or len(atoms) == 0:
                    raise ValueError('no geometry found')
            except Exception as e:
                print_warning(f'Skipping {file}: {e}')
                failed.append(file)
                continue
            out = self.derived_path(file, 'final', '.xyz')
            write_xyz(
                fname=str(out),
                atoms_list=[atoms],
                positions_list=[geom],
                comments_list=['FROG formatted'],
            )
            self.saved(out, 'xyz')
        if failed:
            raise ParseError(f'No geometry written for {len(failed)} file(s): {", ".join(failed)}')


def parse_args():
    return WriteXYZ().build_parser().parse_args()


def main():
    WriteXYZ().main()


if __name__ == '__main__':
    main()
