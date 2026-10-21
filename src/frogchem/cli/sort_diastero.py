"""Sort conformers by the signs of one or more dihedral angles.

Andrea Pellegrini, 2025
"""

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.filetypes import load_frames
from frogchem.io import write_xyz
from frogchem.math_geometry import calculate_dihedral
from frogchem.printer import print_info


SIGNS_MAP = {1: 'P', -1: 'N'}


class SortDiastero(CLICommand, name='frog-sort'):
    """Sort conformers into categories by the signs of one or more dihedral angles.

    Each dihedral contributes one letter to the category code: P (positive, 0 included)
    or N (negative), e.g. 'PN' for two dihedrals. Useful to separate diastereoisomers
    or the faces of attack in an ensemble.
    """
    category = 'geometry'
    outputs = [
        Output('table', 'Dihedral signs (<file>)',
               'One row per conformer (0-based, file order): each dihedral (°) and the category code'),
        Output('table', 'Categories (<file>)', 'Number of conformers in each category'),
        Output('file', '<stem>_<code>.xyz',
               'Conformers of each category (e.g. <stem>_PN.xyz), original comment lines kept', when='-w'),
    ]
    guide = Guide(
        why="""
            A conformer search (CREST, metadynamics, MD snapshots) mixes structures that a chemist would put in
            different boxes: the two faces of attack of a nucleophile, *P* and *M* helices or atropisomers, the
            two orientations of a ligand. When the difference can be read from the **sign of a dihedral**
            (torsion angle), this command sorts the ensemble into those boxes and can write one file per box,
            so that you can compute or Boltzmann-weight each family separately.
            """,
        needs="""
            * A multi-structure `.xyz` (e.g. `crest_conformers.xyz`), or ORCA/Gaussian outputs (last geometry
              each); all structures with the same atom order.
            * One or more dihedrals, each defined by **four atoms, 0-based** (first atom = 0), chosen so that
              the families you want to separate have opposite signs. Measure them first in a viewer or with
              `frog-internals` on two representative conformers.
            """,
        how=[
            "Check that your dihedral really separates the families: `frog-internals ensemble.xyz -i 50 0 4 3` "
            "should show values of both signs.",
            "Sort: `frog-sort ensemble.xyz -i 50 0 4 3`. For two criteria repeat `-i` "
            "(`-i 50 0 4 3 -i 1 2 3 4`): the code then has two letters, one per dihedral in the given order.",
            "Write one file per family with `-w` (`--out-dir DIR` to choose the folder): "
            "`ensemble_N.xyz`, `ensemble_P.xyz`, or `ensemble_PN.xyz` etc. with two dihedrals.",
        ],
        expect="""
            Two tables per file. *Dihedral signs*: one row per conformer (0-based, file order) with each
            dihedral in degrees and its code — **P** for a positive dihedral (0° included), **N** for a negative
            one. *Categories*: how many conformers fall in each code. Example (`ensemble.xyz`, `-i 50 0 4 3`):
            dihedrals from −72.31° (conformer 0) to +80.49° (conformer 14); categories N = 10, P = 8.

            * Only the **sign** is used: −5° and −175° are both N. Dihedrals close to 0° or ±180° flip sign
              with tiny geometric changes, so the split there is not chemically meaningful; pick a dihedral that
              sits clearly on either side (e.g. ±30–150°) or use `frog-sort-ez` for 0°/180° problems.
            * P/N are just the sign of the dihedral, not the *P*/*M* helicity or *R*/*S* descriptor: decide once,
              on one structure, which code corresponds to which isomer.
            * Written files keep the original comment lines (e.g. CREST energies) and the original order
              within each family.
            """,
    )
    examples = [
        ('frog-sort ensemble.xyz -i 0 1 2 3', 'Split the ensemble by the sign of dihedral 0-1-2-3'),
        ('frog-sort ensemble.xyz -i 0 1 2 3 -i 4 5 6 7 -w',
         'Two dihedrals: write ensemble_PP.xyz, ensemble_PN.xyz, ... next to the input'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ (multi-structure), ORCA or Gaussian files')
        parser.add_argument('-i', '--indices', nargs=4, type=int, required=True, action='append',
                            metavar='IDX',
                            help='Four atom indices (0-based) defining a dihedral; repeat for more dihedrals')
        self.add_write_arg(parser, help='Write one XYZ file per category')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.file)
        for f in args.file:
            frames = load_frames(f)
            if not frames:
                raise InputError(f"No structures found in {f}")
            n_atoms = len(frames[0][0])
            bad = sorted({i for idxs in args.indices for i in idxs if not 0 <= i < n_atoms})
            if bad:
                raise InputError(f"Atom indices {bad} out of range for {f} ({n_atoms} atoms, 0-based)")

            print_info(f"Sorting {len(frames)} conformers from {f}")
            categories = {}
            rows = []
            for i, (_atoms, pos, _comment) in enumerate(frames):
                angles = [float(calculate_dihedral(*(pos[j] for j in idxs))) for idxs in args.indices]
                code = ''.join(SIGNS_MAP[1 if np.sign(a) >= 0 else -1] for a in angles)
                categories.setdefault(code, []).append(i)
                rows.append([i, *angles, code])

            dihedral_headers = ['-'.join(map(str, idxs)) + ' (°)' for idxs in args.indices]
            self.output(rows, headers=['Conf', *dihedral_headers, 'Category'],
                        title=f"Dihedral signs ({f})", floatfmt='.2f')
            self.output([[code, len(ids)] for code, ids in sorted(categories.items())],
                        headers=['Category', 'Count'], title=f"Categories ({f})")

            if args.write:
                for code, ids in sorted(categories.items()):
                    out = self.derived_path(f, code, '.xyz')
                    write_xyz(str(out),
                              [frames[i][0] for i in ids],
                              [frames[i][1] for i in ids],
                              [frames[i][2] or f"Conf {i}" for i in ids])
                    self.saved(out, f"{code} ensemble ({len(ids)} structure{'s' * (len(ids) != 1)})")


def parse_args():
    return SortDiastero().build_parser().parse_args()


def main():
    SortDiastero().main()


if __name__ == '__main__':
    main()
