#!/usr/bin/env python3
"""Write the mirror image (enantiomer) of every structure in an ensemble.

Inverts the x coordinate of all atoms. Reads XYZ, ORCA or Gaussian files.
"""

import numpy as np

from frogchem.exceptions import InputError
from frogchem.filetypes import load_frames
from frogchem.io import write_xyz
from frogchem.printer import print_info

from frogchem.cli.base import CLICommand, Guide, Output


class MirrorEnsemble(CLICommand, name='frog-mirror'):
    """Write the mirror image (enantiomer) of every structure in an ensemble.

    The x coordinate of every atom is inverted (reflection through the yz plane).
    Comment lines (e.g. energies) are copied unchanged.
    """
    category = 'geometry'
    has_tables = False
    outputs = [
        Output('file', '<stem>_mirror.xyz', 'Mirrored structures in XYZ format, in input order'),
    ]
    guide = Guide(
        why="""
            Enantiomers (mirror-image molecules) have exactly the same energy and the same achiral
            properties, so you only need to compute one of them. When you need the other — to build the
            opposite enantiomer of a catalyst, to get the second hand of a racemic conformer search, or to
            compare a computed ECD spectrum with the measured one (the mirror image has the opposite ECD
            sign) — this command writes it instantly, without recomputing anything.
            """,
        needs="""
            Any geometry: an `.xyz` file (all structures of a multi-XYZ, e.g. `crest_conformers.xyz`, are
            mirrored) or an ORCA/Gaussian output (its last geometry). No calculation keyword is needed.
            """,
        how=[
            "Run `frog-mirror crest_conformers.xyz`: it writes `crest_conformers_mirror.xyz` next to the "
            "input (use `--out-dir DIR` to put it elsewhere).",
            "Optionally check the result with `frog-stereocenters crest_conformers_mirror.xyz`: every R "
            "centre should now be S and vice versa.",
            "Use the mirrored structures as starting points of new calculations; their energies are the same "
            "as those of the originals, so a re-optimisation is only needed if you change method.",
        ],
        expect="""
            No table: one XYZ file per input, with the structures in the same order and the original comment
            lines (e.g. CREST energies) copied unchanged. Atom order is unchanged too, so atom numbers stay
            valid for other commands.

            * The x coordinate of every atom is inverted (reflection through the yz plane): the molecule may
              appear flipped in the viewer, which is expected.
            * Example: `nmr_iPr.xyz` has C0 = S and C2 = R; `nmr_iPr_mirror.xyz` has C0 = R and C2 = S.
            * Mirroring an **achiral** molecule (or a meso compound) gives the same molecule again.
            * Molecules with axial, helical or planar chirality (atropisomers, helicenes, planar-chiral
              complexes) are mirrored correctly too, even though `frog-stereocenters` finds no R/S centre.
            """,
    )
    examples = [
        ('frog-mirror crest_conformers.xyz', 'Write crest_conformers_mirror.xyz next to the input'),
        ('frog-mirror opt_R.out --out-dir enantiomer', 'Write enantiomer/opt_R_mirror.xyz'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ, ORCA or Gaussian file(s)')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.file)
        for file in args.file:
            frames = load_frames(file)
            if not frames:
                raise InputError(f"No structures found in {file}")
            print_info(f"Mirroring {len(frames)} structures from {file}")

            mirrored = []
            for _atoms, pos, _comment in frames:
                new_pos = np.array(pos, dtype=float)
                new_pos[:, 0] *= -1
                mirrored.append(new_pos)

            out = self.derived_path(file, 'mirror', '.xyz')
            write_xyz(str(out),
                      atoms_list=[f[0] for f in frames],
                      positions_list=mirrored,
                      comments_list=[f[2] or f"Conf {i} mirrored" for i, f in enumerate(frames)])
            self.saved(out, f"mirrored ensemble ({len(frames)} structure{'s' * (len(frames) != 1)})")


def parse_args():
    return MirrorEnsemble().build_parser().parse_args()


def main():
    MirrorEnsemble().main()


if __name__ == '__main__':
    main()
