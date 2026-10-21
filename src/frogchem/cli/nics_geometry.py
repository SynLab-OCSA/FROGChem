#!/usr/bin/env python3
"""Add a NICS ghost atom at the centroid of selected atoms, ready for an ORCA input."""

import numpy as np

from frogchem.exceptions import InputError
from frogchem.filetypes import load_structures

from frogchem.cli.base import CLICommand, Guide, Output

GHOST_BASIS = "NewGTO S 1 1 1e6 1 end NewAuxJGTO S 1 1 2e6 1 end"


def get_center(positions: np.ndarray):
    return np.average(positions, axis=0)


class NicsGeometry(CLICommand, name='frog-nics'):
    """Add a NICS ghost atom at the centroid of selected atoms, ready for an ORCA input.

    The ghost atom is written as 'H:' with a dummy basis (NewGTO/NewAuxJGTO). Its isotropic
    shielding sigma_iso at the ring (or bond) centroid gives NICS(0) = -sigma_iso (sign
    convention of Schleyer et al., JACS 1996: negative NICS = aromatic). Only the in-plane
    centroid is generated (no out-of-plane NICS(1) offset). Repeated indices are counted once.
    """
    category = 'nmr'
    outputs = [
        Output('table', 'NICS Geometry (<file>)',
               'Atoms with coordinates (Å) plus the ghost atom H: and its basis; one table per structure'),
        Output('file', '<stem>_nics.txt',
               'ORCA coordinate lines to paste between "* xyz <charge> <mult>" and "*" '
               '(<stem>_nics_<n>.txt for multi-structure files)', when='-w'),
    ]
    guide = Guide(
        why="""
            NICS (*nucleus-independent chemical shift*) measures aromaticity: the magnetic shielding
            computed at a point with no nucleus, usually the centre of a ring. Negative values
            (strong shielding by a ring current) indicate aromatic rings, positive values antiaromatic
            rings, values near zero non-aromatic rings. This command prepares the geometry for that
            calculation: it places a "ghost" atom (basis functions but no nucleus and no electrons) at
            the centroid of the ring atoms you choose.
            """,
        needs="""
            An optimised geometry (`.xyz`, ORCA or Gaussian output) and the 0-based numbers of the ring
            atoms (look them up in a viewer; the first atom of the file is 0). After running the command
            you need ORCA to compute the NMR shieldings of the new geometry (`! NMR`).
            """,
        how=[
            "Run `frog-nics benzene.xyz -i 0 1 2 3 4 5 -w` (ring atoms 0–5): it prints the coordinates with "
            "the ghost atom `H:` appended and writes `benzene_nics.txt` (`--out-dir` to write it elsewhere).",
            "Paste the lines of `benzene_nics.txt` into an ORCA input between `* xyz 0 1` and `*` (use your "
            "charge and multiplicity) and add `! NMR` to the same method line you use for the molecule.",
            "Run ORCA and read the isotropic shielding σ_iso of the ghost atom (the last atom) in the "
            "*CHEMICAL SHIELDING SUMMARY* of the ORCA output.",
            "NICS(0) = −σ_iso. For a molecule with several rings, repeat the command for each ring.",
        ],
        expect="""
            A table of all atoms (element, X, Y, Z in Å) followed by the ghost line, e.g. for the ring of
            toluene (`-i 0 1 2 3 4 5`): `H:  0.679003  0.000000  0.000233` with the dummy basis
            `NewGTO S 1 1 1e6 1 end NewAuxJGTO S 1 1 2e6 1 end` (it replaces the ghost's normal basis by a
            single, extremely tight s function, so the ghost adds practically nothing to the molecule's basis).

            * Typical NICS(0): benzene about −8 to −10 ppm, cyclobutadiene strongly positive, cyclohexane
              about −2 ppm. Compare rings with the same method and basis.
            * Only the in-plane centroid is generated (NICS(0)); NICS(1), 1 Å above the ring, which is less
              affected by the σ bonds, is not produced: move the ghost atom yourself along the ring normal.
            * Multi-structure files give one table (and with `-w` one file `<stem>_nics_<n>.txt`) per structure.
            * The centroid of a non-planar or wrongly selected set of atoms is not a meaningful NICS point:
              check that the indices are really the ring atoms.
            """,
    )
    examples = [
        ('frog-nics benzene.xyz -i 0 1 2 3 4 5', 'Ghost atom at the centroid of the ring atoms 0-5'),
        ('frog-nics opt.out -i 0 1 2 3 4 5 -w', 'Same from an ORCA output, also writing opt_nics.txt'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', help='XYZ, ORCA or Gaussian file')
        parser.add_argument('-i', '--indices', type=int, nargs='+', required=True,
                            help='Atoms whose centroid hosts the ghost atom, 0-based')
        self.add_write_arg(parser, help='Write the ORCA coordinate lines to a text file')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files([args.file])
        structs = load_structures(args.file)
        if not structs:
            raise InputError(f"No structures found in {args.file}")

        for n, (atoms, geom) in enumerate(structs):
            geom = np.asarray(geom, dtype=float)
            bad = [i for i in args.indices if i < 0 or i >= len(geom)]
            if bad:
                raise InputError(f"Atom indices {bad} out of range (0-{len(geom) - 1})")
            center = get_center(geom[list(dict.fromkeys(args.indices))])

            rows = [[a, *map(float, xyz), ''] for a, xyz in zip(atoms, geom)]
            rows.append(['H:', *map(float, center), GHOST_BASIS])
            suffix = f", structure {n}" if len(structs) > 1 else ''
            self.output(rows, ['Atom', 'X (Å)', 'Y (Å)', 'Z (Å)', 'Ghost basis'],
                        title=f"NICS Geometry ({args.file}{suffix})", floatfmt='.6f')

            if args.write:
                tag = f"nics_{n}" if len(structs) > 1 else 'nics'
                path = self.derived_path(args.file, tag, '.txt')
                lines = [f"{a:<3} {x:>14.6f} {y:>14.6f} {z:>14.6f}" for a, (x, y, z) in zip(atoms, geom)]
                lines.append(f"{'H:':<3} {center[0]:>14.6f} {center[1]:>14.6f} {center[2]:>14.6f}  {GHOST_BASIS}")
                path.write_text('\n'.join(lines) + '\n')
                self.saved(path, 'coordinates')


def parse_args():
    return NicsGeometry().build_parser().parse_args()


def main():
    NicsGeometry().main()


if __name__ == '__main__':
    main()
