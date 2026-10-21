#!/usr/bin/env python3
"""Detect the separate molecules (fragments) of a multimolecular system.

Atoms are bonded when their distance is below the sum of their atomic radii plus a
buffer; each connected component is a fragment. Each fragment can be written to its own
XYZ file.
"""

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.io import write_xyz, compact_list_fragments
from frogchem.filetypes import load_structures
from frogchem.graph_theory import detect_fragments_from_geom
from frogchem.printer import print_info


class FragmentDetection(CLICommand, name='frog-fragments'):
    """Detect the separate molecules (fragments) of a multimolecular system.

Atoms are bonded when their distance is below the sum of their atomic radii plus a
buffer; each connected component is a fragment. Each fragment can be written to its own
XYZ file.
"""
    category = 'geometry'
    outputs = [
        Output('table', 'Fragments (<file>)',
               'One row per conformer: number of fragments and their atoms (0-based, start:end ranges, fragments separated by |)'),
        Output('file', '<stem>_conf<i>_frag<j>.xyz', 'Geometry of fragment j of conformer i', when='-w'),
    ]
    guide = Guide(
        why="""
            A calculation often contains more than one molecule: a catalyst with its counter-ion, a
            host–guest pair, a substrate next to a reagent, a dimer. This command finds the separate
            molecules (*fragments*: groups of atoms connected by bonds) and lists their atom numbers.
            Use it to get the atom ranges you need for other commands (e.g. `--fragment` in `frog-spin`,
            `-E` in `frog-steric`), to check that an optimisation did not split or fuse molecules, or to
            write each molecule to its own XYZ file (e.g. to compute the fragments separately for an
            interaction energy).
            """,
        needs="""
            Any geometry: an `.xyz` file (several structures allowed, each is analysed), or an ORCA/Gaussian
            output (its last geometry). No particular calculation keyword is needed.
            Bonds are guessed from distances only: two atoms are bonded when their distance is below the sum
            of their **covalent radii** plus a buffer (`-b`, default 0.3 Å).
            """,
        how=[
            "Run `frog-fragments complex.xyz` to list the fragments of every structure in the file.",
            "Read the *Atom Indices* column: atoms are **0-based**; `0:11 50:167` means atoms 0 to 11 and 50 "
            "to 167 (both ends included) form one molecule; fragments are separated by `|`.",
            "If two molecules that should be separate come out as one (or a weak metal–ligand bond is cut), "
            "change the buffer: smaller `-b` (e.g. 0.2) bonds fewer atoms, larger `-b` (e.g. 0.4) bonds more.",
            "Add `-w` (and `--out-dir DIR` to keep things tidy) to write each fragment to "
            "`<stem>_conf<i>_frag<j>.xyz`.",
        ],
        expect="""
            One table per input file, one row per structure: *Conformer* (0-based), *N Fragments* and the atom
            ranges. Example (18-conformer Ag complex `ensemble.xyz`): every conformer gives
            `2 | 0:11 50:167 | 12:49`, i.e. the complex (atoms 0–11 and 50–167) and a separate 38-atom
            molecule (atoms 12–49). A single molecule (Ru complex `opt.out`) gives `1 | 0:95`.

            * The fragment numbering (`frag0`, `frag1`, ...) follows the order of the table, not the size of
              the fragments.
            * Rows that differ along an ensemble (e.g. 2 fragments in one conformer, 1 in another) usually
              mean that two molecules came close or a long bond broke in that conformer: open it and check.
            * Long or weak bonds (metal–arene, agostic, hydrogen bonds, transition-state bonds being formed)
              may or may not be counted depending on `-b`; this is a geometric criterion, not a bond order.
            * Written fragment files contain only coordinates: charge and multiplicity of each fragment are
              up to you when you set up new calculations.
            """,
    )
    examples = [
        ('frog-fragments complex.xyz', 'List the fragments of every conformer'),
        ('frog-fragments dimer.out -b 0.4 -w --out-dir frags',
         'Write each fragment to frags/ with a 0.4 Å bonding buffer'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ, ORCA or Gaussian file(s)')
        parser.add_argument('-b', '--buffer', type=float, default=0.3, metavar='Å',
                            help='Distance added to the sum of atomic radii to consider atoms bonded, in Å')
        self.add_write_arg(parser, help='Write each fragment to a separate XYZ file')
        self.add_out_dir_arg(parser)

    def run(self, args):
        files = self.require_files(args.file)
        for p in files:
            fname = str(p)
            print_info(f"Processing {fname}...")
            table_rows = []

            for conf_idx, (atoms, positions) in enumerate(load_structures(fname)):
                fragments = detect_fragments_from_geom(atoms, positions, buffer=args.buffer)
                table_rows.append([conf_idx, len(fragments), " | ".join(compact_list_fragments(f) for f in fragments)])

                if args.write:
                    for frag_idx, indices in enumerate(fragments):
                        out_path = self.derived_path(p, f"conf{conf_idx}_frag{frag_idx}", '.xyz')
                        write_xyz(
                            str(out_path),
                            atoms_list=[[atoms[i] for i in indices]],
                            positions_list=[[positions[i] for i in indices]],
                            comments_list=[f"CONF {conf_idx} - Fragment {frag_idx}"],
                        )
                        self.saved(out_path, 'geometry')

            self.output(table_rows, headers=["Conformer", "N Fragments", "Atom Indices"],
                        title=f"Fragments ({fname})")


def parse_args():
    return FragmentDetection().build_parser().parse_args()


def main():
    FragmentDetection().main()


if __name__ == '__main__':
    main()
