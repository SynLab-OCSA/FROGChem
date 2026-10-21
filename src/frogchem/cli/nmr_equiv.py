"""
Detect NMR Chemical Equivalencies in molecules using rigid heavy-atom
point-group symmetry and local graph topology (methyl rotors).

Andrea Pellegrini, 2026
"""

from pathlib import Path

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import ParseError
from frogchem.filetypes import load_structures
from frogchem.nmr import get_nmr_equivalencies
from frogchem.printer import print_info, print_warning
from frogchem.symmetry import detect_point_group


class NmrEquiv(CLICommand, name='frog-nmr-equiv'):
    """Detect NMR-equivalent atoms from point-group symmetry and topology (methyl rotors).

    The point group is detected on the heavy atoms (unless --include-h-sym); its operations
    and the local graph topology (freely rotating CH3 and similar groups) define the
    groups of chemically equivalent nuclei.
    """
    category = 'nmr'
    outputs = [
        Output('table', 'Equivalencies (<file>)',
               'One row per group of equivalent atoms: element, size, atom indices and a note'),
    ]
    guide = Guide(
        why="""
            A QM calculation gives one shielding per atom, but in the spectrum chemically equivalent
            nuclei (the three H of a methyl group, the two ortho H of a phenyl ring) give one signal. This
            command tells you which atoms are equivalent, so you know which computed values to average
            and how many signals to expect. `frog-nmr` uses the same detection when it simulates
            spectra; run this command to check it or to plan the comparison with experiment.
            """,
        needs="""
            A geometry: an `.xyz` file or an ORCA/Gaussian output (the first structure in the file is
            used). It should be optimised: equivalence is found by symmetry, so a distorted geometry hides
            symmetric atoms. No NMR calculation is needed.
            """,
        how=[
            "Run `frog-nmr-equiv opt.xyz`; several files can be given at once.",
            "Add `--one-based` if you want atom numbers counted from 1, as in most viewers and in `frog-nmr` labels (H24 = index 23).",
            "If a group you expect is missing on a slightly distorted geometry, relax the tolerances: "
            "`-d 3` compares coordinates with 3 decimals instead of 4; `-b` changes the bond-detection tolerance (Å).",
        ],
        expect="""
            An info line with the point group found on the heavy atoms (H ignored, unless `--include-h-sym`)
            and a table with one row per group: element, number of atoms **N**, their indices and a note
            (`Rotor / Sym` for groups of ≥ 3 H, `CH2` for any pair of H, `Aromatic / Ring` for ≥ 6 C).

            * Examples: methane → Td, one group of 4 H; toluene → C2v (heavy atoms), groups C 0,2 / C 3,5,
              ortho H 7,8, meta H 9,11 and the methyl H 12,13,14.
            * The `CH2` note only means "two equivalent H": in toluene it is printed for the ortho and meta
              ring protons, in water for the two OH.
            * A C1 molecule gives mostly groups of one atom; freely rotating CH3 (and similar rotors) are still
              merged because they average fast on the NMR time scale.
            * Diastereotopic CH2 protons are correctly **not** merged (they are not equivalent).
            """,
    )
    examples = [
        ('frog-nmr-equiv opt.xyz', 'Equivalent groups with 0-based indices'),
        ('frog-nmr-equiv conf_*.out --one-based --format csv', 'Several files, 1-based indices, as CSV'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help="Geometry file(s) (.xyz, .out, .log)")
        parser.add_argument('-b', '--buffer', type=float, default=0.3,
                            help='Tolerance added to covalent radii for connectivity in Å')
        parser.add_argument('-d', '--decimals', type=int, default=4,
                            help='Decimals used to compare coordinates')
        parser.add_argument('--one-based', action='store_true', help='Print 1-based atom indices')
        parser.add_argument('--include-h-sym', action='store_true',
                            help='Include H atoms in the point-group detection')

    def run(self, args):
        files = self.require_files(args.file)
        failed = []
        for p in files:
            fname = str(p)
            print_info(f"Analyzing equivalencies for: {fname}")
            try:
                structs = load_structures(fname)
                if not structs:
                    raise ParseError("no structures found")
                atoms, coords = structs[0][0], structs[0][1]

                exclude_h = not args.include_h_sym
                pg_data = detect_point_group(atoms, coords, tolerance=0.1, exclude_h=exclude_h)
                pg_label = "Unknown"
                if pg_data.get('is_valid', False):
                    pg_label = pg_data.get('sch_symbol', 'C1')
                    pg_label += f" (RMSD: {pg_data.get('error_rmsd', 0.0):.4f} Å)"
                    if exclude_h:
                        pg_label += " [heavy atoms only]"
                print_info(f"Global Point Group: {pg_label}")

                equivalencies = get_nmr_equivalencies((atoms, coords), buffer=args.buffer, decimals=args.decimals)
            except Exception as e:
                print_warning(f"Failed to analyze {fname}: {e}")
                failed.append(fname)
                continue

            table_data = []
            for elem in sorted(equivalencies.keys()):
                for group in equivalencies[elem]:
                    count = len(group)
                    indices = [i + 1 for i in group] if args.one_based else group
                    note = ""
                    if elem == 'H' and count >= 3:
                        note = "Rotor / Sym"
                    elif elem == 'H' and count == 2:
                        note = "CH2"
                    elif count >= 6 and elem == 'C':
                        note = "Aromatic / Ring"
                    table_data.append([elem, count, ", ".join(map(str, indices)), note])

            headers = ["Elem", "N", f"Indices ({'1' if args.one_based else '0'}-based)", "Note"]
            self.output(table_data, headers, title=f"Equivalencies ({Path(fname).name})")

        if failed and len(failed) == len(files):
            raise ParseError(f"No file could be analyzed: {', '.join(failed)}")


def main():
    NmrEquiv().main()


if __name__ == '__main__':
    main()
