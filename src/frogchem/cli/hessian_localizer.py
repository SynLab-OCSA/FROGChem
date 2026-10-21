"""Localize vibrational normal modes on atoms and fragments.

Reads the normal modes of an ORCA frequency calculation and reports how much
each atom (or user-defined fragment) contributes to each selected mode, optionally with
the inverse participation ratio, and can write displaced geometries along each mode.

Mode numbers are those printed in the ORCA output (VIBRATIONAL FREQUENCIES /
NORMAL MODES): 0-based, translations and rotations included, so the imaginary
mode of a transition state of a non-linear molecule is mode 6.
"""

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.filetypes import load_calculation
from frogchem.orca_parser.hessian import (
    localize_mode,
    inverse_participation_ratio,
    displace_geometry,
)
from frogchem.printer import print_error, print_info, print_warning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_fragments(fragment_args: list[str]) -> dict[str, list[int]]:
    """
    Parses '--fragment NAME:0,1,2' strings into a dict.

    Parameters
    ----------
    fragment_args : list of str
        Each entry must be 'name:idx1,idx2,...'.

    Returns
    -------
    dict[str, list[int]]
        E.g. {'CO2': [0, 1, 2], 'NH3': [3, 4, 5, 6]}
    """
    fragments: dict[str, list[int]] = {}
    for entry in fragment_args:
        try:
            name, indices_str = entry.split(':', 1)
            indices = [int(i.strip()) for i in indices_str.split(',')]
            fragments[name.strip()] = indices
        except ValueError:
            print_warning(
                f"Cannot parse fragment '{entry}'. "
                "Expected format: 'name:idx1,idx2,...'. Skipping."
            )
    return fragments


def _mode_header(calc, mode: int) -> str:
    return f"Mode {mode} ({calc.freq[mode]:.1f} cm⁻¹)"


def _build_atom_table(calc, localization_data: dict[int, np.ndarray], threshold: float) -> tuple[list, list]:
    """Per-atom contributions (%); values below `threshold` are None, atoms with none above it are dropped."""
    headers = ["Atom"] + [_mode_header(calc, k) for k in localization_data]
    rows = []
    for i, atom in enumerate(calc.atoms):
        values = [float(localization_data[mode][i]) for mode in localization_data]
        if any(v > threshold for v in values):
            rows.append([f"{atom}{i}"] + [v if v > threshold else None for v in values])
    return headers, rows


def _build_fragment_table(calc, localization_data: dict[int, np.ndarray],
                          fragments: dict[str, list[int]]) -> tuple[list, list]:
    """Per-fragment contributions (%): sum of the atomic contributions."""
    headers = ["Fragment"] + [_mode_header(calc, k) for k in localization_data]
    rows = [[name] + [float(np.sum(localization_data[mode][indices])) for mode in localization_data]
            for name, indices in fragments.items()]
    return headers, rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class HessianLocalizer(CLICommand, name='frog-hessian'):
    """Localize vibrational normal modes on atoms and fragments.

Reads the normal modes of an ORCA frequency calculation and reports how much
each atom (or user-defined fragment) contributes to each selected mode, optionally with
the inverse participation ratio, and can write displaced geometries along each mode.

Mode numbers are those printed in the ORCA output (VIBRATIONAL FREQUENCIES /
NORMAL MODES): 0-based, translations and rotations included, so the imaginary
mode of a transition state of a non-linear molecule is mode 6.

Fragment syntax (--fragment):
    --fragment 'name:0,1,2'   fragment named 'name' made of atoms 0, 1, 2
    Multiple fragments: --fragment 'CO2:0,1,2' --fragment 'NH3:3,4,5,6'
"""
    category = 'parsing'
    outputs = [
        Output('table', 'Atomic Contributions (<file>)',
               'One row per atom above the threshold, one column per mode: contribution (%)'),
        Output('table', 'Inverse Participation Ratio (<file>)',
               'One row per mode: frequency (cm⁻¹) and IPR', when='--ipr'),
        Output('table', 'Fragment Contributions (<file>)',
               'One row per fragment, one column per mode: contribution (%)', when='--fragment'),
        Output('file', '<stem>_mode<m>_displ.xyz',
               'Multi-frame XYZ of the geometry displaced along mode m', when='-d'),
    ]
    guide = Guide(
        why="""
            A frequency calculation gives one vibration (normal mode) per 3N−6 degrees of freedom, but the
            output only lists numbers. This command tells you **which atoms move** in a given mode. Its main
            use is checking a **transition state**: the single imaginary frequency (printed as a negative
            number) must move the atoms of the bond being formed/broken; if it mostly moves a methyl rotor
            or a floppy side chain, the TS is not the one you wanted. It also helps to assign IR bands
            (e.g. "is the 1720 cm⁻¹ band the C=O on fragment A or B?") and to write a short movie of a mode
            (displaced geometries) to look at in a viewer, or to restart an optimisation away from a saddle point.
            """,
        needs="""
            An **ORCA output of a frequency calculation**: `! Freq` (or `! NumFreq`, also combined as
            `! Opt Freq` / `! OptTS Freq`). The file must contain the vibrational frequencies and the normal
            modes (ORCA prints them by default in the `NORMAL MODES` block). Gaussian outputs are not
            supported: the Gaussian parser reads frequencies but not normal modes. Atom
            numbers for `--fragment` are **0-based** (first atom = 0); mode numbers are those **printed in the
            ORCA output** (0-based, the zero frequencies of translations/rotations included: for a non-linear
            molecule modes 0-5 are zero and the first vibration, e.g. the imaginary mode of a TS, is mode 6).
            """,
        how=[
            "Check a TS: `frog-hessian ts.out` (by default `-m -1`, i.e. all imaginary modes) and look at "
            "which atoms carry the motion.",
            "Look at other modes with `-m 6 7 10` (numbers as in the ORCA output), or `--all-modes` for every "
            "vibration; "
            "lower `-t` (default 5 %) to see smaller contributions.",
            "Group atoms with `--fragment \"Sub:0,1,2\" --fragment \"Cat:3,4,5,6\"` (name, colon, comma-separated "
            "0-based indices; ranges such as `0-5` are **not** accepted here and the fragment is skipped "
            "with a warning). Add `--ipr` to measure how delocalised each mode is.",
            "Write the motion as a multi-frame XYZ with `-d` (tune with `--max-displacement` and "
            "`--number-geometries`; `--out-dir` to choose the folder) and open `<stem>_mode<m>_displ.xyz` as an "
            "animation in a viewer (Avogadro, Chemcraft, VMD, ChimeraX).",
        ],
        expect="""
            For each file, an *Atomic Contributions* table: one column per mode, labelled with its frequency
            (negative = imaginary), one row per atom (element + 0-based index, e.g. `C40`), value = % of the
            mode carried by that atom. Atoms below `-t` are shown as `-` and atoms below it in every mode are
            left out, so a column does not have to sum to 100. With `--fragment`, a table of fragment sums (%);
            with `--ipr`, one row per mode with the inverse participation ratio.

            * The % of atom *i* is |d_i|² / Σ|d_j|² of the normal-mode vector as printed by the program
              (ORCA prints mass-weighted, normalised displacements), so light atoms (H) tend to show large
              contributions.
            * IPR = Σ p_i² (p_i = fraction on atom *i*): **1** = the mode is on a single atom, about **1/N** =
              spread evenly over all N atoms. A clean TS mode on a few atoms has IPR ≳ 0.2; a value near 1/N
              means a collective, delocalised motion.
            * Frames of `-d` go from −`--max-displacement` to +`--max-displacement` (default ±0.6) times the
              normalised mode vector, so the atom that moves most is displaced by somewhat less than that (Å).
            * "No imaginary frequencies found" with the default `-m -1` means the structure is a minimum:
              pass explicit mode numbers. A minimum with a small imaginary mode (|ν| < 50 cm⁻¹) is often
              numerical noise from a loose optimisation or a coarse grid.

            Example (`opt.out`, TS from `! OptTS Freq`, 96 atoms): one imaginary mode, −180.9 cm⁻¹, with
            C40 63.29 %, C49 15.60 %, H95 6.75 % and IPR 0.4320, i.e. a mode localised on two carbons.
            """,
    )
    examples = [
        ('frog-hessian ts.out', 'Localize the imaginary mode(s) of a transition state'),
        ('frog-hessian freq.out -m 6 7 --ipr --fragment "CO2:0,1,2" --fragment "NH3:3,4,5,6"',
         'Atomic and fragment contributions with IPR for modes 6 and 7 (the first two vibrations)'),
        ('frog-hessian ts.out -d --max-displacement 0.4 --number-geometries 20',
         'Write displaced geometries along the imaginary mode(s)'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help="ORCA output file(s) with frequencies (Gaussian normal modes are not parsed)")
        parser.add_argument('-m', '--mode', nargs='+', default=[-1], type=int, metavar='MODE',
                            help="Normal mode(s) to localize, numbered as in the ORCA output (0-based, "
                                 "translations/rotations included: the first vibration of a non-linear "
                                 "molecule is mode 6); -1 selects all imaginary modes")
        parser.add_argument('-t', '--threshold', type=float, default=5.0, metavar='PCT',
                            help='Minimum atomic contribution shown, in %%')
        parser.add_argument('--ipr', action='store_true',
                            help='Also print the inverse participation ratio of each mode')
        parser.add_argument('--fragment', metavar='NAME:IDX1,IDX2,...', action='append', default=[],
                            help="Fragment for group-level localization: NAME:i,j,k with comma-separated "
                                 "0-based atom indices, no ranges (repeatable)")
        parser.add_argument('--all-modes', action='store_true',
                            help='Analyze all vibrational modes (overrides -m)')

        displ = parser.add_argument_group('displacement')
        displ.add_argument('-d', '--displace', action='store_true',
                           help="Write geometries displaced along each selected mode")
        displ.add_argument('--max-displacement', type=float, default=0.6, metavar='Å',
                           help='Maximum displacement in Å')
        displ.add_argument('--number-geometries', type=int, default=30, metavar='N',
                           help='Number of frames in each displacement trajectory')
        self.add_out_dir_arg(parser)

    def run(self, args):
        fragments = parse_fragments(args.fragment)
        if fragments:
            print_info(f"Fragments defined: {fragments}")

        files = self.require_files(args.file)
        failed = []
        for p in files:
            file = str(p)
            print_info(f"Processing file: {file}")

            try:
                calc = load_calculation(file)
            except Exception as e:
                print_error(f"Failed to parse {file}: {e}")
                failed.append(file)
                continue

            if getattr(calc, 'freq', None) is None or getattr(calc, 'normal_modes', None) is None:
                print_error(f"File {file} contains no frequency data")
                failed.append(file)
                continue

            # Mode numbers are those printed by ORCA (0-based, translations/rotations
            # included as zero frequencies), i.e. column indices of calc.normal_modes.
            n_modes = len(calc.freq)

            if args.all_modes:
                modes = np.flatnonzero(calc.freq != 0).tolist()
                print_info(f"Analyzing all {len(modes)} vibrational modes")
            elif args.mode == [-1]:
                modes = np.flatnonzero(calc.freq < 0).tolist()
                if not modes:
                    print_warning(f"No imaginary frequencies found in {file}")
                    continue
                print_info(f"Selected imaginary modes: {modes}")
            else:
                bad = [m for m in args.mode if not 0 <= m < n_modes]
                if bad:
                    print_error(f"Mode(s) {bad} out of range for {file} (modes 0-{n_modes - 1})")
                    failed.append(file)
                    continue
                modes = args.mode
                zero = [m for m in modes if calc.freq[m] == 0]
                if zero:
                    print_warning(f"Mode(s) {zero} have zero frequency (translation/rotation)")

            localization_data = {mode: localize_mode(calc, mode) for mode in modes}

            if args.displace:
                for mode in modes:
                    path = self.derived_path(p, f'mode{mode}_displ', '.xyz')
                    displace_geometry(calc, mode, max_displ=args.max_displacement,
                                      ngeom=args.number_geometries, path=path)
                    self.saved(path, 'geometries')

            headers, rows = _build_atom_table(calc, localization_data, args.threshold)
            if rows:
                self.output(rows, headers=headers, title=f"Atomic Contributions ({file})", floatfmt='.2f')
            else:
                print_warning(f"No atoms with contributions above {args.threshold}% in {file}")

            if args.ipr:
                ipr_rows = [[mode, float(calc.freq[mode]), float(inverse_participation_ratio(calc, mode))]
                            for mode in modes]
                self.output(ipr_rows, headers=["Mode", "Frequency (cm⁻¹)", "IPR"],
                            title=f"Inverse Participation Ratio ({file})", floatfmt=['d', '.1f', '.4f'])

            if fragments:
                frag_headers, frag_rows = _build_fragment_table(calc, localization_data, fragments)
                self.output(frag_rows, headers=frag_headers,
                            title=f"Fragment Contributions ({file})", floatfmt='.2f')

        if failed:
            raise InputError(f"{len(failed)} of {len(files)} file(s) could not be analyzed: {', '.join(failed)}")


def parse_args():
    return HessianLocalizer().build_parser().parse_args()

def main():
    HessianLocalizer().main()


if __name__ == '__main__':
    main()
