"""Detect the point group of molecules and align them to their principal axes.

Without -t or -g, the point group is detected at several tolerances (0.1, 0.2, 0.5 and
1.0 Å). With -g, each target group is verified by scanning tolerances until it is found.
All atoms are used (true molecular point group); --exclude-h analyses the heavy-atom
framework only.

Andrea Pellegrini, 2025
"""

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.io import write_xyz
from frogchem.filetypes import load_structures
from frogchem.symmetry import (
    calculate_center_of_mass,
    get_inertia_tensor,
    align_molecule,
    detect_point_group,
)
from frogchem.printer import print_info, print_error

SCAN_TOLERANCES = [0.1, 0.2, 0.5, 1.0]
TARGET_SCAN_TOLERANCES = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5]


def rotor_type(evals, rtol: float = 0.02) -> str:
    """Classify the rotor from sorted principal moments of inertia (amu*A^2).

    Two moments count as degenerate when they agree within `rtol` (relative).
    """
    evals = np.sort(np.asarray(evals, dtype=float))
    if evals[0] < 0.1:
        return "Linear"
    if np.isclose(evals[0], evals[2], rtol=rtol):
        return "Spherical Top"
    if np.isclose(evals[0], evals[1], rtol=rtol) or np.isclose(evals[1], evals[2], rtol=rtol):
        return "Symmetric Top"
    return "Asymmetric Top"


class DetectSymmetry(CLICommand, name='frog-sym'):
    """Detect the point group of molecules and align them to their principal axes.

Without -t or -g, the point group is detected at several tolerances (0.1, 0.2, 0.5 and
1.0 Å). With -g, each target group is verified by scanning tolerances until it is found.
All atoms are used (true molecular point group); --exclude-h analyses the heavy-atom
framework only.
"""
    category = 'geometry'
    outputs = [
        Output('table', 'Rotor Properties',
               'One row per structure: rotor type, mass (amu), principal moments of inertia (amu·Å²)'),
        Output('table', 'Symmetry Scan',
               "One row per structure and tolerance (Å): point group and the 'RMSD' column, i.e. the largest "
               'displacement of any atom under the accepted operations (Å)', when='without -g'),
        Output('table', 'Target Verification',
               'One row per structure and target group: detected group, first tolerance at which it matched '
               "and the 'RMSD' (largest atom displacement, Å)", when='-g'),
        Output('file', '<stem>_std.xyz',
               'Geometry in the principal-axes frame (<stem>_std_conf<N>.xyz for multi-structure inputs)',
               when='-w'),
    ]
    guide = Guide(
        why="""
            Find the **point group** (the set of symmetry operations, written as a Schoenflies symbol: C1, Cs,
            C2v, D3h, Td, ...) of a computed structure. You need it to check that an optimisation kept the
            symmetry you expect (or broke it), to decide whether a structure can be run with symmetry, to
            get the **symmetry number** used in the rotational entropy of thermochemistry, and to report the
            symmetry of a minimum or transition state. It also prints the principal moments of inertia and can
            write the molecule in its standard (principal-axes) orientation.
            """,
        needs="""
            A geometry: `.xyz` (multi-structure allowed, each is analysed), ORCA or Gaussian output (last
            geometry, e.g. after `! Opt`). Nothing else. Optimised structures are rarely *exactly* symmetric, so
            the program accepts small deviations: the **tolerance** (in Å) is the largest displacement of an
            atom that still counts as symmetric.
            """,
        how=[
            "Run `frog-sym opt.xyz`: the point group is detected at four tolerances (0.1, 0.2, 0.5 and 1.0 Å).",
            "Take the group found at the **small** tolerances (0.1–0.2 Å) as the answer; groups that only "
            "appear at 0.5–1.0 Å are approximate symmetries of a distorted structure.",
            "To check a specific group, use `-g C2v` (several are allowed: `-g C2v D2h`): tolerances from "
            "0.05 to 1.5 Å are tried until exactly that group is found. Without `-g`, `-t 0.05` replaces the "
            "four-tolerance scan with a single tolerance (`-t` is ignored together with `-g`).",
            "Add `--exclude-h` to get the symmetry of the heavy-atom framework only (useful when methyl or "
            "other hydrogens break a symmetry you consider irrelevant).",
            "Add `-w` to write `<stem>_std.xyz` with the molecule in its principal-axes frame (centre of mass "
            "at the origin), a convenient starting orientation for inputs or figures.",
        ],
        expect="""
            *Rotor Properties*: rotor type (Linear, Spherical/Symmetric/Asymmetric Top), mass (amu) and the
            three principal moments of inertia Ia ≤ Ib ≤ Ic (amu·Å²). *Symmetry Scan*: one row per tolerance
            with the point group and an "RMSD" column, which is the **largest displacement of any atom** under
            the accepted symmetry operations (0.0000 for a perfectly symmetric structure).
            Examples: water C2v (0.0000 Å), methane Td, ethyne `D_inf_h` (D∞h; `C_inf_v` is C∞v), toluene
            **Cs** at 0.1 Å — the methyl hydrogens break the C2v symmetry of the ring — but **C2v** with
            `--exclude-h`.

            * Tolerances that are too large give nonsense: water becomes D8h at 0.5 Å and Td at 1.0 Å. Trust a
              group only if it already appears at 0.1–0.2 Å with a small displacement.
            * With `-g`, *Match* is True only if the detected group is **exactly** the target: a subgroup does
              not match (toluene with `--exclude-h` is C2v, so `-g Cs` gives False even though Cs ⊂ C2v). When
              there is no match, *Detected* shows the group found at the last tolerance tried (1.5 Å), which
              is usually meaningless.
            * `--exclude-h` on a molecule with a single heavy atom (water, ammonia) has no framework to
              analyse: all atoms are used instead, as without the flag.
            * Thermochemistry uses the symmetry number σ of the group (C2v → 2, C3v → 3, Td → 12); a wrong
              group changes the entropy by R·ln σ.
            """,
    )
    examples = [
        ('frog-sym mol.xyz', 'Point group at several tolerances'),
        ('frog-sym mol.xyz -g C2v D2h --exclude-h', 'Verify C2v and D2h for the heavy-atom framework'),
        ('frog-sym opt.out -t 0.2 -w', 'Point group at 0.2 Å and write the aligned geometry'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help="XYZ, ORCA or Gaussian file(s)")
        parser.add_argument('-t', '--threshold', type=float, default=None, metavar='Å',
                            help="Symmetry tolerance in Å; without it, several tolerances are scanned")
        parser.add_argument('-g', '--target-group', type=str, default=None, nargs='+', metavar='GROUP',
                            help="Target point group(s) to verify, e.g. C2v")
        parser.add_argument('--exclude-h', dest='exclude_h', action="store_true", default=False,
                            help="Ignore hydrogen atoms: point group of the heavy-atom framework")
        parser.add_argument('--include-h', dest='exclude_h', action="store_false",
                            help="Include hydrogen atoms (default; kept for backward compatibility)")
        self.add_write_arg(parser, help="Write the geometry aligned to the principal axes")
        self.add_out_dir_arg(parser)

    def run(self, args):
        thresholds = [args.threshold] if args.threshold is not None else SCAN_TOLERANCES

        rotor_rows, scan_rows, target_rows = [], [], []
        files = self.require_files(args.file)
        failed = []
        for p in files:
            file = str(p)
            print_info(f"Analyzing {file}...")
            try:
                structs = load_structures(file)
            except Exception as e:
                print_error(f"Error reading {file}: {e}")
                failed.append(file)
                continue

            for i, (atoms, positions) in enumerate(structs):
                conf = i if len(structs) > 1 else None

                _, mass = calculate_center_of_mass(atoms, positions)
                evals, _ = get_inertia_tensor(atoms, positions)
                evals = np.sort(np.real(evals))
                rotor_rows.append([file, conf, rotor_type(evals), float(mass)] + [float(e) for e in evals])

                if args.target_group:
                    for pg in args.target_group:
                        found, detected = None, None
                        for tol in TARGET_SCAN_TOLERANCES:
                            res = detect_point_group(atoms, positions, tolerance=tol, exclude_h=args.exclude_h)
                            detected = res['sch_symbol']
                            if res['is_valid'] and detected.lower() == pg.lower():
                                found = (tol, float(res['error_rmsd']))
                                break
                        if found:
                            target_rows.append([file, conf, pg, detected, True, found[0], found[1]])
                        else:
                            target_rows.append([file, conf, pg, detected, False, None, None])
                else:
                    for tol in thresholds:
                        res = detect_point_group(atoms, positions, tolerance=tol, exclude_h=args.exclude_h)
                        if res['is_valid']:
                            scan_rows.append([file, conf, tol, res['sch_symbol'], float(res['error_rmsd'])])
                        else:
                            scan_rows.append([file, conf, tol, None, None])

                if args.write:
                    tag = "std" if len(structs) == 1 else f"std_conf{i}"
                    out_path = self.derived_path(p, tag, '.xyz')
                    write_xyz(str(out_path), [atoms], [align_molecule(atoms, positions)],
                              ["Standard orientation (principal axes)"])
                    self.saved(out_path, 'geometry')

        self.output(rotor_rows, headers=["File", "Conf", "Rotor Type", "Mass (amu)",
                                         "Ia (amu·Å²)", "Ib (amu·Å²)", "Ic (amu·Å²)"],
                    title="Rotor Properties", floatfmt='.2f')
        if args.target_group:
            self.output(target_rows, headers=["File", "Conf", "Target", "Detected", "Match",
                                              "Tolerance (Å)", "RMSD (Å)"],
                        title="Target Verification", floatfmt='.4f')
        else:
            self.output(scan_rows, headers=["File", "Conf", "Tolerance (Å)", "Point Group", "RMSD (Å)"],
                        title="Symmetry Scan", floatfmt='.4f')

        if failed:
            raise InputError(f"{len(failed)} of {len(files)} file(s) could not be read: {', '.join(failed)}")


def parse_args():
    return DetectSymmetry().build_parser().parse_args()

def main():
    DetectSymmetry().main()


if __name__ == '__main__':
    main()
