"""Sort conformers by a dihedral angle (E/Z, cis/trans or s-cis/s-trans).

Andrea Pellegrini, 2026
"""

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.filetypes import load_frames
from frogchem.io import write_xyz
from frogchem.math_geometry import calculate_dihedral
from frogchem.printer import print_info


LABEL_SCHEMES = {
    'ez': ('Z', 'E', 'other'),
    'cis-trans': ('cis', 'trans', 'other'),
    'scis-strans': ('s-cis', 's-trans', 'other'),
}


class SortEZ(CLICommand, name='frog-sort-ez'):
    """Sort conformers into E/Z (cis/trans, s-cis/s-trans) classes by a dihedral angle.

    By default |dihedral| < threshold is Z (cis), otherwise E (trans). With --tolerance,
    only dihedrals within the tolerance of 0° (Z) or 180° (E) are classified and the
    rest is labelled 'other'.
    """
    category = 'geometry'
    outputs = [
        Output('table', 'Dihedral classes (<file>)',
               'One row per conformer (0-based, file order): dihedral (°) and class'),
        Output('table', 'Class counts (<file>)', 'Number of conformers in each class'),
        Output('file', '<stem>_<label>.xyz',
               "Conformers of each non-empty class (labels Z/E, cis/trans or scis/strans, plus 'other' "
               'with --tolerance), original comment lines kept', when='-w'),
    ]
    guide = Guide(
        why="""
            Some isomers are defined by whether a dihedral is near **0°** or near **180°**: E/Z alkenes and
            imines, cis/trans amides, *s-cis*/*s-trans* dienes and enones. A conformer search can mix them
            (or isomerise a double bond by mistake). This command classifies every conformer and can write one
            file per class, so that you can check isomer purity or treat each isomer separately.
            """,
        needs="""
            * A multi-structure `.xyz` (e.g. `crest_conformers.xyz`) or ORCA/Gaussian outputs (last
              geometry), all with the same atom order.
            * **Four atoms, 0-based**, defining the dihedral across the bond of interest: substituent – atom –
              atom – substituent (e.g. for an alkene R¹–C=C–R², the atoms R¹, C, C, R²).
            * Note that the command classifies by **geometry**, not by CIP priority: "Z" here means the two
              chosen substituents are on the same side. Choose the higher-priority substituents if you want
              the label to coincide with the true E/Z descriptor.
            """,
        how=[
            "Run `frog-sort-ez ensemble.xyz -i 2 1 3 4`: |dihedral| < 90° is Z (cis), otherwise E (trans); "
            "change the boundary with `-t`.",
            "To flag twisted structures instead of forcing them into a class, use a window: `--tolerance 30` "
            "accepts Z only within 30° of 0° and E only within 30° of 180°; anything in between is *other*.",
            "Pick the wording with `--notation ez` (default), `cis-trans` or `scis-strans`.",
            "Add `-w` (and `--out-dir DIR`) to write one file per non-empty class, e.g. `ensemble_Z.xyz`, "
            "`ensemble_E.xyz`, `ensemble_other.xyz` (`ensemble_scis.xyz`/`ensemble_strans.xyz` for s-cis/s-trans).",
        ],
        expect="""
            Two tables per file: *Dihedral classes* (one row per conformer, 0-based, with the dihedral in
            degrees and its class) and *Class counts*. Example (`ensemble.xyz`, dihedral 2-1-3-4, values from
            −134° to −34°): with the default 90° threshold Z = 8, E = 10; with `--tolerance 60 --notation
            cis-trans` cis = 5, trans = 9, other = 4.

            * The sign of the dihedral does not matter (−175° and +175° are both E/trans).
            * For a real double bond (C=C, C=N) dihedrals stay within ~10–20° of 0° or 180°; many *other* with a
              small tolerance, or values around 90°, mean a strongly twisted bond or a wrong choice of atoms.
              Single bonds (amide, s-cis/s-trans) spread more: use a larger tolerance (30–45°).
            * `--tolerance` must be below 90° and overrides `-t`; `-t` must be between 0 and 180°.
            """,
    )
    examples = [
        ('frog-sort-ez ensemble.xyz -i 0 1 2 3', 'E/Z classification with the 90° threshold'),
        ('frog-sort-ez ensemble.xyz -i 0 1 2 3 --tolerance 30 --notation cis-trans -w',
         'cis within 30° of 0°, trans within 30° of 180°; write ensemble_cis.xyz, ensemble_trans.xyz, '
         'ensemble_other.xyz'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ (multi-structure), ORCA or Gaussian files')
        parser.add_argument('-i', '--indices', nargs=4, type=int, required=True, metavar='IDX',
                            help='Four atom indices (0-based) defining the dihedral')
        parser.add_argument('-t', '--threshold', type=float, default=90.0,
                            help='Boundary angle in degrees: |dihedral| < threshold is Z (cis), '
                                 'otherwise E (trans)')
        parser.add_argument('--tolerance', type=float, default=None,
                            help='Window in degrees around 0° and 180°: |d| <= tol is Z (cis), '
                                 '|d| >= 180 - tol is E (trans), anything else is other; overrides -t')
        parser.add_argument('--notation', choices=list(LABEL_SCHEMES), default='ez',
                            help='Naming convention for the class labels')
        self.add_write_arg(parser, help='Write one XYZ file per class')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.file)
        threshold, tolerance, idx = args.threshold, args.tolerance, args.indices
        if not 0.0 < threshold <= 180.0:
            raise InputError(f"--threshold must be between 0 and 180°, got {threshold}")
        if tolerance is not None and not 0.0 <= tolerance < 90.0:
            raise InputError(f"--tolerance must be between 0 and 90°, got {tolerance}")

        cis_lbl, trans_lbl, other_lbl = LABEL_SCHEMES[args.notation]
        classes = [cis_lbl, trans_lbl] + ([other_lbl] if tolerance is not None else [])
        mode = f"tolerance ±{tolerance}° around 0°/180°" if tolerance is not None else f"threshold {threshold}°"

        for f in args.file:
            frames = load_frames(f)
            if not frames:
                raise InputError(f"No structures found in {f}")
            n_atoms = len(frames[0][0])
            bad = sorted({i for i in idx if not 0 <= i < n_atoms})
            if bad:
                raise InputError(f"Atom indices {bad} out of range for {f} ({n_atoms} atoms, 0-based)")

            print_info(f"Sorting {len(frames)} conformers from {f} ({mode}, notation {args.notation})")
            members = {lbl: [] for lbl in classes}
            rows = []
            for i, (_atoms, pos, _comment) in enumerate(frames):
                angle = float(calculate_dihedral(*(pos[j] for j in idx)))
                abs_angle = abs(angle)
                if tolerance is not None:
                    if abs_angle <= tolerance:
                        label = cis_lbl
                    elif abs_angle >= 180.0 - tolerance:
                        label = trans_lbl
                    else:
                        label = other_lbl
                else:
                    label = cis_lbl if abs_angle < threshold else trans_lbl
                members[label].append(i)
                rows.append([i, angle, label])

            self.output(rows, headers=['Conf', 'Dihedral (°)', 'Class'],
                        title=f"Dihedral classes ({f})", floatfmt='.2f')
            self.output([[lbl, len(ids)] for lbl, ids in members.items()],
                        headers=['Class', 'Count'], title=f"Class counts ({f})")

            if args.write:
                for label, ids in members.items():
                    if not ids:
                        continue
                    out = self.derived_path(f, label.replace('-', ''), '.xyz')
                    write_xyz(str(out),
                              [frames[i][0] for i in ids],
                              [frames[i][1] for i in ids],
                              [frames[i][2] or f"Conf {i}" for i in ids])
                    self.saved(out, f"{label} ensemble ({len(ids)} structure{'s' * (len(ids) != 1)})")


def parse_args():
    return SortEZ().build_parser().parse_args()


def main():
    SortEZ().main()


if __name__ == '__main__':
    main()
