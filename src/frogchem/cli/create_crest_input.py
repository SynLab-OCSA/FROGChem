#!/usr/bin/env python3
"""Write a CREST constraint input that fixes atoms, distances, angles and dihedrals."""

import os
from pathlib import Path

import numpy as np

from frogchem.exceptions import InputError
from frogchem.filetypes import load_structures
from frogchem.io import write_crest_constraints, write_xyz
from frogchem.math_geometry import calculate_distance, calculate_angle, calculate_dihedral
from frogchem.printer import print_info, print_warning
from frogchem.cli.base import CLICommand, Guide, Output


# Map number of atoms of an -i set to function, CREST keyword and unit
INTERNAL_FUNCS = {
    2: (calculate_distance, 'distance', 'Å'),
    3: (calculate_angle, 'angle', '°'),
    4: (calculate_dihedral, 'dihedral', '°'),
}


def atoms_in_spheres(geom, centers, radius: float) -> set:
    """0-based indices of the atoms within `radius` Å of any atom in `centers` (centres included)."""
    geom = np.asarray(geom, dtype=float)
    dist = np.linalg.norm(geom[:, None, :] - geom[list(centers)][None, :, :], axis=-1)
    return {int(i) for i in np.flatnonzero((dist <= radius).any(axis=1))}


class CreateCrestInput(CLICommand, name='frog-crest'):
    """Write a CREST constraint input that fixes atoms, distances, angles and dihedrals.

    Atoms (-a, --mc, or an -i set of 1 or 5+ atoms) are fixed to a reference geometry: 'reference='
    points to the input structure, relative to the folder of the constraint file. ORCA/Gaussian
    outputs and multi-structure XYZ files get a single-structure reference '<stem>_ref.xyz'
    (first structure) next to the constraint file. Each -i set of 2, 3 or 4 atoms, as in
    frog-internals, fixes a distance, angle or dihedral to its value in that structure.
    Atoms are written in CREST's 1-based numbering, force constant 0.5.
    Use it with 'crest <structure> --cinp <file>'.
    """
    category = 'geometry'
    has_tables = False
    outputs = [
        Output('file', 'constrain.inp', 'Constraint input for one input file (-o sets the path)'),
        Output('file', '<stem>_constrain.inp',
               'One constraint input per file when several inputs are given '
               '(next to each input, or --out-dir; the -o file name sets the tag)'),
        Output('file', '<stem>_ref.xyz',
               'Reference geometry, written only when atoms are fixed and the input is an '
               'ORCA/Gaussian or multi-structure XYZ file'),
    ]
    guide = Guide(
        why="""
            CREST (Grimme's conformer search program, based on xtb) explores *all* conformers of a
            molecule. Often you want it to move only part of the system: keep a metal and its donor atoms,
            a reacting core or a transition-state geometry fixed, or hold a single bond length, angle or
            torsion, and let the rest rotate. CREST does this with a *constraint file*; this command writes
            that file for you, so you do not have to renumber atoms or measure values by hand. Use it before
            a constrained CREST run.
            """,
        needs="""
            * The **structure** you want to keep as reference: an `.xyz` file, or an ORCA/Gaussian output
              (the last geometry is used, e.g. from an `! Opt` job). For a multi-structure XYZ only the
              first structure is used.
            * The **atom numbers**, **0-based** (the first atom of the file is 0), as in all FROG
              commands. Look them up in a viewer that shows 0-based indices, or subtract 1 from the numbers
              shown by GaussView/Avogadro/ORCA's 1-based lists.
            * CREST installed to run the result (FROG does not run CREST).
            """,
        how=[
            "Freeze atoms in place with `-a`: `frog-crest struc.xyz -a 0 4 50 83` writes `constrain.inp` in "
            "the current folder (choose another name/folder with `-o`).",
            "Fix internal coordinates with one `-i` per coordinate, exactly as in `frog-internals`: 2 atoms "
            "for a distance, 3 for an angle, 4 for a dihedral, e.g. `-i 0 50 -i 50 0 83 -i 50 0 4 3`. The "
            "target value is measured on the reference structure. A set of 1 or of 5 or more atoms "
            "(`-i 7`, `-i 2 3 36 37 38`) freezes those atoms like `-a`; to freeze 2-4 atoms use `-a`.",
            "To freeze a whole coordination sphere, give the metal(s) and a radius: "
            "`frog-crest struc.xyz --mc 0 -r 2.6` freezes atom 0 and every atom within 2.6 Å of it; with "
            "several centres (`--mc 0 12`) the spheres are merged. The radius is measured from the atom "
            "centres on the reference geometry (the first structure). `-a`, `-i` and `--mc` can be combined.",
            "By default the frozen atoms still feel the metadynamics bias (CREST pushes on every atom). "
            "Add `--exclude-mtd` to leave the frozen atoms (`-a`, `--mc`, `-i` sets of 1 or 5+) out: a `$metadyn` "
            "block then lists only the other atoms, so the bias drives the free part of the molecule and the "
            "frozen core is not pulled against its constraints. Atoms of distances, angles and dihedrals stay "
            "in the bias.",
            "If atoms are frozen and the input is an ORCA/Gaussian output or a multi-structure XYZ, the "
            "reference geometry is also written as `<stem>_ref.xyz` next to the constraint file; a "
            "single-structure XYZ is referenced directly (path relative to the constraint file).",
            "Open `constrain.inp` and check that the atom lists are what you meant: the numbers there are "
            "**1-based** (CREST's numbering), i.e. every index you typed plus 1.",
            "Run CREST from the folder of the constraint file, e.g. `crest struc.xyz --cinp constrain.inp` "
            "(add your usual CREST options: method, solvent, threads).",
            "Several inputs at once (`frog-crest a.xyz b.xyz -a 0 1`) give `a_constrain.inp`, "
            "`b_constrain.inp` next to each input (or in `--out-dir`); then `-o` only sets the name tag. "
            "Values of `-i` coordinates are measured on each file.",
        ],
        expect="""
            No table: one or two files are written. The constraint file looks like this (from the 18-conformer
            Ag complex `ensemble.xyz`, `-a 0 4 50 83 -i 50 0 83 -o crest/constrain.inp`):

            ```text
            $constrain
              atoms: 1,5,51,84
              angle: 51, 1, 84, 89.5748
              force constant=0.5
              reference=ensemble_ref.xyz
            $end
            ```

            * Atom numbers are 1-based: 0 4 50 83 became 1,5,51,84. This is correct, not an error.
            * Distances are in Å, angles and dihedrals in degrees, measured on the reference structure
              (the same values `frog-internals` prints for it); edit them by hand to fix a different value.
            * The force constant is fixed at 0.5 (xtb units, the usual CREST/xtb value); edit the file by
              hand if you need stiffer or softer constraints.
            * A warning *"contains 18 structures: the first is the reference"* means a multi-structure file
              was given; make sure the first structure is the one you want.
            * `Atom indices [...] out of range` means you used 1-based numbers or the wrong file: valid
              indices go from 0 to N−1.
            * The constraint file refers to the reference by a relative path: if you move it, move the
              reference with it (or edit `reference=`).
            """,
    )
    examples = [
        ('frog-crest struc.xyz -a 0 1 2 5', 'Freeze atoms 0, 1, 2 and 5 (0-based) to struc.xyz'),
        ('frog-crest struc.xyz -i 0 1 -i 0 1 2 -i 0 1 2 3',
         'Fix distance 0-1, angle 0-1-2 and dihedral 0-1-2-3 to their values in struc.xyz'),
        ('frog-crest a.xyz b.xyz -a 0 1', 'Write a_constrain.inp and b_constrain.inp'),
        ('frog-crest struc.xyz --mc 0 -r 2.6', 'Freeze atom 0 and every atom within 2.6 Å of it'),
        ('frog-crest struc.xyz --mc 0 -r 2.6 --exclude-mtd',
         'Same, and leave those atoms out of the metadynamics bias'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ, ORCA or Gaussian reference structure(s)')
        parser.add_argument('-i', '--indices', nargs='+', type=int, action='append', metavar='IDX',
                            help='Atom indices, 0-based: 2 (distance), 3 (angle), 4 (dihedral), '
                                 '1 or 5+ (fixed atoms); repeatable')
        parser.add_argument('-a', '--atoms', nargs='+', type=int, default=[], metavar='IDX',
                            help='Atoms fixed to the reference geometry (0-based)')
        parser.add_argument('--mc', '--metal-center', dest='metal_center', nargs='+', type=int, default=[],
                            metavar='IDX',
                            help='Metal centres (0-based): every atom within --radius of any of them is '
                                 'fixed, the centres included')
        parser.add_argument('-r', '--radius', type=float, metavar='Å',
                            help='Sphere radius in Å around each --mc centre')
        parser.add_argument('--exclude-mtd', action='store_true',
                            help='Leave the fixed atoms out of the metadynamics bias '
                                 '($metadyn block listing all the other atoms)')
        self.add_output_arg(parser, default='constrain.inp',
                            help='Constraint file for a single input; with several inputs only its '
                                 'name is used, as tag of <stem>_constrain.inp')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.require_files(args.file)
        groups = args.indices or []
        internals = [idxs for idxs in groups if len(idxs) in INTERNAL_FUNCS]
        for idxs in internals:
            if len(set(idxs)) != len(idxs):
                raise InputError(f"Invalid index set {idxs}: repeated atom")
        # sets of 1 or 5+ indices are fixed atoms
        fixed = set(args.atoms) | {i for idxs in groups if len(idxs) not in INTERNAL_FUNCS for i in idxs}
        if not groups and not args.atoms and not args.metal_center:
            raise InputError("Give the constraints with -a, -i and/or --mc")
        if bool(args.metal_center) != (args.radius is not None):
            raise InputError("--mc and -r/--radius must be given together")
        if args.radius is not None and args.radius <= 0:
            raise InputError(f"--radius must be positive, got {args.radius}")
        multi = len(args.file) > 1
        tag_path = Path(args.output)
        if multi and tag_path.parent != Path('.'):
            raise InputError("With several inputs -o takes a file name only (e.g. -o constrain.inp); "
                             "use --out-dir to choose the folder")
        inputs = {Path(f).resolve() for f in args.file}
        all_indices = set(args.metal_center) | fixed | {i for idxs in internals for i in idxs}

        for f in args.file:
            structs = load_structures(f)
            if not structs:
                raise InputError(f"No geometry found in {f}")
            symbols, geom = structs[0]
            n_atoms = len(symbols)
            bad = sorted(i for i in all_indices if not 0 <= i < n_atoms)
            if bad:
                raise InputError(f"Atom indices {bad} out of range for {f} ({n_atoms} atoms, 0-based)")
            atoms = set(fixed)
            if args.metal_center:
                sphere = atoms_in_spheres(geom, args.metal_center, args.radius)
                print_info(f"{len(sphere)} atoms within {args.radius:g} Å of centres {args.metal_center}")
                atoms |= sphere
            atoms = sorted(atoms)
            values = []
            for idxs in internals:
                calc_func, kind, unit = INTERNAL_FUNCS[len(idxs)]
                value = float(calc_func(*[geom[i] for i in idxs]))
                values.append((idxs, value))
                print_info(f"{kind} {'-'.join(f'{symbols[i]}{i}' for i in idxs)} = {value:.4f} {unit}")
            metadyn = None
            if args.exclude_mtd:
                if not atoms:
                    print_warning("--exclude-mtd has no effect without fixed atoms (-a, --mc)")
                metadyn = sorted(set(range(n_atoms)) - set(atoms))
                if not metadyn:
                    raise InputError(f"All {n_atoms} atoms of {f} are fixed: "
                                     "no atom left for the metadynamics with --exclude-mtd")

            if multi:
                out = self.derived_path(f, tag_path.stem, tag_path.suffix or '.inp')
            else:
                out = self.output_path('constrain.inp', '.inp', known_exts=['.inp', '.c'])
                if out.resolve() in inputs:
                    raise InputError(f"Refusing to overwrite the input file {out}")
                out.parent.mkdir(parents=True, exist_ok=True)

            if len(structs) > 1:
                print_warning(f"{f} contains {len(structs)} structures: the first is the reference")
            reference = None
            if atoms and f.lower().endswith('.xyz') and len(structs) == 1:
                reference = os.path.relpath(Path(f).resolve(), out.resolve().parent)
            elif atoms:
                ref_path = out.parent / f"{Path(f).stem}_ref.xyz"
                if ref_path.resolve() in inputs:
                    raise InputError(f"Refusing to overwrite the input file {ref_path}")
                write_xyz(str(ref_path), [symbols], [geom], [f"Reference from {Path(f).name}"])
                self.saved(ref_path, 'reference geometry')
                reference = ref_path.name
            print_info(f"Constraining {len(atoms)} atoms and {len(values)} internal coordinates of {f}")
            write_crest_constraints(fname=str(out), atom_indexes=atoms, reference_file=reference,
                                    metadyn_atoms=metadyn, internals=values)
            self.saved(out, 'CREST constraints')


# backward compat
def parse_args():
    return CreateCrestInput().build_parser().parse_args()


def main():
    CreateCrestInput().main()


if __name__ == '__main__':
    main()
