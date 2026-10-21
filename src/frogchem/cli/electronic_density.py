#!/usr/bin/env python3
"""Build electron density, spin density and molecular orbital cube files
from ORCA/Gaussian output, from the command line or in an interactive session.
"""

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.density_builder import build_density_cube, build_orbital_cube, build_spin_density_cube
from frogchem.exceptions import InputError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_error, print_info, print_warning


def choose_file() -> str:
    from InquirerPy import inquirer
    from InquirerPy.validator import PathValidator
    return inquirer.filepath(
        message="ORCA/Gaussian output file:",
        default=os.getcwd(),
        validate=PathValidator(is_file=True, message="Not a valid file"),
        only_files=True,
    ).execute()


def get_frontier_orbitals(calc, beta=False) -> Tuple[Optional[int], Optional[int]]:
    """Return (homo_idx, lumo_idx) based on MO occupations, or (None, None)."""
    if hasattr(calc, 'mult') and calc.mult != 1 and hasattr(calc, 'alpha_occ'):
        occ = calc.beta_occ if beta else calc.alpha_occ
        occ = np.asarray(occ)
    elif getattr(calc, 'homo', None) is not None:
        return calc.homo, calc.lumo
    elif hasattr(calc, 'occupation') and calc.occupation is not None and len(calc.occupation) > 0:
        occ = np.asarray(calc.occupation)
    else:
        return None, None

    occ_idx = np.where(occ > 0)[0]
    unocc_idx = np.where(occ == 0)[0]
    homo = int(occ_idx[-1]) if len(occ_idx) > 0 else None
    lumo = int(unocc_idx[0]) if len(unocc_idx) > 0 else None
    return homo, lumo



def _positive_float(text: str, allow_zero: bool = False) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return value >= 0 if allow_zero else value > 0


class ElectronicDensity(CLICommand, name='frog-density'):
    """Build electron density, spin density and molecular orbital cube files.

    Reads the basis set and MO coefficients from ORCA output (! PrintBasis PrintMOs) or a Gaussian log
    (pop=full gfinput). Without a selection the total electron density is computed. Use --interactive for a guided session.
    """

    category = 'spectroscopy'
    has_tables = False
    outputs = [
        Output('file', '<stem>_density.cube', 'Total electron density (e/bohr³)', when='--density, or no selection'),
        Output('file', '<stem>_spin_density.cube', 'Spin density, alpha - beta (open-shell only)', when='--spin'),
        Output('file', '<stem>_homo.cube', 'HOMO', when='--homo'),
        Output('file', '<stem>_lumo.cube', 'LUMO', when='--lumo'),
        Output('file', '<stem>_orbital_<N>.cube', 'Orbital N (0-based)', when='--orbital'),
    ]
    guide = Guide(
        why="""
            To **see** orbitals and densities you need them on a 3D grid, as cube files that any viewer can draw
            as isosurfaces. This command builds them directly from the output file, without the program's own
            plotting tool (`orca_plot`) or the binary `.gbw` file: the HOMO and LUMO for a frontier-orbital
            figure, any orbital listed by `frog-mo` or `frog-tddft`, the total electron density, or the
            **spin density** of a radical or open-shell complex (the input for `frog-spin`). Cubes made with
            the same grid options can then be combined with `frog-cube`.
            """,
        needs="""
            An output that **contains the basis set and all the MO coefficients** (a single point on the
            final geometry is enough):

            * **ORCA**: add `! PrintBasis PrintMOs` to the input.
            * **Gaussian 16**: `#p ... pop=full iop(3/33=1) iop(6/80=1) gfinput` (`gfinput` prints the basis,
              `pop=full` every MO). Pure (`5D 7F`) and Cartesian (`6D 10F`) d/f functions both work, g functions
              do not; no `.chk`, `formchk` or `cubegen` is needed.

            Without them the command stops with an error telling you so. A spin density needs an open-shell
            calculation (multiplicity 2, 3, ...). Orbital numbers are **0-based**, as in ORCA's
            `ORBITAL ENERGIES` table (Gaussian's MO 1 is orbital 0); open-shell calculations have separate α
            and β sets (`--beta`).
            """,
        how=[
            "Frontier orbitals: `frog-density job.out --homo --lumo` writes `job_homo.cube` and `job_lumo.cube` "
            "next to the output (or in `--out-dir`).",
            "Other cubes: `--density` (total density; also the default when no option is given), `--spin` "
            "(spin density, α − β), `--orbital 41 42` (0-based MO numbers); add `--beta` for β orbitals of an "
            "open-shell run.",
            "Control the grid: `--spacing` (Å, default 0.15) or `--voxels N` (N points per axis, overrides the "
            "spacing) and `--padding` (Å of empty space around the molecule, default 3.0). Use `--voxels 50` "
            "for a quick look, the default spacing or finer for numbers (e.g. `frog-spin`).",
            "Name the files with `-o name.cube` (one cube) or `-o prefix` (several cubes: `prefix_homo.cube`, "
            "...), or use `--interactive` for a guided session.",
            "Open the cube in a viewer: isovalue about ±0.03–0.05 for orbitals, 0.02–0.05 e/bohr³ for the total "
            "density, ±0.002–0.005 for a spin density.",
        ],
        expect="""
            One `.cube` file per requested quantity and a *Saved cube* line for each, preceded by the grid
            size and the value range of the cube (hidden with `-q`). Values are in atomic units: e/bohr³ for densities, bohr^−3/2
            for orbitals (whose sign is arbitrary: the two lobes have opposite colours, and swapping them changes
            nothing physically).

            * File size and time grow with the number of points: a 50×50×50 grid is about 1.7 MB, while the
              default 0.15 Å spacing on a large molecule can give tens of MB and take minutes.
            * A spin density of a closed-shell (singlet) calculation is zero: the command prints a warning and
              writes no file.
            * If an isosurface looks cut at the box edge, increase `--padding`; if it looks blocky, use a
              finer grid.
            * HOMO and LUMO are taken from the occupations (α set by default; β with `--beta`).

            Example (water, B3LYP/def2-TZVPP with `! PrintBasis PrintMOs`, `--density --homo --lumo --voxels 50`):
            three 50×50×50 cubes of 1.75 MB each; HOMO = orbital 4 (values ±0.631), LUMO = orbital 5.
            """,
    )
    examples = [
        ('frog-density opt.out', 'Total electron density cube next to opt.out'),
        ('frog-density opt.out --homo --lumo --voxels 60 -o frontier',
         'Write frontier_homo.cube and frontier_lumo.cube on a 60x60x60 grid'),
        ('frog-density opt.out --spin --orbital 41 42 --beta', 'Spin density and beta orbitals 41 and 42'),
    ]

    def setup_args(self, parser):
        parser.add_argument('input', nargs='?', help='ORCA output with ! PrintBasis PrintMOs, or Gaussian log '
                                                     'with pop=full gfinput')
        parser.add_argument('--density', action='store_true', help='Compute the total electron density')
        parser.add_argument('--spin', action='store_true', help='Compute the spin density')
        parser.add_argument('--orbital', type=int, nargs='+', metavar='N',
                            help='Compute orbital(s) by index (0-based)')
        parser.add_argument('--homo', action='store_true', help='Compute the HOMO')
        parser.add_argument('--lumo', action='store_true', help='Compute the LUMO')
        parser.add_argument('--beta', action='store_true', help='Use beta orbitals (open-shell systems)')
        self.add_grid_args(parser, spacing=0.15, padding=3.0)
        self.add_output_arg(parser, help='Output cube path, or prefix ({prefix}_{tag}.cube) when several '
                                         'cubes are written; without it, cubes go next to the input')
        self.add_out_dir_arg(parser)
        self.add_interactive_arg(parser, help='Choose the file and the cubes interactively')

    def run(self, args):
        if args.interactive:
            filepath = args.input or choose_file()
            self.require_files([filepath])
            calc = load_calculation(filepath)
            self._check_basis(calc, filepath)
            self._interactive_session(calc, filepath, args)
            return

        if not args.input:
            raise InputError('No input file given (or use --interactive)')
        self.require_files([args.input])
        calc = load_calculation(args.input)
        self._check_basis(calc, args.input)
        self._run_cli(calc, args)

    @staticmethod
    def _check_basis(calc, path):
        if (getattr(calc, 'mask_atoms', None) is None or getattr(calc, 'orbital_coefficients', None) is None
                or not getattr(calc, 'basis_coeff', None)):
            raise InputError(f'{path} does not contain basis set / MO coefficient data '
                             '(ORCA: ! PrintBasis PrintMOs; Gaussian: pop=full gfinput)')

    def _build(self, op_type, calc, out_path, val=None, beta=False):
        """Run a cube builder (its chatter goes to stderr) and report the file."""
        a = self.args
        grid = dict(spacing=a.spacing, padding=a.padding, n_voxels=a.voxels)
        if op_type == 'spin' and getattr(calc, 'mult', 1) == 1:
            print_warning('Closed-shell calculation (multiplicity 1): spin density is zero, cube not written')
            return
        if op_type == 'density':
            print_info(f'Building electron density cube -> {out_path}')
            build_density_cube(calc, output=str(out_path), **grid)
        elif op_type == 'spin':
            print_info(f'Building spin density cube -> {out_path}')
            build_spin_density_cube(calc, output=str(out_path), **grid)
        else:
            print_info(f'Building orbital {val} cube -> {out_path}')
            build_orbital_cube(calc, val, output=str(out_path), beta=beta, **grid)
        self.saved(out_path, 'cube')

    @staticmethod
    def _n_mos(calc, beta=False):
        if getattr(calc, 'mult', 1) != 1:
            coeff = getattr(calc, 'beta_coeff' if beta else 'alpha_coeff', None)
            if coeff is not None:
                return coeff.shape[1]
        return calc.orbital_coefficients.shape[1]

    def _run_cli(self, calc, args):
        any_flag = args.density or args.spin or args.orbital is not None or args.homo or args.lumo
        operations: List[Tuple[str, Any, str]] = []
        if args.density or not any_flag:
            operations.append(('density', None, 'density'))
        if args.spin:
            operations.append(('spin', None, 'spin_density'))
        if args.homo or args.lumo:
            homo_idx, lumo_idx = get_frontier_orbitals(calc, beta=args.beta)
            if args.homo:
                if homo_idx is None:
                    raise InputError('Could not determine the HOMO index')
                operations.append(('orbital', homo_idx, 'homo'))
            if args.lumo:
                if lumo_idx is None:
                    raise InputError('Could not determine the LUMO index')
                operations.append(('orbital', lumo_idx, 'lumo'))
        n_mos = self._n_mos(calc, args.beta)
        for idx in args.orbital or []:
            if not 0 <= idx < n_mos:
                raise InputError(f'Orbital index {idx} out of range (0-{n_mos - 1})')
            operations.append(('orbital', idx, f'orbital_{idx}'))

        single = len(operations) == 1
        for op_type, val, tag in operations:
            if args.output:
                out_path = self.output_path(args.output, 'cube', None if single else tag)
            else:
                out_path = self.derived_path(args.input, tag, '.cube')
            self._build(op_type, calc, out_path, val, beta=args.beta)

    def _interactive_session(self, calc, filepath, args):
        from InquirerPy import inquirer

        print_info(f'Molecule: {len(calc.atoms)} atoms, {len(calc.mask_atoms)} basis functions')
        # Same grid defaults as the non-interactive run: spacing, unless --voxels was given
        use_voxels = args.voxels is not None

        def default_path(tag):
            return str(self.derived_path(filepath, tag, '.cube'))

        while True:
            mode = inquirer.select(
                message="What do you want to compute?",
                choices=[
                    "Total electron density",
                    "Spin density",
                    "HOMO orbital",
                    "LUMO orbital",
                    "Individual orbital(s)",
                ],
            ).execute()

            if use_voxels:
                args.voxels = int(inquirer.text(message="Number of grid points per axis:",
                                                default=str(args.voxels),
                                                validate=lambda x: x.isdigit() and int(x) > 0).execute())
            else:
                args.spacing = float(inquirer.text(message="Grid spacing (Å):", default=str(args.spacing),
                                                   validate=_positive_float).execute())
            args.padding = float(inquirer.text(message="Padding around the molecule (Å):",
                                               default=str(args.padding),
                                               validate=lambda x: _positive_float(x, allow_zero=True)).execute())

            if mode == "Total electron density":
                output = inquirer.text(message="Output cube file:", default=default_path('density')).execute()
                self._build('density', calc, Path(output))

            elif mode == "Spin density":
                output = inquirer.text(message="Output cube file:", default=default_path('spin_density')).execute()
                self._build('spin', calc, Path(output))

            elif mode in ("HOMO orbital", "LUMO orbital"):
                homo_idx, lumo_idx = get_frontier_orbitals(calc)
                idx, tag = (homo_idx, 'homo') if mode == "HOMO orbital" else (lumo_idx, 'lumo')
                if idx is None:
                    print_error(f'Could not determine the {tag.upper()} index')
                else:
                    output = inquirer.text(message=f"Output cube file for the {tag.upper()} (#{idx}):",
                                           default=default_path(tag)).execute()
                    self._build('orbital', calc, Path(output), idx)

            elif mode == "Individual orbital(s)":
                n_mos = self._n_mos(calc)
                val = inquirer.text(message=f"Orbital index(es), 0-based, space-separated (0-{n_mos - 1}):",
                                    default="0").execute()
                indices = [int(x) for x in val.split() if x.isdigit() and 0 <= int(x) < n_mos]
                if not indices:
                    print_error('No valid orbital index given')
                for idx in indices:
                    output = inquirer.text(message=f"Output cube file for orbital {idx}:",
                                           default=default_path(f'orbital_{idx}')).execute()
                    self._build('orbital', calc, Path(output), idx)

            if not inquirer.confirm(message="Another operation?", default=True).execute():
                break


def parse_args():
    return ElectronicDensity().build_parser().parse_args()


def main():
    ElectronicDensity().main()


if __name__ == '__main__':
    main()
