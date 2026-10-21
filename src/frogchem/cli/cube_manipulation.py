#!/usr/bin/env python3
"""Add, subtract or scale Gaussian cube files.

Performs voxel-by-voxel arithmetic on volumetric data in Gaussian cube format
(the grids must match), from the command line or interactively.
"""

import os
from pathlib import Path

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.grid import add_grids, multiply_grid, subtract_grids
from frogchem.io import parse_gaussian_cube, write_gaussian_cube
from frogchem.printer import print_info


class CubeManipulation(CLICommand, name='frog-cube'):
    """Add, subtract or scale Gaussian cube files.

    Performs voxel-by-voxel arithmetic on volumetric data (the grids must match).
    """

    category = 'spectroscopy'
    has_tables = False
    outputs = [
        Output('file', '<stem2>_plus_<stem1>.cube', 'FILE2 + FILE1, in the current directory (or -o)', when='-a'),
        Output('file', '<stem2>_minus_<stem1>.cube', 'FILE2 - FILE1, in the current directory (or -o)', when='-s'),
        Output('file', '<stem>_x<constant>.cube', 'FILE multiplied by CONSTANT, next to FILE (or -o)', when='-m'),
    ]
    guide = Guide(
        why="""
            A cube file (`.cube`) stores a quantity (electron density, an orbital, a spin density...) on a 3D
            grid of points around the molecule; viewers draw it as an isosurface. Many useful pictures are
            **differences or sums** of two cubes: density difference between an excited and the ground state,
            between a complex and its fragments (where does density move on binding?), between two solvents;
            or a cube **scaled** by a number (e.g. ×−1 to swap the colours of an orbital or a spin density).
            This command does that arithmetic point by point.
            """,
        needs="""
            Two cube files computed on the **same grid**: same number of points, same origin and same spacing
            (the command checks this and stops otherwise). In practice: the same geometry (same atoms in the
            same orientation) and the same grid options, e.g. both made by `frog-density` with identical
            `--voxels`/`--spacing` and `--padding`. Cubes from different programs or from different geometries
            cannot be combined. For `-m` a single cube and a number. Density cubes are in e/bohr³; the result
            keeps the units of the inputs.
            """,
        how=[
            "Subtract: `frog-cube -s A.cube B.cube` computes **B − A** (note the order: the second file minus "
            "the first) and writes `B_minus_A.cube` in the current folder.",
            "Add: `frog-cube -a A.cube B.cube` writes `B_plus_A.cube`; scale: `frog-cube -m spin.cube -1` "
            "writes `spin_x-1.cube` next to `spin.cube`.",
            "Choose the output name with `-o result.cube` (the folder must already exist), or use "
            "`frog-cube --interactive` to pick the operation and the files from menus.",
            "Open the result in a viewer (VMD, ChimeraX, Avogadro, GaussView) and show two isosurfaces of "
            "opposite sign, e.g. ±0.002 e/bohr³ for a density difference.",
        ],
        expect="""
            A new cube file and a *Saved cube* message; nothing else is printed. The atoms and the grid are
            copied from the first file, the values are the voxel-by-voxel result.

            * A density difference has positive (density gained) and negative (density lost) regions; it
              integrates to about zero when both densities have the same number of electrons. Typical isovalues
              are 0.001–0.005 e/bohr³, much smaller than for a total density (0.02–0.05).
            * "Grid sizes must match for addition." (or "Grid origin/voxel vectors must match") means the two
              cubes were not made on the same grid: rebuild them with identical grid options and geometry.
            * The input files are never overwritten; an existing output file with the same name is.

            Example: `frog-cube -s 1_p8_orca_T1_delta_rho.cube 1_p8_orca_S1_delta_rho.cube -o cub/S1_minus_T1.cube`
            (two 60×60×60 difference-density cubes written by `frog-tddft --dct`) writes the S1 − T1 cube;
            adding a cube with a 5×5×5 grid to one of them stops with "Grid sizes must match for addition.".
            """,
    )
    examples = [
        ('frog-cube -s S0_density.cube S1_density.cube', 'Difference density S1 - S0 (S1_density_minus_S0_density.cube)'),
        ('frog-cube -m spin.cube -1 -o spin_inverted.cube', 'Invert the sign of a spin density'),
        ('frog-cube --interactive', 'Choose the operation and the files interactively'),
    ]

    def setup_args(self, parser):
        ops = parser.add_mutually_exclusive_group()
        ops.add_argument('-a', '--add', nargs=2, metavar=('FILE1', 'FILE2'), help='Add two cube files: FILE2 + FILE1')
        ops.add_argument('-s', '--subtract', nargs=2, metavar=('FILE1', 'FILE2'),
                         help='Subtract two cube files: FILE2 - FILE1')
        ops.add_argument('-m', '--multiply', nargs=2, metavar=('FILE', 'CONSTANT'),
                         help='Multiply a cube file by a constant')
        self.add_interactive_arg(parser, help='Choose the operation and the files interactively')
        self.add_output_arg(parser, help='Output cube path')

    def run(self, args):
        if args.interactive:
            inputs, result, default = self._run_interactive()
        else:
            inputs, result, default = self._run_cli(args)
        out = self.output_path(default, 'cube')
        if any(out.resolve() == Path(f).resolve() for f in inputs):
            raise InputError(f'Refusing to overwrite the input file {out}')
        write_gaussian_cube(str(out), result)
        self.saved(out, 'cube')

    @staticmethod
    def _binary(op, f1, f2):
        g1 = parse_gaussian_cube(f1)
        g2 = parse_gaussian_cube(f2)
        if op == 'add':
            return add_grids(g1, g2), Path(f'{Path(f2).stem}_plus_{Path(f1).stem}.cube')
        return subtract_grids(g1, g2), Path(f'{Path(f2).stem}_minus_{Path(f1).stem}.cube')

    def _scale(self, f1, const_str):
        try:
            constant = float(const_str)
        except ValueError:
            raise InputError(f'CONSTANT must be a number, got {const_str!r}') from None
        result = multiply_grid(parse_gaussian_cube(f1), constant)
        return result, self.derived_path(f1, f'x{constant:g}', '.cube')

    def _run_cli(self, args):
        if args.add or args.subtract:
            op = 'add' if args.add else 'subtract'
            f1, f2 = args.add or args.subtract
            self.require_files([f1, f2])
            print_info(f'Adding: {f2} + {f1}' if op == 'add' else f'Subtracting: {f2} - {f1}')
            result, default = self._binary(op, f1, f2)
            return [f1, f2], result, default
        if args.multiply:
            f1, const_str = args.multiply
            self.require_files([f1])
            print_info(f'Multiplying: {f1} * {const_str}')
            result, default = self._scale(f1, const_str)
            return [f1], result, default
        raise InputError('No operation specified: use -a, -s, -m or --interactive')

    def _run_interactive(self):
        from InquirerPy import inquirer
        from InquirerPy.validator import PathValidator

        def pick(message):
            return inquirer.filepath(
                message=message,
                default=os.getcwd(),
                validate=PathValidator(is_file=True, message="Not a valid file"),
                only_files=True,
            ).execute()

        operation = inquirer.select(
            message="Choose the operation:",
            choices=["Add two cube files", "Subtract two cube files", "Multiply a cube by a constant"],
        ).execute()

        if "Multiply" in operation:
            f1 = pick("Select the cube file:")
            const_str = inquirer.text(message="Constant:").execute()
            result, default = self._scale(f1, const_str)
            inputs = [f1]
        else:
            f1 = pick("Select the FIRST cube file (FILE1):")
            f2 = pick("Select the SECOND cube file (FILE2):")
            print_info("Reading files...")
            result, default = self._binary('add' if "Add" in operation else 'subtract', f1, f2)
            inputs = [f1, f2]

        if not self.args.output:
            self.args.output = inquirer.text(message="Output cube file:", default=str(default)).execute()
        return inputs, result, default


def parse_args():
    return CubeManipulation().build_parser().parse_args()


def main():
    CubeManipulation().main()


if __name__ == '__main__':
    main()
