#!/usr/bin/env python3
"""
Integrate a spin density cube on atoms and fragments (Voronoi partition).

Each voxel is assigned to the nearest atom; fragment populations are sums over their atoms.
"""

from typing import List

import numpy as np
import periodictable as pt

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError, ParseError
from frogchem.grid import integrate_density_voronoi
from frogchem.io import parse_gaussian_cube
from frogchem.printer import print_error, print_info, print_warning


def parse_index_string(s: str) -> List[int]:
    """Parse '0-5, 8, 10' into a sorted list of 0-based indices (ranges are inclusive)."""
    indices = set()
    for part in s.replace(' ', '').split(','):
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        except ValueError:
            raise InputError(f"Invalid atom selection {part!r} (expected e.g. '0-5,8')") from None
    return sorted(indices)


def parse_fragment(spec: str) -> tuple[str, List[int]]:
    """Parse 'NAME:0-5,8' into (name, indices)."""
    name, sep, sel = spec.partition(':')
    if not sep or not name or not sel:
        raise InputError(f"Invalid fragment {spec!r} (expected NAME:INDICES, e.g. Cu:0 or L1:1-12)")
    return name, parse_index_string(sel)


def _element(z: int) -> str:
    try:
        return pt.elements[z].symbol
    except Exception:
        return '?'


class CalcLocSpin(CLICommand, name='frog-spin'):
    """Integrate a spin density cube on atoms and fragments (Voronoi partition).

    Each voxel is assigned to the nearest atom. Fragments are given with --fragment NAME:INDICES
    (0-based, ranges inclusive), or built step by step with --interactive.
    """

    category = 'parsing'
    outputs = [
        Output('table', 'Atomic Spin Populations (<file>)', 'One row per atom: integrated spin and % of the total'),
        Output('table', 'Fragment Spin Populations (<file>)',
               'One row per fragment plus "Rest of system": spin and % of the total',
               when='--fragment/--interactive'),
    ]
    guide = Guide(
        why="""
            In a radical, triplet or open-shell metal complex the unpaired electron(s) are spread over
            the molecule. This command tells you **where**: how much of the spin sits on each atom and on
            groups of atoms you define (e.g. "metal" vs "ligand"). Use it to decide whether a radical is
            centred on N or on the ring, whether a complex is metal- or ligand-based, or to compare spin
            delocalisation between conformers or between functionals.
            """,
        needs="""
            A **spin-density cube** (`.cube`) of an open-shell calculation (multiplicity 2, 3, ...).
            The easiest way to get one is `frog-density OUTPUT --spin` on an ORCA output written with
            `! PrintBasis PrintMOs` or a Gaussian 16 log written with `#p ... pop=full iop(3/33=1)
            iop(6/80=1) gfinput` (a cube from `cubegen ... spin` or ORCA `orca_plot` works too). A
            closed-shell (singlet) calculation has no spin density.
            Look up the atom numbers you want to group in a viewer: here they are **0-based**
            (the first atom of the file is 0), as in all FROG commands.
            """,
        how=[
            "Make the cube: `frog-density radical.out --spin` (writes `radical_spin_density.cube`; "
            "keep the default spacing of 0.15 Å or finer for reliable numbers).",
            "Run `frog-spin radical_spin_density.cube` to get the spin on every atom.",
            "Optionally sum atoms into fragments: `--fragment Metal:0 --fragment L:1-24` "
            "(name, colon, 0-based atoms; ranges like `1-24` include both ends; repeat the flag for more "
            "fragments). Atoms you do not list are reported together as *Rest of system*.",
            "Not sure of the numbers? Use `frog-spin --interactive` and build the fragments step by step.",
        ],
        expect="""
            A table with one row per atom: **Spin** (in electrons, α − β) and **% Total**, followed by a
            *Total* row; with `--fragment` a second table with the fragment sums.

            * The *Total* should be close to the number of unpaired electrons: about **1** for a doublet,
              **2** for a triplet, **3** for a quartet. Small deviations (±0.01–0.02) come from the finite
              grid; a larger deviation means the box is too small (increase `--padding` in
              `frog-density`) or the grid too coarse.
            * Negative values are normal: they are *spin polarisation* (β excess) on atoms next to the
              spin-carrying centre.
            * Each grid point is assigned to the **nearest atom** (Voronoi partition), regardless of atom size.
              Numbers are therefore close to, but not the same as, the Mulliken or Löwdin spin populations
              printed by ORCA (e.g. N 0.61 here vs 0.72 Mulliken). Compare trends within the same method,
              not absolute values between methods.

            Example (doublet radical, B3LYP/def2-SVP, 0.25 Å grid): Total 1.0036, N(6) 0.614 (61 %),
            the ring carbons 0.07–0.09 each.
            """,
    )
    examples = [
        ('frog-spin spin.cube', 'Spin population of every atom'),
        ('frog-spin spin.cube --fragment Cu:0 --fragment L:1-24', 'Spin on the metal (atom 0) and on the ligand'),
        ('frog-spin --interactive', 'Pick the cube file and define fragments interactively'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='?', help='Spin density cube file')
        parser.add_argument('--fragment', action='append', default=[], metavar='NAME:INDICES',
                            help="Fragment as NAME:INDICES with 0-based atom indices, e.g. 'L1:1-6,8' "
                                 "(repeatable)")
        self.add_interactive_arg(parser, help='Select the file and define fragments interactively')

    def run(self, args):
        filepath = args.file
        if not filepath:
            if not args.interactive:
                raise InputError('No cube file given (or use --interactive to pick one)')
            from InquirerPy import inquirer
            from InquirerPy.validator import PathValidator
            filepath = inquirer.filepath(
                message='Select the spin density cube file:',
                default='.',
                validate=PathValidator(is_file=True, message='File not found'),
                only_files=True,
            ).execute()
        self.require_files([filepath])

        print_info(f'Parsing {filepath}...')
        try:
            cube_data = parse_gaussian_cube(filepath)
        except Exception as e:
            raise ParseError(f'Error parsing cube file {filepath}: {e}') from e

        n_atoms = abs(cube_data['natoms'])
        print_info(f"Loaded cube: {n_atoms} atoms, grid {cube_data['grid_size']}")
        fragments = {}
        for spec in args.fragment:
            name, indices = parse_fragment(spec)
            bad = [i for i in indices if not 0 <= i < n_atoms]
            if bad:
                shown = ', '.join(map(str, bad[:5])) + (', ...' if len(bad) > 5 else '')
                raise InputError(f'Fragment {name!r}: atom indices {shown} out of range (0-{n_atoms - 1})')
            fragments[name] = indices

        print_info('Integrating spin density (Voronoi partition)...')
        atom_integrals = np.asarray(integrate_density_voronoi(cube_data))
        total_spin = float(np.sum(atom_integrals))
        print_info(f'Total integrated spin: {total_spin:.4f}')

        def pct(value):
            return value / total_spin * 100 if abs(total_spin) > 1e-5 else None

        elements = [_element(a['atomic_number']) for a in cube_data['atoms']]
        atom_rows = [[i, elements[i] if i < len(elements) else '?', float(v), pct(float(v))]
                     for i, v in enumerate(atom_integrals)]
        atom_rows.append(['Total', '', total_spin, pct(total_spin)])
        self.output(atom_rows, ['Idx', 'Element', 'Spin', '% Total'],
                    title=f'Atomic Spin Populations ({filepath})', floatfmt='.4f')

        if args.interactive:
            fragments = self._interactive_fragments(fragments, n_atoms)

        if fragments:
            self.output(self._fragment_rows(fragments, atom_integrals, n_atoms, pct),
                        ['Fragment', 'Spin', '% Total'],
                        title=f'Fragment Spin Populations ({filepath})', floatfmt='.4f')

    @staticmethod
    def _fragment_rows(fragments, atom_integrals, n_atoms, pct):
        rows = []
        for name, idxs in fragments.items():
            spin_val = float(np.sum(atom_integrals[idxs]))
            rows.append([name, spin_val, pct(spin_val)])
        selected = set().union(*fragments.values())
        rest = [i for i in range(n_atoms) if i not in selected]
        if rest:
            spin_rest = float(np.sum(atom_integrals[rest]))
            rows.append(['Rest of system', spin_rest, pct(spin_rest)])
        return rows

    @staticmethod
    def _interactive_fragments(fragments, n_atoms):
        from InquirerPy import inquirer
        print_info('Define the fragments to sum the spin density (0-based atom indices).')
        while True:
            action = inquirer.select(
                message='Manage fragments:',
                choices=['Add fragment', 'Show results'],
                default='Add fragment',
            ).execute()
            if action == 'Show results':
                return fragments

            name = inquirer.text(message='Fragment name:').execute()
            selection = inquirer.text(message=f"Atom indices for '{name}', 0-based (e.g. 0-5, 8):").execute()
            try:
                indices = parse_index_string(selection)
            except InputError as e:
                print_error(str(e))
                continue
            valid = [i for i in indices if 0 <= i < n_atoms]
            if len(valid) != len(indices):
                print_warning(f'Some indices were out of range (0-{n_atoms - 1}) and were ignored')
            if not valid:
                print_error('No valid atoms selected')
                continue
            fragments[name] = valid
            print_info(f"Fragment '{name}' added with {len(valid)} atoms")


def parse_args():
    return CalcLocSpin().build_parser().parse_args()


def main():
    CalcLocSpin().main()


if __name__ == '__main__':
    main()
