#!/usr/bin/env python3
"""Build electron density, spin density and molecular orbital cube files
from ORCA/Gaussian output or a cleave orbital file, from the command line or in an interactive session.
"""

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from frogchem.cleave_orbitals import is_open_shell
from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.density_builder import (build_density_cube, build_orbital_cube, build_orbitals_cube,
                                     build_spin_density_cube, _spin_coeffs)
from frogchem.exceptions import InputError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_error, print_info, print_warning

from frogchem.chemistry import HARTREE_TO_EV as EH_TO_EV
#: ChimeraX 1.12 reads multicubes correctly only up to 9 orbitals (one line of ids)
CHIMERAX_MULTICUBE_LIMIT = 10


def choose_file() -> str:
    from InquirerPy import inquirer
    from InquirerPy.validator import PathValidator
    return inquirer.filepath(
        message="ORCA/Gaussian output or cleave orbital file:",
        default=os.getcwd(),
        validate=PathValidator(is_file=True, message="Not a valid file"),
        only_files=True,
    ).execute()


def get_frontier_orbitals(calc, beta=False) -> Tuple[Optional[int], Optional[int]]:
    """Return (homo_idx, lumo_idx) based on MO occupations, or (None, None)."""
    if is_open_shell(calc) and hasattr(calc, 'alpha_occ'):
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



def _voxel_text(text: str):
    """'80' -> 80, '20 16 12' -> (20, 16, 12); None when it is neither."""
    parts = text.split()
    if len(parts) not in (1, 3) or not all(p.isdigit() and int(p) > 0 for p in parts):
        return None
    return int(parts[0]) if len(parts) == 1 else tuple(int(p) for p in parts)


def _positive_float(text: str, allow_zero: bool = False) -> bool:
    try:
        value = float(text)
    except ValueError:
        return False
    return value >= 0 if allow_zero else value > 0


class ElectronicDensity(CLICommand, name='frog-density'):
    """Build electron density, spin density and molecular orbital cube files.

    Reads the basis set and MO coefficients from ORCA output (! PrintBasis PrintMOs), a Gaussian log
    (pop=full gfinput) or a cleave orbital file (<name>.orbitals.json, written by every almo/cdft run;
    pick the orbital set with --set). Without a selection the total electron density is computed.
    Use --interactive for a guided session.
    """

    category = 'spectroscopy'
    outputs = [
        Output('file', '<stem>_density.cube', 'Total electron density (e/bohr³)', when='--density, or no selection'),
        Output('file', '<stem>_spin_density.cube', 'Spin density, alpha - beta (open-shell only)', when='--spin'),
        Output('file', '<stem>_orbitals.cube', 'Multicube: all the selected orbitals in one file (cube ids are '
                                               'index + 1); open-shell: _orbitals_alpha / _orbitals_beta',
               when='2 or more orbitals (--homo, --lumo, --frontier, --orbital)'),
        Output('file', '<stem>_homo.cube', 'HOMO, one file per orbital', when='--homo with --split'),
        Output('file', '<stem>_lumo.cube', 'LUMO, one file per orbital', when='--lumo with --split'),
        Output('file', '<stem>_orbital_<N>.cube', 'Orbital N (0-based), one file per orbital',
               when='--orbital with --split, or a single orbital'),
        Output('table', 'Orbital sets (<file>)', 'Name, kind, spin channels, number of orbitals, '
                                                 'orthonormality and description of each set of a cleave '
                                                 'orbital file', when='--list-sets'),
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

            * **cleave** (`almo`, `cdft`): nothing to add, every run writes `<name>.orbitals.json` with the
              basis and several **orbital sets**: `full` (canonical MOs of the complex, the default),
              `frag1`, `frag2`, ... (isolated fragments), `frozen` and `polarized` (ALMOs, non-orthogonal:
              their density is computed exactly), `covp_forward`/`covp_backward` (with `almo --covp`: donor
              and acceptor orbitals side by side, orbital cubes only), `cdft` and `hs` (CDFT state and its
              high-spin partner). List them with `--list-sets`, choose one with `--set`; the cube names
              carry the set (`dimer_polarized_density.cube`).

            Without them the command stops with an error telling you so. A spin density needs an open-shell
            calculation (multiplicity 2, 3, ...). Orbital numbers are **0-based**, as in ORCA's
            `ORBITAL ENERGIES` table (Gaussian's MO 1 is orbital 0); open-shell calculations have separate α
            and β sets: the α set is used unless `--beta` is given, and `--beta` on a closed-shell calculation
            is an error.
            """,
        how=[
            "Frontier orbitals: `frog-density job.out --homo --lumo` writes one multicube `job_orbitals.cube` "
            "with both orbitals, next to the output (or in `--out-dir`); `--frontier N` takes HOMO−N … HOMO and "
            "LUMO … LUMO+N (`--frontier 2`: six orbitals). Add `--split` for one file per orbital "
            "(`job_homo.cube`, `job_lumo.cube`, `job_orbital_<N>.cube`). Every run prints the HOMO and LUMO numbers.",
            "Other cubes: `--density` (total density; also the default when no option is given), `--spin` "
            "(spin density, α − β), `--orbital 41 42` (0-based MO numbers). In an open-shell run the orbitals are "
            "α unless you add `--beta`, and the command says which set it uses; `--beta` on a closed-shell run is "
            "an error, since it has no β orbitals.",
            "Control the grid: `--voxels N` (N points on every axis, default 80) or `--voxels NX NY NZ` (one count "
            "per axis), or `--spacing` (Å) instead: give one of the two, not both. `--padding` is the empty space "
            "around the molecule (Å, default 3.0). Use `--voxels 50` for a quick look, a finer grid for numbers "
            "(e.g. `frog-spin`).",
            "cleave runs: `frog-density dimer.orbitals.json --list-sets`, then e.g. `--set polarized --density` "
            "(ALMO density of the polarized state) or `--set covp_forward --orbital 0 1` (leading COVP "
            "donor/acceptor pair); without `--set` the file's default set is used (`full` for `almo`, `cdft` "
            "for `cdft`).",
            "Name the files with `-o name.cube` (one file, a multicube included) or `-o prefix` (several files: "
            "`prefix_density.cube`, `prefix_orbitals.cube`, ...), or use `--interactive` for a guided session "
            "(it asks α or β in an open-shell run).",
            "Open the cube in a viewer: isovalue about ±0.03–0.05 for orbitals, 0.02–0.05 e/bohr³ for the total "
            "density, ±0.002–0.005 for a spin density. ChimeraX's own cube reader misreads multicubes with 10 or "
            "more orbitals: use `--split` there, or `frog-tddft --orbital-grid`, whose script reads the file itself.",
        ],
        expect="""
            One `.cube` file per requested quantity and a *Saved cube* line for each, preceded by the grid
            size and the value range of the cube (hidden with `-q`). Values are in atomic units: e/bohr³ for
            densities, bohr^−3/2 for orbitals.

            * Every run starts with the HOMO and LUMO, e.g. `HOMO = 15, LUMO = 16  (0-based; cube ids 16, 17)`,
              for both spins in an open-shell run. Indices are 0-based, as everywhere in FROG; the ids written
              inside a cube file are 1-based (index + 1), as cubegen writes them.
            * Two or more orbitals go into one **multicube**, `<stem>_orbitals.cube`, listed in a table (index,
              cube id, H/L label, energy in eV). Its header has a negative atom count followed by the orbital
              ids (Gaussian `10I5` record); the values of all the orbitals are stored point by point.
              ChimeraX's own cube reader takes only the first line of ids, so it misreads multicubes with 10 or
              more orbitals (checked with ChimeraX 1.12; the command warns): use `--split` for ChimeraX.
            * The sign of an orbital is arbitrary. FROG flips each orbital so that its largest lobe is
              positive, so the colours stay the same from run to run and between ORCA and Gaussian; densities
              are not affected.
            * File size and time grow with the number of points and of orbitals: the default 80×80×80 grid
              is about 6.7 MB per orbital, and a multicube of N orbitals is N times that.
            * A spin density of a closed-shell (singlet) calculation is zero: the command prints a warning and
              writes no file.
            * If an isosurface looks cut at the box edge, increase `--padding`; if it looks blocky, use a
              finer grid.
            * HOMO and LUMO are taken from the occupations (α set by default; β with `--beta`).

            Example (water, B3LYP/def2-TZVPP with `! PrintBasis PrintMOs`, `--density --homo --lumo --voxels 50`):
            a 50×50×50 density cube of about 1.6 MB and a multicube with HOMO (orbital 4, values ±0.631) and
            LUMO (orbital 5) of about 3.3 MB.
            """,
    )
    examples = [
        ('frog-density opt.out', 'Total electron density cube next to opt.out'),
        ('frog-density opt.out --frontier 2 --voxels 60 -o frontier',
         'One multicube frontier.cube with HOMO-2 ... LUMO+2 on a 60x60x60 grid (cube ids 1-based)'),
        ('frog-density opt.out --homo --lumo --split --voxels 60 -o frontier',
         'Write frontier_homo.cube and frontier_lumo.cube, one file per orbital'),
        ('frog-density opt.out --spin --orbital 41 42 --beta', 'Spin density and beta orbitals 41 and 42'),
        ('frog-density dimer.orbitals.json --list-sets', 'Orbital sets of a cleave run (almo/cdft)'),
        ('frog-density dimer.orbitals.json --set polarized --density',
         'Density of the polarized (ALMO) state of a cleave ALMO-EDA'),
    ]

    def setup_args(self, parser):
        parser.add_argument('input', nargs='?', help='ORCA output with ! PrintBasis PrintMOs, Gaussian log '
                                                     'with pop=full gfinput, or cleave .orbitals.json')
        parser.add_argument('--density', action='store_true', help='Compute the total electron density')
        parser.add_argument('--spin', action='store_true', help='Compute the spin density')
        parser.add_argument('--orbital', type=int, nargs='+', metavar='N',
                            help='Compute orbital(s) by index (0-based)')
        parser.add_argument('--homo', action='store_true', help='Compute the HOMO')
        parser.add_argument('--lumo', action='store_true', help='Compute the LUMO')
        parser.add_argument('--frontier', type=int, metavar='N', default=None,
                            help='HOMO-N ... HOMO and LUMO ... LUMO+N (--frontier 0 is HOMO and LUMO)')
        parser.add_argument('--split', action='store_true',
                            help='Write one cube per orbital instead of one multicube file')
        parser.add_argument('--beta', action='store_true', help='Use beta orbitals (open-shell systems)')
        parser.add_argument('--set', metavar='NAME', help='Orbital set of a cleave orbital file '
                                                          '(default: the file\'s own, e.g. full or cdft)')
        parser.add_argument('--list-sets', action='store_true',
                            help='List the orbital sets of a cleave orbital file and exit')
        self.add_grid_args(parser, spacing=0.15, padding=3.0, voxels=80)
        self.add_output_arg(parser, help='Output cube path, or prefix ({prefix}_{tag}.cube) when several '
                                         'cubes are written; without it, cubes go next to the input')
        self.add_out_dir_arg(parser)
        self.add_interactive_arg(parser, help='Choose the file and the cubes interactively')

    def run(self, args):
        if args.interactive:
            filepath = args.input or choose_file()
            self.require_files([filepath])
            calc = load_calculation(filepath)
            self._select_set(calc, args, filepath, ask=True)
            self._check_basis(calc, filepath)
            self._interactive_session(calc, filepath, args)
            return

        if not args.input:
            raise InputError('No input file given (or use --interactive)')
        self.require_files([args.input])
        calc = load_calculation(args.input)
        if self._select_set(calc, args, args.input):
            return
        self._check_basis(calc, args.input)
        self._run_cli(calc, args)

    def _select_set(self, calc, args, path, ask=False) -> bool:
        """--set / --list-sets of a cleave orbital file; True when only the list was asked."""
        if not hasattr(calc, 'select'):
            if args.set or args.list_sets:
                raise InputError(f'--set/--list-sets need a cleave orbital file, not {path}')
            return False
        if args.list_sets:
            self.output(calc.sets(), ['Set', 'Kind', 'Spin', 'MOs', 'Orthonormal', 'Description'],
                        title=f'Orbital sets ({path})')
            return True
        if args.set:
            calc.select(args.set)
        elif ask and len(calc.set_names) > 1:
            from InquirerPy import inquirer
            calc.select(inquirer.select(message='Orbital set:', choices=calc.set_names,
                                        default=calc.set_name).execute())
        # cleave files carry no "description" since their lean format: show the kind
        about = calc.set_info.get('description') or calc.set_info.get('kind', '')
        print_info(f'Orbital set: {calc.set_name} ({about})')
        return False

    @staticmethod
    def _source(path):
        """Path that names the cubes: 'dimer.orbitals.json' → 'dimer.json' (cubes 'dimer_<tag>.cube')."""
        path = Path(path)
        stem = path.name.split('.orbitals.json')[0]
        return path.with_name(stem + '.json') if stem != path.name else path

    @staticmethod
    def _tag(calc, tag):
        """Cube name tag; cleave files prefix the orbital set (polarized_density, ...),
        except per-set files (dimer.Frag1.orbitals.json), whose name already holds it."""
        if not hasattr(calc, 'set_name'):
            return tag
        stem = Path(calc.fname).name.split('.orbitals.json')[0]
        slug = re.sub(r'[^A-Za-z0-9._-]+', '-', calc.set_name).strip('-')
        return tag if stem.endswith('.' + slug) else f'{calc.set_name}_{tag}'

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
        if op_type != 'orbital' and hasattr(calc, 'density_view'):
            calc.density_view()               # sets without occupations stop here, before the grid
        if op_type == 'spin' and not is_open_shell(calc):
            print_warning('Closed-shell calculation (multiplicity 1): spin density is zero, cube not written')
            return
        if op_type == 'multi':
            print_info(f'Building multicube of {len(val)} orbitals (MO ids {[i + 1 for i in val]}) -> {out_path}')
            build_orbitals_cube(calc, val, output=str(out_path), beta=beta, **grid)
        elif op_type == 'density':
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
        if is_open_shell(calc):
            coeff = getattr(calc, 'beta_coeff' if beta else 'alpha_coeff', None)
            if coeff is not None:
                return coeff.shape[1]
        return calc.orbital_coefficients.shape[1]

    @staticmethod
    def _orbital_energy_ev(calc, idx, beta=False):
        """Orbital energy in eV, or None when the file has no energies."""
        en = getattr(calc, 'orbital_energies', None)
        if en is None:
            return None
        if _spin_coeffs(calc) is not None:
            offset = calc.alpha_coeff.shape[1] if beta else 0  # alpha block first, then beta
            idx = offset + idx
        if idx >= len(en):
            return None
        return float(en[idx]) * EH_TO_EV

    def _report_frontier(self, calc, beta_flag):
        """HOMO and LUMO indices (0-based, cube ids are index + 1) on stderr, always."""
        spins = [False, True] if _spin_coeffs(calc) is not None else [False]
        for beta in spins:
            homo, lumo = get_frontier_orbitals(calc, beta=beta)
            if homo is None or lumo is None:
                continue
            tag = ' (beta)' if beta else (' (alpha)' if len(spins) > 1 else '')
            print_info(f'HOMO = {homo}, LUMO = {lumo}{tag}  (0-based; cube ids {homo + 1}, {lumo + 1})')

    @staticmethod
    def _frontier_span(homo, lumo, n):
        """HOMO-n ... HOMO and LUMO ... LUMO+n as 0-based indices (unique, sorted)."""
        return sorted(set(range(max(0, homo - n), homo + 1)) | set(range(lumo, lumo + n + 1)))

    def _run_cli(self, calc, args):
        any_flag = (args.density or args.spin or args.orbital is not None or args.homo or args.lumo
                    or args.frontier is not None)
        open_shell = _spin_coeffs(calc) is not None
        if args.beta and not open_shell:
            raise InputError('--beta: this calculation is closed-shell (multiplicity 1), '
                             'it has no beta orbitals; drop --beta')
        if open_shell and (args.orbital is not None or args.homo or args.lumo or args.frontier is not None):
            print_info('Open-shell calculation: the orbitals are alpha unless --beta is given')
        self._report_frontier(calc, args.beta)
        operations: List[Tuple[str, Any, str]] = []
        if args.density or not any_flag:
            operations.append(('density', None, 'density'))
        if args.spin:
            operations.append(('spin', None, 'spin_density'))

        n_mos = self._n_mos(calc, args.beta)
        homo_idx = lumo_idx = None
        if args.homo or args.lumo or args.frontier is not None:
            homo_idx, lumo_idx = get_frontier_orbitals(calc, beta=args.beta)
            if homo_idx is None or lumo_idx is None:
                raise InputError('Could not determine the HOMO/LUMO index')

        # (index, tag) in the order the user asked for; duplicates keep the first tag
        wanted: List[Tuple[int, str]] = []
        if args.homo:
            wanted.append((homo_idx, 'homo'))
        if args.lumo:
            wanted.append((lumo_idx, 'lumo'))
        if args.frontier is not None:
            for idx in self._frontier_span(homo_idx, lumo_idx, args.frontier):
                wanted.append((idx, f'orbital_{idx}'))
        for idx in args.orbital or []:
            wanted.append((idx, f'orbital_{idx}'))
        seen, orbitals = set(), []
        for idx, tag in wanted:
            if not 0 <= idx < n_mos:
                raise InputError(f'Orbital index {idx} out of range (0-{n_mos - 1})')
            if idx not in seen:
                seen.add(idx)
                orbitals.append((idx, tag))

        if len(orbitals) >= 2 and not args.split:
            spins = _spin_coeffs(calc)
            tag = ('orbitals_' + ('beta' if args.beta else 'alpha')) if spins is not None else 'orbitals'
            indices = [idx for idx, _ in orbitals]
            operations.append(('multi', indices, tag))
            self._list_orbitals(calc, indices, args.beta)
            if len(indices) >= CHIMERAX_MULTICUBE_LIMIT:
                print_warning(f'{len(indices)} orbitals in one multicube: ChimeraX\'s own cube reader misreads '
                              f'multicubes with {CHIMERAX_MULTICUBE_LIMIT} or more orbitals. For ChimeraX use '
                              '--split, or frog-tddft --orbital-grid (its script reads the file itself)')
        else:
            if len(orbitals) > 1:
                self._list_orbitals(calc, [idx for idx, _ in orbitals], args.beta)
            for idx, tag in orbitals:
                operations.append(('orbital', idx, tag))

        single = len(operations) == 1
        for op_type, val, tag in operations:
            tag = self._tag(calc, tag)
            if args.output:
                out_path = self.output_path(args.output, 'cube', None if single else tag)
            else:
                out_path = self.derived_path(self._source(args.input), tag, '.cube')
            self._build(op_type, calc, out_path, val, beta=args.beta)

    def _list_orbitals(self, calc, indices, beta):
        """The orbitals of a multicube: index, cube id, frontier label and energy (eV)."""
        homo, _ = get_frontier_orbitals(calc, beta=beta)
        rows = []
        for idx in indices:
            if homo is None:
                label = ''
            elif idx <= homo:
                label = 'H' if idx == homo else f'H-{homo - idx}'
            else:
                label = 'L' if idx == homo + 1 else f'L+{idx - homo - 1}'
            en = self._orbital_energy_ev(calc, idx, beta)
            rows.append([idx, idx + 1, label, f'{en:+.2f}' if en is not None else 'N/A'])
        self.output(rows, headers=['Index', 'Cube id', 'Label', 'Energy (eV)'],
                    title='Orbitals in the multicube')

    @staticmethod
    def _ask_spin(calc, inquirer) -> bool:
        """True for beta orbitals. Asked only when the calculation is open-shell."""
        if _spin_coeffs(calc) is None:
            return False
        return inquirer.select(message='Spin of the orbital(s):', choices=['alpha', 'beta'],
                               default='alpha').execute() == 'beta'

    def _interactive_session(self, calc, filepath, args):
        from InquirerPy import inquirer


        print_info(f'Molecule: {len(calc.atoms)} atoms, {len(calc.mask_atoms)} basis functions')
        # Same grid defaults as the non-interactive run: spacing, unless --voxels was given
        use_voxels = args.voxels is not None

        def default_path(tag):
            return str(self.derived_path(self._source(filepath), self._tag(calc, tag), '.cube'))

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
                current = args.voxels if isinstance(args.voxels, tuple) else (args.voxels,)
                args.voxels = _voxel_text(inquirer.text(
                    message="Grid points: N on every axis, or NX NY NZ:",
                    default=" ".join(str(v) for v in current),
                    validate=lambda x: _voxel_text(x) is not None).execute())
            else:
                args.spacing = float(inquirer.text(message="Grid spacing (Å):", default=str(args.spacing or 0.15),
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
                beta_flag = self._ask_spin(calc, inquirer)
                homo_idx, lumo_idx = get_frontier_orbitals(calc, beta=beta_flag)
                idx, tag = (homo_idx, 'homo') if mode == "HOMO orbital" else (lumo_idx, 'lumo')
                if idx is None:
                    print_error(f'Could not determine the {tag.upper()} index')
                else:
                    tag = f"{tag}_beta" if beta_flag else tag
                    output = inquirer.text(message=f"Output cube file for the {tag.upper()} (#{idx}):",
                                           default=default_path(tag)).execute()
                    self._build('orbital', calc, Path(output), idx, beta=beta_flag)

            elif mode == "Individual orbital(s)":
                n_mos = self._n_mos(calc)
                val = inquirer.text(message=f"Orbital index(es), 0-based, space-separated (0-{n_mos - 1}):",
                                    default="0").execute()
                indices = [int(x) for x in val.split() if x.isdigit() and 0 <= int(x) < n_mos]
                if not indices:
                    print_error('No valid orbital index given')
                beta_flag = self._ask_spin(calc, inquirer) if indices else False
                for idx in indices:
                    tag = f'orbital_{idx}' + ('_beta' if beta_flag else '')
                    output = inquirer.text(message=f"Output cube file for orbital {idx}:",
                                           default=default_path(tag)).execute()
                    self._build('orbital', calc, Path(output), idx, beta=beta_flag)

            if not inquirer.confirm(message="Another operation?", default=True).execute():
                break



def main():
    ElectronicDensity().main()


if __name__ == '__main__':
    main()
