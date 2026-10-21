"""
Analyse the atomic-orbital composition of molecular orbitals.
"""

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError, ParseError
from frogchem.filetypes import load_calculation
from frogchem.orca_parser.orb_type import ORB_TYPE
from frogchem.printer import print_info, print_warning

# ponytail: matplotlib is lazily loaded inside plot_coeff_matrix (through frogchem.plotter)

HARTREE_TO_EV = 27.211386245988

TYPE_CODES = {
    's': [0],
    'p': [1, 2, 3],
    # pure functions, then the Cartesian ones of Gaussian 6D/10F
    'd': [4, 5, 6, 7, 8] + [c for n, c in ORB_TYPE.items() if n.startswith('d_')],
    'f': [9, 10, 11, 12, 13, 14, 15] + [c for n, c in ORB_TYPE.items() if n.startswith('f_')],
}


def _open_shell(calc) -> bool:
    return getattr(calc, 'mult', 1) != 1 and hasattr(calc, 'alpha_occ')


def get_homo_index(calc, beta=False):
    if getattr(calc, 'homo', None) is not None:
        return calc.homo
    if _open_shell(calc):
        occ = calc.beta_occ if beta else calc.alpha_occ
    elif getattr(calc, 'occupation', None) is not None:
        occ = calc.occupation
    elif getattr(calc, 'alpha_occ', None) is not None:
        occ = calc.alpha_occ
    else:
        return 0
    occupied_indices = np.where(np.array(occ) > 0.001)[0]
    return occupied_indices[-1] if len(occupied_indices) > 0 else 0


def calculate_full_percent_matrix(calc, beta=False):
    if getattr(calc, 'mult', 1) != 1 and hasattr(calc, 'alpha_coeff'):
        coeffs = calc.beta_coeff if beta else calc.alpha_coeff
    else:
        coeffs = calc.orbital_coefficients
    norm_vector = np.linalg.norm(coeffs, axis=0)
    norm_vector[norm_vector == 0] = 1.0
    coeffs_normalized = coeffs / norm_vector
    return (coeffs_normalized ** 2) * 100


def resolve_atoms(calc, indices=None, elements=None):
    """Atom indices selected by 0-based indices and/or element symbols (None = no filter)."""
    if not indices and not elements:
        return None
    selected = set(indices or [])
    if elements:
        wanted = {e.capitalize() for e in elements}
        selected.update(i for i, atom in enumerate(calc.atoms) if str(atom).capitalize() in wanted)
    bad = [i for i in selected if not 0 <= i < len(calc.atoms)]
    if bad:
        raise InputError(f'Atom indices {sorted(bad)} out of range (0-{len(calc.atoms) - 1})')
    if not selected:
        raise InputError('None of the selected atoms exist in the molecule')
    return sorted(selected)


def composition_table(calc, mo_indices, matrix_percent, atom_indices=None, type_filter=None, beta=False):
    """Rows and headers of the % composition of the selected MOs (per atom/shell and totals)."""
    mask_atoms_int = calc.mask_atoms.astype(int)
    atoms_shown = list(atom_indices) if atom_indices is not None else list(np.unique(mask_atoms_int))
    wanted = {t[0].lower() for t in type_filter} if type_filter else set(TYPE_CODES)
    target_types = [t for t in TYPE_CODES if t in wanted]

    headers = ['MO', 'E (eV)', 'Occ']
    atom_columns = []
    for atom_idx in atoms_shown:
        atom_sym = calc.atoms[atom_idx]
        atom_mask = mask_atoms_int == atom_idx
        for t in target_types:
            if np.any(atom_mask & np.isin(calc.mask_orbital_type, TYPE_CODES[t])):
                headers.append(f'{atom_idx}{atom_sym}_{t}')
                atom_columns.append((atom_idx, TYPE_CODES[t]))
    total_types = [t for t in target_types if np.any(np.isin(calc.mask_orbital_type, TYPE_CODES[t]))]
    headers.extend(f'Total_{t}' for t in total_types)

    if _open_shell(calc):
        n_alpha = calc.alpha_occ.shape[0]
        occ_list = calc.beta_occ if beta else calc.alpha_occ
        energies = getattr(calc, 'orbital_energies', None)
        en_list = [] if energies is None else (energies[n_alpha:] if beta else energies[:n_alpha])
    else:
        occ_list = getattr(calc, 'occupation', None)
        en_list = getattr(calc, 'orbital_energies', None)
    occ_list = [] if occ_list is None else occ_list
    en_list = [] if en_list is None else en_list

    rows = []
    for mo_idx in mo_indices:
        mo_idx = int(mo_idx)
        e_val = float(en_list[mo_idx]) * HARTREE_TO_EV if mo_idx < len(en_list) else None
        occ_val = float(occ_list[mo_idx]) if mo_idx < len(occ_list) else None
        col_vector = matrix_percent[:, mo_idx]
        row = [mo_idx, e_val, occ_val]
        for atom_idx, codes in atom_columns:
            mask = (mask_atoms_int == atom_idx) & np.isin(calc.mask_orbital_type, codes)
            row.append(float(np.sum(col_vector[mask])))
        for t in total_types:
            row.append(float(np.sum(col_vector[np.isin(calc.mask_orbital_type, TYPE_CODES[t])])))
        rows.append(row)
    return rows, headers


def mo_window(calc, matrix_percent, buffer=None, beta=False):
    """(homo_idx, start, end) of the MO range: HOMO ± buffer, or all MOs."""
    homo_idx = int(get_homo_index(calc, beta=beta))
    n_mos = matrix_percent.shape[1]
    if buffer is None:
        return homo_idx, 0, n_mos
    return homo_idx, max(0, homo_idx - buffer), min(n_mos, homo_idx + buffer + 1)


def plot_coeff_matrix(calc, matrix_percent, atom_indices=None, type_filter=None, buffer=None,
                      cmap='rocket_r', beta=False, opts=None):
    """Heatmap of the % contribution of each basis function to the MOs. Returns the figure.

    opts: frogchem.plotter.PlotOptions (title, cmap, colors[0] = HOMO box, ...); they win over the defaults.
    """
    homo_idx, start_idx, end_idx = mo_window(calc, matrix_percent, buffer, beta)
    rev_orb_type = {v: k for k, v in ORB_TYPE.items()}

    if type_filter:
        selector = [ORB_TYPE[k] for k in ORB_TYPE for t in type_filter if k.startswith(t)]
        if not selector:
            raise InputError(f'Orbital type {type_filter} not recognized')
        type_mask = np.isin(calc.mask_orbital_type, selector)
    else:
        type_mask = np.ones(len(calc.mask_orbital_type), dtype=bool)

    mask_atoms_int = calc.mask_atoms.astype(int)
    active = set(range(len(calc.atoms)) if atom_indices is None else atom_indices)
    atom_mask = np.array([a in active for a in mask_atoms_int])

    sliced = matrix_percent[:, start_idx:end_idx]
    rows, labels = [], []
    for i in np.where(type_mask & atom_mask)[0]:
        atom_idx = int(mask_atoms_int[i])
        labels.append(f'{atom_idx}{calc.atoms[atom_idx]}-{rev_orb_type.get(calc.mask_orbital_type[i], "?")}')
        rows.append(sliced[i, :])
    if not rows:
        raise InputError('No basis functions match the selected atoms/orbital types')

    plot_data = np.array(rows)[::-1, :]
    plot_labels = labels[::-1]
    xaxis_labels = np.arange(start_idx, end_idx)

    import matplotlib.patches as patches

    from frogchem import plotter

    zoom = f' | MOs {xaxis_labels[0]}-{xaxis_labels[-1]}' if buffer is not None else ''
    opts = (opts or plotter.PlotOptions()).with_defaults(
        title=f'MO composition | HOMO: {homo_idx}{zoom}' + (' (beta)' if beta else ''),
        xlabel='MO index', ylabel='Contracted basis functions', cmap=cmap, figsize=(12, 8),
        colors=('cyan',),
    )
    fig, ax = plotter.heatmap(plot_data, xticklabels=xaxis_labels, yticklabels=plot_labels,
                              cbar_label='Contribution (%)', opts=opts)
    # HOMO column, in the first --colors colour
    if start_idx <= homo_idx < end_idx:
        ax.add_patch(patches.Rectangle((homo_idx - start_idx, 0), 1, len(plot_labels), linewidth=2,
                                       edgecolor=plotter.series_colors(opts, 1)[0], facecolor='none'))
    return fig


class MoCoeff(CLICommand, name='frog-mo'):
    """Analyse the atomic-orbital composition of molecular orbitals (ORCA and Gaussian outputs).

    Prints the % contribution of each atom and shell (s, p, d, f) to the MOs around the HOMO,
    and optionally saves a heatmap of the full MO coefficient matrix.
    """

    category = 'parsing'
    outputs = [
        Output('table', 'MO Composition (%) (<file>)',
               'One row per MO: energy (eV), occupation and % contribution per atom/shell and per shell type'),
        Output('plot', '<stem>_mo.png', 'Heatmap of the % contribution of each basis function to each MO '
                                         '(<stem>_mo_beta.png with --beta)', when='--plot'),
    ]
    guide = Guide(
        why="""
            Molecular orbitals (MOs) are combinations of atomic orbitals. This command tells you, in %, which
            atoms and which kind of atomic orbital (s, p, d, f) make up each MO: is the HOMO a lone pair on N
            or a π orbital of the ring? Is the LUMO of a complex metal d or ligand π*? How much metal d is
            in the frontier orbitals? Use it to describe frontier orbitals in a paper, to understand an
            excitation found by `frog-tddft` (which lists transitions like H→L), or to check orbital
            character before plotting an orbital cube with `frog-density`.
            """,
        needs="""
            An output that **contains all the MO coefficients** (they are not printed by default; a single
            point on the final geometry is enough):

            * **ORCA**: add `! PrintBasis PrintMOs` to the input.
            * **Gaussian 16**: `#p ... pop=full iop(3/33=1) iop(6/80=1) gfinput` (the same route serves
              `frog-density`, `frog-charges`, `frog-tddft --dct` and `frog-nbo --cube`). `pop=full`
              prints every MO (the default population prints only a few); pure (`5D 7F`) and Cartesian
              (`6D 10F`) d/f functions are both read, g functions are not.

            Open-shell (UKS/UHF) outputs have separate α and β orbitals (`--beta`). MO numbers are
            **0-based** (ORCA's numbering; Gaussian's MO 1 is MO 0 here), atom numbers too.
            """,
        how=[
            "Start with `frog-mo mol.out -b 3`: composition of the MOs from HOMO−3 to HOMO+3 (the LUMO is "
            "HOMO+1).",
            "For a large molecule pick the atoms you care about with `-i 0 1` (0-based) or `--elements Fe N`, "
            "and the shell with `--orbital-type d`; otherwise only per-shell totals are shown.",
            "Add `--plot` (optionally `--plot mo.svg`) to save a heatmap of every basis function vs MO; "
            "the HOMO column is boxed in cyan.",
            "Open shell: run once as is (α orbitals) and once with `--beta`.",
        ],
        expect="""
            A table with one row per MO: **MO** index, **E (eV)**, **Occ** (2 = doubly occupied, 1 = singly
            occupied in an open-shell α/β set, 0 = virtual), then the % of each selected atom and shell
            (`6N_p` = p functions of atom 6, a nitrogen) and the totals per shell type (`Total_s`, `Total_p`...).

            * Percentages are the squared MO coefficients normalised to 100 % per MO (C² analysis, overlap
              between basis functions ignored). They are a good qualitative picture; they differ from the
              Mulliken or Löwdin orbital populations printed by some programs, especially with diffuse basis sets.
            * The atom columns appear only for systems with up to 10 atoms, or for the atoms selected with
              `-i`/`--elements`; with many MOs and no `-b`, only HOMO−5 … HOMO+5 are tabulated.
            * Contributions spread over many atoms are normal for delocalised (π, Rydberg-like) orbitals.
            * With `--beta`, the MO window and the HOMO box of the plot are currently centred on the **α**
              HOMO (for a doublet, the β HOMO is one MO lower): check the **Occ** column to find the β HOMO.

            Example (water, B3LYP/def2-TZVPP, `-b 2`): MO 4 = HOMO, −8.54 eV, 99.37 % O p (the out-of-plane
            lone pair); MO 5 = LUMO, +0.52 eV, 47.91 % O s and 23.3 % on each H s (σ* O–H).
            """,
    )
    examples = [
        ('frog-mo opt.out -b 5', 'Composition of HOMO-5 ... LUMO+5'),
        ('frog-mo opt.out -i 0 1 --orbital-type d --plot', 'd contributions of atoms 0 and 1, plus heatmap'),
        ('frog-mo opt.out --elements Fe -b 3 --plot fe_mo.svg', 'Fe contributions around the HOMO, heatmap as SVG'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='ORCA output files (with ! PrintBasis PrintMOs) or Gaussian '
                                                    'logs (with pop=full)')
        parser.add_argument('-i', '--indices', nargs='+', type=int, default=None,
                            help='Atom indices to show (0-based)')
        parser.add_argument('--elements', nargs='+', default=None, metavar='EL',
                            help="Show all atoms of these elements, e.g. 'C' 'Fe'")
        parser.add_argument('--orbital-type', nargs='+', default=None, metavar='TYPE',
                            help='Restrict to these orbital types: s, p, d, f (the heatmap also accepts '
                                 'px, dxy, ...)')
        parser.add_argument('-b', '--buffer', type=int, default=None,
                            help='Number of MOs shown on each side of the HOMO (all MOs if omitted)')
        parser.add_argument('--beta', action='store_true', help='Use beta orbitals (open-shell calculations)')
        self.add_out_dir_arg(parser)
        self.add_plot_args(parser, optional=True, default_name='<stem>_mo.png', cmap='rocket_r')

    def run(self, args):
        files = self.require_files(args.file)
        failed = []
        for p in files:
            f = str(p)
            calc = load_calculation(f)
            if getattr(calc, 'orbital_coefficients', None) is None or getattr(calc, 'mask_atoms', None) is None:
                print_warning(f'No MO coefficients found in {f} (ORCA: ! PrintBasis PrintMOs; Gaussian: pop=full)')
                failed.append(f)
                continue
            self._process(calc, p, args, multiple=len(files) > 1)
        if failed:
            raise ParseError(f'No MO coefficients in {len(failed)} file(s): {", ".join(failed)}')

    def _process(self, calc, path, args, multiple):
        matrix_percent = calculate_full_percent_matrix(calc, beta=args.beta)
        atom_indices = resolve_atoms(calc, args.indices, args.elements)
        homo_idx, start_idx, end_idx = mo_window(calc, matrix_percent, args.buffer, args.beta)

        table_atoms = atom_indices
        if table_atoms is None:
            unique_atoms = np.unique(calc.mask_atoms.astype(int))
            if len(unique_atoms) <= 10:
                table_atoms = list(unique_atoms)
            else:
                print_info(f'System has {len(unique_atoms)} atoms: the table shows shell totals only '
                           f'(use -i or --elements for atom details)')
                table_atoms = []

        mos = np.arange(start_idx, end_idx)
        if len(mos) > 50 and args.buffer is None:
            print_info('Large number of MOs: the table shows HOMO-5 to HOMO+5 only (use -b to change)')
            mos = np.arange(max(0, homo_idx - 5), min(matrix_percent.shape[1], homo_idx + 6))

        rows, headers = composition_table(calc, mos, matrix_percent, table_atoms, args.orbital_type, args.beta)
        title = f'MO Composition (%){" (beta)" if args.beta else ""} ({path})'
        self.output(rows, headers, title=title, floatfmt='.2f')

        if self.plot_enabled:
            fig = plot_coeff_matrix(calc, matrix_percent, atom_indices, args.orbital_type,
                                    args.buffer, beta=args.beta, opts=self.plot_options())
            default = self.derived_path(path, 'mo_beta' if args.beta else 'mo', '')
            tag = path.stem if (multiple and args.plot) else None
            self.save_figure(fig, self.plot_path(default, tag))


def parse_args():
    return MoCoeff().build_parser().parse_args()


def main():
    MoCoeff().main()


if __name__ == '__main__':
    main()
