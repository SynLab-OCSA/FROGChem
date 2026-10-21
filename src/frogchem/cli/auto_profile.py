"""Reaction profile plot built directly from ORCA/Gaussian output files.

Extracts E, E+ZPVE, H or G (mRRHO, recomputed from the frequencies) of each stationary
point, converts them to kcal/mol relative to the first file and draws the profile.
Optionally annotates each state with the %V_bur around one atom.
"""

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError, ParseError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning
from frogchem.reaction_plotter.plotter import ReactionPlotter
from frogchem.steric_maps import compute_buried_volume, get_sambvca_radii, orient_positions
from frogchem.thermo import free_gibbs_energy

HARTREE_TO_KCAL = 627.5094740631
ENERGY_TERMS = ('E', 'EZPVE', 'H', 'G')


def thermo_energies(calc, temp: float = 298.15) -> dict:
    """E, ZPVE, E+ZPVE, H, G (Eh; mRRHO at `temp`) and imaginary frequencies of a calculation.

    Thermal terms are NaN when the calculation has no frequencies.
    """
    nan = float('nan')
    e = getattr(calc, 'final_energy', None)
    if e is None:
        e = getattr(calc, 'scf_energy', None)
    out = {'E': nan if e is None else float(e), 'ZPVE': nan, 'EZPVE': nan, 'H': nan, 'G': nan, 'imag': []}
    freq = getattr(calc, 'freq', None)
    if e is None or not isinstance(freq, np.ndarray) or freq.size == 0:
        return out
    out['imag'] = [float(f) for f in freq[freq < 0]]
    try:
        g, zpve, h, _s = free_gibbs_energy(SCF=float(e), T=temp, freq=freq[freq > 0], mw=calc.mw,
                                           B=calc.B, m=int(calc.mult))
    except Exception as exc:  # e.g. missing rotational constants
        print_warning(f"Thermochemistry failed for {getattr(calc, 'fname', 'calculation')}: {exc}")
        return out
    out.update(ZPVE=zpve, EZPVE=float(e) + zpve, H=h, G=g)
    return out


def buried_volume(atoms, geom, center: int, radius: float = 3.5) -> float:
    """%V_bur (SambVca radii) in a sphere of `radius` Å around atom `center` (excluded)."""
    coords = orient_positions(np.asarray(geom, dtype=float), center)
    selected = [i for i in range(len(atoms)) if i != center]
    radii = get_sambvca_radii([atoms[i] for i in selected])
    return compute_buried_volume(coords[selected], radii, radius=radius)['total']


class AutoProfile(CLICommand, name='frog-autoprofile'):
    """Plot a reaction profile directly from ORCA/Gaussian output files.

    Files are given in reaction order (e.g. R TS P). Energies (E, E+ZPVE, or mRRHO H/G
    recomputed from the frequencies at -T) are converted to kcal/mol relative to the first
    file. With --steric-center each state is annotated with its %V_bur.
    """
    category = 'thermo'
    outputs = [
        Output('table', 'Reaction Profile', 'One row per state: label, file, energy (Eh), relative energy '
                                            '(kcal/mol) and %V_bur (with --steric-center)'),
        Output('plot', 'reaction_profile.png', 'Profile plot (path from -o)'),
    ]
    guide = Guide(
        why="""
            The quickest way to see a reaction's energy profile: give the outputs of the stationary
            points (reactant, transition state, intermediate, product) in reaction order and get a table
            of relative energies in kcal/mol plus a plot, without typing any number. Use it for a first
            look at a mechanism or to check a barrier; for a figure for a paper, with several pathways,
            branches and custom labels, write an input for `frog-profile` instead.
            """,
        needs="""
            One ORCA or Gaussian output per state, **all at the same level of theory and for the same
            set of atoms** (the energies are only comparable if every structure contains the same
            molecules: add separated reactants into one file, or compute them as a complex).
            For G or H (the default is G) each output must come from a frequency job (`! Opt Freq`,
            `! OptTS Freq`); a file without frequencies falls back to E with a warning.
            For `--steric-center` the atom number (0-based) of the centre, e.g. the metal, which must be
            the same atom in every file.
            """,
        how=[
            "Run `frog-autoprofile R.out TS.out P.out` (order = order along the reaction coordinate).",
            "Give readable names with `-l R TS1 INT P` (one label per file) and choose the energy with "
            "`--energy E`, `EZPVE`, `H` or `G` (default G, recomputed at `-T`, default 298.15 K).",
            "Choose the plot file with `-o profile.svg` (default `reaction_profile.png` in the current "
            "directory; the extension sets the format, otherwise `--fig-format`).",
            "Optionally add `--steric-center 0` to print and annotate the buried volume %V_bur "
            "(share of a 3.5 Å sphere around that atom occupied by the other atoms).",
        ],
        expect="""
            A *Reaction Profile* table (state, file, absolute energy in Eh, relative energy in kcal/mol,
            and %V_bur with `--steric-center`) and the plot: a smooth curve through the states, the first
            state at 0 kcal/mol.

            * The G values are the same as `frog-rrho` with default settings (e.g. `test.out`
              −800.642826 Eh); the relative energies use 1 Eh = 627.509 kcal/mol.
            * Typical barriers are 5–30 kcal/mol, reaction energies −50 to +20 kcal/mol. Relative energies of
              hundreds or thousands of kcal/mol mean the files do not contain the same atoms (a test with
              three unrelated molecules gave −527019 kcal/mol): check composition and charge.
            * A warning `No frequencies in ... using E instead of G` means that file has no frequencies;
              mixing E for one state and G for the others gives a meaningless profile.
            * Only Grimme's quasi-RRHO with default parameters (σ = 1, 1 atm) is used here; use
              `frog-rrho` to change them.
            * %V_bur values: 50.7 % around atom 0 of `test.out`, 83.3 % around atom 0 of `orb.out` (centres
              buried inside a molecule give high values); for the metal of a catalyst with its ligands,
              values of 25–50 % are common.
            """,
    )
    examples = [
        ('frog-autoprofile R.out TS.out P.out', 'G profile saved as reaction_profile.png'),
        ('frog-autoprofile R.out TS.out P.out -l R TS P --energy E -o profile.svg',
         'Electronic-energy profile with custom labels, as SVG'),
        ('frog-autoprofile R.out TS.out P.out --steric-center 0', 'Annotate %V_bur around atom 0'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', help="Output files in reaction order (e.g. R.out TS.out P.out)")
        parser.add_argument('-l', '--labels', nargs='+', help="Labels of the states (default: file names)")
        parser.add_argument('--energy', choices=ENERGY_TERMS, default='G',
                            help='Energy plotted: E, E+ZPVE, H or G (mRRHO from the frequencies)')
        parser.add_argument('-T', '--temp', type=float, default=298.15, help='Temperature in K')
        parser.add_argument('--steric-center', type=int, metavar='INDEX',
                            help="Atom (0-based) around which %%V_bur (3.5 Å sphere) is computed and annotated")
        self.add_output_arg(parser, help='Plot path (default: reaction_profile.<fig-format>)')
        self.add_plot_args(parser, optional=False)

    def run(self, args):
        paths = self.require_files(args.files)
        labels = args.labels if args.labels else [p.stem for p in paths]
        if len(labels) != len(paths):
            raise InputError(f"{len(labels)} labels given for {len(paths)} files")

        energies, v_burs = [], []
        for p in paths:
            print_info(f"Parsing {p.name}...")
            calc = load_calculation(str(p))
            terms = thermo_energies(calc, args.temp)
            val = terms[args.energy]
            if np.isnan(val):
                if np.isnan(terms['E']):
                    raise ParseError(f"Could not extract the energy from {p.name}")
                print_warning(f"No frequencies in {p.name}: using E instead of {args.energy}")
                val = terms['E']
            energies.append(val)

            if args.steric_center is not None:
                if not 0 <= args.steric_center < len(calc.atoms):
                    raise InputError(f"--steric-center {args.steric_center} out of range for {p.name}")
                v_burs.append(buried_volume(calc.atoms, calc.geom, args.steric_center))

        rel_energies = [(e - energies[0]) * HARTREE_TO_KCAL for e in energies]

        headers = ['State', 'File', f'{args.energy} (Eh)', f'Δ{args.energy} (kcal/mol)']
        rows = [[lab, p.name, e, rel] for lab, p, e, rel in zip(labels, paths, energies, rel_energies)]
        if v_burs:
            headers.append('%V_bur')
            for row, v in zip(rows, v_burs):
                row.append(v)
        self.output(rows, headers, title='Reaction Profile', floatfmt='.6f')

        final_labels = [f"{lab}\n(%V_bur: {v:.1f}%)" for lab, v in zip(labels, v_burs)] if v_burs else labels
        x_positions = [i * 2.0 for i in range(len(rel_energies))]

        from frogchem import plotter

        opts = self.plot_options(title="Reaction Profile", ylabel=plotter.energy_label(args.energy, delta=True),
                                 colors=('teal',), figsize=(8, 5), legend=False)
        line_kw = {}
        if opts.linewidth is not None:
            line_kw['linewidth'] = opts.linewidth
        if opts.linestyle:
            line_kw['linestyle'] = plotter.LINESTYLES.get(opts.linestyle, opts.linestyle)
        profile = ReactionPlotter(figsize=opts.figsize)
        profile.plot_reaction_path(
            data=np.array(rel_energies), x_positions=x_positions, color=plotter.series_colors(opts, 1)[0],
            label="Reaction Path", path_index=0, total_paths=1, path_label="Reaction Path", **line_kw,
        )
        profile.add_custom_tick_labels(
            x_positions=x_positions, labels=final_labels, color='black',
            path_index=0, total_paths=1, path_label=None,
        )
        ax = profile.ax
        ax.set_xticks([])
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        plotter.finish(ax, opts)
        self.save_figure(profile.fig, self.plot_path('reaction_profile'))


def parse_args():
    return AutoProfile().build_parser().parse_args()


def main():
    AutoProfile().main()


if __name__ == '__main__':
    main()
