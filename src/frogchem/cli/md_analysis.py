"""
FROG MD Analysis Tool.
Automated analysis using MDTraj for structure and direct parsing for thermodynamics.
Generates a combined plot of Temperature, Pressure, Total Energy, and Fragment RMSD.

Andrea Pellegrini, 2026
"""

import re

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.printer import print_info, print_warning

# Amber prints running averages / fluctuations as extra NSTEP blocks after these markers
_AVERAGE_MARKERS = ("A V E R A G E S", "F L U C T U A T I O N S")


def compute_rmsd(prmtop, traj_files, mask):
    """Fragment RMSD (Å) of every frame vs the first one. Returns a DataFrame or None."""
    import pandas as pd
    import mdtraj as md
    print_info("Loading trajectory files with MDTraj...")
    try:
        traj = md.load(traj_files, top=prmtop)
        print_info(f"Calculating RMSD on mask '{mask}' (reference: first frame)...")
        fragment_indices = traj.topology.select(mask)
        if len(fragment_indices) == 0:
            print_warning(f"No atoms found for mask '{mask}'. Skipping RMSD.")
            return None
        # MDTraj returns RMSD in nm
        data_rmsd = md.rmsd(traj, traj, 0, atom_indices=fragment_indices) * 10.0
        return pd.DataFrame({'Frame': range(len(data_rmsd)), 'RMSD_Angstrom': data_rmsd})
    except Exception as e:
        print_warning(f"Structure analysis failed: {e}")
        return None


def parse_thermo(mdout_files):
    """Parse step, time (ps), temperature (K), pressure (bar) and Etot (kcal/mol) from Amber mdout files.

    The averages / RMS-fluctuation blocks Amber prints are skipped. Returns a DataFrame or None.
    """
    print_info("Parsing thermodynamic data from the mdout files...")
    data = []

    re_nstep = re.compile(r"NSTEP\s*=\s*(\d+)\s+TIME\(PS\)\s*=\s*([-\d\.]+)\s+TEMP\(K\)\s*=\s*([-\d\.]+)")
    re_press = re.compile(r"PRESS\s*=\s*([-\d\.]+)")
    re_ener = re.compile(r"Etot\s*=\s*([-\d\.]+)\s+EKtot\s*=\s*([-\d\.]+)\s+EPtot\s*=\s*([-\d\.]+)")

    for fname in mdout_files:
        current_step = {}
        skip_next_block = False
        with open(fname, 'r') as f:
            for line in f:
                if any(m in line for m in _AVERAGE_MARKERS):
                    if current_step:
                        data.append(current_step)
                    current_step = {}
                    skip_next_block = True
                elif "NSTEP" in line and "TIME" in line:
                    if current_step:
                        data.append(current_step)
                    current_step = {}
                    if skip_next_block:
                        skip_next_block = False
                        continue
                    m = re_nstep.search(line)
                    if m:
                        current_step = {
                            'Step': int(m.group(1)),
                            'Time': float(m.group(2)),
                            'Temp': float(m.group(3)),
                            'Press': None,
                            'Etot': None,
                        }
                        m_p = re_press.search(line)
                        if m_p:
                            current_step['Press'] = float(m_p.group(1))
                elif current_step:
                    if "Etot" in line:
                        m = re_ener.search(line)
                        if m:
                            current_step['Etot'] = float(m.group(1))
                    elif "PRESS" in line and current_step.get('Press') is None:
                        m = re_press.search(line)
                        if m:
                            current_step['Press'] = float(m.group(1))
            if current_step:
                data.append(current_step)

    if not data:
        print_warning("No thermodynamic data found.")
        return None
    import pandas as pd
    return pd.DataFrame(data)


#: Default panel colours: temperature, pressure, total energy, RMSD (--colors overrides them in this order)
MD_COLORS = ('red', 'blue', 'green', 'purple')


def plot_combined_data(df_thermo, df_rmsd, opts=None):
    """One panel per available quantity (temperature, pressure, total energy, RMSD). Returns the figure.

    opts: frogchem.plotter.PlotOptions; the title goes above the panels, colors follow MD_COLORS order.
    """
    from dataclasses import replace

    from frogchem import plotter

    opts = (opts or plotter.PlotOptions()).with_defaults(colors=MD_COLORS, grid=True, legend=False)
    colors = plotter.series_colors(opts, len(MD_COLORS))

    panels = []  # (x, y, colour, ylabel, xlabel)
    if df_thermo is not None and not df_thermo.empty:
        for col, color, ylabel in (('Temp', colors[0], 'Temperature (K)'),
                                   ('Press', colors[1], 'Pressure (bar)'),
                                   ('Etot', colors[2], plotter.energy_label('Total Energy'))):
            if col not in df_thermo.columns:
                continue
            series = df_thermo[col].dropna()  # pressure is missing for NVT runs
            if not series.empty:
                panels.append((df_thermo.loc[series.index, 'Time'], series, color, ylabel, 'Time (ps)'))
    if df_rmsd is not None and not df_rmsd.empty:
        panels.append((df_rmsd['Frame'], df_rmsd['RMSD_Angstrom'], colors[3], 'Fragment RMSD (Å)', 'Frame'))
    if not panels:
        return None

    opts = opts.with_defaults(figsize=(10, 3 * len(panels)))
    fig, axs = plotter.new_figure(len(panels), 1, opts=opts, squeeze=False, layout='constrained')
    for ax, (x, y, color, ylabel, xlabel) in zip(axs[:, 0], panels):
        panel = replace(opts, title=None, colors=(color,)).with_defaults(xlabel=xlabel, ylabel=ylabel)
        plotter.line(x, y, ax=ax, opts=panel)
    if opts.title:
        fig.suptitle(opts.title, fontweight='bold')
    return fig


class MdAnalysis(CLICommand, name='frog-md'):
    """Analyse Amber MD runs: fragment RMSD (MDTraj) and thermodynamics from mdout files.

    The RMSD of the atoms selected by --mask is computed for every frame against the
    first one. Temperature, pressure and total energy are read from the mdout logs
    (Amber's averages / RMS-fluctuation blocks are skipped). Both are written to CSV
    and drawn in one summary figure.
    """
    category = 'geometry'
    has_tables = False
    outputs = [
        Output('file', '<prefix>_rmsd.csv', 'Frame and fragment RMSD (Å) vs the first frame'),
        Output('file', '<prefix>_thermo.csv', 'Step, time (ps), temperature (K), pressure (bar), '
                                              'Etot (kcal/mol)', when='--mdout'),
        Output('plot', '<prefix>_summary.png', 'Temperature, pressure, total energy and RMSD panels '
                                               '(--colors sets the panel colours in this order)'),
    ]
    guide = Guide(
        why="""
            After a molecular dynamics (MD) run with Amber you want to know, before analysing anything else,
            whether the simulation is **stable**: did the temperature and pressure settle at the target values,
            is the total energy steady, and does the molecule (or a ligand in a protein) stay in place or drift
            away? This command gives a one-page check: the RMSD of a chosen group of atoms along the trajectory
            (RMSD = root-mean-square deviation from the first frame, after optimal superposition) and the
            temperature, pressure and total-energy traces from the Amber log.
            """,
        needs="""
            * The Amber **topology** (`.prmtop`) used for the run (`-p`).
            * One or more **trajectory** files (`.nc` NetCDF or `.mdcrd`), given in time order (`-y`); they
              are concatenated.
            * Optionally the Amber **mdout** log(s) (`--mdout`, the file written with `-o` by `pmemd`/`sander`)
              for temperature, pressure and energy; how often they are printed depends on `ntpr` in the
              Amber input.
            * The Python package **MDTraj** (`pip install mdtraj`), used to read the trajectory.
            """,
        how=[
            "Decide which atoms to follow and write an MDTraj selection for `--mask`: `\"resname LIG\"` for a "
            "ligand, `\"protein and name CA\"` for the protein backbone, `\"resid 10 to 20\"` for a segment; "
            "the default is `protein`.",
            "Run `frog-md -p sys.prmtop -y prod.nc --mdout prod.out --mask \"resname LIG\"`.",
            "For a run split in chunks give all of them in order: `-y md1.nc md2.nc --mdout md1.out md2.out`.",
            "Choose the output names with `-o PREFIX` (default `analysis`): you get `PREFIX_rmsd.csv`, "
            "`PREFIX_thermo.csv` and `PREFIX_summary.png` (`--fig-format pdf` / `svg` for vector figures).",
            "Open the summary figure first; use the CSV files to make your own plots or averages.",
        ],
        expect="""
            No table on screen: two CSV files and one figure with one panel per available quantity —
            Temperature (K), Pressure (bar) and Total Energy (kcal/mol) versus time (ps), and fragment RMSD (Å)
            versus frame number.

            * **Temperature** should fluctuate around the thermostat target (e.g. 300 K ± a few K); a slow drift
              means the system is not equilibrated yet.
            * **Pressure** fluctuates a lot (hundreds of bar) even in a good NPT run: look at the average, not
              single points. In NVT runs Amber writes no meaningful pressure (often `PRESS = 0.0`), so that
              panel is flat or absent.
            * **Total energy** should be flat or drift slowly; jumps point to a problem in the run (or to
              chunks given out of order).
            * **RMSD** normally rises in the first frames and then reaches a plateau (often 1–3 Å for a protein
              backbone, below ~2 Å for a ligand that stays bound). A curve that keeps rising means the group is
              changing conformation or leaving its site.
            * The RMSD reference is the first frame of the first trajectory file. The RMSD x-axis is the frame
              index, not ps: multiply by the time between saved frames (`ntwx` × `dt`) to convert.
            * `No atoms found for mask` means the selection matched nothing: RMSD is skipped (check residue
              names in the prmtop). Amber's averages / RMS-fluctuation blocks at the end of the log are skipped
              automatically.
            """,
    )
    examples = [
        ('frog-md -p sys.prmtop -y prod.nc --mdout prod.out --mask "resname LIG"',
         'RMSD of the ligand and thermodynamics, written as analysis_*'),
        ('frog-md -p sys.prmtop -y md1.nc md2.nc --mdout md1.out md2.out -o run1 --fig-format pdf',
         'Several trajectory chunks, files named run1_*'),
    ]

    def setup_args(self, parser):
        parser.add_argument('-p', '--prmtop', required=True, help='Topology file (.prmtop)')
        parser.add_argument('-y', '--trajectory', required=True, nargs='+', help='Trajectory files (.nc, .mdcrd)')
        parser.add_argument('--mdout', nargs='+', default=None, help='Amber output logs (.out) for thermodynamics')
        parser.add_argument('--mask', default='protein',
                            help="MDTraj selection for the RMSD (e.g. 'resname LIG' or 'resid 10 to 20')")
        self.add_output_arg(parser, default='analysis', metavar='PREFIX',
                            help='Prefix of the output files: PREFIX_rmsd.csv, PREFIX_thermo.csv, PREFIX_summary.png')
        self.add_plot_args(parser, optional=False)

    def run(self, args):
        self.require_files([args.prmtop, *args.trajectory, *(args.mdout or [])])

        df_rmsd = compute_rmsd(args.prmtop, args.trajectory, args.mask)
        if df_rmsd is not None:
            path = self.output_path('analysis', 'csv', tag='rmsd')
            df_rmsd.to_csv(path, index=False)
            self.saved(path, 'RMSD')

        df_thermo = parse_thermo(args.mdout) if args.mdout else None
        if df_thermo is not None:
            path = self.output_path('analysis', 'csv', tag='thermo')
            df_thermo.to_csv(path, index=False)
            self.saved(path, 'thermodynamics')

        if df_rmsd is None and df_thermo is None:
            raise InputError("No RMSD or thermodynamic data could be extracted")

        print_info("Generating the summary plot...")
        fig = plot_combined_data(df_thermo, df_rmsd, opts=self.plot_options())
        if fig is None:
            raise InputError("Nothing to plot: all parsed quantities are empty")
        self.save_figure(fig, self.plot_path('analysis', tag='summary'))


def parse_args():
    return MdAnalysis().build_parser().parse_args()


def main():
    MdAnalysis().main()


if __name__ == '__main__':
    main()
