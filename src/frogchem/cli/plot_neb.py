from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.orca_parser.neb import parse_neb_interp
from frogchem.printer import print_warning

# ponytail: matplotlib is lazily loaded inside run()


class PlotNeb(CLICommand, name='frog-plot-neb'):
    """Plot the NEB energy profiles of every iteration from ORCA .interp files.

    Images are drawn as points and the interpolated path as a line: the first iteration
    in black, intermediate iterations in light grey and the last one in red. Several
    .interp files are drawn on the same axes.
    """
    category = 'thermo'
    has_tables = False
    outputs = [
        Output('plot', 'neb.png', 'NEB images and interpolated path of every iteration '
                                  '(path from -o/--output; --colors FIRST MIDDLE LAST colours the '
                                  'first, intermediate and last iteration)'),
    ]
    guide = Guide(
        why="""
            A *nudged elastic band* (NEB) calculation searches a reaction path by optimising a chain of
            structures ("images") between reactant and product; its highest image approximates the
            transition state. This command draws the energy along the chain at every iteration, so you can
            see whether the path has converged, where the barrier is and whether an intermediate appears
            along the way, before you start a transition-state optimisation from the highest image.
            """,
        needs="""
            The `.interp` file that ORCA writes during an NEB job (`! NEB` or `! NEB-CI`, or `! NEB-TS`,
            with `%neb ... end`), e.g. `myjob.interp` next to `myjob.out`. It contains, for each
            iteration, the image energies and an interpolated path. The file grows during the run, so you
            can also plot an NEB that is still running.
            """,
        how=[
            "Run `frog-plot-neb myjob.interp` (writes `neb.png` in the current directory).",
            "Choose name and format with `-o ts_search.pdf` (or `--fig-format svg`) and add `--show` to open the plot window.",
            "To compare runs, give several files: `frog-plot-neb run1.interp run2.interp` draws them on the same axes.",
        ],
        expect="""
            One plot: x = reaction coordinate (first column of the file, normally the path position
            from 0 = reactant to 1 = product), y = energy in **kcal/mol** (third column, converted from Eh
            with no shift; ORCA writes these energies relative to the first image, which therefore sits
            at 0).
            The **first** iteration is drawn in black, intermediate iterations as faint grey lines, the
            **last** iteration in red; dots are the images, lines the interpolation.

            * A converged NEB has red dots lying on a smooth curve with a single maximum: that maximum is
              your TS guess (barrier = its height). Grey lines close to the red one mean the last
              iterations barely changed the path.
            * Two maxima with a minimum between them mean a stepwise path: split it into two NEBs.
            * A barrier of hundreds of kcal/mol usually means atoms are mapped differently in reactant and
              product (atom order must be identical in the two structures).
            * Nothing is printed on the terminal; `No NEB iterations found` means the file is not an ORCA
              `.interp` file or is still empty.
            """,
    )
    examples = [
        ('frog-plot-neb neb.interp', 'Save neb.png in the current directory'),
        ('frog-plot-neb neb.interp -o ts_search.pdf --show', 'Save as PDF and open the plot window'),
    ]

    def setup_args(self, parser):
        parser.add_argument('filename', nargs='+', help='ORCA NEB interpolation files (e.g. neb.interp)')
        self.add_output_arg(parser, help='Plot path (extension overrides --fig-format)')
        self.add_plot_args(parser, optional=False)

    def run(self, args):
        files = self.require_files(args.filename)
        from dataclasses import replace

        from frogchem import plotter

        opts = self.plot_options(xlabel='Reaction coordinate', ylabel=plotter.energy_label(),
                                 colors=NEB_COLORS, legend=False)
        colors = plotter.series_colors(opts, 3)
        # first, intermediate and last iteration: (colour, curve alpha, curve line width)
        looks = [(colors[0], 0.8, 0.8), (colors[1], 0.1, 0.2), (colors[2], 1.0, 1.0)]

        fig, ax = plotter.new_figure(opts=opts)
        n_plotted = 0
        for p in files:
            imags, interpols = parse_neb_interp(str(p))
            if not imags:
                print_warning(f"No NEB iterations found in {p}")
                continue
            n_plotted += 1
            for idx, curve in enumerate(interpols):
                color, alpha, width = looks[_iteration_kind(idx, len(interpols))]
                plotter.line(curve[:, 0], curve[:, 1], ax=ax, opts=replace(opts, colors=(color,)),
                             alpha=alpha, linewidth=width)
            for idx, images in enumerate(imags):
                color = looks[_iteration_kind(idx, len(imags))][0]
                plotter.scatter(images[:, 0], images[:, 1], ax=ax, opts=replace(opts, colors=(color,)))

        if not n_plotted:
            plotter.close(fig)
            raise InputError("No NEB iterations found in the input files")

        self.save_figure(fig, self.plot_path('neb'))


#: Default colours of the first, intermediate and last NEB iteration (--colors overrides them)
NEB_COLORS = ('black', 'gray', 'red')


def _iteration_kind(idx: int, n: int) -> int:
    """0 for the first NEB iteration, 2 for the last one, 1 for those in between."""
    return 0 if idx == 0 else 2 if idx == n - 1 else 1


def parse_args():
    return PlotNeb().build_parser().parse_args()


def main():
    PlotNeb().main()


if __name__ == '__main__':
    main()
