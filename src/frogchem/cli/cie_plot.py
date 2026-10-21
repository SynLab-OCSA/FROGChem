import glob
import os

import numpy as np
import pandas as pd

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.printer import print_info, print_warning

# ponytail: colour-science and matplotlib are lazily imported inside run()


def spectrum_to_XYZ(colour, sd, cmfs, illuminant, spectrum_type: str):
    """CIE XYZ of a spectral distribution.

    'emission': the spectrum is the emitted spectral power itself, X = k∫S(λ)x̄(λ)dλ, with no
    illuminant (colour-science's equal-energy default leaves the chromaticity unchanged).
    'absorption'/'transmittance': `sd` is a transmittance T(λ) (already 10^-A for absorption)
    viewed under `illuminant` (D65), X = (1/k)∫T S_D65 x̄ dλ with k = ∫S_D65 ȳ dλ.
    """
    if spectrum_type == 'emission':
        return colour.sd_to_XYZ(sd, cmfs)
    return colour.sd_to_XYZ(sd, cmfs, illuminant)


class CiePlot(CLICommand, name='frog-cie'):
    """Place emission/absorption spectra on the CIE 1931 chromaticity diagram.

    Each CSV spectrum is converted to CIE 1931 xy chromaticity coordinates with
    colour-science (2° standard observer) and drawn on the diagram in its sRGB colour.
    --spectrum-type selects the physics: 'emission' (default) integrates the emitted
    spectrum directly; 'absorption' converts the absorbance to T = 10^-A and 'transmittance'
    uses T as given, both viewed under the D65 illuminant (perceived colour of the sample).
    Spectra with a non-standard wavelength step are interpolated to 1 nm.
    Requires `pip install colour-science`.
    """
    category = 'spectroscopy'
    outputs = [
        Output('table', 'CIE coordinates', 'One row per spectrum: CIE 1931 x and y chromaticity coordinates'),
        Output('plot', 'cie_diagram.png', 'CIE 1931 chromaticity diagram with one point per spectrum '
                                          '(path from -o/--output)'),
    ]
    guide = Guide(
        why="""
            The colour of an emitter (OLED, fluorescent dye) or of an absorbing sample is reported in papers as
            CIE 1931 chromaticity coordinates (x, y): two numbers that place the colour on the standard
            horseshoe-shaped colour diagram. This command converts spectra (measured or simulated) into (x, y),
            prints them and draws all of them on the diagram, so you can compare compounds, check how close an
            emitter is to a target colour, or compare with literature values.
            """,
        needs="""
            One **CSV file per spectrum** with a header row, a wavelength column in **nm** (default name
            `wavelength`) and an intensity column (default name `intensity`); other names are given with
            `--wl` and `--int`. The spectrum should cover the visible range (about 380–780 nm); parts outside
            it do not contribute. Choose what the intensity means with `--spectrum-type`: an **emission**
            spectrum (any intensity scale), an **absorbance** A (as measured, so its absolute size matters),
            or a **transmittance**/reflectance between 0 and 1. A simulated spectrum from
            `frog-tddft --uv --save-spectra` works too: use `--wl Wavelength_nm --int <curve column>` (it is
            normalised to a maximum of 1, i.e. treated as absorbance 1 at the peak). Requires the optional
            package `colour-science` (`pip install colour-science`).
            """,
        how=[
            "Emission: `frog-cie em1.csv em2.csv` (wildcards like `spectra/*.csv` are accepted) → table of "
            "x, y and `cie_diagram.png` in the current folder.",
            "Colour of an absorbing solution or film: `frog-cie uv.csv --spectrum-type absorption` "
            "(or `transmittance` if the column is T).",
            "Add literature or target points with `--external-coordinates ref.csv` (columns `Name`, `CIEx`, "
            "`CIEy`); change the plot name/format with `-o colours.pdf`, open it with `--show`.",
        ],
        expect="""
            A table with one row per spectrum: **CIE x** and **CIE y** (0–1, 4 decimals), and the diagram with
            one dot per spectrum drawn in its own sRGB colour (reference points as crosses).

            * Emission (`emission`): the spectrum is integrated directly with the 2° standard observer, so only
              its **shape** matters. A narrow band gives a point close to the curved edge (pure, saturated colour);
              broad or multiple bands move it towards the white region (x ≈ y ≈ 0.31–0.33).
            * Absorption/transmittance: the light transmitted under daylight (illuminant D65) is used, i.e. the
              **perceived colour of the sample**, which is the complementary of what it absorbs (absorbing blue
              ⇒ looks yellow). Here the size of A matters: a weak absorber stays close to white.
            * Spectra with a step other than 1, 5, 10 or 20 nm are interpolated to 1 nm (info message).
              A file with wrong column names is skipped with a warning; if nothing can be converted the command
              stops with an error.

            Example (synthetic Gaussian bands): emission at 520 nm (σ 20 nm) → x 0.1468, y 0.6902 (green);
            emission at 620 nm, 2 nm step → x 0.6706, y 0.3292 (red); absorbance 1.0 at 450 nm (σ 30 nm) with
            `--spectrum-type absorption` → x 0.4160, y 0.4748 (yellow).
            """,
    )
    examples = [
        ('frog-cie spectra/*.csv', 'Coordinates of every spectrum and cie_diagram.png'),
        ('frog-cie emission.csv --wl nm --int counts -o emission_cie.pdf',
         'Custom column names, save as PDF'),
        ('frog-cie uv_abs.csv --spectrum-type absorption',
         'Perceived colour of an absorbing sample (T = 10^-A under D65)'),
        ('frog-cie *.csv --external-coordinates literature.csv --show',
         'Also mark reference points (columns Name, CIEx, CIEy) and open the plot window'),
    ]

    def setup_args(self, parser):
        parser.add_argument('inputs', nargs='+', help='CSV spectrum files (glob patterns allowed)')
        parser.add_argument('--wl', default='wavelength', help='Name of the wavelength column (nm)')
        parser.add_argument('--int', default='intensity', help='Name of the intensity column')
        parser.add_argument('--spectrum-type', choices=['emission', 'absorption', 'transmittance'],
                            default='emission',
                            help="How to read the intensity column: 'emission' = emitted spectral power "
                                 "(integrated without illuminant), 'absorption' = absorbance A, converted to "
                                 "T = 10^-A and viewed under D65, 'transmittance' = T (or reflectance) in 0-1, "
                                 "viewed under D65 (default: emission)")
        parser.add_argument('--external-coordinates', metavar='CSV',
                            help='CSV file with extra points to mark (columns Name, CIEx, CIEy)')
        self.add_output_arg(parser, help='Plot path (extension overrides --fig-format)')
        self.add_plot_args(parser, optional=False)

    def run(self, args):
        files = []
        for inp in args.inputs:
            matches = sorted(glob.glob(inp))
            if not matches:
                raise InputError(f"File not found: {inp}")
            files.extend(matches)
        if args.external_coordinates:
            self.require_files([args.external_coordinates])

        try:
            import colour
            import colour.plotting as cl_plot
        except ImportError:
            raise InputError("The library 'colour-science' is not installed. "
                             "Install it with: pip install colour-science")
        from dataclasses import replace
        from itertools import cycle

        from frogchem import plotter

        opts = self.plot_options(title="CIE 1931 Chromaticity Diagram", figsize=(10, 11),
                                 xlim=(-0.1, 0.9), ylim=(-0.1, 0.9))
        fig, ax = plotter.new_figure(opts=opts)
        cl_plot.plot_chromaticity_diagram_CIE1931(
            standalone=False,
            axes=ax,
            title=opts.title,
            show_diagram_colours=True,
            show_spectral_locus=True,
        )
        # Spectra are drawn in their perceived colour; --colors applies to the external points
        point_opts = replace(opts, colors=None, title=None, legend=False)
        ext_colors = cycle(opts.colors) if opts.colors else None

        print_info(f"Processing {len(files)} spectra with colour-science...")

        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        illuminant = colour.SDS_ILLUMINANTS["D65"]

        rows = []
        for fpath in files:
            name = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath)
                df.columns = df.columns.str.strip()
                df = df.sort_values(by=args.wl)

                values = df[args.int].to_numpy(dtype=float)
                if args.spectrum_type == 'absorption':
                    values = 10.0 ** (-values)      # absorbance -> transmittance (tddft.md §3)
                data = dict(zip(df[args.wl], values))
                sd = colour.SpectralDistribution(data, name=name)

                if sd.shape.interval not in (1, 5, 10, 20):
                    print_info(f"{name}: non-standard step of {sd.shape.interval} nm, interpolating to 1 nm")
                    sd = sd.interpolate(colour.SpectralShape(sd.shape.start, sd.shape.end, 1))

                XYZ = spectrum_to_XYZ(colour, sd, cmfs, illuminant, args.spectrum_type)
                x, y = (float(v) for v in colour.XYZ_to_xy(XYZ))
                if args.spectrum_type == 'emission':
                    # emissive colour: arbitrary intensity scale, show the chromaticity at Y = 0.5
                    XYZ_disp = XYZ / XYZ[1] * 0.5 if XYZ[1] > 0 else XYZ
                else:
                    XYZ_disp = XYZ / 100
                RGB = np.clip(colour.XYZ_to_sRGB(XYZ_disp), 0, 1)
            except Exception as e:
                print_warning(f"{fpath}: {e}")
                continue

            label = name[:-4] if name.lower().endswith('.csv') else name
            rows.append([label, x, y])
            plotter.scatter([x], [y], labels=label, annotate=[label], size=80, ax=ax, opts=point_opts,
                            color=[RGB], edgecolors='black', zorder=10)

        n_external = 0
        if args.external_coordinates:
            ext = pd.read_csv(args.external_coordinates)
            for _, row in ext.iterrows():
                try:
                    cx, cy = float(row['CIEx']), float(row['CIEy'])
                except ValueError:
                    print_warning(f"Skipping {row['Name']}: invalid coordinates ({row['CIEx']}, {row['CIEy']})")
                    continue
                color = {'color': next(ext_colors)} if ext_colors else {}
                plotter.scatter([cx], [cy], labels=str(row['Name']), size=30, ax=ax, opts=point_opts,
                                marker='x', zorder=10, **color)
                n_external += 1

        if not rows and not n_external:
            plotter.close(fig)
            raise InputError("No spectrum could be converted to CIE coordinates")

        self.output(rows, ['Spectrum', 'CIE x', 'CIE y'], title='CIE coordinates', floatfmt='.4f')

        n_entries = len(ax.get_legend_handles_labels()[0])
        plotter.finish(ax, opts, loc=opts.legend_loc or 'upper left', bbox_to_anchor=(1.02, 1),
                       borderaxespad=0, ncol=2 if n_entries > 20 else 1, title="Legend",
                       frameon=True, framealpha=1.0, fontsize='small')
        self.save_figure(fig, self.plot_path('cie_diagram'))


def parse_args():
    return CiePlot().build_parser().parse_args()


def main():
    CiePlot().main()


if __name__ == '__main__':
    main()
