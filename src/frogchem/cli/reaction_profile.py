#!/usr/bin/env python3
"""Plot reaction profiles from YAML input files.

Generates publication-quality 2D reaction coordinate diagrams with Bezier curves,
customizable markers, and multi-step annotations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import InputError
from frogchem.printer import print_info


def _load_optional_json(path_str: Optional[str], what: str) -> Optional[Any]:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        # Try parsing as literal JSON string
        try:
            return json.loads(path_str)
        except json.JSONDecodeError:
            raise InputError(f"{what} file not found: '{path_str}'")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise InputError(f"{what} file '{path_str}' is not valid JSON: {e}") from e


class ReactionProfile(CLICommand, name='frog-profile'):
    """Plot reaction energy profiles from YAML input files.

    Draws Bezier-curve reaction coordinate diagrams with point markers, energy labels,
    segment annotations, an optional secondary y-axis and section brackets. One figure
    is saved per input file; the look is set by --style and the input's own options.
    """
    category = 'thermo'
    has_tables = False
    outputs = [
        Output('plot', '<stem>_profile.png', 'Reaction profile, next to the input (or --out-dir); '
                                             '-o sets the path (prefix <output>_<stem> with several inputs)'),
    ]
    guide = Guide(
        why="""
            Draws a publication-style reaction energy diagram from energies you have already computed:
            one or more pathways, branches that leave a common trunk, labelled states, highlighted
            transition states and section brackets. Use it when the quick plot of `frog-autoprofile`
            is not enough, e.g. to compare two mechanisms on one figure or to prepare a figure for a
            paper or a talk.
            """,
        needs="""
            A small YAML file with the **relative energies** (usually ΔG in kcal/mol, e.g. from
            `frog-rrho` or `frog-autoprofile`): `steps:` (state names), `pathways:` (each with `name`
            and `energies`, optionally `branch_at`, `color`, `style`), plus options such as `title` and
            `label_points` (examples in the Reaction Profiles user guide).

            The numbers are plotted as given: no unit conversion is done.
            """,
        how=[
            "Write the input file, e.g. `pes.yaml` with `steps: [R, TS1, INT, TS2, P]` and a pathway "
            "`energies: [0.0, 18.4, -3.2, 12.1, -10.5]`.",
            "Run `frog-profile pes.yaml`: the figure is saved as `pes_profile.png` next to the input "
            "(`--out-dir figs` to put it elsewhere, `-o figure1.svg` to choose the file).",
            "Adjust the look: `--style presentation` (bigger fonts) or `--style straight` (straight segments), "
            "`--point-type bar` (horizontal bars instead of hollow dots), `--curviness 0` (straight lines).",
            "Add the TS barrier to the legend with `--legend-barriers`; extra options can come from JSON "
            "files (`--config`, `--annotations`, `--point-labels`, `--secondary`, `--x`).",
            "No input yet? `frog-profile --gui` opens a browser editor that writes the YAML file for you.",
        ],
        expect="""
            Only a figure (nothing is printed on stdout): the state names along the x axis, each pathway
            as a smooth curve through its states (a legend when there are several pathways), and the
            energy values written at the points you select with `label_points`, typically the transition states. By default no y-axis frame is drawn (`axes` option).
            Example: the model input of the user guide shows 18.4 at TS1 and 12.1/15.8 at TS2 for the two paths.

            * Check the printed values against your table: they are the numbers of the input, rounded to
              one decimal.
            * Branches (`branch_at: INT`) start at the named state; a branch that starts at the first
              state means the name was not found among the `steps`.
            * Several inputs with the same stem (`pes.yaml`, `pes.yml`) write the **same**
              `pes_profile.png`: the last one overwrites the others. Use different names or `-o prefix`.
            * `--gui` needs the optional `streamlit` package.
            """,
    )
    examples = [
        ('frog-profile pes.yaml', 'Save pes_profile.png next to pes.yaml'),
        ('frog-profile pes.yaml -o figure1.svg --style presentation', 'SVG with the presentation preset'),
        ('frog-profile path_a.yaml path_b.yaml -o profiles --fig-format pdf --show',
         'profiles_path_a.pdf and profiles_path_b.pdf, then open the plot windows'),
    ]

    def setup_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('file', nargs='*', default=[], help='Input YAML files')
        parser.add_argument('--style', default='default', choices=['default', 'presentation', 'straight'],
                            help='Plot style preset')
        parser.add_argument('--point-type', choices=['hollow', 'bar', 'dot', '.'], default=None,
                            help='Point marker style')
        parser.add_argument('--curviness', type=float, default=None,
                            help='Curve tension (0 = straight lines; preset value if not given)')
        parser.add_argument('--config', help='JSON file of style options')
        parser.add_argument('--secondary', help='JSON file of series for the right-hand y-axis')
        parser.add_argument('--annotations', help='JSON file or JSON string of segment annotations')
        parser.add_argument('--point-labels', help='JSON file or JSON string of per-point labels')
        parser.add_argument('--x', help='JSON file or JSON string of real x coordinates')
        parser.add_argument('--legend-barriers', dest='legend_barriers', action='store_true', default=None,
                            help='Append the TS barrier (kcal/mol) to the legend labels')
        parser.add_argument('--no-legend-barriers', dest='legend_barriers', action='store_false',
                            help='Do not append TS barriers to the legend labels')
        parser.add_argument('--gui', action='store_true', help='Launch the web GUI (Streamlit) in the browser')
        self.add_output_arg(parser, help='Plot path, or prefix with several inputs '
                                         '(extension overrides --fig-format)')
        self.add_out_dir_arg(parser)
        self.add_plot_args(parser, optional=False, style=False)

    def run(self, args: argparse.Namespace) -> None:
        if args.gui:
            import subprocess
            import sys
            gui_path = Path(__file__).resolve().parent.parent / "reaction_plotter" / "web_gui.py"
            try:
                result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])
            except KeyboardInterrupt:
                return
            if result.returncode:
                raise InputError(f"Streamlit exited with code {result.returncode}")
            return

        if not args.file:
            raise InputError("No input files given: pass file path(s), or use --gui")

        files = self.require_files(args.file)
        import matplotlib as mpl
        from frogchem.plotter import STYLE
        from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer

        style_config = _load_optional_json(args.config, "config") or {}
        secondary_data = _load_optional_json(args.secondary, "secondary")
        ann_data = _load_optional_json(args.annotations, "annotations")
        pt_labels = _load_optional_json(args.point_labels, "point-labels")
        x_coords = _load_optional_json(args.x, "x")

        extra_kwargs: dict[str, Any] = dict(style_config)
        if args.point_type is not None:
            extra_kwargs["point_type"] = args.point_type
        if args.curviness is not None:
            extra_kwargs["curviness"] = args.curviness
        if secondary_data is not None:
            extra_kwargs["secondary"] = secondary_data
        if x_coords is not None:
            extra_kwargs["x"] = x_coords
        if args.legend_barriers is not None:
            extra_kwargs["legend_barriers"] = args.legend_barriers
        # Passed only when given: an explicit None would hide the input file's own labels
        if ann_data is not None:
            extra_kwargs["annotations"] = ann_data
        if pt_labels is not None:
            extra_kwargs["point_labels"] = pt_labels

        # The profile presets (--style) are designed on matplotlib's own defaults, not on the
        # shared FROG style: undo the shared rcParams while drawing and saving.
        own_style = {k: mpl.rcParamsOrig[k] for k in STYLE if k in mpl.rcParamsOrig}

        for p in files:
            print_info(f"Processing {p}")
            try:
                analyzer = ReactionProfileAnalyzer(str(p))
            except (FileNotFoundError, ValueError) as e:
                raise InputError(f"{p}: {e}") from e

            default = self.derived_path(p, 'profile', args.fig_format)
            path = self.plot_path(default, tag=p.stem if len(files) > 1 and args.output else None)

            with mpl.rc_context(own_style):
                try:
                    fig, _ax = analyzer.plot(
                        filename=None,
                        style=args.style,
                        **dict(extra_kwargs),
                    )
                except (FileNotFoundError, ValueError) as e:
                    raise InputError(f"{p}: {e}") from e
                self.save_figure(fig, path, **analyzer.savefig_kwargs)


def main():
    ReactionProfile().main()


if __name__ == '__main__':
    main()
