#!/usr/bin/env python3
"""Plot reaction profiles from formatted input files or JSON.

Generates publication-quality 2D reaction coordinate diagrams with Bezier curves,
customizable markers, and multi-step annotations. Supports both formatted TXT files
and JSON specifications.
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
    """Plot reaction energy profiles from TXT, YAML or JSON input files.

    Draws Bezier-curve reaction coordinate diagrams with point markers, energy labels,
    segment annotations, an optional secondary y-axis and section brackets. One figure
    is saved per input file; the look is set by --style and the input's own options.
    """
    category = 'thermo'
    has_tables = False
    outputs = [
        Output('plot', '<stem>_profile.png', 'Reaction profile, next to the input (or --out-dir); '
                                             '-o sets the path (prefix <output>_<stem> with several inputs)'),
        Output('plot', '<directory>/Rxn_profile.png', 'Legacy TXT inputs with "save: true" (unless -o is given)',
               when='save: true'),
        Output('file', '<directory>/input.txt', 'Copy of the legacy TXT input', when='save: true'),
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
            A small text file with the **relative energies** (usually ΔG in kcal/mol, e.g. from
            `frog-rrho` or `frog-autoprofile`) in one of three formats (examples in the Reaction
            Profiles user guide):

            * **YAML** (`.yaml`, recommended): `steps:` (state names), `pathways:` (each with `name` and
              `energies`, optionally `branch_at`, `color`, `style`), plus options such as `title` and
              `label_points`.
            * **TXT**: sections `--# PES` (one line of `;`-separated energies per pathway), `--# LABELS`
              and `--# GRAPHS` (`key : value` options).
            * **JSON**: `{"Path A": [0.0, 18.4, -3.2]}` or `{"pathways": {...}}`.

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
            "No input yet? `frog-profile --interactive` asks for the data in the terminal and writes the input file for you.",
        ],
        expect="""
            Only a figure (nothing is printed on stdout): the state names along the x axis, each pathway
            as a smooth curve through its states (a legend when there are several pathways), and the
            energy values written at the points you select with `label_points` (YAML) or `labelpoint`
            (TXT), typically the transition states. By default no y-axis frame is drawn (`axes` option).
            Example: the model input of the user guide shows 18.4 at TS1 and 12.1/15.8 at TS2 for the two paths.

            * Check the printed values against your table: they are the numbers of the input, rounded to
              one decimal.
            * Branches (`branch_at: INT` in YAML, `@INT:` in TXT) start at the named state; a branch that
              starts at the first state means the name was not found among the `steps`/labels.
            * Several inputs with the same stem (`pes.yaml`, `pes.txt`) write the **same**
              `pes_profile.png`: the last one overwrites the others. Use different names or `-o prefix`.
            * Legacy TXT files with `save : true` in `--# GRAPHS` are saved as `<directory>/Rxn_profile.png`
              (directory from the `directory` option, default `reaction`) together with a copy `input.txt`,
              unless `-o` is given.
            * `--gui` needs the optional `streamlit` package.
            """,
    )
    examples = [
        ('frog-profile pes.yaml', 'Save pes_profile.png next to pes.yaml'),
        ('frog-profile pes.txt -o figure1.svg --style presentation', 'SVG with the presentation preset'),
        ('frog-profile path_a.json path_b.json -o profiles --fig-format pdf --show',
         'profiles_path_a.pdf and profiles_path_b.pdf, then open the plot windows'),
    ]

    def setup_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument('file', nargs='*', default=[], help='Input files (.txt, .yaml or .json)')
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
        self.add_interactive_arg(parser, help='Launch the terminal wizard to build an input file')
        parser.add_argument('--gui', action='store_true', help='Launch the web GUI (Streamlit) in the browser')
        self.add_output_arg(parser, help='Plot path, or prefix with several inputs '
                                         '(extension overrides --fig-format)')
        self.add_out_dir_arg(parser)
        self.add_plot_args(parser, optional=False, style=False)

    def run(self, args: argparse.Namespace) -> None:
        if args.interactive:
            from frogchem.reaction_plotter.wizard import run_tui_wizard
            run_tui_wizard()
            return

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
            raise InputError("No input files given: pass file path(s), or use --interactive / --gui")

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
                analyzer = ReactionProfileAnalyzer(str(p), auto_plot=False)
            except (FileNotFoundError, ValueError) as e:
                raise InputError(f"{p}: {e}") from e

            legacy_dir = analyzer.config.save and not args.output
            if legacy_dir:
                default = Path(analyzer.config.directory) / 'Rxn_profile'
            else:
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
                if legacy_dir and analyzer.file_content:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    copy = path.parent / 'input.txt'
                    copy.write_text(analyzer.file_content, encoding='utf-8')
                    self.saved(copy, 'input copy')
                self.save_figure(fig, path, **analyzer.savefig_kwargs)


# backward compat
def parse_args():
    return ReactionProfile().build_parser().parse_args()


def main():
    ReactionProfile().main()


if __name__ == '__main__':
    main()
