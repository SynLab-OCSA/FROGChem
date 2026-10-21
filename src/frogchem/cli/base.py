"""CLI framework with auto-registration and shared boilerplate.

Usage:
    from frogchem.cli.base import CLICommand, Output

    class MyTool(CLICommand, name='frog-my-tool'):
        \"\"\"One-line summary shown by `frog` and in the docs.\"\"\"
        category = 'geometry'
        outputs = [Output('table', 'Results', 'One row per input file')]
        examples = [('frog-my-tool mol.xyz', 'Analyse one structure')]

        def setup_args(self, parser):
            parser.add_argument('file', help='Input file')

        def run(self, args):
            self.output(rows, headers, title='Results')

Output, file and plot conventions are described in docs/cli/conventions.md.
"""
import argparse
import contextlib
import inspect
import json
import logging
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from frogchem.exceptions import ParseError, InputError
from frogchem.printer import print_error, print_info, print_title, print_saved, print_table, \
    set_quiet, TableResult, tables_to_json, tables_to_csv, write_tables

logger = logging.getLogger(__name__)

# ponytail: structured exit codes, not just 0/1
EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_PARSE_ERROR = 2

_REGISTRY: dict[str, type['CLICommand']] = {}

#: category → (docs page title, intro). Docs pages: docs/cli/<category>_tools.md
CATEGORIES = {
    'parsing': ('Quantum Parsing & Property',
                'These commands extract electronic properties, wavefunctions, and geometries '
                'from ORCA and Gaussian output files.'),
    'thermo': ('Thermochemistry & Energetics',
               'These commands evaluate thermochemical corrections, selectivities, and reaction pathways.'),
    'geometry': ('Conformer & Geometry',
                 'These commands manipulate 3D molecular structures, detect symmetry, measure internal '
                 'coordinates, sort conformational ensembles, and analyse trajectories.'),
    'nmr': ('NMR Spectroscopy',
            'These commands extract isotropic magnetic shieldings, calibrate chemical shifts, determine '
            'equivalence classes, and position ghost atoms for aromaticity studies.'),
    'spectroscopy': ('Spectroscopy & Grid',
                     'These commands process electronic excited states, UV-Vis/ECD spectra, volumetric '
                     'grids, and real-space densities.'),
    'substituter': ('Fragment Substitution',
                    'These commands automate 3D substituent modification and manage the fragment library.'),
}

OUTPUT_KINDS = ('table', 'file', 'plot')


@dataclass(frozen=True)
class Output:
    """One documented output of a command.

    kind: 'table' (stdout), 'file' or 'plot'.
    name: table title or file name pattern, e.g. '<stem>_UV.png'.
    description: what it contains.
    when: flag that produces it, if not always produced (e.g. '-w', '--plot').
    """
    kind: str
    name: str
    description: str
    when: Optional[str] = None

    def __post_init__(self):
        if self.kind not in OUTPUT_KINDS:
            raise ValueError(f"Output kind must be one of {OUTPUT_KINDS}, got {self.kind!r}")


@dataclass(frozen=True)
class Guide:
    """Plain-language user guide of a command, rendered in the docs (generate_readme.py).

    Written for users who run quantum-chemistry calculations only occasionally.
    Every field is Markdown.

    why:    what the command is for, which question it answers and when in a workflow to use it.
    needs:  what you must have before running it (files, which calculation/keywords produced them).
    how:    ordered steps to go from those files to a result (each step one short sentence;
            commands in backticks).
    expect: what the output looks like and how to read it: units, typical values, what is
            normal and what signals a problem, common warnings and pitfalls.
    """
    why: str
    needs: str
    how: Sequence[str]
    expect: str

    def __post_init__(self):
        for field_name in ('why', 'needs', 'expect'):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Guide.{field_name} must not be empty")
        if not self.how:
            raise ValueError("Guide.how must contain at least one step")


#: Shared declaration of the weights table printed by every ensemble command with -c.
ENSEMBLE_WEIGHTS = Output('table', 'Ensemble Weights (<n> conformers)',
                          'Conformer, protocol, file, Boltzmann population (%) and normalized weight (%)',
                          when='-c')


class HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keeps description/epilog layout and appends '(default: X)' for meaningful defaults."""

    def __init__(self, prog, **kwargs):
        kwargs.setdefault('max_help_position', 32)
        kwargs.setdefault('width', help_width())
        super().__init__(prog, **kwargs)

    def _get_help_string(self, action):
        help_text = action.help or ''
        if '%(default)' in help_text or 'default:' in help_text:
            return help_text
        default = action.default
        if default is None or default is False or default is argparse.SUPPRESS or default == [] or default == '':
            return help_text
        if not action.option_strings and action.nargs not in (argparse.OPTIONAL, argparse.ZERO_OR_MORE):
            return help_text
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._HelpAction)):
            return help_text
        return f"{help_text} (default: %(default)s)"


class CLICommand:
    """Base class for CLI tools. Subclasses auto-register via __init_subclass__."""

    description: str = ''
    name: str = ''
    category: str = ''
    #: Adds -c/--checkpoint, --read-boltzmann, -e, -T, --individual and Boltzmann averaging.
    supports_ensemble: bool = False
    #: Adds --format and --table-file. Set False for commands that print no tables.
    has_tables: bool = True
    #: Append an ensemble-average row to every multi-row table (only with supports_ensemble).
    _auto_ensemble_average: bool = True
    outputs: Sequence[Output] = ()
    examples: Sequence[tuple[str, str]] = ()
    #: Plain-language guide (why / what you need / how / what to expect) shown in the docs.
    guide: Optional[Guide] = None
    _plot_mode: Optional[str] = None

    def __init_subclass__(cls, name: str = '', **kwargs):
        super().__init_subclass__(**kwargs)
        if name:
            cls.name = name
            _REGISTRY[name] = cls

    # ------------------------------------------------------------------ hooks
    def setup_args(self, parser: argparse.ArgumentParser) -> None:
        """Override to add arguments to the parser."""

    def run(self, args: argparse.Namespace) -> None:
        """Override with the tool's logic."""
        raise NotImplementedError

    # ----------------------------------------------------------------- parser
    @classmethod
    def summary(cls) -> str:
        """First line of the description: used by `frog` and the docs."""
        text = (cls.description or cls.__doc__ or '').strip()
        return text.splitlines()[0].strip() if text else ''

    def build_parser(self) -> argparse.ArgumentParser:
        self._plot_mode = None
        parser = argparse.ArgumentParser(
            prog=self.name,
            description=reflow(inspect.cleandoc(self.description or self.__doc__ or ''), help_width()),
            formatter_class=HelpFormatter,
        )
        parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug output')
        parser.add_argument('-q', '--quiet', action='store_true',
                            help='Suppress banner, info and saved-file messages')
        if self.has_tables:
            parser.add_argument('--format', choices=['table', 'json', 'csv'], default='table',
                                help='Format of the tables printed on stdout')
            parser.add_argument('--table-file', metavar='PATH', default=None,
                                help='Also write all tables to PATH (.json, .csv or .txt; '
                                     'otherwise the --format is used)')
        if self.supports_ensemble:
            self._add_ensemble_args(parser)
        self.setup_args(parser)
        self._organize_argument_groups(parser)
        parser.epilog = self.render_epilog(parser)
        return parser

    @staticmethod
    def _add_ensemble_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument('-c', '--checkpoint', metavar='PATH', default=None,
                            help='Checkpoint file (checkpoint.json, or csv/json) with Boltzmann populations for ensemble averaging')
        parser.add_argument('--read-boltzmann', dest='read_boltzmann', metavar='PROTOCOL', default=None,
                            help='Protocol whose populations are read from the checkpoint '
                                 '(default: protocol of each file, or the latest)')
        parser.add_argument('-e', '--energy-term', choices=['E', 'EZPVE', 'H', 'G'], default='G',
                            help='Energy used for on-the-fly Boltzmann weights')
        parser.add_argument('-T', '--temp', type=float, default=298.15,
                            help='Temperature in K')
        parser.add_argument('--individual', action='store_true',
                            help='Also show per-conformer results next to the ensemble average')

    # ---------------------------------------------------- shared arg helpers
    @staticmethod
    def add_output_arg(parser, default=None, help='Primary output path', metavar='PATH'):
        """-o/--output: the primary output path (file, or prefix for several files)."""
        return parser.add_argument('-o', '--output', metavar=metavar, default=default, help=help)

    @staticmethod
    def add_out_dir_arg(parser):
        """--out-dir: where per-input files go (default: next to each input)."""
        return parser.add_argument('--out-dir', metavar='DIR', default=None,
                                   help='Directory for per-input output files (default: next to each input file)')

    @staticmethod
    def add_write_arg(parser, help='Write derived structure files'):
        """-w/--write: write derived structure files."""
        return parser.add_argument('-w', '--write', action='store_true', help=help)

    @staticmethod
    def add_force_arg(parser, help='Overwrite existing output files'):
        return parser.add_argument('-f', '--force', action='store_true', help=help)

    @staticmethod
    def add_grid_args(parser, spacing: float, padding: float) -> None:
        """Cube grid options shared by every command that writes .cube files."""
        parser.add_argument('--spacing', type=float, default=spacing, help='Grid spacing in Å')
        parser.add_argument('--voxels', type=int, default=None,
                            help='Number of grid points per axis (overrides --spacing)')
        parser.add_argument('--padding', type=float, default=padding,
                            help='Box padding around the molecule in Å')

    @staticmethod
    def add_interactive_arg(parser, help='Run in interactive mode'):
        return parser.add_argument('--interactive', action='store_true', help=help)

    def add_plot_args(self, parser, optional: bool = False, default_name: Optional[str] = None,
                      style: bool = True, cmap: Union[bool, str, None] = None):
        """Shared plot options.

        optional=False: the plot is the main product; its path comes from -o/--output
                        (call add_output_arg yourself if you need a custom help/default).
        optional=True:  --plot [PATH] enables the plot.
        style=True:     also the shared look options (--title, --colors, --linewidth, ...),
                        read back with self.plot_options(**command_defaults).
        cmap:           add --cmap for commands that draw maps; a string is its default.
        """
        from frogchem.plotter import DEFAULT_DPI, FIG_FORMATS, LINESTYLES
        if optional:
            hint = f" (default: {default_name})" if default_name else ''
            parser.add_argument('--plot', nargs='?', const='', default=None, metavar='PATH',
                                help=f'Save the plot; PATH is optional{hint}')
            self._plot_mode = 'plot'
        else:
            self._plot_mode = 'output'
        parser.add_argument('--fig-format', choices=list(FIG_FORMATS), default='png',
                            help='Image format, unless the path already has an extension')
        parser.add_argument('--dpi', type=int, default=DEFAULT_DPI, help='Raster resolution of saved plots')
        parser.add_argument('--show', action='store_true', help='Also open the plot in an interactive window')
        if style:
            parser.add_argument('--title', default=None, metavar='TEXT', help="Plot title ('' removes it)")
            parser.add_argument('--xlabel', default=None, metavar='TEXT', help='Label of the x axis')
            parser.add_argument('--ylabel', default=None, metavar='TEXT', help='Label of the y axis')
            parser.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'),
                                help='Range of the x axis')
            parser.add_argument('--ylim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'),
                                help='Range of the y axis')
            parser.add_argument('--colors', nargs='+', default=None, metavar='COLOR',
                                help='Colours of the plotted series, in order (matplotlib names or #hex)')
            parser.add_argument('--linewidth', type=float, default=None, metavar='W', help='Line width in points')
            parser.add_argument('--linestyle', choices=list(LINESTYLES), default=None, help='Line style')
            parser.add_argument('--legend', action=argparse.BooleanOptionalAction, default=None,
                                help='Show or hide the legend (default: decided by the plot)')
            parser.add_argument('--grid', action=argparse.BooleanOptionalAction, default=None,
                                help='Show or hide the grid (default: decided by the plot)')
            parser.add_argument('--figsize', nargs=2, type=float, default=None, metavar=('W', 'H'),
                                help='Figure size in inches')
            parser.add_argument('--font-size', type=float, default=None, metavar='PT',
                                help='Base font size in points (titles and labels scale with it)')
        if cmap is not None and cmap is not False:
            parser.add_argument('--cmap', default=cmap if isinstance(cmap, str) else None, metavar='NAME',
                                help='Colormap of maps and heatmaps (matplotlib or seaborn name)')

    def plot_options(self, **defaults):
        """The plot options of this run: CLI flags, then the command's `defaults`, then STYLE."""
        from frogchem.plotter import PlotOptions
        opts = getattr(self, 'plot_opts', None) or PlotOptions()
        return opts.with_defaults(**defaults)

    # ----------------------------------------------------------- help layout
    _PLOT_DESTS = {'plot', 'fig_format', 'dpi', 'show', 'title', 'xlabel', 'ylabel', 'xlim', 'ylim',
                   'colors', 'cmap', 'linewidth', 'linestyle', 'legend', 'grid', 'figsize', 'font_size'}
    _GROUPS = (
        ('thresholds', None),
        ('plot options', _PLOT_DESTS),
        ('ensemble averaging', {'checkpoint', 'read_boltzmann', 'energy_term', 'temp', 'individual'}),
        ('output', {'output', 'out_dir', 'write', 'force', 'format', 'table_file', 'verbose', 'quiet'}),
    )
    _THR_KEYS = ('threshold', 'thresh', 'tolerance', 'tol', 'cutoff', 'cut_off')

    def _organize_argument_groups(self, parser: argparse.ArgumentParser) -> None:
        """Move shared options out of the default group: thresholds, plot, ensemble, output."""
        default_opt = getattr(parser, '_optionals', None)
        if default_opt is None or not hasattr(default_opt, '_group_actions'):
            return
        buckets = {title: [] for title, _ in self._GROUPS}
        for action in list(default_opt._group_actions):
            if action.dest == 'help':
                continue
            target = None
            for title, dests in self._GROUPS:
                if dests is None:
                    names = [action.dest.lower()] + [s.lower() for s in action.option_strings]
                    if any(k in n for n in names for k in self._THR_KEYS):
                        target = title
                elif action.dest in dests:
                    if title == 'ensemble averaging' and not self.supports_ensemble:
                        continue
                    target = title
            if target:
                default_opt._group_actions.remove(action)
                buckets[target].append(action)
        existing = {g.title: g for g in parser._action_groups}
        for title, _ in self._GROUPS:
            if buckets[title]:
                g = existing.get(title) or parser.add_argument_group(title)
                g._group_actions.extend(buckets[title])

    def render_epilog(self, parser: Optional[argparse.ArgumentParser] = None) -> str:
        """Build the 'outputs:' and 'examples:' sections of --help from the declarations."""
        opts = set(parser._option_string_actions) if parser else set()
        lines = []
        sections = [
            ('table', 'terminal (stdout; --format table|json|csv, --table-file PATH)'
                      if self.has_tables else 'terminal (stdout)'),
            ('file', 'files (next to each input, or --out-dir)' if '--out-dir' in opts else 'files'),
            ('plot', 'plots (saved, not shown; --fig-format, --dpi, --show)'),
        ]
        if self.outputs:
            lines.append('outputs:')
            for kind, header in sections:
                items = [o for o in self.outputs if o.kind == kind]
                if not items:
                    continue
                lines.append(f'  {header}:')
                for o in items:
                    desc = f"[{o.when}] {o.description}" if o.when else o.description
                    lines.extend(_two_columns(o.name, desc, indent=4, col=32, width=help_width()))
        if self.examples:
            if lines:
                lines.append('')
            lines.append('examples:')
            for cmd, desc in self.examples:
                lines.append(f'  {cmd}')
                if desc:
                    lines.extend(textwrap.wrap(desc, help_width() - 6, initial_indent=' ' * 6,
                                               subsequent_indent=' ' * 6))
        if lines:
            lines.append('')
        notes = []
        if self.supports_ensemble and self._auto_ensemble_average:
            notes.append('With -c, multi-row tables get a Boltzmann-weighted "⟨Ensemble Average⟩" row.')
        notes.append('Data go to stdout, messages to stderr. Atom indices are 0-based unless stated otherwise.')
        if self.guide is not None and self.category:
            notes.append(f'Step-by-step guide (what it is for, what you need, what to expect): '
                         f'docs/cli/{self.name}.md.')
        for note in notes:
            lines.extend(textwrap.wrap(note, help_width()))
        return '\n'.join(lines)

    # ----------------------------------------------------------- paths/plots
    def derived_path(self, input_path: Union[str, Path], tag: str, ext: Optional[str] = None) -> Path:
        """Per-input output: '<out-dir or input dir>/<stem>_<tag><ext>' (ext defaults to the input's).

        Compression suffixes are ignored ('a.xyz.gz' → stem 'a', ext '.xyz'). When another input
        of the same run already owns that path (same stem from another folder, e.g.
        conf_1/orca.out and conf_2/orca.out with --out-dir), the parent folder names are
        prepended until the name is unique, so outputs never overwrite each other.
        """
        src = Path(input_path)
        stem, suffix = split_name(src)
        if ext is None:
            ext = suffix
        elif ext and not ext.startswith('.'):
            ext = f'.{ext}'
        out_dir = getattr(getattr(self, 'args', None), 'out_dir', None)
        folder = Path(out_dir) if out_dir else src.parent
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{stem}_{tag}{ext}" if tag else f"{stem}{ext}"
        src_key = src.resolve()
        owners = self.__dict__.setdefault('_derived_owners', {})
        path = folder / name
        parents = [part for part in src_key.parent.parts if part not in ('/', '\\')]
        depth = 0
        while owners.get(path.resolve(), src_key) != src_key:
            depth += 1
            prefix = '_'.join(parents[-depth:]) if depth <= len(parents) else f"{depth}"
            path = folder / f"{prefix}_{name}"
        if depth:
            print_info(f"{src} shares its name with another input: writing {path.name}")
        if path.resolve() == src_key:
            raise InputError(f"Refusing to overwrite the input file {src}")
        owners[path.resolve()] = src_key
        return path

    def output_path(self, default: Union[str, Path], ext: str, tag: Optional[str] = None,
                    known_exts=None) -> Path:
        """Primary output from -o/--output (or `default`), with `{stem}_{tag}` prefix semantics."""
        from frogchem.plotter import resolve_path
        user = getattr(getattr(self, 'args', None), 'output', None)
        return resolve_path(user, default, ext, tag, known_exts=known_exts or [ext])

    @property
    def plot_enabled(self) -> bool:
        if self._plot_mode == 'plot':
            return getattr(self.args, 'plot', None) is not None
        return self._plot_mode == 'output'

    def plot_path(self, default: Union[str, Path], tag: Optional[str] = None) -> Path:
        """Plot path from --plot PATH or -o/--output (or `default`), extension from --fig-format."""
        from frogchem.plotter import resolve_path
        dest = 'plot' if self._plot_mode == 'plot' else 'output'
        user = getattr(self.args, dest, None) or None
        return resolve_path(user, default, getattr(self.args, 'fig_format', 'png'), tag)

    def save_figure(self, fig, path: Union[str, Path], kind: str = 'plot', **savefig_kwargs) -> Path:
        """Save with the shared --dpi/--show options and report the path."""
        from frogchem.plotter import save_figure
        return save_figure(fig, path, dpi=getattr(self.args, 'dpi', 300),
                           show=getattr(self.args, 'show', False), kind=kind, defer_show=True,
                           **savefig_kwargs)

    @staticmethod
    def saved(path: Union[str, Path], kind: str = 'file') -> None:
        """Report a written file on stderr."""
        print_saved(path, kind)

    # --------------------------------------------------------------- ensemble
    def load_populations(self) -> dict[int, float]:
        """Load Boltzmann populations from the checkpoint specified in --checkpoint."""
        ckpt = getattr(self, '_checkpoint', None)
        if not ckpt:
            return {}
        from frogchem.ensemble import load_checkpoint_populations
        return load_checkpoint_populations(ckpt, protocol=getattr(self, '_read_boltzmann', None))

    def build_ensemble(self, items, **kwargs):
        """frogchem.ensemble.build_ensemble with the command's checkpoint, protocol, -e and -T."""
        from frogchem.ensemble import build_ensemble
        kwargs.setdefault('protocol', getattr(self, '_read_boltzmann', None))
        kwargs.setdefault('energy_term', getattr(self, '_energy_term', 'G'))
        kwargs.setdefault('temp', getattr(self, '_temp', 298.15))
        return build_ensemble(items, self._checkpoint, **kwargs)

    def get_ensemble_weights(self, file_labels: Sequence[str]) -> Optional[dict[str, float]]:
        """Compute normalized Boltzmann weights for a sequence of file labels."""
        if not getattr(self, '_checkpoint', None):
            return None
        ens = self.build_ensemble(file_labels)
        return {e.file_path: e.weight for e in ens.entries}

    def output_weights(self, ens) -> None:
        """The 'Ensemble Weights' table, identical for every ensemble command."""
        self._ensemble_weights_printed = True
        self.output(ens.summary_rows, ens.summary_headers,
                    title=f"Ensemble Weights ({len(ens)} conformers)", floatfmt='.2f', average=False)

    def _apply_ensemble_average(self, data, headers, title=""):
        """Automatically compute and append Boltzmann-weighted average row to data tables."""
        if not getattr(self, '_checkpoint', None):
            return data
        ens = self.build_ensemble(data, path_extractor=lambda r: r[0] if r else "")
        if not ens or len(ens) < 2:
            return data

        weights = ens.weights

        if not getattr(self, '_ensemble_weights_printed', False):
            self.output_weights(ens)

        avg_row = ['⟨Ensemble Average⟩']
        for col_idx in range(1, len(headers)):
            col_vals = []
            is_numeric = True
            for r in data:
                if col_idx >= len(r):
                    is_numeric = False
                    break
                v = r[col_idx]
                try:
                    col_vals.append(float(str(v).strip()))
                except (ValueError, TypeError):
                    is_numeric = False
                    col_vals.append(v)

            if is_numeric and col_vals:
                avg_row.append(sum(w * v for w, v in zip(weights, col_vals)))
            else:
                unique_str = set(str(v).strip() for v in col_vals if v is not None)
                avg_row.append(next(iter(unique_str)) if len(unique_str) == 1 else '---')

        return list(data) + [avg_row]

    # ----------------------------------------------------------------- tables
    def output(self, data, headers, title="", floatfmt=".3f", average: bool = True):
        """Emit a result table: printed now in table mode, collected for json/csv and --table-file.

        floatfmt: one format for all columns (".3f") or one per column ([".1f", ".4f", ...]).
        """
        if (average and self.supports_ensemble and getattr(self, '_checkpoint', None) and
                self._auto_ensemble_average and
                data and isinstance(data, list) and isinstance(data[0], (list, tuple)) and len(data) > 1):
            data = self._apply_ensemble_average(data, headers, title)

        if not hasattr(self, '_tables'):
            self._tables = []
        self._tables.append(TableResult(data, headers, title, floatfmt))
        if getattr(self, '_output_format', 'table') == 'table':
            print_table(data, headers=headers, title=title, floatfmt=floatfmt)

    def _flush_tables(self) -> None:
        tables = [t for t in getattr(self, '_tables', []) if t.data]
        fmt = getattr(self, '_output_format', 'table')
        if fmt == 'json':
            # ponytail: stdout is always one valid JSON document, even when nothing was found
            print(tables_to_json(tables))
        elif tables and fmt == 'csv':
            print(tables_to_csv(tables), end='')
        table_file = getattr(self, '_table_file', None)
        if tables and table_file:
            print_saved(write_tables(tables, table_file, fmt), 'tables')
        self._tables = []

    # ------------------------------------------------------------------- main
    def main(self, argv: Sequence[str] | None = None) -> None:
        args = self.build_parser().parse_args(argv)
        self.args = args
        self._output_format = getattr(args, 'format', 'table') or 'table'
        self._table_file = getattr(args, 'table_file', None)
        self._checkpoint = getattr(args, 'checkpoint', None)
        self._read_boltzmann = getattr(args, 'read_boltzmann', None)
        self._energy_term = getattr(args, 'energy_term', 'G')
        self._temp = getattr(args, 'temp', 298.15)
        self._individual = getattr(args, 'individual', False)
        self._ensemble_weights_printed = False
        self._tables = []

        self._derived_owners = {}
        # printer no longer infers quiet mode from the logging level: set it explicitly from -q
        set_quiet(args.quiet)
        _configure_logging(logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO)

        plot_style = contextlib.nullcontext()
        if self._plot_mode:
            from frogchem import plotter
            self.plot_opts = plotter.PlotOptions.from_args(args)
            # ponytail: the style is a context around run(), so matplotlib's global rcParams
            # are restored afterwards (library callers and tests keep their own settings)
            plot_style = plotter.style(self.plot_opts)

        self._parse_errors = []
        print_title(self.name)
        exit_code = EXIT_SUCCESS
        try:
            with plot_style:
                self.run(args)
        except KeyboardInterrupt:
            exit_code = 130
        except SystemExit as e:              # print_error(..., exit_code=N) / require_files
            if isinstance(e.code, str):
                print_error(e.code)
            exit_code = e.code if isinstance(e.code, int) else (EXIT_SUCCESS if e.code is None else EXIT_INPUT_ERROR)
        except InputError as e:
            logger.debug('Input error', exc_info=True)
            print_error(str(e))
            exit_code = EXIT_INPUT_ERROR
        except ParseError as e:
            logger.debug('Parse error', exc_info=True)
            print_error(str(e))
            exit_code = EXIT_PARSE_ERROR
        except Exception as e:
            logger.debug('Unhandled exception', exc_info=True)
            print_error(f"{type(e).__name__}: {e}")
            exit_code = EXIT_INPUT_ERROR

        # Tables collected so far are emitted even after an error (partial results stay usable)
        try:
            self._flush_tables()
        except (OSError, ValueError) as e:
            logger.debug('Table output failed', exc_info=True)
            print_error(f"Cannot write tables to {self._table_file}: {e}")
            exit_code = exit_code or EXIT_INPUT_ERROR

        if exit_code == EXIT_SUCCESS and self._plot_mode and getattr(args, 'show', False):
            from frogchem.plotter import show_figures
            show_figures()

        if self._parse_errors:
            print(json.dumps(self._parse_errors, indent=2), file=sys.stderr)
        if exit_code:
            sys.exit(exit_code)

    @staticmethod
    def require_files(files: Sequence[str]) -> list[Path]:
        """Validate that all files exist. Returns list of Path objects.

        Raises SystemExit with EXIT_INPUT_ERROR if any file is missing.
        """
        paths = []
        for f in files:
            p = Path(f)
            if not p.exists():
                print_error(f"File not found: {f}", exit_code=EXIT_INPUT_ERROR)
            paths.append(p)
        return paths



COMPRESSION_SUFFIXES = ('.gz', '.bz2', '.xz')


def split_name(path: Union[str, Path]) -> tuple[str, str]:
    """(stem, suffix) of a file, ignoring a compression suffix: 'a.cube.gz' → ('a', '.cube')."""
    name = Path(path).name
    for comp in COMPRESSION_SUFFIXES:
        if name.lower().endswith(comp):
            name = name[:-len(comp)]
            break
    p = Path(name)
    return p.stem, p.suffix


def reflow(text: str, width: int) -> str:
    """Re-wrap plain paragraphs of a description to `width`.

    Paragraphs with indented lines or list items (-, *, •, 1.) are kept as written.
    """
    out = []
    for para in text.split('\n\n'):
        rows = para.split('\n')
        structured = any(r[:1].isspace() or r.lstrip()[:2] in ('- ', '* ', '• ')
                         or r.lstrip()[:1].isdigit() and r.lstrip()[1:3].startswith('.')
                         for r in rows)
        out.append(para if structured else textwrap.fill(' '.join(r.strip() for r in rows), width))
    return '\n\n'.join(out)


def help_width() -> int:
    """Help text width: the terminal width, capped at 100 columns (README/docs use 100)."""
    return max(60, min(shutil.get_terminal_size((100, 24)).columns - 2, 100))


def _configure_logging(level: int) -> None:
    """Set FROG's log level without clobbering a host application's logging setup.

    A plain CLI process has no root handlers: add a stderr one. Third-party libraries stay
    at WARNING, so -v shows FROG debug messages only.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(format='%(name)s: %(message)s')
    logging.getLogger('frogchem').setLevel(level)


def _two_columns(left: str, right: str, indent: int, col: int, width: int = 100) -> list[str]:
    pad = ' ' * indent
    wrapped = textwrap.wrap(right, width - col) or ['']
    if len(left) + indent + 2 > col:
        return [f"{pad}{left}"] + [' ' * col + w for w in wrapped]
    return [f"{pad}{left}".ljust(col) + wrapped[0]] + [' ' * col + w for w in wrapped[1:]]


def get_registry() -> dict[str, type[CLICommand]]:
    """Return the tool registry (name → class)."""
    return dict(_REGISTRY)


def load_all_commands() -> dict[str, type[CLICommand]]:
    """Import every CLI module so the registry is complete."""
    import importlib
    for module in _CLI_MODULES.values():
        importlib.import_module(module)
    return get_registry()


def dispatch(name: Optional[str] = None) -> None:
    """Dispatch to a registered tool by name, or show available tools."""
    if name is None or name not in _REGISTRY:
        registry = load_all_commands()
        width = max(len(n) for n in registry)
        listing = '\n'.join(f'  {n.ljust(width)}  {cls.summary()}' for n, cls in sorted(registry.items()))
        print(f'Available tools:\n{listing}\n\nUsage: frog <tool> [args]   (frog <tool> --help for details)')
        sys.exit(1 if name else 0)
    _REGISTRY[name]().main(sys.argv[2:])


# ponytail: lazy-load CLI modules on demand to avoid cold-start penalty
_CLI_MODULES = {
    'frog-rmsd': 'frogchem.cli.calc_rmsd',
    'frog-rrho': 'frogchem.cli.calc_rrho',
    'frog-crest': 'frogchem.cli.create_crest_input',
    'frog-cube': 'frogchem.cli.cube_manipulation',
    'frog-fragments': 'frogchem.cli.fragment_detection',
    'frog-hessian': 'frogchem.cli.hessian_localizer',
    'frog-internals': 'frogchem.cli.internals',
    'frog-mirror': 'frogchem.cli.mirror_ensemble',
    'frog-profile': 'frogchem.cli.reaction_profile',
    'frog-sort': 'frogchem.cli.sort_diastero',
    'frog-sort-ez': 'frogchem.cli.sort_ez',
    'frog-split': 'frogchem.cli.split_xyz',
    'frog-tddft': 'frogchem.cli.tddft_parser',
    'frog-sym': 'frogchem.cli.detect_symmetry',
    'frog-mo': 'frogchem.cli.mo_coeff',
    'frog-nics': 'frogchem.cli.nics_geometry',
    'frog-xyz': 'frogchem.cli.write_xyz',
    'frog-spin': 'frogchem.cli.calc_loc_spin',
    'frog-charges': 'frogchem.cli.get_charges',
    'frog-nbo': 'frogchem.cli.nbo_parser',
    'frog-substitute': 'frogchem.cli.substitute_fragment',
    'frog-sub-db': 'frogchem.cli.manage_sub_db',
    'frog-cie': 'frogchem.cli.cie_plot',
    'frog-md': 'frogchem.cli.md_analysis',
    'frog-er': 'frogchem.cli.calc_er',
    'frog-plot-neb': 'frogchem.cli.plot_neb',
    'frog-stereocenters': 'frogchem.cli.stereocenters',
    'frog-density': 'frogchem.cli.electronic_density',
    'frog-solvent': 'frogchem.cli.solvent_surface',
    'frog-nmr': 'frogchem.cli.calc_nmr',
    'frog-nmr-equiv': 'frogchem.cli.nmr_equiv',
    'frog-cdft': 'frogchem.cli.calc_cdft',
    'frog-si': 'frogchem.cli.generate_si',
    'frog-steric': 'frogchem.cli.calc_steric',
    'frog-autoprofile': 'frogchem.cli.auto_profile',
}


def main():
    """Entry point for the `frog` umbrella command."""
    import importlib
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name in ('-h', '--help'):
        name = None
    if name and not name.startswith('frog-'):
        name = f'frog-{name}'
    if name and name in _CLI_MODULES and name not in _REGISTRY:
        importlib.import_module(_CLI_MODULES[name])
    dispatch(name)
