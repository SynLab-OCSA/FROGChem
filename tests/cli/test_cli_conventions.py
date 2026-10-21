"""Every registered command follows docs/cli/conventions.md."""
import argparse
import json
import re
from pathlib import Path

import pytest

import frogchem
from frogchem.cli.base import CATEGORIES, CLICommand, Output, load_all_commands
from frogchem.printer import TableResult, tables_to_csv, tables_to_json

REGISTRY = load_all_commands()
SRC = Path(frogchem.__file__).parent

RESERVED_SHORT = {
    '-o': {'output'},
    '-i': {'indices'},
    '-t': {'threshold'},
    '-w': {'write'},
    '-f': {'force'},
    '-T': {'temp'},
    '-c': {'checkpoint'},
    '-e': {'energy_term'},
    '-v': {'verbose'},
    '-q': {'quiet'},
}


def _parsers(parser):
    yield parser
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                yield from _parsers(sub)


@pytest.fixture(params=sorted(REGISTRY), scope='module')
def command(request):
    cls = REGISTRY[request.param]
    tool = cls()
    return cls, tool, tool.build_parser()


def test_metadata(command):
    cls, _, _ = command
    assert cls.category in CATEGORIES, f"{cls.name}: category {cls.category!r}"
    assert cls.summary(), f"{cls.name}: missing description"
    assert cls.outputs, f"{cls.name}: no outputs declared"
    assert all(isinstance(o, Output) for o in cls.outputs)
    assert cls.examples, f"{cls.name}: no examples declared"
    for cmd, _desc in cls.examples:
        assert cmd.split()[0] == cls.name, f"{cls.name}: example {cmd!r} does not start with the command"


def test_help_documents_outputs(command):
    cls, _, parser = command
    text = parser.format_help()
    assert 'outputs:' in text and 'examples:' in text
    if any(o.kind == 'plot' for o in cls.outputs):
        assert '--show' in parser._option_string_actions, f"{cls.name}: plots without shared plot options"


SHARED_PLOT_FLAGS = ('--title', '--xlabel', '--ylabel', '--xlim', '--ylim', '--colors', '--linewidth',
                     '--linestyle', '--legend', '--no-legend', '--grid', '--no-grid', '--figsize', '--font-size')
OWN_PLOT_STYLE = {'frog-profile'}  # keeps its own --style presets and --config


def test_plots_expose_shared_style_flags(command):
    cls, _, parser = command
    if not any(o.kind == 'plot' for o in cls.outputs) or cls.name in OWN_PLOT_STYLE:
        return
    missing = [f for f in SHARED_PLOT_FLAGS if f not in parser._option_string_actions]
    assert not missing, f"{cls.name}: plot without the shared style flags {missing}"


RAW_MATPLOTLIB = re.compile(r'plt\.subplots|\.savefig\(|plt\.show\(|plt\.close\(|import seaborn|'
                            r'rcParams\.update|tight_layout')


@pytest.mark.parametrize('path', sorted(SRC.glob('cli/*.py')) + [SRC / 'nmr_sim.py'], ids=lambda p: p.name)
def test_plots_are_drawn_through_plotter(path):
    """Figures are created, styled, saved and shown only through frogchem.plotter."""
    if path.name == 'reaction_profile.py':      # reaction_plotter keeps its own style system
        return
    hits = [f"{i}: {line.strip()}" for i, line in enumerate(path.read_text().splitlines(), 1)
            if RAW_MATPLOTLIB.search(line)]
    assert not hits, f"{path.name} bypasses frogchem.plotter:\n" + "\n".join(hits)


def test_flag_style(command):
    cls, _, parser = command
    for p in _parsers(parser):
        for opt, action in p._option_string_actions.items():
            if opt.startswith('--'):
                assert '_' not in opt, f"{cls.name}: {opt} should be kebab-case"
            else:
                assert len(opt) == 2, f"{cls.name}: single-dash option {opt} must be one letter"
                if opt in RESERVED_SHORT:
                    assert action.dest in RESERVED_SHORT[opt], \
                        f"{cls.name}: {opt} is reserved for {RESERVED_SHORT[opt]}, used for {action.dest!r}"


def test_tables_flags_match_declaration(command):
    cls, _, parser = command
    has_format = '--format' in parser._option_string_actions
    assert has_format == cls.has_tables
    declares_tables = any(o.kind == 'table' for o in cls.outputs)
    assert declares_tables == cls.has_tables, f"{cls.name}: has_tables={cls.has_tables} but table outputs={declares_tables}"


def test_ensemble_flags_only_when_supported(command):
    cls, _, parser = command
    assert ('--checkpoint' in parser._option_string_actions) == cls.supports_ensemble


def test_json_and_csv_documents():
    tables = [TableResult([['a', 1.0, float('nan')]], ['Name', 'E\n[eV]', 'X'], 'T1'),
              TableResult([[1, 2]], ['x', 'y'], 'T2')]
    doc = json.loads(tables_to_json(tables))
    assert doc[0] == {'title': 'T1', 'columns': ['Name', 'E [eV]', 'X'],
                      'rows': [{'Name': 'a', 'E [eV]': 1.0, 'X': None}]}
    assert tables_to_csv(tables) == '# T1\nName,E [eV],X\na,1.0,\n\n# T2\nx,y\n1,2\n'


def test_output_collects_all_tables(tmp_path, capsys):
    class Dummy(CLICommand):
        category = 'geometry'
        outputs = [Output('table', 'T', 'test')]

        def run(self, args):
            self.output([[1, 2]], ['a', 'b'], 'first')
            self.output([[3, 4]], ['a', 'b'], 'second')

    out = tmp_path / 'tables.json'
    Dummy().main(['--format', 'json', '--table-file', str(out)])
    stdout = json.loads(capsys.readouterr().out)
    assert [t['title'] for t in stdout] == ['first', 'second']
    assert json.loads(out.read_text()) == stdout


# ---------------------------------------------------------------- regressions (code review)

class _Writer(CLICommand):
    category = 'geometry'
    has_tables = False
    outputs = [Output('file', '<stem>_x.xyz', 'test')]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+')
        self.add_out_dir_arg(parser)

    def run(self, args):
        self.paths = [self.derived_path(f, 'x', '.xyz') for f in args.files]


def test_json_handles_numpy_arrays_and_infinity():
    import numpy as np
    doc = json.loads(tables_to_json([TableResult([[np.array([1.0, np.nan]), float('inf'), np.float64(2.5)]],
                                                 ['v', 'inf', 'f'], 't')]))
    assert doc[0]['rows'] == [{'v': [1.0, None], 'inf': None, 'f': 2.5}]


def test_derived_path_same_stem_with_out_dir_does_not_collide(tmp_path):
    a, b = tmp_path / 'conf_1' / 'orca.out', tmp_path / 'conf_2' / 'orca.out'
    for f in (a, b):
        f.parent.mkdir()
        f.write_text('x')
    tool = _Writer()
    tool.main([str(a), str(b), '--out-dir', str(tmp_path / 'out')])
    assert tool.paths[0].name == 'orca_x.xyz'
    assert tool.paths[1].name == 'conf_2_orca_x.xyz'


def test_derived_path_ignores_compression_suffix(tmp_path):
    f = tmp_path / 'ens.xyz.gz'
    f.write_text('x')
    tool = _Writer()
    tool.main([str(f)])
    assert tool.paths[0] == tmp_path / 'ens_x.xyz'


def test_unwritable_table_file_is_a_clean_error(tmp_path, capsys):
    class Dummy(CLICommand):
        category = 'geometry'
        outputs = [Output('table', 'T', 'test')]

        def run(self, args):
            self.output([[1]], ['a'], 't')

    with pytest.raises(SystemExit) as exc:
        Dummy().main(['--table-file', str(tmp_path)])   # a directory
    assert exc.value.code == 1
    assert 'Cannot write tables' in capsys.readouterr().err


def test_main_keeps_host_logging_and_messages_visible_in_library_use(capsys):
    import logging
    from frogchem.printer import print_success, set_quiet
    root = logging.getLogger()
    marker = logging.NullHandler()
    root.addHandler(marker)
    try:
        _Writer().main(['dummy.xyz', '-q'])
        assert marker in root.handlers
    finally:
        root.removeHandler(marker)
    set_quiet(False)
    print_success('visible')
    assert 'visible' in capsys.readouterr().err


def test_help_width_follows_terminal(monkeypatch):
    from frogchem.cli.base import help_width
    monkeypatch.setenv('COLUMNS', '80')
    assert help_width() == 78
    monkeypatch.setenv('COLUMNS', '200')
    assert help_width() == 100
