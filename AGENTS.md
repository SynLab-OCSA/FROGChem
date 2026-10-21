# FROG — Agent Guide

## Commands

| Action | Command |
|---|---|
| Install (editable) | `pip install -e .` |
| Run all tests | `pytest` |
| Run one test package | `pytest tests/unit/` |
| Run one test file | `pytest tests/unit/test_thermo.py -v` |
| Regenerate README + `docs/cli/*.md` | `python generate_readme.py` (Makefile uses hardcoded conda path) |

No CI, linter, formatter, typecheck, or pre-commit config exists.

## Architecture

- `src/frogchem/` — core modules (thermo, io, symmetry, grid, density, density_builder, graph_theory, math_geometry, chimera_commands, gui_utils, printer)
- `src/frogchem/cli/` — one module per command, each a `CLICommand` subclass (`base.py`) registered in `pyproject.toml` under `[project.scripts]` and in `base._CLI_MODULES`. Commands declare `category`, `outputs`, `examples`, `has_tables`, `supports_ensemble`; output/file/plot/flag rules are in `docs/cli/conventions.md` and enforced by `tests/cli/test_cli_conventions.py`. Plots go through `frogchem.plotter` (the plotting twin of `frogchem.printer`): `PlotOptions` + `self.plot_options(**command_defaults)`, primitives `line`/`scatter`/`bar`/`hist`/`heatmap`/`contour`, `finish`, `save_figure`; style is a context (`plotter.style`), never global rcParams. No raw `plt.subplots`/`savefig`/`plt.show` in `cli/` or `nmr_sim.py` (checked by `test_cli_conventions.py`); `reaction_plotter/` keeps its own style system.
- `src/frogchem/orca_parser/` — ORCA output parser. `calculation.py` is the main entry (class `Calculation`). Submodules: hessian, opt, tddft, nbo, neb, orbitals, population, chemistry, orb_type, regex_ORCA.
- `src/frogchem/gaussian_parser/` — mirrors orca_parser for Gaussian (class `GaussianCalculation`). **Imports `frogchem.orca_parser.chemistry`** — cross-module dependency.
- `src/frogchem/substituter/` — fragment substitution. `__init__.py` re-exports: `attach_fragment`, `TopologicalOptimizer`, `reset_trajectory_file`. Submodules: builder, optimizer, trajectory, database_sub.
- `src/frogchem/reaction_plotter/` — uses **CamelCase filenames** (`ReactionPlotter.py`, `ReactionProfile.py`), unlike rest of codebase.
- `tests/` — plain pytest (no conftest). Subdirs: unit/, cli/, reaction_plotter/, test_io/, substituter/. **`tests/substituter/` has `__init__.py` but no test files.**
- `src/frogchem/__init__.py` is empty — imports must be explicit per module.
- All imports use absolute paths (`from frogchem.X.Y import Z`, not `from .Y import Z`).

## Gotchas

- `src/frogchem/data/` exists but is **empty** despite `package-data` in pyproject.toml.
- Atom indices in CLI tools are **0-based** (documented per-command).
- `tests/test_io/test_io.py` requires `rdkit` (not in pyproject.toml deps) — pre-existing skip, not a parser failure.
- VS Code `.vscode/settings.json` has `pytestEnabled: false` and `unittestEnabled: true`, but **all tests use pytest**. Run pytest from CLI; VS Code test runner will not work.
- README.md and `docs/cli/*_tools.md` + `docs/cli/reference.md` are generated from the CLI classes (help, `outputs`, `examples`) — edit the CLI classes or `generate_readme.py`, never those files directly. `docs/cli/conventions.md` is hand-written.
- `generate_readme.py` inserts `./src` to `sys.path` — can run without pip install.
- `files/` is gitignored but contains local test data (xyz, out, cube files). Useful for manual testing.
