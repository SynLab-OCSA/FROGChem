# CLI Output Conventions

Every `frog-*` command follows the same rules for what it prints, which files it
writes and how it draws plots. The rules are enforced by `frogchem.cli.base.CLICommand`,
`frogchem.printer` and `frogchem.plotter`; `tests/cli/test_cli_conventions.py` checks them.

---

## 1. Terminal output

| Stream | Content |
|---|---|
| **stdout** | Data only: result tables (`--format table`), or one JSON / CSV document (`--format json\|csv`). |
| **stderr** | Everything else: title banner, progress (`ℹ`), saved-file notices (`✔ Saved …`), warnings, errors. |

* `-q/--quiet` hides banner, info and saved-file notices (warnings and errors stay).
* `-v/--verbose` enables debug logging.
* Tables are emitted only through `self.output(rows, headers, title)`. Commands never
  call `print()` / `console.print()` for data.
* `--format json` prints **one** JSON document (`[]` when nothing was found), always with the same shape:

  ```json
  [
    {"title": "Charges (calc.out)", "columns": ["Atom", "Mulliken"], "rows": [{"Atom": "C0", "Mulliken": -0.12}]}
  ]
  ```

  Numbers stay numbers, `NaN` and `±inf` become `null`, numpy arrays become lists; text is literal (no Rich markup in titles, headers or cells).
* `--format csv` prints every table as `# <title>` + header + rows, tables separated by
  a blank line (read back with `pandas.read_csv(..., comment='#')`).
* `--table-file PATH` writes **all** tables of the run to `PATH`. The format comes from
  the extension (`.json`, `.csv`, `.txt`), otherwise from `--format`.
* Commands without tables do not expose `--format` / `--table-file`.

## 2. Files

* **Per-input files** (derived from one input, e.g. sorted XYZ, cubes, per-file plots) are
  written **next to the input** as `{stem}_{tag}.{ext}`; `--out-dir DIR` redirects them.
  Compression suffixes are ignored (`ens.xyz.gz` → `ens_{tag}.xyz`). When two inputs of the same
  run would produce the same name (e.g. `conf_1/orca.out` and `conf_2/orca.out` with `--out-dir`),
  the later one gets its parent folders as prefix (`conf_2_orca_{tag}.{ext}`) instead of overwriting.
* **Aggregate files** (built from several inputs, e.g. superimposed spectra) go to the
  current directory, or to `-o/--output`.
* `-o/--output PATH` always means *the primary output path* (file, or prefix when the
  command writes several files: `{prefix}_{tag}.{ext}`). It is never an input.
* An input file is never overwritten.
* Every written file is reported on stderr as `✔ Saved <kind>: <path>`.

## 3. Plots

* Plots are **saved, never shown** by default. `--show` also opens all figures of the run, once, at the end.
* Shared options: `--fig-format {png,svg,pdf,eps}` (default `png`), `--dpi` (default `300`).
  An extension given in the output path wins over `--fig-format`.
* Commands whose main product is a plot take the path from `-o/--output`.
  Commands where the plot is optional enable it with `--plot [PATH]`.
* Every plot command accepts the same look options:

  | Flag | Meaning |
  |---|---|
  | `--title TEXT` | Plot title (`--title ''` removes it) |
  | `--xlabel TEXT`, `--ylabel TEXT` | Axis labels |
  | `--xlim MIN MAX`, `--ylim MIN MAX` | Axis ranges (an inverted ppm axis stays inverted) |
  | `--colors C [C ...]` | Colours of the plotted series, in the order the command documents |
  | `--linewidth W`, `--linestyle {solid,dashed,dotted,dashdot}` | Lines |
  | `--legend` / `--no-legend`, `--grid` / `--no-grid` | Force legend / grid on or off |
  | `--figsize W H`, `--font-size PT` | Figure size (inches) and base font size |
  | `--cmap NAME` | Colormap, only on commands that draw maps (`frog-mo`, `frog-nmr`) |

  Precedence: **flag > command default > shared style**. Without a flag the command decides
  (e.g. the legend appears only when there are several labelled curves).
* All figures are drawn with `frogchem.plotter`, the plotting twin of `frogchem.printer`: shared
  style (fonts, sizes, line widths, `bbox_inches="tight"`) applied as a context, so matplotlib's
  global settings are untouched; primitives `line`, `scatter`, `bar`, `hist`, `heatmap`, `contour`;
  `finish()` applies title, labels, limits, legend and grid. In a command:

  ```python
  from frogchem import plotter

  opts = self.plot_options(title='Computed UV Spectrum', xlabel='Wavelength (nm)')  # command defaults
  fig, ax = plotter.line(x, {'conf_1': y1, 'conf_2': y2}, opts=opts)
  self.save_figure(fig, self.plot_path('spectra', tag='UV'))
  ```

  `frog-profile` keeps its own style presets (`--style`, `--config`) and has no look options.

## 4. Flags

* Long flags are kebab-case (`--out-dir`, not `--out_dir`); single-dash flags are a single
  letter (`-t`, not `-thr`).
* Reserved short flags, with one meaning across all commands:

  | Flag | Meaning |
  |---|---|
  | `-o/--output` | primary output path |
  | `-i/--indices` | atom indices, **0-based** |
  | `-t/--threshold` | numeric threshold |
  | `-w/--write` | write derived structure files |
  | `-f/--force` | overwrite existing outputs |
  | `-T/--temp` | temperature in K |
  | `-c/--checkpoint` | Boltzmann populations (ensemble commands only) |
  | `-e/--energy-term` | energy used for Boltzmann weights (ensemble commands only) |
  | `-v`, `-q` | verbose / quiet |

* Interactive modes use `--interactive` (no short flag).
* Help strings: English, start with a capital letter, no trailing period, units in
  the text (Å, K, kcal/mol, eV). Defaults are appended automatically, never written by hand.
* Ensemble flags (`-c`, `--read-boltzmann`, `-e`, `-T`, `--individual`) appear only on
  commands that support Boltzmann averaging.

## 5. Numbering

* **Atoms are 0-based** in every option and every table: the first atom of the input is atom 0. This is the
  numbering ORCA prints; Gaussian, GaussView and Avogadro start at 1. Ranges such as `1-24` include both ends.
* **Excited states** (TD-DFT) are labelled by spin multiplicity and number: **S1, S2, …** (singlets),
  **D1, …** (doublets), **T1, T2, …** (triplets), **Q1, …** (quartets); other multiplicities as `M5-1`.
  Each multiplicity is numbered on its own, as in the program output, and **S0** is the ground state.
  The same labels appear in the excited-state tables, the SOC matrix, the difference-density file names
  (`<stem>_T1_delta_rho.cube`) and are accepted by `--dct-states` / `--soc-states` (e.g. `S1-S3 T1`).
* **Orbitals** are 0-based indices as printed by ORCA; in transition labels `H` is the HOMO and `L` the LUMO
  (`H-1`, `L+2`, …), with `a`/`b` for α/β in open-shell systems (`--raw-orbitals` shows the indices instead).

## 6. Documenting a command

Each command declares its outputs and examples, plus a plain-language `guide`
(`Guide(why, needs, how, expect)`: what it is for, what you need, the steps, what to expect);
`--help`, the README and these docs are generated from the declaration, and
`tests/unit/test_cli_guides.py` fails if a command has no complete guide.

```python
from frogchem.cli.base import CLICommand, Output

class GetCharges(CLICommand, name='frog-charges'):
    """Extract Mulliken and Löwdin atomic charges for selected atoms."""
    category = 'parsing'
    supports_ensemble = True
    outputs = [
        Output('table', 'Charges (<file>)', 'One row per selected atom: Mulliken and Löwdin charge (e)'),
        Output('file', '<stem>_charges.xyz', 'Geometry with charges in the comment line', when='-w'),
        Output('plot', '<stem>_charges.png', 'Bar chart of the charges', when='--plot'),
    ]
    examples = [
        ('frog-charges opt.out -i 0 1 2', 'Charges of atoms 0, 1 and 2'),
        ('frog-charges conf_*.out -i 0 -c checkpoint.json', 'Boltzmann-averaged charge of atom 0'),
    ]
```

The help then ends with:

```text
outputs:
  terminal (stdout; --format table|json|csv, --table-file PATH):
    Charges (<file>)          One row per selected atom: Mulliken and Löwdin charge (e)
  files (next to each input, or --out-dir):
    <stem>_charges.xyz        [-w] Geometry with charges in the comment line
  plots (--fig-format, --dpi, --show):
    <stem>_charges.png        [--plot] Bar chart of the charges

examples:
  frog-charges opt.out -i 0 1 2
      Charges of atoms 0, 1 and 2
```

Regenerate README and the per-command pages `docs/cli/<command>.md` with `python generate_readme.py`
(`--docs-only` leaves README untouched); re-run the real examples with `python docs/build_examples.py`.
