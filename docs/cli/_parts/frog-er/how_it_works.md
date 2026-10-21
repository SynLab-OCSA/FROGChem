The command does no file parsing: it applies the Boltzmann relation between two states separated by a
free-energy difference (`frogchem.thermo.calc_er` and `frogchem.thermo.calc_dg`). ΔG is taken in kcal/mol and
converted to J/mol (×4184) before it is divided by $RT$, with $T$ from `-T` (default 298.15 K).

**`er` (ΔG → ratio).** The ratio minor/major is the Boltzmann factor

$$
\frac{n_\mathrm{minor}}{n_\mathrm{major}} = \exp\!\left(-\frac{\Delta G}{RT}\right)
$$

and the two shares and the enantiomeric (or diastereomeric) excess follow from it:

$$
x_\mathrm{major} = \frac{1}{1 + e^{-\Delta G/RT}}, \qquad
x_\mathrm{minor} = 1 - x_\mathrm{major}, \qquad
ee = (x_\mathrm{major} - x_\mathrm{minor}) \times 100\ \%.
$$

The table prints $x_\mathrm{major}$ and $x_\mathrm{minor}$ in percent (two decimals). The sign of ΔG is not
checked: a negative value gives $x_\mathrm{major} < 50\ \%$ and a negative *ee*, i.e. the labels "major" and
"minor" swap.

**`dg` (ratio → ΔG).** The two amounts are first normalised by their sum, so any unit works (percent,
fractions, HPLC areas). Then

$$
\Delta G = -RT \ln\frac{n_\mathrm{minor}}{n_\mathrm{major}}
$$

is converted back to kcal/mol and printed with three decimals. If the "minor" amount is larger than the
"major" one, ΔG comes out negative.

Both subcommands accept the global output options (`--format`, `-q`, `-v`, table file) after the
subcommand name as well as before it. The computed ΔG used as input normally comes from the G of two
competing transition states (see `frog-rrho` and [Thermochemistry theory](../theory/thermo.md)); the
formula assumes kinetic control by those two states only.
