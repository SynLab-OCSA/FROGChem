# Reaction Profiles API Reference: `frogchem.reaction_plotter`

The `frogchem.reaction_plotter` module provides automated parsing and publication-quality graphical rendering of multi-step reaction pathways, competing routes, and catalytic cycles.

---

## Classes

### `ReactionProfileAnalyzer`

Parses, validates, and analyzes reaction pathways and intermediate relative free energies.

#### Methods

##### `from_json(filepath)`
Class method to construct an analyzer instance from a JSON file.

**Parameters:**
* `filepath` (`str | Path`): Path to JSON profile definition.

**Returns:**
* `ReactionProfileAnalyzer`: Initialized analyzer.

**Esempio:**

```python
from frogchem.reaction_plotter import ReactionProfileAnalyzer

analyzer = ReactionProfileAnalyzer.from_json("reaction.json")
print("Pathways:", list(analyzer.pathways.keys()))
```

**Output atteso:**
```text
Pathways: ['Direct Mechanism', 'Alternative Pathway']
```

##### `get_pathway(name)`
Retrieves a specific reaction pathway list.

##### `re_reference(zero_state_name)`
Re-references all pathway energies so that the specified state has energy $0.0$.

---

### `ReactionPlotter`

Renders publication-ready matplotlib figures of reaction profiles.

#### Constructor

```python
ReactionPlotter(analyzer, style="default")
```

#### Methods

##### `plot(show_labels=True, show_energies=True, energy_unit="kcal/mol", line_style="dashed", color_palette=None, figsize=(10, 6))`

Generates the matplotlib figure.

**Parameters:**

* `show_labels` (`bool`, optional): Whether to annotate intermediate/TS state names. Default: `True`.
* `show_energies` (`bool`, optional): Whether to display relative energy values above energy levels. Default: `True`.
* `energy_unit` (`str`, optional): Energy unit for display (`"kcal/mol"`, `"kJ/mol"`, or `"Hartree"`). Default: `"kcal/mol"`.
* `line_style` (`str`, optional): Matplotlib line style connecting states. Default: `"dashed"`.
* `color_palette` (`list[str] | None`, optional): List of hex colors for pathways.
* `figsize` (`tuple[float, float]`, optional): Figure size in inches. Default: `(10, 6)`.

**Returns:**

* `tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]`: The Matplotlib Figure and Axes objects.

**Esempio:**

```python
from frogchem.reaction_plotter import ReactionPlotter, ReactionProfileAnalyzer

analyzer = ReactionProfileAnalyzer.from_json("reaction.json")
plotter = ReactionPlotter(analyzer)
fig, ax = plotter.plot(show_labels=True, show_energies=True)
fig.savefig("profile.png")
print("Plot successfully rendered.")
```

**Output atteso:**
```text
Plot successfully rendered.
```
