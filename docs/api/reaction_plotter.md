# Reaction Profiles API Reference: `frogchem.reaction_plotter`

Python side of [`frog-profile`](../cli/frog-profile.md): parse a YAML reaction profile and draw it.
The input format is described in the [Reaction Profiles user guide](../user-guide/reaction_profiles.md).

---

## `ReactionProfileAnalyzer(filename=None, content=None)`

Parses a YAML profile, from a file (`filename`) or a string (`content`). Raises `FileNotFoundError`
for a missing file and `ValueError` when the input is not a YAML mapping.

* `energy_data` (`dict[str, list[float | None]]`): energies of each pathway on the common state grid
  (`None` where a branch has no point).
* `point_labels` (`dict[str, list[str | None]]`): state names of each pathway.
* `config` (`PlotConfig`): title, colours, styles, `label_points`, sections and the other plot options.
* `plot(filename=None, file_format="png", dpi=600, style="default", **kwargs)`: draws the profile and
  returns `(fig, ax)`; with `filename` it also saves `<filename>.<file_format>`. `style` is
  `default`, `presentation` or `straight`; `kwargs` override style options (e.g. `point_type`,
  `curviness`, `legend_barriers`, `annotations`, `point_labels`, `x`).

```python
from frogchem.reaction_plotter import ReactionProfileAnalyzer

analyzer = ReactionProfileAnalyzer("pes.yaml")
fig, ax = analyzer.plot(style="presentation")
fig.savefig("pes_profile.png", dpi=300)
```

## `ReactionProfilePlotter(style="default", **style_options)`

The drawing engine used by `plot()`. `plot(energy_data, ...)` takes `{name: [energies]}` (or a list
of lists) directly, without an input file, and returns `(fig, ax)`.

```python
from frogchem.reaction_plotter import ReactionProfilePlotter

fig, ax = ReactionProfilePlotter(point_type="bar").plot({"Path A": [0.0, 18.4, -3.2, 12.1, -10.5]})
```
