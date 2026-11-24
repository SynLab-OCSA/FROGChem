The command does no quantum-chemistry parsing: it reads a table of energies and draws it
(`frogchem.reaction_plotter.ReactionProfileAnalyzer` and `ReactionProfilePlotter`).

1. **Input.** The file is read as YAML and must be a mapping (`steps:`, `pathways:`, options); anything
   else is rejected. JSON written with the same keys also loads, since JSON is valid YAML.
2. **Pathways.** Each pathway becomes a list of energies on a common x grid (state index 0, 1, 2 …). A
   branch (`branch_at:` a state name or index) is padded with empty points up to that state, so it starts
   there. Empty entries (`none`, `nan`, `-`) leave gaps. The numbers are used exactly as written, with no
   unit conversion.
3. **Curves.** Consecutive points are joined by cubic Bézier segments whose inner control points are
   shifted horizontally by `curviness` × Δx (default 0.42; `--curviness`, `0` = straight lines;
   `--style straight` sets 0). With `--point-type bar` the segments start and end at the bar ends.
4. **Labels.** Energies are written at the states selected by `label_points`, with one
   decimal by default. `--legend-barriers` appends to each legend entry the largest energy between the
   first and the last point of the pathway (as given, not relative to the preceding minimum).
5. **Style and saving.** `--style default|presentation|straight` selects a preset; options from the
   input file and from `--config`, `--annotations`, `--point-labels`, `--secondary` and `--x` (JSON files
   or JSON strings) are added on top. The figure is saved with `--fig-format` (PNG) and `--dpi` (300).

For editing profiles interactively, use the **FROG Reaction Profile Designer**, a Streamlit web app
shipped with the package (`src/frogchem/reaction_plotter/web_gui.py`): start it with `frog-profile --gui`
after `pip install streamlit`. It offers a spreadsheet-like table of energies, presets, automatic
detection of TS/MECP peaks, and export to PNG/SVG/PDF and to a YAML file that this command can read.
More input examples are in the
[Reaction Profiles user guide](../user-guide/reaction_profiles.md).
