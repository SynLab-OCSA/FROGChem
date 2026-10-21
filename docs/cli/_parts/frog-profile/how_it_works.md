The command does no quantum-chemistry parsing: it reads a table of energies and draws it
(`frogchem.reaction_plotter.ReactionProfileAnalyzer` and `ReactionProfilePlotter`).

1. **Format detection.** The text is parsed as JSON if it starts with `{` or the file ends in `.json`;
   as YAML if the extension is `.yaml`/`.yml` or a `pathways:`, `steps:`, `energies:` or `title:` key is
   present; otherwise as TXT sections introduced by `--# ` (`PES`, `LABELS`, `GRAPHS`). A TXT file
   without any `--# ` section is rejected with *Input file must contain at least a PES section*.
2. **Pathways.** Each pathway becomes a list of energies on a common x grid (state index 0, 1, 2 …). A
   branch (`branch_at:` in YAML, `@NAME:` or `@index:` in TXT) is padded with empty points up to the named
   state, so it starts there. Empty entries (`none`, `nan`, `-`) leave gaps. The TXT option
   `relative : true` turns step differences into a cumulative profile; otherwise the numbers are used
   exactly as written, with no unit conversion.
3. **Curves.** Consecutive points are joined by cubic Bézier segments whose inner control points are
   shifted horizontally by `curviness` × Δx (default 0.42; `--curviness`, `0` = straight lines;
   `--style straight` sets 0). With `--point-type bar` the segments start and end at the bar ends.
4. **Labels.** Energies are written at the states selected by `label_points` / `labelpoint`, with one
   decimal by default. `--legend-barriers` appends to each legend entry the largest energy between the
   first and the last point of the pathway (as given, not relative to the preceding minimum).
5. **Style and saving.** `--style default|presentation|straight` selects a preset; options from the
   input file and from `--config`, `--annotations`, `--point-labels`, `--secondary` and `--x` (JSON files
   or JSON strings) are added on top. The figure is saved with `--fig-format` (PNG) and `--dpi` (300).

For editing profiles interactively, use the **FROG Reaction Profile Designer**, a Streamlit web app
shipped with the package (`src/frogchem/reaction_plotter/web_gui.py`): start it with `frog-profile --gui`
after `pip install streamlit`. It offers a spreadsheet-like table of energies, presets, automatic
detection of TS/MECP peaks, and export to PNG/SVG/PDF and to an `input.txt` that this command can read.
`frog-profile --interactive` is the terminal counterpart. More input examples are in the
[Reaction Profiles user guide](../user-guide/reaction_profiles.md).
