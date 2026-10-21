"""YAML pathway label dicts whose keys match no step must not crash."""

import logging

import matplotlib
matplotlib.use('Agg')

from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer


def test_unknown_label_key_is_ignored(tmp_path, caplog):
    yaml_file = tmp_path / "labels.yaml"
    yaml_file.write_text("""
steps: [A, B, C]
pathways:
  - name: "Path1"
    energies: [0.0, 10.0, 5.0]
    labels:
      B: "TS1"
      Z: "nowhere"
""", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        analyzer = ReactionProfileAnalyzer(str(yaml_file), auto_plot=False)

    assert analyzer.point_labels["Path1"] == ["A", "TS1", "C"]
    assert "Z" in caplog.text
