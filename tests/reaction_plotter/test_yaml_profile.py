"""Tests for YAML input format and relaxed TXT parsing in reaction_plotter."""

from pathlib import Path
import pytest
import matplotlib
matplotlib.use('Agg')

from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer
from frogchem.reaction_plotter.plotter import ReactionProfilePlotter
from frogchem.cli.reaction_profile import ReactionProfile


SAMPLE_YAML = """
title: "Photochemical Mechanism"
units: "kcal/mol"
marker: hollow
curviness: 0.42
figsize: [12, 7]
xticks: false
excitation: 0

steps:
  - 1a-Br S0
  - 1a-Br S1
  - I
  - I'
  - TS-I
  - II
  - MECP-SET
  - IV
  - TS-IV
  - 2a-HBr

label_points:
  - TS-I
  - MECP-SET

pathways:
  - name: "Trunk"
    color: "#103E84"
    style: solid
    energies: [0.0, 71.7, 69.4, 70.5, 74.5, 42.0]

  - name: "SET pathway"
    color: "#1F56A3"
    style: solid
    branch_at: II
    energies: [42.0, 47.1, 13.9, 16.6, -24.1]

  - name: "HAT pathway"
    color: "#58A4E0"
    style: dashed
    branch_at: 5
    energies: [42.0, 46.9, 39.2, 16.6]
"""

RELAXED_TXT = """
--# PES
Trunk: 0.0; 71.7; 69.4; 70.5; 74.5; 42.0; None; None; None; None
SET: @II: 42.0; 47.1; 13.9; 16.6; -24.1
HAT: @5: 42.0; 46.9; 39.2; 16.6; None

--# LABELS
1a-Br S0; 1a-Br S1; I; I'; TS-I; II; MECP-SET; IV; TS-IV; 2a-HBr

--# GRAPHS
title : Photochemical Test
excitation : 0
point_type : hollow
labelpoint : TS-I; MECP-SET
figsize : 12; 7
save : false
"""


class TestYamlReactionProfile:
    def test_yaml_parsing_and_branching(self, tmp_path):
        yaml_file = tmp_path / "mechanism.yaml"
        yaml_file.write_text(SAMPLE_YAML, encoding="utf-8")

        analyzer = ReactionProfileAnalyzer(str(yaml_file), auto_plot=False)

        # Check energy data
        assert "Trunk" in analyzer.energy_data
        assert "SET pathway" in analyzer.energy_data
        assert "HAT pathway" in analyzer.energy_data

        # Check automatic None padding
        set_vals = analyzer.energy_data["SET pathway"]
        # Step II is index 5, so first 5 entries must be None
        assert set_vals[:5] == [None, None, None, None, None]
        assert set_vals[5] == 42.0
        assert set_vals[6] == 47.1

        hat_vals = analyzer.energy_data["HAT pathway"]
        assert hat_vals[:5] == [None, None, None, None, None]
        assert hat_vals[5] == 42.0

        # Check excitation x coordinates
        assert analyzer.config.x is not None
        # Excitation at 0 means x[0] == 0, x[1] == 0, x[2] == 1, etc.
        assert analyzer.config.x[0] == 0
        assert analyzer.config.x[1] == 0
        assert analyzer.config.x[2] == 1

        # Check label points resolved from species names to indices
        # TS-I is index 4, MECP-SET is index 6
        assert 4 in analyzer.config.labelpoint
        assert 6 in analyzer.config.labelpoint

        # Check visual configs
        assert analyzer.config.point_type == "hollow"
        assert analyzer.config.curviness == 0.42
        assert analyzer.config.show_xticks is False
        assert analyzer.config.figsize == (12.0, 7.0)

    def test_yaml_plot_generation(self, tmp_path):
        yaml_file = tmp_path / "mechanism.yaml"
        yaml_file.write_text(SAMPLE_YAML, encoding="utf-8")
        out_png = tmp_path / "output_plot"

        analyzer = ReactionProfileAnalyzer(str(yaml_file), auto_plot=False)
        fig, ax = analyzer.plot(filename=str(out_png), file_format="png")

        assert fig is not None
        assert ax is not None
        assert (tmp_path / "output_plot.png").exists()
        assert (tmp_path / "output_plot.png").stat().st_size > 0

    def test_relaxed_txt_parsing(self, tmp_path):
        txt_file = tmp_path / "relaxed.txt"
        txt_file.write_text(RELAXED_TXT, encoding="utf-8")

        analyzer = ReactionProfileAnalyzer(str(txt_file), auto_plot=False)

        assert "Trunk" in analyzer.energy_data
        assert "SET" in analyzer.energy_data
        assert "HAT" in analyzer.energy_data

        # Check that @II was branched at index 5
        assert analyzer.energy_data["SET"][:5] == [None, None, None, None, None]
        assert analyzer.energy_data["SET"][5] == 42.0

        # Check that single LABELS line was applied to all pathways
        assert len(analyzer.point_labels["Trunk"]) == 10
        assert analyzer.point_labels["Trunk"][4] == "TS-I"
        assert analyzer.point_labels["SET"][4] == "TS-I"

        # Check excitation
        assert analyzer.config.x[0] == analyzer.config.x[1]

        # Check labelpoint by name
        assert 4 in analyzer.config.labelpoint
        assert 6 in analyzer.config.labelpoint

    def test_cli_with_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "cli_mech.yaml"
        yaml_file.write_text(SAMPLE_YAML, encoding="utf-8")
        out_file = tmp_path / "cli_out"

        cmd = ReactionProfile()
        cmd.main([str(yaml_file), "-o", str(out_file), "--fig-format", "png"])

        assert (tmp_path / "cli_out.png").exists()
        assert (tmp_path / "cli_out.png").stat().st_size > 0

    def test_distinct_pathway_labels(self, tmp_path):
        yaml_content = """
title: "Distinct Labels Test"
steps: [A, B, C, D]
label_points: [C, TS-alt]
pathways:
  - name: "Path1"
    energies: [0.0, 10.0, 20.0, 5.0]
  - name: "Path2"
    branch_at: B
    energies: [10.0, 25.0, 2.0]
    labels:
      C: "TS-alt"
      D: "Prod2"
"""
        yaml_file = tmp_path / "distinct.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")

        analyzer = ReactionProfileAnalyzer(str(yaml_file), auto_plot=False)

        # Check that Path1 has original step names
        assert analyzer.point_labels["Path1"][2] == "C"
        assert analyzer.point_labels["Path1"][3] == "D"

        # Check that Path2 has overridden labels
        assert analyzer.point_labels["Path2"][2] == "TS-alt"
        assert analyzer.point_labels["Path2"][3] == "Prod2"

        # Check label_points resolved TS-alt from Path2
        assert 2 in analyzer.config.labelpoint

        # Test plot generation with xticks combining labels
        fig, ax = analyzer.plot(xticks=True)
        ticks = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert "C / TS-alt" in ticks
        assert "D / Prod2" in ticks
