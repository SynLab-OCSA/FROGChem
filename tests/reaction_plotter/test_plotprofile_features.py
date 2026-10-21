import json
from pathlib import Path
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from frogchem.reaction_plotter import (
    ReactionPlotter,
    ReactionProfilePlotter,
    ReactionProfileAnalyzer,
    plot_reaction_profile,
)
from frogchem.cli.reaction_profile import ReactionProfile


class TestReactionProfilePlotterAPI:
    def test_direct_plot_dict(self, tmp_path):
        energy_data = {
            "Pathway A": [0.00, -2.0, 10.2, 1.4, -1.5, 2.0, -7.2],
            "Pathway B": [None, -2.0, 6.2, 4.3, 5.8, 2.0],
        }
        annotations = {
            "Step 1": (0, 3),
            "Step 2": (3, 5),
        }
        point_labels = {
            "Pathway A": ["R", "Int1", "TS1", "Int2", "TS2", "Int3", "P"],
        }
        out = str(tmp_path / "test_prof")
        plotter = ReactionProfilePlotter(style="presentation", point_type="bar")
        fig, ax = plotter.plot(
            energy_data,
            filename=out,
            annotations=annotations,
            point_labels=point_labels,
        )
        assert fig is not None
        assert ax is not None
        assert (tmp_path / "test_prof.png").exists()

    def test_straight_style_and_secondary_axis(self, tmp_path):
        energy_data = {
            "Path 1": [0.0, 15.0, -10.0],
        }
        secondary = {
            "Distance": [1.5, 2.2, 3.1],
        }
        plotter = ReactionPlotter(style="straight", axes="both")
        fig, ax = plotter.plot(
            energy_data,
            secondary=secondary,
            filename=str(tmp_path / "sec_plot"),
        )
        assert (tmp_path / "sec_plot.png").exists()

    def test_plot_reaction_profile_function(self, tmp_path):
        fig, ax = plot_reaction_profile(
            {"Path 1": [0.0, 5.0, 0.0]},
            style="default",
            point_type="dot",
            filename=str(tmp_path / "func_plot"),
        )
        assert (tmp_path / "func_plot.png").exists()


class TestReactionProfileAnalyzerJSON:
    def test_from_json_structured_frog_format(self, tmp_path):
        payload = {
            "title": "Catalytic Cross-Coupling Mechanism",
            "energy_unit": "kcal/mol",
            "pathways": {
                "Direct Mechanism": [
                    {"name": "Reactants", "energy": 0.0},
                    {"name": "Complex A", "energy": -4.2},
                    {"name": "TS1", "energy": 16.5, "is_ts": True},
                    {"name": "Products", "energy": -22.3},
                ],
                "Alternative Pathway": [
                    {"name": "Reactants", "energy": 0.0},
                    {"name": "Complex A'", "energy": -2.8},
                    {"name": "TS1'", "energy": 21.0, "is_ts": True},
                    {"name": "Products", "energy": -22.3},
                ],
            },
            "annotations": {
                "Oxidative Addition": [0, 2],
            },
        }
        json_file = tmp_path / "reaction.json"
        json_file.write_text(json.dumps(payload))

        analyzer = ReactionProfileAnalyzer.from_json(str(json_file))
        assert len(analyzer.pathways) == 2
        assert "Direct Mechanism" in analyzer.pathways
        assert len(analyzer.pathways["Direct Mechanism"]) == 4
        assert analyzer.config.title == "Catalytic Cross-Coupling Mechanism"
        assert "Oxidative Addition" in analyzer.annotations
        assert analyzer.annotations["Oxidative Addition"] == (0.0, 2.0)

        # Plot output
        out = str(tmp_path / "out_json_struct")
        fig, ax = analyzer.plot(filename=out, style="presentation")
        assert (tmp_path / "out_json_struct.png").exists()

    def test_from_json_plotprofile_format(self, tmp_path):
        payload = {
            "Pathway A": [0.0, -2.0, 10.2, 1.4, -7.2],
            "Pathway B": [None, -2.0, 6.2, 4.3, 2.0],
            "annotations": {
                "Step 1": [0, 2],
            },
            "point_labels": {
                "Pathway A": ["R", None, "TS1", None, "P"],
            },
        }
        json_file = tmp_path / "simple.json"
        json_file.write_text(json.dumps(payload))

        analyzer = ReactionProfileAnalyzer(str(json_file), auto_plot=False)
        assert len(analyzer.energy_data) == 2
        assert analyzer.energy_data["Pathway A"] == [0.0, -2.0, 10.2, 1.4, -7.2]
        assert "Step 1" in analyzer.annotations
        assert analyzer.point_labels["Pathway A"][0] == "R"

        out = str(tmp_path / "out_json_simple")
        fig, ax = analyzer.plot(filename=out)
        assert (tmp_path / "out_json_simple.png").exists()

    def test_from_dict(self):
        d = {
            "Path": [1.0, 2.0, 3.0],
        }
        analyzer = ReactionProfileAnalyzer.from_dict(d)
        assert "Path" in analyzer.energy_data
        assert analyzer.energy_data["Path"] == [1.0, 2.0, 3.0]

    def test_from_txt_explicit(self, tmp_path):
        content = """--# PES
0.0; 10.0; -5.0
--# LABELS
A; B; C
--# GRAPHS
title: Explicit TXT
label: Main Path
"""
        txt_file = tmp_path / "profile.txt"
        txt_file.write_text(content)

        analyzer = ReactionProfileAnalyzer.from_txt(str(txt_file))
        assert "Main Path" in analyzer.energy_data
        assert analyzer.energy_data["Main Path"] == [0.0, 10.0, -5.0]
        assert analyzer.config.title == "Explicit TXT"

        fig, ax = analyzer.plot(filename=str(tmp_path / "out_txt"))
        assert (tmp_path / "out_txt.png").exists()


class TestCLIReactionProfile:
    def test_cli_positional_txt(self, tmp_path):
        content = """--# PES
0.0; 12.0; -2.0
--# LABELS
R; TS; P
--# GRAPHS
title: CLI Test
label: Pathway
"""
        f = tmp_path / "input.txt"
        f.write_text(content)
        out = tmp_path / "cli_out"

        cmd = ReactionProfile()
        cmd.main([str(f), "-o", str(out), "--style", "presentation"])
        assert (tmp_path / "cli_out.png").exists()

    def test_cli_flag_input_json(self, tmp_path):
        data = {
            "pathways": {
                "Reaction 1": [
                    {"name": "Start", "energy": 0.0},
                    {"name": "End", "energy": -15.0},
                ]
            }
        }
        j = tmp_path / "input.json"
        j.write_text(json.dumps(data))
        out = tmp_path / "json_cli_out"

        cmd = ReactionProfile()
        cmd.main([str(j), "-o", str(out), "--fig-format", "svg"])
        assert (tmp_path / "json_cli_out.svg").exists()

    def test_cli_legend_barriers(self, tmp_path):
        content = """--# PES
0.0; 9.5; -10.0
0.5; 14.2; -8.0
--# LABELS
R; TS; P
R; TS; P
--# GRAPHS
legend: true
label: Path A; Path B
"""
        f = tmp_path / "multi_input.txt"
        f.write_text(content)
        out = tmp_path / "barriers_out"

        cmd = ReactionProfile()
        cmd.main([str(f), "-o", str(out), "--legend-barriers"])
        assert (tmp_path / "barriers_out.png").exists()

    def test_smart_anticollision_vertical_split(self, tmp_path):
        # Two competitive pathways with very close energies at the same X (step 1)
        energy_data = {
            "Path Major": [0.0, 20.2, -5.0],
            "Path Minor": [0.0, 20.0, -2.0],
        }
        point_labels = {
            "Path Major": ["R", "TS-Major", "P1"],
            "Path Minor": ["R", "TS-Minor", "P2"],
        }
        plotter = ReactionProfilePlotter(style="default", point_type="hollow", labels=True, sig_figs=1)
        fig, ax = plotter.plot(
            energy_data,
            point_labels=point_labels,
            filename=str(tmp_path / "anticollision"),
        )
        assert (tmp_path / "anticollision.png").exists()

        t_20_2 = next((t for t in ax.texts if "20.2" in t.get_text()), None)
        t_20_0 = next((t for t in ax.texts if "20.0" in t.get_text()), None)
        assert t_20_2 is not None and t_20_0 is not None
        assert t_20_2.get_position()[1] != t_20_0.get_position()[1], "Expected 20.2 and 20.0 to have different vertical positions"

        t_major = next((t for t in ax.texts if "TS-Major" in t.get_text()), None)
        t_minor = next((t for t in ax.texts if "TS-Minor" in t.get_text()), None)
        assert t_major is not None and t_minor is not None
        assert t_major.get_position()[1] > 0, f"Expected TS-Major to have positive vertical offset, got {t_major.get_position()}"
        assert t_minor.get_position()[1] < 0, f"Expected TS-Minor to have negative vertical offset, got {t_minor.get_position()}"
