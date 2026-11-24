import matplotlib
matplotlib.use('Agg')

from frogchem.reaction_plotter import ReactionProfilePlotter
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
        plotter = ReactionProfilePlotter(style="straight", axes="both")
        fig, ax = plotter.plot(
            energy_data,
            secondary=secondary,
            filename=str(tmp_path / "sec_plot"),
        )
        assert (tmp_path / "sec_plot.png").exists()


class TestCLIReactionProfile:
    def test_cli_svg_presentation(self, tmp_path):
        f = tmp_path / "input.yaml"
        f.write_text("title: CLI Test\nsteps: [R, TS, P]\npathways:\n  - {name: Pathway, energies: [0.0, 12.0, -2.0]}\n")
        ReactionProfile().main([str(f), "-o", str(tmp_path / "cli_out"), "--style", "presentation",
                                "--fig-format", "svg"])
        assert (tmp_path / "cli_out.svg").exists()

    def test_cli_legend_barriers(self, tmp_path):
        f = tmp_path / "multi_input.yaml"
        f.write_text("legend: true\nsteps: [R, TS, P]\npathways:\n"
                     "  - {name: Path A, energies: [0.0, 9.5, -10.0]}\n"
                     "  - {name: Path B, energies: [0.5, 14.2, -8.0]}\n")
        ReactionProfile().main([str(f), "-o", str(tmp_path / "barriers_out"), "--legend-barriers"])
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
