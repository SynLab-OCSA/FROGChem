import pytest
import matplotlib
matplotlib.use('Agg')

from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer


class TestReactionProfileAnalyzer:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ReactionProfileAnalyzer("nonexistent_file.yaml")

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("some random content")
        with pytest.raises(ValueError):
            ReactionProfileAnalyzer(str(f))

    def test_legacy_txt_rejected(self, tmp_path):
        f = tmp_path / "old.txt"
        f.write_text("--# PES\n0.0; 10.0\n--# LABELS\nA; B\n")
        with pytest.raises(ValueError):
            ReactionProfileAnalyzer(str(f))

    def test_valid_basic_file(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("title: Test Plot\nlegend: true\nsteps: [A, B, C]\n"
                     "pathways:\n  - name: Path 1\n    energies: [0.0, 10.0, 20.0]\n")
        analyzer = ReactionProfileAnalyzer(str(f))
        assert analyzer.config.legend is True
        assert analyzer.config.title == "Test Plot"
        assert len(analyzer.pes_data) == 1

    def test_multiple_paths(self, tmp_path):
        f = tmp_path / "multi.yaml"
        f.write_text("steps: [A, B, C]\npathways:\n"
                     "  - {name: Path 1, color: red, energies: [0.0, 10.0, 20.0]}\n"
                     "  - {name: Path 2, color: blue, energies: [5.0, 15.0, 25.0]}\n")
        analyzer = ReactionProfileAnalyzer(str(f))
        assert len(analyzer.pes_data) == 2
        assert analyzer.config.colors == ["red", "blue"]
        assert analyzer.config.legend is True

    def test_single_path_no_legend(self, tmp_path):
        f = tmp_path / "single.yaml"
        f.write_text("steps: [A, B, C]\npathways:\n  - {name: Only, energies: [0.0, 10.0, 20.0]}\n")
        assert ReactionProfileAnalyzer(str(f)).config.legend is False

    def test_json_following_yaml_schema(self, tmp_path):
        f = tmp_path / "p.json"
        f.write_text('{"steps": ["R", "P"], "pathways": [{"name": "A", "energies": [0.0, -5.0]}]}')
        assert ReactionProfileAnalyzer(str(f)).energy_data == {"A": [0.0, -5.0]}
