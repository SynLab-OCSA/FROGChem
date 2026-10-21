import pytest
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

from frogchem.reaction_plotter.profile import (
    ReactionProfileAnalyzer,
)


class TestCalculateXPositions:
    def test_single_point(self):
        result = ReactionProfileAnalyzer._calculate_x_positions(1, 5)
        assert result == [0.0]

    def test_two_points(self):
        result = ReactionProfileAnalyzer._calculate_x_positions(2, 5)
        assert result == [0.0, 4.0]

    def test_equal_length(self):
        result = ReactionProfileAnalyzer._calculate_x_positions(5, 5)
        assert result == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_path_longer_than_max(self):
        result = ReactionProfileAnalyzer._calculate_x_positions(5, 3)
        assert result == [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_zero_max_points(self):
        result = ReactionProfileAnalyzer._calculate_x_positions(2, 1)
        assert result == [0.0, 0.0]


class TestReactionProfileAnalyzer:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ReactionProfileAnalyzer("nonexistent_file.txt")

    def test_invalid_format(self, tmp_path):
        f = tmp_path / "bad.txt"
        f.write_text("some random content")
        with pytest.raises((ValueError, IndexError)):
            ReactionProfileAnalyzer(str(f))

    def test_valid_basic_file(self, tmp_path):
        content = """--# PES
0.0; 10.0; 20.0
--# LABELS
A; B; C
--# GRAPHS
legend: true
title: Test Plot
label: Path 1
"""
        f = tmp_path / "test.txt"
        f.write_text(content)
        analyzer = ReactionProfileAnalyzer(str(f))
        assert analyzer.config.legend is True
        assert analyzer.config.title == "Test Plot"
        assert len(analyzer.pes_data) == 1

    def test_multiple_paths(self, tmp_path):
        content = """--# PES
0.0; 10.0; 20.0
5.0; 15.0; 25.0
--# LABELS
A; B; C
--# GRAPHS
colors: red; blue
label: Path 1; Path 2
"""
        f = tmp_path / "multi.txt"
        f.write_text(content)
        analyzer = ReactionProfileAnalyzer(str(f))
        assert len(analyzer.pes_data) == 2
        assert len(analyzer.config.colors) == 2

    def test_relative_energies(self, tmp_path):
        content = """--# PES
10.0; -5.0; 3.0
--# LABELS
A; B; C
--# GRAPHS
relative: true
label: Path 1
"""
        f = tmp_path / "rel.txt"
        f.write_text(content)
        analyzer = ReactionProfileAnalyzer(str(f))
        expected = np.cumsum(np.array([10.0, -5.0, 3.0]))
        np.testing.assert_allclose(analyzer.pes_data[0], expected)

    def test_invalid_zero_param_raises(self, tmp_path):
        content = """--# PES
0.0; 10.0
5.0; 15.0
--# LABELS
A; B
--# GRAPHS
zero: 0; 1; 2
label: Path 1; Path 2
"""
        f = tmp_path / "zero_bad.txt"
        f.write_text(content)
        with pytest.raises(ValueError):
            ReactionProfileAnalyzer(str(f))

    def test_single_label_applied_to_all(self, tmp_path):
        content = """--# PES
0.0; 10.0
5.0; 15.0
--# LABELS
X; Y
--# GRAPHS
label: Path 1; Path 2
"""
        f = tmp_path / "label_test.txt"
        f.write_text(content)
        analyzer = ReactionProfileAnalyzer(str(f))
        assert len(analyzer.labels) == len(analyzer.pes_data)

    def test_optional_label(self, tmp_path):
        content = """--# PES
0.0; 10.0; 20.0
--# LABELS
A; B; C
--# GRAPHS
title: Test Without Label
"""
        f = tmp_path / "no_label.txt"
        f.write_text(content)
        analyzer = ReactionProfileAnalyzer(str(f))
        assert analyzer.config.label == [None]
        assert analyzer.config.legend is False
