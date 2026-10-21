"""frog-sym: hydrogens are included by default; rotor classification tolerance."""

import json

import numpy as np
import pytest

from frogchem.cli.detect_symmetry import DetectSymmetry, rotor_type
from frogchem.io import write_xyz


@pytest.fixture
def water_xyz(tmp_path):
    f = tmp_path / "water.xyz"
    write_xyz(str(f), [["O", "H", "H"]],
              [np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])])
    return str(f)


def _point_group(capsys, *argv):
    DetectSymmetry().main([*argv, "-t", "0.1", "--format", "json"])
    out = capsys.readouterr().out
    return out


def test_default_includes_hydrogens(water_xyz, capsys):
    out = _point_group(capsys, water_xyz)
    assert "C2v" in out


def test_include_h_flag_matches_default(water_xyz, capsys):
    assert "C2v" in _point_group(capsys, water_xyz, "--include-h")


def test_exclude_h_with_single_heavy_atom_falls_back_to_all_atoms(water_xyz, capsys):
    # one heavy atom is no framework: all atoms are used (PointGroupDetector docstring)
    out = _point_group(capsys, water_xyz, "--exclude-h")
    assert "C2v" in out


@pytest.mark.parametrize("evals, expected", [
    ([0.0, 1.7, 1.7], "Linear"),
    ([3.2, 3.2, 3.2], "Spherical Top"),
    ([3.0, 3.0, 5.9], "Symmetric Top"),        # oblate (e.g. NH3-like)
    ([1.0, 9.0, 9.0], "Symmetric Top"),        # prolate
    ([10.0, 10.8, 20.8], "Asymmetric Top"),    # 8% apart: not degenerate
    ([0.61, 1.15, 1.77], "Asymmetric Top"),    # water
])
def test_rotor_type(evals, expected):
    assert rotor_type(np.array(evals)) == expected
