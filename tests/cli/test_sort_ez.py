"""Tests for sort_ez CLI."""
import os
import numpy as np
import pytest

from frogchem.cli.sort_ez import SortEZ
from frogchem.math_geometry import calculate_dihedral
from frogchem.io import write_xyz


def _make_dihedral_xyz(tmp_path, angles_deg):
    """Build a multi-structure XYZ file where atoms 0-1-2-3 have the given dihedral angles."""
    atoms_list = []
    positions_list = []
    comments = []

    for i, angle in enumerate(angles_deg):
        rad = np.radians(angle)
        positions = np.array([
            [0.0, 0.0, 1.0],   # atom 0
            [0.0, 0.0, 0.0],   # atom 1
            [1.0, 0.0, 0.0],   # atom 2
            [1.0, np.sin(rad), np.cos(rad)],  # atom 3
        ])
        atoms_list.append(np.array(['C', 'C', 'C', 'C']))
        positions_list.append(positions)
        comments.append(f"dihedral={angle}")

    fname = os.path.join(str(tmp_path), "test_ensemble.xyz")
    write_xyz(fname, atoms_list, positions_list, comments)
    return fname


class TestSortEZ:
    """Tests for the frog-sort-ez CLI tool."""

    def test_ez_default_threshold(self, tmp_path):
        """Conformers with |dihedral| < 90 are Z, others are E."""
        fname = _make_dihedral_xyz(tmp_path, [30, 150, -45, -120, 0, 180])
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3'])

        from frogchem.filetypes import load_structures
        structs = load_structures(fname)
        labels = []
        for s in structs:
            pos = s[1]
            d = calculate_dihedral(pos[0], pos[1], pos[2], pos[3])
            labels.append('Z' if abs(d) < 90 else 'E')

        assert labels.count('Z') == 3
        assert labels.count('E') == 3

    def test_cis_trans_notation(self, tmp_path):
        """With --notation cis-trans, labels are cis and trans."""
        fname = _make_dihedral_xyz(tmp_path, [20, 160])
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '--notation', 'cis-trans'])

    def test_scis_strans_notation(self, tmp_path):
        """With --notation scis-strans, labels are s-cis and s-trans."""
        fname = _make_dihedral_xyz(tmp_path, [20, 160])
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '--notation', 'scis-strans'])

    def test_custom_threshold(self, tmp_path):
        """A custom threshold changes the boundary."""
        angles = [30, 60, -30, -120]
        fname = _make_dihedral_xyz(tmp_path, angles)
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '-t', '45'])

        from frogchem.filetypes import load_structures
        structs = load_structures(fname)
        labels = []
        for s in structs:
            pos = s[1]
            d = calculate_dihedral(pos[0], pos[1], pos[2], pos[3])
            labels.append('Z' if abs(d) < 45 else 'E')

        assert labels.count('Z') == 2
        assert labels.count('E') == 2

    def test_write_output_files(self, tmp_path):
        """With -w, output XYZ files are generated."""
        angles = [30, 150]
        fname = _make_dihedral_xyz(tmp_path, angles)
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '-w'])

        base = os.path.splitext(fname)[0]
        assert os.path.isfile(f"{base}_Z.xyz")
        assert os.path.isfile(f"{base}_E.xyz")

    def test_tolerance_mode(self, tmp_path):
        """Tolerance defines windows around 0 and 180; intermediate angles become 'other'."""
        angles = [20, 160, 90, -25, -155]
        fname = _make_dihedral_xyz(tmp_path, angles)
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '--tolerance', '30'])

        from frogchem.filetypes import load_structures
        structs = load_structures(fname)
        labels = []
        for s in structs:
            pos = s[1]
            d = calculate_dihedral(pos[0], pos[1], pos[2], pos[3])
            abs_d = abs(d)
            if abs_d <= 30:
                labels.append('Z')
            elif abs_d >= 150:
                labels.append('E')
            else:
                labels.append('other')

        assert labels.count('Z') == 2
        assert labels.count('E') == 2
        assert labels.count('other') == 1

    def test_tolerance_write(self, tmp_path):
        """Tolerance mode writes Z, E, and other XYZ files when requested."""
        angles = [15, 90, 165]
        fname = _make_dihedral_xyz(tmp_path, angles)
        tool = SortEZ()
        tool.main([fname, '-i', '0', '1', '2', '3', '--tolerance', '30', '-w'])

        base = os.path.splitext(fname)[0]
        assert os.path.isfile(f"{base}_Z.xyz")
        assert os.path.isfile(f"{base}_E.xyz")
        assert os.path.isfile(f"{base}_other.xyz")

    def test_missing_file_exits(self, tmp_path):
        """Referencing a nonexistent file exits with an error."""
        with pytest.raises(SystemExit):
            tool = SortEZ()
            tool.main(["nonexistent.xyz", '-i', '0', '1', '2', '3'])
