import os
import numpy as np
import pytest

from frogchem.solvent_surface import (
    _get_vdw_radius,
    _sdf_from_boolean,
    _grid_params,
    compute_sas,
    compute_ses,
    build_sas_cube,
    build_ses_cube,
)
from frogchem.io import parse_gaussian_cube


class TestSolventSurface:
    def test_get_vdw_radius(self):
        r_h = _get_vdw_radius('H', offset=0.4)
        assert r_h > 0.4
        r_c = _get_vdw_radius('C', offset=0.5)
        assert r_c > r_h

    def test_sdf_from_boolean(self):
        # 10x10x10 boolean grid with True at center
        grid = np.zeros((10, 10, 10), dtype=bool)
        grid[4:7, 4:7, 4:7] = True
        sdf = _sdf_from_boolean(grid, spacing=0.5)
        # Inside voxels should be negative
        assert sdf[5, 5, 5] < 0
        # Outside corners should be positive
        assert sdf[0, 0, 0] > 0

    def test_grid_params(self):
        atoms = ['C', 'H']
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        axes, dims = _grid_params(atoms, coords, spacing=0.5, padding=1.0, n_voxels=None)
        assert len(axes) == 3
        assert len(dims) == 3
        assert all(d > 0 for d in dims)

    def test_compute_sas(self):
        atoms = ['O', 'H', 'H']
        coords = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586]
        ])
        phi, axes, dims = compute_sas(
            atoms, coords, spacing=0.5, padding=2.0, n_voxels=15, probe_radius=1.4
        )
        assert phi.shape == tuple(dims)
        # Point near oxygen atom center should be negative (inside SAS)
        # Find closest grid voxel to oxygen (0,0,0)
        idx_x = np.argmin(np.abs(axes[0]))
        idx_y = np.argmin(np.abs(axes[1]))
        idx_z = np.argmin(np.abs(axes[2]))
        assert phi[idx_x, idx_y, idx_z] < 0

    def test_compute_ses(self):
        atoms = ['C']
        coords = np.array([[0.0, 0.0, 0.0]])
        phi, axes, dims = compute_ses(
            atoms, coords, spacing=0.5, padding=2.0, n_voxels=15, probe_radius=1.4
        )
        assert phi.shape == tuple(dims)
        # At origin, should be negative
        idx_x = np.argmin(np.abs(axes[0]))
        idx_y = np.argmin(np.abs(axes[1]))
        idx_z = np.argmin(np.abs(axes[2]))
        assert phi[idx_x, idx_y, idx_z] < 0
        # Far boundary should be positive
        assert phi[0, 0, 0] > 0

    def test_build_sas_and_ses_cube(self, tmp_path):
        atoms = ['H', 'H']
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        sas_file = str(tmp_path / "sas.cube")
        ses_file = str(tmp_path / "ses.cube")

        build_sas_cube(atoms, coords, output=sas_file, spacing=0.5, padding=1.5, n_voxels=12)
        assert os.path.exists(sas_file)
        parsed_sas = parse_gaussian_cube(sas_file)
        assert parsed_sas['title'] == 'Solvent Accessible Surface'
        assert parsed_sas['natoms'] == 2

        build_ses_cube(atoms, coords, output=ses_file, spacing=0.5, padding=1.5, n_voxels=12)
        assert os.path.exists(ses_file)
        parsed_ses = parse_gaussian_cube(ses_file)
        assert parsed_ses['title'] == 'Solvent Excluded Surface'
        assert parsed_ses['natoms'] == 2
