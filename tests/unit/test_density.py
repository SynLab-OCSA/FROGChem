import pytest
import numpy as np
from frogchem.grid import (
    subtract_grids,
    add_grids,
    integrate_density_voronoi,
)
from frogchem.math_geometry import calculate_centroids


@pytest.fixture
def alpha_cube():
    nx, ny, nz = 4, 4, 4
    data = np.ones((nx, ny, nz)) * 2.0
    return {
        'title': 'Alpha density',
        'comment': 'Test',
        'natoms': 1,
        'origin': np.array([0.0, 0.0, 0.0]),
        'grid_size': (nx, ny, nz),
        'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        'units': 'Bohr',
        'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.array([1.5, 1.5, 1.5])}],
        'data': data,
        'data_range': (2.0, 2.0),
    }


@pytest.fixture
def beta_cube():
    nx, ny, nz = 4, 4, 4
    data = np.ones((nx, ny, nz)) * 1.0
    return {
        'title': 'Beta density',
        'comment': 'Test',
        'natoms': 1,
        'origin': np.array([0.0, 0.0, 0.0]),
        'grid_size': (nx, ny, nz),
        'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        'units': 'Bohr',
        'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.array([1.5, 1.5, 1.5])}],
        'data': data,
        'data_range': (1.0, 1.0),
    }


class TestSpinDensity:
    def test_spin_density_calculation(self, alpha_cube, beta_cube):
        spin = subtract_grids(alpha_cube, beta_cube)
        assert spin['title'] == 'Difference'
        np.testing.assert_allclose(spin['data'], -np.ones((4, 4, 4)))
        assert spin['grid_size'] == (4, 4, 4)

    def test_spin_density_grid_mismatch(self, alpha_cube, beta_cube):
        beta_cube['grid_size'] = (3, 3, 3)
        beta_cube['data'] = np.ones((3, 3, 3))
        with pytest.raises(ValueError):
            subtract_grids(alpha_cube, beta_cube)

    def test_spin_density_zero(self, alpha_cube):
        spin = subtract_grids(alpha_cube, alpha_cube)
        np.testing.assert_allclose(spin['data'], np.zeros((4, 4, 4)))


class TestTotalDensity:
    def test_total_density_calculation(self, alpha_cube, beta_cube):
        total = add_grids(alpha_cube, beta_cube)
        assert total['title'] == 'Addition'
        np.testing.assert_allclose(total['data'], np.ones((4, 4, 4)) * 3.0)

    def test_total_density_grid_mismatch(self, alpha_cube, beta_cube):
        beta_cube['grid_size'] = (3, 3, 3)
        beta_cube['data'] = np.ones((3, 3, 3))
        with pytest.raises(ValueError):
            add_grids(alpha_cube, beta_cube)


class TestComputeCentroids:
    def test_delegates_to_math_geometry(self):
        cube = {
            'title': 'test',
            'comment': '',
            'natoms': 0,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (4, 4, 4),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Bohr',
            'atoms': [],
            'data': np.ones((4, 4, 4)),
            'data_range': (1.0, 1.0),
        }
        result = calculate_centroids(cube)
        assert 'positive_centroid' in result
        assert 'negative_centroid' in result


class TestComputeAtomIntegrals:
    def test_delegates_to_grid(self):
        nx, ny, nz = 4, 4, 4
        data = np.ones((nx, ny, nz)) * 0.5
        cube = {
            'title': 'test',
            'comment': '',
            'natoms': 1,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (nx, ny, nz),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Bohr',
            'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.array([1.5, 1.5, 1.5])}],
            'data': data,
            'data_range': (0.5, 0.5),
        }
        integrals = integrate_density_voronoi(cube)
        assert len(integrals) == 1
        assert np.isclose(integrals[0], 32.0)
