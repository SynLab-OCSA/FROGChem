import numpy as np
import pytest
from frogchem.grid import add_grids, subtract_grids, multiply_grid


@pytest.fixture
def grid_a():
    return {
        'grid_size': (3, 2, 2),
        'origin': np.array([0.0, 0.0, 0.0]),
        'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        'data': np.ones((3, 2, 2)) * 2.0,
        'title': 'A',
        'comment': 'Grid A',
        'data_range': (2.0, 2.0),
        'natoms': 0,
        'units': 'Bohr',
        'atoms': [],
    }


@pytest.fixture
def grid_b():
    return {
        'grid_size': (3, 2, 2),
        'origin': np.array([0.0, 0.0, 0.0]),
        'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        'data': np.ones((3, 2, 2)) * 3.0,
        'title': 'B',
        'comment': 'Grid B',
        'data_range': (3.0, 3.0),
        'natoms': 0,
        'units': 'Bohr',
        'atoms': [],
    }


class TestAddGrids:
    def test_add_two_grids(self, grid_a, grid_b):
        result = add_grids(grid_a, grid_b)
        np.testing.assert_allclose(result['data'], np.ones((3, 2, 2)) * 5.0)
        assert result['title'] == 'Addition'
        assert result['grid_size'] == (3, 2, 2)

    def test_add_self(self, grid_a):
        result = add_grids(grid_a, grid_a)
        np.testing.assert_allclose(result['data'], np.ones((3, 2, 2)) * 4.0)

    def test_add_grid_size_mismatch(self, grid_a, grid_b):
        grid_b['grid_size'] = (4, 4, 4)
        grid_b['data'] = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="Grid sizes must match"):
            add_grids(grid_a, grid_b)

    def test_add_does_not_mutate_input(self, grid_a, grid_b):
        data_before = grid_a['data'].copy()
        add_grids(grid_a, grid_b)
        np.testing.assert_array_equal(grid_a['data'], data_before)


class TestSubtractGrids:
    def test_subtract_basic(self, grid_a, grid_b):
        result = subtract_grids(grid_a, grid_b)
        np.testing.assert_allclose(result['data'], np.ones((3, 2, 2)))
        assert result['title'] == 'Difference'

    def test_subtract_negative_result(self, grid_a, grid_b):
        result = subtract_grids(grid_b, grid_a)
        np.testing.assert_allclose(result['data'], -np.ones((3, 2, 2)))

    def test_subtract_zero(self, grid_a):
        result = subtract_grids(grid_a, grid_a)
        np.testing.assert_allclose(result['data'], np.zeros((3, 2, 2)))

    def test_subtract_keeps_metadata(self, grid_a, grid_b):
        result = subtract_grids(grid_a, grid_b)
        assert result['comment'] == 'Difference of grids'

    def test_subtract_grid_size_mismatch(self, grid_a, grid_b):
        grid_b['grid_size'] = (2, 2, 2)
        grid_b['data'] = np.ones((2, 2, 2))
        with pytest.raises(ValueError, match="Grid sizes must match"):
            subtract_grids(grid_a, grid_b)


class TestMultiplyGrid:
    def test_multiply_by_positive(self, grid_a):
        result = multiply_grid(grid_a, 3.0)
        np.testing.assert_allclose(result['data'], np.ones((3, 2, 2)) * 6.0)

    def test_multiply_by_zero(self, grid_a):
        result = multiply_grid(grid_a, 0.0)
        np.testing.assert_allclose(result['data'], np.zeros((3, 2, 2)))

    def test_multiply_by_negative(self, grid_a):
        result = multiply_grid(grid_a, -1.0)
        np.testing.assert_allclose(result['data'], -grid_a['data'])

    def test_multiply_updates_data_range(self, grid_a):
        result = multiply_grid(grid_a, 5.0)
        assert result['data_range'] == (10.0, 10.0)

    def test_multiply_metadata(self, grid_a):
        result = multiply_grid(grid_a, 2.0)
        assert result['title'] == 'Multiplication'
