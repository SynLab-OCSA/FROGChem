import numpy as np
import pytest
from frogchem.math_geometry import (
    calculate_distance,
    calculate_angle,
    calculate_dihedral,
    calc_rmsd,
    calculate_centroids,
    normalize,
    get_rotation_matrix,
    get_reflection_matrix,
    calculate_center_of_mass,
    get_inertia_tensor,
    align_molecule,
)


class TestCalculateDistance:
    def test_euclidean_3_4_5(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        assert calculate_distance(p1, p2) == 5.0

    def test_zero_distance(self):
        p = np.array([1.0, 2.0, 3.0])
        assert calculate_distance(p, p) == 0.0

    def test_negative_coords(self):
        p1 = np.array([-1.0, -1.0, 0.0])
        p2 = np.array([1.0, 1.0, 0.0])
        assert np.isclose(calculate_distance(p1, p2), np.sqrt(8.0))

    def test_3d_distance(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 1.0, 1.0])
        assert np.isclose(calculate_distance(p1, p2), np.sqrt(3.0))


class TestCalculateAngle:
    def test_right_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        assert np.isclose(calculate_angle(p1, center, p3), 90.0)

    def test_straight_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        p3 = np.array([-1.0, 0.0, 0.0])
        assert np.isclose(calculate_angle(p1, center, p3), 180.0)

    def test_zero_angle(self):
        p1 = np.array([1.0, 0.0, 0.0])
        center = np.array([0.0, 0.0, 0.0])
        p3 = np.array([2.0, 0.0, 0.0])
        assert np.isclose(calculate_angle(p1, center, p3), 0.0)

    def test_acute_angle(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        assert np.isclose(calculate_angle(a, b, c), 90.0)


class TestCalculateDihedral:
    def test_cis_planar(self):
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([0, 1, 0])
        p4 = np.array([1, 1, 0])
        assert np.isclose(calculate_dihedral(p1, p2, p3, p4), 0.0, atol=1e-5)

    def test_trans_planar(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([-1.0, 1.0, 0.0])
        assert np.isclose(calculate_dihedral(p1, p2, p3, p4), 180.0, atol=1e-5)

    def test_orthogonal(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([0.0, 1.0, 1.0])
        result = calculate_dihedral(p1, p2, p3, p4)
        assert np.isclose(abs(result), 90.0, atol=1e-5)

    def test_symmetry_of_staggered(self):
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.0])
        p3 = np.array([0.0, 1.0, 0.0])
        p4 = np.array([0.0, 1.0, -1.0])
        result = calculate_dihedral(p1, p2, p3, p4)
        assert np.isclose(abs(result), 90.0, atol=1e-5)


class TestCalcRMSD:
    def test_identical(self):
        eig1 = np.array([1.0, 2.0, 3.0])
        eig2 = np.array([1.0, 2.0, 3.0])
        assert calc_rmsd(eig1, eig2) == 0.0

    def test_single_element(self):
        eig1 = np.array([0.0])
        eig2 = np.array([2.0])
        assert calc_rmsd(eig1, eig2) == 2.0

    def test_three_elements(self):
        eig1 = np.array([0.0, 0.0, 0.0])
        eig2 = np.array([3.0, 4.0, 0.0])
        assert np.isclose(calc_rmsd(eig1, eig2), np.sqrt(25.0 / 3))

    def test_negative_values(self):
        eig1 = np.array([-1.0, -1.0])
        eig2 = np.array([1.0, 1.0])
        assert calc_rmsd(eig1, eig2) == 2.0


class TestCalculateCentroids:
    @pytest.fixture
    def uniform_cube(self):
        nx, ny, nz = 4, 4, 4
        data = np.ones((nx, ny, nz))
        data[1:3, 1:3, 1:3] = 5.0
        return {
            'title': 'test',
            'comment': '',
            'natoms': 0,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (nx, ny, nz),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Bohr',
            'atoms': [],
            'data': data,
            'data_range': (1.0, 5.0),
        }

    @pytest.fixture
    def bipolar_cube(self):
        nx, ny, nz = 4, 4, 4
        data = np.ones((nx, ny, nz))
        data[:2, :, :] = -2.0
        return {
            'title': 'test',
            'comment': '',
            'natoms': 0,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (nx, ny, nz),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Bohr',
            'atoms': [],
            'data': data,
            'data_range': (-2.0, 1.0),
        }

    def test_centroid_positive_region(self, uniform_cube):
        result = calculate_centroids(uniform_cube, threshold_percentile=99.5)
        assert result['positive_centroid'] is not None
        assert result['negative_centroid'] is None
        assert result['positive_charge'] > 0

    def test_centroid_both_signs(self, bipolar_cube):
        result = calculate_centroids(bipolar_cube, threshold_percentile=99.5)
        assert result['positive_centroid'] is not None
        assert result['negative_centroid'] is not None
        assert result['charge_transfer_distance'] is not None
        assert result['dipole_moment'] is not None

    def test_centroid_key_structure(self, bipolar_cube):
        result = calculate_centroids(bipolar_cube, threshold_percentile=99.5)
        assert 'data' in result
        assert 'positive_centroid' in result
        assert 'negative_centroid' in result
        assert 'positive_charge' in result
        assert 'negative_charge' in result
        assert 'charge_transfer_distance' in result
        assert 'dipole_moment' in result
        assert 'thresholds' in result

    def test_zero_data(self):
        cube = {
            'title': 'zero',
            'comment': '',
            'natoms': 0,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (2, 2, 2),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Bohr',
            'atoms': [],
            'data': np.zeros((2, 2, 2)),
            'data_range': (0.0, 0.0),
        }
        result = calculate_centroids(cube)
        assert result['positive_centroid'] is None
        assert result['negative_centroid'] is None

    def test_angstrom_conversion(self):
        nx, ny, nz = 4, 4, 4
        data = np.ones((nx, ny, nz))
        data[1:3, 1:3, 1:3] = 5.0
        cube = {
            'title': 'test',
            'comment': '',
            'natoms': 0,
            'origin': np.array([0.0, 0.0, 0.0]),
            'grid_size': (nx, ny, nz),
            'voxel_vectors': np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            'units': 'Angstrom',
            'atoms': [],
            'data': data,
            'data_range': (1.0, 5.0),
        }
        result_ang = calculate_centroids(cube, threshold_percentile=99.5)
        assert result_ang['positive_centroid'] is not None


class Test3DTransformations:
    def test_normalize(self):
        v = np.array([3.0, 4.0, 0.0])
        normed = normalize(v)
        assert np.isclose(np.linalg.norm(normed), 1.0)
        assert np.allclose(normalize(np.array([0.0, 0.0, 0.0])), np.array([0.0, 0.0, 0.0]))

    def test_rotation_matrix(self):
        rot = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        assert np.isclose(np.linalg.det(rot), 1.0)
        v = np.array([1.0, 0.0, 0.0])
        v_rot = np.dot(rot, v)
        assert np.allclose(v_rot, [0.0, 1.0, 0.0], atol=1e-7)

    def test_reflection_matrix(self):
        ref = get_reflection_matrix(np.array([0.0, 0.0, 1.0]))
        assert np.isclose(np.linalg.det(ref), -1.0)
        assert np.allclose(np.dot(ref, ref), np.eye(3))

    def test_center_of_mass_and_align(self):
        symbols = ['H', 'H']
        positions = np.array([[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]])
        com, mass = calculate_center_of_mass(symbols, positions)
        assert np.allclose(com, [0.0, 0.0, 0.0])
        assert mass > 0

        evals, evecs = get_inertia_tensor(symbols, positions)
        assert len(evals) == 3

        aligned = align_molecule(symbols, positions)
        com_aligned, _ = calculate_center_of_mass(symbols, aligned)
        assert np.allclose(com_aligned, [0.0, 0.0, 0.0], atol=1e-7)
