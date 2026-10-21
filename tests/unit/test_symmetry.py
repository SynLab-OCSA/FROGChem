import numpy as np
import pytest

rdkit = pytest.importorskip("rdkit", reason="rdkit not installed")

from frogchem.symmetry import (
    normalize,
    get_rotation_matrix,
    get_reflection_matrix,
    calculate_center_of_mass,
    get_inertia_tensor,
    align_molecule,
)


class TestNormalize:
    def test_unit_vector(self):
        v = np.array([1.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, v)

    def test_arbitrary_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        result = normalize(v)
        assert np.isclose(np.linalg.norm(result), 1.0)
        np.testing.assert_allclose(result, np.array([0.6, 0.8, 0.0]))

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_allclose(result, v)


class TestRotationMatrix:
    def test_identity_rotation(self):
        mat = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), 0.0)
        np.testing.assert_allclose(mat, np.eye(3), atol=1e-10)

    def test_90_deg_around_z(self):
        mat = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ])
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_180_deg_around_z(self):
        mat = get_rotation_matrix(np.array([0.0, 0.0, 1.0]), np.pi)
        expected = np.diag([-1, -1, 1])
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_rotation_is_orthogonal(self):
        mat = get_rotation_matrix(np.array([1.0, 2.0, 3.0]), 0.5)
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-10)

    def test_determinant_is_one(self):
        mat = get_rotation_matrix(np.array([1.0, 1.0, 1.0]), 1.0)
        assert np.isclose(np.linalg.det(mat), 1.0, atol=1e-10)


class TestReflectionMatrix:
    def test_reflection_xy_plane(self):
        mat = get_reflection_matrix(np.array([0.0, 0.0, 1.0]))
        expected = np.diag([1, 1, -1])
        np.testing.assert_allclose(mat, expected, atol=1e-10)

    def test_reflection_is_involutive(self):
        normal = np.array([1.0, 2.0, 3.0])
        mat = get_reflection_matrix(normal)
        np.testing.assert_allclose(mat @ mat, np.eye(3), atol=1e-10)

    def test_reflection_determinant(self):
        mat = get_reflection_matrix(np.array([1.0, 0.0, 0.0]))
        assert np.isclose(np.linalg.det(mat), -1.0, atol=1e-10)


class TestCalculateCenterOfMass:
    def test_single_atom(self):
        symbols = np.array(["C"])
        positions = np.array([[1.0, 2.0, 3.0]])
        com, mass = calculate_center_of_mass(symbols, positions)
        np.testing.assert_allclose(com, [1.0, 2.0, 3.0])
        assert mass > 0

    def test_symmetric_diatomic(self):
        symbols = np.array(["C", "C"])
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        com, mass = calculate_center_of_mass(symbols, positions)
        np.testing.assert_allclose(com, [0.0, 0.0, 0.0])

    def test_asymmetric_diatomic(self):
        symbols = np.array(["H", "O"])
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        com, mass = calculate_center_of_mass(symbols, positions)
        assert com[2] > 0  # COM closer to O (heavier)


class TestGetInertiaTensor:
    def test_returns_eigenvalues_and_vectors(self):
        symbols = ["C", "C"]
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        evals, evecs = get_inertia_tensor(symbols, positions)
        assert len(evals) == 3
        assert evecs.shape == (3, 3)

    def test_linear_molecule(self):
        symbols = ["C", "C"]
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        evals, evecs = get_inertia_tensor(symbols, positions)
        assert np.isclose(evals[0], 0.0, atol=1e-10)


class TestAlignMolecule:
    def test_diatomic(self):
        symbols = ["C", "C"]
        positions = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        aligned = align_molecule(symbols, positions)
        assert aligned.shape == (2, 3)

    def test_output_centered(self):
        symbols = ["H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        aligned = align_molecule(symbols, positions)
        com = np.mean(aligned, axis=0)
        np.testing.assert_allclose(com, np.zeros(3), atol=1e-10)
