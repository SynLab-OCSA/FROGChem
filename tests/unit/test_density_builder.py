import numpy as np
import pytest

from frogchem.density_builder import (
    _orbital_letter,
    _solid_harmonic_poly,
    _ang_momentum,
    _guess_atomic_number,
    _NORM_CONST,
    generate_grid,
    build_basis_table,
    eval_basis_1,
    eval_all_basis,
    compute_density_on_points,
    build_density_cube,
    build_spin_density_cube,
    build_orbital_cube,
)


class TestOrbitalLetter:
    def test_s_orbitals(self):
        assert _orbital_letter(0) == 's'

    def test_p_orbitals(self):
        assert _orbital_letter(1) == 'p'
        assert _orbital_letter(2) == 'p'
        assert _orbital_letter(3) == 'p'

    def test_d_orbitals(self):
        assert _orbital_letter(4) == 'd'
        assert _orbital_letter(5) == 'd'
        assert _orbital_letter(6) == 'd'
        assert _orbital_letter(7) == 'd'
        assert _orbital_letter(8) == 'd'

    def test_f_orbitals(self):
        for ac in range(9, 16):
            assert _orbital_letter(ac) == 'f'

    def test_invalid_code(self):
        with pytest.raises(KeyError):
            _orbital_letter(100)


class TestSolidHarmonicPoly:
    def test_s_orbital(self):
        dx, dy, dz = np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])
        result = _solid_harmonic_poly(0, dx, dy, dz)
        np.testing.assert_allclose(result, np.ones_like(dx))

    def test_px(self):
        dx, dy, dz = np.array([1.0, 2.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0])
        result = _solid_harmonic_poly(1, dx, dy, dz)
        np.testing.assert_allclose(result, dx)

    def test_py(self):
        dx = np.array([0.0, 0.0])
        dy = np.array([1.0, 2.0])
        dz = np.array([0.0, 0.0])
        result = _solid_harmonic_poly(2, dx, dy, dz)
        np.testing.assert_allclose(result, np.array([1.0, 2.0]))

    def test_pz(self):
        dx, dz = np.array([0.0, 0.0]), np.array([1.0, 2.0])
        result = _solid_harmonic_poly(3, dx, np.array([0.0, 0.0]), dz)
        np.testing.assert_allclose(result, np.array([1.0, 2.0]))

    def test_dz2(self):
        dx, dy, dz = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 2.0])
        result = _solid_harmonic_poly(4, dx, dy, dz)
        expected = 2.0 * dz**2 - dx**2 - dy**2
        np.testing.assert_allclose(result, expected)

    def test_dxz(self):
        dx, dz = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        result = _solid_harmonic_poly(5, dx, np.zeros_like(dx), dz)
        np.testing.assert_allclose(result, dx * dz)

    def test_dyz(self):
        dy, dz = np.array([1.0, 2.0]), np.array([3.0, 4.0])
        result = _solid_harmonic_poly(6, np.zeros_like(dy), dy, dz)
        np.testing.assert_allclose(result, dy * dz)

    def test_dx2y2(self):
        dx, dy = np.array([2.0, 3.0]), np.array([1.0, 2.0])
        result = _solid_harmonic_poly(7, dx, dy, np.zeros_like(dx))
        np.testing.assert_allclose(result, dx**2 - dy**2)

    def test_dxy(self):
        dx, dy = np.array([2.0, 3.0]), np.array([4.0, 5.0])
        result = _solid_harmonic_poly(8, dx, dy, np.zeros_like(dx))
        np.testing.assert_allclose(result, dx * dy)

    def test_f0(self):
        dx, dy, dz = np.array([1.0]), np.array([1.0]), np.array([1.0])
        result = _solid_harmonic_poly(9, dx, dy, dz)
        expected = dz[0] * (2.0 * dz[0]**2 - 3.0 * dx[0]**2 - 3.0 * dy[0]**2)
        assert np.isclose(result[0], expected)

    def test_invalid_code_returns_ones(self):
        result = _solid_harmonic_poly(99, np.array([1.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(result, np.ones(1))


class TestAngMomentum:
    def test_s(self):
        assert _ang_momentum(0) == 0

    def test_p(self):
        for ac in range(1, 4):
            assert _ang_momentum(ac) == 1

    def test_d(self):
        for ac in range(4, 9):
            assert _ang_momentum(ac) == 2

    def test_f(self):
        for ac in range(9, 16):
            assert _ang_momentum(ac) == 3


class TestGuessAtomicNumber:
    def test_hydrogen(self):
        assert _guess_atomic_number('H') == 1

    def test_carbon(self):
        assert _guess_atomic_number('C') == 6

    def test_oxygen(self):
        assert _guess_atomic_number('O') == 8

    def test_unknown_fallback(self):
        assert _guess_atomic_number('Xx') == 6

    def test_case_insensitive(self):
        assert _guess_atomic_number('h') == 1
        assert _guess_atomic_number('c') == 6


class TestNormConst:
    def test_all_defined(self):
        expected_codes = list(range(16))
        for ac in expected_codes:
            assert ac in _NORM_CONST
            assert _NORM_CONST[ac] > 0

    def test_s_smaller_than_p(self):
        assert _NORM_CONST[0] < _NORM_CONST[1]  # s < px

    def test_px_py_pz_equal(self):
        assert _NORM_CONST[1] == _NORM_CONST[2] == _NORM_CONST[3]


class TestGenerateGrid:
    def test_basic_grid(self):
        geom = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        grid_3d, axes, (nx, ny, nz) = generate_grid(geom, spacing=1.0, padding=0.0)
        assert nx > 0 and ny > 0 and nz > 0
        assert grid_3d.shape[-1] == 3

    def test_with_padding(self):
        geom = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        grid_3d, axes, (nx, ny, nz) = generate_grid(geom, spacing=1.0, padding=2.0)
        assert nx > 0 and ny > 0 and nz > 0

    def test_n_voxels(self):
        geom = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        grid_3d, axes, (nx, ny, nz) = generate_grid(geom, n_voxels=10, padding=2.0)
        assert nx == ny == nz == 10

    def test_grid_contains_atoms(self):
        geom = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        grid_3d, axes, _ = generate_grid(geom, spacing=0.5, padding=0.5)
        points = grid_3d.reshape(-1, 3)
        for atom_pos in geom:
            dists = np.linalg.norm(points - atom_pos, axis=1)
            assert np.any(dists < 0.3)

    def test_uniform_axes(self):
        geom = np.array([[0.0, 0.0, 0.0]])
        grid_3d, axes, (nx, ny, nz) = generate_grid(geom, spacing=0.1, padding=1.0)
        ax, ay, az = axes
        assert len(ax) > 0
        assert np.allclose(np.diff(ax), 0.1)


class TestDensityBuilderRealCalc:
    @pytest.fixture
    def orb_calc(self):
        import os
        from frogchem.orca_parser.calculation import Calculation
        if not os.path.exists("files/orb.out"):
            pytest.skip("files/orb.out not found")
        return Calculation("files/orb.out")

    def test_basis_table_and_eval(self, orb_calc):
        table = build_basis_table(orb_calc)
        assert len(table) == 828
        assert "primitives" in table[0]
        assert "ang_code" in table[0]

        # Test single basis evaluation
        sample_pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        vals = eval_basis_1(table[0], sample_pts)
        assert vals.shape == (2,)

        # Test eval_all_basis
        all_vals = eval_all_basis(table[:5], sample_pts)
        assert all_vals.shape == (5, 2)

    def test_build_orbital_cube(self, orb_calc, tmp_path):
        out_cube = str(tmp_path / "mo_0.cube")
        build_orbital_cube(orb_calc, orbital_index=0, output=out_cube, n_voxels=3)
        assert (tmp_path / "mo_0.cube").exists()

    def test_build_density_cube(self, orb_calc, tmp_path):
        out_cube = str(tmp_path / "density.cube")
        build_density_cube(orb_calc, output=out_cube, n_voxels=3)
        assert (tmp_path / "density.cube").exists()

    def test_build_spin_density_closed_shell(self, orb_calc, tmp_path, capsys):
        # Closed shell mult == 1
        orb_calc.mult = 1
        out_cube = str(tmp_path / "spin_density.cube")
        build_spin_density_cube(orb_calc, output=out_cube, n_voxels=3)
        captured = capsys.readouterr()
        assert "closed-shell" in captured.err.lower()  # library warnings go to stderr

    def test_build_spin_density_open_shell(self, orb_calc, tmp_path):
        # Create an open-shell mock with matching dimensions
        import copy
        mock_open = copy.copy(orb_calc)
        mock_open.mult = 2
        mock_open.alpha_coeff = orb_calc.orbital_coefficients[:, :2]
        mock_open.beta_coeff = orb_calc.orbital_coefficients[:, :2]
        mock_open.alpha_occ = np.array([1.0, 1.0])
        mock_open.beta_occ = np.array([1.0, 0.0])

        out_cube = str(tmp_path / "spin_open.cube")
        build_spin_density_cube(mock_open, output=out_cube, n_voxels=3)
        assert (tmp_path / "spin_open.cube").exists()

    def test_compute_difference_density(self, orb_calc):
        from frogchem.density_builder import (
            compute_difference_density_on_points,
            generate_grid, eval_all_basis, _AU,
        )
        from frogchem.math_geometry import calculate_centroids

        homo = orb_calc.homo
        lumo = orb_calc.lumo
        assert homo is not None and lumo is not None

        geom_bohr = orb_calc.geom * _AU
        table = build_basis_table(orb_calc, geom_bohr)
        grid_3d, axes, (nx, ny, nz) = generate_grid(
            geom_bohr, n_voxels=20, padding=3.0 * _AU,
        )
        points = grid_3d.reshape(-1, 3)
        bf = eval_all_basis(table, points)

        transitions = [(homo, lumo, 1.0)]
        delta_rho = compute_difference_density_on_points(
            orb_calc, transitions, points, bf_matrix=bf,
        )

        # Δρ should have both positive and negative regions
        assert np.any(delta_rho > 0), "Δρ should have positive (particle) regions"
        assert np.any(delta_rho < 0), "Δρ should have negative (hole) regions"
        # The integrated Δρ should be approximately 0 (charge conservation)
        ax, ay, az = axes
        dvol = abs((ax[1]-ax[0]) * (ay[1]-ay[0]) * (az[1]-az[0]))
        integral = delta_rho.sum() * dvol
        assert abs(integral) < 0.5, f"Integral of Δρ should be near 0, got {integral:.4f}"

    def test_compute_difference_density_skips_tiny_weights(self, orb_calc):
        from frogchem.density_builder import (
            compute_difference_density_on_points,
            generate_grid, eval_all_basis, _AU,
        )
        homo = orb_calc.homo
        lumo = orb_calc.lumo

        geom_bohr = orb_calc.geom * _AU
        table = build_basis_table(orb_calc, geom_bohr)
        grid_3d, axes, (nx, ny, nz) = generate_grid(
            geom_bohr, n_voxels=5, padding=3.0 * _AU,
        )
        points = grid_3d.reshape(-1, 3)
        bf = eval_all_basis(table, points)

        # Tiny weight → should be all zeros
        transitions = [(homo, lumo, 1e-8)]
        delta_rho = compute_difference_density_on_points(
            orb_calc, transitions, points, bf_matrix=bf,
        )
        np.testing.assert_allclose(delta_rho, 0.0)


class TestMakeCubeData:
    def test_returns_valid_dict(self):
        from frogchem.density_builder import _make_cube_data

        class FakeCalc:
            atoms = ['H', 'H']
            geom = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        data_3d = np.ones((3, 3, 3))
        axes = (np.linspace(0, 1, 3), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
        result = _make_cube_data(data_3d, axes, FakeCalc(), title='t', comment='c')

        assert result['title'] == 't'
        assert result['comment'] == 'c'
        assert result['natoms'] == 2
        assert result['grid_size'] == (3, 3, 3)
        assert result['units'] == 'Bohr'
        assert result['data'] is data_3d
        assert len(result['atoms']) == 2
        assert result['voxel_vectors'].shape == (3, 3)


