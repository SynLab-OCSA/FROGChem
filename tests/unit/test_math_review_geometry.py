"""Regression tests for the geometry / symmetry / sterics / surfaces review fixes.

Each test pins one confirmed finding (expected values from analytic results or
independent implementations).
"""

import numpy as np
import pytest

from frogchem.math_geometry import (
    calculate_angle,
    calculate_dihedral,
    calculate_centroids,
    calc_distance_matrix,
    get_inertia_tensor,
    align_molecule,
)
from frogchem.symmetry import detect_point_group
from frogchem.steric_maps import (
    compute_buried_volume,
    compute_steric_map,
    orient_positions,
    get_sambvca_radii,
)
from frogchem.solvent_surface import compute_ses, _get_vdw_radius, _axes_spacing
from frogchem.chemistry import BOHR_TO_ANGSTROM


# ---------------------------------------------------------------- helpers

def _rot(seed=7):
    """Deterministic random proper rotation matrix (QR of a Gaussian matrix)."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


A_CH = 1.09 / np.sqrt(3)
CH4 = (['C', 'H', 'H', 'H', 'H'],
       np.array([[0, 0, 0], [A_CH, A_CH, A_CH], [A_CH, -A_CH, -A_CH],
                 [-A_CH, A_CH, -A_CH], [-A_CH, -A_CH, A_CH]]))
H2O = (['O', 'H', 'H'], np.array([[0, 0, 0.117], [0, 0.757, -0.469], [0, -0.757, -0.469]]))
NH3 = (['N', 'H', 'H', 'H'], np.array([[0, 0, 0.116], [0, 0.938, -0.271],
                                       [0.812, -0.469, -0.271], [-0.812, -0.469, -0.271]]))
ALLENE_F = (['C', 'C', 'C', 'F', 'F', 'F', 'F'],
            np.array([[0, 0, 0], [0, 0, 1.31], [0, 0, -1.31], [0.93, 0, 1.87],
                      [-0.93, 0, 1.87], [0, 0.93, -1.87], [0, -0.93, -1.87]]))


def _s4_orbit(p):
    pts, q = [], np.array(p, float)
    for _ in range(4):
        pts.append(q.copy())
        q = np.array([q[1], -q[0], -q[2]])  # S4 about z
    return pts


# ---------------------------------------------------------------- symmetry

def test_default_point_group_includes_hydrogens():
    # was: H2O/NH3 -> C_inf_v, CH4 -> D_inf_h with the default exclude_h=True
    assert detect_point_group(*H2O)['sch_symbol'] == 'C2v'
    assert detect_point_group(*NH3)['sch_symbol'] == 'C3v'
    assert detect_point_group(*CH4)['sch_symbol'] == 'Td'


def test_exclude_h_single_heavy_atom_falls_back_to_all_atoms():
    assert detect_point_group(*H2O, exclude_h=True)['sch_symbol'] == 'C2v'
    assert detect_point_group(*CH4, exclude_h=True)['sch_symbol'] == 'Td'


def test_exclude_h_framework_centred_on_framework_com():
    # H2O2 framework (O-O) is linear about its own COM -> D_inf_h (was 'D8h')
    s = ['O', 'O', 'H', 'H']
    x = np.array([[0, 0.7375, -0.06], [0, -0.7375, -0.06], [0.8, 0.9, 0.48], [-0.8, -0.9, 0.48]])
    assert detect_point_group(s, x, exclude_h=True)['sch_symbol'] == 'D_inf_h'
    assert detect_point_group(s, x)['sch_symbol'] == 'C2'


def test_point_group_is_rotation_invariant():
    # was: D2d in the input frame, C2 after rotation (candidate dedupe at 11.5 deg)
    s, x = ALLENE_F
    assert detect_point_group(s, x)['sch_symbol'] == 'D2d'
    assert detect_point_group(s, x @ _rot().T)['sch_symbol'] == 'D2d'


def test_improper_axis_s4_detected():
    # two inequivalent S4 orbits: only {E, S4, C2, S4^3} -> S4 (was C2)
    x = np.vstack([[[0, 0, 0]], _s4_orbit([1.4, 0.3, 0.6]), _s4_orbit([0.4, 2.1, -0.9])])
    s = ['Si'] + ['Cl'] * 4 + ['F'] * 4
    assert detect_point_group(s, x)['sch_symbol'] == 'S4'


def test_d2d_disphenoid_principal_axis_is_s4():
    # one S4 orbit is a D2d disphenoid (was D2)
    x = np.vstack([_s4_orbit([1.0, 0.3, 0.5]), [[0, 0, 0]]])
    assert detect_point_group(['Cl'] * 4 + ['C'], x)['sch_symbol'] == 'D2d'


def test_single_atom_is_kh():
    assert detect_point_group(['Ar'], np.zeros((1, 3)))['sch_symbol'] == 'Kh'


def test_s4_equivalent_atoms_mapped():
    x = np.vstack([[[0, 0, 0]], _s4_orbit([1.4, 0.3, 0.6]), _s4_orbit([0.4, 2.1, -0.9])])
    s = ['Si'] + ['Cl'] * 4 + ['F'] * 4
    eq = detect_point_group(s, x)['equivalencies']
    assert eq['Cl'] == [[1, 2, 3, 4]] and eq['F'] == [[5, 6, 7, 8]]


# ---------------------------------------------------------------- inertia / alignment

def test_align_molecule_is_rigid_for_spherical_top():
    # was: np.linalg.eig -> non-orthogonal eigenvectors, CH4 distorted by 0.418 A
    s, x = CH4
    y = x @ _rot(3).T + np.array([1.0, -2.0, 3.0])
    _, v = get_inertia_tensor(s, y)
    assert np.allclose(v.T @ v, np.eye(3), atol=1e-10)
    al = align_molecule(s, y)
    d0 = np.linalg.norm(y[:, None] - y[None], axis=-1)
    d1 = np.linalg.norm(al[:, None] - al[None], axis=-1)
    assert np.abs(d0 - d1).max() < 1e-10


# ---------------------------------------------------------------- internal coordinates

def test_angle_precise_near_180():
    a, b = np.array([-1.0, 0, 0]), np.zeros(3)
    c = np.array([1.0, np.tan(np.radians(1e-7)), 0])
    assert np.isclose(180.0 - calculate_angle(a, b, c), 1e-7, rtol=1e-3)  # was 0.0


def test_angle_coincident_points_raises():
    with pytest.raises(ValueError):
        calculate_angle(np.zeros(3), np.zeros(3), np.ones(3))


def test_dihedral_sign_iupac():
    # looking along 2->3 (+z), front bond on +x, rear bond rotated clockwise by +60 deg
    p1, p2, p3 = np.array([1.0, 0, 0]), np.zeros(3), np.array([0, 0, 1.0])
    for th in (60.0, -60.0, 120.0):
        p4 = np.array([np.cos(np.radians(th)), np.sin(np.radians(th)), 1.0])
        assert np.isclose(calculate_dihedral(p1, p2, p3, p4), th)


def test_distance_matrix_excludes_lowercase_h():
    dm = calc_distance_matrix(np.array([[0, 0, 0], [1.0, 0, 0]]), np.array(['C', 'h']), exclude_H=True)
    assert dm.shape == (1, 1)


# ---------------------------------------------------------------- CT centroids

def _dipole_cube(units='Bohr'):
    n = 20
    g = np.arange(n)
    i, j, k = np.meshgrid(g, g, g, indexing='ij')
    data = (np.exp(-((i - 6) ** 2 + (j - 10) ** 2 + (k - 10) ** 2) / 4.0)
            - np.exp(-((i - 14) ** 2 + (j - 10) ** 2 + (k - 10) ** 2) / 4.0))
    h_bohr = 0.2
    vv = np.eye(3) * (h_bohr if units == 'Bohr' else h_bohr * BOHR_TO_ANGSTROM)
    return {'data': data, 'origin': np.zeros(3), 'voxel_vectors': vv,
            'grid_size': (n, n, n), 'units': units}


def test_centroids_default_integrates_all_voxels():
    cube = _dipole_cube()
    res = calculate_centroids(cube)
    q_all = cube['data'][cube['data'] > 0].sum() * 0.2 ** 3
    assert np.isclose(res['positive_charge'], q_all)  # was 29 % low (99.5 percentile trim)
    # lobes overlap slightly, so d_CT is ~0.6 % above the lobe-centre separation
    assert np.isclose(res['charge_transfer_distance'], 8 * 0.2 * BOHR_TO_ANGSTROM, rtol=1e-2)


def test_centroids_angstrom_cube_same_charge():
    # cube data are e/bohr^3 also on Angstrom grids (was off by 1/0.529^3 = 6.748)
    rb = calculate_centroids(_dipole_cube('Bohr'))
    ra = calculate_centroids(_dipole_cube('Angstrom'))
    assert np.isclose(ra['positive_charge'], rb['positive_charge'], rtol=1e-10)
    assert np.allclose(ra['positive_centroid'], rb['positive_centroid'], atol=1e-10)


# ---------------------------------------------------------------- buried volume / steric map

def test_buried_volume_single_central_atom_analytic():
    r, big_r = 1.70 * 1.17, 3.5
    res = compute_buried_volume(np.zeros((1, 3)), np.array([r]), radius=big_r, spacing=0.1)
    assert abs(res['total'] - (r / big_r) ** 3 * 100) < 0.1


def test_buried_volume_quadrants_unbiased():
    # was NE 18.98 / SW 17.78 % for a centrosymmetric sphere
    res = compute_buried_volume(np.zeros((1, 3)), np.array([1.989]), radius=3.5, spacing=0.1)
    q = np.array(list(res['quadrants'].values()))
    o = np.array(list(res['octants'].values()))
    assert np.ptp(q) < 1e-9 and np.ptp(o) < 1e-9
    assert np.isclose(q.mean(), res['total'])


def test_buried_volume_octants_mirror_symmetric():
    res = compute_buried_volume(np.array([[0, 0, 2.0], [0, 0, -2.0]]), np.array([1.5, 1.5]), radius=3.5)
    assert np.isclose(res['octants']['Front NE'], res['octants']['Back NE'])


def test_buried_volume_non_commensurate_spacing_symmetric():
    res = compute_buried_volume(np.zeros((1, 3)), np.array([1.5]), radius=3.5, spacing=0.3)
    assert np.ptp(list(res['quadrants'].values())) < 1e-9


def test_steric_map_ignores_atoms_outside_sphere():
    _, _, z = compute_steric_map(np.array([[0, 0, 6.0]]), np.array([2.0]), radius=3.5)
    assert not np.isfinite(z).any()  # was 1249 points painted at z = 3.5


def test_steric_map_keeps_intersecting_atom():
    _, _, z = compute_steric_map(np.array([[0, 0, 4.0]]), np.array([2.0]), radius=3.5)
    assert np.isfinite(z).any() and np.nanmax(z) <= 3.5 + 1e-12


def test_orient_positions_undefined_z_axis_raises():
    p = np.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
    with pytest.raises(ValueError):
        orient_positions(p, 0, z_axis_idx=[1])


def test_sambvca_radii_case_insensitive():
    assert np.isclose(get_sambvca_radii(['CL'])[0], 1.75 * 1.17)
    assert np.isclose(get_sambvca_radii(['c'])[0], 1.70 * 1.17)


# ---------------------------------------------------------------- solvent surfaces

def test_vdw_radius_is_bondi():
    assert np.isclose(_get_vdw_radius('C'), 1.70)  # was covalent + 0.4 = 1.16
    assert np.isclose(_get_vdw_radius('H'), 1.20)


def test_ses_uses_actual_grid_spacing_with_n_voxels():
    # was: empty SES (EDT sampled with the nominal 0.1 A while the grid had 0.285 A)
    rv = _get_vdw_radius('C')
    phi, axes, _ = compute_ses(['C'], np.zeros((1, 3)), spacing=0.1, n_voxels=60)
    h = _axes_spacing(axes)
    vol = np.sum(phi < 0) * np.prod(h)
    assert h[0] > 0.1
    assert abs(vol / (4 / 3 * np.pi * rv ** 3) - 1) < 0.15


def test_ses_single_atom_volume_matches_vdw_sphere():
    rv = _get_vdw_radius('C')
    phi, axes, _ = compute_ses(['C'], np.zeros((1, 3)), spacing=0.1)
    vol = np.sum(phi < 0) * np.prod(_axes_spacing(axes))
    assert abs(vol / (4 / 3 * np.pi * rv ** 3) - 1) < 0.03


# ---------------------------------------------------------------- substituter

def test_attach_fragment_points_away_from_core():
    ase = pytest.importorskip("ase")
    from frogchem.substituter.builder import attach_fragment
    mol = ase.Atoms('CHHHH', positions=[[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0],
                                        [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])
    frag = ase.Atoms('HCHHH', positions=[[-1.09, 0, 0], [0, 0, 0], [0.36, 1.03, 0],
                                         [0.36, -0.51, 0.89], [0.36, -0.51, -0.89]])
    new, added, (a, b) = attach_fragment(mol, frag, (1, 0), (0, 1), n_rotations=12, clash_thresh=0.5)
    p = new.positions
    for h in added:
        if h == b:
            continue
        u, v = p[a] - p[b], p[h] - p[b]
        ang = np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
        assert 105.0 < ang < 114.0  # was 70.7 deg (fragment inverted)
    assert np.isclose(np.linalg.norm(p[a] - p[b]), 1.54)
