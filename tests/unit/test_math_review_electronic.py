"""Regression tests for the maths/physics review of the electronic-structure tools.

One test (or small group) per fixed finding: Voronoi ∫ρdV, ORCA f(±3) phase, ECD ν̃·R weighting,
empty peak lists / FWHM validation, charges block selection and open-shell Löwdin, MO energies
block, open-shell CDFT frontier orbitals, CIE absorption/emission, Gaussian TD-DFT states,
cube lattice checks.
"""
import os
import types
from argparse import Namespace
from math import comb, pi, sqrt

import numpy as np
import pytest

from frogchem.grid import integrate_density_voronoi, add_grids, subtract_grids
from frogchem.density_builder import (
    _NORM_CONST, _ang_momentum, _solid_harmonic_poly, _AU, _make_cube_data,
    build_basis_table, compute_density_on_points, generate_grid,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FILES = os.path.join(REPO, 'files')

# ------------------------------------------------------------------ analytic overlap helpers
# Cartesian expansion of the polynomials of density_builder._solid_harmonic_poly (ORCA phase)
_POLY = {0: {(0, 0, 0): 1}, 1: {(1, 0, 0): 1}, 2: {(0, 1, 0): 1}, 3: {(0, 0, 1): 1},
         4: {(0, 0, 2): 2, (2, 0, 0): -1, (0, 2, 0): -1}, 5: {(1, 0, 1): 1}, 6: {(0, 1, 1): 1},
         7: {(2, 0, 0): 1, (0, 2, 0): -1}, 8: {(1, 1, 0): 1},
         9: {(0, 0, 3): 2, (2, 0, 1): -3, (0, 2, 1): -3}, 10: {(1, 0, 2): 4, (3, 0, 0): -1, (1, 2, 0): -1},
         11: {(0, 1, 2): 4, (2, 1, 0): -1, (0, 3, 0): -1}, 12: {(2, 0, 1): 1, (0, 2, 1): -1},
         13: {(1, 1, 1): 1}, 14: {(3, 0, 0): -1, (1, 2, 0): 3}, 15: {(2, 1, 0): -3, (0, 3, 0): 1}}


def _dfact(n):
    return 1.0 if n <= 0 else n * _dfact(n - 2)


def _i1(la, lb, A, B, a, b):
    p = a + b
    P = (a * A + b * B) / p
    s = 0.0
    for i in range(la + 1):
        for j in range(lb + 1):
            if (i + j) % 2 == 0:
                s += (comb(la, i) * comb(lb, j) * (P - A) ** (la - i) * (P - B) ** (lb - j)
                      * _dfact(i + j - 1) / (2 * p) ** ((i + j) / 2))
    return sqrt(pi / p) * np.exp(-a * b / p * (A - B) ** 2) * s


def _overlap(table):
    n = len(table)
    S = np.zeros((n, n))
    for i in range(n):
        bi = table[i]
        li = _ang_momentum(bi['ang_code'])
        for j in range(i, n):
            bj = table[j]
            lj = _ang_momentum(bj['ang_code'])
            v = 0.0
            for a, ca in bi['primitives']:
                na = _NORM_CONST[bi['ang_code']] * a ** (0.5 * li + 0.75)
                for b, cb in bj['primitives']:
                    nb = _NORM_CONST[bj['ang_code']] * b ** (0.5 * lj + 0.75)
                    for mi, ti in _POLY[bi['ang_code']].items():
                        for mj, tj in _POLY[bj['ang_code']].items():
                            v += ca * cb * na * nb * ti * tj * np.prod(
                                [_i1(mi[d], mj[d], bi['center'][d], bj['center'][d], a, b) for d in range(3)])
            S[i, j] = S[j, i] = v
    return S


def _h2_sto3g():
    prims = np.array([[3.42525091, 0.15432897], [0.62391373, 0.53532814], [0.16885540, 0.44463454]])
    calc = types.SimpleNamespace(
        atoms=np.array(['H', 'H']), geom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        mask_atoms=np.array(['0', '1']), mask_orbital_type=np.array([0, 0]),
        basis_coeff={'H': {'1s': prims}}, mult=1, fname='synthetic')
    table = build_basis_table(calc)
    s01 = _overlap(table)[0, 1]
    cg = 1.0 / np.sqrt(2.0 * (1.0 + s01))
    calc.orbital_coefficients = np.array([[cg, 0.0], [cg, 0.0]])
    calc.occupation = np.array([2.0, 0.0])
    return calc, table


# ------------------------------------------------------------------ grid.py
def test_voronoi_integral_includes_voxel_volume():
    calc, table = _h2_sto3g()
    grid, axes, (nx, ny, nz) = generate_grid(calc.geom * _AU, spacing=0.15 * _AU, padding=5 * _AU)
    rho = compute_density_on_points(calc, grid.reshape(-1, 3), table)
    cube = _make_cube_data(rho.reshape(nx, ny, nz), axes, calc)
    per_atom = integrate_density_voronoi(cube)
    assert abs(per_atom.sum() - 2.0) < 1e-3          # was 87.8 (raw voxel sum) before the fix


def test_voronoi_constant_cube_non_unit_voxels():
    cube = {'grid_size': (4, 4, 4), 'origin': np.zeros(3), 'data': np.full((4, 4, 4), 0.5),
            'voxel_vectors': np.diag([0.5, 0.5, 0.25]),
            'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.array([0.75, 0.75, 0.375])}]}
    assert np.isclose(integrate_density_voronoi(cube)[0], 64 * 0.5 * 0.0625)


def test_grid_arithmetic_rejects_different_lattices():
    g1 = {'grid_size': (2, 2, 2), 'origin': np.zeros(3), 'voxel_vectors': np.eye(3), 'data': np.ones((2, 2, 2))}
    g2 = dict(g1, origin=np.array([1.0, 0.0, 0.0]))
    g3 = dict(g1, voxel_vectors=0.5 * np.eye(3))
    for other in (g2, g3):
        with pytest.raises(ValueError):
            add_grids(g1, other)
        with pytest.raises(ValueError):
            subtract_grids(g1, other)
    assert np.allclose(subtract_grids(g1, dict(g1, data=3 * np.ones((2, 2, 2))))['data'], 2.0)


# ------------------------------------------------------------------ density_builder.py
def test_orca_f_pm3_phase():
    p = np.array([1.0]), np.array([0.0]), np.array([0.0])
    assert _solid_harmonic_poly(14, *p)[0] == -1.0                      # -x(x²-3y²)
    assert _solid_harmonic_poly(15, np.array([0.0]), np.array([1.0]), np.array([0.0]))[0] == 1.0  # -y(3x²-y²)


def test_orca_mos_orthonormal_with_f_functions():
    path = os.path.join(FILES, 'tddft_mos.out')
    if not os.path.exists(path):
        pytest.skip('files/tddft_mos.out not found')
    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation(path)
    table = build_basis_table(calc)
    S = _overlap(table)
    C = calc.orbital_coefficients
    M = C.T @ S @ C
    assert np.abs(M - np.eye(len(M))).max() < 1e-4                   # was 5.30 with the wrong f(±3) sign
    assert abs(np.sum(calc.occupation * np.diag(M)) - 10.0) < 1e-4    # H2O: 10 electrons


# ------------------------------------------------------------------ tddft_parser.py
def test_ecd_bands_weighted_by_transition_energy():
    from frogchem.cli.tddft_parser import TddftParser
    cmd = TddftParser()
    cmd.args = Namespace(shift=0.0, fwhm=0.3)
    calc = types.SimpleNamespace(tddft_states={1: {1: {'energy': 3.0, 'R': 10.0},
                                                   2: {'energy': 6.0, 'R': -10.0}}})
    x = np.linspace(2, 8, 1000)
    y = cmd._spectrum(calc, 'R', x)
    i3, i6 = np.argmin(abs(x - 3)), np.argmin(abs(x - 6))
    assert abs(abs(y[i6] / y[i3]) - 2.0) < 0.01                       # Δε ∝ ν̃_k R_k
    assert y[i3] > 0 > y[i6]
    # UV stays proportional to f alone
    calc_uv = types.SimpleNamespace(tddft_states={1: {1: {'energy': 3.0, 'fosc': 0.1},
                                                      2: {'energy': 6.0, 'fosc': 0.1}}})
    yu = cmd._spectrum(calc_uv, 'fosc', x)
    assert abs(yu[i6] / yu[i3] - 1.0) < 0.01


def test_extract_peaks_no_intensity_returns_empty():
    from frogchem.cli.tddft_parser import extract_peaks, gaussian_convolution
    for states in ({1: {1: {'energy': 3.0, 'fosc': 0.0}}}, {3: {1: {'energy': 3.0}}}, {}):
        e, i = extract_peaks(states, 'fosc', 0.0)
        assert e.size == 0 and i.size == 0
    x = np.linspace(2, 8, 50)
    assert np.all(gaussian_convolution(x, e, i, 0.3) == 0)


def test_gaussian_convolution_area_and_fwhm_validation():
    from frogchem.cli.tddft_parser import gaussian_convolution
    from frogchem.exceptions import InputError
    x = np.linspace(0, 12, 240001)
    y = gaussian_convolution(x, np.array([4.0, 6.0]), np.array([0.3, 0.1]), 0.3)
    assert abs(np.trapezoid(y, x) - 0.4) < 1e-6
    for bad in (0.0, -0.3, float('nan')):
        with pytest.raises(InputError):
            gaussian_convolution(x, np.array([4.0]), np.array([0.1]), bad)


# ------------------------------------------------------------------ population / orbitals
_LOEWDIN_OPEN = """-------------------------------------------
LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS
-------------------------------------------
   0 C :   -0.123456    0.500000
   1 H :    0.123456    0.500000

----------------------------------------------------
LOEWDIN REDUCED ORBITAL CHARGES AND SPIN POPULATIONS
----------------------------------------------------
"""
_LOEWDIN_CLOSED = """----------------------
LOEWDIN ATOMIC CHARGES
----------------------
   0 C :   {c0}
   1 H :   {c1}

-------------------------------
LOEWDIN REDUCED ORBITAL CHARGES
-------------------------------
"""
_MULLIKEN = """-----------------------
MULLIKEN ATOMIC CHARGES
-----------------------
   0 C :   {c0}
   1 H :   {c1}
Sum of atomic charges:    0.0000000
"""


def test_loewdin_open_shell_header():
    from frogchem.orca_parser.population import parse_charges
    from frogchem.orca_parser.regex_ORCA import REGEXS
    q = parse_charges(_LOEWDIN_OPEN, 'loewdin', REGEXS['6'])
    assert q == {0: {'atom': 'C', 'charge': -0.123456, 'spin': 0.5},
                 1: {'atom': 'H', 'charge': 0.123456, 'spin': 0.5}}


def test_charges_taken_from_last_block():
    from frogchem.orca_parser.population import parse_charges
    from frogchem.orca_parser.regex_ORCA import REGEXS
    txt = (_MULLIKEN.format(c0='-0.100000', c1='0.100000') + _LOEWDIN_CLOSED.format(c0='-0.200000', c1='0.200000')
           + "\n GEOMETRY OPTIMIZATION CYCLE 2\n"
           + _MULLIKEN.format(c0='-0.150000', c1='0.150000') + _LOEWDIN_CLOSED.format(c0='-0.250000', c1='0.250000'))
    assert parse_charges(txt, 'mulliken', REGEXS['6'])[0]['charge'] == -0.15
    assert parse_charges(txt, 'loewdin', REGEXS['6'])[1]['charge'] == 0.25


def test_charges_two_letter_elements():
    from frogchem.orca_parser.population import parse_charges
    from frogchem.orca_parser.regex_ORCA import REGEXS
    txt = _MULLIKEN.format(c0='-0.100000', c1='0.100000').replace('   1 H :', '   1 Br:')
    q = parse_charges(txt, 'mulliken', REGEXS['6'])
    assert q[1] == {'atom': 'Br', 'charge': 0.1}
    path = os.path.join(FILES, 'nbo.out')
    if os.path.exists(path):
        from frogchem.orca_parser.calculation import Calculation
        calc = Calculation(path)
        assert len(calc.mulliken) == len(calc.loewdin) == len(calc.atoms) == 37
        assert calc.mulliken[32]['atom'] == 'Br' and calc.mulliken[32]['charge'] == -0.138023


def test_real_open_shell_loewdin():
    path = os.path.join(FILES, 'tddft.out')
    if not os.path.exists(path):
        pytest.skip('files/tddft.out not found')
    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation(path)
    assert calc.loewdin and len(calc.loewdin) == len(calc.mulliken) == 60


def test_orbital_energies_from_last_block():
    from frogchem.orca_parser.orbitals import get_MOs_energies
    from frogchem.orca_parser.regex_ORCA import REGEXS
    block = ("ORBITAL ENERGIES\n----------------\n\n  NO   OCC          E(Eh)            E(eV) \n"
             "   0   2.0000      {e0:.6f}     0.0000 \n   1   0.0000      {e1:.6f}     0.0000 \n\n"
             "MOLECULAR ORBITALS\n")
    txt = block.format(e0=-0.5, e1=0.1) + block.format(e0=-0.6, e1=0.2)
    en, occ = get_MOs_energies(txt, REGEXS['6'])
    assert np.allclose(en, [-0.6, 0.2]) and np.allclose(occ, [2.0, 0.0])


# ------------------------------------------------------------------ calc_cdft.py
_ORCA_OPEN = """ORBITAL ENERGIES
----------------
                 SPIN UP ORBITALS
  NO   OCC          E(Eh)            E(eV) 
   0   1.0000      -0.500000     -13.6057 
   1   1.0000      -0.206630      -5.6227 
   2   0.0000      -0.024770      -0.6740 

                 SPIN DOWN ORBITALS
  NO   OCC          E(Eh)            E(eV) 
   0   1.0000      -0.500000     -13.6057 
   1   1.0000      -0.223940      -6.0937 
   2   0.0000      -0.119200      -3.2436 

------------------
MOLECULAR ORBITALS
------------------
   0   1.0000      9.999999      9.9999 
"""


def test_cdft_open_shell_uses_both_spins():
    from frogchem.cli.calc_cdft import _get_homo_lumo_energies
    h, l = _get_homo_lumo_energies(types.SimpleNamespace(fl=_ORCA_OPEN))
    assert (h, l) == (-0.20663, -0.1192)                               # was LUMO = -0.02477 (alpha only)
    closed = _ORCA_OPEN.split('                 SPIN DOWN')[0].replace('                 SPIN UP ORBITALS\n', '')
    assert _get_homo_lumo_energies(types.SimpleNamespace(fl=closed)) == (-0.20663, -0.02477)


def test_cdft_gaussian_beta_orbitals():
    from frogchem.cli.calc_cdft import _get_homo_lumo_energies
    fl = (" Alpha  occ. eigenvalues --  -10.20000  -0.30000\n"
          " Alpha virt. eigenvalues --   -0.05000   0.10000\n"
          "  Beta  occ. eigenvalues --  -10.10000  -0.32000\n"
          "  Beta virt. eigenvalues --   -0.15000   0.20000\n")
    assert _get_homo_lumo_energies(types.SimpleNamespace(fl=fl)) == (-0.3, -0.15)


# ------------------------------------------------------------------ cie_plot.py
def test_cie_absorption_and_emission():
    try:
        import colour
    except ImportError:
        pytest.skip('colour-science not installed')
    from frogchem.cli.cie_plot import spectrum_to_XYZ
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    d65 = colour.SDS_ILLUMINANTS["D65"]
    wl = np.arange(360, 781, 1.0)
    A = np.exp(-0.5 * ((wl - 450) / 25) ** 2)
    sd_t = colour.SpectralDistribution(dict(zip(wl, 10 ** -A)))
    x, y = colour.XYZ_to_xy(spectrum_to_XYZ(colour, sd_t, cmfs, d65, 'absorption'))
    assert abs(x - 0.4032) < 2e-3 and abs(y - 0.4720) < 2e-3          # yellow, not blue
    em = colour.SpectralDistribution(dict(zip(wl, np.exp(-0.5 * ((wl - 450) / 10) ** 2))))
    x, y = colour.XYZ_to_xy(spectrum_to_XYZ(colour, em, cmfs, d65, 'emission'))
    assert abs(x - 0.1546) < 5e-4 and abs(y - 0.0203) < 5e-4          # no D65 weighting


# ------------------------------------------------------------------ gaussian_parser
_G_TD = """
 Excitation energies and oscillator strengths:

 Excited State   1:      Singlet-A      3.4560 eV   358.75 nm  f=0.0450  <S**2>=0.000
       5 ->  6         0.70710
 Excited State   2:      Triplet-B      4.1230 eV   300.71 nm  f=0.0000  <S**2>=2.000
       4 ->  6         0.65432
 Excited State   3:      Singlet-A     -0.2345 eV -5287.23 nm  f=-0.0000  <S**2>=0.000
       5 ->  7         0.70000
***
 <0|del|b> * <b|rxdel|0> + <0|del|b> * <b|delr+rdel|0>
 1   0.1234   0.5678   0.9101   -12.3456
 2   0.2345   0.6789   0.1234    25.6789
 3   0.2345   0.6789   0.1234     1.0000
1/2[<0|r|b>*<b|rxdel|0> + (<0|rxdel|b>*<b|r|0>)*]
"""


def test_gaussian_tddft_states():
    from frogchem.gaussian_parser.tddft import parse_tddft_states, parse_uv
    from frogchem.gaussian_parser.regex_gaussian import REGEXS
    assert parse_uv(_G_TD, REGEXS).shape == (3, 2)                     # negative-energy state kept
    st = parse_tddft_states(_G_TD, REGEXS)
    assert set(st) == {1, 3}
    s1 = st[1][1]
    assert s1['energy'] == 3.456 and s1['fosc'] == 0.045 and s1['R'] == -12.3456
    assert s1['population'][0][:2] == (4, 5) and abs(s1['population'][0][2] - 1.0) < 1e-4  # 2c², 0-based
    # numbered per multiplicity (T1 is Gaussian's state 2, S2 its state 3), as in ORCA
    assert st[3][1]['R'] == 25.6789 and st[3][1]['S**2'] == 2.0 and st[3][1]['gaussian_state'] == 2
    assert st[1][2]['energy'] == -0.2345 and st[1][2]['gaussian_state'] == 3
