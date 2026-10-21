"""Gaussian 16 orbitals, basis, overlap and charges validated against the ORCA 6 twins.

Reference set: tests/data/gaussian_ref/{gaussian,orca}/ (single points at the same geometry, nosymm,
G16 A.03 with `#p ... pop=full iop(3/33=1) iop(6/80=1) gfinput`, ORCA 6.1.1 with PrintBasis PrintMOs NoRI).
Every test skips when the outputs are missing.
"""

import os
import types
from functools import lru_cache
from math import pi

import numpy as np
import pytest

from frogchem.density_builder import (
    _AU, _CART_EXP, _NORM_CONST, _ang_momentum, build_basis_table, eval_all_basis, generate_grid,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, 'tests', 'data', 'gaussian_ref')

GAUSSIAN = {
    'g01': 'gaussian/g01_h2o_pure.log', 'g02': 'gaussian/g02_h2o_cart.log', 'g03': 'gaussian/g03_ch3_uks.log',
    'g04': 'gaussian/g04_ticl4.log', 'g05': 'gaussian/g05_methyloxirane_td.log', 'g06': 'gaussian/g06_h20_nbo.log',
}
ORCA = {
    'g01': 'orca/g01_h2o_pure.out', 'g03': 'orca/g03_ch3_uks.out', 'g04': 'orca/g04_ticl4.out',
    'g05': 'orca/g05_methyloxirane_td.out', 'g06': 'orca/g06_h2o_nbo.out',
}
Z = {'H': 1, 'C': 6, 'O': 8, 'Cl': 17, 'Ti': 22}


def _path(rel):
    path = os.path.join(REF, rel)
    if not os.path.exists(path):
        pytest.skip(f'{rel} not found (Gaussian reference set not available)')
    return path


@lru_cache(maxsize=None)
def gauss(key):
    from frogchem.gaussian_parser.calculation import GaussianCalculation
    calc = GaussianCalculation(_path(GAUSSIAN[key]))
    calc.atoms  # parse now
    return calc


@lru_cache(maxsize=None)
def orca(key):
    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation(_path(ORCA[key]))
    calc.atoms
    return calc


def _coeff_sets(calc):
    """[(label, C, occ, energies)]: the closed-shell set, or the alpha and the beta set."""
    if getattr(calc, 'alpha_coeff', None) is not None:
        n = calc.alpha_coeff.shape[1]
        return [('alpha', calc.alpha_coeff, calc.alpha_occ, calc.orbital_energies[:n]),
                ('beta', calc.beta_coeff, calc.beta_occ, calc.orbital_energies[n:])]
    return [('closed', calc.orbital_coefficients, calc.occupation, calc.orbital_energies)]


def _n_electrons(calc):
    return sum(Z[str(a)] for a in calc.atoms) - calc.charge


# ------------------------------------------------------------------ analytic overlap of the basis table
# Cartesian expansion of density_builder._solid_harmonic_poly (ORCA phase) plus the Cartesian codes
_POLY = {0: {(0, 0, 0): 1}, 1: {(1, 0, 0): 1}, 2: {(0, 1, 0): 1}, 3: {(0, 0, 1): 1},
         4: {(0, 0, 2): 2, (2, 0, 0): -1, (0, 2, 0): -1}, 5: {(1, 0, 1): 1}, 6: {(0, 1, 1): 1},
         7: {(2, 0, 0): 1, (0, 2, 0): -1}, 8: {(1, 1, 0): 1},
         9: {(0, 0, 3): 2, (2, 0, 1): -3, (0, 2, 1): -3}, 10: {(1, 0, 2): 4, (3, 0, 0): -1, (1, 2, 0): -1},
         11: {(0, 1, 2): 4, (2, 1, 0): -1, (0, 3, 0): -1}, 12: {(2, 0, 1): 1, (0, 2, 1): -1},
         13: {(1, 1, 1): 1}, 14: {(3, 0, 0): -1, (1, 2, 0): 3}, 15: {(2, 1, 0): -3, (0, 3, 0): 1}}
_POLY.update({code: {abc: 1} for code, abc in _CART_EXP.items()})


def _os_1d(la, lb, A, B, a, b):
    """Obara-Saika 1D overlaps S[i][j] (i<=la, j<=lb), each an array over the primitive pairs (a x b)."""
    p = a[:, None] + b[None, :]
    P = (a[:, None] * A + b[None, :] * B) / p
    PA, PB = P - A, P - B
    S = [[None] * (lb + 1) for _ in range(la + 1)]
    S[0][0] = np.sqrt(pi / p) * np.exp(-a[:, None] * b[None, :] / p * (A - B) ** 2)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                v = PA * S[i - 1][j]
                if i > 1:
                    v = v + (i - 1) / (2 * p) * S[i - 2][j]
                if j > 0:
                    v = v + j / (2 * p) * S[i - 1][j - 1]
            else:
                v = PB * S[i][j - 1]
                if j > 1:
                    v = v + (j - 1) / (2 * p) * S[i][j - 2]
            S[i][j] = v
    return S


def basis_overlap(table):
    """Analytic overlap matrix of a density_builder basis table (vectorised over primitives)."""
    n = len(table)
    funcs = []
    for bf in table:
        prims = np.asarray(bf['primitives'], dtype=float)
        l = _ang_momentum(bf['ang_code'])
        w = prims[:, 1] * _NORM_CONST[bf['ang_code']] * prims[:, 0] ** (0.5 * l + 0.75)
        funcs.append((np.asarray(bf['center'], dtype=float), prims[:, 0], w, _POLY[bf['ang_code']], l))
    S = np.zeros((n, n))
    for i in range(n):
        Ai, ai, wi, poly_i, li = funcs[i]
        for j in range(i, n):
            Bj, bj, wj, poly_j, lj = funcs[j]
            tabs = [_os_1d(li, lj, Ai[d], Bj[d], ai, bj) for d in range(3)]
            tot = np.zeros((len(ai), len(bj)))
            for mi, ti in poly_i.items():
                for mj, tj in poly_j.items():
                    tot += ti * tj * tabs[0][mi[0]][mj[0]] * tabs[1][mi[1]][mj[1]] * tabs[2][mi[2]][mj[2]]
            S[i, j] = S[j, i] = wi @ tot @ wj
    return S


@lru_cache(maxsize=None)
def rebuilt_overlap(key):
    return basis_overlap(build_basis_table(gauss(key)))


# ------------------------------------------------------------------ 0. what the parser exposes
@pytest.mark.parametrize('key', ['g01', 'g02', 'g03', 'g04', 'g05', 'g06'])
def test_gaussian_parser_exposes_orca_layout(key):
    calc = gauss(key)
    n_bf = len(calc.mask_atoms)
    assert calc.orbital_coefficients.shape[0] == n_bf
    assert len(calc.mask_type_n) == len(calc.mask_orbital_type) == n_bf
    assert calc.orbital_energies.shape == calc.occupation.shape == (calc.orbital_coefficients.shape[1],)
    assert calc.overlap.shape == (n_bf, n_bf)
    assert set(calc.basis_coeff) == {str(a) for a in calc.atoms}
    assert sorted({int(a) for a in calc.mask_atoms}) == list(range(len(calc.atoms)))
    assert abs(calc.occupation.sum() - _n_electrons(calc)) < 1e-12
    assert calc.mulliken and calc.loewdin and len(calc.mulliken) == len(calc.atoms)
    assert abs(sum(v['charge'] for v in calc.mulliken.values()) - calc.charge) < 1e-4


def test_gaussian_open_shell_layout():
    calc = gauss('g03')
    n_mo = calc.alpha_coeff.shape[1]
    assert calc.orbital_coefficients.shape[1] == 2 * n_mo
    assert (calc.alpha_occ.sum(), calc.beta_occ.sum()) == (5, 4)
    assert (calc.homo, calc.beta_homo) == (4, 3)
    assert abs(sum(v['spin'] for v in calc.mulliken.values()) - 1.0) < 1e-4


# ------------------------------------------------------------------ 1. basis, overlap, orthonormality
@pytest.mark.parametrize('key', ['g01', 'g02', 'g03', 'g04', 'g05'])
def test_rebuilt_overlap_matches_gaussian(key):
    """S from the parsed basis (density_builder conventions) == S printed by Gaussian (iop(3/33=1)).

    Checks basis parsing, AO order, normalisation and the f(±3) phase change in one go.
    """
    calc = gauss(key)
    assert np.abs(rebuilt_overlap(key) - calc.overlap).max() < 2e-6


@pytest.mark.parametrize('key', ['g01', 'g02', 'g03', 'g04', 'g05'])
@pytest.mark.parametrize('source', ['printed', 'rebuilt'])
def test_gaussian_mos_orthonormal(key, source):
    """C^T S C = 1 (coefficients are printed with 5 decimals) and Σ occ <φ|φ> = number of electrons."""
    calc = gauss(key)
    S = calc.overlap if source == 'printed' else rebuilt_overlap(key)
    n_el = 0.0
    for _, C, occ, _ in _coeff_sets(calc):
        M = C.T @ S @ C
        assert np.abs(M - np.eye(len(M))).max() < 2e-3
        n_el += np.sum(occ * np.diag(M))
    assert abs(n_el - _n_electrons(calc)) < 5e-4 * _n_electrons(calc)


# ------------------------------------------------------------------ 2. energies and composition vs ORCA
@pytest.mark.parametrize('key', ['g01', 'g03', 'g04', 'g05'])
def test_orbital_energies_match_orca(key):
    g, o = gauss(key), orca(key)
    assert g.orbital_energies.shape == o.orbital_energies.shape
    assert np.abs(g.orbital_energies - o.orbital_energies).max() < 3e-4
    assert np.array_equal(g.occupation, o.occupation)


def _degenerate_groups(energies, tol=1e-4):
    groups, cur = [], [0]
    for i in range(1, len(energies)):
        if energies[i] - energies[i - 1] < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def _composition(calc, C, group):
    """% per (atom, shell letter) of a set of MOs; invariant to rotations inside a degenerate set."""
    from frogchem.cli.mo_coeff import TYPE_CODES
    sq = (C[:, group] ** 2).sum(axis=1)
    sq = 100 * sq / sq.sum()
    atoms = calc.mask_atoms.astype(int)
    out = np.zeros((len(calc.atoms), 4))
    for k, letter in enumerate('spdf'):
        sel = np.isin(calc.mask_orbital_type, TYPE_CODES[letter])
        out[:, k] = np.bincount(atoms[sel], weights=sq[sel], minlength=len(calc.atoms))
    return out


@pytest.mark.parametrize('key', ['g01', 'g03', 'g04', 'g05'])
def test_mo_composition_matches_orca(key):
    """frog-mo C² composition per atom and shell type, Gaussian vs ORCA (degenerate sets summed)."""
    from frogchem.cli.mo_coeff import calculate_full_percent_matrix, composition_table
    g, o = gauss(key), orca(key)
    worst = 0.0
    for (_, Cg, _, eg), (_, Co, _, _) in zip(_coeff_sets(g), _coeff_sets(o)):
        for group in _degenerate_groups(eg):
            worst = max(worst, np.abs(_composition(g, Cg, group) - _composition(o, Co, group)).max())
    assert worst < 0.5
    # the CLI path runs on the Gaussian output
    rows, headers = composition_table(g, [g.homo, g.homo + 1], calculate_full_percent_matrix(g), None)
    totals = [i for i, h in enumerate(headers) if h.startswith('Total_')]
    assert len(rows) == 2 and all(abs(sum(row[i] for i in totals) - 100.0) < 1e-6 for row in rows)


# ------------------------------------------------------------------ 3. charges vs ORCA
@pytest.mark.parametrize('key', ['g01', 'g03', 'g04', 'g05', 'g06'])
@pytest.mark.parametrize('scheme', ['mulliken', 'loewdin'])
def test_charges_match_orca(key, scheme):
    g, o = getattr(gauss(key), scheme), getattr(orca(key), scheme)
    assert g.keys() == o.keys()
    for idx in g:
        assert g[idx]['atom'] == o[idx]['atom']
        assert abs(g[idx]['charge'] - o[idx]['charge']) < 0.01, (idx, g[idx], o[idx])


def test_mulliken_spin_populations_match_orca():
    g, o = gauss('g03').mulliken, orca('g03').mulliken
    for idx in g:
        assert abs(g[idx]['spin'] - o[idx]['spin']) < 0.01, (idx, g[idx], o[idx])


# ------------------------------------------------------------------ 4. densities vs ORCA
def _grid(calc, spacing=0.25, padding=3.0):
    grid, _, _ = generate_grid(calc.geom * _AU, spacing * _AU, padding * _AU)
    return grid.reshape(-1, 3)


def _densities(calc, points):
    """(total, spin) density on the points (e/bohr³) from the occupied MOs and the parsed basis."""
    bf = eval_all_basis(build_basis_table(calc), points)
    rho = []
    for _, C, occ, _ in _coeff_sets(calc):
        phi = bf.T @ C[:, occ > 0]
        rho.append((phi ** 2) @ occ[occ > 0])
    total = sum(rho)
    spin = rho[0] - rho[1] if len(rho) == 2 else np.zeros_like(total)
    return total, spin


def _electrons_from_basis(calc, S):
    """Σ occ <φ|φ> per spin set, with S of the very basis used on the grid."""
    return [np.sum(occ * np.einsum('ik,ij,jk->k', C, S, C)) for _, C, occ, _ in _coeff_sets(calc)]


@pytest.mark.parametrize('key', ['g01', 'g03'])
def test_density_matches_orca(key):
    from frogchem.density_builder import compute_density_on_points
    g, o = gauss(key), orca(key)
    points = _grid(g)
    rho_g, spin_g = _densities(g, points)
    rho_o, spin_o = _densities(o, points)
    assert np.allclose(compute_density_on_points(g, points), rho_g)  # code path of frog-density
    assert np.abs(rho_g - rho_o).max() < 1e-3 * rho_g.max()
    # ∫ρ = Σ occ <φ|φ> (a uniform grid cannot integrate the 1s cusp, so the integral is analytic)
    per_spin = _electrons_from_basis(g, rebuilt_overlap(key))
    assert abs(sum(per_spin) - _n_electrons(g)) < 5e-3
    if key == 'g03':
        assert np.abs(spin_g - spin_o).max() < 1e-3 * np.abs(spin_g).max()
        assert abs(per_spin[0] - per_spin[1] - 1.0) < 2e-3  # ∫ spin density = Nα - Nβ
        assert spin_g.max() > 0.05                           # the unpaired electron on C


def test_cartesian_density_matches_pure_basis():
    """6D/10F (g02) and 5D/7F (g01) water: same density to within the basis-set difference."""
    g1, g2 = gauss('g01'), gauss('g02')
    points = _grid(g1, spacing=0.3)
    rho1, _ = _densities(g1, points)
    rho2, _ = _densities(g2, points)
    assert np.abs(rho1 - rho2).max() < 1e-2 * rho1.max()


# ------------------------------------------------------------------ 5. TD-DFT vs ORCA, DCT
def _gaussian_r_length(fl):
    """{Gaussian state: R} of the length-gauge table '1/2[<0|r|b>*<b|rxdel|0> + ...]' (4th column)."""
    import re
    from frogchem.gaussian_parser.regex_gaussian import REGEXS
    block = fl.split(REGEXS['ecd_section_end'])[-1].split('1/2[<0|del|b>')[0]
    return {int(n): float(r) for n, r in
            re.findall(r'^\s*(\d+)\s+\S+\s+\S+\s+\S+\s+(-?\d+\.\d+)\s*$', block, re.M)}


def test_tddft_states_match_orca():
    """td=(50-50,nstates=10) vs ORCA tda false: same S1..S10 / T1..T10, energies, f and R.

    frog reports Gaussian's velocity-gauge R (origin independent); ORCA prints the length-gauge R,
    which is compared with Gaussian's length-gauge column (they differ by up to ~8 %, the velocity
    form by up to ~30 % on this molecule).
    """
    gc = gauss('g05')
    g, o = gc.tddft_states, orca('g05').tddft_states
    r_length = _gaussian_r_length(gc.fl)
    assert sorted(g) == sorted(o) == [1, 3]
    for mult in (1, 3):
        assert sorted(g[mult]) == sorted(o[mult]) == list(range(1, 11))
        for n in range(1, 11):
            sg, so = g[mult][n], o[mult][n]
            assert abs(sg['energy'] - so['energy']) < 0.01, (mult, n, sg['energy'], so['energy'])
            if mult == 1:
                assert abs(sg['fosc'] - so['fosc']) < max(0.002, 0.05 * so['fosc']), (n, sg['fosc'], so['fosc'])
                r = r_length[sg['gaussian_state']]
                assert abs(r - so['R']) < max(0.5, 0.1 * abs(so['R'])), (n, r, so['R'])
                assert np.sign(sg['R']) == np.sign(so['R'])


def test_dct_runs_on_gaussian_and_matches_orca():
    from frogchem.cli.tddft_parser import process_calc_dct

    def run(calc):
        rows = []
        cmd = types.SimpleNamespace(output=lambda r, headers=None, title=None: rows.extend(r),
                                    derived_path=None, saved=None)
        args = types.SimpleNamespace(voxels=30, padding=3.0, dct_states=['S1', 'S2'], save_cube=False,
                                     dct_threshold=100.0, le_threshold=1.5, ct_threshold=2.0)
        process_calc_dct(calc, calc.fname, args, cmd)
        return {r[0]: float(r[4].split()[0]) for r in rows}

    dg, do = run(gauss('g05')), run(orca('g05'))
    assert set(dg) == set(do) == {'S1', 'S2'}
    for state in dg:
        assert abs(dg[state] - do[state]) < 0.2, (state, dg[state], do[state])
