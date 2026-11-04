"""cleave orbital files (``<name>.orbitals.json`` from ``almo``/``cdft``) in frog-density.

The fixture (tests/data/cleave/make_fixture.py) is water B3LYP/def2-TZVP with three sets
and PySCF reference values: orbitals and densities must match to 1e-10 (AO order, exact
normalization, ORCA phase of f±3), the non-orthogonal set must give the same density.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from frogchem.cleave_orbitals import CleaveOrbitals, is_open_shell
from frogchem.cli.electronic_density import ElectronicDensity
from frogchem.density_builder import build_basis_table, compute_density_on_points, eval_all_basis
from frogchem.exceptions import InputError, ParseError
from frogchem.filetypes import load_calculation

FIXTURE = Path(__file__).parents[1] / 'data' / 'cleave' / 'water.orbitals.json'


@pytest.fixture(scope='module')
def water():
    return CleaveOrbitals(FIXTURE)


@pytest.fixture(scope='module')
def check():
    data = json.loads(FIXTURE.read_text())['check']
    return {k: np.asarray(v) for k, v in data.items()}


def test_loader_layout(water):
    assert isinstance(load_calculation(str(FIXTURE)), CleaveOrbitals)
    assert water.set_names == ['scf', 'cation', 'almo'] and water.set_name == 'scf'
    assert water.atoms == ['O', 'H', 'H'] and water.mult == 1
    assert water.nao == len(water.mask_atoms) == water.orbital_coefficients.shape[0] == 43
    assert set(water.basis_coeff) == {'O', 'H'} and '1f' in water.basis_coeff['O']
    # PySCF order of O's first p shell (x, y, z) and of the f shell (m = -3..3)
    p0 = int(np.flatnonzero(water.mask_orbital_type == 1)[0])       # after O's 5 s shells
    assert p0 == 5 and list(water.mask_orbital_type[p0:p0 + 3]) == [1, 2, 3]
    f = list(water.mask_orbital_type[water.mask_orbital_type >= 9])
    assert f == [15, 13, 11, 9, 10, 12, 14]
    assert not is_open_shell(water)


def test_orbitals_and_density_match_pyscf(water, check):
    water.select('scf')
    pts = check['points_bohr']
    bf = eval_all_basis(build_basis_table(water), pts)
    phi = bf.T @ water.orbital_coefficients[:, [0, 4, 5]]
    np.testing.assert_allclose(phi.T, check['scf_orbitals_0_4_5'], atol=1e-10)
    np.testing.assert_allclose(compute_density_on_points(water, pts), check['scf_density'], atol=1e-10)


def test_unrestricted_set(water, check):
    water.select('cation')
    try:
        assert is_open_shell(water) and water.alpha_coeff.shape == water.beta_coeff.shape == (43, 43)
        assert water.alpha_occ.sum() == 5 and water.beta_occ.sum() == 4
        pts = check['points_bohr']
        np.testing.assert_allclose(compute_density_on_points(water, pts), check['cation_density'],
                                   atol=1e-10)
        bf = eval_all_basis(build_basis_table(water), pts).T
        spin = ((bf @ water.alpha_coeff) ** 2 @ water.alpha_occ
                - (bf @ water.beta_coeff) ** 2 @ water.beta_occ)
        np.testing.assert_allclose(spin, check['cation_spin_density'], atol=1e-10)
    finally:
        water.select('scf')
    assert not hasattr(water, 'alpha_coeff')


def test_non_orthogonal_set_gives_the_same_density(water, check):
    water.select('almo')
    try:
        assert not water.set_info['orthonormal'] and water.orbital_energies is None
        np.testing.assert_allclose(compute_density_on_points(water, check['points_bohr']),
                                   check['scf_density'], atol=1e-10)
    finally:
        water.select('scf')


def test_errors(tmp_path, water):
    with pytest.raises(InputError, match='No orbital set'):
        water.select('nope')
    bad = tmp_path / 'x.json'
    bad.write_text('{"format": "something-else"}')
    with pytest.raises(ParseError, match='not a cleave orbital file'):
        CleaveOrbitals(bad)
    data = json.loads(FIXTURE.read_text())
    data['sets']['scf']['channels'][0]['occupations'] = None
    nocc = tmp_path / 'nocc.orbitals.json'
    nocc.write_text(json.dumps(data))
    with pytest.raises(InputError, match='no occupations'):
        CleaveOrbitals(nocc).density_view()


def test_frog_density_cli(tmp_path, capsys):
    src = tmp_path / 'water.orbitals.json'
    shutil.copy(FIXTURE, src)
    ElectronicDensity().main([str(src), '--list-sets', '--format', 'table'])
    out = capsys.readouterr().out
    assert 'cation' in out and 'almo' in out
    ElectronicDensity().main([str(src), '--set', 'cation', '--density', '--spin', '--homo',
                              '--voxels', '12', '-q'])
    names = {p.name for p in tmp_path.glob('*.cube')}
    assert names == {'water_cation_density.cube', 'water_cation_spin_density.cube',
                     'water_cation_homo.cube'}
    with pytest.raises(SystemExit) as exc:
        ElectronicDensity().main([str(src), '--set', 'missing', '-q'])
    assert exc.value.code == 1


def test_lean_per_set_file_without_description(tmp_path):
    """Current cleave files: no "description", one file per listed set
    (water.Frag1.orbitals.json): the cube name does not repeat the set."""
    data = json.loads(FIXTURE.read_text())
    name, st = 'cation', data['sets']['cation']
    st.pop('description', None)
    data['sets'] = {name: st}
    src = tmp_path / f'water.{name}.orbitals.json'
    src.write_text(json.dumps(data))
    ElectronicDensity().main([str(src), '--density', '--voxels', '12'])
    assert {p.name for p in tmp_path.glob('*.cube')} == {'water.cation_density.cube'}
