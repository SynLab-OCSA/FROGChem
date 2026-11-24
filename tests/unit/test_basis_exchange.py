"""Basis set from the Basis Set Exchange when the output does not print it (no PrintBasis / gfinput).

Every reference output prints its basis: the same file with the printout hidden must give the same
basis_coeff (shell keys, exponents and normalised contraction coefficients).
"""
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('basis_set_exchange')

from frogchem.basis_exchange import fetch_basis, resolve_basis_name, check_basis_vs_mask  # noqa: E402
from frogchem.orca_parser.calculation import Calculation  # noqa: E402
from frogchem.gaussian_parser.calculation import GaussianCalculation  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / 'data'
ORCA = sorted((DATA / 'gaussian_ref' / 'orca').glob('*.out')) + sorted((DATA / 'tddft').glob('*.out'))
GAUSSIAN = sorted((DATA / 'gaussian_ref' / 'gaussian').glob('*.log'))


def _strip_orca(src: Path, dst: Path) -> Path:
    # Drop PrintBasis from the echoed input: the printed block is then ignored by the parser
    dst.write_text(re.sub(r'printbasis', '', src.read_text(), flags=re.IGNORECASE))
    return dst


def _strip_gaussian(src: Path, dst: Path) -> Path:
    dst.write_text(src.read_text().replace('AO basis set in the form of general basis input', 'removed'))
    return dst


def _assert_same_basis(ref, new):
    assert ref.keys() == new.keys()
    for el in ref:
        assert ref[el].keys() == new[el].keys(), el
        for k in ref[el]:
            np.testing.assert_allclose(new[el][k], ref[el][k], rtol=1e-6, err_msg=f'{el} {k}')


@pytest.mark.parametrize('path', ORCA, ids=lambda p: p.name)
def test_orca_without_printbasis(path, tmp_path):
    ref = Calculation(path).basis_coeff
    calc = Calculation(_strip_orca(path, tmp_path / path.name))
    assert 'printbasis' not in calc.__dict__
    _assert_same_basis(ref, calc.basis_coeff)


@pytest.mark.parametrize('path', GAUSSIAN, ids=lambda p: p.name)
def test_gaussian_without_gfinput(path, tmp_path):
    ref = GaussianCalculation(path).basis_coeff
    calc = GaussianCalculation(_strip_gaussian(path, tmp_path / path.name))
    _assert_same_basis(ref, calc.basis_coeff)


def test_density_unchanged_without_printbasis(tmp_path):
    from frogchem.density_builder import build_basis_table, compute_density_on_points
    path = DATA / 'gaussian_ref' / 'orca' / 'g04_ticl4.out'
    ref, new = Calculation(path), Calculation(_strip_orca(path, tmp_path / path.name))
    pts = ref.geom * 1.8897259886 + 0.3
    np.testing.assert_allclose(compute_density_on_points(new, pts, build_basis_table(new)),
                               compute_density_on_points(ref, pts, build_basis_table(ref)), rtol=1e-6)


def test_gaussian_name_without_hyphen():
    assert resolve_basis_name('def2SVP') == 'def2-SVP'
    assert resolve_basis_name('def2-TZVP') == 'def2-TZVP'
    assert resolve_basis_name('not-a-basis') is None


def test_sp_and_general_contractions_are_split():
    # 6-31G: SP shells -> s + p; cc-pVDZ: general contraction of the s functions -> separate shells
    b = fetch_basis('6-31G', ['C'])['C']
    assert [k for k in b] == ['1s', '2s', '3s', '1p', '2p']
    b = fetch_basis('cc-pVDZ', ['O'])['O']
    assert sum(k.endswith('s') for k in b) == 3 and sum(k.endswith('p') for k in b) == 2


def test_modified_basis_is_rejected():
    calc = Calculation(DATA / 'gaussian_ref' / 'orca' / 'g01_h2o_pure.out')
    assert not check_basis_vs_mask(calc, fetch_basis('def2-SVP', calc.atoms))
