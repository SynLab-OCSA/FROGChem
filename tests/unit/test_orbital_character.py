"""Orbital types (n / π / σ), state character (n→π*, π→π*, CT between fragments)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from frogchem.exceptions import InputError
from frogchem.filetypes import load_calculation
from frogchem.orbital_character import (analyse_orbitals, local_normals, parse_fragments, parse_indices,
                                        pick_state, state_character, SpinOrbitals)

DATA = Path(__file__).resolve().parents[1] / "data"
H2CO = DATA / "tddft" / "h2co.out"            # ORCA 6.1 B3LYP/def2-SVP TDA, PrintBasis PrintMOs
ALLYL = DATA / "tddft" / "allyl.out"          # same, UKS doublet
G05_G = DATA / "gaussian_ref" / "gaussian" / "g05_methyloxirane_td.log"
G05_O = DATA / "gaussian_ref" / "orca" / "g05_methyloxirane_td.out"


def _states(calc):
    return sorted(((st['energy'], m, n, st) for m in calc.tddft_states
                   for n, st in calc.tddft_states[m].items()), key=lambda t: t[0])


class TestInput:
    def test_parse_indices(self):
        assert parse_indices("1,5,8-10") == [1, 5, 8, 9, 10]
        assert parse_indices(["14-15", "20"]) == [14, 15, 20]

    def test_fragments(self):
        assert parse_fragments(["1,5,8-10", "2,3-4"], 12) == [[1, 5, 8, 9, 10], [2, 3, 4]]

    @pytest.mark.parametrize("tokens", [["0-3", "3-5"], ["0-20"], ["a"], ["0", "1", "2"]])
    def test_bad_fragments(self, tokens):
        with pytest.raises(InputError):
            parse_fragments(tokens, 10)


class TestNormals:
    def test_carbonyl_and_sp3(self):
        # formaldehyde in the yz plane, plus a far methane: π normal along x for C and O only
        geom = np.array([[0, 0, 0], [0, 0, 1.205], [0, 0.943, -0.587], [0, -0.943, -0.587],
                         [10, 0, 0], [10.63, 0.63, 0.63], [9.37, -0.63, 0.63], [9.37, 0.63, -0.63],
                         [10.63, -0.63, -0.63]], dtype=float)
        n = local_normals(['C', 'O', 'H', 'H', 'C', 'H', 'H', 'H', 'H'], geom)
        assert np.allclose(np.abs(n[0]), [1, 0, 0], atol=1e-6)
        assert np.allclose(np.abs(n[1]), [1, 0, 0], atol=1e-6)
        assert np.isnan(n[2:]).all()


class TestStateCharacter:
    def test_types_and_ct(self):
        # MOs: 0 n on A, 1 π on A, 2 π* on B, 3 π* on A
        orb = SpinOrbitals(['n', 'π', 'π*', 'π*'], np.zeros(4),
                           np.array([[1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]))
        an = {'': orb}
        ct = state_character({'population': [(1, 2, 0.9, ''), (1, 3, 0.1, '')]}, an)
        assert ct['A→B'] == pytest.approx(0.9) and ct['A loc'] == pytest.approx(0.1)
        assert ct['type'] == 'CT A→B'          # CT wins over π→π*
        npi = state_character({'population': [(0, 3, 0.95, '')]}, an)
        assert npi['type'] == 'n→π*' and npi['n→π*'] == pytest.approx(1.0)
        assert pick_state([npi, ct], 'ct') == (1, None)
        k, best = pick_state([npi], 'pipi')
        assert k is None and best == 0.0


@pytest.mark.skipif(not H2CO.exists(), reason="formaldehyde sample not available")
class TestFormaldehyde:
    def test_npi_and_pipi(self):
        calc = load_calculation(str(H2CO))
        an = analyse_orbitals(calc)
        types = an[''].types
        assert types[calc.homo] == 'n' and types[calc.homo - 1] == 'π' and types[calc.homo + 1] == 'π*'
        chars = [state_character(st, an) for *_, st in _states(calc)]
        k, _ = pick_state(chars, 'npi')
        assert k == 0                                       # S1 is n→π*
        k, _ = pick_state(chars, 'pipi')
        assert _states(calc)[k][3]['fosc'] > 0.3            # the bright π→π*


@pytest.mark.skipif(not ALLYL.exists(), reason="allyl sample not available")
def test_open_shell_allyl():
    calc = load_calculation(str(ALLYL))
    an = analyse_orbitals(calc)
    assert set(an) == {'a', 'b'}
    assert an['a'].types[calc.homo] == 'π'                  # SOMO
    assert an['b'].types[calc.homo] == 'π*'                 # its empty beta partner
    first = _states(calc)[0][3]
    assert state_character(first, an)['type'] == 'π→π*'


@pytest.mark.skipif(not (G05_G.exists() and G05_O.exists()), reason="methyloxirane samples not available")
def test_gaussian_and_orca_agree_on_fragments():
    res = []
    for path in (G05_G, G05_O):
        calc = load_calculation(str(path))
        frags = parse_fragments(["0-3", "4-9"], len(calc.atoms))
        an = analyse_orbitals(calc, frags)
        assert np.allclose(an[''].fragments.sum(axis=0), 1.0)
        assert 'π' not in an[''].types                      # saturated molecule
        assert an[''].types[calc.homo] == 'n'               # oxygen lone pair
        s1 = calc.tddft_states[1][1]
        res.append(state_character(s1, an))
    assert res[0]['A→B'] == pytest.approx(res[1]['A→B'], abs=0.02)
    with pytest.raises(InputError):
        analyse_orbitals(SimpleNamespace(fname='x', orbital_coefficients=None, mask_atoms=None))
