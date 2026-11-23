"""Orbital transition diagram: energy levels, layout data, labels and the frog-tddft flags."""
import shutil
from pathlib import Path

import pytest

from frogchem.filetypes import load_calculation
from frogchem.transition_diagram import (_gaussian_levels, _orca_levels, collect, orbital_levels,
                                         orbital_name, parse_level_labels, weight_definition)

DATA = Path(__file__).resolve().parents[1] / "data"
H2CO = DATA / "tddft" / "h2co.out"
ALLYL = DATA / "tddft" / "allyl.out"
G05_G = DATA / "gaussian_ref" / "gaussian" / "g05_methyloxirane_td.log"
G05_O = DATA / "gaussian_ref" / "orca" / "g05_methyloxirane_td.out"


def test_level_labels():
    labels = parse_level_labels("a5=pi*, b4=n(S),7=x")
    assert labels[('a', 5)] == 'pi*' and labels[('b', 4)] == 'n(S)'
    assert labels[('', 7)] == labels[('a', 7)] == labels[('b', 7)] == 'x'


@pytest.mark.skipif(not (G05_G.exists() and G05_O.exists()), reason="methyloxirane samples not available")
def test_energies_from_text_match_both_programs():
    g = load_calculation(str(G05_G))
    o = load_calculation(str(G05_O))
    eg = _gaussian_levels(g.__dict__['fl'])['a']
    eo = _orca_levels(o.__dict__['fl'])['a']
    assert eg[g.homo] == pytest.approx(eo[o.homo], abs=0.01)
    assert weight_definition(g, False) == "weight = 2c\u00b2" and weight_definition(g, True) == "weight = c\u00b2"
    assert "ORCA" in weight_definition(o, False)
    assert orbital_levels(g, False)[''][g.homo] == pytest.approx(eg[g.homo], abs=1e-3)


@pytest.mark.skipif(not ALLYL.exists(), reason="allyl sample not available")
def test_open_shell_layout():
    calc = load_calculation(str(ALLYL))
    st = calc.tddft_states
    first = sorted((s for m in st for s in st[m].values()), key=lambda s: s['energy'])[0]
    data = collect(calc, [("D1", first, None)], wmin=0.1)
    assert data.open_shell and data.columns == ['a', 'b']
    h = calc.homo
    assert h in data.levels['a'] and h in data.levels['b']      # SOMO and its empty beta partner
    assert orbital_name(h, 'a', data) == 'SOMO'
    assert orbital_name(h, 'b', data) == 'LUMO'
    assert sum(w for *_, w in data.arrows) <= 1.01              # c² weights


@pytest.mark.skipif(not H2CO.exists(), reason="formaldehyde sample not available")
class TestCLI:
    def _run(self, tmp_path, *flags):
        from frogchem.cli.tddft_parser import TddftParser
        src = tmp_path / "h2co.out"
        shutil.copy(H2CO, src)
        TddftParser().main([str(src), "--transition-diagram", "--fig-format", "svg", *flags])
        return tmp_path / "h2co_transitions.svg"

    def test_default_states(self, tmp_path):
        svg = self._run(tmp_path)
        text = svg.read_text(encoding="utf-8")
        assert "S1" in text and "S6" in text and "S7" not in text

    def test_pick(self, tmp_path, capsys):
        svg = self._run(tmp_path, "--diagram-pick", "npi", "pipi")
        text = svg.read_text(encoding="utf-8")
        assert "S1" in text and "S8" in text and "S2" not in text
        out = capsys.readouterr().out
        assert "State Character" in out and "n→π*" in out

    def test_ct_needs_two_fragments(self, tmp_path):
        with pytest.raises(SystemExit):
            self._run(tmp_path, "--diagram-pick", "ct")
