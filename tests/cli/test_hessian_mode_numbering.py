"""frog-hessian -m uses the mode numbers printed by ORCA (translations/rotations included)."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import frogchem.cli.hessian_localizer as hl


def _fake_water():
    # 9 modes as ORCA prints them: 0-5 zero (translations/rotations), 6 imaginary, 7-8 real.
    # Mode 6 moves only O0, mode 7 only H1, mode 8 only H2.
    modes = np.zeros((9, 9))
    modes[0:3, 6] = [1.0, 0.0, 0.0]
    modes[3:6, 7] = [0.0, 1.0, 0.0]
    modes[6:9, 8] = [0.0, 0.0, 1.0]
    return SimpleNamespace(
        atoms=['O', 'H', 'H'],
        geom=np.zeros((3, 3)),
        freq=np.array([0.0] * 6 + [-500.0, 1600.0, 3700.0]),
        number_modes=np.arange(9),
        normal_modes=modes,
    )


@pytest.fixture
def fake_calc(monkeypatch):
    monkeypatch.setattr(hl, "load_calculation", lambda f: _fake_water())


def _run(capsys, tmp_path, *argv):
    f = tmp_path / "w.out"
    f.write_text("")
    hl.HessianLocalizer().main([str(f), "--format", "json", "-t", "1", *argv])
    tables = json.loads(capsys.readouterr().out)
    (atoms,) = [t for t in tables if t["title"].startswith("Atomic Contributions")]
    return atoms["rows"]


def test_explicit_mode_matches_orca_numbering(fake_calc, capsys, tmp_path):
    rows = _run(capsys, tmp_path, "-m", "7")
    assert len(rows) == 1
    assert rows[0]["Atom"] == "H1"
    (key,) = [k for k in rows[0] if k != "Atom"]
    assert key.startswith("Mode 7") and "1600.0" in key


def test_default_selects_imaginary_mode_by_orca_number(fake_calc, capsys, tmp_path):
    rows = _run(capsys, tmp_path)
    assert rows[0]["Atom"] == "O0"
    (key,) = [k for k in rows[0] if k != "Atom"]
    assert key.startswith("Mode 6") and "-500.0" in key


def test_all_modes_skips_zero_frequencies(fake_calc, capsys, tmp_path):
    rows = _run(capsys, tmp_path, "--all-modes")
    keys = [k for k in rows[0] if k != "Atom"]
    assert [k.split()[1] for k in keys] == ["6", "7", "8"]


def test_real_ts_imaginary_mode_is_orca_mode_6(capsys):
    if not os.path.exists("files/opt.out"):
        pytest.skip("files/opt.out not found")
    hl.HessianLocalizer().main(["files/opt.out", "-m", "6", "--format", "json"])
    out = capsys.readouterr().out
    assert "Mode 6" in out and "-180.9" in out
