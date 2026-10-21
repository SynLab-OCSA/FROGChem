import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from frogchem.cli.get_charges import GetCharges
from frogchem.cli.internals import Internals
from frogchem.cli.calc_rrho import CalcRrho
from frogchem.cli.tddft_parser import TddftParser
from frogchem.cli.nbo_parser import NboParser


@pytest.fixture
def enan_checkpoint(tmp_path):
    ckpt_file = tmp_path / "checkpoint.json"
    data = {
        "conformers": {
            "1": {
                "number": 1,
                "energies": {
                    "1": {"Pop": 25.0}
                }
            },
            "2": {
                "number": 2,
                "energies": {
                    "1": {"Pop": 75.0}
                }
            }
        }
    }
    ckpt_file.write_text(json.dumps(data))
    return str(ckpt_file)


def test_get_charges_ensemble_averaging(tmp_path, enan_checkpoint):
    calc1 = MagicMock()
    calc1.mulliken = {0: {'atom': 'C', 'charge': -0.4}, 1: {'atom': 'H', 'charge': 0.1}}
    calc1.loewdin = {0: {'atom': 'C', 'charge': -0.2}, 1: {'atom': 'H', 'charge': 0.05}}

    calc2 = MagicMock()
    calc2.mulliken = {0: {'atom': 'C', 'charge': -0.8}, 1: {'atom': 'H', 'charge': 0.2}}
    calc2.loewdin = {0: {'atom': 'C', 'charge': -0.6}, 1: {'atom': 'H', 'charge': 0.15}}

    p1 = tmp_path / "conf_1" / "protocol_1" / "orca.out"
    p2 = tmp_path / "conf_2" / "protocol_1" / "orca.out"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("dummy")
    p2.write_text("dummy")

    with patch('frogchem.cli.get_charges.load_calculation', side_effect=[calc1, calc2]):
        tool = GetCharges()
        output_tables = []
        tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

        tool.main([str(p1), str(p2), '-i', '0', '1', '-c', enan_checkpoint])

        assert len(output_tables) == 2
        # First table is Ensemble Weights
        weights_data, weights_headers, weights_title = output_tables[0]
        assert "Weights" in weights_title
        assert len(weights_data) == 2

        # Second table is Ensemble-Averaged Charges
        avg_data, avg_headers, avg_title = output_tables[1]
        assert "Ensemble-Averaged Charges" in avg_title
        assert len(avg_data) == 2  # atoms 0 and 1
        # C0 Mulliken: 0.25 * (-0.4) + 0.75 * (-0.8) = -0.1 - 0.6 = -0.7
        assert pytest.approx(avg_data[0][3]) == -0.7
        # C0 Loewdin: 0.25 * (-0.2) + 0.75 * (-0.6) = -0.05 - 0.45 = -0.5
        assert pytest.approx(avg_data[0][4]) == -0.5


def test_internals_ensemble_averaging(tmp_path, enan_checkpoint):
    p1 = tmp_path / "conf_1" / "protocol_1" / "opt.xyz"
    p2 = tmp_path / "conf_2" / "protocol_1" / "opt.xyz"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    
    xyz1 = "2\nconf1\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n"
    xyz2 = "2\nconf2\nC 0.0 0.0 0.0\nH 2.0 0.0 0.0\n"
    p1.write_text(xyz1)
    p2.write_text(xyz2)

    tool = Internals()
    output_tables = []
    tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

    tool.main([str(p1), str(p2), '-i', '0', '1', '-c', enan_checkpoint])

    assert len(output_tables) == 2
    avg_data, avg_headers, avg_title = output_tables[1]
    assert "Ensemble-Averaged Internal Coordinates" in avg_title
    # Linear average: 0.25 * 1.0 + 0.75 * 2.0 = 1.75 Å
    assert avg_data[0][2] == pytest.approx(1.75)
    # NOE (r^-6): (0.25 * 1.0^-6 + 0.75 * 2.0^-6)^(-1/6)
    expected_noe = (0.25 * 1.0 + 0.75 * (2.0 ** -6)) ** (-1.0 / 6.0)
    assert avg_data[0][3] == pytest.approx(expected_noe)


def test_calc_rrho_ensemble_averaging(tmp_path, enan_checkpoint):
    p1 = tmp_path / "conf_1" / "protocol_1" / "freq.out"
    p2 = tmp_path / "conf_2" / "protocol_1" / "freq.out"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("dummy")
    p2.write_text("dummy")

    calc1 = MagicMock()
    calc1.final_energy = -100.0
    calc1.freq = np.array([100.0, 500.0, 1000.0])
    calc1.imag_freq = []
    calc1.atoms = ['C']
    calc1.mult = 1
    calc1.mw = 12.0
    calc1.B = np.array([1.0, 1.0, 1.0])
    calc1.geom = np.array([[0,0,0]])

    calc2 = MagicMock()
    calc2.final_energy = -100.2
    calc2.freq = np.array([100.0, 500.0, 1000.0])
    calc2.imag_freq = []
    calc2.atoms = ['C']
    calc2.mult = 1
    calc2.mw = 12.0
    calc2.B = np.array([1.0, 1.0, 1.0])
    calc2.geom = np.array([[0,0,0]])

    with patch('frogchem.cli.calc_rrho.load_calculation', side_effect=[calc1, calc2]):
        with patch('frogchem.cli.calc_rrho.free_gibbs_energy', side_effect=[
            (-99.9, 0.05, -99.95, 0.0001),
            (-100.1, 0.05, -100.15, 0.0001),
        ]):
            tool = CalcRrho()
            output_tables = []
            tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

            tool.main([str(p1), str(p2), '-c', enan_checkpoint])

            assert len(output_tables) == 2
            avg_data, avg_headers, avg_title = output_tables[1]
            assert "Ensemble-Averaged Thermodynamics" in avg_title
            # E average: 0.25 * (-100.0) + 0.75 * (-100.2) = -100.15
            e_row = [r for r in avg_data if "Electronic Energy" in r[0]][0]
            assert float(e_row[1]) == pytest.approx(-100.15)


def test_tddft_ensemble_averaging(tmp_path, enan_checkpoint):
    p1 = tmp_path / "conf_1" / "protocol_1" / "tddft.out"
    p2 = tmp_path / "conf_2" / "protocol_1" / "tddft.out"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("dummy")
    p2.write_text("dummy")

    calc1 = MagicMock()
    calc1.tddft = True
    calc1.soc = False
    calc1.tddft_states = {
        1: {
            1: {"energy": 3.0, "fosc": 0.10, "R": 10.0, "S**2": 0.0, "population": []},
        }
    }

    calc2 = MagicMock()
    calc2.tddft = True
    calc2.soc = False
    calc2.tddft_states = {
        1: {
            1: {"energy": 4.0, "fosc": 0.20, "R": 30.0, "S**2": 0.0, "population": []},
        }
    }

    with patch('frogchem.cli.tddft_parser.load_calculation', side_effect=[calc1, calc2]):
        tool = TddftParser()
        output_tables = []
        tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

        tool.main([str(p1), str(p2), '-c', enan_checkpoint])

        assert len(output_tables) == 2
        avg_data, avg_headers, avg_title = output_tables[1]
        assert "Ensemble-Averaged Excited States" in avg_title
        # State 1 Energy: 0.25 * 3.0 + 0.75 * 4.0 = 0.75 + 3.0 = 3.75 eV
        assert pytest.approx(avg_data[0][2]) == 3.75
        # Oscillator strength f: 0.25 * 0.10 + 0.75 * 0.20 = 0.025 + 0.15 = 0.175
        assert pytest.approx(avg_data[0][3]) == 0.175
        # Rotatory strength R: 0.25 * 10.0 + 0.75 * 30.0 = 2.5 + 22.5 = 25.0
        assert pytest.approx(avg_data[0][4]) == 25.0


def test_tddft_soc_ensemble_averaging(tmp_path, enan_checkpoint):
    p1 = tmp_path / "conf_1" / "protocol_1" / "tddft.out"
    p2 = tmp_path / "conf_2" / "protocol_1" / "tddft.out"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("dummy")
    p2.write_text("dummy")

    calc1 = MagicMock()
    calc1.tddft = True
    calc1.soc = True
    calc1.tddft_states = {
        1: {1: {"energy": 3.0, "fosc": 0.10, "R": 10.0, "S**2": 0.0, "population": []}},
        3: {1: {"energy": 2.5, "fosc": 0.0, "R": 0.0, "S**2": 2.0, "population": []}},
    }
    calc1.tddft_soc = np.array([[4.0], [2.0]])

    calc2 = MagicMock()
    calc2.tddft = True
    calc2.soc = True
    calc2.tddft_states = {
        1: {1: {"energy": 3.0, "fosc": 0.10, "R": 10.0, "S**2": 0.0, "population": []}},
        3: {1: {"energy": 2.5, "fosc": 0.0, "R": 0.0, "S**2": 2.0, "population": []}},
    }
    calc2.tddft_soc = np.array([[8.0], [6.0]])

    with patch('frogchem.cli.tddft_parser.load_calculation', side_effect=[calc1, calc2]):
        tool = TddftParser()
        output_tables = []
        tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

        tool.main([str(p1), str(p2), '-c', enan_checkpoint])

        assert len(output_tables) == 3
        soc_data, soc_headers, soc_title = output_tables[2]
        assert "Ensemble-Averaged Spin-Orbit Coupling" in soc_title
        assert soc_headers == ['State', 'T1']
        # S0: 0.25 * 4.0 + 0.75 * 8.0 = 1.0 + 6.0 = 7.0
        assert soc_data[0] == ['S0', 7.0]
        # S1: 0.25 * 2.0 + 0.75 * 6.0 = 0.5 + 4.5 = 5.0
        assert soc_data[1] == ['S1', 5.0]




def test_nbo_ensemble_averaging(tmp_path, enan_checkpoint):
    p1 = tmp_path / "conf_1" / "protocol_1" / "nbo.out"
    p2 = tmp_path / "conf_2" / "protocol_1" / "nbo.out"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("dummy")
    p2.write_text("dummy")

    nbo1 = MagicMock()
    nbo1.npa_charges = pd.DataFrame([
        {'Atom': 'C', 'Index': 0, 'Charge': -0.2, 'Valence': 4.1, 'Rydberg': 0.02},
    ])
    nbo1.wiberg_matrix = None
    nbo1.wiberg_totals = {}
    nbo1.perturbation_theory = None

    nbo2 = MagicMock()
    nbo2.npa_charges = pd.DataFrame([
        {'Atom': 'C', 'Index': 0, 'Charge': -0.6, 'Valence': 3.9, 'Rydberg': 0.06},
    ])
    nbo2.wiberg_matrix = None
    nbo2.wiberg_totals = {}
    nbo2.perturbation_theory = None

    calc1 = MagicMock()
    calc1.nbo = nbo1
    calc2 = MagicMock()
    calc2.nbo = nbo2

    with patch('frogchem.cli.nbo_parser.load_calculation', side_effect=[calc1, calc2]):
        tool = NboParser()
        output_tables = []
        tool.output = lambda data, headers, title="", **kw: output_tables.append((data, headers, title))

        tool.main([str(p1), str(p2), '--npa', '-c', enan_checkpoint])

        assert len(output_tables) == 2
        avg_data, avg_headers, avg_title = output_tables[1]
        assert "Ensemble-Averaged NPA Charges" in avg_title
        # C0 Charge: 0.25 * (-0.2) + 0.75 * (-0.6) = -0.05 - 0.45 = -0.50
        assert pytest.approx(avg_data[0][2]) == -0.50
        # C0 Valence: 0.25 * 4.1 + 0.75 * 3.9 = 1.025 + 2.925 = 3.95
        assert pytest.approx(avg_data[0][3]) == 3.95


def test_rmsd_ensemble_averaging(tmp_path, enan_checkpoint, capsys):
    from frogchem.cli.calc_rmsd import CalcRmsd

    p1 = tmp_path / "conf_1" / "protocol_1" / "geom.xyz"
    p2 = tmp_path / "conf_2" / "protocol_1" / "geom.xyz"
    p1.parent.mkdir(parents=True)
    p2.parent.mkdir(parents=True)
    p1.write_text("1\nconf1\nC 0.0 0.0 0.0\n")
    p2.write_text("1\nconf2\nC 0.0 0.0 0.0\n")

    tool = CalcRmsd()

    with patch('frogchem.cli.calc_rmsd.calc_distance_matrix', return_value=np.array([[0.0]])):
        with patch('frogchem.cli.calc_rmsd.calc_rmsd', side_effect=[0.4, 0.8]):
            tool.main([str(p1), str(p2), '-c', enan_checkpoint, '--format', 'json'])

    tables = json.loads(capsys.readouterr().out)
    rmsd_table = [t for t in tables if t['title'].startswith('λ-RMSD')][-1]
    avg_row = rmsd_table['rows'][-1]
    assert avg_row['Structure'] == '⟨Ensemble Average⟩'
    # 0.25 * 0.4 + 0.75 * 0.8 = 0.1 + 0.6 = 0.7
    assert avg_row['λ-RMSD (Å)'] == pytest.approx(0.7)

