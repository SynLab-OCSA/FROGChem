import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from frogchem.orca_parser.nmr import NMRAnalysis
from frogchem.ensemble import extract_conf_and_protocol, load_checkpoint_populations
from frogchem.nmr import (
    convert_shielding_to_shift,
    extract_reference_shieldings,
    resolve_reference_shielding,
)


SAMPLE_ORCA_NMR = """
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  C      0.000000    0.000000    0.000000
  H      0.000000    0.000000    1.090000
  H      1.027000    0.000000   -0.363000
  H     -0.513000    0.890000   -0.363000
  H     -0.513000   -0.890000   -0.363000

--------------------------------
CHEMICAL SHIELDING SUMMARY (ppm)
--------------------------------


  Nucleus  Element    Isotropic     Anisotropy
  -------  -------  ------------   ------------
      1       H           31.250          5.120 
      2       H           31.100          5.200 
      3       H           31.100          5.200 
      4       H           31.100          5.200 

-----------------------------------------------------------------------
                     NMR SPIN-SPIN COUPLING CONSTANTS
                     ================================
-----------------------------------------------------------
 NUCLEUS A = H    1 NUCLEUS B = H    2 
-----------------------------------------------------------
Total spin-spin coupling tensor J (Hz):
        -12.5000        0.0000        0.0000
         0.0000       -12.5000        0.0000
         0.0000        0.0000       -12.5000

 Diagonalized JT*J matrix:
 
 J[1,2](Total)         -12.500          -12.500          -12.500  iso=     -12.500
"""


def test_nmr_parsing():
    geom = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.027, 0.0, -0.363],
        [-0.513, 0.89, -0.363],
        [-0.513, -0.89, -0.363],
    ]
    atoms = ['C', 'H', 'H', 'H', 'H']

    nmr = NMRAnalysis(SAMPLE_ORCA_NMR, geom, atoms)
    
    # 1. Shifts check
    assert 1 in nmr.shifts
    assert nmr.shifts[1]['element'] == 'H'
    assert nmr.shifts[1]['iso'] == 31.250
    assert nmr.shifts[1]['aniso'] == 5.120

    # 2. J-coupling check
    assert (1, 2) in nmr.j_couplings
    assert (2, 1) in nmr.j_couplings
    assert nmr.j_couplings[(1, 2)] == -12.500

    # 3. Tables and filters
    shifts_table = nmr.get_shifts_table(atoms=[1, 2])
    assert 'H2' in shifts_table
    assert 'H3' in shifts_table
    assert '31.250' in shifts_table

    j_matrix = nmr.get_j_matrix(atoms=[1, 2])
    assert '-12.500' in j_matrix

    dist_matrix = nmr.get_distances(atoms=[1, 2])
    assert 'H2' in dist_matrix
    assert 'H3' in dist_matrix


def test_ensemble_extraction():
    p1 = Path("/path/to/conf_3/protocol_2/3_p2_orca.out")
    c_id, proto_id = extract_conf_and_protocol(p1)
    assert c_id == 3
    assert proto_id == 2

    p2 = Path("conf_10/protocol_1/job.out")
    c_id, proto_id = extract_conf_and_protocol(p2)
    assert c_id == 10
    assert proto_id == 1

    p3 = Path("1_p2_orca.out")
    c_id, proto_id = extract_conf_and_protocol(p3)
    assert c_id == 1
    assert proto_id == 2


def test_checkpoint_population_loading():
    ckpt_data = {
        "1": {
            "number": 1,
            "energies": {
                "1": {"Pop": 30.0},
                "2": {"Pop": 70.0}
            }
        },
        "2": {
            "number": 2,
            "energies": {
                "1": {"Pop": 70.0},
                "2": {"Pop": 30.0}
            }
        }
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(ckpt_data, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Default loads latest protocol (2)
        pops_default = load_checkpoint_populations(tmp_path)
        assert pops_default[1] == 70.0
        assert pops_default[2] == 30.0

        # Specify protocol 1
        pops_p1 = load_checkpoint_populations(tmp_path, protocol=1)
        assert pops_p1[1] == 30.0
        assert pops_p1[2] == 70.0
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_clicommand_ensemble_averaging():
    from frogchem.cli.base import CLICommand

    class DummyTool(CLICommand, name='dummy-tool'):
        def run(self, args):
            pass

    tool = DummyTool()
    tool._output_format = 'table'
    tool._ensemble_weights_printed = True

    ckpt_data = {
        "1": {"energies": {"1": {"Pop": 25.0}}},
        "2": {"energies": {"1": {"Pop": 75.0}}}
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(ckpt_data, tmp)
        tmp_path = Path(tmp.name)

    try:
        tool._checkpoint = str(tmp_path)
        tool._read_boltzmann = "1"

        data = [
            ["conf_1/protocol_1/1.out", 10.0, 20.0, "C"],
            ["conf_2/protocol_1/2.out", 20.0, 40.0, "C"],
        ]
        headers = ["Structure", "Val1", "Val2", "Element"]
        averaged_data = tool._apply_ensemble_average(data, headers)

        # 1 original row for each conf + 1 average row
        assert len(averaged_data) == 3
        avg_row = averaged_data[-1]
        assert avg_row[0] == '⟨Ensemble Average⟩'
        # Val1: 0.25 * 10 + 0.75 * 20 = 2.5 + 15 = 17.5
        assert pytest.approx(avg_row[1]) == 17.5
        # Val2: 0.25 * 20 + 0.75 * 40 = 5 + 30 = 35.0
        assert pytest.approx(avg_row[2]) == 35.0
        # Element: common "C"
        assert avg_row[3] == "C"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_reference_shielding_numeric():
    # Direct float
    assert resolve_reference_shielding(31.8, '1H') == 31.8
    assert resolve_reference_shielding("31.8", '1H') == 31.8
    assert resolve_reference_shielding(185.5, '13C') == 185.5
    assert resolve_reference_shielding(None, '1H') == 31.8
    assert resolve_reference_shielding(None, '13C') == 185.0
    assert resolve_reference_shielding(None, '29Si') == 360.0

    # convert_shielding_to_shift with numeric
    assert convert_shielding_to_shift(28.0, '1H', ref_sigma=31.8) == pytest.approx((31.8 - 28.0) / (1 - 31.8e-6))
    assert convert_shielding_to_shift(28.0, '1H', ref_sigma="31.8") == pytest.approx((31.8 - 28.0) / (1 - 31.8e-6))


def test_reference_shielding_from_calculation_object():
    # Mock calculation with TMS-like data (1 Si, 4 C, 12 H)
    class MockNMR:
        def __init__(self):
            self.shifts = {
                0: {'element': 'Si', 'iso': 365.4, 'aniso': 1.0},
                1: {'element': 'C', 'iso': 184.0, 'aniso': 2.0},
                2: {'element': 'C', 'iso': 184.4, 'aniso': 2.0},
                3: {'element': 'C', 'iso': 184.0, 'aniso': 2.0},
                4: {'element': 'C', 'iso': 184.4, 'aniso': 2.0},
                5: {'element': 'H', 'iso': 31.5, 'aniso': 0.5},
                6: {'element': 'H', 'iso': 31.7, 'aniso': 0.5},
            }

    class MockCalc:
        def __init__(self):
            self.nmr = MockNMR()

    calc = MockCalc()
    ref_map = extract_reference_shieldings(calc)
    assert 'H' in ref_map
    assert '1H' in ref_map
    assert 'C' in ref_map
    assert '13C' in ref_map
    assert 'Si' in ref_map
    assert '29Si' in ref_map

    assert pytest.approx(ref_map['H']) == 31.6
    assert pytest.approx(ref_map['C']) == 184.2
    assert pytest.approx(ref_map['Si']) == 365.4

    # Resolve specific nucleus
    assert pytest.approx(resolve_reference_shielding(calc, '1H')) == 31.6
    assert pytest.approx(resolve_reference_shielding(calc, 'H')) == 31.6
    assert pytest.approx(resolve_reference_shielding(calc, '13C')) == 184.2
    assert pytest.approx(resolve_reference_shielding(calc, '29Si')) == 365.4

    # Convert shifts using the calculation object
    assert pytest.approx(convert_shielding_to_shift(25.0, '1H', ref_sigma=calc)) == (31.6 - 25.0) / (1 - 31.6e-6)
    assert pytest.approx(convert_shielding_to_shift(120.0, '13C', ref_sigma=calc)) == (184.2 - 120.0) / (1 - 184.2e-6)
    assert pytest.approx(convert_shielding_to_shift(300.0, '29Si', ref_sigma=calc)) == (365.4 - 300.0) / (1 - 365.4e-6)

    # Missing nucleus in reference raises ValueError
    with pytest.raises(ValueError, match="does not contain NMR shielding data for nucleus '19F'"):
        resolve_reference_shielding(calc, '19F')


# files/ is gitignored (local test data): tests that read it are skipped elsewhere, e.g. in CI
needs_nmr_out = pytest.mark.skipif(not Path("files/nmr.out").exists(), reason="files/nmr.out not found")


@needs_nmr_out
def test_reference_shielding_from_file():
    # Using files/nmr.out
    ref_file = "files/nmr.out"
    ref_h = resolve_reference_shielding(ref_file, '1H')
    # Mean of 16 protons in nmr.out should be ~25.6 ppm
    assert 20.0 < ref_h < 30.0

    shift = convert_shielding_to_shift(26.121, '1H', ref_sigma=ref_file)
    assert pytest.approx(shift) == (ref_h - 26.121) / (1 - ref_h * 1e-6)


def test_multiple_reference_specifications():
    # 1. Comma-separated string with key=value pairs
    ref_str = "1H=31.8, 13C=185.0, 31P=280.5"
    d = extract_reference_shieldings(ref_str)
    assert pytest.approx(d['1H']) == 31.8
    assert pytest.approx(d['H']) == 31.8
    assert pytest.approx(d['13C']) == 185.0
    assert pytest.approx(d['C']) == 185.0
    assert pytest.approx(d['31P']) == 280.5
    assert pytest.approx(d['P']) == 280.5

    assert pytest.approx(resolve_reference_shielding(ref_str, '1H')) == 31.8
    assert pytest.approx(resolve_reference_shielding(ref_str, '13C')) == 185.0
    assert pytest.approx(resolve_reference_shielding(ref_str, '31P')) == 280.5

    # 2. List of key=value pairs (as received from CLI nargs='+')
    ref_list = ["1H=31.8", "13C=185.0"]
    assert pytest.approx(resolve_reference_shielding(ref_list, '1H')) == 31.8
    assert pytest.approx(resolve_reference_shielding(ref_list, '13C')) == 185.0

    # 3. Dict input
    ref_dict = {"1H": 31.8, "13C": 185.0}
    assert pytest.approx(resolve_reference_shielding(ref_dict, '1H')) == 31.8
    assert pytest.approx(resolve_reference_shielding(ref_dict, '13C')) == 185.0

    # 4. Converting shifts with multi-ref
    assert pytest.approx(convert_shielding_to_shift(25.0, '1H', ref_sigma=ref_str)) == (31.8 - 25.0) / (1 - 31.8e-6)
    assert pytest.approx(convert_shielding_to_shift(120.0, '13C', ref_sigma=ref_str)) == (185.0 - 120.0) / (1 - 185.0e-6)


@needs_nmr_out
def test_mixed_numeric_and_file_reference():
    # Mixed numeric and file reference: 1H=31.8, 13C=files/nmr.out
    ref_mixed = ["1H=31.8", "13C=files/nmr.out"]
    assert pytest.approx(resolve_reference_shielding(ref_mixed, '1H')) == 31.8
    ref_c = resolve_reference_shielding(ref_mixed, '13C')
    assert 50.0 < ref_c < 100.0


def test_nuclei_parsing_and_normalization():
    from frogchem.nmr import parse_nuclei_list, normalize_nucleus_name, extract_element_from_nucleus

    # Normalization
    assert normalize_nucleus_name('1h') == '1H'
    assert normalize_nucleus_name('c') == '13C'
    assert normalize_nucleus_name('s') == '33S'
    assert normalize_nucleus_name('b') == '11B'
    assert normalize_nucleus_name('o') == '17O'
    assert normalize_nucleus_name('33S') == '33S'
    assert normalize_nucleus_name('19F') == '19F'

    # Element extraction
    assert extract_element_from_nucleus('1H') == 'H'
    assert extract_element_from_nucleus('13C') == 'C'
    assert extract_element_from_nucleus('33S') == 'S'
    assert extract_element_from_nucleus('31P') == 'P'
    assert extract_element_from_nucleus('77Se') == 'Se'

    # parse_nuclei_list
    assert parse_nuclei_list(None) == ['1H']
    assert parse_nuclei_list('1H') == ['1H']
    assert parse_nuclei_list(['1H', '13C', '33S']) == ['1H', '13C', '33S']
    assert parse_nuclei_list('1H, 13C, 33S') == ['1H', '13C', '33S']
    assert parse_nuclei_list('1H 13C 33S') == ['1H', '13C', '33S']
    assert parse_nuclei_list(['1H,13C', '33S']) == ['1H', '13C', '33S']


@needs_nmr_out
def test_calc_nmr_cli_cartesian_product(tmp_path):
    from frogchem.cli.calc_nmr import CalcNmr
    out_stem = str(tmp_path / "cartesian.png")
    
    CalcNmr().main([
        "files/nmr.out",
        "-n", "1H", "13C", "33S",
        "--all-spectra",
        "--plot", out_stem,
        "-q",
    ])

    # Check 1D plots generated
    assert (tmp_path / "cartesian_1d_1h.png").exists()
    assert (tmp_path / "cartesian_1d_13c.png").exists()
    assert (tmp_path / "cartesian_1d_33s.png").exists()

    # Check 2D homonuclear plots generated
    assert (tmp_path / "cartesian_cosy_1h_1h.png").exists()
    assert (tmp_path / "cartesian_noesy_1h_1h.png").exists()
    assert (tmp_path / "cartesian_cosy_13c_13c.png").exists()
    assert (tmp_path / "cartesian_cosy_33s_33s.png").exists()

    # Check 2D heteronuclear Cartesian plots generated
    assert (tmp_path / "cartesian_hsqc_1h_13c.png").exists()
    assert (tmp_path / "cartesian_hmbc_1h_13c.png").exists()
    assert (tmp_path / "cartesian_hmbc_1h_33s.png").exists()
    assert (tmp_path / "cartesian_hsqc_13c_33s.png").exists()
    assert (tmp_path / "cartesian_hmbc_13c_33s.png").exists()





