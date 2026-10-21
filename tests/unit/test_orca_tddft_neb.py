import numpy as np
import pytest
from pathlib import Path
from frogchem.orca_parser.calculation import Calculation

from frogchem.orca_parser.tddft import parse_tddft_states, get_tddft_soc, parse_tddft_fosc
from frogchem.orca_parser.neb import parse_neb_interp
from frogchem.orca_parser.hessian import get_freq, get_normal_modes, get_hessian_graphs_impulses
from frogchem.orca_parser.regex_ORCA import REGEXS


ORCA_REG = REGEXS['6']

SAMPLE_TDDFT_OUTPUT = """
TD-DFT EXCITED STATES (SINGLETS)
--------------------------------

STATE  1:  E=   0.123456 au     3.359 eV    27100.0 cm**-1   <S**2> =   0.0000   Mult 1
   012a -> 013a  :     0.7071
   011a -> 014a  :     0.7071

STATE  2:  E=   0.150000 au     4.082 eV    32920.0 cm**-1   <S**2> =   0.0000   Mult 1
   012a -> 014a  :     0.9500

TD-DFT-EXCITATION SPECTRA

TD-DFT EXCITED STATES (TRIPLETS)
--------------------------------

STATE  1:  E=   0.090000 au     2.449 eV    19750.0 cm**-1   <S**2> =   2.0000   Mult 3
   012a -> 013a  :     1.0000

TD-DFT SPIN-ORBIT COUPLING

CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS
----------------------------------------------
  1  0   (   0.00,   12.50)  (   5.00,    0.00)  (   0.00,    0.00)
  1  1   (   1.00,    2.00)  (   0.50,    0.00)  (   0.00,    1.50)

SOC stabilization

 ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
 ----------------------------------------------------------
         States           Energy   Wavelength   dummy    fosc
   0-1A  ->   1-1A        3.359     369.1      1000.0   0.0450
   0-1A  ->   2-1A        4.082     303.8      2000.0   0.1200

 CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
 --------------------------------------------------
         States           Energy   Wavelength   dummy     R
   0-1A  ->   1-1A        3.359     369.1      1000.0   -15.40
   0-1A  ->   2-1A        4.082     303.8      2000.0    22.80

"""

SAMPLE_HESSIAN_OUTPUT = """
VIBRATIONAL FREQUENCIES
-----------------------

   0:    0.00 cm**-1
   1:    0.00 cm**-1
   2:    0.00 cm**-1
   3:  150.25 cm**-1
   4:  500.50 cm**-1
   5: 1650.00 cm**-1



 Mode   freq       eps      Int      T**2         TX        TY        TZ
-------------------------------------------------------------------------
    0:      0.00   0.000     0.00    0.0000     -0.00     -0.00     -0.00
    1:      0.00   0.000     0.00    0.0000     -0.00     -0.00     -0.00
    2:      0.00   0.000     0.00    0.0000     -0.00     -0.00     -0.00
    3:    150.25   0.000    12.34    0.1234      0.10      0.05      0.02
    4:    500.50   0.000    45.67    0.5678      0.30      0.20      0.10


NORMAL MODES
------------

         0         1
    0  0.10000   0.00000
    1  0.20000   0.15000



"""


class TestOrcaTDDFT:
    def test_parse_tddft_states(self):
        states = parse_tddft_states(SAMPLE_TDDFT_OUTPUT, ORCA_REG)
        assert 1 in states  # Singlets
        assert 3 in states  # Triplets

        # Singlet 1
        s1 = states[1][1]
        assert s1['energy'] == 3.359
        assert len(s1['population']) == 2
        assert s1['population'][0] == (12, 13, 0.7071)

        # Triplet 1
        t1 = states[3][1]
        assert t1['energy'] == 2.449
        assert len(t1['population']) == 1

    def test_tddft_soc_and_fosc(self):
        states = parse_tddft_states(SAMPLE_TDDFT_OUTPUT, ORCA_REG)
        soc_matrix = get_tddft_soc(SAMPLE_TDDFT_OUTPUT, ORCA_REG, states)
        assert soc_matrix.shape[0] == 3  # len(singlets) + 1
        assert soc_matrix.shape[1] == 1  # len(triplets)
        assert soc_matrix[0, 0] > 0  # S0 - T1 coupling
        assert soc_matrix[1, 0] > 0  # S1 - T1 coupling

        parse_tddft_fosc(SAMPLE_TDDFT_OUTPUT, states)
        assert states[1][1]['fosc'] == 0.045
        assert states[1][2]['fosc'] == 0.12
        assert states[1][1]['R'] == -15.4
        assert states[1][2]['R'] == 22.8

    def test_tddft_soc_norm_formula_exact(self):
        # Test case where Re_y != 0 to ensure re_y**2 is not duplicated
        # Z = (1.0, 2.0), X = (3.0, 4.0), Y = (5.0, 6.0)
        # Expected norm = sqrt(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) = sqrt(91) = 9.539392014169456
        soc_text = """
CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS
----------------------------------------------
  1  0   (   1.00,    2.00)  (   3.00,    4.00)  (   5.00,    6.00)

SOC stabilization
"""
        states = {1: {1: {}}, 3: {1: {}}}
        soc = get_tddft_soc(soc_text, ORCA_REG, states)
        expected = np.sqrt(1.0**2 + 2.0**2 + 3.0**2 + 4.0**2 + 5.0**2 + 6.0**2)
        assert np.isclose(soc[0, 0], expected)
        assert np.isclose(soc[0, 0], 9.539392014169456)

    def test_soc_out_file_if_present(self):
        from pathlib import Path
        from frogchem.orca_parser.calculation import Calculation
        soc_file = Path("files/soc.out")
        if not soc_file.exists():
            soc_file = Path(__file__).resolve().parent.parent.parent / "files" / "soc.out"
        if not soc_file.exists():
            pytest.skip("soc.out not found in files/")

        calc = Calculation(str(soc_file))
        assert calc.soc is True
        assert hasattr(calc, 'tddft_soc')
        assert calc.tddft_soc.shape == (2, 1)
        # S0-T1 and S1-T1 couplings
        assert np.isclose(calc.tddft_soc[0, 0], 6.857295, atol=1e-3)
        assert np.isclose(calc.tddft_soc[1, 0], 1.040192, atol=1e-3)

        # Ensure triplet transitions are not duplicated
        assert len(calc.tddft_states[3][1]['population']) == 3

        # Ensure fosc and R are extracted correctly from uncorrected spectra
        assert np.isclose(calc.tddft_states[1][1]['fosc'], 0.080, atol=1e-3)
        assert np.isclose(calc.tddft_states[1][1]['R'], 20.696, atol=1e-2)

    def test_soc_state_selection_and_filtering(self):
        from frogchem.cli.tddft_parser import parse_state_selection, filter_soc_table

        # 1. Parsing token combinations
        sel_s, sel_t = parse_state_selection(soc_states=["S0", "S1", "T1", "T3"])
        assert sel_s == {0, 1}
        assert sel_t == {1, 3}

        # Range syntax
        sel_s, sel_t = parse_state_selection(soc_states=["S0-S2", "T1-T3"])
        assert sel_s == {0, 1, 2}
        assert sel_t == {1, 2, 3}

        # Only singlets specified -> triplets remain None (all)
        sel_s, sel_t = parse_state_selection(soc_states=["S0", "S1"])
        assert sel_s == {0, 1}
        assert sel_t is None

        # Only triplets specified -> singlets remain None (all)
        sel_s, sel_t = parse_state_selection(soc_states=["T2"])
        assert sel_s is None
        assert sel_t == {2}

        # 2. Filtering table
        headers = ['State', 'T1', 'T2', 'T3']
        rows = [
            ['S0', 1.0, 2.0, 3.0],
            ['S1', 4.0, 5.0, 6.0],
            ['S2', 7.0, 8.0, 9.0],
        ]
        filt_headers, filt_rows = filter_soc_table(headers, rows, sel_s={0, 2}, sel_t={1, 3})
        assert filt_headers == ['State', 'T1', 'T3']
        assert len(filt_rows) == 2
        assert filt_rows[0] == ['S0', 1.0, 3.0]
        assert filt_rows[1] == ['S2', 7.0, 9.0]

    def test_homo_lumo_and_transition_labels(self):
        from frogchem.cli.tddft_parser import format_orbital_label, format_transition, prepare_states_table

        # Test format_orbital_label with homo = 50 (so lumo = 51)
        assert format_orbital_label(50, homo=50) == "H"
        assert format_orbital_label(49, homo=50) == "H-1"
        assert format_orbital_label(48, homo=50) == "H-2"
        assert format_orbital_label(46, homo=50) == "H-4"
        assert format_orbital_label(51, homo=50) == "L"
        assert format_orbital_label(52, homo=50) == "L+1"
        assert format_orbital_label(53, homo=50) == "L+2"
        assert format_orbital_label(55, homo=50) == "L+4"
        # Fallback when homo is None
        assert format_orbital_label(50, homo=None) == "50"
        assert format_orbital_label(51, homo=None) == "51"

        # Test format_transition
        assert format_transition(50, 51, 0.93, homo=50) == "H->L (0.93)"
        assert format_transition(49, 51, 0.99, homo=50) == "H-1->L (0.99)"
        assert format_transition(50, 52, 0.21, homo=50) == "H->L+1 (0.21)"
        assert format_transition(50, 51, 0.93, homo=None) == "50->51 (0.93)"

        # Test prepare_states_table with and without homo
        dummy_states = {
            1: {
                1: {
                    "energy": 3.14,
                    "fosc": 0.08,
                    "R": 20.0,
                    "S**2": 0.0,
                    "population": [(99, 100, 0.94), (98, 100, 0.30)],
                }
            }
        }
        # With homo=99
        rows_hl = prepare_states_table(dummy_states, threshold=0.20, homo=99)
        assert rows_hl[0][6] == "H->L (0.94), H-1->L (0.30)"
        # Without homo (raw)
        rows_raw = prepare_states_table(dummy_states, threshold=0.20, homo=None)
        assert rows_raw[0][6] == "99->100 (0.94), 98->100 (0.30)"

    def test_calculation_homo_property(self):
        tddft_file = Path("files/tddft.out")
        if tddft_file.exists():
            calc = Calculation(str(tddft_file))
            assert calc.homo in (50, 51)
            assert calc.lumo in (51, 52)





class TestOrcaNEB:
    def test_parse_neb_interp(self, tmp_path):
        interp_file = tmp_path / "neb.interp"
        content = """Iteration: 0
------------------------------------------------------------------
0   0.000000   -1.000000
1   0.500000   -0.980000
2   1.000000   -1.005000

Interpolated Path
------------------------------------------------------------------
0.000000   0.000000   -1.000000
0.250000   0.250000   -0.985000
0.500000   0.500000   -0.980000
0.750000   0.750000   -0.990000
1.000000   1.000000   -1.005000

"""
        interp_file.write_text(content)
        imags, interpols = parse_neb_interp(str(interp_file))
        assert len(imags) == 1
        assert len(interpols) == 1
        assert imags[0].shape == (3, 2)
        assert interpols[0].shape == (5, 2)
        # Check conversion to kcal/mol
        np.testing.assert_allclose(imags[0][0, 0], 0.0)
        assert imags[0][0, 1] < 0  # Energy in kcal/mol


    def test_parse_neb_interp_orca_format(self, tmp_path):
        # Layout written by ORCA (<basename>.interp): one header line per block, no separators
        interp_file = tmp_path / "neb.interp"
        interp_file.write_text(
            "Iteration: 0\n"
            "Images: Distance  (Bohr), Energy (Eh)\n"
            "0.000000   0.000000   0.000000\n"
            "0.500000   1.200000   0.031000\n"
            "1.000000   2.400000   -0.010000\n"
            "\n"
            "Interp.: Distance (Bohr), Energy (Eh)\n"
            "0.000000   0.000000   0.000000\n"
            "0.250000   0.600000   0.018000\n"
            "0.500000   1.200000   0.031000\n"
            "0.750000   1.800000   0.012000\n"
            "1.000000   2.400000   -0.010000\n"
            "\n"
        )
        imags, interpols = parse_neb_interp(str(interp_file))
        assert imags[0].shape == (3, 2)
        assert interpols[0].shape == (5, 2)
        np.testing.assert_allclose(interpols[0][:, 0], [0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(interpols[0][0, 1], 0.0)

class TestOrcaHessian:
    def test_get_freq(self):
        modes, freqs = get_freq(SAMPLE_HESSIAN_OUTPUT, ORCA_REG)
        assert len(modes) == 6
        assert len(freqs) == 6
        assert freqs[3] == 150.25
        assert freqs[5] == 1650.0

    def test_get_normal_modes(self):
        modes = get_normal_modes(SAMPLE_HESSIAN_OUTPUT, ORCA_REG, N=2)
        assert modes.shape == (2, 2)
        assert modes[0, 0] == 0.1
        assert modes[1, 1] == 0.15

    def test_get_hessian_graphs_impulses(self):
        graphs = get_hessian_graphs_impulses(SAMPLE_HESSIAN_OUTPUT, ORCA_REG)
        assert 'ir' in graphs
        assert len(graphs['ir']) == 4
        assert graphs['ir'][2] == '12.34'

    def test_mode_localization_and_ipr(self):
        from frogchem.orca_parser.hessian import (
            localize_mode,
            localize_mode_fragment,
            inverse_participation_ratio,
            localize_all_modes,
        )
        from types import SimpleNamespace
        # Mock calculation with 2 atoms and 6 normal modes
        mock_calc = SimpleNamespace(
            atoms=['O', 'H'],
            geom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            normal_modes=np.zeros((6, 2)),
        )
        mock_calc.normal_modes[:, 0] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        mock_calc.normal_modes[:, 1] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        loc = localize_mode(mock_calc, 0)
        assert loc.shape == (2,)
        assert np.isclose(loc[0], 100.0)
        assert np.isclose(loc[1], 0.0)

        ipr = inverse_participation_ratio(mock_calc, 0)
        assert np.isclose(ipr, 1.0)

        frag = localize_mode_fragment(mock_calc, 0, {'frag_O': [0], 'frag_H': [1]})
        assert np.isclose(frag['frag_O'], 100.0)
        assert np.isclose(frag['frag_H'], 0.0)

        all_res = localize_all_modes(mock_calc, {'frag_O': [0]})
        assert len(all_res['ipr']) == 2
        assert all_res['atomic'].shape == (2, 2)
        assert 'fragments' in all_res

    def test_displace_geometry(self, tmp_path, monkeypatch):
        from frogchem.orca_parser.hessian import displace_geometry
        from types import SimpleNamespace
        monkeypatch.chdir(tmp_path)
        mock_calc = SimpleNamespace(
            atoms=['O', 'H'],
            geom=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            normal_modes=np.zeros((6, 1)),
        )
        mock_calc.normal_modes[:, 0] = [0.1, 0.0, 0.0, -0.1, 0.0, 0.0]
        res = displace_geometry(mock_calc, mode=0, max_displ=0.5, ngeom=5)
        assert res is True
        assert (tmp_path / "displ_geometry_0.xyz").exists()
