import numpy as np
import pytest

from frogchem.gaussian_parser.hessian import get_freq, get_ir_intensities
from frogchem.gaussian_parser.tddft import parse_uv, parse_ecd
from frogchem.gaussian_parser.regex_gaussian import REGEXS


SAMPLE_GAUSSIAN_FREQ = """
 Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering
 activities (A**4/AMU), depolarization ratios for plane and unpolarized
 incident light, reduced masses (AMU), force constants (mDyne/A)
 and normal coordinates:
                     1                      2                      3
                    A1                     B2                     A1
 Frequencies --   1605.4231              3778.2145              3920.8521
 Red. masses --      1.0825                 1.0456                 1.0812
 Frc consts  --      1.6452                 8.7942                 9.7845
 IR Inten    --      5.3210                32.1450                45.6780
 Raman Activ --      8.4200                 2.3100                18.5200



"""

SAMPLE_GAUSSIAN_TDDFT = """
 Excitation energies and oscillator strengths:

 Excited State   1:      Singlet-A      3.4560 eV   358.75 nm  f=0.0450  <S**2>=0.000
       5 ->  6         0.70710
 Excited State   2:      Singlet-B      4.1230 eV   300.71 nm  f=0.1250  <S**2>=0.000
       4 ->  6         0.65432
***
"""

SAMPLE_GAUSSIAN_ECD = SAMPLE_GAUSSIAN_TDDFT + """
 <0|del|b> * <b|rxdel|0> + <0|del|b> * <b|delr+rdel|0>
 1   0.1234   0.5678   0.9101   -12.3456
 2   0.2345   0.6789   0.1234    25.6789
1/2[<0|r|b>*<b|rxdel|0> + (<0|rxdel|b>*<b|r|0>)*]
"""


class TestGaussianHessian:
    def test_get_freq(self):
        freqs = get_freq(SAMPLE_GAUSSIAN_FREQ, REGEXS)
        assert len(freqs) == 3
        assert freqs[0] == 1605.4231
        assert freqs[2] == 3920.8521

    def test_get_freq_missing(self):
        freqs = get_freq("No frequency section here", REGEXS)
        assert len(freqs) == 0

    def test_get_ir_intensities(self):
        ir = get_ir_intensities(SAMPLE_GAUSSIAN_FREQ, REGEXS)
        assert ir.shape == (3, 2)
        assert ir[0, 0] == 1605.4231
        assert ir[0, 1] == 5.3210
        assert ir[2, 1] == 45.6780

    def test_get_ir_intensities_missing(self):
        ir = get_ir_intensities("No freq section", REGEXS)
        assert ir.shape == (1, 2)


class TestGaussianTDDFT:
    def test_parse_uv(self):
        uv = parse_uv(SAMPLE_GAUSSIAN_TDDFT, REGEXS)
        assert uv.shape == (2, 2)
        assert uv[0, 0] == 3.4560
        assert uv[0, 1] == 0.0450
        assert uv[1, 0] == 4.1230
        assert uv[1, 1] == 0.1250

    def test_parse_uv_missing(self):
        uv = parse_uv("No tddft", REGEXS)
        assert uv.shape == (1, 2)

    def test_parse_ecd(self):
        ecd = parse_ecd(SAMPLE_GAUSSIAN_ECD, REGEXS)
        assert ecd.shape == (2, 2)
        assert ecd[0, 0] == 3.4560
        assert ecd[0, 1] == -12.3456
        assert ecd[1, 1] == 25.6789

    def test_parse_ecd_prefers_length_gauge(self):
        # G16 prints R(velocity), then R(length), then the dipole strengths: R(length) is used (as ORCA)
        sample = SAMPLE_GAUSSIAN_ECD + """ 1   0.1000   0.2000   0.3000    -3.1200
 2   0.1000   0.2000   0.3000    22.9100
  1/2[<0|del|b>*<b|r|0> + (<0|r|b>*<b|del|0>)*] (Au)
 1   0.0001   0.0025   0.0018    0.0045    0.0030
 2   0.0080   0.0457   0.0105    0.0642    0.0428
"""
        ecd = parse_ecd(sample, REGEXS)
        assert ecd[0, 1] == -3.12
        assert ecd[1, 1] == 22.91

    def test_parse_ecd_missing(self):
        ecd = parse_ecd("No ecd", REGEXS)
        assert ecd.shape == (1, 2)


def _orientation(header, rows, atomic_type=True):
    lines = [f"                         {header}",
             " " + "-" * 69,
             " Center     Atomic      Atomic             Coordinates (Angstroms)",
             " Number     Number       Type             X           Y           Z",
             " " + "-" * 69]
    for i, (z, x, y, w) in enumerate(rows, 1):
        typ = "           0" if atomic_type else ""
        lines.append(f" {i:6d} {z:10d}{typ}    {x:12.6f}{y:12.6f}{w:12.6f}")
    lines.append(" " + "-" * 69)
    return "\n".join(lines) + "\n"


WATER = [(8, 0.0, 0.0, 0.117), (1, 0.0, 0.757, -0.469), (1, 0.0, -0.757, -0.469)]
WATER_STD = [(8, 0.0, 0.0, 0.119), (1, 0.0, 0.763, -0.477), (1, 0.0, -0.763, -0.477)]
# Text after the last table that the old parser read as atoms (6+ columns, integer second field)
TRAILER = """
 Rotational constants (GHZ):    835.0510100    434.1200300    285.6131800
 Charge =  0 Multiplicity = 1
      1        792         1         2          3         4
 Normal termination of Gaussian 16
"""


def _write_log(tmp_path, text):
    path = tmp_path / "job.log"
    path.write_text(" Entering Gaussian System, Link 0=g16\n" + text)
    return str(path)


class TestGaussianGeometry:
    def test_only_standard_orientation(self, tmp_path):
        """opt + freq jobs with symmetry print no 'Input orientation' (06-conf27.log)."""
        from frogchem.gaussian_parser.calculation import GaussianCalculation
        calc = GaussianCalculation(_write_log(tmp_path, _orientation("Standard orientation:", WATER_STD) + TRAILER))
        assert list(calc.atoms) == ["O", "H", "H"]
        assert calc.geom.shape == (3, 3)
        assert calc.geom[1, 1] == pytest.approx(0.763)

    def test_input_orientation_preferred_and_atoms_not_doubled(self, tmp_path):
        from frogchem.gaussian_parser.calculation import GaussianCalculation
        text = (_orientation("Input orientation:", WATER) + _orientation("Standard orientation:", WATER_STD)
                + TRAILER)
        calc = GaussianCalculation(_write_log(tmp_path, text))
        assert list(calc.atoms) == ["O", "H", "H"]
        assert calc.geom[1, 1] == pytest.approx(0.757)

    def test_last_table_is_used(self, tmp_path):
        from frogchem.gaussian_parser.opt import parse_orientation
        moved = [(z, x, y + 1.0, w) for z, x, y, w in WATER]
        text = _orientation("Input orientation:", WATER) + _orientation("Input orientation:", moved)
        numbers, coords = parse_orientation(text, REGEXS)
        assert list(numbers) == [8, 1, 1]
        assert coords[1, 1] == pytest.approx(1.757)

    def test_old_format_without_atomic_type_column(self):
        from frogchem.gaussian_parser.opt import parse_orientation
        numbers, coords = parse_orientation(_orientation("Z-Matrix orientation:", WATER, atomic_type=False),
                                            REGEXS)
        assert list(numbers) == [8, 1, 1]
        assert coords.shape == (3, 3)

    def test_no_geometry_is_a_parse_error(self, tmp_path):
        from frogchem.exceptions import ParseError
        from frogchem.gaussian_parser.calculation import GaussianCalculation
        with pytest.raises(ParseError):
            GaussianCalculation(_write_log(tmp_path, TRAILER)).atoms

    def test_charge_and_multiplicity(self, tmp_path):
        from frogchem.gaussian_parser.calculation import GaussianCalculation
        text = (" Charge =  0 Multiplicity = 1\n" + _orientation("Standard orientation:", WATER_STD)
                + " Charge = -1 Multiplicity = 2\n")
        calc = GaussianCalculation(_write_log(tmp_path, text))
        assert (calc.charge, calc.mult) == (-1, 2)              # last job of the file
