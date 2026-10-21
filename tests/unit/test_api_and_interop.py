import gzip
import pytest
from pathlib import Path

import frogchem as frog
from frogchem.parser_base import BaseParser


class DummyGzParser(BaseParser):
    def _parse_file(self):
        self.lines = [line.strip() for line in self.fl.splitlines() if line.strip()]


def test_package_version():
    assert hasattr(frog, "__version__")
    assert isinstance(frog.__version__, str)
    assert len(frog.__version__.split(".")) >= 3


def test_base_parser_gzip(tmp_path):
    gz_file = tmp_path / "sample.out.gz"
    content = "Line 1: ORCA calculation\nLine 2: Total Energy: -100.5\n"
    with gzip.open(gz_file, "wt", encoding="utf-8") as f:
        f.write(content)

    parser = DummyGzParser(str(gz_file))
    assert parser.lines == ["Line 1: ORCA calculation", "Line 2: Total Energy: -100.5"]


def test_read_detection_orca(tmp_path):
    orca_file = tmp_path / "job.out"
    orca_file.write_text("* O   R   C   A *\nProgram Version 5.0.0\n")
    calc = frog.read(str(orca_file))
    assert isinstance(calc, frog.OrcaCalculation)


def test_read_detection_gaussian(tmp_path):
    gau_file = tmp_path / "job.log"
    gau_file.write_text("Entering Gaussian System\nGaussian 16: ES64L-G16RevC.01\n")
    calc = frog.read(str(gau_file))
    assert isinstance(calc, frog.GaussianCalculation)


def test_to_ase_and_rdkit():
    opt_file = Path("files/opt.out")
    if not opt_file.exists():
        pytest.skip("files/opt.out not found in local workspace")

    calc = frog.read(str(opt_file))
    atoms = calc.to_ase()
    assert len(atoms) == len(calc.atoms)
    assert atoms.get_chemical_formula() != ""

    mol = calc.to_rdkit()
    assert mol.GetNumAtoms() == len(calc.atoms)


def test_load_calculation_xyz_and_gz(tmp_path):
    from frogchem.filetypes import load_calculation, load_structures
    xyz_path = tmp_path / "water.xyz"
    content = "3\nwater molecule\nO 0.0 0.0 0.0\nH 0.0 0.7 0.6\nH 0.0 -0.7 0.6\n"
    xyz_path.write_text(content)

    calc = load_calculation(str(xyz_path))
    assert len(calc.atoms) == 3
    assert calc.geom.shape == (3, 3)

    # test .xyz.gz
    gz_path = tmp_path / "water.xyz.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as f:
        f.write(content)

    calc_gz = load_calculation(str(gz_path))
    assert len(calc_gz.atoms) == 3

    # test load_structures
    structs = load_structures(str(gz_path))
    assert len(structs) == 1
    assert len(structs[0][0]) == 3


def test_load_calculation_gaussian_fallback_and_error(tmp_path):
    from frogchem.filetypes import load_calculation, load_structures
    from frogchem.exceptions import ParseError

    # Gaussian log file with geometry
    gau_file = tmp_path / "calc.log"
    content = """Entering Gaussian System
Gaussian 16: ES64L-G16RevC.01
                         Input orientation:                          
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.117176
      2          1           0        0.000000    0.757034   -0.468704
      3          1           0        0.000000   -0.757034   -0.468704
 ---------------------------------------------------------------------
"""
    gau_file.write_text(content)
    calc = load_calculation(str(gau_file))
    assert calc.__class__.__name__ == "GaussianCalculation"

    # load_structures from calc
    structs = load_structures(str(gau_file))
    assert len(structs) == 1

    # Completely invalid computation file raises ParseError
    bad_file = tmp_path / "bad.out"
    bad_file.write_text("gibberish that matches neither ORCA nor Gaussian\n")
    with pytest.raises(ParseError):
        load_calculation(str(bad_file))

