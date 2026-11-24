import pytest
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from frogchem.orca_parser.nbo import NBOAnalysis
from frogchem.orca_parser.calculation import Calculation
from frogchem.gaussian_parser.calculation import GaussianCalculation
from frogchem.cli.nbo_parser import NboParser


SAMPLE_WIBERG_OUTPUT = """
 Summary of Natural Population Analysis:
                                                     Natural Population
  Atom  No    Charge        Core      Valence    Rydberg      Total
 ------------------------------------------------------------------
   C     1   -0.45000     1.99990     4.43000    0.02010     6.45000
   H     2    0.15000     0.00000     0.84000    0.01000     0.85000
   H     3    0.15000     0.00000     0.84000    0.01000     0.85000
   H     4    0.15000     0.00000     0.84000    0.01000     0.85000
 ==================================================================

 NATURAL BOND ORBITAL ANALYSIS:
      (Occupancy)   Bond orbital / Coefficients / Hybrids
 ------------------ Lewis ------------------------------------------------------
   1. (1.99999) CR ( 1) C  1            s(100.00%)
   2. (1.99400) BD ( 1) C  1- H  2
   3. (1.99400) BD ( 1) C  1- H  3
   4. (1.99400) BD ( 1) C  1- H  4
   5. (1.95000) LP ( 1) C  1            s( 65.00%)p 0.54( 35.00%)
 ------------------ non-Lewis --------------------------------------------------
   6. (0.01200) BD*( 1) C  1- H  2
   7. (0.01200) BD*( 1) C  1- H  3
   8. (0.01200) BD*( 1) C  1- H  4

 SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS
  within       Threshold for printing:   0.50 kcal/mol
   5. LP ( 1) C  1 /   6. BD*( 1) C  1- H  2        4.50    0.85    0.055
   5. LP ( 1) C  1 /   7. BD*( 1) C  1- H  3        4.50    0.85    0.055

 Wiberg bond index matrix in the NAO basis:

     Atom    1       2       3       4
     ---- ------  ------  ------  ------
   1.  C  0.0000  0.9850  0.9850  0.9850
   2.  H  0.9850  0.0000  0.0010  0.0010
   3.  H  0.9850  0.0010  0.0000  0.0010
   4.  H  0.9850  0.0010  0.0010  0.0000

 Wiberg bond index, Totals by atom:

     Atom    Total
     ----  -------
   1.  C    2.9550
   2.  H    0.9870
   3.  H    0.9870
   4.  H    0.9870
"""

# Sample with multi-column chunking (11 atoms: columns 1-9 then 10-11)
SAMPLE_CHUNKED_WIBERG = """
 Summary of Natural Population Analysis:
  Atom  No    Charge        Core      Valence    Rydberg      Total
 ------------------------------------------------------------------
   C     1   -0.10000     1.99990     4.08000    0.02010     6.10000
 ==================================================================

 NATURAL BOND ORBITAL ANALYSIS:
   1. (1.99999) CR ( 1) C  1            s(100.00%)

 Wiberg bond index matrix in the NAO basis:

     Atom    1       2       3       4       5       6       7       8       9
     ---- ------  ------  ------  ------  ------  ------  ------  ------  ------
   1.  C  0.0000  1.0200  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
   2.  C  1.0200  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  10.  C  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  11.  H  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

     Atom   10      11
     ---- ------  ------
   1.  C  0.0000  0.0000
   2.  C  0.0000  0.0000
  10.  C  0.0000  1.0150
  11.  H  1.0150  0.0000

 Wiberg bond index, Totals by atom:

     Atom    Total
     ----  -------
   1.  C    1.0200
   2.  C    1.0200
  10.  C    1.0150
  11.  H    1.0150
"""


def test_wiberg_parsing_and_queries():
    nbo = NBOAnalysis(SAMPLE_WIBERG_OUTPUT)

    assert nbo.wiberg_matrix is not None
    assert nbo.wiberg_matrix.shape == (4, 4)
    assert np.isclose(nbo.wiberg_matrix[0, 1], 0.9850)
    assert np.isclose(nbo.wiberg_matrix[1, 0], 0.9850)
    assert np.isclose(nbo.wiberg_matrix[1, 2], 0.0010)
    assert np.isclose(nbo.wiberg_totals[0], 2.9550)


def test_wiberg_chunked_matrix():
    nbo = NBOAnalysis(SAMPLE_CHUNKED_WIBERG)
    assert nbo.wiberg_matrix is not None
    assert nbo.wiberg_matrix.shape == (11, 11)
    # Check bond between 1 and 2 (0 and 1 in 0-based)
    assert np.isclose(nbo.wiberg_matrix[0, 1], 1.0200)
    # Check bond in second chunk between 10 and 11 (9 and 10 in 0-based)
    assert np.isclose(nbo.wiberg_matrix[9, 10], 1.0150)
    assert np.isclose(nbo.wiberg_matrix[10, 9], 1.0150)
    assert np.isclose(nbo.wiberg_totals[9], 1.0150)


def test_wiberg_missing():
    output_no_wiberg = """
 Summary of Natural Population Analysis:
   Atom  No    Charge        Core      Valence    Rydberg      Total
 ------------------------------------------------------------------
   C     1   -0.45000     1.99990     4.43000    0.02010     6.45000
 ==================================================================
 NATURAL BOND ORBITAL ANALYSIS:
    """
    nbo = NBOAnalysis(output_no_wiberg)
    assert nbo.wiberg_matrix is None


def test_nbo_query_api():
    nbo = NBOAnalysis(SAMPLE_WIBERG_OUTPUT)

    # Bonds
    bonds_all = nbo.get_bonds()
    assert len(bonds_all) == 6  # 3 BD + 3 BD*

    bonds_c_h1 = nbo.get_bonds(atom1=0, atom2=1)
    assert len(bonds_c_h1) == 2  # 1 BD + 1 BD*
    assert "C  1- H  2" in bonds_c_h1.iloc[0]['Label']

    # Filter by bond type
    bd_only = nbo.get_bonds(bond_type="BD")
    assert len(bd_only) == 3
    assert all(bd_only['Type'] == 'BD')

    bd_star_only = nbo.get_bonds(bond_type="BD*")
    assert len(bd_star_only) == 3
    assert all(bd_star_only['Type'] == 'BD*')

    # Lone pairs
    lps = nbo.get_lone_pairs()
    assert len(lps) == 1
    assert lps.iloc[0]['Type'] == 'LP'

    lp_c0 = nbo.get_lone_pairs(atom=0)
    assert len(lp_c0) == 1

    lp_h1 = nbo.get_lone_pairs(atom=1)
    assert lp_h1.empty

    # E2 and NPA tables
    assert np.isclose(nbo.perturbation_theory['E(2)'].max(), 4.50)
    charges = nbo.npa_charges.set_index('Index')['Charge']
    assert np.isclose(charges[0], -0.45) and np.isclose(charges[1], 0.15)


def test_real_nbo_file_and_cycle_selection():
    nbo_path = Path("files/nbo.out")
    if not nbo_path.exists():
        pytest.skip("files/nbo.out not found")

    # Test Calculation property
    calc = Calculation(str(nbo_path))
    assert calc.has_nbo() is True
    assert calc.nbo is not None

    # Compare cycle 0 (first step) vs cycle -1 (last step)
    text = nbo_path.read_text(encoding="utf-8", errors="replace")
    nbo_c0 = NBOAnalysis(text, cycle=0)
    nbo_last = NBOAnalysis(text, cycle=-1)

    assert nbo_c0.cycle == 0
    assert nbo_last.cycle == -1

    # Geometry changed during optimization, so charges and energies differ
    q0_c0 = nbo_c0.npa_charges.set_index('Index')['Charge'][0]
    q0_last = nbo_last.npa_charges.set_index('Index')['Charge'][0]
    assert not np.isclose(q0_c0, q0_last, atol=1e-3)

    e2_max_c0 = nbo_c0.perturbation_theory['E(2)'].max()
    e2_max_last = nbo_last.perturbation_theory['E(2)'].max()
    assert not np.isclose(e2_max_c0, e2_max_last, atol=1e-2)

    # Bonds query on real file
    bonds_0_1 = calc.nbo.get_bonds(atom1=0, atom2=1)
    assert not bonds_0_1.empty
    assert any(bonds_0_1['Type'] == 'BD')

    # Lone pairs query
    lps = calc.nbo.get_lone_pairs()
    assert len(lps) > 0

    assert calc.nbo.perturbation_theory['E(2)'].max() >= 15.0


def test_gaussian_nbo_mock(tmp_path):
    mock_log = tmp_path / "gaussian_nbo.log"
    content = """
 Entering Gaussian System
 Gaussian 16:  ES64L-G16RevC.01
 ----------------------------------------------------------------------
 #p opt b3lyp/6-31g* pop=nbo
 ----------------------------------------------------------------------
                         Input orientation:                         
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    1.090000
 ---------------------------------------------------------------------
 Rotational constants (GHZ):      150.0000000     150.0000000     150.0000000
 Dipole moment (field-independent basis, Debye):
    X=     0.0000    Y=     0.0000    Z=     0.0000  Tot=     0.0000
 SCF Done:  E(RB3LYP) =  -40.50000000     A.U. after   10 cycles

 Summary of Natural Population Analysis:
                                                     Natural Population
  Atom  No    Charge        Core      Valence    Rydberg      Total
 ------------------------------------------------------------------
   C     1   -0.30000     1.99990     4.28000    0.02010     6.30000
   H     2    0.30000     0.00000     0.69000    0.01000     0.70000
 ==================================================================

 NATURAL BOND ORBITAL ANALYSIS:
      (Occupancy)   Bond orbital / Coefficients / Hybrids
 ------------------ Lewis ------------------------------------------------------
   1. (1.99999) CR ( 1) C  1            s(100.00%)
   2. (1.98000) BD ( 1) C  1- H  2
 ------------------ non-Lewis --------------------------------------------------
   3. (0.02000) BD*( 1) C  1- H  2

 Normal termination of Gaussian 16 at Sun Sep 13 12:00:00 2026.
"""
    mock_log.write_text(content)

    calc = GaussianCalculation(str(mock_log))
    assert calc.has_nbo() is True
    assert calc.nbo is not None
    assert np.isclose(calc.nbo.npa_charges.set_index('Index')['Charge'][0], -0.30)
    assert len(calc.nbo.get_bonds(atom1=0, atom2=1)) == 2


def test_cli_nbo_parser(capsys):
    nbo_path = Path("files/nbo.out")
    if not nbo_path.exists():
        pytest.skip("files/nbo.out not found")

    cmd = NboParser()
    parser = cmd.build_parser()

    # Test --bonds flag
    args = parser.parse_args([str(nbo_path), "--bonds", "-i", "0", "1"])
    cmd.run(args)
    captured = capsys.readouterr()
    assert "NBO Bonds" in captured.out
    assert "C  1- C  2" in captured.out

    # Test --lp flag with atom 16 (Oxygen 17 has lone pairs)
    args_lp = parser.parse_args([str(nbo_path), "--lp", "-i", "16"])
    cmd.run(args_lp)
    captured_lp = capsys.readouterr()
    assert "NBO Lone Pairs" in captured_lp.out

    # Test --npa flag
    args_npa = parser.parse_args([str(nbo_path), "--npa", "-i", "0"])
    cmd.run(args_npa)
    captured_npa = capsys.readouterr()
    assert "NPA Charges" in captured_npa.out
