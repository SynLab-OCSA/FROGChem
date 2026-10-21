"""NBO orbital energies come from the 'NATURAL BOND ORBITALS (Summary)' table."""

from pathlib import Path

import numpy as np
import pytest

from frogchem.orca_parser.nbo import NBOAnalysis


SAMPLE = """
 Summary of Natural Population Analysis:
  Atom  No    Charge        Core      Valence    Rydberg      Total
 ------------------------------------------------------------------
   C     1   -0.45000     1.99990     4.43000    0.02010     6.45000
   H     2    0.15000     0.00000     0.84000    0.01000     0.85000
 ==================================================================

 NATURAL BOND ORBITAL ANALYSIS:
     (Occupancy)   Bond orbital / Coefficients / Hybrids
 ------------------ Lewis ------------------------------------------------------
   1. (1.99999) CR ( 1) C  1            s(100.00%)
                                         1.0000  0.0000  0.0000
   2. (1.99400) BD ( 1) C  1- H  2
               ( 60.00%)   0.7746* C  1 s( 25.00%)p 3.00( 75.00%)
               ( 40.00%)   0.6325* H  2 s(100.00%)
 ------------------ non-Lewis --------------------------------------------------
   3. (0.01200) BD*( 1) C  1- H  2
               ( 40.00%)   0.6325* C  1 s( 25.00%)p 3.00( 75.00%)
               ( 60.00%)  -0.7746* H  2 s(100.00%)

 SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS
    1. CR ( 1) C  1          /   3. BD*( 1) C  1- H  2        0.60   10.50    0.071

 NATURAL BOND ORBITALS (Summary):

                                                     Principal Delocalizations
           NBO                 Occupancy    Energy   (geminal,vicinal,remote)
 ===============================================================================
 Molecular unit  1  (CH)
 ------ Lewis --------------------------------------
    1. CR ( 1) C  1             1.99999   -10.17087
    2. BD ( 1) C  1- H  2       1.99400    -0.61234  3(g)
 ------ non-Lewis ----------------------------------
    3. BD*( 1) C  1- H  2       0.01200     0.45678
          -------------------------------
                 Total Lewis    3.99399  ( 99.7000%)
          -------------------------------
               Total unit  1    4.00599  (100.0000%)
"""


def test_energies_from_summary():
    nbo = NBOAnalysis(SAMPLE)
    df = nbo.nbo_details.set_index('Index')
    np.testing.assert_allclose(df.loc[[1, 2, 3], 'Energy'], [-10.17087, -0.61234, 0.45678])
    np.testing.assert_allclose(df.loc[[1, 2, 3], 'Occupancy'], [1.99999, 1.99400, 0.01200])


def test_real_file_energies_not_zero():
    path = Path("files/nbo.out")
    if not path.exists():
        pytest.skip("files/nbo.out not found")
    nbo = NBOAnalysis(path.read_text(encoding="utf-8", errors="replace"))
    df = nbo.nbo_details
    assert (df['Energy'] != 0.0).all()
    # Last cycle of files/nbo.out: first core orbital, CR ( 1) C 1
    assert df.iloc[0]['Energy'] < -9.0


# NBO 7 layout: coefficient rows starting with "0.", a hybrid wrapped onto an "f" row,
# no space in "*Br 33" or "p99.99(" -- none of these may cut the composition short;
# one-centre NBOs carry their hybrid on the header row
COMPOSITION_SAMPLE = """
 NATURAL BOND ORBITAL ANALYSIS:
 ------------------ Lewis ------------------------------------------------------
   1. (1.98278) BD ( 1) C  1-Br  2
               ( 46.87%)   0.6846* C  1 s( 21.47%)p 3.64( 78.13%)d 0.02(  0.32%)
                                                  f 0.00(  0.08%)
                                         0.0000  0.4627 -0.0190  0.0159  0.0000
                                        -0.0084 -0.3297 -0.0077 -0.0122 -0.0177
               ( 53.13%)   0.7289*Br  2 s( 13.51%)p 6.39( 86.29%)d 0.01(  0.19%)
                                                  f 0.00(  0.02%)
                                         0.0001 -0.4922 -0.0293  0.7071  0.0374
   2. (1.65038) BD ( 2) C  1- C  3
               ( 51.74%)   0.7193* C  1 s(  0.02%)p99.99( 99.92%)d 1.37(  0.02%)
                                                  f 2.34(  0.04%)
                                         0.0000  0.0002  0.0000  0.0000  0.0000
               ( 48.26%)   0.6947* C  3 s(  0.01%)p99.99( 99.92%)d 1.00(  0.03%)
                                                  f 1.00(  0.04%)
                                         0.0000  0.0001  0.0000  0.0000  0.0000
 ------------------ non-Lewis --------------------------------------------------
   3. (0.00021) BD*( 1) C  1-Br  2
               ( 53.13%)   0.7289* C  1 s( 21.47%)p 3.64( 78.13%)d 0.02(  0.32%)
                                         0.0000  0.4627 -0.0190  0.0159  0.0000
               ( 46.87%)  -0.6846*Br  2 s( 13.51%)p 6.39( 86.29%)d 0.01(  0.19%)
                                         0.0001 -0.4922 -0.0293  0.7071  0.0374
   4. (1.99996) CR ( 5) Br  2           s(  0.00%)p 1.00(100.00%)
                                         0.0000  0.0000  0.0000  0.0000  1.0000
   5. (1.99722) LP ( 1) Br  2           s( 86.56%)p 0.16( 13.43%)d 0.00(  0.01%)
                                                  f 0.00(  0.00%)
                                         0.0000  0.0000  0.9300  0.0012 -0.0001
"""


def test_composition_lists_atoms_and_full_hybrid():
    df = NBOAnalysis(COMPOSITION_SAMPLE).nbo_details.set_index('Index')
    assert df.loc[1, 'Composition'] == [
        'C 1 (46.87%) s(21.47%) | p(78.13%) | d(0.32%) | f(0.08%)',
        'Br 2 (53.13%) s(13.51%) | p(86.29%) | d(0.19%) | f(0.02%)',
    ]
    assert df.loc[2, 'Composition'] == [
        'C 1 (51.74%) s(0.02%) | p(99.92%) | d(0.02%) | f(0.04%)',
        'C 3 (48.26%) s(0.01%) | p(99.92%) | d(0.03%) | f(0.04%)',
    ]
    assert [c.split(' (')[0] for c in df.loc[3, 'Composition']] == ['C 1', 'Br 2']
    assert df.loc[4, 'Composition'] == ['Br 2 (100.00%) s(0.00%) | p(100.00%)']
    assert df.loc[5, 'Composition'] == ['Br 2 (100.00%) s(86.56%) | p(13.43%) | d(0.01%) | f(0.00%)']
