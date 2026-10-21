"""Gaussian NBO 3.1 (`pop=(full,nbo)`) read by NBOAnalysis, checked against the ORCA (NBO 7) twin.

NBO 3.1 prints the section titles in mixed case, the E(2) pairs as 'donor / acceptor' and Rydbergs as
RY*; NBO 7 numbers the NBOs differently (Lewis first), so NBOs are matched by label: type, atoms and
the ordinal in parentheses. The file tests skip when tests/data/gaussian_ref is missing.
"""

import os

import pytest

from frogchem.orca_parser.nbo import NBOAnalysis

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(REPO, 'tests', 'data', 'gaussian_ref')
GAUSSIAN = os.path.join(REF, 'gaussian', 'g06_h20_nbo.log')
ORCA = os.path.join(REF, 'orca', 'g06_h2o_nbo.out')

SAMPLE_NBO31 = """
 Summary of Natural Population Analysis:

                                     Natural Population
             Natural  -----------------------------------------------
    Atom No    Charge         Core      Valence    Rydberg      Total
 -----------------------------------------------------------------------
      O    1   -0.91418      1.99983     8.89768    0.01667    10.91418
      H    2    0.45709      0.00000     0.54045    0.00246     0.54291
 =======================================================================

 NATURAL BOND ORBITAL ANALYSIS:

       (Occupancy)   Bond orbital/ Coefficients/ Hybrids
 ---------------------------------------------------------------------------------
     1. (1.99811) BD ( 1) O   1 - H   2
                ( 72.46%)   0.8512* O   1 s( 24.31%)p 3.11( 75.58%)d 0.00(  0.11%)
                                            0.0001 -0.4922 -0.0293  0.7071  0.0374
                ( 27.54%)   0.5248* H   2 s( 99.73%)p 0.00(  0.27%)
                                           -0.9987  0.0016 -0.0449  0.0256  0.0000
     2. (1.99675) LP ( 1) O   1           s(  0.00%)p 1.00( 99.94%)d 0.00(  0.06%)
                                            0.0000  0.0000  0.0000  0.0000  0.0000
     3. (0.00162) RY*( 1) H   2           s(  0.00%)p 1.00(100.00%)
                                            0.0000  0.0000  0.0000  0.0000  1.0000
     4. (0.00021) BD*( 1) O   1 - H   2
                ( 27.54%)   0.5248* O   1 s( 24.31%)p 3.11( 75.58%)d 0.00(  0.11%)
                                           -0.0001  0.4922  0.0293 -0.7071 -0.0374
                ( 72.46%)  -0.8512* H   2 s( 99.73%)p 0.00(  0.27%)
                                            0.9987 -0.0016  0.0449 -0.0256 -0.0000


 NHO Directionality and "Bond Bending" (deviations from line of nuclear centers)

                NBO               Theta   Phi    Theta   Phi    Dev    Theta   Phi    Dev
 ========================================================================================
     1. BD (   1) O   1 - H   2    90.0  143.9    90.0  145.7   1.8      --     --    --
     2. LP (   1) O   1             --     --      0.0    0.0   --       --     --    --


 Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis

     Threshold for printing:   0.50 kcal/mol
                                                                              E(2)  E(j)-E(i) F(i,j)
         Donor NBO (i)                     Acceptor NBO (j)                 kcal/mol   a.u.    a.u.
 ===================================================================================================

 within unit  1
   1. BD (   1) O   1 - H   2        /  4. BD*(   1) O   1 - H   2            0.55    1.18    0.022
   2. LP (   1) O   1                /  3. RY*(   1) H   2                    2.18    1.92    0.058


 Natural Bond Orbitals (Summary):

                                                            Principal Delocalizations
           NBO                        Occupancy    Energy   (geminal,vicinal,remote)
 ====================================================================================
 Molecular unit  1  (OH)
     1. BD (   1) O   1 - H   2          1.99811    -0.72332  4(g)
     2. LP (   1) O   1                  1.99675    -0.28832  3(v)
     3. RY*(   1) H   2                  0.00162     1.62927
     4. BD*(   1) O   1 - H   2          0.00021     0.45267
       -------------------------------
              Total Lewis    3.99486  ( 99.8853%)
"""


def test_nbo31_sample():
    nbo = NBOAnalysis(SAMPLE_NBO31)
    df = nbo.nbo_details
    # The NHO, E(2) and summary rows must not be read as further NBOs
    assert df['Index'].tolist() == [1, 2, 3, 4]
    assert df['Type'].tolist() == ['BD', 'LP', 'RY*', 'BD*']
    assert df['Energy'].tolist() == pytest.approx([-0.72332, -0.28832, 1.62927, 0.45267])
    assert df['Occupancy'].tolist() == pytest.approx([1.99811, 1.99675, 0.00162, 0.00021])
    assert df.iloc[0]['Atom_Indices'] == [0, 1]

    e2 = nbo.perturbation_theory
    assert e2 is not None and len(e2) == 2
    row = e2.iloc[1]
    assert (row['Donor_Index'], row['Donor_Type'], row['Donor_Atoms']) == (2, 'LP', [0])
    assert (row['Acceptor_Index'], row['Acceptor_Type'], row['Acceptor_Atoms']) == (3, 'RY*', [1])
    assert (row['E(2)'], row['E_diff'], row['F_ij']) == pytest.approx((2.18, 1.92, 0.058))
    assert e2.iloc[0]['Acceptor_Type'] == 'BD*'


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f'{os.path.relpath(path, REPO)} not found (Gaussian reference set not available)')
    with open(path, encoding='utf-8', errors='replace') as fh:
        return NBOAnalysis(fh.read())


def _keys(df):
    """NBO index -> (type, atoms, ordinal); NBO 3.1 'RY*' is NBO 7 'RY'."""
    df = df.sort_values('Index')
    types = df['Type'].replace({'RY*': 'RY'})
    atoms = df['Atom_Indices'].apply(tuple)
    ordinal = df.groupby([types, atoms]).cumcount() + 1
    return {int(i): (t, a, int(o)) for i, t, a, o in zip(df['Index'], types, atoms, ordinal)}


@pytest.fixture(scope='module')
def twins():
    return _load(GAUSSIAN), _load(ORCA)


def test_gaussian_nbo_list(twins):
    g, o = twins
    assert g.nbo_details['Index'].tolist() == list(range(1, 25))
    assert sorted(_keys(g.nbo_details).values()) == sorted(_keys(o.nbo_details).values())


def test_gaussian_nbo_energies_match_orca(twins):
    g, o = twins
    g_by_key = {k: i for i, k in _keys(g.nbo_details).items()}
    o_by_key = {k: i for i, k in _keys(o.nbo_details).items()}
    g_df = g.nbo_details.set_index('Index')
    o_df = o.nbo_details.set_index('Index')
    assert (g_df['Energy'] != 0.0).all()
    for key in [k for k in g_by_key if k[0] in ('CR', 'BD', 'LP', 'BD*')]:
        eg, eo = g_df.loc[g_by_key[key], 'Energy'], o_df.loc[o_by_key[key], 'Energy']
        # Same SCF (ΔE 7e-7 Eh); NBO 3.1 and NBO 7 hybrids differ: 0.16 Eh on CR, < 0.08 Eh on valence
        assert eg == pytest.approx(eo, abs=0.2 if key[0] == 'CR' else 0.1), key
        assert g_df.loc[g_by_key[key], 'Occupancy'] == pytest.approx(o_df.loc[o_by_key[key], 'Occupancy'], abs=3e-3)
    # Pure 2p lone pair: no hybridisation freedom, the two programs agree closely
    assert g_df.loc[g_by_key[('LP', (0,), 1)], 'Energy'] == pytest.approx(-0.2887, abs=1e-3)


def test_gaussian_e2_matches_orca(twins):
    g, o = twins

    def table(nbo):
        keys = _keys(nbo.nbo_details)
        return {(keys[int(r['Donor_Index'])], keys[int(r['Acceptor_Index'])]): r['E(2)']
                for _, r in nbo.perturbation_theory.iterrows()}

    g_e2, o_e2 = table(g), table(o)
    assert len(g_e2) == 8 and len(o_e2) == 6
    # Every interaction above the 0.5 kcal/mol print threshold in NBO 7 is in NBO 3.1 too (which also
    # prints n(O) -> RY(3) H at 0.67); Rydberg-dependent values differ by up to 0.5 kcal/mol
    for pair, e2 in o_e2.items():
        assert g_e2[pair] == pytest.approx(e2, abs=0.5), pair
    assert max(g_e2.values()) == pytest.approx(2.18)
