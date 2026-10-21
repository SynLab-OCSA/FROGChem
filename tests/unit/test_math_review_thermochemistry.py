"""Regression tests for the thermochemistry math review (reference values computed independently)."""
import json
import re
import warnings

import numpy as np
from scipy import constants as C

from frogchem import thermo as T
from frogchem.ensemble import load_checkpoint_populations
from frogchem.orca_parser.regex_ORCA import REGEXS

EH_J_MOL = C.physical_constants['Hartree energy'][0] * C.N_A   # Eh -> J/mol
EH_KCAL = EH_J_MOL / 4184.0

# H2O: rotational constants [cm-1], harmonic frequencies [cm-1], isotopic mass [g/mol]
H2O_B = np.array([27.8806, 14.5216, 9.2778])
H2O_FREQ = np.array([1594.6, 3657.1, 3755.9])
H2O_MW = 18.01056


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# --- finding 1: ORCA rotational constants >= 10 cm-1 / missing B -------------------------

def test_orca_rot_const_regex_two_digit_constants():
    line = "Rotational constants in cm-1:    27.264292    14.578823     9.499082"
    for reg in REGEXS.values():
        assert re.findall(reg['rot_const'], line) == [('27.264292', '14.578823', '9.499082')]


def test_free_gibbs_energy_warns_without_rotational_constants():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        T.free_gibbs_energy(-76.4, 298.15, H2O_FREQ, H2O_MW, None, 1, symno=2)
    assert any(issubclass(x.category, RuntimeWarning) and 'rotational' in str(x.message) for x in w)


def test_water_rrho_entropy_reference():
    # independent RRHO implementation: S = 188.713 J/mol/K at 298.15 K, 1 bar (NIST 188.835)
    _, _, _, S = T.free_gibbs_energy(0.0, 298.15, H2O_FREQ, H2O_MW, H2O_B, 1, P=100.0, symno=2)
    assert abs(S * EH_J_MOL - 188.713) < 2e-3


# --- finding 2: E and G must not be mixed in one Boltzmann pool ---------------------------

def test_checkpoint_does_not_mix_E_and_G(tmp_path):
    p = _write(tmp_path, 'ck.json', {
        "1": {"energies": {"1": {"E": -500.0, "G": -499.700}}},
        "2": {"energies": {"1": {"E": -500.001}}},          # no G stored
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pops = load_checkpoint_populations(p, energy_term='G')
    assert abs(pops[1] - 100.0) < 1e-9          # was {1: 3e-137, 2: 100.0}
    assert 2 not in pops


def test_checkpoint_all_E_fallback_still_consistent(tmp_path):
    p = _write(tmp_path, 'ck.json', {
        "1": {"energies": {"1": {"E": -500.0}}},
        "2": {"energies": {"1": {"E": -500.0}}},
    })
    pops = load_checkpoint_populations(p, energy_term='G')
    assert abs(pops[1] - 50.0) < 1e-9 and abs(pops[2] - 50.0) < 1e-9


# --- finding 3: no default B for only some conformers ------------------------------------

def test_checkpoint_missing_B_vec_does_not_bias(tmp_path):
    fr = [30., 80., 150., 400., 800., 1200., 1600., 3000.]
    p = _write(tmp_path, 'ck.json', {
        "1": {"energies": {"1": {"E": -500.0, "G": -499.8, "Freq": fr, "B_vec": [0.03, 0.02, 0.015]}}},
        "2": {"energies": {"1": {"E": -500.0, "G": -499.8, "Freq": fr}}},
    })
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pops = load_checkpoint_populations(p, energy_term='G')
    assert abs(pops[1] - 50.0) < 1e-9 and abs(pops[2] - 50.0) < 1e-9   # was 99.71 / 0.29


def test_checkpoint_recompute_with_B_vec(tmp_path):
    fr = [30., 80., 150., 400., 800., 1200., 1600., 3000.]
    B = [0.03, 0.02, 0.015]
    p = _write(tmp_path, 'ck.json', {
        "1": {"atoms": ["C", "Cl"], "energies": {"1": {"E": -500.0, "Freq": fr, "B_vec": B}}},
        "2": {"atoms": ["C", "Cl"], "energies": {"1": {"E": -500.001, "Freq": fr, "B_vec": B}}},
    })
    pops = load_checkpoint_populations(p, energy_term='G')
    ref = T.calculate_boltzmann(np.array([-500.0, -500.001]), 298.15) * 100   # same thermal corr.
    assert abs(pops[1] - ref[0]) < 1e-6 and abs(pops[2] - ref[1]) < 1e-6


# --- finding 4: CODATA constants in ensemble Boltzmann -----------------------------------

def test_checkpoint_boltzmann_constants(tmp_path):
    p = _write(tmp_path, 'ck.json', {
        "1": {"energies": {"1": {"E": -500.0, "G": -499.700}}},
        "2": {"energies": {"1": {"E": -500.001, "G": -499.699}}},
    })
    pops = load_checkpoint_populations(p, energy_term='G')
    x = np.exp(-0.001 * EH_J_MOL / (C.R * 298.15))
    assert abs(pops[1] - 100 / (1 + x)) < 1e-9          # 74.2521, was 74.2618 (4.186 J/cal)


# --- finding 5: linear molecules detected from B -----------------------------------------

def test_linear_rotor_autodetected():
    B = np.array([0.0, 0.3902, 0.3902])
    s_flag = T.calc_rotational_entropy(B, 298.15, 2, True) * EH_J_MOL
    s_auto = T.calc_rotational_entropy(B, 298.15, 2, False) * EH_J_MOL
    assert abs(s_flag - 54.724) < 1e-3          # independent: k(ln(T/(sigma*theta)) + 1)
    assert abs(s_auto - s_flag) < 1e-12         # was 87.326 J/mol/K


def test_linear_free_gibbs_energy_uses_linear_U_rot():
    fr = np.array([667.4, 667.4, 1333.0, 2349.0])
    B = np.array([0.0, 0.3902, 0.3902])
    g_a, _, h_a, _ = T.free_gibbs_energy(0.0, 298.15, fr, 43.98983, B, 1, linear=True, symno=2)
    g_b, _, h_b, _ = T.free_gibbs_energy(0.0, 298.15, fr, 43.98983, B, 1, linear=False, symno=2)
    assert abs(g_a - g_b) < 1e-14 and abs(h_a - h_b) < 1e-14


# --- finding 7: calculate_boltzmann uses CODATA Eh ---------------------------------------

def test_calculate_boltzmann_codata():
    E = np.array([0.0, 0.001, 0.002])
    ref = np.exp(-E * EH_J_MOL / (C.R * 298.15))
    ref /= ref.sum()
    assert np.max(np.abs(T.calculate_boltzmann(E, 298.15) / ref - 1)) < 1e-12   # was 1.4e-6


def test_calculate_boltzmann_empty_raises():
    try:
        T.calculate_boltzmann(np.array([]), 298.15)
    except ValueError as e:
        assert 'empty' in str(e)
    else:
        raise AssertionError("no ValueError")


# --- notes fixed: numerical stability, list input, docstring -----------------------------

def test_harmonic_entropy_stable():
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # no overflow RuntimeWarning
        s = T.calc_S_V_grimme(np.array([3000.0]), 4.0)
    assert s[0] >= 0.0
    x = 1e-9
    s_small = T.calc_S_V_grimme(np.array([x * C.k * 298.15 / (C.h * C.c * 100)]), 298.15)[0] / C.k
    assert abs(s_small - (x / np.expm1(x) - np.log(-np.expm1(-x)))) < 1e-12


def test_free_gibbs_energy_accepts_list():
    a = T.free_gibbs_energy(0.0, 298.15, list(H2O_FREQ), H2O_MW, H2O_B, 1, symno=2)
    b = T.free_gibbs_energy(0.0, 298.15, H2O_FREQ, H2O_MW, H2O_B, 1, symno=2)
    assert np.allclose(a, b, rtol=0, atol=0)


def test_calc_zpe_has_docstring():
    assert T.calc_zpe.__doc__ and 'Zero Point Energy' in T.calc_zpe.__doc__
    assert T.calc_zpe() == 0.0
