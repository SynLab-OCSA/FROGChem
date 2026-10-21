"""
Regression tests for the NMR mathematics/physics review (one test per fixed finding).

Expected values are analytic (AB quartet, first-order Pascal multiplets, Solomon
equations, <r^-6> averaging) or IUPAC constants (Harris et al., Pure Appl. Chem. 73, 1795 (2001)).
"""

import json
import math
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.linalg

from frogchem.nmr import get_nmr_equivalencies, symmetrize_nmr_data
from frogchem.nmr_sim import (
    SpinTransition,
    simulate_1d_lines,
    simulate_1d_spectrum,
    simulate_noesy,
    simulate_quantum_transitions,
    solve_spin_cluster,
)


def _aggregate(transitions, decimals=1):
    """Sum intensities of lines that coincide to `decimals` Hz."""
    agg = defaultdict(float)
    for t in transitions:
        agg[round(t.freq_hz, decimals)] += t.intensity
    return dict(agg)


def _bins(transitions, gap=0.5):
    """Merge lines closer than `gap` Hz; returns [(intensity-weighted centre, total intensity)]."""
    out = []
    for f, i in sorted((t.freq_hz, t.intensity) for t in transitions):
        if out and f - out[-1][2] < gap:
            c, s, _ = out[-1]
            out[-1] = ((c * s + f * i) / (s + i), s + i, f)
        else:
            out.append((f, i, f))
    return [(c, s) for c, s, _ in out]


def _ab_analytic(nu_a, nu_b, J):
    d = nu_a - nu_b
    C = math.sqrt(d * d + J * J)
    c = 0.5 * (nu_a + nu_b)
    s = J / C
    return sorted([(c + J / 2 + C / 2, 1 - s), (c - J / 2 + C / 2, 1 + s),
                   (c + J / 2 - C / 2, 1 + s), (c - J / 2 - C / 2, 1 - s)])


# --- Finding 1: equivalent groups expanded into individual spins -----------------------------
def test_group_expansion_ethyl_matches_explicit_spins():
    J = np.array([[0.0, 7.0], [7.0, 0.0]])
    grouped = simulate_quantum_transitions([1.2, 3.6], J, group_counts=[3, 2], freq_mhz=500.0)

    J5 = np.zeros((5, 5))
    for i in range(3):
        for j in range(3, 5):
            J5[i, j] = J5[j, i] = 7.0
    explicit = simulate_quantum_transitions([1.2] * 3 + [3.6] * 2, J5, freq_mhz=500.0)

    a, b = _aggregate(grouped, 3), _aggregate(explicit, 3)
    assert set(a) == set(b)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-3

    # CH3: 1:2:1 triplet (J = 7 Hz) around 600 Hz; CH2: 1:3:3:1 quartet around 1800 Hz
    # (second order at J/dnu = 0.006 shifts lines by < 0.1 Hz and intensities by < 2 %)
    ch3 = _bins([t for t in grouped if t.freq_hz < 1000])
    ch2 = _bins([t for t in grouped if t.freq_hz > 1000])
    np.testing.assert_allclose([c for c, _ in ch3], [593.0, 600.0, 607.0], atol=0.1)
    np.testing.assert_allclose([s for _, s in ch3], [0.75, 1.5, 0.75], atol=0.02)
    np.testing.assert_allclose([c for c, _ in ch2], [1789.5, 1796.5, 1803.5, 1810.5], atol=0.1)
    np.testing.assert_allclose([s for _, s in ch2], [0.25, 0.75, 0.75, 0.25], atol=0.02)


# --- Finding 2: integrals proportional to group counts ----------------------------------------
def test_group_integrals_follow_counts_when_coupled():
    for J_val in (0.0, 7.0):
        tr = simulate_quantum_transitions([1.0, 3.0], np.array([[0.0, J_val], [J_val, 0.0]]),
                                          group_counts=[3, 1], freq_mhz=500.0)
        s1 = sum(t.intensity for t in tr if t.delta_ppm < 2.0)
        s2 = sum(t.intensity for t in tr if t.delta_ppm > 2.0)
        assert abs(s1 - 3.0) < 1e-3 and abs(s2 - 1.0) < 1e-3


# --- First-order fallback above the exact-cluster cap -----------------------------------------
def test_first_order_fallback_pascal_multiplets():
    J = np.array([[0.0, 7.0], [7.0, 0.0]])
    tr = simulate_quantum_transitions([1.2, 3.6], J, group_counts=[3, 2], freq_mhz=500.0,
                                      max_cluster_size=4)   # 5 spins > 4 -> first order
    ch3 = _aggregate([t for t in tr if t.freq_hz < 1000], decimals=3)
    ch2 = _aggregate([t for t in tr if t.freq_hz > 1000], decimals=3)
    assert ch3 == {593.0: 0.75, 600.0: 1.5, 607.0: 0.75}
    assert ch2 == {1789.5: 0.25, 1796.5: 0.75, 1803.5: 0.75, 1810.5: 0.25}


def test_first_order_fallback_oversize_group_binomial():
    # tBu-like group of 9 equivalent spins (dim 512 > 256) coupled to one spin: not exactly
    # solvable and not splittable -> first order: X is a 10-line binomial C(9,k)/2^9 multiplet.
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    tr = simulate_quantum_transitions([1.0, 5.0], J, group_counts=[9, 1], freq_mhz=500.0)
    x = sorted((t.freq_hz, t.intensity) for t in tr if t.freq_hz > 1000)
    t9 = sorted((t.freq_hz, t.intensity) for t in tr if t.freq_hz < 1000)
    np.testing.assert_allclose([f for f, _ in x], 2500.0 + np.arange(-4.5, 4.6, 1.0), atol=1e-9)
    np.testing.assert_allclose([i for _, i in x], [math.comb(9, k) / 512 for k in range(10)], atol=1e-12)
    np.testing.assert_allclose(t9, [(499.5, 4.5), (500.5, 4.5)], atol=1e-9)


# --- Finding 3: negative frequencies are kept -------------------------------------------------
def test_negative_frequency_transitions_kept():
    nu = np.array([-1000.0, -750.0])
    J = np.array([[0.0, 7.0], [7.0, 0.0]])
    tr = sorted(solve_spin_cluster(nu, J))
    ref = _ab_analytic(-1000.0, -750.0, 7.0)
    assert len(tr) == 4
    np.testing.assert_allclose([t[0] for t in tr], [r[0] for r in ref], atol=1e-8)
    np.testing.assert_allclose([t[1] for t in tr], [r[1] for r in ref], atol=1e-8)

    mixed = simulate_quantum_transitions([-2.0, 1.5], J, freq_mhz=500.0)
    neg = sum(t.intensity for t in mixed if t.delta_ppm < 0)
    pos = sum(t.intensity for t in mixed if t.delta_ppm > 0)
    assert abs(neg - 1.0) < 1e-3 and abs(pos - 1.0) < 1e-3


# --- Finding 4: automatic ppm ranges include negative shifts ----------------------------------
def test_auto_range_includes_negative_shifts():
    tr = [SpinTransition(freq_hz=-1500.0, intensity=1.0, delta_ppm=-3.0)]
    ax, inten = simulate_1d_spectrum(tr, freq_mhz=500.0, fwhm_hz=1.0)
    assert ax[0] < -3.0 < ax[-1]
    assert inten.max() > 0.9 * 1.0 / (math.pi * 0.001)   # Lorentzian peak height 1/(pi*gamma)

    ax, inten = simulate_1d_lines([-120.0, -60.0])
    assert ax[0] < -120.0 and ax[-1] > -60.0

    # Usual 1H window is unchanged when nothing is below the clamp
    ax, _ = simulate_1d_spectrum([SpinTransition(freq_hz=3500.0, intensity=1.0, delta_ppm=7.0),
                                  SpinTransition(freq_hz=500.0, intensity=1.0, delta_ppm=1.0)], freq_mhz=500.0)
    assert ax[0] == 0.0 and ax[-1] == 8.0


# --- Finding 5: NOESY sign follows sigma (Solomon equations) ----------------------------------
def _noesy_ratio(tau_ps):
    f1, _, m = simulate_noesy([1.0, 3.0], np.array([[0.0, 2.5], [2.5, 0.0]]), freq_mhz=500.0,
                              tau_c_ps=tau_ps, mixing_time=0.5, nucleus='1H', n_points=256)
    i1, i3 = np.argmin(abs(f1 - 1.0)), np.argmin(abs(f1 - 3.0))
    return m[i1, i1], m[i1, i3]


def _solomon_ratio(tau_ps, r=2.5e-10, tm=0.5, freq=500e6, leak=0.5):
    mu0, hbar, g = 4e-7 * math.pi, 1.054571817e-34, 267.522e6
    w, t = 2 * math.pi * freq, tau_ps * 1e-12
    Jf = lambda x: t / (1 + (x * t) ** 2)
    K = (mu0 / (4 * math.pi) * hbar * g ** 2) ** 2 / 10
    sig = K * (6 * Jf(2 * w) - Jf(0)) / r ** 6
    rho = K * (Jf(0) + 3 * Jf(w) + 6 * Jf(2 * w)) / r ** 6
    A = scipy.linalg.expm(-np.array([[rho + leak, sig], [sig, rho + leak]]) * tm)
    return A[0, 1] / A[0, 0]


def test_noesy_cross_peak_sign_follows_sigma():
    for tau in (100.0, 1000.0, 10000.0):
        diag, cross = _noesy_ratio(tau)
        expected = _solomon_ratio(tau)
        assert diag < 0.0
        assert np.sign(cross / diag) == np.sign(expected)
    # extreme narrowing: opposite sign to the diagonal; slow tumbling: same sign
    d, c = _noesy_ratio(100.0)
    assert c > 0.0
    d, c = _noesy_ratio(10000.0)
    assert c < 0.0


# --- Finding 6: coupled heteronuclei use their own Larmor frequency in the CLI ----------------
def test_cli_coupled_13c_uses_13c_larmor_frequency():
    import frogchem.cli.calc_nmr as cm
    captured = {}
    orig = cm.simulate_quantum_transitions

    def spy(*a, **k):
        captured['freq'] = k['freq_mhz']
        return orig(*a, **k)

    cm.simulate_quantum_transitions = spy
    try:
        cmd = cm.CalcNmr()
        cmd.args = types.SimpleNamespace(plot=None, save_matrix=False, save_xy=False, out_dir=None)
        cmd.output = lambda *a, **k: None
        args = types.SimpleNamespace(
            nuclei=['13C'], custom_spins={}, no_equiv=True, ref=185.0, coupled=True, freq=500.0,
            j_cutoff=0.3, fwhm=1.0, sim_1d=False, sim_1d_c13=False, cosy=False, noesy=False,
            hsqc=False, hmbc=False, save_xy=False, save_matrix=False, r_cutoff=5.0, all_spectra=False)
        shifts = {0: {'element': 'C', 'iso': 165.0, 'aniso': 0.0}, 1: {'element': 'C', 'iso': 164.7, 'aniso': 0.0}}
        cmd._run_simulation(shifts, {(0, 1): 35.0, (1, 0): 35.0}, np.array([[0, 0, 0], [1.5, 0, 0.0]]),
                            ['C', 'C'], args)
    finally:
        cm.simulate_quantum_transitions = orig
    assert abs(captured['freq'] - 500.0 * 67.2828e6 / 267.522e6) < 1e-6


# --- Finding 7: NOE-effective group distances <r^-6>^(-1/6) -----------------------------------
def test_symmetrize_group_distance_is_r6_average():
    geom = np.array([[0.0, 0.0, 0.0], [2.3, 0.0, 0.0], [3.0, 0.0, 0.0], [3.6, 0.0, 0.0]])
    shifts = {i: {'element': 'H', 'iso': 30.0, 'aniso': 0.0} for i in range(4)}
    equiv = {'H': [[0], [1, 2, 3]]}
    _, _, gdist, _ = symmetrize_nmr_data(shifts, {}, geom, equiv, element='H')
    r = np.array([2.3, 3.0, 3.6])
    assert abs(gdist[0, 1] - np.mean(r ** -6) ** (-1 / 6)) < 1e-12
    assert abs(gdist[0, 1] - 2.65391036116223) < 1e-9


# --- Finding 9: gyromagnetic ratios (IUPAC 2001) ----------------------------------------------
def test_nuclei_json_gamma_values_iupac():
    import frogchem
    data = json.loads((Path(frogchem.__file__).parent / 'data' / 'nuclei.json').read_text())
    iupac = {'1H': 26.7522128e7, '13C': 6.728284e7, '19F': 25.18148e7, '31P': 10.8394e7,
             '77Se': 5.1253857e7, '113Cd': -5.9609155e7, '119Sn': -10.0317e7,
             '127I': 5.389573e7, '195Pt': 5.8385e7}
    for nuc, g in iupac.items():
        assert abs(data[nuc]['gamma'] - g) / abs(g) < 1e-4, nuc


# --- Heavy-atom symmetry broken by H must not merge inequivalent atoms ------------------------
def test_equivalence_rejects_heavy_atom_symmetry_broken_by_h():
    atoms = ['C', 'C', 'H', 'H', 'H', 'H']
    coords = [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [-0.5, 0.9, 0.0],
              [1.9, 0.9, 0.0], [1.9, -0.45, 0.78], [1.9, -0.45, -0.78]]
    equiv = get_nmr_equivalencies((atoms, coords))
    groups = [set(g) for g in equiv['H']]
    assert {3, 4, 5} in groups and {2} in groups
    assert [set(g) for g in equiv['C']] == [{0}, {1}]
