"""
Unit tests for quantum NMR spin simulation, equivalencies, and 2D COSY/NOESY.
"""

import math
import numpy as np
import pytest

from frogchem.nmr import (
    convert_shielding_to_shift,
    format_spin,
    get_nmr_equivalencies,
    get_nucleus_spin,
    get_periodictable_spin,
    get_spin_multiplicity,
    is_nucleus_decoupled,
    is_quadrupolar,
    load_nucleus_database,
    parse_nuclei_set,
    is_quadrupolar,
    load_nucleus_database,
    parse_spin,
    parse_spins_arg,
    set_nucleus_spin,
    symmetrize_nmr_data,
    NUCLEUS_DATA,
)
from frogchem.nmr_sim import (
    solve_spin_cluster,
    simulate_quantum_transitions,
    simulate_1d_spectrum,
    simulate_fid,
    simulate_1d_from_fid,
    simulate_cosy,
    simulate_noesy,
    simulate_1d_lines,
    simulate_hsqc,
    simulate_hmbc,
    plot_2d_spectrum,
    partition_spin_network,
)


def test_convert_shielding_to_shift():
    # IUPAC definition (Harris et al., Pure Appl. Chem. 2008, 80, 59):
    # delta = (sigma_ref - sigma) / (1 - sigma_ref), with sigma in ppm (x 1e-6).
    # Standard 1H TMS default is 31.8 ppm
    delta = convert_shielding_to_shift(24.0, '1H')
    assert np.isclose(delta, (31.8 - 24.0) / (1 - 31.8e-6))

    # Custom TMS reference
    delta_custom = convert_shielding_to_shift(24.0, '1H', ref_sigma=30.0)
    assert np.isclose(delta_custom, 6.0 / (1 - 30.0e-6))
    # The correction is well below the precision of DFT shifts for 1H
    assert abs(delta_custom - 6.0) < 1e-3

    # Empirical linear scaling: delta = (sigma - intercept) / slope
    delta_scaled = convert_shielding_to_shift(24.0, slope=-1.0, intercept=31.5)
    assert np.isclose(delta_scaled, (24.0 - 31.5) / -1.0, rtol=0.0002)


def test_nmr_equivalencies_methane():
    # Tetrahedral methane
    atoms = ['C', 'H', 'H', 'H', 'H']
    coords = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.027, 0.0, -0.363],
        [-0.513, 0.890, -0.363],
        [-0.513, -0.890, -0.363],
    ]
    equiv = get_nmr_equivalencies((atoms, coords))
    assert 'H' in equiv
    # All 4 hydrogens must be grouped together into 1 equivalent group of 4
    assert len(equiv['H']) == 1
    assert set(equiv['H'][0]) == {1, 2, 3, 4}


def test_nmr_equivalencies_methyl_rotor():
    # Molecule with a methyl group: C1-C2 where C2 is bonded to 3 H
    atoms = ['C', 'C', 'H', 'H', 'H', 'H']
    coords = [
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [-0.5, 0.9, 0.0],      # H on C1
        [1.9, 0.9, 0.0],       # H on C2
        [1.9, -0.45, 0.78],    # H on C2
        [1.9, -0.45, -0.78],   # H on C2
    ]
    equiv = get_nmr_equivalencies((atoms, coords))
    assert 'H' in equiv
    # The 3 methyl protons (indices 3, 4, 5) must be grouped together
    methyl_found = False
    for group in equiv['H']:
        if set(group) == {3, 4, 5}:
            methyl_found = True
    assert methyl_found, f"Methyl rotor grouping failed, got: {equiv['H']}"


def test_spin_hamiltonian_ab_system_analytical():
    """Verify that exact quantum diagonalization matches analytical AB system formulas:
    Positions: nu0 +/- C/2 +/- J/2
    Intensities: 1 +/- J/C (roof effect)
    """
    nu_A = 200.0
    nu_B = 220.0
    delta_nu = nu_B - nu_A  # 20 Hz
    J_val = 10.0            # 10 Hz
    nu_0 = (nu_A + nu_B) / 2.0  # 210 Hz

    C = math.sqrt(delta_nu**2 + J_val**2)  # sqrt(500) = 22.36068 Hz

    expected_freqs = sorted([
        nu_0 - C / 2.0 - J_val / 2.0,
        nu_0 - C / 2.0 + J_val / 2.0,
        nu_0 + C / 2.0 - J_val / 2.0,
        nu_0 + C / 2.0 + J_val / 2.0,
    ])
    expected_intensities = [
        1.0 - J_val / C,
        1.0 + J_val / C,
        1.0 + J_val / C,
        1.0 - J_val / C,
    ]

    nu = np.array([nu_A, nu_B])
    J = np.array([[0.0, J_val], [J_val, 0.0]])

    transitions = solve_spin_cluster(nu, J)
    transitions.sort(key=lambda x: x[0])

    assert len(transitions) == 4
    calc_freqs = [t[0] for t in transitions]
    calc_intensities = [t[1] for t in transitions]

    np.testing.assert_allclose(calc_freqs, expected_freqs, atol=1e-4)
    np.testing.assert_allclose(calc_intensities, expected_intensities, atol=1e-4)


def test_spin_hamiltonian_identical_spins():
    """For two magnetically equivalent spins (A2 system), mutual J coupling
    produces zero splitting and the spectrum collapses to a single singlet.
    """
    nu = np.array([150.0, 150.0])
    J = np.array([[0.0, 12.0], [12.0, 0.0]])

    transitions = solve_spin_cluster(nu, J)
    # Both allowed transitions in the triplet manifold (-1 -> 0 and 0 -> +1)
    # occur at the exact same frequency 150.0 Hz (zero splitting between equivalent spins).
    assert len(transitions) == 2
    for freq, prob in transitions:
        assert np.isclose(freq, 150.0)
        assert prob > 0.0


def test_1d_simulation_spectrum():
    nu = np.array([100.0, 200.0])  # in Hz, corresponding to 0.2 and 0.4 ppm at 500 MHz
    J = np.array([[0.0, 0.0], [0.0, 0.0]])
    transitions = simulate_quantum_transitions(
        shifts_ppm=[0.2, 0.4],
        j_matrix=J,
        freq_mhz=500.0,
    )
    ppm_axis, intensity = simulate_1d_spectrum(
        transitions,
        ppm_range=(0.0, 0.6),
        n_points=1000,
        freq_mhz=500.0,
        fwhm_hz=2.0,
    )
    assert len(ppm_axis) == 1000
    assert len(intensity) == 1000
    assert np.all(intensity >= 0.0)

    # Peaks should be centered near 0.2 and 0.4 ppm
    peak1_idx = np.argmin(np.abs(ppm_axis - 0.2))
    peak2_idx = np.argmin(np.abs(ppm_axis - 0.4))
    assert intensity[peak1_idx] > 10.0
    assert intensity[peak2_idx] > 10.0


def test_cosy_simulation():
    shifts = [2.0, 3.0]
    J = np.array([[0.0, 7.5], [7.5, 0.0]])
    f1, f2, cosy_mat = simulate_cosy(
        group_shifts_ppm=shifts,
        group_j=J,
        ppm_range=(1.0, 4.0),
        n_points=64,
    )
    assert cosy_mat.shape == (64, 64)
    # Check that diagonal has strong intensity
    idx_2 = np.argmin(np.abs(f1 - 2.0))
    idx_3 = np.argmin(np.abs(f1 - 3.0))
    assert cosy_mat[idx_2, idx_2] > 0.0
    assert cosy_mat[idx_3, idx_3] > 0.0

    # Cross-peaks at (2.0, 3.0) and (3.0, 2.0)
    assert cosy_mat[idx_2, idx_3] > 0.0
    assert cosy_mat[idx_3, idx_2] > 0.0


def test_noesy_simulation():
    shifts = [2.0, 5.0]
    # Close distance: 2.2 Angstrom -> strong NOE
    dist = np.array([[0.0, 2.2], [2.2, 0.0]])
    f1, f2, noesy_mat = simulate_noesy(
        group_shifts_ppm=shifts,
        group_dist=dist,
        ppm_range=(1.0, 6.0),
        n_points=64,
        r_cutoff=5.0,
    )
    assert noesy_mat.shape == (64, 64)
    idx_2 = np.argmin(np.abs(f1 - 2.0))
    idx_5 = np.argmin(np.abs(f1 - 5.0))
    # Diagonal peaks are negative
    assert noesy_mat[idx_2, idx_2] < 0.0
    assert noesy_mat[idx_5, idx_5] < 0.0
    # Cross-peaks are positive
    assert noesy_mat[idx_2, idx_5] > 0.0
    assert abs(noesy_mat[idx_2, idx_5]) > 0.1
    assert noesy_mat[idx_5, idx_2] > 0.0
    np.testing.assert_allclose(noesy_mat[idx_2, idx_5], noesy_mat[idx_5, idx_2])


def test_fid_simulation_and_fft(tmp_path):
    shifts = [2.0, 4.0]
    J = np.zeros((2, 2))
    trans = simulate_quantum_transitions(shifts, J, freq_mhz=500.0)
    assert len(trans) == 2

    # Simulate time-domain FID
    t, fid = simulate_fid(trans, freq_mhz=500.0, fwhm_hz=2.0, n_points=4096)
    assert len(t) == 4096
    assert len(fid) == 4096
    assert fid.dtype == np.complex128
    assert np.abs(fid[0]) > 0.0
    # Decays over time
    assert np.abs(fid[-1]) < np.abs(fid[0])

    # Compute 1D via FFT
    ppm, intensity = simulate_1d_from_fid(trans, freq_mhz=500.0, fwhm_hz=2.0, n_points=4096)
    assert len(ppm) == 8192
    assert np.all(intensity >= 0.0)
    # Check peaks near 2.0 and 4.0 ppm
    idx_2 = np.argmin(np.abs(ppm - 2.0))
    idx_4 = np.argmin(np.abs(ppm - 4.0))
    assert intensity[idx_2] > 0.05
    assert intensity[idx_4] > 0.05


def test_plot_2d_with_trace_1d(tmp_path):
    shifts = [2.0, 4.0]
    J = np.array([[0.0, 5.0], [5.0, 0.0]])
    trans = simulate_quantum_transitions(shifts, J, freq_mhz=500.0)
    ppm_1d, int_1d = simulate_1d_spectrum(trans, freq_mhz=500.0)
    f1, f2, cosy_mat = simulate_cosy(shifts, J, ppm_range=(1.0, 5.0), n_points=32)

    out_cosy = str(tmp_path / "test_cosy_trace.png")
    plot_2d_spectrum(
        f1, f2, cosy_mat,
        title="COSY with 1D Trace",
        save_path=out_cosy,
        trace_1d=(ppm_1d, int_1d),
    )
    assert (tmp_path / "test_cosy_trace.png").exists()


def test_simulate_1d_lines():
    c_shifts = [25.0, 55.0, 130.0, 175.0]
    ppm, intensity = simulate_1d_lines(c_shifts, ppm_range=(0.0, 200.0), n_points=1000, fwhm_ppm=1.0)
    assert len(ppm) == 1000
    assert len(intensity) == 1000
    assert np.all(intensity >= 0.0)

    for cs in c_shifts:
        idx = np.argmin(np.abs(ppm - cs))
        assert intensity[idx] > 0.1


def test_simulate_hsqc():
    # 1H shifts: H0 (CH) at 3.0 ppm, H1 (CH2) at 4.5 ppm
    h_shifts = [3.0, 4.5]
    # 13C shifts: C0 (CH) at 50.0 ppm, C1 (CH2) at 70.0 ppm
    c_shifts = [50.0, 70.0]
    correlations = [
        (0, 0, 'CH'),   # H0-C0 -> Positive
        (1, 1, 'CH2'),  # H1-C1 -> Negative
    ]

    f1, f2, hsqc_mat = simulate_hsqc(
        h_shifts, c_shifts, correlations,
        ppm_range_f2=(2.0, 6.0),
        ppm_range_f1=(40.0, 80.0),
        n_points_f2=64, n_points_f1=64,
    )
    assert hsqc_mat.shape == (64, 64)

    # Check that (C0, H0) is positive (CH)
    idx_c0 = np.argmin(np.abs(f1 - 50.0))
    idx_h0 = np.argmin(np.abs(f2 - 3.0))
    assert hsqc_mat[idx_c0, idx_h0] > 0.0

    # Check that (C1, H1) is negative (CH2)
    idx_c1 = np.argmin(np.abs(f1 - 70.0))
    idx_h1 = np.argmin(np.abs(f2 - 4.5))
    assert hsqc_mat[idx_c1, idx_h1] < 0.0


def test_simulate_hmbc():
    h_shifts = [2.0, 7.5]
    c_shifts = [30.0, 170.0] # C1 (170 ppm) is quaternary carbonyl
    correlations = [
        (0, 1, 7.0), # H0 (2.0 ppm) couples to C1 (170.0 ppm) across 2 bonds (J = 7 Hz)
    ]

    f1, f2, hmbc_mat = simulate_hmbc(
        h_shifts, c_shifts, correlations,
        ppm_range_f2=(1.0, 8.0),
        ppm_range_f1=(20.0, 180.0),
        n_points_f2=64, n_points_f1=64,
    )
    assert hmbc_mat.shape == (64, 64)

    idx_c_quat = np.argmin(np.abs(f1 - 170.0))
    idx_h = np.argmin(np.abs(f2 - 2.0))
    assert hmbc_mat[idx_c_quat, idx_h] > 0.0


def test_plot_2d_heteronuclear(tmp_path):
    h_shifts = [3.0, 4.5]
    c_shifts = [50.0, 70.0]
    ppm_h, int_h = simulate_1d_lines(h_shifts, ppm_range=(1.0, 6.0), n_points=500, fwhm_ppm=0.05)
    ppm_c, int_c = simulate_1d_lines(c_shifts, ppm_range=(30.0, 90.0), n_points=500, fwhm_ppm=0.5)

    f1, f2, hsqc_mat = simulate_hsqc(
        h_shifts, c_shifts, [(0, 0, 'CH'), (1, 1, 'CH2')],
        ppm_range_f2=(1.0, 6.0), ppm_range_f1=(30.0, 90.0),
        n_points_f2=32, n_points_f1=32,
    )

    out_hsqc = str(tmp_path / "test_hsqc.png")
    plot_2d_spectrum(
        f1, f2, hsqc_mat,
        title="1H-13C HSQC",
        spectrum_type="HSQC",
        nucleus_f1="13C",
        nucleus_f2="1H",
        trace_1d_top=(ppm_h, int_h),
        trace_1d_left=(ppm_c, int_c),
        save_path=out_hsqc,
    )
    assert (tmp_path / "test_hsqc.png").exists()


def test_simulate_heteronuclear_2d(tmp_path):
    from frogchem.nmr_sim import simulate_heteronuclear_2d

    c_shifts = [54.5, 43.2]
    s_shifts = [0.0]
    correlations = [(0, 0, 1.0), (1, 0, 0.8)]

    f1, f2, mat = simulate_heteronuclear_2d(
        c_shifts, s_shifts, correlations,
        ppm_range_f2=(30.0, 70.0),
        ppm_range_f1=(-20.0, 20.0),
        n_points_f2=64,
        n_points_f1=64,
    )
    assert mat.shape == (64, 64)
    assert np.max(mat) > 0.0

    out_file = str(tmp_path / "test_cs.png")
    plot_2d_spectrum(
        f1, f2, mat,
        title="13C-33S Heteronuclear Correlation",
        spectrum_type="COSY",
        nucleus_f1="33S",
        nucleus_f2="13C",
        save_path=out_file,
    )
    assert (tmp_path / "test_cs.png").exists()


def test_spin_helpers():
    """Verify nuclear spin constants and helper functions."""
    assert np.isclose(get_nucleus_spin('1H'), 0.5)
    assert np.isclose(get_nucleus_spin('H'), 0.5)
    assert np.isclose(get_nucleus_spin('13C'), 0.5)
    assert np.isclose(get_nucleus_spin('C'), 0.5)
    assert np.isclose(get_nucleus_spin('17O'), 2.5)
    assert np.isclose(get_nucleus_spin('O'), 2.5)
    assert np.isclose(get_nucleus_spin('33S'), 1.5)
    assert np.isclose(get_nucleus_spin('S'), 1.5)
    assert np.isclose(get_nucleus_spin('2H'), 1.0)
    assert np.isclose(get_nucleus_spin('51V'), 3.5)

    assert format_spin(0.5) == "1/2"
    assert format_spin(1.0) == "1"
    assert format_spin(1.5) == "3/2"
    assert format_spin(2.5) == "5/2"
    assert format_spin(3.5) == "7/2"

    assert not is_quadrupolar('1H')
    assert not is_quadrupolar('13C')
    assert is_quadrupolar('17O')
    assert is_quadrupolar('33S')
    assert is_quadrupolar('2H')

    # Multiplicity 2nI + 1
    assert get_spin_multiplicity(0.5, 1) == 2  # doublet
    assert get_spin_multiplicity(0.5, 3) == 4  # quartet
    assert get_spin_multiplicity(1.0, 1) == 3  # CDCl3 triplet in 13C
    assert get_spin_multiplicity(2.5, 1) == 6  # 17O sextet


def test_solve_spin_cluster_oxygen17_equivalent():
    """Two magnetically equivalent Oxygen-17 spins (I = 5/2) should collapse
    into a single unsplit resonance frequency at nu0.
    """
    nu = np.array([250.0, 250.0])
    J = np.array([[0.0, 15.0], [15.0, 0.0]])
    trans = solve_spin_cluster(nu, J, spins=2.5)
    assert len(trans) > 0
    # Every allowed transition must occur at exactly 250.0 Hz
    for freq, prob in trans:
        assert np.isclose(freq, 250.0, atol=1e-3)
        assert prob > 0.0


def test_solve_spin_cluster_oxygen17_coupled_ab():
    """Two non-equivalent Oxygen-17 spins (I = 5/2) coupled together."""
    nu = np.array([200.0, 300.0])
    J = np.array([[0.0, 20.0], [20.0, 0.0]])
    trans = solve_spin_cluster(nu, J, spins=2.5)
    assert len(trans) > 10
    # Center of gravity of transitions should match average frequency
    weighted_freq = sum(f * p for f, p in trans) / sum(p for _, p in trans)
    assert np.isclose(weighted_freq, 250.0, atol=1e-2)


def test_solve_spin_cluster_mixed_spins():
    """Solve spin cluster with mixed spins: spin 1/2 (e.g. 1H) coupled to spin 5/2 (e.g. 17O)."""
    nu = np.array([500.0, 100.0])
    J = np.array([[0.0, 12.0], [12.0, 0.0]])
    trans = solve_spin_cluster(nu, J, spins=[0.5, 2.5])
    assert len(trans) > 0
    # Hilbert space is 2 * 6 = 12 states
    for freq, prob in trans:
        assert freq > 0.0
        assert prob > 0.0


def test_partition_spin_network_high_spin():
    """Verify that partition_spin_network respects max_dim for high-spin nuclei."""
    # 4 spins of spin 5/2: Hilbert space is 6^4 = 1296 > 256.
    # The partitioner must subdivide this into smaller clusters.
    J = np.array([
        [0.0, 10.0, 1.0, 0.5],
        [10.0, 0.0, 0.5, 1.0],
        [1.0, 0.5, 0.0, 10.0],
        [0.5, 1.0, 10.0, 0.0],
    ])
    spins = [2.5, 2.5, 2.5, 2.5]
    clusters = partition_spin_network(4, J, cutoff=0.3, max_dim=256, spins=spins)
    for c in clusters:
        dim = 1
        for idx in c:
            dim *= int(round(2.0 * spins[idx] + 1.0))
        assert dim <= 256, f"Cluster {c} exceeded max_dim 256 with dim {dim}"


def test_simulate_quantum_transitions_oxygen():
    """Verify simulate_quantum_transitions works automatically with nucleus='17O'."""
    shifts = [10.0, 20.0]
    J = np.array([[0.0, 15.0], [15.0, 0.0]])
    transitions = simulate_quantum_transitions(
        shifts_ppm=shifts,
        j_matrix=J,
        freq_mhz=500.0,
        nucleus='17O',
    )
    assert len(transitions) > 0
    # Total intensity must integrate to exactly 2.0 (2 atoms)
    total_intensity = sum(t.intensity for t in transitions)
    assert np.isclose(total_intensity, 2.0, atol=1e-3)


def test_dynamic_nuclei_database_and_spin_overrides(tmp_path):
    """Verify that nuclear database is loaded dynamically from JSON and can be customized/overridden."""
    # 1. Base database loaded from JSON
    db = load_nucleus_database()
    assert '1H' in db
    assert '17O' in db
    assert np.isclose(db['17O']['spin'], 2.5)

    # 2. Custom JSON database loading
    custom_json = tmp_path / "custom_nuclei.json"
    custom_json.write_text('{"X": {"spin": 4.5, "element": "X", "gamma": 1000.0}}')
    custom_db = load_nucleus_database(custom_json)
    assert 'X' in custom_db
    assert np.isclose(custom_db['X']['spin'], 4.5)

    # 3. parse_spin handles floats, ints, and fractions
    assert np.isclose(parse_spin("5/2"), 2.5)
    assert np.isclose(parse_spin("1/2"), 0.5)
    assert np.isclose(parse_spin("3/2"), 1.5)
    assert np.isclose(parse_spin(2.5), 2.5)
    assert np.isclose(parse_spin(1), 1.0)

    # 4. parse_spins_arg handles comma/semicolon/equals/colon syntax and JSON
    spins_dict = parse_spins_arg("O=5/2,H=1/2,C:1/2")
    assert np.isclose(spins_dict['17O'], 2.5)
    assert np.isclose(spins_dict['1H'], 0.5)
    assert np.isclose(spins_dict['13C'], 0.5)

    spins_json = parse_spins_arg('{"17O": "5/2", "33S": 1.5}')
    assert np.isclose(spins_json['17O'], 2.5)
    assert np.isclose(spins_json['33S'], 1.5)

    # 5. Dynamic set_nucleus_spin and custom_spins priority
    set_nucleus_spin('CustomNuc', '7/2')
    assert np.isclose(get_nucleus_spin('CustomNuc'), 3.5)

    # custom_spins override takes precedence
    override_val = get_nucleus_spin('1H', custom_spins={'1H': 1.0})
    assert np.isclose(override_val, 1.0)


def test_orca_detected_isotopes_parsing():
    """Verify that NMRAnalysis extracts isotopes and g-factors from ORCA output."""
    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation('files/nmr.out')
    assert calc.nmr is not None
    isotopes = calc.nmr.get_isotopes()
    assert len(isotopes) in (39, 172)
    if len(isotopes) == 172:
        assert isotopes[0] == '15N'
        assert isotopes[1] == '13C'
        assert isotopes[26] == '1H'
        assert isotopes[45] == '107Ag'
    else:
        assert isotopes[0] == '13C'
        assert isotopes[7] == '17O'
        assert isotopes[11] == '33S'
        assert isotopes[23] == '1H'


def test_periodictable_nuclear_spin():
    """Verify dynamic nuclear spin resolution via periodictable."""
    # Specific isotopes
    assert np.isclose(get_periodictable_spin('1H'), 0.5)
    assert np.isclose(get_periodictable_spin('2H'), 1.0)
    assert np.isclose(get_periodictable_spin('13C'), 0.5)
    assert np.isclose(get_periodictable_spin('17O'), 2.5)
    assert np.isclose(get_periodictable_spin('33S'), 1.5)
    assert np.isclose(get_periodictable_spin('14N'), 1.0)
    assert np.isclose(get_periodictable_spin('15N'), 0.5)
    assert np.isclose(get_periodictable_spin('19F'), 0.5)
    assert np.isclose(get_periodictable_spin('31P'), 0.5)
    assert np.isclose(get_periodictable_spin('51V'), 3.5)

    # Element symbols (auto-selecting most abundant NMR-active isotope)
    assert np.isclose(get_periodictable_spin('H'), 0.5)
    assert np.isclose(get_periodictable_spin('C'), 0.5)
    assert np.isclose(get_periodictable_spin('O'), 2.5)
    assert np.isclose(get_periodictable_spin('S'), 1.5)
    assert np.isclose(get_periodictable_spin('N'), 1.0)
    assert np.isclose(get_periodictable_spin('F'), 0.5)
    assert np.isclose(get_periodictable_spin('P'), 0.5)

    # Seamless fallback via get_nucleus_spin
    assert np.isclose(get_nucleus_spin('17O'), 2.5)
    assert np.isclose(get_nucleus_spin('O'), 2.5)
    assert np.isclose(get_nucleus_spin('33S'), 1.5)
    assert np.isclose(get_nucleus_spin('S'), 1.5)
    assert np.isclose(get_nucleus_spin('51V'), 3.5)


def test_nucleus_decoupled_resolution():
    """Verify that is_nucleus_decoupled correctly resolves coupling/decoupling per nucleus."""
    # 1. Defaults: 1H is coupled (decoupled=False), heteronuclei decoupled (decoupled=True)
    assert is_nucleus_decoupled('1H') is False
    assert is_nucleus_decoupled('H') is False
    assert is_nucleus_decoupled('13C') is True
    assert is_nucleus_decoupled('C') is True
    assert is_nucleus_decoupled('17O') is True
    assert is_nucleus_decoupled('O') is True
    assert is_nucleus_decoupled('33S') is True

    # 2. Bare flags: --coupled (all coupled) vs --decoupled (all decoupled)
    assert is_nucleus_decoupled('1H', coupled_arg=[]) is False
    assert is_nucleus_decoupled('13C', coupled_arg=[]) is False
    assert is_nucleus_decoupled('17O', coupled_arg=[]) is False

    assert is_nucleus_decoupled('1H', decoupled_arg=[]) is True
    assert is_nucleus_decoupled('13C', decoupled_arg=[]) is True
    assert is_nucleus_decoupled('17O', decoupled_arg=[]) is True

    # 3. Specific lists / comma-separated
    # Couple only Carbon:
    assert is_nucleus_decoupled('13C', coupled_arg=['C']) is False
    assert is_nucleus_decoupled('1H', coupled_arg=['C']) is False  # 1H remains coupled by default
    assert is_nucleus_decoupled('17O', coupled_arg=['C']) is True  # 17O remains decoupled by default

    # Decouple only Proton:
    assert is_nucleus_decoupled('1H', decoupled_arg=['H']) is True
    assert is_nucleus_decoupled('13C', decoupled_arg=['H']) is True  # 13C remains decoupled by default

    # Mixed CLI arguments: decouple H, couple C and O
    assert is_nucleus_decoupled('1H', coupled_arg=['C', 'O'], decoupled_arg=['H']) is True
    assert is_nucleus_decoupled('13C', coupled_arg=['C', 'O'], decoupled_arg=['H']) is False
    assert is_nucleus_decoupled('17O', coupled_arg=['C', 'O'], decoupled_arg=['H']) is False
    assert is_nucleus_decoupled('33S', coupled_arg=['C', 'O'], decoupled_arg=['H']) is True

    # Comma-separated strings
    assert is_nucleus_decoupled('13C', coupled_arg='C,17O') is False
    assert is_nucleus_decoupled('17O', coupled_arg='C,17O') is False
    assert is_nucleus_decoupled('33S', coupled_arg='C,17O') is True

    # 4. JSON database explicit flag
    db = {'17O': {'decoupled': False}}
    assert is_nucleus_decoupled('17O', nuclei_data=db) is False



