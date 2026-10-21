import numpy as np
import pytest
from frogchem.thermo import (
    calc_zpe,
    calc_damp,
    calc_translational_energy,
    calc_rotational_energy,
    calc_qRRHO_energy,
    calc_vibrational_energy,
    calc_translational_entropy,
    calc_rotational_entropy,
    calc_electronic_entropy,
    calc_vibrational_entropy,
    calc_vibrational_entropy_truhlar,
    calc_S_V_grimme,
    calc_S_R_grimme,
    calculate_boltzmann,
    calculate_bolzmann,
    calc_er,
    calc_dg,
    free_gibbs_energy,
    GRIMME_BAV,
)


class TestCalcDamp:
    def test_damp_below_cutoff(self):
        result = calc_damp(np.array([50.0]), cut_off=100.0, alpha=4)
        expected = 1 / (1 + (100 / 50) ** 4)
        assert np.isclose(result[0], expected)

    def test_damp_above_cutoff(self):
        result = calc_damp(np.array([200.0]), cut_off=100.0, alpha=4)
        expected = 1 / (1 + (100 / 200) ** 4)
        assert np.isclose(result[0], expected)

    def test_damp_at_cutoff(self):
        result = calc_damp(np.array([100.0]), cut_off=100.0, alpha=4)
        assert np.isclose(result[0], 0.5)

    def test_damp_multiple_freqs(self):
        freqs = np.array([50.0, 100.0, 200.0])
        result = calc_damp(freqs, cut_off=100.0, alpha=4)
        assert len(result) == 3
        assert np.all(result > 0) and np.all(result <= 1)


class TestCalcZPE:
    def test_zero_frequency(self):
        assert calc_zpe(np.array([0.0])) == 0.0

    def test_single_frequency(self):
        zpe = calc_zpe(np.array([1000.0]))
        assert zpe > 0

    def test_multiple_frequencies(self):
        zpe_1 = calc_zpe(np.array([1000.0]))
        zpe_3 = calc_zpe(np.array([1000.0, 2000.0, 3000.0]))
        assert zpe_3 > zpe_1

    def test_no_frequencies(self):
        zpe = calc_zpe(np.array([]))
        assert zpe == 0.0


class TestTranslationalEnergy:
    def test_at_zero_kelvin(self):
        assert calc_translational_energy(0.0) == 0.0

    def test_at_room_temp(self):
        e = calc_translational_energy(298.15)
        assert e > 0

    def test_linear_with_temperature(self):
        e298 = calc_translational_energy(298.15)
        e596 = calc_translational_energy(596.3)
        assert np.isclose(e596 / e298, 2.0, rtol=1e-3)


class TestRotationalEnergy:
    def test_nonlinear_at_zero(self):
        assert calc_rotational_energy(0.0, linear=False) == 0.0

    def test_linear_at_zero(self):
        assert calc_rotational_energy(0.0, linear=True) == 0.0

    def test_nonlinear_vs_linear(self):
        e_nl = calc_rotational_energy(298.15, linear=False)
        e_l = calc_rotational_energy(298.15, linear=True)
        assert e_nl > e_l

    def test_nonlinear_ratio(self):
        e_nl = calc_rotational_energy(298.15, linear=False)
        e_l = calc_rotational_energy(298.15, linear=True)
        assert np.isclose(e_nl / e_l, 1.5)


class TestQRRHOEnergy:
    def test_zero_freq_nan(self):
        result = calc_qRRHO_energy(np.array([0.0]), 298.15)
        assert np.isnan(result[0])

    def test_positive_freq(self):
        result = calc_qRRHO_energy(np.array([1000.0]), 298.15)
        assert result[0] > 0

    def test_lower_freq_higher_population(self):
        e_low = calc_qRRHO_energy(np.array([10.0]), 298.15)
        e_high = calc_qRRHO_energy(np.array([100.0]), 298.15)
        assert e_low[0] > e_high[0]


class TestVibrationalEnergy:
    def test_no_freqs(self):
        e = calc_vibrational_energy(np.array([]), 298.15, cut_off=100, alpha=4)
        assert e >= 0

    def test_positive_freq(self):
        e = calc_vibrational_energy(np.array([1000.0]), 298.15, cut_off=100, alpha=4)
        assert e > 0

    def test_temperature_dependence(self):
        e_low = calc_vibrational_energy(np.array([1000.0]), 100.0, cut_off=100, alpha=4)
        e_high = calc_vibrational_energy(np.array([1000.0]), 500.0, cut_off=100, alpha=4)
        assert e_high != e_low


class TestTranslationalEntropy:
    def test_positive_entropy(self):
        s = calc_translational_entropy(MW=18.0, T=298.15, P=101.325)
        assert s > 0

    def test_increasing_with_temperature(self):
        s_low = calc_translational_entropy(MW=18.0, T=200.0, P=101.325)
        s_high = calc_translational_entropy(MW=18.0, T=400.0, P=101.325)
        assert s_high > s_low


class TestRotationalEntropy:
    def test_nonlinear_positive(self):
        B = np.array([1.0, 2.0, 3.0])
        s = calc_rotational_entropy(B, T=298.15, symno=1, linear=False)
        assert s > 0

    def test_linear_positive(self):
        B = np.array([1.0])
        s = calc_rotational_entropy(B, T=298.15, symno=1, linear=True)
        assert s > 0

    def test_symmetry_reduces_entropy(self):
        B = np.array([1.0, 2.0, 3.0])
        s1 = calc_rotational_entropy(B, T=298.15, symno=1, linear=False)
        s2 = calc_rotational_entropy(B, T=298.15, symno=2, linear=False)
        assert s2 < s1

    def test_linear_with_zeros(self):
        B = np.array([0.0, 1.5, 1.5])
        s = calc_rotational_entropy(B, T=298.15, symno=1, linear=True)
        assert s > 0



class TestElectronicEntropy:
    def test_singlet(self):
        assert calc_electronic_entropy(1) == 0.0

    def test_doublet(self):
        s = calc_electronic_entropy(2)
        assert s > 0

    def test_increasing_with_multiplicity(self):
        s2 = calc_electronic_entropy(2)
        s4 = calc_electronic_entropy(4)
        assert s4 > s2


class TestSVGrimme:
    def test_small_freq(self):
        result = calc_S_V_grimme(np.array([1.0]), 298.15)
        assert np.isfinite(result[0])

    def test_positive_freq(self):
        result = calc_S_V_grimme(np.array([1000.0]), 298.15)
        assert np.isfinite(result[0]) and result[0] >= 0


class TestSRGrimme:
    def test_returns_finite(self):
        result = calc_S_R_grimme(np.array([50.0]), 298.15, B=np.array([1.0, 2.0, 3.0]))
        assert np.isfinite(result[0])

    def test_default_bav_when_none(self):
        result1 = calc_S_R_grimme(np.array([50.0]), 298.15, B=None)
        result2 = calc_S_R_grimme(np.array([50.0]), 298.15, B=GRIMME_BAV)
        assert np.isclose(result1[0], result2[0])

    def test_filter_zero_constants(self):
        result = calc_S_R_grimme(np.array([50.0]), 298.15, B=np.array([0.0, 1.5, 1.5]))
        assert np.isfinite(result[0])

    def test_scalar_bav(self):
        result = calc_S_R_grimme(np.array([50.0]), 298.15, B=2.0e-44)
        assert np.isfinite(result[0])


class TestVibrationalEntropy:
    def test_no_freqs(self):
        s = calc_vibrational_entropy(np.array([]), T=298.15, B=np.array([1.0, 2.0, 3.0]))
        assert s == 0.0

    def test_positive_freqs(self):
        s = calc_vibrational_entropy(
            np.array([500.0, 1000.0]), T=298.15, B=np.array([1.0, 2.0, 3.0])
        )
        assert s >= 0


class TestTruhlarEntropy:
    def test_empty_frequencies(self):
        assert calc_vibrational_entropy_truhlar(np.array([]), 298.15) == 0.0

    def test_cutoff_raising(self):
        # Frequencies [20, 50] with cut_off=100 should equal harmonic entropy evaluated at [100, 100]
        s_truhlar = calc_vibrational_entropy_truhlar(np.array([20.0, 50.0]), 298.15, cut_off=100.0)
        from frogchem.thermo import J_TO_H
        s_ho_100 = np.sum(calc_S_V_grimme(np.array([100.0, 100.0]), 298.15)) * J_TO_H
        assert np.isclose(s_truhlar, s_ho_100)

    def test_higher_frequencies_unchanged(self):
        s_truhlar = calc_vibrational_entropy_truhlar(np.array([200.0, 500.0]), 298.15, cut_off=100.0)
        from frogchem.thermo import J_TO_H
        s_ho = np.sum(calc_S_V_grimme(np.array([200.0, 500.0]), 298.15)) * J_TO_H
        assert np.isclose(s_truhlar, s_ho)


class TestBoltzmann:
    def test_equal_energies(self):
        probs = calculate_bolzmann(np.array([100.0, 100.0]), T=298.15)
        np.testing.assert_allclose(probs, [0.5, 0.5])
        assert np.isclose(np.sum(probs), 1.0)

    def test_one_state(self):
        probs = calculate_bolzmann(np.array([100.0]), T=298.15)
        np.testing.assert_allclose(probs, [1.0])

    def test_three_states_unequal(self):
        probs = calculate_bolzmann(np.array([0.0, 0.001, 0.002]), T=298.15)
        assert np.isclose(np.sum(probs), 1.0)
        assert probs[0] > probs[1] > probs[2]

    def test_correct_spelling_alias(self):
        arr = np.array([0.0, 0.001, 0.002])
        probs1 = calculate_boltzmann(arr, T=298.15)
        probs2 = calculate_bolzmann(arr, T=298.15)
        np.testing.assert_allclose(probs1, probs2)


class TestER:
    def test_er_zero_dg(self):
        maj, minor, ee = calc_er(0.0, 298.15)
        assert np.isclose(maj, 0.5)
        assert np.isclose(ee, 0.0)

    def test_er_positive_dg(self):
        maj, minor, ee = calc_er(1.0, 298.15)
        assert maj > 0.5
        assert 0 < ee <= 100

    def test_er_negative_dg(self):
        maj, minor, ee = calc_er(-1.0, 298.15)
        assert maj < 0.5


class TestDG:
    def test_reversible(self):
        dg = calc_dg(0.5, 0.5, 298.15)
        assert np.isclose(dg, 0.0, atol=1e-6)

    def test_positive_dg_from_ratio(self):
        dg = calc_dg(0.75, 0.25, 298.15)
        assert dg > 0

    def test_roundtrip(self):
        dg_orig = 2.0
        maj, minor, _ = calc_er(dg_orig, 298.15)
        dg_back = calc_dg(maj, minor, 298.15)
        assert np.isclose(dg_orig, dg_back, rtol=1e-3)


class TestFreeGibbsEnergy:
    def test_basic_call(self):
        freq = np.array([500.0, 1000.0, 1500.0])
        result = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0, 2.0, 3.0]), m=1,
        )
        G, zpve, H, S = result
        assert np.isfinite(G)
        assert zpve > 0
        assert np.isfinite(H)
        assert np.isfinite(S)

    def test_linear_molecule(self):
        freq = np.array([500.0, 1000.0])
        result = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0]), m=1, linear=True,
        )
        G, zpve, H, S = result
        assert np.isfinite(G)

    def test_linear_molecule_with_zero_constant(self):
        freq = np.array([500.0, 1000.0])
        result = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([0.0, 1.5, 1.5]), m=1, linear=True,
        )
        G, zpve, H, S = result
        assert np.isfinite(G)

    def test_symmetry_number(self):
        freq = np.array([500.0, 1000.0, 1500.0])
        from scipy.constants import Boltzmann, physical_constants
        J_TO_H = physical_constants["joule-hartree relationship"][0]
        res1 = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0, 2.0, 3.0]), m=1, symno=1,
        )
        res2 = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0, 2.0, 3.0]), m=1, symno=2,
        )
        delta_S = res1[3] - res2[3]
        expected_delta_S = Boltzmann * np.log(2) * J_TO_H
        assert np.isclose(delta_S, expected_delta_S)
        assert np.isclose(res2[0] - res1[0], 298.15 * expected_delta_S)

    def test_truhlar_model(self):
        freq = np.array([50.0, 1000.0, 1500.0])
        res_grimme = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0, 2.0, 3.0]), m=1, model="grimme",
        )
        res_truhlar = free_gibbs_energy(
            SCF=-100.0, T=298.15, freq=freq, mw=18.0,
            B=np.array([1.0, 2.0, 3.0]), m=1, model="truhlar",
        )
        assert np.isfinite(res_truhlar[0])
        assert res_grimme[2] == res_truhlar[2]  # Enthalpy is the same
        assert res_grimme[3] != res_truhlar[3]  # Entropy differs due to quasi-harmonic model
