import warnings

import numpy as np
from scipy.constants import R, h, c, Boltzmann, N_A, calorie
from scipy.constants import physical_constants

__all__ = [
    "calc_damp", "calc_zpe", "free_gibbs_energy",
    "calculate_boltzmann", "calculate_bolzmann",
    "calc_er", "calc_dg",
    "calc_S_V_grimme", "calc_S_R_grimme",
    "calc_vibrational_entropy", "calc_vibrational_entropy_truhlar",
    "calc_vibrational_energy", "calc_qRRHO_energy",
    "calc_translational_energy", "calc_rotational_energy",
    "calc_translational_entropy", "calc_rotational_entropy",
    "calc_electronic_entropy", "GRIMME_BAV",
]


C_CM_S = c * 100  # speed of light in cm/s (used in spectroscopy formulas)
J_TO_H = physical_constants["joule-hartree relationship"][0]
AMU_TO_KG = physical_constants["atomic mass constant"][0]
CAL_TO_J = calorie
GRIMME_BAV = 1.00e-44  # kg * m^2 (default average moment of inertia, Grimme 2012)

# R = 8.314462618 J/(mol K)
# h = 6.62607015e-34 J*s
# c = 2.9979245800E+10 cm/s
# Boltzmann = 1.380649e-23 J/K
# J_TO_H = 2.2937122783963e+17 Eh/J
# AMU_TO_KG = 1.6605390666e-27 kg*mol/g


def calc_damp(frequency: np.ndarray, cut_off: float, alpha: int) -> np.ndarray:
    r"""
    Damping factor proportionate to frequency

    .. math::
        \frac {1}{1+(\frac {\text{cut_off}}{ν})^α}

    Damping factory NO measure unit

    :param frequency: Frequency list
    :type frequency: np.ndarray
    :param cut_off: cut off value, default is 100 cm-1
    :type cut_off: float
    :param alpha: dumping factor, default is 4
    :type alpha: int
    :return: dumping factor
    :rtype: np.ndarray
    """
    return 1 / (1 + (cut_off / frequency) ** alpha)


def calc_zpe(frequency: np.ndarray | None = None) -> float:
    r"""Calculate the Zero Point Energy

    .. math::
        ZPE = \sum_{\nu}^\text{freq} \frac 12 h\nu c

    Return in Eh

    :param frequency: frequency list, defaults to np.array([0])
    :type frequency: np.ndarray, optional
    :return: zero point energy
    :rtype: float
    """
    if frequency is None:
        frequency = np.array([0])
    return np.sum((h * frequency * C_CM_S) / (2)) * J_TO_H


def calc_translational_energy(T: float) -> float:
    r"""Translational energy

    .. math::
        U_{trans} = \frac 32 K_bT

    Return in Eh

    :param T: temperature [K]
    :type T: float
    :return: translational energy
    :rtype: float
    """

    return 1.5 * Boltzmann * T * J_TO_H


def calc_rotational_energy(T: float, linear=False) -> float:
    r"""Rotational energy

    .. math::
        U_{rot} = \frac 32 K_bT\\
        U_{rot} = K_bT 

    Return in Eh

    :param T: temperature [K]
    :type T: float
    :param linear: if the molecule is linear, defaults to False
    :type linear: bool, optional
    :return: rotational energy
    :rtype: float
    """
    if linear:
        return Boltzmann * T * J_TO_H
    return 1.5 * Boltzmann * T * J_TO_H


def calc_qRRHO_energy(freq: np.ndarray, T: float) -> np.ndarray:
    r"""quasi-Rigid Rotor Harmonic Oscillator energy

    .. math::
        U = h\nu c \frac { e^{-\frac {h\nu c}{k_bT}} }{1-e^{-\frac {h\nu c}{k_bT}}}


    :param freq: frequency list
    :type freq: np.ndarray
    :param T: temperature [K]
    :type T: float
    :return: vibrational energy for each vibrational mode in Joule
    :rtype: np.ndarray
    """
    f = h * freq * C_CM_S / (Boltzmann * T)
    return h * freq * C_CM_S * np.exp(-f) / (1 - np.exp(-f))


def calc_vibrational_energy(
    freq: np.ndarray, T: float, cut_off: float = 100, alpha: int = 4
) -> float:
    r"""
    Harmonic vibrational thermal energy contribution (Grimme mRRHO style).

    .. math::
        U_{\text{vib}} = \sum_{\nu}^{\text{freq}} \frac{h\nu c}{e^{\frac{h\nu c}{k_bT}} - 1}

    In Grimme's 2012 mRRHO approach, the vibrational internal energy and enthalpy
    are computed with the standard harmonic oscillator partition function.

    :param freq: frequency
    :type freq: array
    :param T: temperature
    :type T: float
    :param cut_off: frequency cutoff (retained for API compatibility)
    :type cut_off: float, optional
    :param alpha: damping factor (retained for API compatibility)
    :type alpha: int, optional

    :return: vibrational energy in Eh
    :rtype: float
    """
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]
    if len(freq) == 0:
        return 0.0
    return float(np.sum(calc_qRRHO_energy(freq, T)) * J_TO_H)


def calc_translational_entropy(MW: float, T: float, P: float) -> float:
    r"""
    Translational entropy

    .. math::
        S_{trans} = k_b \left(\frac 52 + \ln\left(\sqrt{\frac{2πMWk_bT}{N_A*h^2}}^3 \frac {k_bT}{p}\right)\right)


    :param MW: molecular weight
    :type MW: float
    :param T: temperature
    :type T: float
    :param P: pressure [kPa]
    :type P: float


    :return: translational entropy in Eh/K
    :rtype: float
    """

    lambda_ = np.sqrt((2 * np.pi * MW * Boltzmann * T) / (1000 * N_A * h**2))
    V = (Boltzmann * T) / (P * 1000)

    return Boltzmann * (5 / 2 + np.log(lambda_**3 * V)) * J_TO_H


def calc_rotational_entropy(B, T, symno: int = 1, linear: bool = False) -> float:
    r"""
    Rotational entropy

    .. math::
        θ_R &=& \frac {hcB}{k_b}\\
        q_{rot} &=& \sqrt{\frac {πT^3}{θ_{Rx}θ_{Ry}θ_{Rz}}}\\
        S_R &=& k_b \left(\ln\left(\frac{q_{rot}}{σ}\right) + 1.5\right)

    :param B: rotational constant [cm-1]
    :type B: np.array or list
    :param T: temperature [K]
    :type T: float
    :param symno: number of symmetry (σ), default is 1
    :type symno: int
    :param linear: if molecule is linear, default is False
    :type linear: bool

    :return: rotational entropy in Eh/K
    :rtype: float

    A rotor with fewer than three positive rotational constants (e.g. ``[0, B, B]``)
    is linear and is always treated as such, whatever ``linear`` says.
    """
    B_arr = np.atleast_1d(np.asarray(B if B is not None else [], dtype=float))
    pos_B = B_arr[B_arr > 0]
    if len(pos_B) == 0:
        return 0.0
    if len(pos_B) < 3:
        linear = True

    rot_temperature = h * C_CM_S * pos_B / Boltzmann

    if linear:
        qrot = T / rot_temperature[0]
    else:
        qrot = np.sqrt(np.pi * T**3 / np.prod(rot_temperature))

    symno = max(1, int(symno))
    return Boltzmann * (np.log(qrot / symno) + 1 + (0 if linear else 0.5)) * J_TO_H


def calc_S_V_grimme(freq: np.array, T) -> np.array:
    r"""
    V factor used for the damping of the frequency

    .. math::
        V = \frac {\frac {hc\nu}{k_bT} k_b}{e^{\frac {hc\nu}{k_bT}} - 1} - k_b \ln\left(1 - e^{-\frac {hc\nu}{k_bT}}\right)

    :param freq: frequencies [cm-1]
    :type freq: np.array
    :param T: temperature [K]
    :type T: float

    :return: V factor in J
    :rtype: float
    """
    f = h * np.asarray(freq, dtype=float) * C_CM_S / (Boltzmann * T)
    # expm1 forms: no overflow for large f, no cancellation for small f
    with np.errstate(over='ignore'):
        return (f * Boltzmann) / np.expm1(f) - Boltzmann * np.log(-np.expm1(-f))


def calc_S_R_grimme(freq: np.ndarray, T: float, B: np.ndarray | float | None = None) -> np.ndarray:
    r"""
    Free rotor entropy factor used for the quasi-RRHO interpolation (Grimme 2012).

    .. math::
        \mu &= \frac{h}{8\pi^2 \nu c}\\
        \mu' &= \frac{\mu B_{\text{av}}}{\mu + B_{\text{av}}}\\
        S_{\text{rot}} &= k_b \left( \frac{1}{2} + \ln\sqrt{\frac{8\pi^3 \mu' k_b T}{h^2}} \right)

    :param freq: frequencies [cm-1]
    :type freq: np.ndarray
    :param T: temperature [K]
    :type T: float
    :param B: rotational constants [cm-1] or average moment of inertia [kg*m^2].
              If None or empty, GRIMME_BAV (1.00e-44 kg*m^2) is used.
    :type B: np.ndarray, float, or None

    :return: free rotor entropy in J/K
    :rtype: np.ndarray
    """
    freq = np.asarray(freq, dtype=float)
    if B is None:
        bav = GRIMME_BAV
    elif isinstance(B, (int, float, np.floating, np.integer)):
        bav = float(B) if B > 0 else GRIMME_BAV
    else:
        B_arr = np.asarray(B, dtype=float)
        pos_B = B_arr[B_arr > 0]
        if len(pos_B) > 0:
            # B_i in cm^-1; moment of inertia I_i = h / (8 * pi^2 * c * B_i) in kg * m^2
            moments = h / (8 * np.pi**2 * pos_B * C_CM_S)
            bav = float(np.mean(moments))
        else:
            bav = GRIMME_BAV

    mu = h / (8 * np.pi**2 * freq * C_CM_S)
    mu_prime = mu * bav / (mu + bav)
    f = 8 * np.pi**3 * mu_prime * Boltzmann * T / h**2
    return (0.5 + np.log(f**0.5)) * Boltzmann


def calc_vibrational_entropy(freq, T, B=None, cut_off=100, alpha=4) -> float:
    r"""
    Vibrational entropy using Grimme's quasi-RRHO interpolation model (2012).

    .. math::
        \sum_{\nu}^{freq} \left(w(\nu) S_{\text{vib,HO}}(\nu) + (1-w(\nu)) S_{\text{rot}}(\nu, T, B)\right)

    :param freq: frequencies [cm-1]
    :type freq: list or np.ndarray
    :param T: temperature [K]
    :type T: float
    :param B: rotational constants [cm-1], moment of inertia, or None (defaults to GRIMME_BAV)
    :type B: np.array, float, or None
    :param cut_off: cut off for the damping of the frequency [cm-1]
    :type cut_off: float
    :param alpha: damping factor
    :type alpha: float

    :return: vibrational entropy [Eh/K]
    :rtype: float
    """
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]
    if len(freq) == 0:
        return 0.0

    s_damp = calc_damp(freq, cut_off, alpha)
    return float(
        np.sum(
            calc_S_V_grimme(freq, T) * s_damp
            + (1 - s_damp) * calc_S_R_grimme(freq, T, B)
        )
        * J_TO_H
    )


def calc_vibrational_entropy_truhlar(
    freq: np.ndarray, T: float, cut_off: float = 100.0
) -> float:
    r"""
    Vibrational entropy calculated with Truhlar's quasi-harmonic cutoff model (2011).

    All frequencies below cut_off are raised to cut_off:
    .. math::
        \tilde{\omega}_i = \max(\omega_i, \omega_{\text{cut}})

    :param freq: vibrational frequencies [cm-1]
    :type freq: np.ndarray
    :param T: temperature [K]
    :type T: float
    :param cut_off: frequency cutoff in cm-1, default 100.0
    :type cut_off: float

    :return: vibrational entropy in Eh/K
    :rtype: float
    """
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]
    if len(freq) == 0:
        return 0.0
    shifted_freq = np.maximum(freq, cut_off)
    return float(np.sum(calc_S_V_grimme(shifted_freq, T)) * J_TO_H)


def calc_electronic_entropy(m) -> float:
    r"""
    Electronic entropy

    .. math::
        S_{el} = k_b \ln(m)

    :param m: electronic multiplicity
    :type m: int

    :return: electronic entropy in Eh/K
    :rtype: float
    """
    return Boltzmann * np.log(m) * J_TO_H


def free_gibbs_energy(
    SCF: float,
    T: float,
    freq: np.ndarray,
    mw: float,
    B: np.ndarray,
    m: int,
    # defaults
    linear: bool = False,
    cut_off=100,
    alpha=4,
    P: float = 101.325,
    symno: int = 1,
    model: str = "grimme",
) -> tuple[float, float, float, float]:
    r"""
    Calculate Gibbs energy

    .. math::
        H &=& SCF + ZPVE + U_{trans} + U_{rot} + U_{vib} + k_bT\\
        S &=& S_{trans} + S_{rot} + S_{vib} + S_{el}\\
        G &=& H - TS

    :param SCF: self consistent field energy [Eh] + dispersions
    :type SCF: float
    :param T: temperature [K]
    :type T: float
    :param P: pressure [kPa], default is 101.325
    :type P: float
    :param B: rotational constant [cm-1]
    :type B: np.array
    :param m: spin multiplicity
    :type m: int

    :param linear: if molecule is linear
    :type linear: bool
    :param cut_off: frequency cut_off
    :type cut_off: float
    :param alpha: frequency damping factor
    :type alpha: int
    :param symno: rotational symmetry number (sigma), default is 1
    :type symno: int
    :param model: quasi-RRHO model for vibrational entropy, 'grimme' or 'truhlar' (default: 'grimme')
    :type model: str

    :return: Tuple of (G [Eh], ZPVE [Eh], H [Eh], S [Eh/K])
    :rtype: tuple[float, float, float, float]

    If there are vibrational frequencies but no positive rotational constant in ``B``
    (e.g. the parser could not read them), a ``RuntimeWarning`` is issued because
    S_rot is then 0. ``B`` with fewer than three positive constants is treated as linear.
    """
    freq = np.asarray(freq, dtype=float)
    freq = freq[freq > 0]

    B_arr = np.atleast_1d(np.asarray(B if B is not None else [], dtype=float))
    n_pos_B = int(np.sum(B_arr > 0))
    if len(freq) > 0 and n_pos_B == 0:
        warnings.warn("free_gibbs_energy: frequencies given but no rotational constants (B); "
                      "rotational entropy is set to 0 and G is not reliable", RuntimeWarning)
    if 0 < n_pos_B < 3:
        linear = True

    zpve = calc_zpe(freq)

    U_trans = calc_translational_energy(T)
    U_rot = calc_rotational_energy(T, linear) if zpve > 0 else 0
    U_vib = calc_vibrational_energy(freq, T, cut_off, alpha)

    H = SCF + zpve + U_trans + U_rot + U_vib + Boltzmann * T * J_TO_H

    S_elec = calc_electronic_entropy(m)
    if model.lower() == "truhlar":
        S_vib = calc_vibrational_entropy_truhlar(freq, T, cut_off=cut_off)
    else:
        S_vib = calc_vibrational_entropy(freq, T, B, cut_off, alpha)

    S_rot = calc_rotational_entropy(B, T, symno=symno, linear=linear)
    S_trans = calc_translational_entropy(mw, T, P)

    S = S_trans + S_rot + S_vib + S_elec

    return H - T * S, zpve, H, S


def calculate_boltzmann(energy: np.ndarray, T: float) -> np.ndarray:
    """Calculate Boltzmann distribution populations.

    :param energy: array of energies [Hartree]
    :param T: temperature [K]
    :return: normalized Boltzmann population fractions (sum to 1.0)
    """
    energy = np.asarray(energy, dtype=float)
    if energy.size == 0:
        raise ValueError("calculate_boltzmann: empty energy array")
    rel_energy = (energy - np.min(energy)) / J_TO_H * N_A  # J/mol (CODATA Hartree)
    boltz_factor = np.exp(-rel_energy / (R * T))
    return boltz_factor / np.sum(boltz_factor)


# Backward-compatible alias for typo
calculate_bolzmann = calculate_boltzmann


def calc_er(dg: float, T: float) -> tuple[float, float, float]:
    """Calculate enantiomeric/diastereomeric ratio and ee from free energy difference.

    :param dg: free energy difference Delta(Delta G) [kcal/mol]
    :param T: temperature [K]
    :return: tuple of (major_ratio, minor_ratio, ee_percent)
    """
    er = np.exp(-(dg * (CAL_TO_J * 1000)) / (R * T))
    maj = 1 / (er + 1)
    minor = 1 - maj
    ee = (maj - minor) / (maj + minor) * 100
    return maj, minor, ee


def calc_dg(maj: float, minor: float, T: float) -> float:
    """Calculate free energy difference Delta(Delta G) from isomer ratios.

    :param maj: major isomer ratio or percentage
    :param minor: minor isomer ratio or percentage
    :param T: temperature [K]
    :return: Delta(Delta G) [kcal/mol]
    """
    s = maj + minor
    minor = minor / s
    maj = maj / s
    er = minor / maj
    dg = -(np.log(er) * R * T) / (CAL_TO_J * 1000)
    return dg