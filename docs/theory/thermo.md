# Quasi-RRHO Thermochemistry

Gas-phase thermochemical quantities (enthalpy $H$, entropy $S$, and Gibbs free energy $G$) calculated via the standard Rigid Rotor-Harmonic Oscillator (RRHO) approximation suffer from catastrophic breakdown for low-frequency vibrational modes ($\omega < 100\text{ cm}^{-1}$).

Because the vibrational entropy of a harmonic oscillator diverges as $\omega \to 0$:
$$S_{\text{vib}} \propto -\ln\left(\frac{\hbar \omega}{k_B T}\right) \xrightarrow{\omega \to 0} +\infty$$
low vibrational frequencies (typically representing hindered internal rotations or large-amplitude conformational flexions) lead to severe overestimation of entropy and unphysically negative Gibbs free energies.

FROG implements the **quasi-RRHO** approximations to remedy this failure.

---

## 1. Grimme's Interpolation Model (qRRHO)

Grimme (*Chem. Eur. J.* **2012**, *18*, 9955–9964) proposed interpolating continuously between the harmonic oscillator entropy $S_{\text{HO}}$ and a free-rotor entropy $S_{\text{rot}}$ using a damping function:

$$S_{\text{qRRHO}}(\omega_i) = w(\omega_i) S_{\text{HO}}(\omega_i) + \left[1 - w(\omega_i)\right] S_{\text{rot}}(\omega_i)$$

where the weighting function is:
$$w(\omega_i) = \frac{1}{1 + \left(\frac{\omega_0}{\omega_i}\right)^4}$$

Here, $\omega_0$ is the cutoff parameter (typically set to $100\text{ cm}^{-1}$).

### Free Rotor Entropy
For a free rotor, the effective moment of inertia $\mu$ of mode $\omega_i$ is given by:
$$\mu = \frac{h}{8\pi^2 c \omega_i}$$
To prevent unphysical divergence as $\omega_i \to 0$, an effective moment of inertia $\mu'$ is defined by interpolation with the average molecular moment of inertia $B_{\text{av}}$:
$$\mu' = \frac{\mu B_{\text{av}}}{\mu + B_{\text{av}}}$$
where $B_{\text{av}}$ is either derived from the principal moments of inertia ($B_{\text{av}} = \frac{1}{3}\sum_i \frac{h}{8\pi^2 c B_i}$) or taken as Grimme's universal default ($1.00 \times 10^{-44}\text{ kg}\cdot\text{m}^2$). The corresponding free rotor entropy is:
$$S_{\text{rot}}(\omega_i) = R \left( \frac{1}{2} + \ln\sqrt{\frac{8\pi^3 \mu' k_B T}{h^2}} \right)$$

As $\omega_i \to 0$, $w(\omega_i) \to 0$, $\mu' \to B_{\text{av}}$, and the entropy smoothly converges to the finite value of a 1D free rotor instead of diverging to infinity.

---

## 2. Truhlar's Cutoff Model

Truhlar and coworkers (*J. Phys. Chem. B* **2011**, *115*, 14556–14562) introduced a hard frequency shift cutoff: all vibrational frequencies below $\omega_{\text{cut}}$ (typically $100\text{ cm}^{-1}$) are raised to $\omega_{\text{cut}}$:

$$\tilde{\omega}_i = \max(\omega_i, \omega_{\text{cut}})$$

The standard harmonic oscillator formulas are then evaluated using the shifted frequencies $\tilde{\omega}_i$.

---

## 3. Total Gibbs Free Energy

The overall Gibbs free energy $G(T, p)$ at temperature $T$ and pressure $p$ is computed as:

$$G(T, p) = E_{\text{el}} + \text{ZPE} + \Delta H_{0 \to T} - T \cdot S_{\text{total}}(T, p)$$

where:
* $E_{\text{el}}$ is the electronic SCF/DFT single-point energy (in Hartree).
* $\text{ZPE}$ is the zero-point vibrational energy:
  $$\text{ZPE} = \sum_{i} \frac{1}{2} h c \omega_i$$
* $\Delta H_{0 \to T} = \Delta H_{\text{trans}} + \Delta H_{\text{rot}} + \Delta H_{\text{vib}} + k_B T$.
* $S_{\text{total}} = S_{\text{trans}} + S_{\text{rot}} + S_{\text{vib,qRRHO}} + S_{\text{elec}}$.

### Linear rotors

The rotational terms depend on the shape of the molecule. A non-linear molecule has
$U_{\text{rot}} = \tfrac32 k_B T$ and $q_{\text{rot}} = \sqrt{\pi T^3/(\theta_A\theta_B\theta_C)}$;
a linear molecule has $U_{\text{rot}} = k_B T$, $q_{\text{rot}} = T/\theta_B$ and
$S_{\text{rot}} = k_B[\ln(q_{\text{rot}}/\sigma) + 1]$ instead of $k_B[\ln(q_{\text{rot}}/\sigma) + \tfrac32]$
($\theta_i = hcB_i/k_B$, $\sigma$ = rotational symmetry number). FROG detects linear molecules
automatically: when fewer than three rotational constants are positive (e.g. ORCA prints
`0.000000 B B`), the linear formulas are used whatever the `linear` argument (`--linear` in
`frog-rrho`) says; `linear=True` forces them for any molecule.

### Missing rotational constants

If frequencies are available but no positive rotational constant is (the parser could not read
them), `free_gibbs_energy` issues a `RuntimeWarning`
(`frequencies given but no rotational constants (B)`): $S_{\text{rot}}$ is set to zero and the free
rotor of Grimme's model falls back to $B_{\text{av}} = 1.00\times10^{-44}$ kg·m². $G$ is then too high
by roughly $T S_{\text{rot}}$, several kcal/mol (≈ 9 kcal/mol for the repository example
`files/test.out`), and must not be used.

---

## Literature References

If you use the Quasi-RRHO thermochemistry features in FROG, please cite:

1. **Grimme, S.** (2012). *Supramolecular Binding Thermodynamics: Beyond the Rigid-Rotor-Harmonic-Oscillator Approximation.* Chem. Eur. J., 18: 9955-9964. [doi:10.1002/chem.201200497](https://doi.org/10.1002/chem.201200497)
2. **Ribeiro, R. F., Marenich, A. V., Cramer, C. J., Truhlar, D. G.** (2011). *Use of Solution-Phase Vibrational Frequencies in Continuum Models for the Free Energy of Solvation.* J. Phys. Chem. B, 115(49): 14556–14562. [doi:10.1021/jp205508z](https://doi.org/10.1021/jp205508z)
