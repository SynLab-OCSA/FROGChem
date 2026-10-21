# Spectroscopy & Excited States (TD-DFT)

FROG includes specialized parsers and visualization tools for excited-state calculations performed with ORCA and Gaussian.

---

## 1. TD-DFT Spectral Convolution

Quantum chemistry calculations provide discrete excitation energies $E_k$ (in $\text{eV}$ or $\text{cm}^{-1}$) and dimensionless oscillator strengths $f_k$.

To generate a continuous UV-Vis absorption spectrum $\varepsilon(\tilde{\nu})$ comparable to experimental spectrophotometry, each discrete transition is convoluted with a Gaussian line-broadening function:

$$\varepsilon(\tilde{\nu}) = \frac{\pi e^2 N_A}{1000 \ln(10) m_e c^2} \sum_{k} \frac{f_k}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(\tilde{\nu} - \tilde{\nu}_k)^2}{2\sigma^2} \right)$$

with $\pi e^2 N_A / (1000 \ln(10) m_e c^2) = 2.315 \times 10^{8}\ \text{L mol}^{-1}\text{cm}^{-2}$ (cgs, $\sigma$ in $\text{cm}^{-1}$); e.g. $f = 1$, $\sigma = 1000\ \text{cm}^{-1}$ gives $\varepsilon_{\max} \approx 9.24 \times 10^{4}\ \text{L mol}^{-1}\text{cm}^{-1}$.

> `frog-tddft` plots area-normalised Gaussians $\sum_k f_k\, g_k$ (FWHM given with `--fwhm`, in eV); the curve has the shape of $\varepsilon$ but no absolute prefactor.

where:
* $\tilde{\nu}_k$ is the transition wavenumber.
* $\sigma$ is the standard deviation parameter controlling the half-width at half-maximum (HWHM):
  $$\text{HWHM} = \sigma \sqrt{2\ln 2} \approx 1.1774\,\sigma$$
* Typical broadening values used in practice are $\sigma \approx 0.1\text{ to }0.3\text{ eV}$ ($800\text{ to }2500\text{ cm}^{-1}$).

---

## 2. Electronic Circular Dichroism (ECD)

For chiral molecules, the rotatory strength $R_k$ in velocity or dipole representation (given in cgs units $10^{-40}\text{ esu}\cdot\text{cm}\cdot\text{erg/G}$) governs the differential absorption $\Delta \varepsilon = \varepsilon_L - \varepsilon_R$:

$$\Delta \varepsilon(\tilde{\nu}) = \frac{1}{22.96} \frac{1}{\sigma \sqrt{2\pi}} \sum_k R_k \tilde{\nu}_k \exp\left( -\frac{(\tilde{\nu} - \tilde{\nu}_k)^2}{2\sigma^2} \right)$$

with $R_k$ in units of $10^{-40}$ cgs and $\tilde{\nu}$, $\sigma$ in $\text{cm}^{-1}$ (equivalently $1/(2.296 \times 10^{-39})$ when $R_k$ is in absolute cgs units, $\text{esu}^2\text{cm}^2$).

> `frog-tddft --ecd` plots $\sum_k E_k R_k\, g_k$ (area-normalised Gaussians weighted by the transition energy), i.e. the shape of $\Delta\varepsilon$ without the absolute prefactor.

---

## 3. CIE 1931 Chromaticity Coordinates

Using `colour-science`, `frog-cie` can convert transmitted UV-Vis absorption spectra into perceived perceived colors in the **CIE 1931 color space**.

Given a transmission spectrum $T(\lambda) = 10^{-A(\lambda)}$ and the CIE Standard Illuminant $S(\lambda)$ (e.g., standard daylight D65), the tristimulus values $(X, Y, Z)$ are computed by integrating against the color matching functions $(\bar{x}(\lambda), \bar{y}(\lambda), \bar{z}(\lambda))$:

$$X = \frac{1}{k} \int_{380}^{780} T(\lambda) S(\lambda) \bar{x}(\lambda) d\lambda$$
$$Y = \frac{1}{k} \int_{380}^{780} T(\lambda) S(\lambda) \bar{y}(\lambda) d\lambda$$
$$Z = \frac{1}{k} \int_{380}^{780} T(\lambda) S(\lambda) \bar{z}(\lambda) d\lambda$$

with normalization factor:
$$k = \int_{380}^{780} S(\lambda) \bar{y}(\lambda) d\lambda$$

The 2D chromaticity coordinates are then:
$$x = \frac{X}{X + Y + Z}, \quad y = \frac{Y}{X + Y + Z}$$
This enables the direct prediction of perceived chemical sample colors from TD-DFT computed spectra.
