# NMR Theory & Spin Simulation

FROG implements comprehensive methods for extracting quantum chemical magnetic properties and simulating 1D and 2D NMR spectra from first principles.

---

## 1. Magnetic Shielding & Chemical Shifts

Quantum chemical calculations (e.g. via ORCA's `EPRNMR` block or Gaussian's `NMR` keyword) compute the second derivative of the electronic energy with respect to external magnetic field $\mathbf{B}$ and nuclear magnetic dipole moment $\boldsymbol{\mu}_k$:

$$\boldsymbol{\sigma}_k = \frac{\partial^2 E}{\partial \mathbf{B} \partial \boldsymbol{\mu}_k}$$

The isotropic magnetic shielding $\sigma_{\text{iso}}$ is one-third of the trace of the symmetric shielding tensor:

$$\sigma_{\text{iso}} = \frac{1}{3} \left( \sigma_{xx} + \sigma_{yy} + \sigma_{zz} \right)$$

### Linear Shift Calibration
To obtain chemical shifts $\delta$ comparable to experimental spectra (referenced against tetramethylsilane, TMS), FROG supports both direct difference and empirical linear scaling:

$$\delta_k = \frac{\sigma_{\text{ref}} - \sigma_{\text{iso}, k}}{1 - \sigma_{\text{ref}} \cdot 10^{-6}} \qquad \text{(reference shielding)}$$

$$\delta_k = \frac{\sigma_{\text{iso}, k} - b}{m} \qquad \text{(empirical linear scaling)}$$

where:
* $\sigma_{\text{ref}}$ is the isotropic shielding of the reference compound (e.g. computed at the identical level of theory), in ppm.
* $m$ and $b$ are the slope and intercept of the empirical regression $\sigma_{\text{iso}} = m\,\delta_{\text{exp}} + b$ (CHESHIRE convention, $m < 0$), compensating systematic functional/basis-set errors.

---

## 2. Chemical Equivalency Detection

In flexible molecules, observed NMR spectra reflect both **rigid point group symmetry** (e.g., $C_2$, $C_s$) and **rapid dynamic averaging** of rotators (such as methyl $-\text{CH}_3$ and phenyl rings) occurring on the NMR chemical shift timescale ($10^{-3}\text{ s}$).

FROG implements a hybrid topological and geometric detection algorithm:

1. **Global Point Group Symmetry**: Detects 3D molecular symmetry operations using the Kabsch algorithm and coordinate permutations. Nuclei mapped into each other by symmetry operations form equivalent sets.
2. **Local Graph Topology**: Identifies rotating terminal groups (e.g., $sp^3$ methyls bonded to a common heavy atom) via graph adjacency.
3. **Disjoint Set Union**: Merges topological equivalence classes with geometric symmetry classes to determine the exact observable equivalencies.

---

## 3. Quantum Spin Hamiltonian Simulation

For coupled spin-1/2 systems, first-order perturbation theory breaks down whenever the chemical shift difference $|\nu_i - \nu_j|$ is comparable to or smaller than the scalar coupling constant $|J_{ij}|$ (strong coupling regime, leading to the "roof effect" and extra transition lines).

FROG solves the exact isotropic spin Hamiltonian:

$$\hat{H} = \sum_{i} \nu_i \hat{I}_{z,i} + \sum_{i < j} J_{ij} \left( \hat{I}_{x,i} \hat{I}_{x,j} + \hat{I}_{y,i} \hat{I}_{y,j} + \hat{I}_{z,i} \hat{I}_{z,j} \right)$$

where:
* $\nu_i = \delta_i \cdot \nu_{\text{spectrometer}}$ is the resonance frequency of spin $i$ in Hz.
* $J_{ij}$ is the scalar coupling constant between spins $i$ and $j$ in Hz.
* $\hat{I}_{x,i}, \hat{I}_{y,i}, \hat{I}_{z,i}$ are the angular momentum operators for spin $i$.

### Block Diagonalization by Total $M$
Because the isotropic scalar coupling operator commutes with the total longitudinal angular momentum operator $\hat{I}_z = \sum_i \hat{I}_{z,i}$:

$$\left[ \hat{H}, \hat{I}_z \right] = 0$$

the Hamiltonian matrix of dimension $2^N \times 2^N$ decomposes into independent uncoupled sub-blocks labeled by the total magnetic quantum number $M = \sum_k m_k$:

$$\hat{H} = \bigoplus_{M = -N/2}^{+N/2} \hat{H}_M$$

Each sub-block $\hat{H}_M$ has dimension $\binom{N}{N/2 - M}$, drastically reducing the computational complexity:

| Number of Spins $N$ | Full $2^N$ Matrix | Largest Sub-block $\binom{N}{N/2}$ |
|:---:|:---:|:---:|
| 4 | 16 | 6 |
| 6 | 64 | 20 |
| 8 | 256 | 70 |
| 10 | 1024 | 252 |

### Graph Partitioning for Large Molecules
Each chemically equivalent group (e.g. a CH$_3$ with count $n=3$) is expanded into $n$ explicit spins sharing the group shift and the group–group couplings (couplings inside a magnetically equivalent group have no effect on the spectrum and are omitted), so $n$ equivalent neighbours produce the correct $(n+1)$-line pattern and integrals proportional to the counts.

FROG builds a connectivity graph with edges where $|J_{ij}| > J_{\text{threshold}}$ and partitions the groups (never splitting a group) into clusters that are diagonalized exactly, as long as a cluster has at most 8 spins and a Hilbert-space dimension $\prod (2I+1) \le 256$. Couplings between different clusters are then added to first order. If a single group alone exceeds the limit, a first-order (Pascal/binomial) multiplet is used: each line of group $k$ sits at $\nu_k + \sum_j J_{kj} M_j$, weighted by the binomial distribution of the total $M_j$ of each coupled group.

### Transition Frequencies & Probabilities
Allowed magnetic dipole transitions occur between eigenstates $|\psi_a\rangle$ and $|\psi_b\rangle$ satisfying $\Delta M = M_b - M_a = \pm 1$:

$$\Delta E_{ab} = E_b - E_a = h \nu_{ab}$$

The transition intensity (probability) is proportional to the square of the transition dipole moment:

$$I_{ab} = \left| \langle \psi_b | \hat{I}^{+} | \psi_a \rangle \right|^2 \;\propto\; \left| \langle \psi_b | \hat{I}_x | \psi_a \rangle \right|^2, \qquad \hat{I}^{+} = \sum_i \hat{I}^{+}_i$$

Only $\Delta M = +1$ pairs are enumerated, so every line is counted once; transitions at negative frequency (nuclei with $\delta < 0$) are kept.

---

## 4. 2D NMR Simulations

### 2D COSY (Correlation Spectroscopy)
2D COSY correlates spins coupled via scalar through-bond couplings $J_{ij}$:

* **Diagonal peaks**: Appear at $(\delta_i, \delta_i)$ with intensity proportional to the spin multiplicity.
* **Cross peaks**: Appear at $(\delta_i, \delta_j)$ and $(\delta_j, \delta_i)$ if and only if $J_{ij} \neq 0$.

### 2D NOESY (Nuclear Overhauser Effect Spectroscopy)
2D NOESY correlates spins through spatial proximity via dipole-dipole cross-relaxation:

* The cross-relaxation rate $\sigma_{ij}$ between two nuclei $i$ and $j$ in the extreme narrowing limit ($\omega \tau_c \ll 1$) scales with the inverse sixth power of their interatomic distance $r_{ij}$:

$$\sigma_{ij} \propto \frac{\gamma_i^2 \gamma_j^2 \tau_c}{r_{ij}^6}$$

Cross-peaks in FROG's NOESY simulations are generated with intensities proportional to $r_{ij}^{-6}$, with an interactive distance cutoff (typically $5.0\text{ \AA}$). For an equivalent group the effective distance is the $\langle r^{-6} \rangle^{-1/6}$ average over its members.

Peak volumes follow the Solomon equations, $\mathbf{I}(\tau_m) = \exp(-\mathbf{R}\,\tau_m)\,\mathbf{M}_0$ with $R_{ii} = \rho_i$ and $R_{ij} = +\sigma_{ij}$, plotted with one global sign. Consequently, cross-peaks have the **opposite** sign to the diagonal for small molecules (extreme narrowing, $\sigma_{ij} > 0$) and the **same** sign for slow tumbling ($\omega_0 \tau_c \gtrsim 1.12$, $\sigma_{ij} < 0$, negative NOE). $\omega_0$ is the Larmor frequency of the observed nucleus.

---

## Literature References

1. **Keeler, J.** (2010). *Understanding NMR Spectroscopy* (2nd ed.). John Wiley & Sons.
2. **Levitt, M. H.** (2008). *Spin Dynamics: Basics of Nuclear Magnetic Resonance* (2nd ed.). John Wiley & Sons.
3. **Ernst, R. R., Bodenhausen, G., & Wokaun, A.** (1987). *Principles of Nuclear Magnetic Resonance in One and Two Dimensions*. Oxford University Press.
