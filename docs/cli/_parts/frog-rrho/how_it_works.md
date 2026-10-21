For every file the command reads the electronic energy $E$, the vibrational frequencies, the rotational
constants $B$, the molecular mass and the spin multiplicity. It then rebuilds the thermochemistry from scratch
with the ideal-gas / rigid-rotor / quasi-harmonic-oscillator model (`frogchem.thermo.free_gibbs_energy`):

$$
H = E + \mathrm{ZPVE} + U_\mathrm{trans} + U_\mathrm{rot} + U_\mathrm{vib} + k_\mathrm{B}T
\qquad
S = S_\mathrm{trans} + S_\mathrm{rot} + S_\mathrm{vib} + S_\mathrm{el}
\qquad
G = H - TS
$$

1. **Frequencies.** Only real (positive) frequencies enter ZPVE, $U_\mathrm{vib}$ and $S_\mathrm{vib}$.
   Imaginary ones are listed in the table but left out of the thermochemistry.
2. **Low modes (quasi-RRHO).** Below the cut-off ($\nu_0$ = `--cutoff`, 100 cm⁻¹) the harmonic entropy
   diverges. With `-m grimme` (default) the entropy of each mode is a blend of the harmonic value and the
   entropy of a free rotor with the same moment of inertia. The blend is weighted by the damping function
   $w(\nu) = 1/\left[1 + (\nu_0/\nu)^{\alpha}\right]$ (`-a`, $\alpha = 4$). With `-m truhlar` every
   frequency below $\nu_0$ is simply raised to $\nu_0$.
3. **Rotation and translation.** $S_\mathrm{rot}$ uses the rotational constants and the symmetry number σ
   (`-s`). A molecule with fewer than three non-zero constants is treated as linear. $S_\mathrm{trans}$ uses
   the mass, $T$ (`-T`) and $P$ (`-P`, kPa).
4. **Electronic entropy.** $S_\mathrm{el} = R \ln(2S+1)$ from the multiplicity.
5. **Single-point correction** (`--sp-path`). The single-point energy replaces $E$, and the thermal
   correction $G - E$ of the frequency job is added to it.
6. **Populations** (`-b`). $p_i \propto \exp(-G_i/RT)$ over all files that have frequencies.

The full derivation is in [Thermochemistry theory](../theory/thermo.md).
