1. **Read the structures.** `load_structures` returns every structure of an XYZ file (each one is
   analysed) or the geometry of an ORCA/Gaussian output.
2. **Rotor properties.** Centre of mass and inertia tensor are computed with atomic masses; the sorted
   eigenvalues are $I_a\le I_b\le I_c$ (amu·Å²). The rotor type is: *Linear* if $I_a<0.1$, *Spherical
   Top* if $I_a\approx I_c$, *Symmetric Top* if two moments agree, otherwise *Asymmetric Top*
   (relative tolerance 2 %).
3. **Point group** (`frogchem.symmetry.PointGroupDetector`). Coordinates are centred on the centre of
   mass of the analysed atoms (heavy atoms only with `--exclude-h`).
   * *Candidate axes*: directions of the atoms, and of the sums, differences and cross products of
     atom pairs (pairs only for fewer than 50 atoms), plus the three principal axes.
   * *Operations tested*: inversion, $C_n$ ($n$ = 8, 6, 5, 4, 3, 2) and $S_n$ ($n$ = 8, 6, 4) about
     every candidate, and a mirror plane normal to every candidate.
   * *Acceptance*: an operation $\hat O$ is accepted when every transformed atom has an atom of the
     same element within the tolerance $\delta$:
     $\max_i \min_{j\,\in\,\mathrm{el}(i)} |\hat O\mathbf r_i-\mathbf r_j| \le \delta$.
     The *RMSD* column is the largest of these displacements over all accepted operations: it is a
     maximum deviation, not a root mean square.
   * *Group*: the Schoenflies symbol is assigned from the accepted elements (several high-order
     axes → T/O/I families; otherwise principal axis, perpendicular $C_2$, $\sigma_h$, $\sigma_v$,
     $S_{2n}$). Linear molecules give `D_inf_h` or `C_inf_v`.
4. **Tolerances.** Without `-t`/`-g`: δ = 0.1, 0.2, 0.5, 1.0 Å. `-t` uses one value. With `-g`, δ is
   raised through 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 1.5 Å until the detected group equals
   the target (case-insensitive). Only an exact match counts, so a subgroup gives *False*.
5. **Standard orientation** (`-w`). The structure is moved to its centre of mass and rotated so that
   the principal axes $I_a, I_b, I_c$ lie along $x, y, z$ (right-handed frame), then written to
   `<stem>_std.xyz` (`<stem>_std_conf<N>.xyz` for multi-structure files).
