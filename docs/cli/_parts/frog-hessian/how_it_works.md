The command works on the normal-mode matrix printed by ORCA. It does not rebuild or re-diagonalise the
Hessian.

1. **Parsing.** From the ORCA output it reads the last `VIBRATIONAL FREQUENCIES` list (cm⁻¹; imaginary
   modes are negative) and the last `NORMAL MODES` block, a $3N \times 3N$ matrix. Column $k$ is mode $k$
   (0-based, with the six zero translations/rotations included) and holds the Cartesian displacements
   $(d_{ix}, d_{iy}, d_{iz})$ of every atom $i$. A file without both is reported as an error, and the command
   then exits non-zero after the other files.
2. **Mode selection.** By default (`-m -1`) all modes with $\nu < 0$ are used. `-m 6 7 …` takes the
   given columns, and `--all-modes` takes every mode with $\nu \neq 0$.
3. **Atomic contributions.** For atom $i$ in mode $k$:

$$
p_i = \frac{|\mathbf{d}_i|^2}{\sum_j |\mathbf{d}_j|^2}
\qquad
\text{shown as } 100\,p_i\ (\%)
$$

   Values at or below `-t` (5 %) are blanked, and atoms with no value above it in any selected mode are dropped.
4. **Fragments** (`--fragment NAME:i,j,k`). The fragment value is $\sum_{i \in F} 100\,p_i$, computed from the full,
   unthresholded vector.
5. **IPR** (`--ipr`). $\mathrm{IPR} = \sum_i p_i^2$. It ranges from $1/N$ (motion spread evenly over all atoms)
   to 1 (a single atom).
6. **Displaced geometries** (`-d`). For `--number-geometries` evenly spaced steps $s$ from
   −`--max-displacement` to +`--max-displacement`, the command writes the frame
   $\mathbf{R}(s) = \mathbf{R}_0 + s\,\mathbf{d}^{(k)}$ to `<stem>_mode<k>_displ.xyz`. $\mathbf{d}^{(k)}$ is the
   printed mode vector (normalised over all $3N$ components), so no atom moves by more than |s| Å.

Because ORCA prints mass-weighted normalised modes, light atoms (H) get larger $p_i$ than heavy ones for the
same motion. See [Thermochemistry theory](../theory/thermo.md) for how the frequencies enter G.
