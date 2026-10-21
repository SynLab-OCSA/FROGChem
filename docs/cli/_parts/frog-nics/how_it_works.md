The command only does geometry: it reads every structure in the file: all frames of an XYZ, or the single parsed
geometry of an ORCA/Gaussian output. For each one:

1. **Index check.** The indices given with `-i` are 0-based. Any index outside $0 \ldots N-1$ stops the
   command, and repeated indices count once.
2. **Centroid.** The ghost position is the unweighted mean of the selected coordinates:

   $$
   \mathbf r_\mathrm{ghost} = \frac{1}{n}\sum_{k \in \texttt{-i}} \mathbf r_k
   $$

   No mass weighting, plane fitting or normal vector is computed. For a ring this is the NICS(0) point. The
   point is not moved out of the plane.
3. **Ghost atom.** An `H:` line is appended to the atom list. In ORCA the colon marks a ghost centre:
   basis functions, but no nucleus and no electrons. Its basis is replaced by the fixed string
   `NewGTO S 1 1 1e6 1 end NewAuxJGTO S 1 1 2e6 1 end`. That is a single s function with exponent 10⁶
   (2·10⁶ in the auxiliary basis). It is so tight that it has practically no overlap with the molecule's
   basis.
4. **Output.** The table lists every atom (Å, 6 decimals) and the ghost. `-w` writes the same lines as
   plain ORCA coordinates to `<stem>_nics.txt`. Multi-structure files give `<stem>_nics_<n>.txt`.

ORCA then computes the shielding tensor of the ghost with `! NMR`. NICS is its isotropic shielding with the
sign reversed, $\mathrm{NICS} = -\sigma_\mathrm{iso}$, so negative values mean aromatic. The command does not
read or compute any shielding. Shieldings in general are discussed in [NMR theory](../theory/nmr.md).
