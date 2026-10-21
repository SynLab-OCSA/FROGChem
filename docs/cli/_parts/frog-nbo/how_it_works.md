frog does not run NBO. It parses the text that the NBO program (NBO 7 in ORCA, NBO 3.1 in Gaussian)
wrote into the output, and builds orbital cubes on request.

1. **Parsing.** From the NBO section of the output the parser builds four tables:
   * NPA charges (charge, valence and Rydberg populations per atom);
   * the NBO list (number, type `CR/LP/BD/BD*/RY…`, label, atoms, occupancy, energy, hybrid composition);
   * the second-order perturbation table (donor, acceptor, $E^{(2)}$ in kcal/mol, $\Delta E = \varepsilon_j - \varepsilon_i$ in au);
   * the Wiberg matrix and atom totals, when the output contains them (`BNDIDX`).

   Optimisations print one analysis per cycle. The last one is used, or the one chosen with `--cycle`.
2. **Summary** (`--summary`). Total electrons = Σ occupancies over all NBOs. Non-Lewis = the occupancies of
   `BD*`, `LP*`, `LV`, `RY`, `RY*`, and Lewis = the rest. The summary also shows the three most positive and three most
   negative NPA charges and the five largest $E^{(2)}$.
3. **Filters.** `-i` (0-based) keeps the rows that involve any of the given atoms. `--dtype`/`--atype` keep $E^{(2)}$
   rows whose donor/acceptor type *contains* the string (so `BD` also matches `BD*`). `-t` (2 kcal/mol) is the
   threshold. The *Aggregated* table sums $E^{(2)}$ over rows with the same donor type + label and acceptor
   type + label, e.g. the separate entries of two lone pairs of one O into the same σ*.
4. **Cubes** (`--cube N…`, `--cube-e2 K`). The NBO coefficients in the AO basis come from, in order:
   * a `.37` file (`--coeff-file`, or found next to the input);
   * a Gaussian `.fchk` written with `savenbo`;
   * a reconstruction from the hybrid coefficients printed in the output, placed on the AOs of each atom in order.
     For every `BD*` that is not printed, the coefficients are rebuilt from its `BD` partner,
     $\sigma^* = c_2 h_1 - c_1 h_2$ (with $\sigma = c_1 h_1 + c_2 h_2$).

   Each orbital $\phi(\mathbf{r}) = \sum_\mu c_\mu \chi_\mu(\mathbf{r})$ is evaluated on a box around the
   molecule (`--spacing` 0.15 Å, `--padding` 5 Å, or `--voxels` per axis). The cubes are written to
   `<stem>_nbo_cubes/` together with ChimeraX/PyMOL scripts.
5. **Ensemble** (`-c`). NPA charges, Wiberg indices and $E^{(2)}$ are Boltzmann-averaged
   ($\sum_k w_k x_k / \sum_k w_k$), see [Conformer ensembles](../user-guide/ensemble.md).
