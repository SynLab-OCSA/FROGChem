The command recomputes nothing: it copies the atomic charges that the quantum-chemistry program already
printed, and lines them up by atom index.

1. **Parsing.** Each file is read with the ORCA or Gaussian parser, which stores two dictionaries
   indexed by the **0-based** atom number, `mulliken` and `loewdin` (element + charge in e). The last block of
   the file is kept:
   * ORCA: `MULLIKEN ATOMIC CHARGES` and `LOEWDIN ATOMIC CHARGES` (for open shell, the
     `... CHARGES AND SPIN POPULATIONS` blocks; only the charge column is used);
   * Gaussian: `Mulliken charges` and, with `iop(6/80=1)`, `Lowdin Atomic Charges`.
2. **Selection.** For each index of `-i` the element is taken from the Mulliken block (or, if it is missing,
   from the Löwdin block). Then the requested scheme(s) are added: `--type all` (default) gives both columns,
   `mulliken` or `loewdin` gives one. An index that is not in the file gives an empty cell. A file that cannot be
   loaded is skipped with a warning.
3. **Ensemble average** (`-c checkpoint.json`). The Boltzmann weights $w_k$ of the conformers are read from
   the checkpoint (see [Conformer ensembles](../user-guide/ensemble.md)). For atom $i$:

$$
\langle q_i \rangle = \frac{\sum_k w_k\, q_i^{(k)}}{\sum_k w_k}
$$

   The sum runs only over the files that contain that atom and scheme, so the weights are renormalised when
   a file has no charge block. The per-file table is then printed only with `--individual`.

What the two schemes are, in short: with $\mathbf{P}$ the density matrix, $\mathbf{S}$ the overlap matrix and
$Z_A$ the nuclear charge,
Mulliken uses $q_A = Z_A - \sum_{\mu \in A} (\mathbf{PS})_{\mu\mu}$, and Löwdin uses
$q_A = Z_A - \sum_{\mu \in A} (\mathbf{S}^{1/2}\mathbf{P}\mathbf{S}^{1/2})_{\mu\mu}$. Both are computed by the
program, not by frogchem. Both depend on the basis set; NPA charges from `frog-nbo --npa` depend on it much less.
