The command compares the *spectra* of the interatomic distance matrices, not the aligned Cartesian
coordinates.

1. **Reference.** With `-r FILE`, the first structure of that file is the reference (a warning is
   printed if it contains more). Without `-r`, the first structure of the first input is used.
2. **Distance matrix.** For each structure, the $N\times N$ Euclidean distance matrix
   $D_{ij}=|\mathbf r_i-\mathbf r_j|$ is built (`frogchem.math_geometry.calc_distance_matrix`); with
   `--no-hydrogen` the H atoms are removed first, so $N$ is the number of heavy atoms.
3. **Eigenvalues.** $D$ is real and symmetric; its eigenvalues $\lambda_1\le\dots\le\lambda_N$ are
   computed with `numpy.linalg.eigvalsh` and are already sorted.
4. **λ-RMSD** (`calc_rmsd`):

   $$
   \mathrm{RMSD}_\lambda=\sqrt{\frac{1}{N}\sum_{k=1}^{N}\big(\lambda_k-\lambda_k^\mathrm{ref}\big)^2}\quad(\text{Å}).
   $$

   The structures must have the same $N$ as the reference, otherwise the command stops.

Because the spectrum does not change under translation, rotation, reflection or renumbering of the
atoms, no alignment or atom mapping is done. For the same reasons, the value does not see the element
of each atom, and a structure and its mirror image give exactly 0. A value of 0 means the same set of
distances; it is not a proof that two structures are identical. Different structures can, in
principle, have the same spectrum.

With `-c checkpoint.json` the Boltzmann weights are taken from the checkpoint and an *Ensemble Weights*
table and a weighted average row are added (see [Ensembles](../user-guide/ensemble.md)).
