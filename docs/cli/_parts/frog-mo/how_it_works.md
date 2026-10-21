1. **Parsing.** From the output the parser reads the MO coefficient matrix $C_{\mu i}$ (basis function μ,
   MO $i$), the orbital energies and occupations, and two labels for each basis function: the atom it sits on
   and its type (s, p$_x$ … d$_{xy}$ …, f…, both pure and Gaussian Cartesian 6D/10F functions). For an
   open-shell file with multiplicity ≠ 1, the α or the β set (`--beta`) is used.
2. **C² composition.** Each MO column is normalised and squared:

$$
\%_{\mu i} = 100\,\frac{C_{\mu i}^2}{\sum_\nu C_{\nu i}^2}
$$

   The overlap between basis functions is ignored, so this is not a Mulliken or Löwdin orbital
   population.
3. **Grouping.** For each selected atom $A$ and shell type $t \in \{s, p, d, f\}$, the column `A_t` is
   $\sum_{\mu \in A,\ \mu \in t} \%_{\mu i}$. `Total_t` is the same sum over all atoms. Atoms come from `-i`
   (0-based) and/or `--elements`; with neither, all atoms are shown when there are at most 10, otherwise only
   the totals. `--orbital-type` keeps only the listed shells.
4. **MO window.** The HOMO is the last MO with occupation > 0.001 (α set in an open-shell job, also with
   `--beta`). `-b n` shows HOMO−n … HOMO+n. Without `-b` all MOs are shown, but if there are more than 50
   the table is cut to HOMO−5 … HOMO+5. Energies are converted from Eh to eV.
5. **Heatmap** (`--plot`). One row per basis function that passes the atom and type filters (labels
   `0Ti-dxy`, …), one column per MO of the window, colour = $\%_{\mu i}$. The HOMO column is boxed in cyan
   (first `--colors` value). For the heatmap `--orbital-type` also accepts sub-shell names such as `px` or
   `dxy`. The default name is `<stem>_mo.png` (`<stem>_mo_beta.png` with `--beta`), next to the input or in
   `--out-dir`; `--dpi` sets the resolution.

To look at an orbital in 3D, write its cube with `frog-density` (see
[Density and orbital cubes](../user-guide/density_cubes.md)).
