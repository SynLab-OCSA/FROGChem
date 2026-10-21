For every file the command needs two numbers, the energies of the HOMO and of the LUMO. Everything else
follows from them through Koopmans' theorem ($\mathrm{IP} \approx -\varepsilon_\mathrm{HOMO}$,
$\mathrm{EA} \approx -\varepsilon_\mathrm{LUMO}$). No new calculation and no finite-difference ($N\pm1$) energies are used.

1. **Reading the orbital energies** (Hartree), in this order:
   * the HOMO/LUMO energies already stored by the parser, if any;
   * ORCA: the **last** `ORBITAL ENERGIES` block. The command reads only the table rows (`NO OCC E(Eh) E(eV)`),
     and for an open-shell job both the `SPIN UP` and `SPIN DOWN` tables. An orbital counts as occupied when its
     occupation is above 0.001;
   * Gaussian: the last `Alpha occ. eigenvalues` line and the `Alpha virt. eigenvalues` line after it. When
     `Beta` lines follow (UHF/UKS), they are read too.
2. **Open shell.** HOMO = the highest occupied spin-orbital of either spin, LUMO = the lowest empty
   spin-orbital of either spin. For a doublet the LUMO is often a β orbital.
3. **Indices.** The energies are converted to eV (1 Eh = 27.211386 eV), then

$$
\mu = \frac{\varepsilon_\mathrm{HOMO} + \varepsilon_\mathrm{LUMO}}{2}
\qquad
\eta = \varepsilon_\mathrm{LUMO} - \varepsilon_\mathrm{HOMO}
\qquad
\omega = \frac{\mu^2}{2\eta}
\qquad
S = \frac{1}{\eta}
$$

   The hardness has **no factor ½** ($\eta = \mathrm{IP} - \mathrm{EA}$, the convention of Parr, Szentpály
   and Liu for ω), so η is the same number as the gap. With $\eta = 0$, ω and S are `nan`.
4. **Missing data.** A file that cannot be loaded is skipped with a warning. A file without orbital energies
   gives a row of `nan`. If no file gives HOMO/LUMO energies, the command exits with an error after printing
   the table.

The command has no options of its own; the shared `--format`/`--table-file` options export the table.
For where the orbital energies come from, see [Parsing outputs](../user-guide/parsing.md).
