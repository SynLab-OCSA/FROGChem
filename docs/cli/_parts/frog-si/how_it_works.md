For each input file, in the order given, the command builds one Markdown section. It then writes all
sections into a single file.

1. **Energy.** $E$ is the final single-point energy stored by the parser. If there is none, the last SCF energy
   is used.
2. **Thermochemistry.** If the file has frequencies, ZPVE, H and G are **recomputed** with
   `frogchem.thermo.free_gibbs_energy`, the same mRRHO model as `frog-rrho` with its defaults: Grimme's
   quasi-RRHO entropy with a 100 cm⁻¹ cut-off, 1 atm, symmetry number 1. The inputs are the real frequencies, the
   molecular mass, the rotational constants and the multiplicity, at the temperature `-T` (298.15 K).
   Imaginary frequencies are left out. If any input is missing (no frequencies, or Gaussian files, where
   the multiplicity is not read), the terms are written as `N/A` and a warning is printed. The model is
   described in [Thermochemistry theory](../theory/thermo.md).
3. **Classification** by the number of imaginary (negative) frequencies:
   0 → *Minimum*, 1 → *TS*, more → *Saddle Point*. A file without frequencies has zero imaginary modes and
   is therefore labelled *Minimum*.
4. **Geometry.** The last geometry of the file is written as an `xyz` block (atom count, comment
   `<stem> optimized geometry`, coordinates in Å with 6 decimals).
5. **Report.** The section title is the file stem: `## <stem> (<class>)`. All sections go under
   `# Computational Details & Cartesian Coordinates` into `SI_report.md` in the current folder, or into
   the path given with `-o`; missing folders are created. The command refuses to overwrite an input
   file. Unreadable files are skipped. If none can be read, no report is written and the command ends
   with an error.

Energies are in Hartree with 6 decimals and frequencies in cm⁻¹ with 2 decimals. Nothing is printed
except progress messages. The same thermochemistry, with more control (cut-off, model, σ, pressure),
is available from `frog-rrho`.
