Each CSV is read with pandas. Header names are stripped of spaces and the rows are sorted by the
wavelength column (`--wl`, nm). The intensity column (`--int`) is then turned into a spectral distribution
with `colour-science`:

1. **Physical meaning** (`--spectrum-type`). `emission` uses the intensity as emitted power $S(\lambda)$.
   `absorption` converts the absorbance to transmittance, $T(\lambda) = 10^{-A(\lambda)}$. `transmittance`
   uses $T$ as given.
2. **Sampling.** The spectrum must have a step of 1, 5, 10 or 20 nm. If the step is different, it is
   interpolated to 1 nm between its first and last wavelength, and an info message is printed.
3. **Tristimulus values** with the CIE 1931 2° observer $\bar x, \bar y, \bar z$:

   $$
   X_\mathrm{em} = k\!\int S\,\bar x\,d\lambda
   \qquad
   X_\mathrm{abs} = \frac{\int T\,S_\mathrm{D65}\,\bar x\,d\lambda}{\int S_\mathrm{D65}\,\bar y\,d\lambda}
   $$

   $Y$ and $Z$ are computed in the same way. Emission is integrated without an illuminant, so only the shape
   of the spectrum matters. Absorption and transmittance are viewed under D65, which gives the colour the
   sample appears to have.
4. **Chromaticity.** $x = X/(X+Y+Z)$ and $y = Y/(X+Y+Z)$ go into the *CIE coordinates* table.
5. **Plot.** Each point is drawn on the horseshoe diagram in its own sRGB colour. For emission the colour is
   shown at $Y = 0.5$, for the other types at $XYZ/100$, clipped to the sRGB gamut. Points from
   `--external-coordinates` (columns `Name`, `CIEx`, `CIEy`) are drawn as crosses. `--colors` applies only to
   them.

A file that cannot be converted (missing column, wrong values) is skipped with a warning. The command fails
only when no point at all is left. The colour-science formulas are summarised in
[TD-DFT theory, §3](../theory/tddft.md).
