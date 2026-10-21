#!/usr/bin/env python3
"""Recalculate mRRHO thermodynamics with Grimme's approximation.

This script parses ORCA output files and recalculates thermodynamic properties
(G, H, S) using the modified Rigid-Rotor Harmonic-Oscillator (mRRHO) approximation.
It can also compute Boltzmann populations.

If a file has no frequency data, only its electronic energy is shown (thermo
columns become N/A).

SP correction (``-sp SP_DIR``):
    For each frequency file ``conformer/opt/opt.out``, the script looks for the
    single-point calculation inside ``conformer/SP_DIR/`` (one directory up from
    the file's parent, then down into SP_DIR).  It expects exactly one ``.out``
    or ``.log`` file in that directory.  When found:

    * the electronic energy column (E) uses the SP energy
    * the Gibbs and enthalpy values become::

        G = E_SP + (G_freq - E_freq)
        H = E_SP + (H_freq - E_freq)

    * the G-E column still shows only the thermal correction (G_freq - E_freq)

    If the SP file is missing or cannot be parsed, a warning is printed and the
    original calculation energy is used instead.
"""

import glob
import os

import numpy as np

from frogchem.filetypes import load_calculation
from frogchem.thermo import free_gibbs_energy, calculate_boltzmann
from frogchem.printer import print_error, print_warning, print_info

from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS

HARTREE_TO_KCAL = 627.509474


class CalcRrho(CLICommand, name='frog-rrho'):
    """Recompute mRRHO thermochemistry (G, H, S) of ORCA/Gaussian frequency calculations.

    Uses the quasi-RRHO treatment of low frequencies (Grimme or Truhlar) and can compute
    Boltzmann populations (-b) or average over an ensemble checkpoint (-c). Files without
    frequencies only report the electronic energy (thermal columns are empty).

    Single-point correction (--sp-path SP_DIR): for each file conformer/opt/opt.out the
    single point is the only .out/.log file in conformer/SP_DIR/. E then is the SP energy,
    G = E_SP + (G_freq - E_freq), H = E_SP + (H_freq - E_freq), and G-E keeps the thermal
    correction. A missing or unreadable SP file gives a warning and the original energy.
    """
    category = 'thermo'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Thermodynamics', 'One row per file: E, E+ZPVE, H, TS, G, G-E (Eh), imaginary '
                                          'frequencies (cm⁻¹) and Boltz % with -b'),
        Output('table', 'Boltzmann Weighted Averages', 'Population-weighted E, E+ZPVE, H, G (Eh)', when='-b'),
        ENSEMBLE_WEIGHTS,
        Output('table', 'Ensemble-Averaged Thermodynamics (Boltzmann Weighted)',
               'Weighted E, E+ZPVE, H, TS, G, G-E (Eh) and kcal/mol relative to the lowest conformer',
               when='-c'),
    ]
    guide = Guide(
        why="""
            Free energies (G) decide which conformer, isomer or pathway wins, but the G printed by the
            QM program depends on how it treats **low-frequency vibrations** (soft modes below
            ~100 cm⁻¹, e.g. methyl or aryl rotations), whose harmonic entropy is unrealistically large.
            This command recomputes H, S and G from the frequencies with a *quasi-RRHO* model
            (low modes are blended into free rotors, Grimme 2012, or raised to the cut-off, Truhlar 2011),
            so that ORCA and Gaussian results, different temperatures and different pressures are
            treated the same way. Use it to get G at another temperature, to combine an accurate
            single-point energy with the thermal correction of a cheaper frequency job, and to get
            Boltzmann populations of conformers.
            """,
        needs="""
            * One output per structure from a **frequency calculation** at a true minimum (or TS):
              ORCA `! Opt Freq` (or `! Freq`), Gaussian `opt freq`. The rotational constants, the
              multiplicity and the frequencies are read from it.
            * Optional, for `--sp-path`: a single-point calculation of the same geometry in a sibling
              folder, e.g. `conf1/opt/opt.out` (frequencies) and `conf1/sp/sp.out` (single point,
              the only `.out`/`.log` file in that folder).
            * Optional, for `-c`: a checkpoint file with populations (see the Ensemble user guide).

            Files without frequencies are accepted: only their electronic energy is shown.
            """,
        how=[
            "Run `frog-rrho freq.out` (several files can be given at once, e.g. `frog-rrho conf_*/opt.out`).",
            "Choose the conditions if they are not 298.15 K and 1 atm: `-T 195` (K), `-P 101.325` (kPa). "
            "For a molecule with rotational symmetry give the symmetry number σ with `-s` "
            "(e.g. `-s 2` for water, `-s 12` for benzene); the default is 1.",
            "To compare conformers add `-b` (Boltzmann populations from G) and `--sort` (lowest G first).",
            "To use a better electronic energy, put each single point in a folder next to the frequency "
            "folder and add `--sp-path sp`: E becomes the single-point energy and G = E_SP + (G − E)_freq.",
            "To switch to Truhlar's model use `-m truhlar`; `--cutoff` (default 100 cm⁻¹) and `-a` "
            "(default 4) set where and how sharply the low modes are corrected.",
        ],
        expect="""
            A *Thermodynamics* table with one row per file, all energies in hartree (Eh;
            1 Eh = 627.5 kcal/mol): **E**, **E+ZPVE** (with zero-point energy), **H**, **TS** (temperature ×
            entropy), **G**, **G−E** (the thermal correction to add to any electronic energy) and the
            **imaginary frequencies** (negative numbers, cm⁻¹). With `-b` a *Boltz %* column and a table of
            population-weighted averages follow.

            * Imaginary frequencies are left out of the thermochemistry. A minimum must show `-`; a
              transition state exactly one value (e.g. −356.93 cm⁻¹ for `orb.out`). An extra small one
              (> −50 cm⁻¹) usually means a loose optimisation: reoptimise rather than ignore it.
            * G−E is typically 0.1–0.7 Eh for organic molecules (60–420 kcal/mol): it grows with size.
            * With the default settings the numbers reproduce ORCA's own *Final Gibbs free energy* for a
              C1 molecule (example `test.out`: G = −800.642826 Eh, G−E = 0.111330 Eh in both). For a
              molecule with symmetry ORCA uses its detected σ, FROG the `-s` you give: G differs by
              RT ln σ (0.41 kcal/mol for σ = 2 at 298 K).
            * Lowering T increases G (here −800.639195 Eh at 273.15 K); `-m truhlar` changes G by a few
              tenths of a kcal/mol (−800.643094 Eh).
            * Linear molecules are recognised automatically (fewer than three non-zero rotational
              constants); `--linear` forces this treatment.
            * Warning `frequencies given but no rotational constants (B)`: the rotational entropy is set to
              zero and G is wrong by several kcal/mol (≈ 9 kcal/mol for `test.out`). Check that the output
              is complete and comes from a frequency job.
            * Boltzmann populations make sense only for conformers of the **same molecule** computed at the
              same level. Populations below 1 % rarely matter; a population of 100 % with all others 0.00 %
              usually means the energies differ by > 3 kcal/mol, or that unrelated files were mixed.
            """,
    )
    examples = [
        ('frog-rrho freq.out', 'mRRHO thermochemistry at 298.15 K'),
        ('frog-rrho conf_*/opt/opt.out --sp-path sp -b --sort -T 273.15',
         'SP-corrected G, Boltzmann populations, sorted by G, at 273.15 K'),
        ('frog-rrho conf_*/opt.out -c checkpoint.json', 'Ensemble-averaged thermochemistry'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help="ORCA/Gaussian output files")
        parser.add_argument('-b', '--boltz', action='store_true',
                            help='Compute Boltzmann populations from G')
        parser.add_argument('--sort', action='store_true', help='Sort the files by G')
        parser.add_argument('--sp-path', default=None, metavar='SP_DIR',
                            help='Directory name with the single-point calculation of each conformer '
                                 '(see the description)')

        parameters = parser.add_argument_group('calculation parameters')
        parameters.add_argument('--linear', action='store_true', help="Force the linear-rotor treatment (linear molecules are detected automatically "
                                     "from the rotational constants)")
        parameters.add_argument('-m', '--model', choices=['grimme', 'truhlar'], default='grimme',
                                help="Quasi-RRHO model for the vibrational entropy")
        parameters.add_argument('-s', '--sigma', type=int, default=1, help="Rotational symmetry number")
        parameters.add_argument('-a', '--alpha', default=4, type=float, help="Damping exponent of Grimme's model")
        parameters.add_argument('--cutoff', dest='cut_off', default=100, type=float,
                                help="Cut-off frequency of the quasi-RRHO interpolation in cm⁻¹")
        parameters.add_argument('-P', '--pressure', default=101.325, type=float, help="Pressure in kPa")

    def run(self, args):
        self.require_files(args.file)
        data_rows = []

        for file in args.file:
            try:
                calc = load_calculation(file)
            except Exception as e:
                self._parse_errors.append({'file': file, 'error_type': type(e).__name__, 'message': str(e)})
                print_warning(f"Skipping {file}: {e}")
                continue

            e_base = calc.final_energy
            e_display = e_base
            has_sp = False

            if args.sp_path:
                parent = os.path.dirname(file)
                conformer_dir = os.path.dirname(parent)
                sp_dir = os.path.join(conformer_dir, args.sp_path)
                sp_files = glob.glob(os.path.join(sp_dir, "*.out")) + glob.glob(os.path.join(sp_dir, "*.log"))
                if len(sp_files) == 1:
                    try:
                        sp_calc = load_calculation(sp_files[0])
                        e_display = sp_calc.final_energy
                        has_sp = True
                    except Exception as e:
                        print_warning(f"Failed to parse SP file {sp_files[0]}: {e}")
                elif len(sp_files) == 0:
                    print_warning(f"No .out or .log found in {sp_dir} (--sp-path {args.sp_path}) for {file}")
                else:
                    print_warning(f"Multiple .out/.log files in {sp_dir} (--sp-path {args.sp_path}), expected exactly one")

            has_freq = hasattr(calc, 'freq') and isinstance(calc.freq, np.ndarray) and calc.freq.size > 0

            if has_freq:
                real_freqs = calc.freq[calc.freq > 0]

                g, zpve, h, s = free_gibbs_energy(
                    SCF=calc.final_energy,
                    T=args.temp,
                    freq=real_freqs,
                    mw=calc.mw,
                    B=calc.B,
                    m=int(calc.mult),
                    linear=args.linear,
                    cut_off=args.cut_off,
                    alpha=args.alpha,
                    P=args.pressure,
                    symno=args.sigma,
                    model=args.model,
                )

                if has_sp:
                    g = e_display + (g - e_base)
                    h = e_display + (h - e_base)

                imag_freqs = calc.freq[calc.freq < 0]
                imag_str = ", ".join([f'{i:.2f}' for i in imag_freqs]) if imag_freqs.size > 0 else "-"

            else:
                g, zpve, h, s = np.nan, np.nan, np.nan, np.nan
                imag_str = "N/A"

            data_rows.append([
                file,
                e_display,
                e_display + zpve,
                h,
                args.temp * s,
                g,
                g - e_display,
                imag_str
            ])

        if not data_rows:
            print_error("No valid data found.", exit_code=1)

        headers = [
            'File', 'E [Eh]', 'E+ZPVE [Eh]', 'H [Eh]',
            'TS [Eh]', 'G [Eh]', 'G-E [Eh]', 'Imag Freq (cm⁻¹)'
        ]
        avgs = None
        if args.boltz:
            g_values = np.array([row[5] for row in data_rows], dtype=float)
            ok = ~np.isnan(g_values)
            if not ok.all():
                print_warning("Files without frequencies are left out of the Boltzmann populations")
            pops = np.full(len(data_rows), np.nan)
            if ok.any():
                pops[ok] = calculate_boltzmann(g_values[ok], args.temp)
            for row, pop in zip(data_rows, pops):
                row.append(pop * 100)
            headers.append('Boltz %')
            sel = [r for r, keep in zip(data_rows, ok) if keep]
            cols = [np.array([r[i] for r in sel], dtype=float) for i in (1, 2, 3, 5)]
            avgs = [[float(np.sum(c * pops[ok])) for c in cols]] if sel else None

        if args.sort:
            data_rows.sort(key=lambda x: (np.isnan(x[5]), x[5]))

        if args.checkpoint:
            self._run_ensemble_average(data_rows)
            if not args.individual:
                return

        self.output(data_rows, headers=headers, title="Thermodynamics",
                    floatfmt=['s'] + ['.6f'] * 6 + ['.1f', '.2f'])
        if avgs is not None:
            self.output(avgs, headers=['E_av [Eh]', 'E+ZPVE_av [Eh]', 'H_av [Eh]', 'G_av [Eh]'],
                        title="Boltzmann Weighted Averages", floatfmt=".6f")

    def _run_ensemble_average(self, data_rows):
        print_info(f"Loading Boltzmann populations from checkpoint: {self._checkpoint}")
        ens = self.build_ensemble([str(r[0]) for r in data_rows])
        self.output_weights(ens)
        weights = np.array(ens.weights, dtype=float)

        names = ["E (Electronic Energy)", "E+ZPVE", "H (Enthalpy)", "TS (Entropy)",
                 "G (Gibbs Free Energy)", "G-E (Thermal Corr)"]
        rows = []
        for col, name in enumerate(names, start=1):
            vals = np.array([r[col] for r in data_rows], dtype=float)
            avg = float(np.sum(weights * vals)) if not np.isnan(vals).any() else float('nan')
            if col == 4:          # TS: no meaningful "relative to minimum"
                rel = float('nan')
            elif col == 6:        # thermal correction itself in kcal/mol
                rel = avg * HARTREE_TO_KCAL
            else:
                rel = (avg - float(np.nanmin(vals))) * HARTREE_TO_KCAL
            rows.append([name, avg, rel])

        self.output(rows, headers=["Property", "⟨Value⟩ [Eh]", "⟨Δ⟩ [kcal/mol rel. min]"],
                    title="Ensemble-Averaged Thermodynamics (Boltzmann Weighted)",
                    floatfmt=['s', '.6f', '.2f'])


def parse_args():
    return CalcRrho().build_parser().parse_args()


def main():
    CalcRrho().main()


if __name__ == '__main__':
    main()
