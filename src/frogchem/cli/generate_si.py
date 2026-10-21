"""Supporting Information (SI) Markdown report from quantum chemistry outputs.

Energies, mRRHO thermochemistry, imaginary frequencies and Cartesian coordinates of
each structure, formatted for an SI document.
"""

from frogchem.cli.auto_profile import thermo_energies
from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import ParseError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning


def _fmt_e(val) -> str:
    return "N/A" if val is None or val != val else f"{val:.6f} Eh"


class GenerateSI(CLICommand, name='frog-si'):
    """Write a Supporting Information (SI) Markdown report from ORCA/Gaussian outputs.

    For each file: classification (minimum / TS / saddle point) from the number of imaginary
    frequencies, E, ZPVE, H and G (mRRHO recomputed from the frequencies at -T), the
    imaginary frequencies and the Cartesian coordinates (XYZ block).
    """
    category = 'parsing'
    has_tables = False
    outputs = [
        Output('file', 'SI_report.md', 'Markdown report, one section per input file '
                                       '(current directory, or -o)'),
    ]
    guide = Guide(
        why="""
            Journals ask for the energies and Cartesian coordinates of every computed structure in the
            Supporting Information (SI). Copying them by hand from dozens of outputs is slow and error-prone.
            This command collects, for each output file, the electronic energy, the thermochemistry, the
            imaginary frequencies (which also tell whether the structure is a minimum or a transition state)
            and the final geometry, and writes them into one Markdown file that you can paste into Word or
            convert with pandoc.
            """,
        needs="""
            The final ORCA or Gaussian **output files**, one per structure, ideally of an optimisation with
            frequencies (ORCA `! Opt Freq`, `! OptTS Freq`; Gaussian `opt freq`). The thermal terms (ZPVE, H, G)
            are recomputed by FROG from the frequencies at the temperature `-T` with Grimme's mRRHO model
            (the entropy of modes below ~100 cm⁻¹ is interpolated towards that of a free rotor; 1 atm,
            symmetry number 1), so they need a frequency calculation in the same file;
            without one the report shows `N/A`. The thermochemistry needs the multiplicity, which only the
            ORCA parser reads: for **Gaussian** files E, imaginary frequencies and coordinates are written but
            ZPVE/H/G are `N/A` (with a warning). The file name (without extension) becomes the section title,
            so name the files as you want them to appear (`TS1.out`, `INT2.out`...).
            """,
        how=[
            "Run `frog-si R.out TS1.out P.out`: the report is written to `SI_report.md` in the current folder.",
            "Choose another path with `-o si/structures.md` and another temperature with `-T 273.15` "
            "(kelvin; default 298.15 K).",
            "Open the Markdown file, check the classification of each structure, and convert it if needed "
            "(e.g. `pandoc SI_report.md -o SI.docx`).",
        ],
        expect="""
            A Markdown file with a title and one section per input, e.g. `## TS1 (TS)`, listing **E**, **ZPVE**,
            **H** and **G** in Hartree (Eh), the imaginary frequencies in cm⁻¹ (negative numbers) and an `xyz`
            block with the number of atoms, a comment line and the final coordinates in Å. Nothing is printed
            on screen except progress messages.

            * Classification: no imaginary frequency → *Minimum*, exactly one → *TS*, more → *Saddle Point*.
              A file **without frequencies** is also labelled *Minimum* (with ZPVE/H/G `N/A`): that label then
              means nothing, check it.
            * A minimum with a small imaginary frequency (e.g. −20 cm⁻¹) is usually numerical noise; a TS
              should have one clear imaginary mode (typically −100 to −2000 cm⁻¹; check what moves with
              `frog-hessian`).
            * G and H are FROG's mRRHO values and differ slightly from the numbers printed by ORCA or
              Gaussian (different low-frequency treatment); imaginary modes are left out of the thermochemistry.
              Use one source consistently in a paper.
            * Files that cannot be read are skipped with a warning; if none can be read, no report is written.

            Example (`opt.out`, r²SCAN-3c `! OptTS Freq`): `## opt (TS)`, E −3265.187260 Eh, ZPVE 0.753166 Eh,
            H −3264.382645 Eh, G −3264.509635 Eh at 298.15 K, imaginary frequency −180.94 cm⁻¹, 96 atoms.
            """,
    )
    examples = [
        ('frog-si R.out TS.out P.out', 'Write SI_report.md'),
        ('frog-si conf_*/opt.out -o si/structures.md -T 273.15', 'Thermochemistry at 0 °C, custom path'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', help="ORCA/Gaussian output files")
        parser.add_argument('-T', '--temp', type=float, default=298.15, help='Temperature in K')
        self.add_output_arg(parser, help='Markdown report path (default: SI_report.md)')

    def run(self, args):
        paths = self.require_files(args.files)
        out_path = self.output_path('SI_report', '.md')
        if any(out_path.resolve() == p.resolve() for p in paths):
            raise ParseError(f"Refusing to overwrite the input file {out_path}")

        sections = []
        for p in paths:
            print_info(f"Parsing {p.name}...")
            try:
                calc = load_calculation(str(p))
            except Exception as e:
                print_warning(f"Skipping {p.name}: {e}")
                continue
            sections.append(self._section(p.stem, calc, args.temp))

        if not sections:
            raise ParseError("No input file could be parsed; report not written")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# Computational Details & Cartesian Coordinates\n\n" + ''.join(sections))
        self.saved(out_path, 'report')

    @staticmethod
    def _section(name: str, calc, temp: float) -> str:
        terms = thermo_energies(calc, temp)
        imag = terms['imag']
        if not imag:
            title, freq_str = f"## {name} (Minimum)", "None"
        else:
            kind = "TS" if len(imag) == 1 else "Saddle Point"
            title, freq_str = f"## {name} ({kind})", ", ".join(f"{f:.2f} cm⁻¹" for f in imag)

        lines = [
            f"{title}\n",
            f"- **Electronic Energy (E)**: {_fmt_e(terms['E'])}",
            f"- **Zero-Point Vibrational Energy (ZPVE)**: {_fmt_e(terms['ZPVE'])}",
            f"- **Enthalpy (H, {temp:.2f} K)**: {_fmt_e(terms['H'])}",
            f"- **Gibbs Free Energy (G, {temp:.2f} K)**: {_fmt_e(terms['G'])}",
            f"- **Imaginary Frequencies**: {freq_str}\n",
        ]
        atoms = getattr(calc, 'atoms', None)
        geom = getattr(calc, 'geom', None)
        if atoms is not None and geom is not None and len(atoms):
            lines += ["### Cartesian Coordinates (Å)", "```xyz", str(len(atoms)), f"{name} optimized geometry"]
            lines += [f"{sym:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}" for sym, (x, y, z) in zip(atoms, geom)]
            lines.append("```")
        return '\n'.join(lines) + "\n\n---\n\n"


def parse_args():
    return GenerateSI().build_parser().parse_args()


def main():
    GenerateSI().main()


if __name__ == '__main__':
    main()
