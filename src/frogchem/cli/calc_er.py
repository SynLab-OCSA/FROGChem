"""Convert between a free-energy difference and an enantiomeric/diastereomeric ratio."""
import argparse

from frogchem.thermo import calc_er, calc_dg

from frogchem.cli.base import CLICommand, Guide, Output

# Global options that also work after the subcommand (e.g. `frog-er er 1.4 --format json`).
_SHARED_DESTS = ('verbose', 'quiet', 'format', 'table_file')


class CalcEr(CLICommand, name='frog-er'):
    """Convert a free-energy difference ΔG into e.r./ee, or a ratio into ΔG.

    `er` computes the Boltzmann ratio of two stereoisomers separated by ΔG;
    `dg` computes ΔG from the measured amounts of the major and minor isomer.
    """
    category = 'thermo'
    outputs = [
        Output('table', 'Enantiomeric Ratio', '[er] ΔG (kcal/mol), T (K), major/minor share and ee (%)'),
        Output('table', 'ΔG from Ratio', '[dg] Major and minor amounts, T (K) and ΔG (kcal/mol)'),
    ]
    guide = Guide(
        why="""
            Stereoselectivity is decided by the free-energy difference **ΔΔG‡** between the competing
            transition states (or between the products, under thermodynamic control). This command turns
            a computed ΔΔG into the ratio and enantiomeric excess (*ee*) you would measure, or an
            experimental ratio into the ΔΔG it implies, so that calculation and experiment can be compared
            on the same scale.
            """,
        needs="""
            Nothing but numbers. For `er`: the free-energy difference in **kcal/mol**, normally the
            difference between the G of the two lowest transition states leading to the two stereoisomers
            (e.g. from `frog-rrho`; 1 Eh = 627.5 kcal/mol). For `dg`: the amounts of the major and the
            minor isomer in any unit (95 and 5, 0.95 and 0.05, or HPLC areas). The reaction temperature
            in K if it is not 298.15 K.
            """,
        how=[
            "Computed → experiment: `frog-er er 1.42` (ΔΔG = 1.42 kcal/mol). Add `-T 195` for a reaction run at −78 °C.",
            "Experiment → computed: `frog-er dg 95 5` gives the ΔΔG corresponding to a 95:5 ratio.",
            "Global options such as `--format json` may be written after the subcommand.",
        ],
        expect="""
            One small table. `er` prints ΔG, T, the **Major (%)** and **Minor (%)** shares and **ee (%)**;
            `dg` prints the two amounts, T and **dG (kcal/mol)**.

            * Real values: `er 1.42` → 91.66 : 8.34, ee 83.31 % at 298.15 K; the same ΔG at 195 K gives
              97.50 : 2.50 (ee 95.00 %). `dg 95 5` → 1.745 kcal/mol.
            * The relation is exponential: about 1.3 kcal/mol for 90:10, 1.7 for 95:5, 2.7 for 99:1 at room
              temperature. DFT errors on ΔΔG of ±0.5–1 kcal/mol are normal, so a predicted 99:1 and a
              measured 95:5 are in reasonable agreement.
            * A negative ΔG in `er` just swaps major and minor.
            * The ratio refers only to the two states compared; with several competing transition states,
              use Boltzmann populations of all of them (`frog-rrho -b`).
            """,
    )
    examples = [
        ('frog-er er 1.42', 'e.r. and ee for ΔG = 1.42 kcal/mol at 298.15 K'),
        ('frog-er er 1.42 -T 195 --format json', 'Same at 195 K, as JSON'),
        ('frog-er dg 95 5', 'ΔG (kcal/mol) corresponding to a 95:5 ratio'),
    ]

    def setup_args(self, parser):
        shared = [a for a in parser._actions if a.dest in _SHARED_DESTS]
        subparsers = parser.add_subparsers(dest='command', required=True, metavar='{er,dg}')

        parser_er = subparsers.add_parser('er', help='Compute e.r. and ee from ΔG',
                                          description='Compute e.r. and ee from ΔG')
        parser_er.add_argument('dg', type=float, help='Free-energy difference ΔG in kcal/mol')

        parser_dg = subparsers.add_parser('dg', help='Compute ΔG from a ratio',
                                          description='Compute ΔG from a ratio')
        parser_dg.add_argument('maj', type=float, help='Amount of the major stereoisomer (any unit)')
        parser_dg.add_argument('minor', type=float, help='Amount of the minor stereoisomer (same unit)')

        for sub in (parser_er, parser_dg):
            sub.formatter_class = parser.formatter_class
            sub.add_argument('-T', '--temp', type=float, default=298.15, help='Temperature in K')
            # Same dest as the global option and no default: only overrides when given.
            for action in shared:
                kwargs = dict(dest=action.dest, default=argparse.SUPPRESS, help=action.help)
                if action.nargs == 0:
                    kwargs['action'] = 'store_true'
                else:
                    kwargs.update(choices=action.choices, metavar=action.metavar)
                sub.add_argument(*action.option_strings, **kwargs)

    def run(self, args):
        if args.command == 'er':
            maj, minor, ee = calc_er(args.dg, args.temp)
            headers = ["dG (kcal/mol)", "T (K)", "Major (%)", "Minor (%)", "ee (%)"]
            data = [[args.dg, args.temp, maj * 100, minor * 100, ee]]
            self.output(data, headers=headers, title="Enantiomeric Ratio", floatfmt=".2f")
        else:
            dg = calc_dg(args.maj, args.minor, args.temp)
            headers = ["Major", "Minor", "T (K)", "dG (kcal/mol)"]
            data = [[args.maj, args.minor, args.temp, dg]]
            self.output(data, headers=headers, title="ΔG from Ratio", floatfmt=".3f")


def parse_args():
    return CalcEr().build_parser().parse_args()


def main():
    CalcEr().main()


if __name__ == '__main__':
    main()
