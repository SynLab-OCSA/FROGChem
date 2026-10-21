"""
Extract Mulliken and Löwdin atomic charges for selected atoms from ORCA/Gaussian output.
"""

from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS
from frogchem.exceptions import InputError, ParseError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning

_LABELS = {'mulliken': 'Mulliken', 'loewdin': 'Löwdin', 'all': 'Mulliken and Löwdin'}


class GetCharges(CLICommand, name='frog-charges'):
    """Extract Mulliken and Löwdin atomic charges for selected atoms.

    Reads ORCA output files and Gaussian logs (Löwdin with iop(6/80=1)) and prints one row per file with the
    element and the charge of each selected atom. With -c the charges are Boltzmann-averaged over the ensemble.
    """

    category = 'parsing'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Charges (<type>)', 'One row per input file: element and charge (e) of each selected atom'),
        ENSEMBLE_WEIGHTS,
        Output('table', 'Ensemble-Averaged Charges (<type>)',
               'One row per selected atom: Boltzmann-weighted charges (e)', when='-c'),
    ]
    guide = Guide(
        why="""
            Partial atomic charges are a quick way to see where electron density goes: which atom of a
            substrate is the most positive (likely site of nucleophilic attack), how much charge a ligand
            donates to a metal, or how a substituent changes the charge on a reactive centre along a
            series. This command pulls the **Mulliken** and **Löwdin** charges (the two population schemes
            ORCA prints by default; Gaussian prints Löwdin on request) for the atoms you ask for, from one or many files, into a single
            table. With a conformer ensemble it can also give Boltzmann-averaged charges.
            """,
        needs="""
            One or more output files with the population analysis:

            * **ORCA**: the `MULLIKEN ATOMIC CHARGES` and `LOEWDIN ATOMIC CHARGES` blocks are printed by
              default after every SCF, unless the input has `! NoPop` / `! MiniPrint`.
            * **Gaussian 16**: Mulliken charges are printed by default; Löwdin charges need `iop(6/80=1)`
              (e.g. `#p ... pop=full iop(3/33=1) iop(6/80=1) gfinput`, the route that also serves
              `frog-mo` and `frog-density`). Without it the Löwdin column is empty.

            The **last** block of the file is used (final geometry of an optimisation). Look up the atom
            numbers in a viewer: they are **0-based** (first atom = 0).
            For ensemble averaging you also need a `checkpoint.json` (or csv/json) with Boltzmann populations,
            e.g. from a conformer-search workflow (`-c`).
            """,
        how=[
            "Run `frog-charges opt.out -i 0 1 2` to get the element and both charges of atoms 0, 1 and 2.",
            "Compare several molecules or conformers at once: `frog-charges conf_*.out -i 5`; keep only one "
            "scheme with `--type mulliken` or `--type loewdin`.",
            "For a conformer ensemble add `-c checkpoint.json`: the charges are averaged with the Boltzmann "
            "weights (add `--individual` to also see the per-file table).",
            "Export with `--format csv --table-file charges.csv` if you want to plot the numbers.",
        ],
        expect="""
            A table with one row per file and, for each selected atom, the element (`#i El`) and the charge in
            units of the elementary charge e (`#i Mulliken`, `#i Löwdin`). With `-c`, first an *Ensemble
            Weights* table, then one row per atom with the weighted ⟨Mulliken⟩ / ⟨Löwdin⟩ charges.

            * Mulliken and Löwdin charges of the same atom can differ a lot, even in **sign**
              (e.g. `test.out`, an anion at r²SCAN-3c: atom 0 (C) Mulliken −0.373, Löwdin −0.196;
              atom 1 (C) Mulliken **+0.307**, Löwdin **−0.050**). Neither is "the" charge: both depend on the basis set
              (Mulliken strongly, and badly with diffuse functions). Use them for **trends within one
              method and basis**, never as absolute numbers; for more robust charges use NPA
              (`frog-nbo --npa`).
            * The charges of all atoms add up to the total molecular charge.
            * An empty cell (`-`) means the file has no charge block (e.g. `! NoPop`) or the index does
              not exist in that molecule; check the 0-based numbering.
            """,
    )
    examples = [
        ('frog-charges opt.out -i 0 1 2', 'Mulliken and Löwdin charges of atoms 0, 1 and 2'),
        ('frog-charges conf_*/protocol_1/orca.out -i 0 -c checkpoint.json',
         'Boltzmann-averaged charges of atom 0 over a conformer ensemble'),
        ('frog-charges opt.out -i 3 --type loewdin --format csv', 'Löwdin charge of atom 3 as CSV'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', help='ORCA output files or Gaussian logs')
        parser.add_argument('-i', '--indices', dest='indices', nargs='+', type=int, required=True,
                            help='Atom indices (0-based)')
        parser.add_argument('--type', choices=['mulliken', 'loewdin', 'all'], default='all',
                            help='Charge scheme to extract')

    def run(self, args):
        self.require_files(args.files)
        want_m = args.type in ('mulliken', 'all')
        want_l = args.type in ('loewdin', 'all')

        headers = ['File']
        for idx in args.indices:
            headers.append(f'#{idx} El')
            if want_m:
                headers.append(f'#{idx} Mulliken')
            if want_l:
                headers.append(f'#{idx} Löwdin')

        data = []
        calcs_data = []
        for fname in args.files:
            print_info(f'Processing: {fname}')
            try:
                calc = load_calculation(fname)
            except Exception as e:
                print_warning(f'Could not load {fname}: {e}')
                continue

            calcs_data.append({'fname': fname, 'calc': calc})
            mulliken = calc.mulliken or {}
            loewdin = calc.loewdin or {}

            row = [fname]
            for idx in args.indices:
                m = mulliken.get(idx)
                lw = loewdin.get(idx)
                row.append((m or lw or {}).get('atom'))
                if want_m:
                    row.append(m['charge'] if m else None)
                if want_l:
                    row.append(lw['charge'] if lw else None)
            data.append(row)

        if not data:
            raise ParseError('No charges could be extracted from the input files')

        if args.checkpoint:
            self._run_ensemble_average(calcs_data, args)
            if not args.individual:
                return

        self.output(data, headers, title=f'Charges ({_LABELS[args.type]})')

    def _run_ensemble_average(self, calcs_data, args):
        print_info(f'Loading Boltzmann populations from checkpoint: {args.checkpoint}')
        try:
            ens = self.build_ensemble(calcs_data, path_extractor=lambda x: x['fname'])
        except Exception as e:
            raise InputError(f'Failed to load checkpoint {args.checkpoint}: {e}') from e

        self.output_weights(ens)
        weights = ens.weights

        avg_headers = ['Atom', 'Idx', 'Element']
        if args.type in ('mulliken', 'all'):
            avg_headers.append('⟨Mulliken⟩')
        if args.type in ('loewdin', 'all'):
            avg_headers.append('⟨Löwdin⟩')

        avg_rows = []
        for idx in args.indices:
            atom_label = None
            w_m_sum, w_m_val = 0.0, 0.0
            w_l_sum, w_l_val = 0.0, 0.0
            for item, w in zip(calcs_data, weights):
                calc = item['calc']
                if calc.mulliken and idx in calc.mulliken:
                    atom_label = calc.mulliken[idx]['atom']
                    w_m_val += w * calc.mulliken[idx]['charge']
                    w_m_sum += w
                if calc.loewdin and idx in calc.loewdin:
                    atom_label = atom_label or calc.loewdin[idx]['atom']
                    w_l_val += w * calc.loewdin[idx]['charge']
                    w_l_sum += w

            row = [f'{atom_label or "X"}{idx}', idx, atom_label]
            if args.type in ('mulliken', 'all'):
                row.append(w_m_val / w_m_sum if w_m_sum > 0 else None)
            if args.type in ('loewdin', 'all'):
                row.append(w_l_val / w_l_sum if w_l_sum > 0 else None)
            avg_rows.append(row)

        self.output(avg_rows, avg_headers,
                    title=f'Ensemble-Averaged Charges ({_LABELS[args.type]}, Boltzmann weighted)')


def parse_args():
    return GetCharges().build_parser().parse_args()


def main():
    GetCharges().main()


if __name__ == '__main__':
    main()
