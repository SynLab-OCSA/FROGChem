"""Conceptual DFT (CDFT) global reactivity indices from frontier orbital energies (Koopmans)."""

import re

from frogchem.cli.base import CLICommand, Guide, Output
from frogchem.exceptions import ParseError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning

HARTREE_TO_EV = 27.211386245988


def _get_homo_lumo_energies(calc) -> tuple[float | None, float | None]:
    """Fallback parser for HOMO and LUMO energies in Hartree."""
    # If the parser already has them:
    if hasattr(calc, 'homo_energy') and hasattr(calc, 'lumo_energy'):
        if calc.homo_energy is not None and calc.lumo_energy is not None:
            return float(calc.homo_energy), float(calc.lumo_energy)
            
    # ORCA "ORBITAL ENERGIES" block (last one). Read the contiguous orbital lines after each
    # "NO OCC E(Eh) E(eV)" header only, so that the MO-coefficient blocks printed afterwards are
    # not picked up. Open shell: both the SPIN UP and SPIN DOWN tables are read and the HOMO is
    # the highest occupied, the LUMO the lowest unoccupied spin-orbital of either spin.
    if hasattr(calc, 'fl'):
        pos = calc.fl.rfind('ORBITAL ENERGIES')
        if pos != -1:
            row = re.compile(r'^\s*\d+\s+([0-9.]+)\s+([-+]?[0-9.]+)\s+[-+]?[0-9.]+\s*$')
            blocks, current = [], []
            for line in calc.fl[pos:].splitlines():
                m = row.match(line)
                if m:
                    current.append((float(m.group(1)), float(m.group(2))))
                    continue
                if current:                      # a table just ended
                    blocks.append(current)
                    current = []
                    if len(blocks) == 2:         # alpha + beta read
                        break
                    continue
                if blocks and line.strip():
                    # between the two tables only the SPIN DOWN title and the column header may appear
                    if 'SPIN DOWN' in line or line.split()[:2] == ['NO', 'OCC']:
                        continue
                    break
            if current:
                blocks.append(current)
            orbitals = [o for b in blocks for o in b]
            if orbitals:
                occupied = [e for occ, e in orbitals if occ > 0.001]
                unoccupied = [e for occ, e in orbitals if occ <= 0.001]
                return (max(occupied) if occupied else None), (min(unoccupied) if unoccupied else None)

    # Gaussian "Alpha  occ. eigenvalues --" / "Alpha virt. eigenvalues --" (last population analysis);
    # unrestricted runs also print "Beta ..." lines: HOMO = max over spins, LUMO = min over spins.
    if hasattr(calc, 'fl'):
        num = re.compile(r'[-+]?\d+\.\d+')  # F10.5 fields may run together ("-10.12345-10.23456")
        occ = list(re.finditer(r'Alpha\s+occ\.\s+eigenvalues\s+--\s+(.+)', calc.fl))
        if occ:
            virt = re.compile(r'Alpha\s+virt\.\s+eigenvalues\s+--\s+(.+)').search(calc.fl, occ[-1].end())
            if virt:
                try:
                    homo = float(num.findall(occ[-1].group(1))[-1])
                    lumo = float(num.findall(virt.group(1))[0])
                except (ValueError, IndexError):
                    homo = lumo = None
                if homo is not None:
                    b_virt = re.compile(r'Beta\s+virt\.\s+eigenvalues\s+--\s+(.+)').search(calc.fl, virt.end())
                    nxt_alpha = calc.fl.find('Alpha  occ.', virt.end())
                    if b_virt and (nxt_alpha == -1 or b_virt.start() < nxt_alpha):
                        b_occ = [m for m in re.finditer(r'Beta\s+occ\.\s+eigenvalues\s+--\s+(.+)',
                                                        calc.fl[:b_virt.start()]) if m.start() > virt.end()]
                        if b_occ:
                            homo = max(homo, float(num.findall(b_occ[-1].group(1))[-1]))
                        lumo = min(lumo, float(num.findall(b_virt.group(1))[0]))
                    return homo, lumo

    return None, None


class CalcCDFT(CLICommand, name='frog-cdft'):
    """Compute Conceptual DFT global indices (μ, η, ω, S) from HOMO/LUMO energies.

    Koopmans' theorem approximation (IP = -E_HOMO, EA = -E_LUMO):
    chemical potential μ = (E_HOMO + E_LUMO) / 2, hardness η = E_LUMO - E_HOMO,
    electrophilicity ω = μ² / (2η), softness S = 1 / η (η = IP − EA convention, no 1/2).
    Open-shell: HOMO = highest occupied and LUMO = lowest unoccupied orbital over both spins.
    """
    category = 'parsing'
    outputs = [
        Output('table', 'Conceptual DFT (CDFT) Global Indices',
               'One row per file: HOMO, LUMO, gap, μ, η, ω (eV) and S (1/eV)'),
    ]
    guide = Guide(
        why="""
            Conceptual DFT turns two numbers you already have, the energies of the HOMO and the LUMO
            (highest occupied and lowest unoccupied molecular orbital), into simple reactivity descriptors:
            how easily the molecule gives or takes electrons (chemical potential μ), how much it resists
            changing its electron count (hardness η) and how strongly it behaves as an electrophile
            (electrophilicity ω). Use it to **rank a series** of related molecules (substituents,
            catalysts, substrates) computed with the same method, e.g. "which Michael acceptor is the most
            electrophilic". It costs nothing extra: no new calculation is needed.
            """,
        needs="""
            One or more ORCA or Gaussian **output files** of a converged SCF calculation (a single point or
            an optimisation; any job that prints the orbital energies). ORCA prints the `ORBITAL ENERGIES`
            block by default; Gaussian prints the `Alpha occ./virt. eigenvalues` lines in its population
            analysis (default unless `pop=none`). The **last** orbital-energy block of the file is used,
            i.e. the final geometry of an optimisation. Open-shell (UKS/UHF) outputs are fine: both spins are read.
            To compare molecules, every file must come from the **same functional and basis set**.
            """,
        how=[
            "Run `frog-cdft mol.out` for one molecule, or `frog-cdft *.out` to get one row per file "
            "in the same table.",
            "Save the table for a spreadsheet with `--format csv --table-file cdft.csv` (or `.json`).",
            "Compare the columns within the series: higher ω = better electrophile, higher (less negative) "
            "μ = better electron donor, larger η (= gap) = harder, less reactive molecule.",
        ],
        expect="""
            A table with one row per file: **HOMO**, **LUMO** and **Gap** in eV, **μ**, **η** and **ω** in eV
            and the softness **S** in 1/eV. Definitions (Koopmans approximation, IP ≈ −E_HOMO, EA ≈ −E_LUMO):
            μ = (E_HOMO + E_LUMO)/2, η = E_LUMO − E_HOMO (so η is the same number as the gap; no factor ½),
            ω = μ²/2η, S = 1/η.

            * Typical values for neutral organic molecules with hybrid functionals: HOMO −5 to −7 eV,
              gap 3–6 eV, ω 1–4 eV. Some authors define η = (IP − EA)/2: their η is half and their ω twice
              ours, so only compare with literature that uses the same convention.
            * Absolute numbers depend strongly on the functional (pure GGA functionals give much smaller gaps
              than hybrids); only **trends within one method** are meaningful.
            * Anions often have a **positive** LUMO energy (an unbound extra electron) and a high HOMO:
              that is a known limit of the orbital picture, not a parsing error. A diffuse basis set
              changes these numbers considerably.
            * Open shell: HOMO is the highest occupied and LUMO the lowest empty spin-orbital over α and β.
            * A file without orbital energies gives a warning and a row of `nan`; if no file has them the
              command ends with an error.

            Example (`opt.out`, r²SCAN-3c): HOMO −5.158, LUMO −3.065, gap = η 2.093, μ −4.111, ω 4.038 eV,
            S 0.478 eV⁻¹; an anion computed with B3LYP/def2-TZVP gives HOMO −3.342 and LUMO **+0.283** eV.
            """,
    )
    examples = [
        ('frog-cdft opt.out', 'CDFT indices of one calculation'),
        ('frog-cdft *.out --format csv --table-file cdft.csv', 'Compare several molecules and save the table'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', help="ORCA/Gaussian output files")

    def run(self, args):
        paths = self.require_files(args.files)
        headers = ["File", "HOMO (eV)", "LUMO (eV)", "Gap (eV)", "μ (eV)", "η (eV)", "ω (eV)", "S (1/eV)"]
        nan = float('nan')
        data = []
        n_ok = 0

        for p in paths:
            print_info(f"Parsing {p.name}...")
            try:
                calc = load_calculation(str(p))
            except Exception as e:
                print_warning(f"Failed to load {p.name}: {e}")
                continue

            homo_eh, lumo_eh = _get_homo_lumo_energies(calc)
            if homo_eh is None or lumo_eh is None:
                print_warning(f"Could not find HOMO/LUMO energies in {p.name}")
                data.append([p.name] + [nan] * 7)
                continue

            homo_ev = homo_eh * HARTREE_TO_EV
            lumo_ev = lumo_eh * HARTREE_TO_EV
            # Hardness η = IP - EA = LUMO - HOMO (no factor 1/2, as in Parr, Szentpály & Liu,
            # JACS 1999, for ω = μ²/2η; Parr & Pearson 1983 used η = (IP - EA)/2).
            # Open shell: HOMO/LUMO are taken over both spins.
            eta = lumo_ev - homo_ev
            mu = (homo_ev + lumo_ev) / 2.0
            s = 1.0 / eta if eta != 0 else nan
            omega = mu ** 2 / (2 * eta) if eta != 0 else nan
            data.append([p.name, homo_ev, lumo_ev, eta, mu, eta, omega, s])
            n_ok += 1

        self.output(data, headers, title="Conceptual DFT (CDFT) Global Indices")
        if n_ok == 0:
            raise ParseError("No HOMO/LUMO energies found in any input file")


def parse_args():
    return CalcCDFT().build_parser().parse_args()


def main():
    CalcCDFT().main()


if __name__ == '__main__':
    main()
