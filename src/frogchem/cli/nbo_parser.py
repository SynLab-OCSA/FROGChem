"""
Extract Natural Bond Orbital (NBO) data from ORCA/Gaussian output files.
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_error, print_warning
from frogchem.exceptions import InputError, ParseError
from frogchem.io import read_nbo_plot_37
from frogchem.density_builder import build_nbo_cube, generate_visualization_scripts

from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS


def _find_nbo_plot_file(calc_path: Path, explicit_plot_file: Optional[str] = None) -> Optional[Path]:
    """Find the NBO plot (.37) file associated with a calculation."""
    if explicit_plot_file:
        p = Path(explicit_plot_file)
        if p.is_file():
            return p
        raise InputError(f"NBO coefficient file {explicit_plot_file} not found")

    stem = calc_path.stem
    parent = calc_path.parent
    candidates = [
        calc_path.with_suffix('.37'),
        parent / f"{stem}.37",
        parent / f"{stem}.nbo.37",
        parent / "FILE.37",
        parent / "file.37",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    # ponytail: a lone .37 in the folder is taken only if unambiguous (never another job's file)
    matches = list(parent.glob("*.37"))
    if len(matches) == 1:
        print_warning(f"Using {matches[0]} as NBO coefficient file for {calc_path.name}")
        return matches[0]
    if len(matches) > 1:
        print_warning(f"Several .37 files next to {calc_path.name}: pass the right one with --coeff-file")

    return None


def _extract_nbo_coefficients_from_calc(calc) -> Optional[np.ndarray]:
    """Extract NBO coefficient matrix in AO basis directly from calculation output text.
    
    Parses natural hybrid orbital expansions for Lewis orbitals (CR, LP, BD) and
    reconstructs the orthogonal complement for valence antibonding orbitals (BD*).
    """
    fl = getattr(calc, 'fl', None)
    if fl is None:
        fname = getattr(calc, 'fname', None)
        if fname and Path(fname).is_file():
            try:
                fl = Path(fname).read_text(encoding='utf-8', errors='replace')
            except Exception:
                return None
        else:
            return None

    # NBO 7 (ORCA): 'Bond orbital / Coefficients / Hybrids'; NBO 3.1 (Gaussian): 'Bond orbital/ Coefficients/ Hybrids'
    starts = list(re.finditer(r'\(Occupancy\)\s+Bond orbital\s*/\s*Coefficients\s*/\s*Hybrids', fl))
    if not starts:
        return None

    block = fl[starts[-1].start():]
    stop = re.search(r'SECOND ORDER PERTURBATION THEORY', block, re.IGNORECASE)
    if stop:
        block = block[:stop.start()]

    if not hasattr(calc, 'mask_atoms') or not hasattr(calc, 'atoms'):
        return None

    mask_int = calc.mask_atoms.astype(int)
    atom_ao_indices = {a: np.where(mask_int == a)[0] for a in range(len(calc.atoms))}
    n_ao = len(calc.mask_atoms)
    n_nbo = len(calc.nbo.nbo_details) if (hasattr(calc, 'nbo') and getattr(calc.nbo, 'nbo_details', None) is not None) else n_ao
    mat = np.zeros((n_ao, n_nbo), dtype=np.float64)

    lines = block.splitlines()
    cur_nbo = None
    cur_atom = None
    cur_coeff = 1.0
    cur_vals = []
    cur_atom_data = []

    def flush_atom():
        nonlocal cur_vals
        if cur_atom is not None and cur_vals:
            cur_atom_data.append((cur_atom, cur_coeff, list(cur_vals)))
            cur_vals = []

    def flush_nbo():
        nonlocal cur_atom_data
        flush_atom()
        if cur_nbo is not None and cur_atom_data:
            col = cur_nbo - 1
            if 0 <= col < n_nbo:
                for a_idx, c, vals in cur_atom_data:
                    if a_idx in atom_ao_indices:
                        ao_inds = atom_ao_indices[a_idx]
                        n_needed = len(ao_inds)
                        if len(vals) >= n_needed:
                            v = np.array(vals[:n_needed])
                            mat[ao_inds, col] += c * v
            cur_atom_data = []

    for line in lines:
        m_nbo = re.match(r'^\s*(\d+)\.\s+(\((-?\d+\.\d+)\)|[A-Z]+\*?)\s*([A-Z]+\*?)\s*\(\s*(\d+)\)\s*(.*)', line)
        if m_nbo:
            flush_nbo()
            cur_nbo = int(m_nbo.group(1))
            cur_atom = None
            cur_coeff = 1.0
            cur_vals = []
            raw_tail = m_nbo.group(6)
            if '-' not in raw_tail and ' - ' not in raw_tail:
                m_inline = re.search(r'([A-Z][a-z]?)\s+(\d+)', raw_tail)
                if m_inline:
                    cur_atom = int(m_inline.group(2)) - 1
                    cur_coeff = 1.0
            continue

        m_hybrid = re.search(r'\(\s*([0-9.]+)%\)\s+([0-9.]+)\*?\s+([A-Z][a-z]?)\s+(\d+)', line)
        if m_hybrid:
            flush_atom()
            cur_atom = int(m_hybrid.group(4)) - 1
            cur_coeff = float(m_hybrid.group(2))
            cur_vals = []
            continue

        if cur_nbo is not None and cur_atom is not None:
            parts = line.split()
            for p in parts:
                try:
                    cur_vals.append(float(p))
                except ValueError:
                    pass

    flush_nbo()

    # Normalize Lewis NBOs
    for col in range(n_nbo):
        norm = np.linalg.norm(mat[:, col])
        if norm > 1e-4:
            mat[:, col] /= norm

    # Reconstruct BD* antibonding orbitals from their corresponding BD bonds
    if hasattr(calc, 'nbo') and getattr(calc.nbo, 'nbo_details', None) is not None:
        df = calc.nbo.nbo_details
        bd_df = df[df['Type'] == 'BD']
        bd_star_df = df[df['Type'] == 'BD*']
        for _, r_star in bd_star_df.iterrows():
            star_col = int(r_star['Index']) - 1
            if star_col >= n_nbo or np.linalg.norm(mat[:, star_col]) > 1e-4:
                continue
            m = bd_df[bd_df['Label'] == r_star['Label']]
            if not m.empty:
                bd_col = int(m.iloc[0]['Index']) - 1
                atom_indices = m.iloc[0]['Atom_Indices']
                if len(atom_indices) >= 2:
                    a1, a2 = atom_indices[0], atom_indices[1]
                    inds1 = atom_ao_indices.get(a1, [])
                    inds2 = atom_ao_indices.get(a2, [])
                    if len(inds1) > 0 and len(inds2) > 0:
                        v1 = mat[inds1, bd_col]
                        v2 = mat[inds2, bd_col]
                        c1 = np.linalg.norm(v1)
                        c2 = np.linalg.norm(v2)
                        if c1 > 1e-4 and c2 > 1e-4:
                            h1 = v1 / c1
                            h2 = v2 / c2
                            v_star = np.zeros(n_ao)
                            v_star[inds1] = c2 * h1
                            v_star[inds2] = -c1 * h2
                            norm_star = np.linalg.norm(v_star)
                            if norm_star > 1e-4:
                                mat[:, star_col] = v_star / norm_star

    non_zero_count = np.sum(np.linalg.norm(mat, axis=0) > 1e-4)
    if non_zero_count > 0:
        return mat
    return None


def _get_nbo_coefficients(calc, calc_path: Path, explicit_plot_file: Optional[str] = None) -> Tuple[Optional[np.ndarray], str]:
    """Retrieve NBO coefficient matrix in AO basis and its description."""
    # 1. Try NBO plot file (.37)
    plot_file = _find_nbo_plot_file(calc_path, explicit_plot_file)
    if plot_file is not None:
        try:
            n_ao = len(calc.mask_atoms) if hasattr(calc, 'mask_atoms') else None
            mat = read_nbo_plot_37(plot_file, n_ao=n_ao)
            return mat, f"coefficient file {plot_file.name}"
        except Exception as e:
            print_warning(f"Failed to read NBO coefficient file {plot_file}: {e}")

    # 2. Gaussian .fchk written with pop=(nbo,savenbo): its "MOs" are the NBOs. In a .log the
    # orbital_coefficients are the canonical MOs (pop=full), never NBOs.
    if getattr(calc, 'orbital_coefficients', None) is not None and str(calc_path).endswith(('.fchk', '.FCHK')):
        return calc.orbital_coefficients, "checkpoint (savenbo)"

    # 3. Fallback: extract from calculation output text directly
    mat_text = _extract_nbo_coefficients_from_calc(calc)
    if mat_text is not None:
        return mat_text, "calculation output (NBO hybrids & bonds)"

    return None, ""


ORB_MAP = {
    'BD': 'σ/π (Bond)',
    'BD*': 'σ*/π* (AntiBond)',
    'LP': 'n (LonePair)',
    'LP*': 'n* (Vacant)',
    'LV': 'v (Vacant)',
    'RY': 'Rydberg',
    'RY*': 'Rydberg*',
    'CR': 'Core'
}

def translate_type(otype):
    return ORB_MAP.get(otype, otype)


def _format_orbital_rows(df: pd.DataFrame, translate: bool = False) -> List[List[Any]]:
    """Format orbital DataFrame rows for CLI table presentation."""
    table_data = []
    for _, row in df.iterrows():
        comp_str = " | ".join(row['Composition']) if row['Composition'] else "N/A"
        otype = translate_type(row['Type']) if translate else row['Type']
        table_data.append([
            row['Index'],
            otype,
            row['Label'],
            row['Occupancy'],
            row['Energy'],
            comp_str,
        ])
    return table_data


def _involves_atoms(row, atoms) -> bool:
    """True if the donor or the acceptor of an E(2) row involves any of `atoms`."""
    user_atoms = set(atoms)
    return not set(row['Donor_Atoms']).isdisjoint(user_atoms) or not set(row['Acceptor_Atoms']).isdisjoint(user_atoms)


class NboParser(CLICommand, name='frog-nbo'):
    """Extract Natural Bond Orbital (NBO) data from ORCA output (! NBO).

    Prints NPA charges, NBO occupancies and compositions, bonds, lone pairs, Wiberg bond indices
    and second-order E(2) donor-acceptor energies, and builds cube files of selected NBOs.
    NBO numbers (--cube) are 1-based as in the NBO output; atom indices (-i) are 0-based.
    """

    category = 'parsing'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Natural Population Summary (<file>)', 'Total, Lewis and non-Lewis populations',
               when='--summary/--all'),
        Output('table', 'NPA Charge Extremes (<file>)', 'Three most positive and most negative NPA charges (e)',
               when='--summary/--all'),
        Output('table', 'Top 5 E(2) Interactions (<file>)', 'Strongest donor-acceptor interactions (kcal/mol)',
               when='--summary/--all'),
        Output('table', 'NPA Charges (<file>)', 'One row per atom: natural charge, valence and Rydberg population',
               when='--npa'),
        Output('table', 'Natural Bond Orbital Analysis (<file>)',
               'One row per NBO: type, atoms, occupancy, energy (au) and hybrid composition', when='--analysis'),
        Output('table', 'NBO Bonds (<file>)', 'Bond orbitals (BD/BD*) of the selected atoms', when='--bonds'),
        Output('table', 'NBO Lone Pairs (<file>)', 'Lone pairs (LP) of the selected atom', when='--lp'),
        Output('table', 'Wiberg Bond Index Matrix (<file>)', 'Atom x atom Wiberg bond indices', when='--wiberg'),
        Output('table', 'Wiberg Totals (<file>)', 'Total Wiberg valence of each atom', when='--wiberg'),
        Output('table', 'Detailed Interactions (> <t> kcal/mol) (<file>)',
               'Donor-acceptor pairs with E(2) above -t (kcal/mol) and ΔE (au)', when='--e2'),
        Output('table', 'Aggregated Interactions (Sum > <t> kcal/mol) (<file>)',
               'E(2) summed over equivalent donor-acceptor pairs', when='--e2'),
        Output('table', 'NBO Cubes (<file>)', 'One row per cube: NBO, type, label, occupancy, energy and path',
               when='--cube/--cube-e2'),
        ENSEMBLE_WEIGHTS,
        Output('table', 'Ensemble-Averaged NPA Charges / Wiberg / E(2) Interactions',
               'Boltzmann-weighted versions of the --npa, --wiberg and --e2 tables', when='-c'),
        Output('file', '<stem>_nbo_cubes/<stem>_nbo<N>_<label>.cube',
               'NBO N on a grid (E(2) pairs: <stem>_e2p<k>_donor|acceptor_nbo<N>_<label>.cube)',
               when='--cube/--cube-e2'),
        Output('file', '<stem>_nbo_cubes/view_chimerax.cxc, view_pymol.pml',
               'ChimeraX and PyMOL scripts that open all cubes of the input', when='--cube/--cube-e2'),
    ]
    guide = Guide(
        why="""
            NBO (Natural Bond Orbital) analysis rewrites the wavefunction as a Lewis structure: core orbitals
            (CR), lone pairs (LP), bonds (BD) and their empty counterparts (antibonds BD*, Rydberg RY). It
            answers chemists' questions directly: **NPA charges** (natural population analysis, much less
            basis-set dependent than Mulliken), hybridisation of a bond (sp², % s character), **Wiberg bond
            indices** (bond orders) and the **E(2) donor→acceptor energies** that quantify hyperconjugation
            and delocalisation (e.g. n(O) → σ*(C–X), anomeric effects, π conjugation). The NBO output is
            thousands of lines long; this command extracts the parts you need, filtered by atom, and can
            turn selected NBOs into cube files to look at.
            """,
        needs="""
            An **ORCA output with an NBO analysis**: add `! NBO` to the input (ORCA 5/6 with the NBO 7
            program installed and the `NBOEXE` environment variable set). NPA charges, the NBO list and the
            E(2) table are printed by default. **Wiberg bond indices** need an extra keyword:
            `%nbo NBOKEYLIST = "$NBO BNDIDX $END" end`. A **Gaussian log** with `pop=nbo` (Gaussian's
            NBO 3.1) works too: NPA charges, the NBO list with occupancies and energies and the E(2) table
            are checked against an ORCA twin (Wiberg indices from Gaussian are not checked). NBO 3.1 numbers
            the NBOs in a different order than NBO 7 and calls the Rydbergs `RY*`, so compare the two
            programs by label (`LP ( 2) O 1`), not by NBO number; E(2) values involving Rydbergs differ by
            a few tenths of a kcal/mol between the two NBO versions.

            For `--cube`/`--cube-e2` the output must also contain the basis set: ORCA `! PrintBasis
            PrintMOs`; Gaussian `#p ... pop=(full,nbo) iop(3/33=1) iop(6/80=1) gfinput`. The NBO
            coefficients are read from the `.37` file written by the NBO `PLOT` keyword
            (`%nbo NBOKEYLIST = "$NBO PLOT $END" end`; give it with `--coeff-file` if it is not next to the
            output); without it they are rebuilt from the hybrids printed in the output, which covers
            core, lone-pair, bond and antibond NBOs only.

            Numbering: atom indices (`-i`) are **0-based** (first atom = 0), while the labels printed by NBO
            (e.g. `O 17`) and the NBO numbers (`--cube 44`) are **1-based**, as in the NBO output.
            """,
        how=[
            "Get an overview first: `frog-nbo job.out --summary` (Lewis/non-Lewis population, most "
            "positive/negative NPA charges, five strongest E(2) interactions).",
            "Ask for the sections you need: `--npa` (charges), `--analysis` (all NBOs with occupancy and "
            "hybrids), `--e2` (donor→acceptor energies), `--wiberg`, or `--all` for everything.",
            "Focus on a site with `-i` (0-based atoms): `--e2 -i 16 --dtype LP -t 5` keeps only lone-pair "
            "donations involving atom 16 above 5 kcal/mol; `--bonds -i 14 18` shows the bond orbitals between "
            "atoms 14 and 18, `--lp -i 16` the lone pairs of atom 16.",
            "Visualise orbitals: `--cube-e2 3` writes the donor and acceptor NBOs of the 3 strongest E(2) "
            "interactions, `--cube 44 144` specific NBOs, into `<stem>_nbo_cubes/` together with ChimeraX "
            "(`view_chimerax.cxc`) and PyMOL (`view_pymol.pml`) scripts; use `--voxels 60` for a quick test.",
            "For a conformer ensemble add `-c checkpoint.json` to Boltzmann-average NPA charges, Wiberg "
            "indices and E(2) values.",
        ],
        expect="""
            Tables in the terminal, one block per file. Units: charges in e, occupancies in electrons,
            E(2) in kcal/mol, orbital energies and ΔE (donor–acceptor energy gap) in Hartree (au).

            * **Lewis population** of a well-described closed-shell organic molecule is ≥ 97–98 % of the
              electrons; a much lower value means strong delocalisation (or that one Lewis structure is a
              poor description, e.g. in some metal complexes).
            * Bond orbital occupancies are close to 2.0 (e.g. 1.96), antibonds close to 0 (0.02–0.1);
              a BD* with ≥ 0.3 e is strongly populated by hyperconjugation.
            * E(2) above ~5 kcal/mol is a significant interaction, above ~20 kcal/mol a strong one
              (typical for n → σ*/π* in conjugated or anionic systems). The *Aggregated* table sums
              E(2) over the separate entries NBO prints for the same donor/acceptor pair (e.g. two lone pairs
              of the same oxygen).
            * An optimisation with `! Opt NBO` prints one NBO analysis per cycle in ORCA: the **last** one
              (`--cycle -1`, default) belongs to the final geometry.
            * "No Wiberg bond index matrix found" means the `BNDIDX` keyword was not in the input.
            * Known limitations: with ORCA/NBO 7 outputs the **E (au)** column of `--analysis`, `--bonds` and
              `--lp` currently shows 0.0000 (the energies are in the NBO summary of the output), and the
              *Hybrid Composition* lists only the first atom of a two-centre bond.

            Example (`nbo.out`, a 37-atom anion C16H13BrF3O3S⁻, B3LYP/def2-TZVP): 211.9999 electrons, Lewis
            97.88 %, 845 NBOs; most positive NPA charge S15 (NBO label; FROG index 14) +2.048, most negative O18 −0.942; strongest E(2)
            LP O 17 → BD* S 15–C 19, 26.48 kcal/mol (ΔE 0.33 au), BD* S 15–C 19 occupancy 0.35.
            """,
    )
    examples = [
        ('frog-nbo nbo.out --summary', 'Populations, charge extremes and strongest E(2) interactions'),
        ('frog-nbo nbo.out --e2 -t 5 -i 16 --dtype LP', 'Lone-pair donations of atom 16 above 5 kcal/mol'),
        ('frog-nbo nbo.out --cube-e2 3 --voxels 60',
         'Cubes of the donor/acceptor NBOs of the 3 strongest E(2) interactions in nbo_nbo_cubes/'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help="ORCA output files with an NBO analysis (! NBO)")

        # What to show
        parser.add_argument('--all', action='store_true',
                            help="Show the full analysis (summary, NPA, NBOs, E(2), Wiberg; bonds and lone "
                                 "pairs with -i)")
        parser.add_argument('--summary', action='store_true', help="Show a short summary of the NBO analysis")
        parser.add_argument('--npa', action='store_true', help="Show NPA charges")
        parser.add_argument('--analysis', action='store_true',
                            help="Show the NBO list (occupancy, energy, hybrid composition)")
        parser.add_argument('--e2', action='store_true', help="Show second-order E(2) perturbation energies")
        parser.add_argument('--wiberg', action='store_true', help="Show the Wiberg bond index matrix and atom totals")
        parser.add_argument('--bonds', action='store_true',
                            help="Show bond orbitals (BD) of the selected atoms (first two of -i)")
        parser.add_argument('--lp', action='store_true',
                            help="Show lone pair orbitals (LP) of the selected atom (first of -i)")
        parser.add_argument('--cycle', type=int, default=-1,
                            help="NBO cycle to inspect (-1 = last/converged, 0 = first)")

        # Filters
        parser.add_argument('-i', '--indices', nargs='+', type=int,
                            help="Keep only results involving any of these atoms (0-based)")
        parser.add_argument('--dtype', type=str, help="Filter E(2) by donor type (e.g. LP, BD)")
        parser.add_argument('--atype', type=str, help="Filter E(2) by acceptor type (e.g. BD*, RY)")
        parser.add_argument('-t', '--threshold', type=float, default=2.0, help="E(2) threshold in kcal/mol")

        # NBO cubes
        parser.add_argument('--cube', nargs='+', type=int, metavar='N',
                            help="Write cube files of these NBOs (1-based NBO numbers)")
        parser.add_argument('--cube-e2', type=int, nargs='?', const=3, default=None, metavar='K',
                            help="Write cube files of the donor and acceptor NBOs of the K strongest E(2) "
                                 "interactions (K = 3 if omitted)")
        parser.add_argument('--coeff-file', type=str, default=None, metavar='PATH',
                            help="NBO coefficient file (.37, from the NBO PLOT keyword); auto-detected next "
                                 "to the input if omitted")
        self.add_grid_args(parser, spacing=0.15, padding=5.0)
        parser.add_argument('--out-dir', metavar='DIR', default=None,
                            help="Directory for the <stem>_nbo_cubes/ folders (default: next to each input file)")

    _SECTIONS = ('all', 'summary', 'npa', 'analysis', 'e2', 'wiberg', 'bonds', 'lp')

    def run(self, args):
        if not any(getattr(args, k) for k in self._SECTIONS) and not args.cube and args.cube_e2 is None:
            raise InputError("Nothing to show: use --summary, --all, --npa, --analysis, --e2, --bonds, --lp, "
                             "--wiberg, --cube or --cube-e2")

        files = self.require_files(args.file)
        self._failures = []
        items_data = []

        for p in files:
            fname = str(p)
            print_info(f"Reading file: {fname}")

            try:
                calc = load_calculation(fname)
            except Exception as e:
                self._fail(f"Failed to load calculation from {fname}: {e}")
                continue

            if not hasattr(calc, 'nbo') or calc.nbo is None:
                self._fail(f"{fname} does not contain NBO data (or it could not be parsed)")
                continue

            nbo = calc.nbo
            if args.cycle != -1:
                from frogchem.orca_parser.nbo import NBOAnalysis
                nbo = NBOAnalysis(calc.fl, cycle=args.cycle)

            items_data.append({
                'path': p,
                'fname': fname,
                'calc': calc,
                'nbo': nbo,
            })

        if not items_data:
            raise ParseError("No valid NBO data found in any of the input files")

        if args.all:
            args.npa = True
            args.analysis = True
            args.e2 = True
            args.wiberg = True
            if args.indices:
                args.bonds = True
                args.lp = True

        if args.checkpoint:
            self._run_ensemble_average(items_data, args)
            if not args.individual:
                self._raise_failures()
                return

        for item in items_data:
            self._process_item(item, args)
        self._raise_failures()

    def _fail(self, message: str) -> None:
        """Report a non-fatal error; the run then ends with a nonzero exit code."""
        print_error(message)
        self._failures.append(message)

    def _raise_failures(self) -> None:
        if self._failures:
            raise ParseError(f"{len(self._failures)} error(s) occurred, see the messages above")

    def _process_item(self, item: Dict[str, Any], args: argparse.Namespace):
        fname = item['fname']
        nbo = item['nbo']
        atoms = args.indices

        # --- Summary ---
        if args.summary or args.all:
            self._print_nbo_summary(fname, nbo, args)

        # --- NPA Charges ---
        if args.npa:
            try:
                if nbo.npa_charges is not None:
                    df_npa = nbo.npa_charges
                    if atoms:
                        df_npa = df_npa[df_npa['Index'].isin(atoms)]
                    if not df_npa.empty:
                        data = df_npa[['Atom', 'Index', 'Charge', 'Valence', 'Rydberg']].values.tolist()
                        headers = ['Atom', 'Idx', 'Charge (e)', 'Valence', 'Rydberg']
                        self.output(data, headers, title=f"NPA Charges ({fname})")
                    else:
                        print_info("No NPA charges found for the selected atoms")
                else:
                    print_warning(f"No NPA charges found in {fname}")
            except Exception as e:
                self._fail(f"Error parsing NPA charges in {fname}: {e}")

        # --- NBO Analysis (Orbital Details) ---
        if args.analysis:
            try:
                if nbo.nbo_details is not None:
                    df_nbo = nbo.nbo_details
                    if atoms:
                        user_atoms = set(atoms)
                        df_nbo = df_nbo[df_nbo['Atom_Indices'].apply(lambda idx: not set(idx).isdisjoint(user_atoms))]
                    if not df_nbo.empty:
                        table_data = _format_orbital_rows(df_nbo, translate=True)
                        headers = ['NBO', 'Type', 'Atoms', 'Occ', 'E (au)', 'Hybrid Composition']
                        self.output(table_data, headers, title=f"Natural Bond Orbital Analysis ({fname})",
                                    floatfmt='.4f')
                    else:
                        print_info("No orbitals found involving the selected atoms")
                else:
                    print_warning(f"No 'NATURAL BOND ORBITAL ANALYSIS' section found in {fname}")
            except Exception as e:
                self._fail(f"Error parsing NBO details in {fname}: {e}")

        # --- Bonds Query ---
        if args.bonds:
            try:
                a1 = atoms[0] if atoms and len(atoms) >= 1 else None
                a2 = atoms[1] if atoms and len(atoms) >= 2 else None
                df_bonds = nbo.get_bonds(atom1=a1, atom2=a2)
                if not df_bonds.empty:
                    table_data = _format_orbital_rows(df_bonds)
                    headers = ['NBO', 'Type', 'Bond', 'Occ', 'E (au)', 'Hybrid Composition']
                    self.output(table_data, headers, title=f"NBO Bonds ({fname})", floatfmt='.4f')
                else:
                    print_info("No bond orbitals found matching the criteria")
            except Exception as e:
                self._fail(f"Error parsing NBO bonds in {fname}: {e}")

        # --- Lone Pairs Query ---
        if args.lp:
            try:
                a1 = atoms[0] if atoms else None
                df_lp = nbo.get_lone_pairs(atom=a1)
                if not df_lp.empty:
                    table_data = _format_orbital_rows(df_lp)
                    headers = ['NBO', 'Type', 'Lone Pair', 'Occ', 'E (au)', 'Hybrid Composition']
                    self.output(table_data, headers, title=f"NBO Lone Pairs ({fname})", floatfmt='.4f')
                else:
                    print_info("No lone pair orbitals found matching the criteria")
            except Exception as e:
                self._fail(f"Error parsing NBO lone pairs in {fname}: {e}")

        # --- Wiberg Bond Index ---
        if args.wiberg:
            try:
                if nbo.wiberg_matrix is not None:
                    df_wib = nbo.wiberg_df
                    if atoms:
                        sel = [a for a in atoms if a in df_wib.index]
                        df_wib = df_wib.loc[sel, sel]

                    headers = ['Atom'] + [str(c) for c in df_wib.columns]
                    table_data = [[str(idx)] + [float(v) for v in row.values] for idx, row in df_wib.iterrows()]
                    self.output(table_data, headers, title=f"Wiberg Bond Index Matrix ({fname})", floatfmt='.4f')

                    if nbo.wiberg_totals:
                        tot_data = [
                            [idx, float(tot)]
                            for idx, tot in nbo.wiberg_totals.items()
                            if not atoms or idx in atoms
                        ]
                        self.output(tot_data, ['Atom Idx', 'Total Valence'], title=f"Wiberg Totals ({fname})",
                                    floatfmt='.4f')
                else:
                    print_info(f"No Wiberg bond index matrix found in {fname}")
            except Exception as e:
                self._fail(f"Error parsing Wiberg bond indices in {fname}: {e}")

        # --- E(2) Perturbation ---
        if args.e2:
            try:
                self._print_e2(fname, nbo, args)
            except Exception as e:
                self._fail(f"Error parsing E(2) perturbation energies in {fname}: {e}")

        # --- 3D CUBE Generation ---
        if args.cube or args.cube_e2 is not None:
            self._generate_nbo_cubes(item, args)

    def _print_e2(self, fname: str, nbo, args: argparse.Namespace):
        if nbo.perturbation_theory is None:
            print_warning(f"No E(2) analysis found in {fname}")
            return
        df_e2 = nbo.perturbation_theory
        thr = args.threshold

        if args.dtype:
            df_e2 = df_e2[df_e2['Donor_Type'].str.upper().str.contains(args.dtype.upper(), regex=False)]
        if args.atype:
            df_e2 = df_e2[df_e2['Acceptor_Type'].str.upper().str.contains(args.atype.upper(), regex=False)]
        if args.indices and not df_e2.empty:
            df_e2 = df_e2[df_e2.apply(lambda row: _involves_atoms(row, args.indices), axis=1)]

        if df_e2.empty:
            print_info("No E(2) interactions found matching the criteria")
            return

        # Detailed Table
        df_detail = df_e2[df_e2['E(2)'] > thr].sort_values(by='E(2)', ascending=False)
        if not df_detail.empty:
            table_data = []
            for _, row in df_detail.iterrows():
                d_str = f"{translate_type(row['Donor_Type'])} {row['Donor_Label']}"
                a_str = f"{translate_type(row['Acceptor_Type'])} {row['Acceptor_Label']}"
                table_data.append([d_str, a_str, row['E(2)'], row['E_diff']])
            headers = ['Donor Orbital', 'Acceptor Orbital', 'E(2) (kcal/mol)', 'ΔE (au)']
            self.output(table_data, headers, title=f"Detailed Interactions (> {thr} kcal/mol) ({fname})")
        else:
            print_info(f"No individual interactions > {thr} kcal/mol found")

        # Aggregated Table
        group_cols = ['Donor_Type', 'Donor_Label', 'Acceptor_Type', 'Acceptor_Label']
        df_agg = df_e2.groupby(group_cols)['E(2)'].sum().reset_index()
        df_agg = df_agg[df_agg['E(2)'] > thr].sort_values(by='E(2)', ascending=False)
        if not df_agg.empty:
            table_data = []
            for _, row in df_agg.iterrows():
                d_str = f"{translate_type(row['Donor_Type'])} {row['Donor_Label']}"
                a_str = f"{translate_type(row['Acceptor_Type'])} {row['Acceptor_Label']}"
                table_data.append([d_str, a_str, row['E(2)']])
            headers = ['Donor Orbital', 'Acceptor Orbital', 'Total E(2) (kcal/mol)']
            self.output(table_data, headers, title=f"Aggregated Interactions (Sum > {thr} kcal/mol) ({fname})")
        else:
            print_info(f"No aggregated interactions > {thr} kcal/mol found")

    def _print_nbo_summary(self, fname: str, nbo, args):
        """Print a short summary of the NBO calculation."""
        # 1. Total Lewis vs non-Lewis populations if available
        if nbo.nbo_details is not None and not nbo.nbo_details.empty:
            df_det = nbo.nbo_details
            total_elec = float(df_det['Occupancy'].sum())
            non_lewis_types = {'BD*', 'LP*', 'LV', 'RY', 'RY*'}
            lewis_occ = float(df_det[~df_det['Type'].isin(non_lewis_types)]['Occupancy'].sum())
            non_lewis_occ = float(df_det[df_det['Type'].isin(non_lewis_types)]['Occupancy'].sum())
            lewis_pct = (lewis_occ / total_elec * 100.0) if total_elec > 0 else None
            non_lewis_pct = (non_lewis_occ / total_elec * 100.0) if total_elec > 0 else None

            summary_rows = [
                ["Total Electrons", total_elec, 100.0 if total_elec > 0 else None],
                ["Lewis Population", lewis_occ, lewis_pct],
                ["Non-Lewis Population", non_lewis_occ, non_lewis_pct],
                ["Total NBOs", len(df_det), None],
            ]
            self.output(summary_rows, ["Metric", "Value", "%"], title=f"Natural Population Summary ({fname})",
                        floatfmt='.4f')

        # 2. NPA Charges Extremes
        if nbo.npa_charges is not None and not nbo.npa_charges.empty:
            df_npa = nbo.npa_charges.sort_values(by='Charge')
            n_top = min(3, len(df_npa))
            most_neg = df_npa.head(n_top)
            most_pos = df_npa.tail(n_top).iloc[::-1]
            npa_extreme_rows = []
            for _, r in most_pos.iterrows():
                npa_extreme_rows.append([f"{r['Atom']}{int(r['Index']) + 1}", int(r['Index']), float(r['Charge']),
                                         "Most positive"])
            for _, r in most_neg.iterrows():
                npa_extreme_rows.append([f"{r['Atom']}{int(r['Index']) + 1}", int(r['Index']), float(r['Charge']),
                                         "Most negative"])
            self.output(npa_extreme_rows, ["Atom (NBO label)", "Idx", "Charge (e)", "Role"],
                        title=f"NPA Charge Extremes ({fname})")

        # 3. Top E(2) Interactions
        if nbo.perturbation_theory is not None and not nbo.perturbation_theory.empty:
            df_top = nbo.perturbation_theory.sort_values(by='E(2)', ascending=False).head(5)
            top_rows = []
            for _, r in df_top.iterrows():
                d_str = f"#{r['Donor_Index']} {translate_type(r['Donor_Type'])} {r['Donor_Label']}"
                a_str = f"#{r['Acceptor_Index']} {translate_type(r['Acceptor_Type'])} {r['Acceptor_Label']}"
                top_rows.append([d_str, a_str, float(r['E(2)']), float(r['E_diff'])])
            self.output(top_rows, ["Donor Orbital", "Acceptor Orbital", "E(2) (kcal/mol)", "ΔE (au)"],
                        title=f"Top 5 E(2) Interactions ({fname})")

    def _generate_nbo_cubes(self, item: Dict[str, Any], args: argparse.Namespace):
        """Generate 3D CUBE files for requested NBOs (manual indices and/or top E(2) pairs)."""
        fname = item['fname']
        calc = item['calc']
        nbo = item['nbo']
        calc_path = Path(item['path'])

        coeff_mat, coeff_src = _get_nbo_coefficients(calc, calc_path, args.coeff_file)
        if coeff_mat is None:
            self._fail(
                f"Cannot generate cube files for {fname}: NBO coefficients (.37 file) not found. "
                "ORCA: add %nbo nbokeylist = \"$nbo plot $end\" end; Gaussian: add $NBO PLOT $END "
                "(or pop=(nbo,savenbo) with .fchk); or pass --coeff-file PATH."
            )
            return

        print_info(f"Using NBO coefficients from {coeff_src} "
                   f"({coeff_mat.shape[0]} AOs x {coeff_mat.shape[1]} NBOs)")

        requests = {}  # nbo_idx -> [(tag, note), ...]

        # Manual selection: --cube 42 78
        for idx in args.cube or []:
            requests.setdefault(int(idx), []).append(('manual', 'user-selected'))

        # Automatic E(2) pairs: --cube-e2 [K]
        if args.cube_e2 is not None:
            n_pairs = max(1, int(args.cube_e2))
            if nbo.perturbation_theory is not None and not nbo.perturbation_theory.empty:
                df_e2 = nbo.perturbation_theory.sort_values(by='E(2)', ascending=False)
                if args.indices:
                    df_e2 = df_e2[df_e2.apply(lambda row: _involves_atoms(row, args.indices), axis=1)]

                top_pairs = df_e2.head(n_pairs)
                for pair_rank, (_, row) in enumerate(top_pairs.iterrows(), start=1):
                    note = f"E(2) = {float(row['E(2)']):.1f} kcal/mol"
                    requests.setdefault(int(row['Donor_Index']), []).append((f'e2p{pair_rank}_donor', note))
                    requests.setdefault(int(row['Acceptor_Index']), []).append((f'e2p{pair_rank}_acceptor', note))
            else:
                print_warning(f"No E(2) perturbation data in {fname} for --cube-e2")

        if not requests:
            print_info("No NBOs selected for cube generation")
            return

        out_dir = self.derived_path(calc_path, 'nbo_cubes', '')
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = calc_path.stem
        rows = []
        generated_cubes = []

        print_info(f"Writing NBO cube files to {out_dir}")

        for nbo_idx, tags in requests.items():
            col = nbo_idx - 1
            if col < 0 or col >= coeff_mat.shape[1]:
                self._fail(f"NBO {nbo_idx} out of range (1-{coeff_mat.shape[1]}) in {fname}, skipped")
                continue

            otype, olabel, occ, energy = "NBO", "", None, None
            if nbo.nbo_details is not None and not nbo.nbo_details.empty:
                matches = nbo.nbo_details[nbo.nbo_details['Index'] == nbo_idx]
                if not matches.empty:
                    row = matches.iloc[0]
                    otype = str(row['Type'])
                    olabel = str(row['Label'])
                    occ = float(row['Occupancy'])
                    energy = float(row['Energy'])

            clean_lbl = re.sub(r'_+', '_', f"{otype}_{olabel}".replace("*", "star")
                               .replace(" ", "_").replace("-", "_").replace("/", "_")
                               .replace("(", "").replace(")", "")).strip("_")
            roles = [t[0] for t in tags if t[0] != 'manual']
            prefix = f"{stem}_{'_'.join(roles)}" if roles else stem
            cube_path = out_dir / f"{prefix}_nbo{nbo_idx}_{clean_lbl}.cube"

            desc = f"{otype} {olabel}".strip()
            build_nbo_cube(
                calc=calc,
                nbo_index=nbo_idx,
                coeff_vector=coeff_mat[:, col],
                output=str(cube_path),
                spacing=args.spacing,
                padding=args.padding,
                n_voxels=args.voxels,
                label=desc,
                occupancy=occ,
                energy=energy,
            )
            self.saved(cube_path, 'cube')
            generated_cubes.append(str(cube_path))
            rows.append([nbo_idx, otype, olabel, occ, energy, '; '.join(t[1] for t in tags), str(cube_path)])

        if generated_cubes:
            scripts = generate_visualization_scripts(generated_cubes, output_dir=out_dir)
            self.saved(scripts['chimerax'], 'ChimeraX script')
            self.saved(scripts['pymol'], 'PyMOL script')
            self.output(rows, ['NBO', 'Type', 'Label', 'Occ', 'E (au)', 'Selection', 'Cube'],
                        title=f"NBO Cubes ({fname})", floatfmt='.4f')

    def _run_ensemble_average(self, items_data: List[Dict[str, Any]], args: argparse.Namespace):
        """Compute Boltzmann-weighted averages of NPA charges, WBI, and E(2) interactions."""
        print_info(f"Loading Boltzmann populations from checkpoint: {args.checkpoint}")
        try:
            ens = self.build_ensemble(items_data, path_extractor=lambda x: x['fname'])
        except Exception as e:
            raise InputError(f"Failed to load checkpoint {args.checkpoint}: {e}") from e

        self.output_weights(ens)
        weights = ens.weights

        # 1. NPA Charges
        if args.npa:
            atom_indices = []
            for item in items_data:
                df = item['nbo'].npa_charges
                if df is not None and not df.empty:
                    for idx in df['Index'].tolist():
                        if idx not in atom_indices:
                            atom_indices.append(idx)
            if args.indices:
                atom_indices = [idx for idx in atom_indices if idx in args.indices]
            atom_indices.sort()

            npa_avg_rows = []
            for a_idx in atom_indices:
                w_sum = 0.0
                weighted_charge = 0.0
                weighted_val = 0.0
                weighted_ryd = 0.0
                atom_symbol = "X"

                for item, w in zip(items_data, weights):
                    df = item['nbo'].npa_charges
                    if df is not None and not df.empty:
                        row = df[df['Index'] == a_idx]
                        if not row.empty:
                            atom_symbol = str(row.iloc[0]['Atom'])
                            w_sum += w
                            weighted_charge += w * float(row.iloc[0]['Charge'])
                            weighted_val += w * float(row.iloc[0]['Valence'])
                            weighted_ryd += w * float(row.iloc[0]['Rydberg'])

                if w_sum > 0:
                    weighted_charge /= w_sum
                    weighted_val /= w_sum
                    weighted_ryd /= w_sum
                    npa_avg_rows.append([
                        atom_symbol, a_idx, weighted_charge, weighted_val, weighted_ryd
                    ])

            if npa_avg_rows:
                self.output(
                    npa_avg_rows,
                    headers=['Atom', 'Idx', '⟨Charge⟩ (e)', '⟨Valence⟩', '⟨Rydberg⟩'],
                    title="Ensemble-Averaged NPA Charges (Boltzmann Weighted)"
                )

        # 2. Wiberg Bond Index Matrix and Totals
        if args.wiberg:
            first_wib = None
            for item in items_data:
                if item['nbo'].wiberg_df is not None:
                    first_wib = item['nbo'].wiberg_df
                    break
            if first_wib is not None:
                avg_mat = np.zeros_like(first_wib.values, dtype=float)
                w_tot = 0.0
                for item, w in zip(items_data, weights):
                    if item['nbo'].wiberg_df is not None:
                        avg_mat += w * item['nbo'].wiberg_df.values
                        w_tot += w
                if w_tot > 0:
                    avg_mat /= w_tot

                w_headers = ['Idx'] + [str(c) for c in first_wib.columns]
                w_rows = []
                for idx_label, row in zip(first_wib.index, avg_mat):
                    w_rows.append([str(idx_label)] + [float(v) for v in row])

                self.output(w_rows, headers=w_headers, title="Ensemble-Averaged Wiberg Bond Index Matrix")

                all_tot_indices = sorted(list(set(
                    idx for item in items_data if item['nbo'].wiberg_totals
                    for idx in item['nbo'].wiberg_totals.keys()
                )))
                if args.indices:
                    all_tot_indices = [idx for idx in all_tot_indices if idx in args.indices]

                tot_avg_rows = []
                for idx in all_tot_indices:
                    w_sum = 0.0
                    weighted_tot = 0.0
                    for item, w in zip(items_data, weights):
                        if item['nbo'].wiberg_totals and idx in item['nbo'].wiberg_totals:
                            w_sum += w
                            weighted_tot += w * item['nbo'].wiberg_totals[idx]
                    if w_sum > 0:
                        tot_avg_rows.append([idx, weighted_tot / w_sum])

                if tot_avg_rows:
                    self.output(tot_avg_rows, headers=['Atom Idx', '⟨Total Valence⟩'], title="Ensemble-Averaged Wiberg Totals")

        # 3. E(2) Perturbation
        if args.e2:
            e2_pairs = {}
            for item, w in zip(items_data, weights):
                df = item['nbo'].perturbation_theory
                if df is not None and not df.empty:
                    for _, row in df.iterrows():
                        pair_key = (
                            str(row['Donor_Type']), str(row['Donor_Label']),
                            str(row['Acceptor_Type']), str(row['Acceptor_Label'])
                        )
                        if pair_key not in e2_pairs:
                            e2_pairs[pair_key] = {'e2_sum': 0.0, 'w_sum': 0.0, 'ediff_sum': 0.0}
                        e2_pairs[pair_key]['e2_sum'] += w * float(row['E(2)'])
                        e2_pairs[pair_key]['ediff_sum'] += w * float(row.get('E_diff', 0.0))
                        e2_pairs[pair_key]['w_sum'] += w

            e2_rows = []
            for (dt, dl, at, al), vals in e2_pairs.items():
                if vals['w_sum'] > 0:
                    avg_e2 = vals['e2_sum'] / vals['w_sum']
                    avg_ediff = vals['ediff_sum'] / vals['w_sum']
                    if avg_e2 >= args.threshold:
                        d_str = f"{translate_type(dt)} {dl}"
                        a_str = f"{translate_type(at)} {al}"
                        e2_rows.append([d_str, a_str, avg_e2, avg_ediff])

            e2_rows.sort(key=lambda x: x[2], reverse=True)
            if e2_rows:
                self.output(
                    e2_rows,
                    headers=['Donor Orbital', 'Acceptor Orbital', '⟨E(2)⟩ (kcal/mol)', '⟨ΔE⟩ (au)'],
                    title=f"Ensemble-Averaged E(2) Interactions (> {args.threshold} kcal/mol)"
                )



# backward compat
def parse_args():
    return NboParser().build_parser().parse_args()

def main():
    NboParser().main()


if __name__ == '__main__':
    main()
