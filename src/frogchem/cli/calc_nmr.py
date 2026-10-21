"""
Extract and average NMR Chemical Shifts, J-Couplings, and interatomic distances (NOE)
from ORCA output files, with optional Boltzmann weighting from conformer checkpoint files.

Andrea Pellegrini, 2026
"""

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS
from frogchem.ensemble import extract_conf_and_protocol
from frogchem.exceptions import InputError, ParseError
from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning
from frogchem.nmr import (
    NUCLEUS_DATA,
    convert_shielding_to_shift,
    extract_element_from_nucleus,
    extract_reference_shieldings,
    format_spin,
    get_nmr_equivalencies,
    get_nucleus_spin,
    is_nucleus_decoupled,
    is_quadrupolar,
    load_nucleus_database,
    normalize_nucleus_name,
    parse_nuclei_list,
    parse_spin,
    parse_spins_arg,
    symmetrize_nmr_data,
)
from frogchem.nmr_sim import (
    plot_1d_spectrum,
    plot_2d_spectrum,
    simulate_1d_spectrum,
    simulate_1d_lines,
    simulate_cosy,
    simulate_noesy,
    simulate_hsqc,
    simulate_hmbc,
    simulate_heteronuclear_2d,
    simulate_quantum_transitions,
)

logger = logging.getLogger(__name__)

#: Square matrices are printed up to this many atoms (or always with -i); otherwise pair lists.
MAX_MATRIX_ATOMS = 15
#: Pair lists are truncated to this many rows unless the section is requested explicitly.
MAX_PAIRS = 40
ENSEMBLE_SCOPE = 'Boltzmann average'
ENSEMBLE_STEM = 'nmr_ensemble'


def _pairwise_distances(geom: np.ndarray) -> np.ndarray:
    geom = np.asarray(geom, dtype=float)
    if geom.size == 0:
        return np.zeros((0, 0))
    return np.linalg.norm(geom[:, None, :] - geom[None, :, :], axis=-1)


def _noe_class(r: float) -> str:
    return "Strong" if r < 2.5 else ("Medium" if r < 3.5 else "Weak")


class CalcNmr(CLICommand, name='frog-nmr'):
    """NMR shifts, J couplings and NOE distances from ORCA outputs, with spectrum simulation.

    Prints shieldings/shifts, J-coupling and distance tables (with NOE estimates) for each
    file, or their Boltzmann average over a conformer ensemble (-c). Simulates 1D spectra
    (full quantum spin simulation for coupled nuclei, Lorentzian lines otherwise) and 2D
    COSY, NOESY, HSQC and HMBC spectra for the selected nuclei (-n).
    Atom labels such as H5 are 1-based; -i and the Index column are 0-based.
    """
    category = 'nmr'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Chemical Shifts & Shieldings (<file>)',
               'Per atom: σ_iso, δ (vs --ref or the default reference) and anisotropy (ppm)'),
        Output('table', 'Spin-Spin Couplings J (Hz) (<file>)',
               f'Square J matrix (≤ {MAX_MATRIX_ATOMS} atoms or with -i); otherwise '
               '"Significant Spin-Spin Couplings" pair list with |J| ≥ 0.5 Hz and distance'),
        Output('table', 'Interatomic Distances (Å) (<file>)',
               f'Square distance matrix (≤ {MAX_MATRIX_ATOMS} atoms or with -i)'),
        Output('table', 'NOE Estimation (<file>)',
               'Atom pairs within --r-cutoff: distance, relative NOE (∝ r⁻⁶) and intensity class'),
        Output('table', 'Simulated <nucleus> NMR Signals (<file>)',
               'Equivalence groups with σ_iso, δ and member atoms', when='simulation flags'),
        ENSEMBLE_WEIGHTS,
        Output('table', '... (Boltzmann average)',
               'The tables above for the ensemble; distances as ⟨r⟩ and ⟨r⁻⁶⟩⁻¹ᐟ⁶', when='-c'),
        Output('file', '<stem>_1d_<nucleus>.dat', '1D spectrum, columns ppm and intensity', when='--save-xy'),
        Output('file', '<stem>_<exp>_<f2>_<f1>.csv',
               '2D spectrum matrix; first row F2 axis, first column F1 axis (ppm)', when='--save-matrix'),
        Output('plot', '<stem>_1d_<nucleus>.png',
               '1D spectrum of each simulated nucleus. Ensemble: nmr_ensemble_<tag> in the current '
               'directory; --plot PATH: PATH_<tag> (PATH_<stem>_<tag> for several inputs)', when='--plot'),
        Output('plot', '<stem>_<exp>_<f2>_<f1>.png',
               '2D spectra: exp = cosy, noesy, hsqc, hmbc (e.g. nmr_hsqc_1h_13c.png)', when='--plot'),
    ]
    guide = Guide(
        why="""
            Turns an ORCA NMR calculation into numbers you can compare with an experimental spectrum:
            chemical shifts δ (ppm), J couplings (Hz) and H···H distances with an NOE estimate, and
            simulated 1D and 2D spectra (COSY, NOESY, HSQC, HMBC). Use it to assign signals, to decide
            between candidate structures or diastereomers (the one whose computed shifts fit the
            experiment best), or to Boltzmann-average the NMR of a conformer ensemble.
            """,
        needs="""
            * An **ORCA output of an NMR calculation** on an optimised geometry: `! NMR` gives shieldings
              of all nuclei; J couplings need an `%eprnmr` block, e.g.
              `%eprnmr nuclei = all H {shift, ssall} end` (`ssall` = all spin-spin coupling terms).
              Couplings are expensive: request them only for the nuclei you need.
            * A **reference** to convert shielding σ into shift δ ≈ σ_ref − σ. Best: an output of TMS (for
              1H, 13C) computed with exactly the same method, basis and solvent, passed with `--ref tms.out`.
              Without `--ref` built-in default values are used (σ_ref 31.8 ppm for 1H, 185.0 ppm for 13C).
            * For `-c`: one NMR output per conformer and a checkpoint with populations (see the Ensemble
              user guide). For `--hsqc`/`--hmbc`: shieldings of both nuclei (e.g. H and C) in the same output.
            """,
        how=[
            "Look at the data: `frog-nmr nmr.out -n H` prints the 1H shifts, the H–H J couplings and the "
            "NOE table. Without `-n` the tables contain every nucleus in the output.",
            "Use a proper reference: `--ref tms.out`, or one value per nucleus `--ref 1H=31.8,13C=185.0`. "
            "A single number (`--ref 31.8`) is applied to every nucleus shown, so combine it with `-n 1H`.",
            "Show only one table with `--shifts`, `--couplings` or `--distances`; `-i 23 24 25` restricts "
            "everything to those atoms (0-based) and prints full matrices.",
            "Simulate spectra: `--sim-1d --plot` (1D 1H, saved as `nmr_1d_1h.png` next to the input), add "
            "`--cosy`, `--noesy`, `--hsqc`, `--hmbc` with `-n 1H 13C` for 2D maps; `--freq 400` sets the "
            "spectrometer frequency (MHz), `--save-xy`/`--save-matrix` write the data for other programs.",
            "For a conformer ensemble: `frog-nmr conf_*/nmr.out -c checkpoint.json --sim-1d --plot`.",
        ],
        expect="""
            * **Chemical Shifts & Shieldings**: per atom σ_iso, δ and the shielding anisotropy (ppm).
              Atom labels such as `H24` are 1-based, the *Index* column is 0-based (H24 = index 23).
              1H shifts should fall in 0–12 ppm, 13C in 0–220 ppm; with a good method and a matching TMS
              reference, typical errors are ±0.1–0.2 ppm (1H) and ±2–5 ppm (13C). A shift far outside these
              ranges usually means a wrong reference (e.g. a 1H reference applied to C or N: δ(N) = −51 ppm
              in one test) rather than a wrong structure.
            * **J couplings** (Hz): a matrix for ≤ 15 atoms, otherwise a list of pairs with |J| ≥ 0.5 Hz.
              Geminal ²J(H,H) are negative (−9 to −15 Hz), vicinal ³J positive (2–12 Hz). Example
              (`NMR_ORCA.out`): CH2 pair H30/H31 −12.0 Hz, aromatic ortho pairs 6.4–6.9 Hz.
            * **NOE Estimation**: pairs within 5 Å (`--r-cutoff`), relative NOE ∝ r⁻⁶ (normalised to 1.8 Å)
              and a class: Strong < 2.5 Å, Medium < 3.5 Å, Weak beyond.
            * **Simulated signals**: chemically equivalent atoms (by symmetry and fast-rotating CH3) are
              merged; e.g. the methyl H37–H39 become one 3H signal at 4.248 ppm. The 1D spectrum shows
              multiplets from the full spin simulation (by default only 1H is coupled; 13C is shown
              decoupled), with the ppm axis decreasing to the right as in experiment.
            * With `-c`, shifts and J are population-weighted averages and the NOE uses ⟨r⁻⁶⟩⁻¹ᐟ⁶; the
              *Ensemble Weights* table shows which file got which population.
            * Warning `2D spectra are only produced with --plot or --save-matrix`: add one of the two.
            """,
    )
    examples = [
        ('frog-nmr nmr.out -n H', 'Proton shieldings, shifts, J couplings and NOE table'),
        ('frog-nmr nmr.out --sim-1d --cosy --hsqc -n 1H 13C --plot',
         'Simulate 1D, COSY and HSQC spectra; plots saved next to nmr.out (nmr_1d_1h.png, ...)'),
        ('frog-nmr conf_*/nmr.out -c checkpoint.json --sim-1d --plot spectra/ens.svg',
         'Boltzmann-averaged data and 1D spectra as spectra/ens_1d_1h.svg'),
    ]

    def setup_args(self, parser: argparse.ArgumentParser):
        parser.add_argument('file', nargs='+', help="ORCA output file(s)")
        parser.add_argument('-i', '--indices', nargs='+', type=int,
                            help="Show only these atoms, 0-based")
        parser.add_argument('--shifts', action='store_true', help="Show only chemical shifts")
        parser.add_argument('--couplings', action='store_true', help="Show only J couplings")
        parser.add_argument('--distances', action='store_true', help="Show only interatomic distances and NOE")

        sim = parser.add_argument_group('simulation')
        sim.add_argument('--sim-1d', action='store_true',
                         help="Simulate the 1D spectrum of each nucleus (quantum spin simulation or Lorentzian lines)")
        sim.add_argument('--sim-1d-c13', action='store_true', help="Simulate the 1D 13C{1H} spectrum")
        sim.add_argument('--cosy', action='store_true', help="Simulate homonuclear COSY spectra")
        sim.add_argument('--noesy', action='store_true', help="Simulate the 1H-1H NOESY spectrum")
        sim.add_argument('--hsqc', action='store_true', help="Simulate edited HSQC / direct (1-bond) correlation spectra")
        sim.add_argument('--hmbc', action='store_true', help="Simulate HMBC / long-range correlation spectra")
        sim.add_argument('--all-spectra', action='store_true',
                         help="Simulate all 1D and 2D spectra for all nuclei (or those given with -n)")
        sim.add_argument('-n', '--nuclei', dest='nucleus', nargs='+', default=None, metavar='NUC',
                         help="Nuclei by element or isotope (e.g. H C S or 1H 13C 33S). Restricts the tables "
                              "to these elements (default: every nucleus in the output) and sets the "
                              "simulated nuclei (default: 1H)")
        sim.add_argument('--freq', type=float, default=500.0, help="Spectrometer 1H frequency in MHz")
        sim.add_argument('--ref', nargs='+', default=None,
                         help="Reference shielding in ppm ('31.8', '1H=31.8,13C=185.0') or reference "
                              "calculation file(s), e.g. a TMS output. A single number is applied to "
                              "EVERY nucleus shown: combine it with -n, or give one value per nucleus")
        sim.add_argument('--fwhm', type=float, default=1.0, help="Line width (FWHM) of 1D Lorentzian lines in Hz")
        sim.add_argument('--j-cutoff', type=float, default=0.3,
                         help="Minimum |J| in Hz for spin-cluster partitioning")
        sim.add_argument('--r-cutoff', type=float, default=5.0,
                         help="Maximum distance in Å for NOE (NOESY and NOE table)")
        sim.add_argument('--no-equiv', action='store_true', help="Disable chemical-equivalence averaging")
        sim.add_argument('--coupled', action='store_true',
                         help="Simulate all nuclei coupled (multiplets); by default only 1H is coupled")
        sim.add_argument('--spin', default=None, help="Nuclear spin of the first nucleus (e.g. '1/2', '5/2', '2.5')")
        sim.add_argument('--spins', default=None,
                         help="Nuclear spins mapping (e.g. 'O=5/2,H=1/2' or '{\"17O\": 2.5}')")
        sim.add_argument('--nuclei-file', default=None, metavar='PATH',
                         help="JSON file with custom nuclear constants and spins")

        self.add_plot_args(parser, optional=True, cmap=True)
        parser.add_argument('--save-xy', action='store_true',
                            help="Also write each simulated 1D spectrum to <stem>_1d_<nucleus>.dat")
        parser.add_argument('--save-matrix', action='store_true',
                            help="Also write each simulated 2D spectrum to <stem>_<exp>_<f2>_<f1>.csv")
        self.add_out_dir_arg(parser)

    # ------------------------------------------------------------------ run
    def run(self, args: argparse.Namespace):
        files = self.require_files(args.file)

        if args.nuclei_file:
            NUCLEUS_DATA.update(load_nucleus_database(args.nuclei_file))

        user_spins = parse_spins_arg(args.spins)
        if args.spin is not None:
            active_nuc = parse_nuclei_list(args.nucleus)[0] if args.nucleus else '1H'
            user_spins[normalize_nucleus_name(active_nuc)] = parse_spin(args.spin)
            user_spins[extract_element_from_nucleus(active_nuc)] = parse_spin(args.spin)
        args.custom_spins = user_spins

        if args.all_spectra:
            args.sim_1d = args.cosy = args.noesy = args.hsqc = args.hmbc = True

        nuclei = parse_nuclei_list(args.nucleus) if args.nucleus is not None else []
        if not nuclei:
            nuclei = ['1H', '13C'] if args.all_spectra else ['1H']
        if args.sim_1d_c13 and '13C' not in nuclei:
            nuclei.append('13C')
        args.nuclei = nuclei

        is_sim = args.sim_1d or args.sim_1d_c13 or args.cosy or args.noesy or args.hsqc or args.hmbc
        show_all = not (args.shifts or args.couplings or args.distances or is_sim)
        self._show = {
            'shifts': show_all or args.shifts,
            'couplings': show_all or args.couplings,
            'distances': show_all or args.distances,
        }
        if (args.cosy or args.noesy or args.hsqc or args.hmbc) and not (self.plot_enabled or args.save_matrix):
            print_warning("2D spectra are only produced with --plot or --save-matrix")

        args.ref = self._load_reference(args.ref)

        confs_data = []
        for p in files:
            fname = str(p)
            print_info(f"Reading file: {fname}")
            try:
                calc = load_calculation(fname)
            except Exception as e:
                print_warning(f"Failed to load {fname}: {e}")
                continue
            if getattr(calc, 'nmr', None) is None:
                print_warning(f"File {fname} does not contain NMR data or parsing failed.")
                continue
            conf_id, proto_id = extract_conf_and_protocol(fname)
            confs_data.append({'path': p, 'fname': fname, 'calc': calc, 'nmr': calc.nmr,
                               'conf_id': conf_id, 'proto_id': proto_id})

        if not confs_data:
            raise ParseError("No valid NMR data found in any of the provided files.")

        # --all-spectra without -n: every element present in the calculations
        if args.all_spectra and args.nucleus is None:
            elements = []
            for c in confs_data:
                for d in c['nmr'].shifts.values():
                    el = d.get('element', '').strip().capitalize()
                    if el and el not in elements:
                        elements.append(el)
            elements.sort(key=lambda e: ({'H': 0, 'C': 1}.get(e, 2), e))
            args.nuclei = [normalize_nucleus_name(el) for el in elements]
            print_info(f"--all-spectra: NMR elements found: {', '.join(elements)} -> {', '.join(args.nuclei)}")

        if isinstance(args.ref, (int, float)):
            ref_elements = {extract_element_from_nucleus(n) for n in args.nuclei}
            if len(ref_elements) > 1:
                print_warning(f"A single numeric --ref ({args.ref:.3f} ppm) is applied to ALL nuclei "
                              f"({', '.join(args.nuclei)}); use e.g. --ref 1H=31.8,13C=185.0 for per-nucleus references")

        run_individual = not args.checkpoint or args.individual
        stems = [c['path'].stem for c in confs_data]
        unique = len(set(stems)) == len(stems)
        for i, c in enumerate(confs_data):
            c['label'] = c['path'].stem if unique else f"{c['path'].stem}_{i + 1}"
        self._n_sets = (len(confs_data) if run_individual else 0) + (1 if args.checkpoint else 0)

        if args.checkpoint:
            self._run_ensemble_average(confs_data, args, is_sim)

        if not run_individual:
            return

        for item in confs_data:
            nmr = item['nmr']
            filter_atoms = args.indices
            if filter_atoms is None and args.nucleus is not None and not args.all_spectra:
                req = {extract_element_from_nucleus(n).upper() for n in args.nuclei}
                filter_atoms = [i for i, d in nmr.shifts.items() if d.get('element', '').upper() in req]
            dist = _pairwise_distances(item['calc'].geom)
            self._report(item['fname'], nmr.shifts, nmr.j_couplings, list(nmr.atoms), dist, None,
                         filter_atoms, args)
            if is_sim:
                self._run_simulation(nmr.shifts, nmr.j_couplings, item['calc'].geom, item['calc'].atoms,
                                     args, scope=item['path'].name, source=item)

    def _load_reference(self, ref):
        """Resolve --ref once (files are parsed here, not at every conversion)."""
        if ref is None:
            return None
        ref = ref[0] if len(ref) == 1 else ref
        try:
            ref_dict = extract_reference_shieldings(ref)
        except Exception as e:
            raise InputError(f"Failed to load reference shielding from '{ref}': {e}") from e
        if '*' in ref_dict and len(ref_dict) == 1:
            print_info(f"Using manual reference shielding: {ref_dict['*']:.3f} ppm")
            return ref_dict['*']
        nucs = [k for k in sorted(ref_dict) if k in NUCLEUS_DATA] or \
               [k for k in sorted(ref_dict) if len(k) <= 2 and k.isalpha()]
        print_info("Loaded reference shieldings (" + ", ".join(f"{k}: {ref_dict[k]:.3f} ppm" for k in nucs) + ")")
        return {k: ref_dict[k] for k in nucs} or ref_dict

    @staticmethod
    def _shift(iso: float, nucleus: str, ref) -> float:
        try:
            return convert_shielding_to_shift(iso, nucleus=nucleus, ref_sigma=ref)
        except Exception:
            return float('nan')

    # --------------------------------------------------------------- tables
    def _report(self, scope: str, shifts: Dict[int, Dict[str, Any]], j_couplings: Dict[Tuple[int, int], float],
                atoms: List[str], dist: np.ndarray, noe_dist: Optional[np.ndarray],
                filter_atoms: Optional[List[int]], args: argparse.Namespace) -> None:
        """Shift, J and distance/NOE tables of one file or of the ensemble average.

        dist: distance matrix (Å); noe_dist: ⟨r⁻⁶⟩⁻¹ᐟ⁶ matrix for ensembles (None: use dist).
        """
        explicit = args.indices is not None

        def label(idx):
            el = atoms[idx] if idx < len(atoms) else shifts.get(idx, {}).get('element', 'X')
            return f"{el}{idx + 1}"

        if self._show['shifts']:
            selected = sorted(shifts) if filter_atoms is None else [a for a in filter_atoms if a in shifts]
            rows = []
            for idx in selected:
                d = shifts[idx]
                rows.append([idx, label(idx), d['element'], d['iso'],
                             self._shift(d['iso'], d['element'], args.ref), d['aniso']])
            if rows:
                self.output(rows, ['Index', 'Atom', 'Element', 'σ_iso (ppm)', 'δ (ppm)', 'Anisotropy (ppm)'],
                            title=f"Chemical Shifts & Shieldings ({scope})")
            else:
                print_info(f"No chemical shifts found for the selected atoms ({scope}).")

        def r_of(a, b):
            return float(dist[a, b]) if max(a, b) < len(dist) else float('nan')

        if self._show['couplings']:
            j_atoms = sorted({k[0] for k in j_couplings})
            sel = j_atoms if filter_atoms is None else list(filter_atoms)
            if not sel or not j_couplings:
                print_info(f"No J couplings found for the selected atoms ({scope}).")
            elif len(sel) <= MAX_MATRIX_ATOMS or explicit:
                labels = [label(i) for i in sel]
                rows = [[labels[r]] + [0.0 if a == b else float(j_couplings.get((a, b), 0.0)) for b in sel]
                        for r, a in enumerate(sel)]
                self.output(rows, ['Atom'] + labels, title=f"Spin-Spin Couplings J (Hz) ({scope})")
            else:
                pairs = [[label(a), label(b), float(j_couplings.get((a, b), 0.0)), r_of(a, b)]
                         for a, b in combinations(sel, 2) if abs(j_couplings.get((a, b), 0.0)) >= 0.5]
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                n = len(pairs) if args.couplings else min(MAX_PAIRS, len(pairs))
                if pairs:
                    self.output(pairs[:n], ['Atom 1', 'Atom 2', 'J (Hz)', 'Distance (Å)'],
                                title=f"Significant Spin-Spin Couplings, |J| ≥ 0.5 Hz ({scope}; "
                                      f"{n} of {len(pairs)} pairs)")
                print_info(f"The J matrix has {len(sel)} atoms: use -i to print a square matrix.")

        if self._show['distances']:
            sel = list(range(len(atoms))) if filter_atoms is None else [a for a in filter_atoms if a < len(dist)]
            if not sel:
                print_info(f"No atomic distances available ({scope}).")
                return
            labels = [label(i) for i in sel]
            eff = dist if noe_dist is None else noe_dist
            if len(sel) <= MAX_MATRIX_ATOMS or explicit:
                lin_title = "Interatomic Distances (Å)" if noe_dist is None else "Interatomic Distances ⟨r⟩ (Å)"
                self.output([[labels[r]] + [float(dist[a, b]) for b in sel] for r, a in enumerate(sel)],
                            ['Atom'] + labels, title=f"{lin_title} ({scope})")
                if noe_dist is not None:
                    self.output([[labels[r]] + [float(noe_dist[a, b]) for b in sel] for r, a in enumerate(sel)],
                                ['Atom'] + labels, title=f"NOE Effective Distances ⟨r⁻⁶⟩⁻¹ᐟ⁶ (Å) ({scope})")
            else:
                print_info(f"The distance matrix has {len(sel)} atoms: use -i to print a square matrix.")

            pairs = []
            for (ia, a), (ib, b) in combinations(enumerate(sel), 2):
                r_eff = float(eff[a, b])
                if 0.0 < r_eff <= args.r_cutoff:
                    row = [labels[ia], labels[ib], float(dist[a, b])]
                    if noe_dist is not None:
                        row.append(r_eff)
                    pairs.append(row + [(1.8 / r_eff) ** 6, _noe_class(r_eff)])
            pairs.sort(key=lambda x: x[-2], reverse=True)
            if pairs:
                n = len(pairs) if args.distances else min(MAX_PAIRS, len(pairs))
                headers = ['Atom 1', 'Atom 2', 'Distance (Å)'] + \
                          (['r_NOE (Å)'] if noe_dist is not None else []) + ['Rel NOE (∝ r⁻⁶)', 'Intensity']
                self.output(pairs[:n], headers,
                            title=f"NOE Estimation, r ≤ {args.r_cutoff:.1f} Å ({scope}; {n} of {len(pairs)} pairs)")

    # ------------------------------------------------------------- ensemble
    def _run_ensemble_average(self, confs_data: List[Dict[str, Any]], args: argparse.Namespace, is_sim: bool):
        """Boltzmann-weighted shifts, J couplings and distances over the conformer ensemble."""
        print_info(f"Loading Boltzmann populations from checkpoint: {args.checkpoint}")
        ens = self.build_ensemble(confs_data, path_extractor=lambda x: x['fname'])
        weights = np.array(ens.weights, dtype=float)
        self.output_weights(ens)

        # Shifts: weighted over the conformers that contain the atom
        ens_shifts: Dict[int, Dict[str, Any]] = {}
        for idx in sorted({k for c in confs_data for k in c['nmr'].shifts}):
            w_sum = iso = aniso = 0.0
            entry: Dict[str, Any] = {}
            for c, w in zip(confs_data, weights):
                d = c['nmr'].shifts.get(idx)
                if d is None:
                    continue
                entry.setdefault('element', d['element'])
                if 'isotope' in d:
                    entry.setdefault('isotope', d['isotope'])
                iso += w * d['iso']
                aniso += w * d['aniso']
                w_sum += w
            if w_sum > 0:
                entry.update(iso=iso / w_sum, aniso=aniso / w_sum)
                ens_shifts[idx] = entry

        ens_j: Dict[Tuple[int, int], float] = {}
        for key in {k for c in confs_data for k in c['nmr'].j_couplings}:
            ens_j[key] = float(sum(w * c['nmr'].j_couplings.get(key, 0.0) for c, w in zip(confs_data, weights)))

        first = confs_data[0]
        n_atoms = len(first['nmr'].atoms)
        dists = [_pairwise_distances(c['nmr'].geom) for c in confs_data]
        if all(d.shape == (n_atoms, n_atoms) for d in dists):
            stack = np.array(dists)
            lin = np.tensordot(weights, stack, axes=1)
            with np.errstate(divide='ignore'):
                inv6 = np.where(stack > 0, stack ** -6.0, 0.0)
            r6 = np.tensordot(weights, inv6, axes=1)
            with np.errstate(divide='ignore'):
                noe = np.where(r6 > 0, r6 ** (-1.0 / 6.0), 0.0)
        else:
            print_warning("Conformers have different numbers of atoms: distances are not averaged")
            lin = noe = np.zeros((0, 0))

        self._report(ENSEMBLE_SCOPE, ens_shifts, ens_j, list(first['nmr'].atoms), lin, noe, args.indices, args)

        if is_sim:
            self._run_simulation(ens_shifts, ens_j, first['calc'].geom, first['calc'].atoms,
                                 args, scope=ENSEMBLE_SCOPE, source=None)

    # --------------------------------------------------------- output paths
    def _plot_file(self, source: Optional[Dict[str, Any]], tag: str) -> Path:
        """Plot path: --plot PATH as prefix (plus the input name when several inputs are plotted),
        otherwise <stem>_<tag> next to the input (or nmr_ensemble_<tag> in the current directory)."""
        if self.args.plot:
            if self._n_sets > 1:
                tag = f"{source['label'] if source else 'ensemble'}_{tag}"
            return self.plot_path('unused', tag)
        base = self.derived_path(source['path'], '', '') if source else Path(ENSEMBLE_STEM)
        return self.plot_path(base, tag)

    def _data_file(self, source: Optional[Dict[str, Any]], tag: str, ext: str) -> Path:
        if source:
            return self.derived_path(source['path'], tag, ext)
        out_dir = Path(self.args.out_dir) if self.args.out_dir else Path('.')
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{ENSEMBLE_STEM}_{tag}{ext}"

    def _emit_2d(self, source, tag: str, f1, f2, mat, **plot_kwargs) -> None:
        if self.plot_enabled:
            fig = plot_2d_spectrum(f1, f2, mat, opts=self.plot_options(), **plot_kwargs)
            self.save_figure(fig, self._plot_file(source, tag))
        if self.args.save_matrix:
            out = np.full((len(f1) + 1, len(f2) + 1), np.nan)
            out[0, 1:] = f2
            out[1:, 0] = f1
            out[1:, 1:] = mat
            path = self._data_file(source, tag, '.csv')
            np.savetxt(path, out, delimiter=',', fmt='%.6g',
                       header='first row: F2 axis (ppm); first column: F1 axis (ppm)')
            self.saved(path, '2D spectrum')

    # ----------------------------------------------------------- simulation
    def _run_simulation(
        self,
        shifts_dict: Dict[int, Dict[str, Any]],
        j_couplings_dict: Dict[Tuple[int, int], float],
        geom_arr: Any,
        atoms_list: List[str],
        args: argparse.Namespace,
        scope: str = "",
        source: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Run 1D NMR simulation and/or 2D homonuclear & heteronuclear correlation simulations."""
        suffix = f" ({scope})" if scope else ""
        nuclei: List[str] = args.nuclei
        geom = np.array(geom_arr, dtype=float) if geom_arr is not None else np.array([])
        want_2d_output = self.plot_enabled or args.save_matrix

        trace_1d_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        nuc_data_dict: Dict[str, Dict[str, Any]] = {}
        spin_by_nuc: Dict[str, float] = {}

        # 1. Process and simulate 1D for each requested nucleus
        for nuc in nuclei:
            element = extract_element_from_nucleus(nuc)
            active_atoms = [i for i, d in shifts_dict.items() if d.get('element', '').upper() == element.upper()]
            nuc_info = NUCLEUS_DATA.get(nuc, {})

            custom_spins = args.custom_spins
            detected_iso = next((shifts_dict[i]['isotope'] for i in active_atoms if 'isotope' in shifts_dict[i]), None)
            spin_target = detected_iso if (detected_iso and nuc not in custom_spins and element not in custom_spins) else nuc
            spin_val = get_nucleus_spin(spin_target, custom_spins=custom_spins)
            spin_by_nuc[nuc] = spin_val
            spin_str = format_spin(spin_val)

            group_shifts: Dict[int, Dict[str, Any]] = {}
            shifts_ppm: List[float] = []
            group_counts: List[int] = []
            j_mat = np.zeros((0, 0), dtype=float)
            group_dist = np.zeros((0, 0), dtype=float)

            if not active_atoms:
                print_warning(f"No {nuc} (element {element}) NMR data found in calculation.")
            else:
                if not args.no_equiv:
                    try:
                        equiv = get_nmr_equivalencies((atoms_list, geom))
                    except Exception as e:
                        logger.warning(f"Equivalency detection failed for {nuc}: {e}")
                        equiv = {}
                    group_shifts, group_j, group_dist, _groups = symmetrize_nmr_data(
                        shifts_dict, j_couplings_dict, geom, equiv, element=element
                    )
                else:
                    group_shifts = {i: dict(shifts_dict[i], members=[i], count=1) for i in active_atoms}
                    group_j = j_couplings_dict
                    n_act = len(active_atoms)
                    group_dist = np.zeros((n_act, n_act), dtype=float)
                    for a, ia in enumerate(active_atoms):
                        for b, ib in enumerate(active_atoms):
                            if len(geom) > max(ia, ib):
                                group_dist[a, b] = float(np.linalg.norm(geom[ia] - geom[ib]))

                if group_shifts:
                    group_keys = list(group_shifts.keys())
                    shifts_ppm = [convert_shielding_to_shift(group_shifts[k]['iso'], nucleus=nuc, ref_sigma=args.ref)
                                  for k in group_keys]
                    group_counts = [group_shifts[k]['count'] for k in group_keys]
                    j_mat = np.array([[group_j.get((ki, kj), 0.0) for kj in group_keys] for ki in group_keys],
                                     dtype=float)
                    summary_rows = [
                        [f"{element}{k + 1}", d['count'], d['iso'], delta,
                         ", ".join(f"{element}{m + 1}" for m in d['members'])]
                        for k, d, delta in zip(group_keys, group_shifts.values(), shifts_ppm)
                    ]
                    self.output(summary_rows, ['Group Rep', 'N', 'σ_iso (ppm)', f'δ ({nuc}, ppm)', 'Equivalent Atoms'],
                                title=f"Simulated {nuc} (I = {spin_str}) NMR Signals{suffix}")

            nuc_data_dict[nuc] = {
                'element': element, 'shifts_ppm': shifts_ppm, 'group_counts': group_counts,
                'j_mat': j_mat, 'group_dist': group_dist,
            }

            # 1D simulation
            ppm_axis = intensity = None
            is_decoupled = is_nucleus_decoupled(nuc, coupled_arg=args.coupled or None,
                                                decoupled_arg=None, nuclei_data=NUCLEUS_DATA)
            is_13c = (nuc == '13C' or element == 'C')

            # Larmor frequency of this nucleus (MHz): --freq is the 1H frequency
            gamma_ratio = abs(nuc_info.get('gamma', 67.3e6) / NUCLEUS_DATA.get('1H', {}).get('gamma', 267.5e6))
            nuc_freq = args.freq * (gamma_ratio if gamma_ratio > 0 else 0.25)

            if shifts_ppm and not is_decoupled:
                try:
                    transitions = simulate_quantum_transitions(
                        shifts_ppm, j_mat, group_counts=group_counts, freq_mhz=nuc_freq,
                        j_cutoff=args.j_cutoff, spins=spin_val, nucleus=nuc,
                    )
                    ppm_axis, intensity = simulate_1d_spectrum(transitions, freq_mhz=nuc_freq, fwhm_hz=args.fwhm)
                except Exception as e:
                    logger.warning(f"Quantum transitions failed for {nuc}: {e}")

            if ppm_axis is None or intensity is None:
                # Decoupled / Lorentzian line simulation
                nuc_range = nuc_info.get('default_range', None)
                if nuc_range is not None:
                    min_p = min(nuc_range[0], float(np.min(shifts_ppm)) - 5.0) if shifts_ppm else nuc_range[0]
                    max_p = max(nuc_range[1], float(np.max(shifts_ppm)) + 5.0) if shifts_ppm else nuc_range[1]
                    r_ppm = (min_p, max_p)
                elif shifts_ppm:
                    pad = max(2.0, 0.15 * (float(np.max(shifts_ppm)) - float(np.min(shifts_ppm))))
                    r_ppm = (float(np.min(shifts_ppm)) - pad, float(np.max(shifts_ppm)) + pad)
                else:
                    r_ppm = (-10.0, 200.0)

                if is_quadrupolar(spin_val):
                    fwhm_ppm = max(0.04, min(0.3, (args.fwhm * 2.0 * spin_val) / nuc_freq if nuc_freq > 0 else 0.1))
                else:
                    fwhm_ppm = max(0.015, min(0.04, args.fwhm / nuc_freq if nuc_freq > 0 else 0.025))
                ppm_axis, intensity = simulate_1d_lines(shifts_ppm, intensities=group_counts, ppm_range=r_ppm,
                                                        n_points=16384, fwhm_ppm=fwhm_ppm)

            trace_1d_dict[nuc] = (ppm_axis, intensity)

            wants_1d = (
                args.sim_1d
                or (args.sim_1d_c13 and nuc == '13C')
                or (len(nuclei) == 1 and not (args.cosy or args.noesy or args.hsqc or args.hmbc))
            )
            if not wants_1d:
                continue
            tag = f"1d_{nuc.lower()}"
            if self.plot_enabled:
                if is_decoupled:
                    kind = "13C{1H}" if is_13c else f"{nuc}"
                    detail = f"I = {spin_str}" + ("" if is_13c else ", Decoupled")
                else:
                    kind = nuc
                    detail = f"I = {spin_str}" + ("" if nuc == '1H' else ", Coupled")
                spec_title = f"Simulated {kind} ({detail}) NMR Spectrum{suffix} ({args.freq:.0f} MHz)"
                fig = plot_1d_spectrum(ppm_axis, intensity, title=spec_title, nucleus=nuc,
                                       opts=self.plot_options())
                self.save_figure(fig, self._plot_file(source, tag))
            if args.save_xy:
                path = self._data_file(source, tag, '.dat')
                np.savetxt(path, np.column_stack([ppm_axis, intensity]), header="ppm intensity", fmt="%.5f %.6e")
                self.saved(path, '1D spectrum')

        valid_nuclei = [n for n in nuclei if n in nuc_data_dict]
        if not valid_nuclei or not want_2d_output:
            return

        # 2. 2D homonuclear simulations
        for nuc in valid_nuclei if (args.cosy or args.noesy) else []:
            data = nuc_data_dict[nuc]
            shifts_ppm, j_mat = data['shifts_ppm'], data['j_mat']
            group_counts, group_dist = data['group_counts'], data['group_dist']
            trace_1d = trace_1d_dict.get(nuc)
            n_low = nuc.lower()

            if args.cosy:
                print_info(f"Computing 2D {nuc}-{nuc} COSY spectrum...")
                nuc_range = NUCLEUS_DATA.get(nuc, {}).get('default_range', (-100.0, 100.0))
                f1, f2, cosy_mat = simulate_cosy(shifts_ppm, j_mat, group_counts=group_counts,
                                                 ppm_range=nuc_range if not shifts_ppm else None,
                                                 j_cutoff=args.j_cutoff)
                self._emit_2d(source, f"cosy_{n_low}_{n_low}", f1, f2, cosy_mat,
                              title=f"Simulated {nuc}-{nuc} COSY Spectrum{suffix}", spectrum_type='COSY',
                              nucleus=nuc, trace_1d=trace_1d)

            # NOESY: homonuclear 1H-1H only
            if args.noesy and (nuc == '1H' or data['element'] == 'H'):
                print_info(f"Computing 2D {nuc}-{nuc} NOESY spectrum (r_cutoff = {args.r_cutoff:.1f} Å)...")
                nuc_range = NUCLEUS_DATA.get(nuc, {}).get('default_range', (-0.5, 14.0))
                f1, f2, noesy_mat = simulate_noesy(shifts_ppm, group_dist, group_counts=group_counts,
                                                   ppm_range=nuc_range if not shifts_ppm else None,
                                                   r_cutoff=args.r_cutoff, nucleus=nuc, spin=spin_by_nuc[nuc])
                self._emit_2d(source, f"noesy_{n_low}_{n_low}", f1, f2, noesy_mat,
                              title=f"Simulated {nuc}-{nuc} NOESY Spectrum{suffix}", spectrum_type='NOESY',
                              nucleus=nuc, trace_1d=trace_1d)

        # 3. 2D heteronuclear simulations (all nucleus pairs)
        if not (args.hsqc or args.hmbc):
            return
        for nuc_a, nuc_b in combinations(valid_nuclei, 2):
            gamma_a = abs(NUCLEUS_DATA.get(nuc_a, {}).get('gamma', 1.0))
            gamma_b = abs(NUCLEUS_DATA.get(nuc_b, {}).get('gamma', 1.0))
            nuc_f2, nuc_f1 = (nuc_a, nuc_b) if gamma_a >= gamma_b else (nuc_b, nuc_a)
            el_f2 = extract_element_from_nucleus(nuc_f2)
            el_f1 = extract_element_from_nucleus(nuc_f1)

            atoms_f2 = [i for i, d in shifts_dict.items() if d.get('element', '').upper() == el_f2.upper()]
            atoms_f1 = [i for i, d in shifts_dict.items() if d.get('element', '').upper() == el_f1.upper()]
            shifts_f2_all = [convert_shielding_to_shift(shifts_dict[i]['iso'], nucleus=nuc_f2, ref_sigma=args.ref)
                             for i in atoms_f2]
            shifts_f1_all = [convert_shielding_to_shift(shifts_dict[j]['iso'], nucleus=nuc_f1, ref_sigma=args.ref)
                             for j in atoms_f1]

            def j_and_d(at1, at2):
                j_val = j_couplings_dict.get((at1, at2), j_couplings_dict.get((at2, at1), 0.0))
                d_val = float(np.linalg.norm(geom[at1] - geom[at2])) if len(geom) > max(at1, at2) else 999.0
                return j_val, d_val

            # Direct (1-bond) and long-range correlations
            hsqc_corrs, hmbc_corrs = [], []
            for idx1, at1 in enumerate(atoms_f1):
                for idx2, at2 in enumerate(atoms_f2):
                    j_val, d_val = j_and_d(at1, at2)
                    is_direct = (abs(j_val) > 40.0) or (d_val < 1.45)
                    if not is_direct and ((el_f1 != 'H' and el_f2 != 'H') or 'S' in (el_f1, el_f2)) and d_val < 2.05:
                        is_direct = True
                    if is_direct:
                        hsqc_corrs.append((idx2, idx1, float(abs(j_val)) if abs(j_val) > 0 else 1.0))
                    elif 0.5 <= abs(j_val) <= 30.0:
                        hmbc_corrs.append((idx2, idx1, float(j_val)))
                    elif 1.8 <= d_val <= 3.8 and abs(j_val) < 0.5:
                        hmbc_corrs.append((idx2, idx1, 5.0))

            trace_f2 = trace_1d_dict.get(nuc_f2)
            trace_f1 = trace_1d_dict.get(nuc_f1)
            range_f2 = NUCLEUS_DATA.get(nuc_f2, {}).get('default_range', (-100.0, 100.0))
            range_f1 = NUCLEUS_DATA.get(nuc_f1, {}).get('default_range', (-100.0, 100.0))
            ranges = dict(ppm_range_f2=range_f2 if not shifts_f2_all else None,
                          ppm_range_f1=range_f1 if not shifts_f1_all else None)
            pair_tag = f"{nuc_f2.lower()}_{nuc_f1.lower()}"
            is_ch = (el_f2 == 'H' and el_f1 == 'C')
            traces = dict(nucleus_f1=nuc_f1, nucleus_f2=nuc_f2, trace_1d_top=trace_f2, trace_1d_left=trace_f1)

            allow_empty_hsqc = args.all_spectra and not (el_f2 == 'H' and el_f1 == 'S')
            if args.hsqc and (hsqc_corrs or allow_empty_hsqc):
                print_info(f"Computing 2D {nuc_f2}-{nuc_f1} HSQC / direct correlation spectrum...")
                if is_ch:
                    ch_corrs = []
                    for idx1, at1 in enumerate(atoms_f1):
                        att = [idx2 for idx2, at2 in enumerate(atoms_f2)
                               if abs(j_and_d(at1, at2)[0]) > 50.0 or j_and_d(at1, at2)[1] < 1.35]
                        mult = ('C', 'CH', 'CH2', 'CH3')[min(len(att), 3)]
                        ch_corrs.extend((idx2, idx1, mult) for idx2 in att)
                    f1, f2, mat = simulate_hsqc(shifts_f2_all, shifts_f1_all, ch_corrs, **ranges)
                else:
                    f1, f2, mat = simulate_heteronuclear_2d(shifts_f2_all, shifts_f1_all, hsqc_corrs, **ranges)
                self._emit_2d(source, f"hsqc_{pair_tag}", f1, f2, mat,
                              title=f"Simulated {nuc_f2}-{nuc_f1} Direct Correlation Spectrum{suffix}",
                              spectrum_type='HSQC' if is_ch else 'COSY', **traces)

            if args.hmbc and (hmbc_corrs or hsqc_corrs or args.all_spectra):
                print_info(f"Computing 2D {nuc_f2}-{nuc_f1} HMBC / long-range correlation spectrum...")
                if is_ch:
                    f1, f2, mat = simulate_hmbc(shifts_f2_all, shifts_f1_all, hmbc_corrs, **ranges)
                else:
                    norm_corrs = [(i2, i1, max(0.2, min(1.0, abs(w) / 10.0)))
                                  for i2, i1, w in (hmbc_corrs or hsqc_corrs)]
                    f1, f2, mat = simulate_heteronuclear_2d(shifts_f2_all, shifts_f1_all, norm_corrs, **ranges)
                self._emit_2d(source, f"hmbc_{pair_tag}", f1, f2, mat,
                              title=f"Simulated {nuc_f2}-{nuc_f1} HMBC Correlation Spectrum{suffix}",
                              spectrum_type='HMBC', **traces)


def parse_args():
    return CalcNmr().build_parser().parse_args()


def main():
    CalcNmr().main()


if __name__ == '__main__':
    main()
