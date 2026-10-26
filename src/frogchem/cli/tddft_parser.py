#!/usr/bin/env python3
"""Obtain TDDFT information from output files and plot UV/ECD spectra."""

import argparse
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

from frogchem.filetypes import load_calculation
from frogchem.printer import print_info, print_warning
from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS
from frogchem.exceptions import InputError
from frogchem.io import parse_gaussian_cube, write_gaussian_cube
from frogchem.math_geometry import calculate_centroids
from frogchem.chimera_commands import chimerax_add_sphere

# ponytail: matplotlib is lazily imported inside create_plot (through frogchem.plotter)


def format_orbital_label(idx: int, homo: Optional[int] = None) -> str:
    """Format an orbital index relative to HOMO/LUMO.

    If homo is provided:
        idx == homo       -> 'H'
        idx < homo        -> f'H-{homo - idx}'
        idx == homo + 1   -> 'L'
        idx > homo + 1    -> f'L+{idx - (homo + 1)}'
    If homo is None:
        returns str(idx)
    """
    if homo is None:
        return str(idx)
    if idx == homo:
        return "H"
    elif idx < homo:
        return f"H-{homo - idx}"
    elif idx == homo + 1:
        return "L"
    else:
        return f"L+{idx - (homo + 1)}"


def format_transition(donor: int, acceptor: int, weight: float, spin: str = '', homo: Optional[int] = None, beta_homo: Optional[int] = None) -> str:
    """Format an electronic transition using frontier orbital notation (e.g. Hα->Lα (0.94))."""
    eff_homo = beta_homo if spin == 'b' and beta_homo is not None else homo
    d_label = format_orbital_label(donor, eff_homo)
    a_label = format_orbital_label(acceptor, eff_homo)
    
    spin_char = 'α' if spin == 'a' else 'β' if spin == 'b' else ''
    if spin_char:
        d_label += spin_char
        a_label += spin_char
        
    return f"{d_label}->{a_label} ({weight:.2f})"


#: Letter used in state labels for each spin multiplicity (S1, D1, T1, Q1 ...).
MULT_LETTER = {1: 'S', 2: 'D', 3: 'T', 4: 'Q'}
_LETTER_MULT = {v: k for k, v in MULT_LETTER.items()}


def state_label(multiplicity: int, number: int) -> str:
    """Label of an excited state: S1, T2, D1 ... (M5-1 for multiplicities without a letter).

    The number is the root index printed by the program within its multiplicity block, so
    singlets and triplets are numbered independently (S1, S2, ..., T1, T2, ...) and S0 is the
    ground state, as in the SOC matrix and in --dct-states.
    """
    letter = MULT_LETTER.get(int(multiplicity))
    return f"{letter}{number}" if letter else f"M{multiplicity}-{number}"


def prepare_states_table(states: Dict[int, Any], threshold: float, homo: Optional[int] = None, beta_homo: Optional[int] = None) -> List[List[Any]]:
    table_rows = []
    for multiplicity, state_data in states.items():
        for state, details in state_data.items():
            formatted_trans = ', '.join(
                format_transition(t[0], t[1], t[2], spin=t[3] if len(t) > 3 else '', homo=homo, beta_homo=beta_homo)
                for t in details.get("population", []) if t[2] > threshold
            )
            table_rows.append([
                state_label(multiplicity, state), multiplicity, details.get('energy', 0.0),
                details.get('fosc', 0.0), details.get('R', 0.0),
                details.get('S**2', 0.0), formatted_trans
            ])
    return table_rows


def prepare_soc_table(states: Dict[int, Any], soc: Any) -> Tuple[List[str], List[List[Any]]]:
    if soc is None or (isinstance(soc, np.ndarray) and soc.size == 0):
        return [], []

    num_cols = soc.shape[1] if hasattr(soc, 'shape') else (len(soc[0]) if soc else 0)
    headers = ['State'] + [f'T{i+1}' for i in range(num_cols)]
    rows = [[f'S{idx}'] + [round(float(v), 3) for v in row] for idx, row in enumerate(soc)]
    return headers, rows


def parse_state_selection(
    soc_states: Optional[List[str]] = None,
    soc_singlets: Optional[List[int]] = None,
    soc_triplets: Optional[List[int]] = None
) -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
    """Parse singlet and triplet indices from state selection arguments."""
    sel_s: Optional[Set[int]] = set(soc_singlets) if soc_singlets is not None else None
    sel_t: Optional[Set[int]] = set(soc_triplets) if soc_triplets is not None else None

    if soc_states:
        if sel_s is None:
            sel_s = set()
        if sel_t is None:
            sel_t = set()

        for token in soc_states:
            token = token.strip()
            # Range syntax like S0-S3 or T1-T5
            if '-' in token and not token.startswith('-'):
                parts = token.split('-', 1)
                p1, p2 = parts[0].strip(), parts[1].strip()
                if (p1.upper().startswith('S') and p2.upper().startswith('S')) or (p1.upper().startswith('T') and p2.upper().startswith('T')):
                    prefix = p1[0].upper()
                    try:
                        start_idx = int(p1[1:])
                        end_idx = int(p2[1:])
                        step = 1 if end_idx >= start_idx else -1
                        indices = set(range(start_idx, end_idx + step, step))
                        if prefix == 'S':
                            sel_s.update(indices)
                        else:
                            sel_t.update(indices)
                        continue
                    except ValueError:
                        pass

            upper = token.upper()
            if upper.startswith('S'):
                try:
                    sel_s.add(int(upper[1:]))
                    continue
                except ValueError:
                    pass
            elif upper.startswith('T'):
                try:
                    sel_t.add(int(upper[1:]))
                    continue
                except ValueError:
                    pass

            try:
                val = int(token)
                if val == 0:
                    sel_s.add(0)
                else:
                    sel_s.add(val)
                    sel_t.add(val)
            except ValueError:
                pass

        # If user only specified S or only T, keep the other as None (meaning all)
        s_specified = any(tok.strip().upper().startswith('S') or tok.strip() == '0' for tok in soc_states)
        t_specified = any(tok.strip().upper().startswith('T') for tok in soc_states)
        bare_ints = any(tok.strip().isdigit() and int(tok.strip()) > 0 for tok in soc_states)

        if s_specified and not t_specified and not bare_ints and soc_triplets is None:
            sel_t = None
        if t_specified and not s_specified and not bare_ints and soc_singlets is None:
            sel_s = None

    return sel_s, sel_t


def filter_soc_table(
    headers: List[str],
    rows: List[List[Any]],
    sel_s: Optional[Set[int]],
    sel_t: Optional[Set[int]]
) -> Tuple[List[str], List[List[Any]]]:
    """Filter SOC table columns and rows according to selected state indices."""
    if not headers or not rows or (sel_s is None and sel_t is None):
        return headers, rows

    col_indices = [0]
    for col_idx, h in enumerate(headers[1:], start=1):
        try:
            t_num = int(h.lstrip('Tt'))
        except ValueError:
            t_num = col_idx
        if sel_t is None or t_num in sel_t:
            col_indices.append(col_idx)

    filt_headers = [headers[i] for i in col_indices]

    filt_rows = []
    for row in rows:
        label = str(row[0])
        try:
            s_num = int(label.lstrip('Ss'))
        except ValueError:
            s_num = None
        if sel_s is None or (s_num is not None and s_num in sel_s):
            filt_rows.append([row[i] for i in col_indices])

    return filt_headers, filt_rows


def parse_dct_state_selection(tokens: Optional[List[str]]) -> Optional[Dict[Optional[int], Set[int]]]:
    """Parse state selection for DCT analysis (e.g. S1 T1, S1-S3, T1, 1)."""
    if not tokens:
        return None
    selected: Dict[Optional[int], Set[int]] = {m: set() for m in MULT_LETTER}
    selected[None] = set()
    specified_mults: Set[int] = set()

    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # Check range e.g. S1-S3 or T1-T5 or D1-D3
        if '-' in token and not token.startswith('-'):
            parts = token.split('-', 1)
            p1, p2 = parts[0].strip().upper(), parts[1].strip().upper()
            if len(p1) > 1 and len(p2) > 1 and p1[0] == p2[0] and p1[0] in _LETTER_MULT:
                prefix = p1[0]
                mult = _LETTER_MULT[prefix]
                specified_mults.add(mult)
                try:
                    s_start, s_end = int(p1[1:]), int(p2[1:])
                    step = 1 if s_end >= s_start else -1
                    selected[mult].update(range(s_start, s_end + step, step))
                    continue
                except ValueError:
                    pass

        upper = token.upper()
        if upper[:1] in _LETTER_MULT and upper[1:].isdigit():
            mult = _LETTER_MULT[upper[0]]
            specified_mults.add(mult)
            selected[mult].add(int(upper[1:]))
        elif token.isdigit():
            val = int(token)
            selected[None].add(val)
        else:
            m_match = re.match(r'^M(\d+)[_-](\d+)$', upper)
            if m_match:
                m_val, s_val = int(m_match.group(1)), int(m_match.group(2))
                specified_mults.add(m_val)
                selected.setdefault(m_val, set()).add(s_val)

    res: Dict[Optional[int], Set[int]] = {}
    if specified_mults:
        for m in specified_mults:
            if selected.get(m):
                res[m] = selected[m]
    if selected[None]:
        res[None] = selected[None]
    return res if res else None


def gaussian_convolution(x_grid: np.ndarray, centers: np.ndarray, intensities: np.ndarray, fwhm: float) -> np.ndarray:
    """Sum of area-normalised Gaussians (FWHM in the units of x_grid) weighted by intensities."""
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise InputError(f"The Gaussian FWHM must be a positive number, got {fwhm}")
    mask = intensities != 0
    valid_centers, valid_ints = centers[mask], intensities[mask]
    
    if not valid_centers.size:
        return np.zeros_like(x_grid)
    
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    
    dx = x_grid[:, None] - valid_centers[None, :]
    gaussians = norm * np.exp(-0.5 * (dx / sigma)**2)
    
    return np.sum(valid_ints * gaussians, axis=1)


def extract_peaks(states: Dict[int, Any], target_key: str, shift: float) -> Tuple[np.ndarray, np.ndarray]:
    """(energies + shift, intensities) of the states with a non-zero `target_key`.

    Returns two empty arrays when no state carries a non-zero intensity
    (e.g. triplet-only runs or --ecd without rotatory strengths).
    """
    peaks = [
        (details.get('energy', 0.0) + shift, details.get(target_key, 0.0))
        for state_data in (states or {}).values()
        for details in state_data.values()
        if details.get(target_key, 0.0) != 0.0
    ]
    if not peaks:
        return np.array([], dtype=float), np.array([], dtype=float)
    energies, intensities = zip(*peaks)
    return np.array(energies, dtype=float), np.array(intensities, dtype=float)


def ecd_band_intensities(energies: np.ndarray, rotatory_strengths: np.ndarray) -> np.ndarray:
    """Band weights for the ECD curve: Δε(ν̃) ∝ Σ_k ν̃_k R_k g(ν̃ − ν̃_k).

    The rotatory strength enters Δε weighted by the transition energy
    (Δε = ν̃_k R_k / (22.96 σ√(2π)) exp(...) with R in 1e-40 cgs and ν̃ in cm⁻¹),
    unlike ε, which is proportional to f_k alone. Energies may be in any unit
    (eV here): the curves are normalised, so only the ratios matter.
    """
    return np.asarray(rotatory_strengths, dtype=float) * np.asarray(energies, dtype=float)


def ev_to_nm(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return np.where(x == 0, np.inf, 1239.84193 / x)


def nm_to_ev(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore'):
        return np.where(x == 0, np.inf, 1239.84193 / x)




_HC_NM_EV = 1239.84193


def classify_ct(delta_r: Optional[float], le_thr: float, ct_thr: float) -> str:
    """Classify an excitation based on Δr thresholds."""
    if delta_r is None:
        return ''
    if delta_r < le_thr:
        return 'Local Exc.'
    elif delta_r > ct_thr:
        return 'Charge Trans.'
    return 'Mixed CT/LE'


def normalize_spectra(y_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Scale all curves by the same factor so the largest |intensity| is 1."""
    global_max = max(np.max(np.abs(y)) for y in y_data.values())
    global_max = global_max if global_max > 0 else 1.0
    return {label: y / global_max for label, y in y_data.items()}


def create_plot(x_ev: np.ndarray, y_data: Dict[str, np.ndarray], graph_type: str,
                x_start: float, x_end: float, superimposed: bool = False, opts=None):
    """Plot normalized UV/ECD curves on a wavelength axis (energy on top). Returns the figure.

    opts: frogchem.plotter.PlotOptions (title, colors, limits in nm, ...); they win over the defaults.
    """
    from frogchem import plotter
    opts = (opts or plotter.PlotOptions()).with_defaults(
        title=f"Superimposed {graph_type} Spectra" if superimposed else f"Computed {graph_type} Spectrum",
        xlabel="Wavelength (nm)", ylabel="Normalized Intensity", grid=True, legend=superimposed,
        xlim=(ev_to_nm(x_end), ev_to_nm(x_start)),
    )
    curves = {label if superimposed else f"{graph_type} Spectrum": y
              for label, y in normalize_spectra(y_data).items()}
    with plotter.style(opts):
        fig, ax_nm = plotter.line(ev_to_nm(x_ev), curves, opts=opts)
        ax_ev = ax_nm.secondary_xaxis('top', functions=(nm_to_ev, ev_to_nm))
        ax_ev.set_xlabel("Energy (eV)")
        # room for the energy axis on top
        ax_nm.set_title(ax_nm.get_title(), pad=20)
    return fig


def export_csv(x_ev: np.ndarray, y_data: Dict[str, np.ndarray], out_file) -> Path:
    """Write the plotted (normalized) curves: Energy_eV, Wavelength_nm, one column per curve."""
    headers = ["Energy_eV", "Wavelength_nm"] + list(y_data.keys())
    data_matrix = np.column_stack([x_ev, ev_to_nm(x_ev)] + list(normalize_spectra(y_data).values()))
    np.savetxt(out_file, data_matrix, delimiter=",", header=",".join(headers), comments='')
    return Path(out_file)


def _centroid_commands(res: Dict[str, Any], comment: str) -> List[str]:
    if res['positive_centroid'] is None:
        return []
    return [f"# {comment}",
            chimerax_add_sphere(res['positive_centroid'], color="blue", radius=0.5),
            chimerax_add_sphere(res['negative_centroid'], color="red", radius=0.5)]


def process_cube_dct(filepath: str, args) -> Tuple[List[Any], List[str]]:
    """Barycenters and charge-transfer distance of a CUBE density difference.

    Returns the table row and the ChimeraX commands for the centroid spheres.
    """
    data = parse_gaussian_cube(filepath)
    res = calculate_centroids(data, threshold_percentile=args.dct_threshold)

    ct_dist = res['charge_transfer_distance']
    label = classify_ct(ct_dist, args.le_threshold, args.ct_threshold)
    dist_str = f"{ct_dist:.2f} ({label})" if ct_dist is not None else 'N/A'
    dipole_val = (np.linalg.norm(res['dipole_moment'])
                  if res['dipole_moment'] is not None else None)

    row = [
        filepath,
        dist_str,
        f"{dipole_val:.4f}" if dipole_val is not None else 'N/A',
        str(res['data']['grid_size']),
    ]
    return row, _centroid_commands(res, Path(filepath).name)


def process_calc_dct(calc, file_path: str, args, cmd: CLICommand) -> List[str]:
    """Excited-state difference densities and DCT analysis of a TD-DFT calculation.

    Emits the DCT table through cmd.output, writes one cube per state (unless --no-save-cube)
    and returns the ChimeraX commands for the centroid spheres.
    """
    from frogchem.density_builder import (
        build_basis_table,
        generate_grid,
        eval_all_basis,
        compute_difference_density_on_points,
        _make_cube_data,
        _AU,
    )

    has_mos = (hasattr(calc, 'orbital_coefficients')
               and calc.orbital_coefficients is not None
               and hasattr(calc, 'basis_coeff')
               and calc.basis_coeff)
    if not has_mos:
        print_warning(
            f"No MO/basis data in {file_path}: DCT skipped. "
            "To compute difference densities add '! PrintMOs PrintBasis' to the ORCA input "
            "or 'pop=full gfinput' to the Gaussian route"
        )
        return []

    print_info(f"Computing DCT analysis for {file_path}")
    geom_bohr = calc.geom * _AU
    table = build_basis_table(calc, geom_bohr)
    grid_3d, axes, (nx, ny, nz) = generate_grid(
        geom_bohr, n_voxels=args.voxels, padding=args.padding * _AU,
    )
    points = grid_3d.reshape(-1, 3)
    n_pts = len(points)
    print_info(f"  grid: {nx}×{ny}×{nz} = {n_pts:,} pts, "
               f"basis funcs: {len(table)}, "
               f"MOs: {calc.orbital_coefficients.shape[1]}")

    print_info("  evaluating basis functions on grid …")
    bf = eval_all_basis(table, points)

    dct_filter = parse_dct_state_selection(args.dct_states)

    table_rows = []
    commands: List[str] = []
    for mult in sorted(calc.tddft_states.keys()):
        states = calc.tddft_states[mult]

        for state_num in sorted(states.keys()):
            if dct_filter is not None:
                if mult in dct_filter:
                    if state_num not in dct_filter[mult]:
                        continue
                elif None in dct_filter:
                    if state_num not in dct_filter[None]:
                        continue
                else:
                    continue
            state_data = states[state_num]
            transitions = state_data.get('population', [])
            energy_ev = state_data.get('energy', 0.0)
            wavelength_nm = (_HC_NM_EV / energy_ev if energy_ev > 0 else float('inf'))
            fosc = state_data.get('fosc', 0.0)
            state_tag = state_label(mult, state_num)

            if not transitions:
                table_rows.append([
                    state_tag,
                    f"{energy_ev:.2f}",
                    f"{wavelength_nm:.1f}",
                    f"{fosc:.4f}",
                    'N/A',
                    'N/A',
                ])
                continue

            print_info(f"  {state_tag}: computing Δρ …")
            delta_rho = compute_difference_density_on_points(
                calc, transitions, points, bf_matrix=bf,
            )
            data_3d = delta_rho.reshape(nx, ny, nz)

            cube_data = _make_cube_data(
                data_3d, axes, calc,
                title=f'Difference Density {state_tag} from {file_path}',
                comment=f'Delta rho = rho_exc - rho_GS, Grid: {nx}x{ny}x{nz}',
            )

            if args.save_cube:
                cube_path = cmd.derived_path(file_path, f"{state_tag}_delta_rho", '.cube')
                write_gaussian_cube(filename=str(cube_path), cube_data=cube_data)
                cmd.saved(cube_path, 'cube')

            res = calculate_centroids(
                cube_data, threshold_percentile=args.dct_threshold,
            )
            ct_dist = res['charge_transfer_distance']
            dipole_raw = res['dipole_moment']
            dipole_val = (f"{np.linalg.norm(dipole_raw):.4f}"
                          if dipole_raw is not None else 'N/A')

            if ct_dist is not None:
                label = classify_ct(ct_dist, args.le_threshold, args.ct_threshold)
                dist_str = f"{ct_dist:.2f} ({label})"
            else:
                dist_str = 'N/A'

            table_rows.append([
                state_tag,
                f"{energy_ev:.2f}",
                f"{wavelength_nm:.1f}",
                f"{fosc:.4f}",
                dist_str,
                dipole_val,
            ])
            commands += _centroid_commands(res, state_tag)

    del bf

    cmd.output(
        table_rows,
        headers=['State', 'Energy (eV)', 'Wavelength (nm)',
                 'Osc. Strength (f)', 'Δr (Å)', 'Dipole Moment'],
        title=f"Charge Transfer Distance Analysis ({Path(file_path).name})",
    )
    return commands


_SPECTRA = (('uv', 'UV', 'fosc'), ('ecd', 'ECD', 'R'))


class TddftParser(CLICommand, name='frog-tddft'):
    """Summarise TD-DFT excited states, SOC and charge transfer, and plot UV/ECD spectra.

    Reads ORCA or Gaussian outputs: one table of excited states per file (energy,
    oscillator and rotatory strength, <S²>, dominant orbital transitions) and the SOC
    matrix when present. --dct computes the excited-state density differences on a grid
    and the charge-transfer distance Δr; CUBE inputs are analysed directly. --uv/--ecd
    broaden the transitions with Gaussians and save the spectra; with -c the excited
    states, SOC and spectra are Boltzmann averaged over the conformers.
    """
    category = 'spectroscopy'
    supports_ensemble = True
    _auto_ensemble_average = False
    outputs = [
        Output('table', 'Excited States Summary (<file>)',
               'One row per state: multiplicity, energy (eV), f, R (1e-40 cgs), <S²> and the '
               'orbital transitions with weight above -t'),
        Output('table', 'Spin-Orbit Coupling (SOC) Matrix (<file>)',
               'SOC between singlets (rows) and triplets (columns), cm⁻¹; only for SOC calculations, '
               'filtered by --soc-states/--soc-singlets/--soc-triplets'),
        Output('table', 'Charge Transfer Distance Analysis (<file>)',
               'Per state: energy, wavelength, f, Δr (Å) with CT/LE class, CT dipole q_CT·Δr (e·Å)', when='--dct'),
        Output('table', 'Charge Transfer Distance (CUBE)', 'Δr (Å), CT dipole q_CT·Δr (e·Å) and grid size of each CUBE input'),
        ENSEMBLE_WEIGHTS,
        Output('table', 'Ensemble-Averaged Excited States (Boltzmann Weighted)',
               'Weighted energy, f, R and <S²> per state, transitions of the most populated conformer',
               when='-c'),
        Output('table', 'Ensemble-Averaged Spin-Orbit Coupling (SOC) Matrix (Boltzmann Weighted)',
               'Weighted SOC matrix (cm⁻¹)', when='-c'),
        Output('file', '<stem>_<state>_delta_rho.cube', 'Difference density of each state '
                                                        '(not with --no-save-cube)', when='--dct'),
        Output('file', '<stem>_dct.cxc', 'ChimeraX script with the +/− centroid spheres of Δρ '
                                         '(blue/red), for --dct or CUBE inputs', when='--chimerax'),
        Output('file', '<plot>.csv', 'Normalized curves of every saved spectrum plot, same name: '
                                     'Energy_eV, Wavelength_nm, one column per curve', when='--save-spectra'),
        Output('file', '<plot>.pickle', 'Pickled matplotlib figure of every saved spectrum plot',
               when='--save-pickle'),
        Output('plot', '<stem>_UV.png, <stem>_ECD.png', 'Spectrum of each file, next to it (or --out-dir)',
               when='--uv/--ecd'),
        Output('plot', '<prefix>_UV_superimposed.png', 'All files on one plot, only with 2 or more files; '
                                                       'prefix from -o (spectra, in the current directory)',
               when='--uv/--ecd'),
        Output('plot', '<prefix>_UV_ensemble.png', 'Boltzmann-averaged spectrum (ECD likewise)',
               when='-c --uv/--ecd'),
    ]
    guide = Guide(
        why="""
            A TD-DFT calculation gives the vertical excited states of a molecule: their energies, how bright
            they are in absorption (oscillator strength f), their sign in circular dichroism (rotatory
            strength R) and which orbitals are involved. This command turns the long ORCA/Gaussian output into
            one readable table, draws the **UV-Vis** and **ECD** spectra (to compare with experiment or to
            assign the absolute configuration from ECD), shows the **spin-orbit couplings** (SOC) between
            singlets and triplets (relevant for intersystem crossing, phosphorescence, TADF), and measures
            whether an excitation is **local** or **charge transfer** (CT) with the D_CT index (distance
            between the regions where electron density is lost and gained). With a conformer ensemble it
            Boltzmann-averages states and spectra.
            """,
        needs="""
            An **ORCA output of a TD-DFT (or TDA) calculation**, e.g. `! B3LYP def2-TZVP` with
            `%tddft nroots 10 end` (more roots = wider spectrum; 10–30 is typical for UV-Vis). Rotatory
            strengths for ECD are printed by ORCA automatically. For triplets add `triplets true`, for SOC
            `dosoc true` inside `%tddft`. A **Gaussian 16** output with `td=(nstates=10)` is also read
            (energies, f, R and transitions; with `td=(50-50,nstates=10)` singlets and triplets are numbered
            S1…, T1… as in ORCA; checked against ORCA `tda false` on (R)-methyloxirane).

            For `--dct` the output must also contain the basis set and all the MO coefficients: ORCA
            `! PrintBasis PrintMOs`; Gaussian `#p ... pop=full iop(3/33=1) iop(6/80=1) gfinput` (pure `5D 7F` or
            Cartesian `6D 10F` functions). Alternatively give an existing difference-density cube
            (`.cube`, e.g. written by an earlier `--dct` run) and it is analysed directly. For `-c` you need a
            `checkpoint.json` (or csv/json) with Boltzmann populations of the conformers.
            """,
        how=[
            "Run `frog-tddft job.out` to print the excited-state table (and the SOC matrix if the job had "
            "`dosoc true`); lower `-t` (default 0.20) to see more orbital transitions per state, "
            "`--raw-orbitals` to see MO numbers instead of H/L labels.",
            "Plot spectra with `--uv` and/or `--ecd`: the files `<stem>_UV.png` / `<stem>_ECD.png` go next to the "
            "output (or `--out-dir`). Adjust the band width with `--fwhm` (eV, default 0.3), shift all "
            "energies with `--shift` (eV, e.g. −0.2 to match experiment), set the window with `--x-start`/"
            "`--x-end` (eV, default 2–8 eV = 620–155 nm), and add `--save-spectra` to get the curves as CSV.",
            "Several files (e.g. functionals, or conformers) → one plot each plus a superimposed plot "
            "`spectra_UV_superimposed.png` (prefix from `-o`); with `-c checkpoint.json` the states and the "
            "spectrum are Boltzmann-averaged (`spectra_UV_ensemble.png`).",
            "Charge-transfer analysis: `--dct` for all states, or `--dct --dct-states S1-S3 T1` for some of "
            "them, with the same labels as the table; `--voxels` (default 60) sets the grid, `--chimerax` writes "
            "a ChimeraX script with the +/− centroids, `--no-save-cube` skips the Δρ cube files.",
            "Restrict the SOC matrix to the states you care about with `--soc-states S0 S1 T1 T2` "
            "(ranges like `T1-T5` work).",
        ],
        expect="""
            **Excited States Summary**: one row per state with its label, multiplicity, **Energy (eV)**,
            **f** (dimensionless), **R** (rotatory strength, 10⁻⁴⁰ cgs, length gauge for both ORCA and
            Gaussian: Gaussian's `R(length)` column, not `R(velocity)`), **<S\\*\\*2>** and the orbital
            transitions with their weight, e.g. `H->L (0.99)` (H = HOMO, L = LUMO, H-1 = HOMO−1; α/β marks the
            spin in open-shell runs). States are labelled per multiplicity: **S1, S2… T1, T2… D1… Q1…**
            (S0 is the ground state), and the same labels are used in the SOC matrix and in `--dct-states`.
            To convert: λ (nm) = 1239.84 / E (eV).

            * f > 0.1 is a bright transition, f < 0.01 essentially dark. R has a sign: its pattern
              (+/−) is what you compare with an experimental ECD spectrum; for an achiral molecule R ≈ 0.
            * <S\\*\\*2> should be 0 for singlets and 2 for triplets (closed-shell reference); in open-shell
              (UKS) runs a value far from S(S+1) means spin contamination: be careful with that state.
            * **SOC matrix**: rows S0, S1…, columns T1, T2…, values in cm⁻¹. ⟨S|H_SO|T⟩ of a few cm⁻¹ is
              typical for organic molecules; tens to hundreds of cm⁻¹ appear with heavy atoms or n→π*/π→π*
              (El-Sayed) state pairs.
            * Spectra are **normalised** to a maximum of 1 (shape only, no ε in L mol⁻¹ cm⁻¹), on a wavelength
              axis with energy on top. The ECD bands are weighted by the transition energy (Δε ∝ E·R).
              TD-DFT with hybrid functionals typically overestimates energies by 0.1–0.4 eV, hence `--shift`.
            * **D_CT** (Δr, Å): **< 1.5 Å** local excitation (*Local Exc.*), **> 2 Å** charge transfer
              (*Charge Trans.*), in between *Mixed CT/LE* (change with `--le-threshold`/`--ct-threshold`). The
              *Dipole Moment* column is q_CT·Δr in e·Å. A grid that is too coarse or a too small `--padding`
              makes Δr noisy: compare states computed with the same grid.
            * An output whose TD-DFT part did not finish has no states: no table is printed (the UV/ECD
              plots are then empty). Check that the job ended normally.

            Example (`soc.out`, PBE0/def2-TZVPP, `nroots 1`, `triplets true`, `dosoc true`): S1 2.778 eV,
            f 0.080, R 20.696, `H->L (0.99)`; T1 2.506 eV, <S\\*\\*2> 2.000; SOC S0–T1 6.857 cm⁻¹ and S1–T1
            1.040 cm⁻¹. Water (B3LYP/def2-TZVPP, `--dct --voxels 40`): S1 7.577 eV, f 0.034, Δr 0.86 Å
            (Local Exc.).
            """,
    )
    examples = [
        ('frog-tddft tddft.out -t 0.1', 'Excited states with transitions of weight above 0.1'),
        ('frog-tddft conf_*/tddft.out --uv --ecd --fwhm 0.25 --shift -0.2 --save-spectra',
         'Per-file and superimposed spectra, plus their curves as CSV'),
        ('frog-tddft tddft.out --dct --dct-states S1-S3 T1 --chimerax --out-dir dct',
         'Charge-transfer distance of S1-S3 and T1, cubes and ChimeraX script in dct/'),
    ]

    def setup_args(self, parser):
        parser.add_argument('files', nargs='+', type=Path, help='ORCA/Gaussian output files or CUBE files')
        parser.add_argument('-t', '--threshold', type=float, default=0.20,
                            help='Minimum weight of an orbital transition to list it')
        parser.add_argument('--raw-orbitals', action='store_true',
                            help='Show orbital numbers instead of frontier labels (H, L, H-1, ...)')

        parser.add_argument('--soc-states', nargs='+', default=None,
                            help='States to show in the SOC matrix (e.g. S0 S1 T1 T2, S0-S3, T1-T5)')
        parser.add_argument('--soc-singlets', nargs='+', type=int, default=None,
                            help='Singlet indices to show in the SOC matrix (e.g. 0 1)')
        parser.add_argument('--soc-triplets', nargs='+', type=int, default=None,
                            help='Triplet indices to show in the SOC matrix (e.g. 1 2)')

        spectra = parser.add_argument_group('spectra')
        spectra.add_argument('--uv', action='store_true', help='Plot the UV-Vis spectrum')
        spectra.add_argument('--ecd', action='store_true', help='Plot the ECD spectrum')
        spectra.add_argument('--shift', type=float, default=0.0, help='Shift of the excitation energies in eV')
        spectra.add_argument('--fwhm', type=float, default=0.3, help='Gaussian broadening FWHM in eV')
        spectra.add_argument('--x-start', type=float, default=2.0, help='Lower end of the energy window in eV')
        spectra.add_argument('--x-end', type=float, default=8.0, help='Upper end of the energy window in eV')
        spectra.add_argument('--save-spectra', action='store_true',
                             help='Also write the plotted curves as CSV next to each spectrum plot')
        spectra.add_argument('--save-pickle', action='store_true',
                             help='Also pickle each spectrum figure (.pickle next to the plot)')

        dct = parser.add_argument_group('charge transfer (DCT)')
        dct.add_argument('--dct', action='store_true',
                         help='Compute the difference densities and the charge-transfer distance Δr')
        dct.add_argument('--dct-states', nargs='+', default=None,
                         help='States for the DCT analysis (e.g. S1 T1, S1-S3, T1); all if not given')
        dct.add_argument('--dct-threshold', type=float, default=100.0,
                         help='Density percentile kept when locating the Δρ centroids '
                              '(100 = no trimming, as in the Le Bahers D_CT definition)')
        dct.add_argument('--ct-threshold', type=float, default=2.0,
                         help='Δr in Å above which a state is labelled charge transfer')
        dct.add_argument('--le-threshold', type=float, default=1.5,
                         help='Δr in Å below which a state is labelled local excitation')
        dct.add_argument('--voxels', type=int, default=60, help='Grid points per dimension for the DCT grid')
        dct.add_argument('--padding', type=float, default=3.0, help='Grid padding around the molecule in Å')
        dct.add_argument('--no-save-cube', dest='save_cube', action='store_false',
                         help='Do not write the difference-density cube files')
        dct.add_argument('--chimerax', action='store_true',
                         help='Write the ChimeraX commands for the Δρ centroid spheres to <stem>_dct.cxc')

        self.add_output_arg(parser, default='spectra', metavar='PREFIX',
                            help='Prefix of the aggregate spectra (superimposed, ensemble average), '
                                 'relative to the current directory')
        self.add_out_dir_arg(parser)
        self.add_plot_args(parser, optional=False)

    # --------------------------------------------------------------- helpers
    def _save_spectrum(self, x_ev, y_data, graph_type, path, superimposed=False):
        """Plot, then save the figure (and optional pickle / CSV twins with the same name)."""
        fig = create_plot(x_ev, y_data, graph_type, self.args.x_start, self.args.x_end,
                          superimposed=superimposed, opts=self.plot_options())
        if self.args.save_pickle:
            pkl = path.with_suffix('.pickle')
            with open(pkl, 'wb') as f:
                pickle.dump(fig, f)
            self.saved(pkl, 'figure pickle')
        if self.args.save_spectra:
            self.saved(export_csv(x_ev, y_data, path.with_suffix('.csv')), 'spectrum data')
        self.save_figure(fig, path)

    def _spectrum(self, calc, key, x_ev):
        energies, intensities = extract_peaks(calc.tddft_states, key, self.args.shift)
        if key == 'R':
            # Δε ∝ ν̃_k R_k: weight each rotatory strength by its (shifted) transition energy
            intensities = ecd_band_intensities(energies, intensities)
        return gaussian_convolution(x_ev, energies, intensities, self.args.fwhm)

    def _write_chimerax(self, input_path, commands):
        if not (self.args.chimerax and commands):
            return
        path = self.derived_path(input_path, 'dct', '.cxc')
        path.write_text(f"# Δρ centroids of {Path(input_path).name} (blue: +, red: −)\n"
                        + "\n".join(commands) + "\n", encoding='utf-8')
        self.saved(path, 'ChimeraX script')

    def _soc_table(self, headers, rows, title, where):
        has_filter = bool(self.args.soc_states or self.args.soc_singlets is not None
                          or self.args.soc_triplets is not None)
        if has_filter:
            sel_s, sel_t = parse_state_selection(self.args.soc_states, self.args.soc_singlets,
                                                 self.args.soc_triplets)
            headers, rows = filter_soc_table(headers, rows, sel_s, sel_t)
        if rows and len(headers) > 1:
            self.output(rows, headers=headers, title=title)
        elif has_filter:
            print_info(f"No matching SOC states found for the selection in {where}.")

    # ------------------------------------------------------------------- run
    def run(self, args):
        x_ev = np.linspace(args.x_start, args.x_end, num=1000)
        files = self.require_files(args.files)
        confs_data = []
        cube_rows = []

        for p in files:
            file_path = str(p)
            if p.suffix.lower() == '.cube' or file_path.endswith('.cube.gz'):
                row, commands = process_cube_dct(file_path, args)
                cube_rows.append(row)
                self._write_chimerax(p, commands)
                continue

            print_info(f"Processing file: {file_path}")
            try:
                calc = load_calculation(file_path)
            except Exception as e:
                print_warning(f"Failed to parse {file_path}: {e}")
                continue

            if not getattr(calc, 'tddft', False):
                print_warning(f"{file_path} does not contain a TDDFT section.")
                continue

            confs_data.append({'path': p, 'file_path': file_path, 'calc': calc})

        if cube_rows:
            self.output(cube_rows, headers=['File', 'Δr (Å)', 'Dipole Moment', 'Grid Size'],
                        title="Charge Transfer Distance (CUBE)")

        if not confs_data:
            if cube_rows and len(cube_rows) == len(files):
                return
            raise InputError("No valid TDDFT calculations found in the provided files.")

        if args.checkpoint:
            self._run_ensemble_average(confs_data, args, x_ev)
            if not args.individual:
                return

        spectra = {name: {} for _, name, _ in _SPECTRA}
        for item in confs_data:
            file_path = item['file_path']
            calc = item['calc']

            homo = None if args.raw_orbitals else getattr(calc, 'homo', None)
            beta_homo = getattr(calc, 'beta_homo', None)
            state_rows = prepare_states_table(calc.tddft_states, args.threshold, homo=homo, beta_homo=beta_homo)
            self.output(state_rows,
                        headers=["State", "Mult", "Energy (eV)", 'f', 'R (1e-40 cgs)', '<S**2>', "Population"],
                        title=f"Excited States Summary ({file_path})")

            if getattr(calc, 'soc', False) and hasattr(calc, 'tddft_soc'):
                soc_headers, soc_rows = prepare_soc_table(calc.tddft_states, calc.tddft_soc)
                if soc_rows:
                    self._soc_table(soc_headers, soc_rows,
                                    f"Spin-Orbit Coupling (SOC) Matrix ({file_path})", file_path)
                else:
                    print_warning(f"SOC data found in {file_path} but no SOC table could be built.")

            if args.dct or args.dct_states:
                self._write_chimerax(item['path'], process_calc_dct(calc, file_path, args, self))

            for flag, name, key in _SPECTRA:
                if getattr(args, flag):
                    y = self._spectrum(calc, key, x_ev)
                    spectra[name][file_path] = y
                    self._save_spectrum(x_ev, {file_path: y}, name,
                                        self.derived_path(item['path'], name, args.fig_format))

        for _, name, _ in _SPECTRA:
            if len(spectra[name]) >= 2:
                print_info(f"Generating superimposed {name} spectra...")
                self._save_spectrum(x_ev, spectra[name], name,
                                    self.plot_path('spectra', tag=f'{name}_superimposed'), superimposed=True)

    def _run_ensemble_average(self, confs_data: List[Dict[str, Any]], args: argparse.Namespace, x_ev: np.ndarray):
        """Compute Boltzmann-weighted averages of excited states and spectra."""
        print_info(f"Loading Boltzmann populations from checkpoint: {args.checkpoint}")
        try:
            ens = self.build_ensemble(confs_data, path_extractor=lambda x: x['file_path'])
        except Exception as e:
            raise InputError(f"Failed to load checkpoint {args.checkpoint}: {e}") from e

        self.output_weights(ens)

        weights = ens.weights

        # 1. Boltzmann Averaged Excited States
        all_state_keys = []
        for item in confs_data:
            calc = item['calc']
            for mult, state_data in calc.tddft_states.items():
                for st in state_data.keys():
                    if (mult, st) not in all_state_keys:
                        all_state_keys.append((mult, st))
        all_state_keys.sort(key=lambda x: (x[0], x[1]))

        avg_rows = []
        for mult, st in all_state_keys:
            w_sum = 0.0
            weighted_e = 0.0
            weighted_f = 0.0
            weighted_r = 0.0
            weighted_s2 = 0.0
            dominant_trans = ""
            max_w = -1.0

            for item, w in zip(confs_data, weights):
                calc = item['calc']
                details = calc.tddft_states.get(mult, {}).get(st)
                if details is not None:
                    w_sum += w
                    weighted_e += w * details.get('energy', 0.0)
                    weighted_f += w * details.get('fosc', 0.0)
                    weighted_r += w * details.get('R', 0.0)
                    weighted_s2 += w * details.get('S**2', 0.0)
                    if w > max_w:
                        max_w = w
                        homo = None if args.raw_orbitals else getattr(calc, 'homo', None)
                        beta_homo = getattr(calc, 'beta_homo', None)
                        dominant_trans = ', '.join(
                            format_transition(t[0], t[1], t[2], spin=t[3] if len(t) > 3 else '', homo=homo, beta_homo=beta_homo)
                            for t in details.get("population", []) if t[2] > args.threshold
                        )

            if w_sum > 0:
                weighted_e /= w_sum
                weighted_f /= w_sum
                weighted_r /= w_sum
                weighted_s2 /= w_sum

            avg_rows.append([
                state_label(mult, st), mult, weighted_e, weighted_f, weighted_r, weighted_s2, dominant_trans
            ])

        if avg_rows:
            self.output(
                avg_rows,
                headers=["State", "Mult", "⟨Energy (eV)⟩", '⟨f⟩', '⟨R (1e-40 cgs)⟩', '⟨<S**2>⟩', "Dominant Transitions"],
                title="Ensemble-Averaged Excited States (Boltzmann Weighted)"
            )

        # 2. Averaged UV / ECD spectra
        for flag, name, key in _SPECTRA:
            if getattr(args, flag):
                y_avg = np.zeros_like(x_ev)
                for item, w in zip(confs_data, weights):
                    y_avg += w * self._spectrum(item['calc'], key, x_ev)
                self._save_spectrum(x_ev, {"Ensemble Average": y_avg}, name,
                                    self.plot_path('spectra', tag=f'{name}_ensemble'))

        # 3. Averaged Spin-Orbit Coupling (SOC) Matrix
        soc_confs = [
            (item, w) for item, w in zip(confs_data, weights)
            if getattr(item['calc'], 'soc', False) and getattr(item['calc'], 'tddft_soc', None) is not None and item['calc'].tddft_soc.size > 0
        ]
        if soc_confs:
            max_rows = max(item['calc'].tddft_soc.shape[0] for item, _ in soc_confs)
            max_cols = max(item['calc'].tddft_soc.shape[1] for item, _ in soc_confs)
            avg_soc = np.zeros((max_rows, max_cols))
            soc_w_sum = 0.0

            for item, w in soc_confs:
                soc_w_sum += w
                mat = item['calc'].tddft_soc
                avg_soc[:mat.shape[0], :mat.shape[1]] += w * mat

            if soc_w_sum > 0:
                avg_soc /= soc_w_sum

            soc_headers = ['State'] + [f'T{i+1}' for i in range(max_cols)]
            soc_rows = [[f'S{idx}'] + [round(float(v), 3) for v in row] for idx, row in enumerate(avg_soc)]
            self._soc_table(soc_headers, soc_rows,
                            "Ensemble-Averaged Spin-Orbit Coupling (SOC) Matrix (Boltzmann Weighted)",
                            "the ensemble")


def parse_args():
    return TddftParser().build_parser().parse_args()


def main():
    TddftParser().main()


if __name__ == '__main__':
    main()
