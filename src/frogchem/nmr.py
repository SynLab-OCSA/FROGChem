"""
NMR Equivalency Detection and Data Processing Module.

Provides algorithms for:
1. Detecting NMR chemical equivalencies by combining global point group symmetry
   (rigid heavy-atom symmetry) with local topology (graph connectivity for rotators like methyls).
2. Converting calculated magnetic isotropic shieldings (sigma, ppm) to chemical shifts (delta, ppm).
3. Symmetrizing NMR parameters (shifts, J-couplings, and interatomic distances) over equivalent nuclei.
4. Physical constants for common NMR-active spin-1/2 nuclei.

Andrea Pellegrini, 2026
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from frogchem.chemistry import get_atomic_radius
from frogchem.graph_theory import create_graph_matrix, DisjointSet
from frogchem.io import read_xyz
from frogchem.symmetry import detect_point_group


import json
from pathlib import Path


def load_nucleus_database(filepath: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Load nucleus constants and spins from an external JSON data file.

    Search priority:
    1. Explicit filepath argument if passed
    2. FROG_NUCLEI_FILE environment variable
    3. Bundled frogchem/data/nuclei.json package data
    """
    target: Optional[Path] = None
    if filepath is not None:
        target = Path(filepath)
    elif os.environ.get('FROG_NUCLEI_FILE'):
        target = Path(os.environ['FROG_NUCLEI_FILE'])
    elif os.environ.get('PACLAB_NUCLEI_FILE'):          # former name, still accepted
        target = Path(os.environ['PACLAB_NUCLEI_FILE'])
    else:
        pkg_data = Path(__file__).parent / 'data' / 'nuclei.json'
        if pkg_data.exists():
            target = pkg_data

    if target and target.exists():
        try:
            with open(target, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load nuclear database from {target}: {e}")
    return {}


# Nuclear database loaded dynamically from external JSON data file (not hardcoded)
NUCLEUS_DATA: Dict[str, Dict[str, Any]] = load_nucleus_database()


def parse_spin(spin_val: Any) -> float:
    """Parse a nuclear spin value represented as float, int, or fraction string (e.g. '5/2' -> 2.5, '1/2' -> 0.5)."""
    if isinstance(spin_val, (int, float)):
        return float(spin_val)
    s = str(spin_val).strip()
    if '/' in s:
        parts = s.split('/')
        return float(parts[0]) / float(parts[1])
    return float(s)


def parse_spins_arg(arg: Any) -> Dict[str, float]:
    """Parse user command line specification for spins (e.g. 'O=5/2,H=1/2' or '17O:2.5' or JSON string)."""
    if not arg:
        return {}
    if isinstance(arg, dict):
        return {normalize_nucleus_name(k): parse_spin(v) for k, v in arg.items()}
    if isinstance(arg, (list, tuple)):
        res: Dict[str, float] = {}
        for item in arg:
            res.update(parse_spins_arg(item))
        return res
    s = str(arg).strip()
    if s.startswith('{') and s.endswith('}'):
        raw = json.loads(s)
        return {normalize_nucleus_name(k): parse_spin(v) for k, v in raw.items()}
    res = {}
    for part in s.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            k, v = part.split('=', 1)
        elif ':' in part:
            k, v = part.split(':', 1)
        else:
            continue
        res[normalize_nucleus_name(k.strip())] = parse_spin(v.strip())
    return res


def set_nucleus_spin(nucleus: str, spin: Union[float, str]) -> None:
    """Dynamically set or override the nuclear spin for a nucleus without altering the source code."""
    norm = normalize_nucleus_name(nucleus)
    s_val = parse_spin(spin)
    if norm not in NUCLEUS_DATA:
        NUCLEUS_DATA[norm] = {'spin': s_val, 'element': extract_element_from_nucleus(nucleus)}
    else:
        NUCLEUS_DATA[norm]['spin'] = s_val

# Synonyms mapping (e.g., 'H' -> '1H', 'C' -> '13C', 'S' -> '33S')
_NUCLEUS_ALIASES: Dict[str, str] = {
    'H': '1H',
    'C': '13C',
    'F': '19F',
    'P': '31P',
    'N': '15N',
    'SI': '29Si',
    'S': '33S',
    'B': '11B',
    'O': '17O',
    'SE': '77Se',
    'CL': '35Cl',
    'BR': '79Br',
    'I': '127I',
    'SN': '119Sn',
    'PT': '195Pt',
    'V': '51V',
    'LI': '7Li',
    'NA': '23Na',
    'AL': '27Al',
    'PB': '207Pb',
    'HG': '199Hg',
    'CD': '113Cd',
}


def normalize_nucleus_name(nucleus: str) -> str:
    """Normalize nucleus symbol (e.g. 'H' -> '1H', '1h' -> '1H', 'c' -> '13C', 's' -> '33S')."""
    clean = str(nucleus).strip()
    clean_upper = clean.upper()
    if clean_upper in _NUCLEUS_ALIASES:
        return _NUCLEUS_ALIASES[clean_upper]
    # Check if capitalized matches key
    for k in NUCLEUS_DATA:
        if k.upper() == clean_upper:
            return k
    return clean


try:
    import periodictable
    import periodictable.nsf
    periodictable.nsf.init(periodictable.elements)
except Exception:
    periodictable = None


def get_periodictable_spin(symbol_or_nuc: str) -> Optional[float]:
    """Retrieve nuclear spin quantum number I directly from the periodictable package for any isotope or element symbol.

    Examples:
        get_periodictable_spin('1H') -> 0.5
        get_periodictable_spin('17O') -> 2.5
        get_periodictable_spin('33S') -> 1.5
        get_periodictable_spin('O') -> 2.5
    """
    if periodictable is None:
        return None
    import re
    clean = str(symbol_or_nuc).strip()
    m = re.match(r'^(\d+)?([A-Za-z]+)$', clean)
    if not m:
        return None
    mass_str, elem_str = m.group(1), m.group(2).capitalize()
    if not hasattr(periodictable, elem_str):
        return None
    el = getattr(periodictable, elem_str)
    if mass_str:
        try:
            iso = el[int(mass_str)]
            sp = getattr(iso, 'nuclear_spin', None)
            if sp is not None and sp != '':
                return abs(parse_spin(sp))
        except Exception:
            pass
    else:
        # Search for NMR-active isotopes (nuclear_spin != '0')
        candidates = []
        for iso in el:
            sp = getattr(iso, 'nuclear_spin', None)
            if sp and sp != '0':
                ab = getattr(iso, 'abundance', 0.0) or 0.0
                candidates.append((iso.isotope, sp, ab))
        if candidates:
            # Sort by natural abundance descending
            candidates.sort(key=lambda x: x[2], reverse=True)
            return abs(parse_spin(candidates[0][1]))
    return None


def get_nucleus_spin(nucleus: str, custom_spins: Optional[Dict[str, float]] = None) -> float:
    """Get the nuclear spin quantum number I for a nucleus or element (e.g. '1H' -> 0.5, '17O' -> 2.5, 'O' -> 2.5).

    Search priority:
    1. custom_spins mapping (CLI --spin / --spins overrides)
    2. periodictable package database
    3. NUCLEUS_DATA database
    4. Default 0.5
    """
    clean = str(nucleus).strip()
    norm = normalize_nucleus_name(clean)
    if custom_spins:
        if clean in custom_spins:
            return float(custom_spins[clean])
        if norm in custom_spins:
            return float(custom_spins[norm])
        el = extract_element_from_nucleus(clean)
        if el in custom_spins:
            return float(custom_spins[el])

    # Query periodictable directly
    pt_spin = get_periodictable_spin(clean)
    if pt_spin is None and norm != clean:
        pt_spin = get_periodictable_spin(norm)
    if pt_spin is not None:
        return float(pt_spin)

    # Query NUCLEUS_DATA
    if norm in NUCLEUS_DATA and 'spin' in NUCLEUS_DATA[norm]:
        return float(NUCLEUS_DATA[norm]['spin'])
    return 0.5


def format_spin(spin: float) -> str:
    """Format nuclear spin as a human-readable fraction (e.g. 0.5 -> '1/2', 1.0 -> '1', 2.5 -> '5/2')."""
    two_s = int(round(2.0 * spin))
    if two_s % 2 == 0:
        return str(two_s // 2)
    return f"{two_s}/2"


def is_quadrupolar(nucleus_or_spin: Union[str, float]) -> bool:
    """Return True if the nucleus has spin I > 1/2 (quadrupolar nucleus with electric quadrupole moment)."""
    if isinstance(nucleus_or_spin, (int, float)):
        return float(nucleus_or_spin) > 0.5
    return get_nucleus_spin(str(nucleus_or_spin)) > 0.5


def get_spin_multiplicity(spin: float, n_equivalent: int = 1) -> int:
    """Return the first-order multiplet splitting multiplicity (2 * n * I + 1)."""
    return int(round(2.0 * n_equivalent * spin + 1.0))


def extract_element_from_nucleus(nucleus: str) -> str:
    """Extract chemical element symbol from nucleus string (e.g. '1H' -> 'H', '13C' -> 'C', '33S' -> 'S')."""
    nuc = normalize_nucleus_name(nucleus)
    if nuc in NUCLEUS_DATA and 'element' in NUCLEUS_DATA[nuc]:
        return NUCLEUS_DATA[nuc]['element']
    import re
    m = re.match(r'^\d*([A-Za-z]+)$', str(nucleus).strip())
    if m:
        return m.group(1).capitalize()
    return str(nucleus).strip().capitalize()


def parse_nuclei_list(nuclei_arg: Any) -> List[str]:
    """Parse a single string, sequence of strings, or comma/space-separated list into normalized nuclei list.
    
    Examples:
        parse_nuclei_list('1H') -> ['1H']
        parse_nuclei_list(['1H', '13C', '33S']) -> ['1H', '13C', '33S']
        parse_nuclei_list('1H,13C,33S') -> ['1H', '13C', '33S']
        parse_nuclei_list(['1H 13C', '33S']) -> ['1H', '13C', '33S']
    """
    if nuclei_arg is None:
        return ['1H']
    
    raw_tokens: List[str] = []
    if isinstance(nuclei_arg, (str, bytes)):
        items = [str(nuclei_arg)]
    elif isinstance(nuclei_arg, (list, tuple, set)):
        items = [str(x) for x in nuclei_arg]
    else:
        items = [str(nuclei_arg)]

    import re
    for item in items:
        # Split on comma, semicolon, or whitespace
        parts = re.split(r'[,;\s]+', item.strip())
        for p in parts:
            p_clean = p.strip()
            if p_clean:
                raw_tokens.append(p_clean)

    if not raw_tokens:
        return ['1H']

    # Normalize and deduplicate preserving order
    seen = set()
    result: List[str] = []
    for tok in raw_tokens:
        norm = normalize_nucleus_name(tok)
        if norm not in seen:
            seen.add(norm)
            result.append(norm)

    return result

def parse_nuclei_set(nuclei_arg: Any) -> Set[str]:
    """Parse nuclei argument into a set containing normalized nucleus symbols and element symbols."""
    if nuclei_arg is None or isinstance(nuclei_arg, bool):
        return set()
    n_list = parse_nuclei_list(nuclei_arg)
    res = set()
    for item in n_list:
        res.add(item)
        res.add(item.upper())
        elem = extract_element_from_nucleus(item)
        if elem:
            res.add(elem)
            res.add(elem.upper())
    return res


def is_nucleus_decoupled(
    nucleus: str,
    coupled_arg: Any = None,
    decoupled_arg: Any = None,
    nuclei_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """Determine whether a given nucleus should be simulated as decoupled (single lines) or coupled (multiplet splitting).

    Precedence:
    1. Explicit `--decoupled` / `--decouple` argument:
       - Bare flag / empty list / True -> ALL nuclei decoupled (returns True).
       - List containing nucleus/element -> decoupled (returns True).
    2. Explicit `--coupled` / `--couple` argument:
       - Bare flag / empty list / True -> ALL nuclei coupled (returns False).
       - List containing nucleus/element -> coupled (returns False).
    3. JSON database setting `decoupled` in `nuclei_data`.
    4. Default convention:
       - '1H' / 'H' -> coupled by default (returns False).
       - Heteronuclei ('13C', '17O', '33S', etc.) -> decoupled by default (returns True).
    """
    if nuclei_data is None:
        nuclei_data = NUCLEUS_DATA

    norm = normalize_nucleus_name(nucleus)
    elem = extract_element_from_nucleus(norm)

    # 1. Check decoupled argument
    if decoupled_arg is True or (isinstance(decoupled_arg, (list, tuple)) and len(decoupled_arg) == 0):
        return True
    if decoupled_arg is not None:
        dec_set = parse_nuclei_set(decoupled_arg)
        if norm in dec_set or elem in dec_set or str(nucleus) in dec_set or str(nucleus).upper() in dec_set:
            return True

    # 2. Check coupled argument
    if coupled_arg is True or (isinstance(coupled_arg, (list, tuple)) and len(coupled_arg) == 0):
        return False
    if coupled_arg is not None:
        cpl_set = parse_nuclei_set(coupled_arg)
        if norm in cpl_set or elem in cpl_set or str(nucleus) in cpl_set or str(nucleus).upper() in cpl_set:
            return False

    # 3. Check database specification
    info = nuclei_data.get(norm, {})
    if 'decoupled' in info:
        return bool(info['decoupled'])

    # 4. Fallback convention: 1H is coupled, others decoupled
    if norm in ('1H', 'H') or elem == 'H':
        return False
    return True

def extract_reference_shieldings(
    ref: Union[float, int, str, os.PathLike, Sequence[Any], Dict[str, Any], Any]
) -> Dict[str, float]:
    """Extract reference isotropic shieldings from numeric values, key-value definitions,
    or quantum chemistry calculation files.

    Parameters:
        ref: Can be:
            - A numeric float or int (or string representing float): universal reference ('*'),
              applied to EVERY nucleus (use key-value definitions for per-nucleus references).
            - A key-value string: e.g. '1H=31.8', '13C:185.0', '1H=31.8,13C=185.0', '1H=tms.out,31P=h3po4.out'.
            - A list/tuple of references: e.g. ['1H=31.8', '13C=185.0'] or ['tms.out', 'h3po4.out'].
            - A dictionary: e.g. {'1H': 31.8, '13C': 185.0} or {'1H': 'tms.out', '31P': 'h3po4.out'}.
            - Path to an output file (.out, .log) containing NMR calculation (e.g. TMS calculation).
            - A parsed Calculation / GaussianCalculation object.

    Returns:
        Dict mapping element symbol (and nucleus names) to average isotropic shielding (ppm).
        For example: {'H': 31.8, '1H': 31.8, 'C': 185.0, '13C': 185.0, 'Si': 360.0, '29Si': 360.0}
    """
    if ref is None:
        return {}

    # 1. Direct number
    if isinstance(ref, (int, float)):
        return {'*': float(ref)}

    # 2. Sequence (list, tuple) of references
    if isinstance(ref, (list, tuple)):
        merged: Dict[str, float] = {}
        for item in ref:
            sub = extract_reference_shieldings(item)
            merged.update(sub)
        return merged


    def _add_shielding_entry(mapping: Dict[str, float], nuc: str, sub: Dict[str, float]) -> None:
        norm = normalize_nucleus_name(nuc)
        val = sub.get('*', sub.get(norm, next(iter(sub.values())) if sub else 0.0))
        mapping[norm] = val
        mapping[norm.upper()] = val
        el = NUCLEUS_DATA.get(norm, {}).get('element', '')
        if el:
            mapping[el] = val
            mapping[el.upper()] = val

    # 3. Dictionary of references: {'1H': 31.8, '13C': 185.0}
    if isinstance(ref, dict):
        merged = {}
        for k, v in ref.items():
            _add_shielding_entry(merged, str(k), extract_reference_shieldings(v))
        return merged

    # 4. String handling
    if isinstance(ref, str):
        ref_str = ref.strip()
        try:
            return {'*': float(ref_str)}
        except ValueError:
            pass

        if not os.path.exists(ref_str):
            if ',' in ref_str:
                merged = {}
                for part in ref_str.split(','):
                    p = part.strip()
                    if p:
                        merged.update(extract_reference_shieldings(p))
                return merged

            import re
            m = re.match(r'^([0-9]*[A-Za-z]+)\s*[:=]\s*(.+)$', ref_str)
            if m:
                merged = {}
                _add_shielding_entry(merged, m.group(1), extract_reference_shieldings(m.group(2).strip()))
                return merged

    # 5. Calculation object or single file path
    calc = ref
    if isinstance(ref, (str, os.PathLike)):
        from frogchem.filetypes import load_calculation
        path_str = str(ref)
        if not os.path.exists(path_str):
            raise FileNotFoundError(f"Reference file not found: {path_str}")
        calc = load_calculation(path_str)

    if not hasattr(calc, 'nmr') or calc.nmr is None or not getattr(calc.nmr, 'shifts', None):
        ref_name = getattr(calc, 'filename', str(ref))
        raise ValueError(f"Reference calculation '{ref_name}' does not contain any NMR shielding data.")

    # Group isotropic shielding by element
    by_element: Dict[str, List[float]] = defaultdict(list)
    for data in calc.nmr.shifts.values():
        el = data.get('element', '').strip().capitalize()
        if el and 'iso' in data:
            by_element[el].append(float(data['iso']))

    result: Dict[str, float] = {}
    for el, vals in by_element.items():
        avg = float(np.mean(vals))
        result[el] = avg
        result[el.upper()] = avg
        # Also map standard isotopes (e.g. H -> 1H, C -> 13C, Si -> 29Si, P -> 31P, F -> 19F, N -> 15N)
        for nuc_name, nuc_info in NUCLEUS_DATA.items():
            if nuc_info.get('element', '').upper() == el.upper():
                result[nuc_name] = avg
                result[nuc_name.upper()] = avg

    return result



def resolve_reference_shielding(
    ref: Union[float, int, str, os.PathLike, Any, None],
    nucleus: str = '1H',
) -> float:
    """Resolve the reference isotropic shielding (ppm) for a given nucleus.

    Parameters:
        ref: Numeric value (float/int/str), calculation file path, Calculation object, or None.
        nucleus: Active NMR nucleus (e.g. '1H', '13C', '29Si', 'H', 'C', 'Si').

    Returns:
        float: Reference isotropic shielding in ppm.
    """
    nuc_key = normalize_nucleus_name(nucleus)
    default_ref = NUCLEUS_DATA.get(nuc_key, {}).get('default_tms', 0.0)

    if ref is None:
        return float(default_ref)

    ref_map = extract_reference_shieldings(ref)
    if not ref_map:
        return float(default_ref)

    # Universal numeric reference ('*')
    if '*' in ref_map:
        return float(ref_map['*'])

    # Lookup by exact nucleus key (e.g. '1H', '13C', '29Si')
    if nuc_key in ref_map:
        return float(ref_map[nuc_key])
    if nuc_key.upper() in ref_map:
        return float(ref_map[nuc_key.upper()])

    # Lookup by element symbol (e.g. 'H', 'C', 'Si')
    target_element = NUCLEUS_DATA.get(nuc_key, {}).get('element', '')
    if not target_element:
        import re
        target_element = re.sub(r'^\d+', '', nucleus).capitalize()

    if target_element in ref_map:
        return float(ref_map[target_element])
    if target_element.upper() in ref_map:
        return float(ref_map[target_element.upper()])

    avail = sorted(list(set(k for k in ref_map.keys() if not k.startswith('*'))))
    ref_desc = getattr(ref, 'filename', str(ref))
    raise ValueError(
        f"Reference calculation '{ref_desc}' does not contain NMR shielding data for nucleus '{nucleus}' "
        f"(element '{target_element}'). Available elements/nuclei in reference: {avail}"
    )


def convert_shielding_to_shift(
    sigma: float,
    nucleus: str = '1H',
    ref_sigma: Optional[Union[float, int, str, os.PathLike, Any]] = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
) -> float:
    """Convert calculated absolute isotropic shielding (sigma, ppm) to chemical shift (delta, ppm).

    Parameters:
        sigma: Isotropic magnetic shielding in ppm.
        nucleus: Target nucleus ('1H', '13C', '19F', etc.).
        ref_sigma: Reference absolute shielding in ppm (e.g. float 31.8) OR
                   path to an output calculation file (e.g. 'tms.out' containing TMS shielding data).
                   If None, uses default DFT reference for the nucleus.
        slope: Optional empirical scaling factor slope.
        intercept: Optional empirical scaling factor intercept.

    Formula:
        If slope and intercept are provided:
            delta = (sigma - intercept) / slope
        Else:
            delta = (ref_sigma - sigma) / (1 - ref_sigma * 1e-6)
    """
    if slope is not None and intercept is not None:
        return (sigma - intercept) / slope

    ref_val = resolve_reference_shielding(ref_sigma, nucleus=nucleus)
    return float((ref_val - sigma) / (1 - ref_val * 1e-6))



def get_nmr_equivalencies(
    xyz_input: Any,
    buffer: float = 0.3,
    decimals: int = 4,
) -> Dict[str, List[List[int]]]:
    """Detect NMR chemical equivalencies by combining:
    1. Global rigid point group symmetry (ignoring hydrogens during detection,
       then applying symmetry operations to all atoms). An equivalence is accepted only
       between atoms with the same topological (Weisfeiler-Lehman) colour on the full
       bonded graph, so a heavy-atom symmetry broken by the H atoms is not applied.
    2. Local graph topology to detect rotators (e.g., methyl -CH3 groups).

    Parameters:
        xyz_input: File path, Calculation object, tuple of (atoms, coords),
                   or string in XYZ format.
        buffer: Distance buffer (Angstrom) for covalent bond connectivity.
        decimals: Coordinate rounding precision for symmetry mapping.

    Returns:
        Dict mapping element symbol (e.g., 'H', 'C') to a list of groups,
        where each group is a list of 0-based atom indices.
    """
    # 1. Parse input
    if hasattr(xyz_input, 'atoms') and hasattr(xyz_input, 'geom'):
        atoms = list(xyz_input.atoms)
        coords = np.array(xyz_input.geom, dtype=float)
    elif isinstance(xyz_input, str):
        if os.path.isfile(xyz_input):
            atoms_l, coords_l, _ = read_xyz(xyz_input)
            atoms = atoms_l[0]
            coords = coords_l[0]
        else:
            lines = xyz_input.strip().split('\n')
            start = 2 if len(lines) > 2 and lines[0].strip().isdigit() else 0
            atoms = [l.split()[0] for l in lines[start:] if l.strip()]
            coords = np.array([[float(x) for x in l.split()[1:4]] for l in lines[start:] if l.strip()])
    else:
        atoms, coords = xyz_input

    atoms = [str(a) for a in atoms]
    coords = np.array(coords, dtype=float)
    n_atoms = len(atoms)
    if n_atoms == 0:
        return {}

    # 2. Rigid global symmetry detection
    # exclude_h=True detects group on heavy framework, then maps all atoms including H
    pg_data = detect_point_group(atoms, coords, tolerance=0.1, exclude_h=True)
    global_equiv = pg_data.get('equivalencies', {})

    # 3. Local topology for rotators
    radii = {k: get_atomic_radius(k) for k in set(atoms)}
    adj = create_graph_matrix(atoms, coords, radii, buffer).astype(bool)

    # Topological colours (Weisfeiler-Lehman refinement on the full graph, H included).
    # Symmetry-equivalent atoms must share the same colour; this rejects spurious
    # equivalencies from a heavy-atom-only point group that the H atoms do not obey
    # (e.g. D_inf_h of a C-C framework carrying CH and CH3 ends).
    colours = list(atoms)
    for _ in range(min(n_atoms, 6)):
        colours = [colours[i] + '(' + ','.join(sorted(colours[n] for n in np.where(adj[i])[0])) + ')'
                   for i in range(n_atoms)]
        uniq = {c: str(k) for k, c in enumerate(sorted(set(colours)))}
        colours = [atoms[i] + uniq[colours[i]] for i in range(n_atoms)]

    # 4. Disjoint Set Union (DSU) merge
    dsu = DisjointSet(n_atoms)
    # Apply global symmetry equivalencies (only between topologically identical atoms)
    for el, groups in global_equiv.items():
        for group in groups:
            if len(group) > 1:
                by_colour: Dict[str, List[int]] = defaultdict(list)
                for member in group:
                    by_colour[colours[int(member)]].append(int(member))
                for sub in by_colour.values():
                    for member in sub[1:]:
                        dsu.union(sub[0], member)

    # Apply rotor equivalencies (e.g. -CH3, -NH3+)
    for i, atom in enumerate(atoms):
        if atom != 'H':
            neighbors = np.where(adj[i])[0]
            h_neighbors = [int(n) for n in neighbors if atoms[n] == 'H']
            if len(h_neighbors) >= 3:
                root = h_neighbors[0]
                for h in h_neighbors[1:]:
                    dsu.union(root, h)

    # Build final groupings grouped by element
    output: Dict[str, List[List[int]]] = defaultdict(list)
    for g in dsu.get_groups().values():
        el = atoms[g[0]]
        output[el].append(sorted(g))

    for k in output:
        output[k].sort(key=lambda x: x[0])

    return dict(output)


def symmetrize_nmr_data(
    shifts: Dict[int, Dict[str, Any]],
    j_couplings: Dict[Tuple[int, int], float],
    geom: np.ndarray,
    equivalencies: Dict[str, List[List[int]]],
    element: str = 'H',
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], float], np.ndarray, List[List[int]]]:
    """Symmetrize NMR isotropic shieldings, J-couplings, and inter-group distances
    over chemically equivalent groups of nuclei.

    Parameters:
        shifts: Dict mapping 0-based atom index to {'element': str, 'iso': float, ...}.
        j_couplings: Dict mapping (idx1, idx2) to scalar coupling in Hz.
        geom: (N, 3) Cartesian coordinates.
        equivalencies: Dict mapping element to list of equivalent atom index lists.
        element: Element to filter and symmetrize (default: 'H').

    Returns:
        Tuple containing:
        - group_shifts: Dict[int, Dict[str, Any]] keyed by group representative index.
        - group_j: Dict[Tuple[int, int], float] keyed by (rep1, rep2).
        - group_dist: 2D np.ndarray of NOE-effective distances <r^-6>^(-1/6) between groups (Å).
        - active_groups: List of groups for the selected element.
    """
    element_groups = equivalencies.get(element, [])
    # Only keep groups where atoms actually have NMR shifts calculated
    active_groups: List[List[int]] = []
    for g in element_groups:
        present = [i for i in g if i in shifts]
        if present:
            active_groups.append(present)

    # Group shifts: average isotropic shielding and anisotropy
    group_shifts: Dict[int, Dict[str, Any]] = {}
    for g in active_groups:
        rep = g[0]
        el = shifts[rep]['element']
        iso_mean = float(np.mean([shifts[i]['iso'] for i in g]))
        aniso_mean = float(np.mean([shifts[i].get('aniso', 0.0) for i in g]))
        group_shifts[rep] = {
            'element': el,
            'iso': iso_mean,
            'aniso': aniso_mean,
            'members': list(g),
            'count': len(g),
        }

    # Inter-group J couplings: average coupling between groups
    group_j: Dict[Tuple[int, int], float] = {}
    n_groups = len(active_groups)
    for a in range(n_groups):
        rep_a = active_groups[a][0]
        for b in range(a + 1, n_groups):
            rep_b = active_groups[b][0]
            vals = []
            for ia in active_groups[a]:
                for ib in active_groups[b]:
                    val = j_couplings.get((ia, ib), j_couplings.get((ib, ia), 0.0))
                    vals.append(val)
            mean_j = float(np.mean(vals)) if vals else 0.0
            group_j[(rep_a, rep_b)] = mean_j
            group_j[(rep_b, rep_a)] = mean_j

    # Inter-group NOE-effective distances <r^-6>^(-1/6) over member pairs (an arithmetic
    # mean <r> would underestimate the NOE, which scales as r^-6)
    group_dist = np.zeros((n_groups, n_groups), dtype=float)
    geom_arr = np.array(geom, dtype=float) if geom is not None else np.array([])
    for a in range(n_groups):
        for b in range(n_groups):
            if a == b:
                group_dist[a, b] = 0.0
            elif len(geom_arr) > 0:
                dist_vals = []
                for ia in active_groups[a]:
                    for ib in active_groups[b]:
                        if len(geom_arr) > max(ia, ib):
                            dist_vals.append(float(np.linalg.norm(geom_arr[ia] - geom_arr[ib])))
                dist_vals = [d for d in dist_vals if d > 0.0]
                group_dist[a, b] = float(np.mean(np.array(dist_vals) ** -6.0) ** (-1.0 / 6.0)) if dist_vals else 0.0

    return group_shifts, group_j, group_dist, active_groups