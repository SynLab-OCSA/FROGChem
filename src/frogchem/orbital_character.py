"""Character of molecular orbitals and excited states: n / π / σ orbitals, n→π*, π→π* and charge transfer.

Works on any parsed calculation with MO coefficients (ORCA ``! PrintBasis PrintMOs``, Gaussian
``pop=full gfinput``): ``orbital_coefficients`` (n_bf x n_MO, alpha then beta columns for open shell,
or ``alpha_coeff``/``beta_coeff``), ``mask_atoms``, ``mask_type_n`` and ``mask_orbital_type``.

* Orbital weights are squared coefficients normalised per MO (C² analysis, no overlap), as in frog-mo.
* π character: every atom of an unsaturated (or conjugated) group gets a local normal, the normal of
  the plane through it and its bonded neighbours; the π weight of an MO is Σ (c_p · n)² over the p
  shells of those atoms, divided by Σ C². sp3 atoms and isolated lone pairs have no π part.
* Orbital types: π / π* (π weight ≥ PI_MIN), n (occupied, π weight < N_MAX_PI and at least
  N_MIN_HETERO of the MO on one heteroatom), otherwise σ / σ*.
* State character: weights of the transitions of a state (normalised to their sum) split into
  n→π*, π→π* and, with two fragments A and B, A→B, B→A, A local, B local (w · p_X(donor) · p_Y(acceptor)).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from frogchem.exceptions import InputError

PI_MIN = 0.6          # π weight of a π / π* orbital
N_MAX_PI = 0.4        # π weight below which an occupied orbital can be a lone pair
N_MIN_HETERO = 0.4    # weight on one heteroatom of a lone pair
DOMINANT = 0.5        # share of the transitions that names the type of a state
PLANAR_TOL = 0.25     # Å: distance of a 3-coordinated atom from the plane of its neighbours
HETEROATOMS = {'N', 'O', 'S', 'Se', 'P', 'F', 'Cl', 'Br', 'I'}
_P_CODES = (1, 2, 3)  # px, py, pz (orca_parser.orb_type.ORB_TYPE)


# ------------------------------------------------------------------- input
def parse_indices(tokens) -> List[int]:
    """Sorted 0-based indices from tokens such as '14-17', '20' or '14,15 16'."""
    if isinstance(tokens, str):
        tokens = [tokens]
    out = set()
    for tok in tokens:
        for part in str(tok).replace(' ', '').split(','):
            if not part:
                continue
            if '-' in part:
                a, b = part.split('-', 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(part))
    return sorted(out)


def parse_fragments(tokens: Sequence[str], n_atoms: int) -> List[List[int]]:
    """Fragments from tokens like '1,5,8-10' (0-based atoms); at most two, disjoint, in range."""
    if not tokens:
        return []
    if len(tokens) > 2:
        raise InputError(f"--frag takes one or two fragments, got {len(tokens)}")
    frags = []
    for tok in tokens:
        try:
            atoms = parse_indices(tok)
        except ValueError:
            raise InputError(f"Fragment not understood: {tok!r} (use e.g. 1,5,8-10)") from None
        bad = [a for a in atoms if not 0 <= a < n_atoms]
        if bad or not atoms:
            raise InputError(f"Fragment {tok!r}: atom indices must be between 0 and {n_atoms - 1}")
        frags.append(atoms)
    if len(frags) == 2 and set(frags[0]) & set(frags[1]):
        raise InputError(f"The two fragments share atoms {sorted(set(frags[0]) & set(frags[1]))}")
    return frags


def has_mos(calc) -> bool:
    return (getattr(calc, 'orbital_coefficients', None) is not None
            and getattr(calc, 'mask_atoms', None) is not None)


def transition_spin(t) -> str:
    """'' (restricted), 'a' or 'b' of a transition tuple (from, to, weight[, spin])."""
    spin = getattr(t, 'spin', None)
    if spin is None:
        spin = t[3] if len(t) > 3 else ''
    return spin or ''


# ---------------------------------------------------------------- orbitals
def spin_coefficients(calc, spin: str) -> np.ndarray:
    """MO coefficients (n_bf x n_MO) of one spin ('', 'a' or 'b')."""
    if getattr(calc, 'alpha_coeff', None) is not None:
        return calc.beta_coeff if spin == 'b' else calc.alpha_coeff
    return calc.orbital_coefficients


def mo_weights(coeffs: np.ndarray) -> np.ndarray:
    """C² normalised to 1 per MO (column)."""
    sq = np.asarray(coeffs, dtype=float) ** 2
    norm = sq.sum(axis=0)
    norm[norm == 0] = 1.0
    return sq / norm


def atom_weights(calc, coeffs: np.ndarray) -> np.ndarray:
    """(n_atoms, n_MO) weight of every MO on every atom."""
    w = mo_weights(coeffs)
    atoms = np.asarray(calc.mask_atoms).astype(int)
    out = np.zeros((len(calc.atoms), w.shape[1]))
    np.add.at(out, atoms, w)
    return out


def _mo_geometry(calc) -> np.ndarray:
    """Coordinates in the frame of the MO coefficients: Gaussian prints them in the standard orientation."""
    fl = calc.__dict__.get('fl') or ''
    if 'Standard orientation:' in fl:
        from frogchem.gaussian_parser.hessian import _last_orientation
        std = _last_orientation(fl, 'Standard orientation:')
        if std is not None and len(std) == len(calc.atoms):
            return std
    return np.asarray(calc.geom, dtype=float)


def _plane_normal(points: np.ndarray) -> Optional[np.ndarray]:
    """Unit normal of the best plane through points, None when they are (nearly) collinear."""
    centred = points - points.mean(axis=0)
    _, s, vt = np.linalg.svd(centred)
    if len(s) < 2 or s[1] < 0.1 * s[0]:
        return None
    return vt[-1]


def local_normals(symbols: Sequence[str], geom: np.ndarray) -> np.ndarray:
    """(n_atoms, 3) normal of the π system at each atom, NaN where the atom has no π part.

    π atoms: C with 2-3 neighbours, N/P with 1-2, B with 3 (3-coordinated ones only if planar);
    terminal O/S/Se on such an atom (carbonyl, thiocarbonyl); N/O/S/Se with lone pairs bonded to
    one of them (amide N, furan O, aniline N). Atoms with too few neighbours or on a linear chain
    take the normal of a bonded π atom.
    """
    from frogchem.chemistry import get_atomic_radius
    from frogchem.graph_theory import create_graph_matrix

    symbols = [str(s) for s in symbols]
    geom = np.asarray(geom, dtype=float)
    n = len(symbols)
    radii = {s: get_atomic_radius(s) for s in set(symbols)}
    graph = create_graph_matrix(symbols, geom, radii, buffer=0.3)
    neigh = [list(np.nonzero(graph[i])[0]) for i in range(n)]

    def planar(i):
        if len(neigh[i]) != 3:
            return True
        normal = _plane_normal(geom[neigh[i]])
        return normal is not None and abs(np.dot(geom[i] - geom[neigh[i]].mean(axis=0), normal)) < PLANAR_TOL

    def unsaturated(i):
        s, nb = symbols[i], len(neigh[i])
        ok = ((s == 'C' and nb in (2, 3)) or (s in ('N', 'P') and nb in (1, 2)) or (s == 'B' and nb == 3))
        return ok and planar(i)

    core = {i for i in range(n) if unsaturated(i)}
    pi_atoms = set(core)
    for i in range(n):
        if i in core or not any(j in core for j in neigh[i]):
            continue
        s, nb = symbols[i], len(neigh[i])
        if (s in ('O', 'S', 'Se') and nb in (1, 2)) or (s == 'N' and nb == 3):
            pi_atoms.add(i)

    normals = np.full((n, 3), np.nan)
    for i in pi_atoms:
        if len(neigh[i]) >= 2:
            nv = _plane_normal(geom[[i] + neigh[i]])
            if nv is not None:
                normals[i] = nv
    for _ in range(3):  # terminal / linear atoms: normal of a bonded π atom
        for i in pi_atoms:
            if np.isnan(normals[i, 0]):
                for j in neigh[i]:
                    if j in pi_atoms and not np.isnan(normals[j, 0]):
                        normals[i] = normals[j]
                        break
    return normals


def pi_fractions(calc, coeffs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """π weight of every MO: Σ (c_p · n)² over the p shells of the π atoms, over Σ C²."""
    c = np.asarray(coeffs, dtype=float)
    atoms = np.asarray(calc.mask_atoms).astype(int)
    shells = np.asarray(calc.mask_type_n).astype(str)
    codes = np.asarray(calc.mask_orbital_type).astype(int)
    pi = np.zeros(c.shape[1])
    rows: Dict[tuple, Dict[int, int]] = {}
    for r, (a, sh, code) in enumerate(zip(atoms, shells, codes)):
        if code in _P_CODES and not np.isnan(normals[a, 0]):
            rows.setdefault((a, sh), {})[code] = r
    for (a, _), comp in rows.items():
        if len(comp) != 3:
            continue
        proj = sum(normals[a, k] * c[comp[code]] for k, code in enumerate(_P_CODES))
        pi += proj ** 2
    total = (c ** 2).sum(axis=0)
    total[total == 0] = 1.0
    return pi / total


@dataclass
class SpinOrbitals:
    """Per-MO data of one spin: type, π weight, weight on each fragment (n_frag x n_MO)."""
    types: List[str]
    pi: np.ndarray
    fragments: Optional[np.ndarray]


def _classify(pi: np.ndarray, hetero: np.ndarray, n_occ: int) -> List[str]:
    types = []
    for i, (p, h) in enumerate(zip(pi, hetero)):
        occ = i < n_occ
        if p >= PI_MIN:
            types.append('π' if occ else 'π*')
        elif occ and p < N_MAX_PI and h >= N_MIN_HETERO:
            types.append('n')
        else:
            types.append('σ' if occ else 'σ*')
    return types


def analyse_orbitals(calc, fragments: Sequence[Sequence[int]] = ()) -> Dict[str, SpinOrbitals]:
    """{spin: SpinOrbitals} for '' (closed shell) or 'a' and 'b' (open shell)."""
    if not has_mos(calc):
        raise InputError(f"No MO coefficients in {calc.fname}: the orbital character needs "
                         "'! PrintBasis PrintMOs' (ORCA) or 'pop=full gfinput' (Gaussian)")
    open_shell = getattr(calc, 'alpha_coeff', None) is not None
    homo = calc.homo
    n_occ = {'': homo + 1, 'a': homo + 1}
    if open_shell:
        beta_homo = getattr(calc, 'beta_homo', None)
        n_occ['b'] = (beta_homo + 1) if beta_homo is not None else homo + 1
    normals = local_normals(calc.atoms, _mo_geometry(calc))
    hetero_mask = np.array([str(s) in HETEROATOMS for s in calc.atoms])
    out = {}
    for spin in (('a', 'b') if open_shell else ('',)):
        coeffs = spin_coefficients(calc, spin)
        per_atom = atom_weights(calc, coeffs)
        hetero = per_atom[hetero_mask].max(axis=0) if hetero_mask.any() else np.zeros(coeffs.shape[1])
        pi = pi_fractions(calc, coeffs, normals)
        frag = np.array([per_atom[list(f)].sum(axis=0) for f in fragments]) if fragments else None
        out[spin] = SpinOrbitals(_classify(pi, hetero, n_occ[spin]), pi, frag)
    return out


def orbitals_of(analysis: Dict[str, SpinOrbitals], spin: str) -> SpinOrbitals:
    """SpinOrbitals of a transition spin ('' falls back to alpha in open shell)."""
    if spin in analysis:
        return analysis[spin]
    return analysis.get('a') or analysis['']


# ------------------------------------------------------------------ states
def state_character(state: dict, analysis: Dict[str, SpinOrbitals]) -> Dict[str, float]:
    """Shares of n→π*, π→π* (and with two fragments A→B, B→A, A loc, B loc) and the dominant 'type'."""
    keys = ['n→π*', 'π→π*']
    two = any(o.fragments is not None and len(o.fragments) == 2 for o in analysis.values())
    if two:
        keys += ['A→B', 'B→A', 'A loc', 'B loc']
    res = {k: 0.0 for k in keys}
    wsum = 0.0
    for t in state.get('population', []):
        i, a, w = int(t[0]), int(t[1]), float(t[2])
        orb = orbitals_of(analysis, transition_spin(t))
        if max(i, a) >= len(orb.types):
            continue
        wsum += w
        ti, ta = orb.types[i], orb.types[a]
        if ti == 'n' and ta == 'π*':
            res['n→π*'] += w
        elif ti == 'π' and ta == 'π*':
            res['π→π*'] += w
        if two:
            fa, fb = orb.fragments[0], orb.fragments[1]
            res['A→B'] += w * fa[i] * fb[a]
            res['B→A'] += w * fb[i] * fa[a]
            res['A loc'] += w * fa[i] * fa[a]
            res['B loc'] += w * fb[i] * fb[a]
    if wsum > 0:
        res = {k: float(v / wsum) for k, v in res.items()}
    # CT between the fragments is the more specific label: a π(A)→π*(B) state is 'CT A→B'
    res['type'] = 'mixed'
    for group in ((('CT A→B', 'A→B'), ('CT B→A', 'B→A')) if two else (),
                  (('n→π*', 'n→π*'), ('π→π*', 'π→π*'))):
        if group:
            name, key = max(group, key=lambda g: res[g[1]])
            if res[key] >= DOMINANT:
                res['type'] = name
                break
    return res


#: --diagram-pick choice → state types it accepts
PICKS = {'npi': ('n→π*',), 'pipi': ('π→π*',), 'ct': ('CT A→B', 'CT B→A')}


def pick_state(characters: Sequence[dict], pick: str):
    """(index of the first character of the `pick` type or None, best share found among all)."""
    types = PICKS[pick]
    for k, ch in enumerate(characters):
        if ch['type'] in types:
            return k, None
    keys = ['A→B', 'B→A'] if pick == 'ct' else list(types)
    shares = [max(ch.get(key, 0.0) for key in keys) for ch in characters]
    return None, (max(shares) if shares else 0.0)
