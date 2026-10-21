"""Basis set, AO overlap and MO coefficients from a Gaussian log, in the layout of the ORCA parser.

Needs ``#p ... pop=full iop(3/33=1) gfinput``: ``gfinput`` prints the basis ("AO basis set in the form of
general basis input"), ``pop=full`` all MO coefficients, ``iop(3/33=1)`` the overlap matrix.

Gaussian conventions are mapped onto the ones of frogchem.density_builder (written for ORCA):

* the AO rows keep Gaussian's order (p: x, y, z; pure d: D0, D+1, D-1, D+2, D-2; pure f: F0 ... F-3) and
  ``mask_orbital_type`` holds the matching ORB_TYPE code of every row, so ``build_basis_table`` follows it;
* pure f(+3) and f(-3) have the standard phase in Gaussian and the opposite one in ORCA (and in
  density_builder): their coefficient rows (and overlap rows/columns) change sign;
* Cartesian d/f shells (6D/10F) use the Cartesian codes of orb_type (each function normalised to 1).
"""

import re
from collections import defaultdict

import numpy as np

from frogchem.exceptions import ParseError
from frogchem.orca_parser.orb_type import ORB_TYPE

_PURE = {
    'S': 's', 'PX': 'px', 'PY': 'py', 'PZ': 'pz',
    'D0': 'dz2', 'D+1': 'dxz', 'D-1': 'dyz', 'D+2': 'dx2y2', 'D-2': 'dxy',
    'F0': 'f0', 'F+1': 'f+1', 'F-1': 'f-1', 'F+2': 'f+2', 'F-2': 'f-2', 'F+3': 'f+3', 'F-3': 'f-3',
}
# Gaussian standard phase -> ORCA phase used by density_builder
_FLIP = {'F+3', 'F-3'}

_MO_HEADER = re.compile(r'^\s+(Alpha |Beta )?Molecular Orbital Coefficients:\s*$', re.M)
# '   1 1   O  1S    ...', '   2        2S ...', '  15        9D 0 ...', '  16        9D+1 ...', '  15        9XX ...'
_AO_LABEL = re.compile(r'^\s*(\d+)\s+(?:(\d+)\s+([A-Z][a-z]?)\s+)?(\d+)([A-Z]+)(?: ?(\d|[+-]\d))?')
_OCC_TOKEN = re.compile(r'^(?:\(\S+\)--)?[OV]$')
_FLOAT = re.compile(r'-?\d+\.\d+')
_FORTRAN = re.compile(r'-?\d+\.\d+D[+-]\d+')
_SHELL = re.compile(r'^\s*(S|SP|L|P|D|F|G|H|I)\s+(\d+)\s+(\d+\.\d+)')
_CENTER = re.compile(r'^\s*(\d+)\s+0\s*$')


def _ffloat(s: str) -> float:
    return float(s.replace('D', 'E'))


def _ao_code(letters: str, suffix: str):
    """(ORB_TYPE code, letter s/p/d/f, phase flip) of a Gaussian AO label such as 'PX', 'D' '+1', 'XXY'."""
    if letters in ('S', 'PX', 'PY', 'PZ', 'D', 'F'):
        key = letters + (suffix or '')
        if key not in _PURE:
            raise ParseError(f"Unknown Gaussian basis function label '{key}'")
        name = _PURE[key]
        return ORB_TYPE[name], name[0], key in _FLIP
    if set(letters) <= set('XYZ') and len(letters) in (2, 3):
        name = ('d_' if len(letters) == 2 else 'f_') + letters.lower()
        if name in ORB_TYPE:
            return ORB_TYPE[name], name[0], False
    raise ParseError(f"Gaussian basis functions '{letters}{suffix or ''}' (g or higher angular momentum) "
                     "are not supported: FROG builds orbitals with s, p, d and f functions only")


def _parse_mo_block(lines):
    """Raw MO block -> (labels per AO row, coefficients n_bf x n_mo, eigenvalues)."""
    labels = {}
    columns = {}
    energies = {}
    cols = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            break
        if all(t.isdigit() for t in tokens):
            cols = [int(t) - 1 for t in tokens]
            continue
        if all(_OCC_TOKEN.match(t) for t in tokens):
            continue
        if 'Eigenvalues --' in line:
            for c, v in zip(cols, _FLOAT.findall(line.split('--', 1)[1])):
                energies[c] = float(v)
            continue
        m = _AO_LABEL.match(line)
        if not m or not cols:
            break
        row = int(m.group(1)) - 1
        values = _FLOAT.findall(line[m.end():])
        if len(values) != len(cols):
            raise ParseError(f'Malformed MO coefficient line (expected {len(cols)} values): {line.strip()}')
        if row not in labels:
            labels[row] = (m.group(2), m.group(4), m.group(5), m.group(6))
        for c, v in zip(cols, values):
            columns.setdefault(c, {})[row] = float(v)

    n_bf, n_mo = len(labels), len(columns)
    if n_bf == 0 or n_mo == 0 or len(energies) != n_mo:
        raise ParseError('Incomplete MO coefficient block')
    if set(columns) != set(range(n_mo)) or set(labels) != set(range(n_bf)):
        raise ParseError('only some MOs are printed (default pop=regular): use pop=full to print all of them')
    coeff = np.zeros((n_bf, n_mo))
    for c, col in columns.items():
        for r, v in col.items():
            coeff[r, c] = v
    return [labels[r] for r in range(n_bf)], coeff, np.array([energies[c] for c in range(n_mo)])


def _ao_masks(labels):
    """mask_atoms (0-based str), mask_type_n (shell number per type, str), mask_orbital_type, phase signs."""
    mask_atoms, mask_type_n, mask_type, signs = [], [], [], []
    atom = None
    shell_count = defaultdict(int)
    seen = {}
    for atom_num, shell_no, letters, suffix in labels:
        if atom_num is not None:
            atom = int(atom_num) - 1
            shell_count.clear()
            seen.clear()
        if atom is None:
            raise ParseError('MO coefficient block does not start with an atom label')
        code, letter, flip = _ao_code(letters, suffix)
        if (shell_no, letter) not in seen:
            shell_count[letter] += 1
            seen[(shell_no, letter)] = shell_count[letter]
        mask_atoms.append(str(atom))
        mask_type_n.append(str(seen[(shell_no, letter)]))
        mask_type.append(code)
        signs.append(-1.0 if flip else 1.0)
    return np.array(mask_atoms), np.array(mask_type_n), np.array(mask_type), np.array(signs)


def get_MOs(fl):
    """MO data of the last 'Molecular Orbital Coefficients' block(s), or None if not printed.

    Returns a dict with mask_atoms, mask_type_n, mask_orbital_type, signs (phase changes, already
    applied), 'alpha' = (coeff, energies) and, for unrestricted runs, 'beta'.
    """
    headers = list(_MO_HEADER.finditer(fl))
    if not headers:
        return None
    last = {}
    for h in headers:
        last[(h.group(1) or '').strip().lower() or 'restricted'] = h
    if 'alpha' in last and ('restricted' not in last or last['alpha'].start() > last['restricted'].start()):
        spins = ['alpha', 'beta'] if 'beta' in last else ['alpha']
    else:
        spins = ['restricted']

    out = {}
    for spin in spins:
        h = last[spin]
        labels, coeff, energies = _parse_mo_block(fl[h.end():].splitlines()[1:])
        if 'mask_atoms' not in out:
            out['mask_atoms'], out['mask_type_n'], out['mask_orbital_type'], out['signs'] = _ao_masks(labels)
        elif coeff.shape[0] != len(out['mask_atoms']):
            raise ParseError('Alpha and beta MO blocks have a different number of basis functions')
        out['beta' if spin == 'beta' else 'alpha'] = (coeff * out['signs'][:, None], energies)
    return out


def get_overlap(fl, signs=None):
    """AO overlap matrix printed by iop(3/33=1) ('*** Overlap ***'), with the f(±3) phase changes, or None."""
    marker = '*** Overlap ***'
    pos = fl.rfind(marker)
    if pos == -1:
        return None
    entries = {}
    cols = []
    for line in fl[pos + len(marker):].splitlines()[1:]:
        tokens = line.split()
        if not tokens:
            break
        if all(t.isdigit() for t in tokens):
            cols = [int(t) - 1 for t in tokens]
            continue
        if not tokens[0].isdigit() or not cols:
            break
        values = _FORTRAN.findall(line)
        if not values:
            break
        row = int(tokens[0]) - 1
        for c, v in zip(cols, values):
            entries[(row, c)] = _ffloat(v)
    if not entries:
        return None
    n = max(max(k) for k in entries) + 1
    S = np.zeros((n, n))
    for (r, c), v in entries.items():
        S[r, c] = S[c, r] = v
    if signs is not None and len(signs) == n:
        S *= np.outer(signs, signs)
    return S


def get_basis_coefficients(fl, atoms):
    """Basis set of the gfinput block as ORCA's basis_coeff: {element: {'1s': [[exp, coeff], ...], ...}}.

    SP (L) shells are split into an s and a p shell. Raises ParseError if two atoms of the same
    element carry different basis sets (density_builder keys the basis by element).
    """
    marker = 'AO basis set in the form of general basis input'
    pos = fl.rfind(marker)
    if pos == -1:
        return None

    per_center = {}
    center = None
    shells = None
    for line in fl[pos:].splitlines()[1:]:
        if not line.strip():
            break
        if line.strip() == '****':
            center = None
            continue
        m = _CENTER.match(line)
        if m and center is None:
            center = int(m.group(1)) - 1
            shells = per_center.setdefault(center, [])
            continue
        m = _SHELL.match(line)
        if m and center is not None:
            kind, scale = m.group(1), float(m.group(3))
            if kind in ('G', 'H', 'I'):
                raise ParseError(f'{kind} functions in the basis set are not supported')
            shells.append([kind, scale, []])
            continue
        values = _FORTRAN.findall(line)
        if values and shells:
            shells[-1][2].append([_ffloat(v) for v in values])

    basis = {}
    for center, shell_list in sorted(per_center.items()):
        el = str(atoms[center])
        element_basis = {}
        count = defaultdict(int)
        for kind, scale, prims in shell_list:
            prims = np.array(prims)
            exps = prims[:, 0] * scale ** 2
            parts = [('s', prims[:, 1]), ('p', prims[:, 2])] if kind in ('SP', 'L') else [(kind.lower(), prims[:, 1])]
            for letter, coeffs in parts:
                count[letter] += 1
                element_basis[f'{count[letter]}{letter}'] = np.column_stack((exps, coeffs))
        if el in basis:
            same = basis[el].keys() == element_basis.keys() and all(
                np.allclose(basis[el][k], element_basis[k]) for k in element_basis)
            if not same:
                raise ParseError(f'Atoms of element {el} carry different basis sets: FROG needs one '
                                 'basis per element to build orbitals and densities')
        else:
            basis[el] = element_basis
    return basis or None
