"""Orbital files written by cleave (``almo``, ``cdft``): the ``cleave-orbitals`` JSON.

A file holds the geometry (Bohr), the basis set per element and several named orbital
sets (``full``, ``polarized``, ``frag1``, ``cdft``, ...). :class:`CleaveOrbitals` exposes one
set at a time with the attributes of a parsed ORCA/Gaussian calculation (``atoms``, ``geom``,
``mask_atoms``, ``mask_orbital_type``, ``orbital_coefficients``, ``occupation``, ...), so
``frog-density`` and ``frog-mo`` work on it unchanged; ``select(name)`` switches set.

AO order is PySCF's spherical one: atom by atom, shell by shell, contraction by contraction,
the 2l+1 components (p: x, y, z; d/f: m = -l ... l), coefficients of normalized primitives.
"""
import copy
import gzip
import json
from pathlib import Path

import numpy as np

from frogchem.exceptions import InputError, ParseError
from frogchem.printer import print_warning

FORMAT = 'cleave-orbitals'
VERSION = 1
_AU = 1.8897259886  # Å → Bohr (as density_builder)

# PySCF (l, m) → FROGChem angular code (orca_parser.orb_type.ORB_TYPE)
_CODES = {
    0: [0],
    1: [1, 2, 3],                      # px py pz
    2: [8, 6, 4, 5, 7],                # m = -2..2: dxy dyz dz2 dxz dx2y2
    3: [15, 13, 11, 9, 10, 12, 14],    # m = -3..3: f-3 f-2 f-1 f0 f+1 f+2 f+3
}
# ponytail: density_builder evaluates f(+3)/f(-3) with ORCA's phase, opposite to the real solid
# harmonics of PySCF; the coefficient rows of those functions change sign on load (checked
# against PySCF's eval_gto on def2-TZVP: every AO agrees to 1e-15 after the flip).
_FLIPPED = {14, 15}


def _read(path) -> dict:
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    try:
        with opener(path, 'rt', encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise ParseError(f'{path}: not a readable JSON file ({e})') from e
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ParseError(f'{path}: not a cleave orbital file (format {FORMAT!r})')
    if data.get('version', 0) > VERSION:
        raise ParseError(f'{path}: {FORMAT} version {data["version"]} is newer than this reader '
                         f'(version {VERSION}); update frogchem')
    return data


def is_open_shell(calc) -> bool:
    """Separate α/β orbitals: a cleave set says so itself, otherwise multiplicity > 1."""
    # ponytail: a broken-symmetry singlet (cleave CDFT, mult 1) is unrestricted all the same.
    # Read from the instance dict: getattr would trigger the lazy parse of an ORCA/Gaussian
    # Calculation (parser_base.__getattr__) and overwrite attributes set before it.
    unrestricted = getattr(calc, '__dict__', {}).get('unrestricted')
    if unrestricted is not None:
        return bool(unrestricted)
    return getattr(calc, 'mult', 1) != 1 and hasattr(calc, 'alpha_coeff')


class CleaveOrbitals:
    """One orbital set of a cleave orbital file, shaped like a parsed calculation."""

    def __init__(self, path, set_name=None):
        self.fname = str(path)
        self.data = _read(path)
        d = self.data
        self.program = d.get('program', 'cleave')
        self.atoms = [a['symbol'] for a in d['atoms']]
        self.geom = np.array([a['coords'] for a in d['atoms']], dtype=float) / _AU
        self.charge = int(d.get('charge', 0))
        self.mult = int(d.get('multiplicity', 1))
        self.fragments = d.get('fragments', [])
        self.ecp = d.get('ecp', {})
        if self.ecp:
            print_warning(f'{Path(path).name}: ECP on atoms {", ".join(self.ecp)}: densities are '
                          f'valence only ({sum(self.ecp.values())} core electrons not included)')
        self._build_basis()
        self.set_names = list(d['sets'])
        if not self.set_names:
            raise ParseError(f'{path}: no orbital sets')
        self.default_set = d.get('default_set') if d.get('default_set') in d['sets'] else self.set_names[0]
        self.select(set_name or self.default_set)

    # ------------------------------------------------------------------ basis
    def _build_basis(self):
        b = self.data['basis']
        if not b.get('spherical', True):
            raise ParseError(f'{self.fname}: Cartesian basis functions are not supported')
        table, codes, atoms = [], [], []
        self.basis_coeff = {}
        for ia, label in enumerate(b['atoms']):
            counts = {}
            for sh in b['elements'][label]:
                l = int(sh['l'])
                if l not in _CODES:
                    raise ParseError(f'{self.fname}: l = {l} functions are not supported (s, p, d, f only)')
                exps = [float(e) for e in sh['exponents']]
                cs = np.asarray(sh['coefficients'], dtype=float).reshape(len(exps), -1)
                for col in cs.T:
                    prims = [(e, float(c)) for e, c in zip(exps, col)]
                    letter = 'spdf'[l]
                    counts[letter] = counts.get(letter, 0) + 1
                    self.basis_coeff.setdefault(label, {})[f'{counts[letter]}{letter}'] = prims
                    for code in _CODES[l]:
                        table.append({'atom_idx': ia, 'element': self.atoms[ia], 'ang_code': code,
                                      'primitives': prims})
                        codes.append(code)
                        atoms.append(ia)
        if len(table) != len(b['ao_labels']):
            raise ParseError(f'{self.fname}: basis gives {len(table)} functions, '
                             f'{len(b["ao_labels"])} AO labels')
        self.basis_table = table
        self.ao_labels = b['ao_labels']
        self.mask_atoms = np.array(atoms)
        self.mask_orbital_type = np.array(codes)
        self._sign = np.where(np.isin(codes, list(_FLIPPED)), -1.0, 1.0)
        self.nao = len(table)

    # ------------------------------------------------------------------- sets
    def sets(self) -> list:
        """(name, kind, channels, n orbitals, orthonormal, description) of every set."""
        rows = []
        for name, st in self.data['sets'].items():
            spins = '/'.join(ch['spin'] for ch in st['channels'])
            nmo = '/'.join(str(len(ch['coefficients'][0]) if ch['coefficients'] else 0)
                           for ch in st['channels'])
            rows.append([name, st['kind'], spins, nmo, 'yes' if st['orthonormal'] else 'no',
                         st.get('description', '')])
        return rows

    def _coeff(self, ch) -> np.ndarray:
        c = np.asarray(ch['coefficients'], dtype=float).reshape(self.nao, -1)
        return c * self._sign[:, None]

    def select(self, name: str) -> 'CleaveOrbitals':
        """Make set ``name`` the current orbitals."""
        if name not in self.data['sets']:
            raise InputError(f'No orbital set {name!r} in {self.fname} '
                             f'(available: {", ".join(self.set_names)})')
        st = self.data['sets'][name]
        self.set_name, self.set_info = name, st
        chans = {ch['spin']: ch for ch in st['channels']}
        self.unrestricted = 'both' not in chans
        self.has_occupations = all(ch.get('occupations') is not None for ch in st['channels'])
        for attr in ('alpha_coeff', 'beta_coeff', 'alpha_occ', 'beta_occ'):
            self.__dict__.pop(attr, None)

        def parts(ch):
            if ch is None:                    # e.g. COVP pairs in one spin channel only
                return np.zeros((self.nao, 0)), np.zeros(0), np.zeros(0), []
            c = self._coeff(ch)
            n = c.shape[1]
            occ = np.zeros(n) if ch.get('occupations') is None else np.asarray(ch['occupations'], float)
            en = np.full(n, np.nan) if ch.get('energies') is None else np.asarray(ch['energies'], float)
            return c, occ, en, ch.get('labels') or []

        if not self.unrestricted:
            c, occ, en, labels = parts(chans['both'])
            self.orbital_coefficients, self.occupation, self.orbital_labels = c, occ, labels
        else:
            ca, oa, ea, la = parts(chans.get('alpha'))
            cb, ob, eb, lb = parts(chans.get('beta'))
            self.alpha_coeff, self.beta_coeff, self.alpha_occ, self.beta_occ = ca, cb, oa, ob
            self.orbital_coefficients = np.hstack((ca, cb))
            self.occupation = np.concatenate((oa, ob))
            self.orbital_labels = la + lb
            en = np.concatenate((ea, eb))
        self.orbital_energies = None if en.size == 0 or np.all(np.isnan(en)) else en
        return self

    # ---------------------------------------------------------------- density
    def density_view(self) -> 'CleaveOrbitals':
        """Orbitals whose Σ n|φ|² is the density of the current set.

        Orthonormal sets are returned as they are; ALMO sets (non-orthogonal across fragments)
        are Löwdin-orthogonalized within each channel, C σ^-½ with σ = CᵀSC from the file: the
        density C σ⁻¹ n Cᵀ is unchanged (ALMO occupations are uniform in a channel).
        """
        if not self.has_occupations:
            raise InputError(f'Set {self.set_name!r} has no occupations ({self.set_info["kind"]}: '
                             'orbitals only); build orbital cubes with --orbital instead')
        if self.set_info['orthonormal']:
            return self
        view = copy.copy(self)
        chans = {ch['spin']: ch for ch in self.set_info['channels']}

        def lowdin(c, ch):
            if ch is None or c.shape[1] == 0:
                return c
            w, u = np.linalg.eigh(np.asarray(ch['overlap'], dtype=float))
            return c @ (u / np.sqrt(w)) @ u.T

        if not self.unrestricted:
            view.orbital_coefficients = lowdin(self.orbital_coefficients, chans['both'])
        else:
            view.alpha_coeff = lowdin(self.alpha_coeff, chans.get('alpha'))
            view.beta_coeff = lowdin(self.beta_coeff, chans.get('beta'))
            view.orbital_coefficients = np.hstack((view.alpha_coeff, view.beta_coeff))
        return view

    def __repr__(self):
        return (f'CleaveOrbitals({Path(self.fname).name!r}, set={self.set_name!r}, '
                f'{len(self.atoms)} atoms, {self.nao} AOs)')
