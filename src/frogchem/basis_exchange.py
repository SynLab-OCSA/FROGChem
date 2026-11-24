"""Basis set from the Basis Set Exchange (https://www.basissetexchange.org) when the output does not print it.

ORCA prints the basis only with ``! PrintBasis`` and Gaussian only with ``gfinput``. Without them the
basis is rebuilt from its name (ORCA "Your calculation utilizes the basis: ...", Gaussian "Standard
basis: ...") with the ``basis_set_exchange`` package, in the layout of ``basis_coeff``:
``{element: {'1s': [[exp, coeff], ...], '2s': ..., '1p': ...}}``.

General contractions are split into one shell per coefficient column and SP shells into an s and a
p shell, in the order of the BSE entry (the order of the program's basis library). ECPs are ignored:
only the electron shells enter the orbitals. The basis is returned only if its shells match the AO
rows of the MO coefficients (``check_basis_vs_mask``), so a renamed or modified basis is never used.
"""

import logging
import re
from collections import Counter, defaultdict
from functools import lru_cache

import basis_set_exchange as bse
import numpy as np
import periodictable as pt

logger = logging.getLogger(__name__)

_LETTERS = 'spdfghi'


def _squash(name: str) -> str:
    return re.sub(r'[^0-9a-z*+()]', '', name.lower())


@lru_cache(maxsize=None)
def resolve_basis_name(name: str):
    """BSE name of a basis as printed by ORCA/Gaussian (e.g. 'def2SVP' -> 'def2-SVP'), or None."""
    # Gaussian drops the hyphen (def2SVP): accept a unique match ignoring '-', '_', spaces
    matches = [n for n in bse.get_all_basis_names() if _squash(n) == _squash(name)]
    return matches[0] if len(matches) == 1 else None


def _normalize_contraction(prims: np.ndarray, l: int) -> np.ndarray:
    """Unit-norm contraction of normalised primitives: what ORCA/Gaussian print and density_builder expects."""
    a, c = prims[:, 0], prims[:, 1]
    overlap = (2.0 * np.sqrt(np.outer(a, a)) / np.add.outer(a, a)) ** (l + 1.5)
    return np.column_stack((a, c / np.sqrt(c @ overlap @ c)))


def fetch_basis(name: str, elements):
    """basis_coeff dict of basis ``name`` for ``elements`` (symbols); ValueError if unknown or not covered."""
    bse_name = resolve_basis_name(name)
    if bse_name is None:
        raise ValueError(f"basis '{name}' not found in the Basis Set Exchange")
    numbers = {s: pt.elements.symbol(s).number for s in {str(e) for e in elements}}
    data = bse.get_basis(bse_name, elements=list(numbers.values()), header=False)

    basis = {}
    for sym, z in numbers.items():
        shells = data['elements'].get(str(z), {}).get('electron_shells')
        if not shells:
            raise ValueError(f"basis '{bse_name}' has no functions for {sym}")
        contractions = []
        for shell in shells:
            exps = np.array(shell['exponents'], dtype=float)
            ams = shell['angular_momentum']
            # SP shells: one am per coefficient row; general contractions: one am, many rows
            for am, c in (zip(ams, shell['coefficients']) if len(ams) > 1
                          else ((ams[0], c) for c in shell['coefficients'])):
                c = np.array(c, dtype=float)
                keep = c != 0.0  # general contractions pad with zeros
                contractions.append((am, _normalize_contraction(np.column_stack((exps[keep], c[keep])), am)))
        # ORCA/Gaussian order shells of one am from tightest to most diffuse; BSE keeps the source
        # order (e.g. def2-TZVP Cl: d 0.339 before d 4.61/1.011)
        contractions.sort(key=lambda t: (t[0], -t[1][:, 0].max()))
        count = defaultdict(int)
        element_basis = {}
        for am, prims in contractions:
            count[am] += 1
            element_basis[f'{count[am]}{_LETTERS[am]}'] = prims
        basis[sym] = element_basis
    return basis


def check_basis_vs_mask(calc, basis) -> bool:
    """True if every atom has, per angular momentum, as many shells in ``basis`` as AO blocks in the MOs."""
    # imported here: frogchem.orca_parser imports this module
    from frogchem.orca_parser.orb_type import ORB_POLINOMIAL, CARTESIAN_CODES
    cart = set(CARTESIAN_CODES.values())
    size = {'s': 1, 'p': 3, 'd': 5, 'f': 7}
    shells = Counter()
    for a, code in zip(calc.mask_atoms, calc.mask_orbital_type):
        letter = ORB_POLINOMIAL[int(code)][0][0]
        n = {'d': 6, 'f': 10}[letter] if int(code) in cart else size[letter]
        shells[int(a), letter] += 1 / n
    for a in {a for a, _ in shells}:
        el = str(calc.atoms[a])
        have = Counter(k.lstrip('0123456789') for k in basis.get(el, {}))
        if {l: round(n) for (b, l), n in shells.items() if b == a} != dict(have):
            return False
    return True


def basis_from_bse(calc, name: str):
    """Basis of ``calc`` from the BSE, checked against its MO rows; None (with a warning) if unusable."""
    try:
        basis = fetch_basis(name, calc.atoms)
    except Exception as e:  # unknown name/element must not break parsing
        logger.warning(f"{calc.fname}: basis not printed and not available from the Basis Set Exchange: {e}")
        return None
    if not check_basis_vs_mask(calc, basis):
        logger.warning(f"{calc.fname}: basis '{name}' from the Basis Set Exchange does not match the "
                       "MO coefficients (modified basis?): print it with ! PrintBasis (ORCA) or gfinput (Gaussian)")
        return None
    logger.info(f"{calc.fname}: basis '{name}' not printed, taken from the Basis Set Exchange")
    return basis
