import periodictable as pt

__all__ = [
    "get_atomic_radius", "get_atomic_weights", "atomic_number_to_symbol",
    "HARTREE_TO_KCAL", "HARTREE_TO_EV", "BOHR_TO_ANGSTROM", "ANGSTROM_TO_BOHR",
]

# ponytail: canonical physical conversion constants (CODATA recommendations)
HARTREE_TO_KCAL: float = 627.5094740631
HARTREE_TO_EV: float = 27.211386245988
BOHR_TO_ANGSTROM: float = 0.529177210903
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM


def get_atomic_radius(symbol) -> float:
    """Get the atomic radius of an element

    Args:
        symbol (str): Atom symbol

    Returns:
        float: covalent radius of the element in A
    """
    return getattr(pt, symbol).covalent_radius


def get_atomic_weights(symbol) -> float: 
    """Get the atomic weight of an element

    Args:
        symbol (str): Atom symbol

    Returns:
        float: atomic weight of the atom
    """
    return getattr(pt, symbol).mass


def atomic_number_to_symbol(n: int) -> str:
    """Convert atomic number to element symbol.

    Args:
        n (int): Atomic number (0 returns 'X')

    Returns:
        str: Element symbol or 'X' for unknown
    """
    if n <= 0:
        return 'X'
    try:
        return pt.elements[n].symbol
    except (AttributeError, IndexError):
        return 'X'