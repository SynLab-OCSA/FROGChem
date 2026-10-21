"""Unified file loading for ORCA, Gaussian, and XYZ formats."""
from pathlib import Path

from frogchem.exceptions import ParseError


def load_calculation(path):
    """Load a computation output file.

    Returns an ORCA Calculation, Gaussian GaussianCalculation, or a
    simple namespace with (atoms, geom) for XYZ files.

    Raises ParseError if the file cannot be parsed by any supported parser.
    """
    path_lower = str(path).lower()
    if path_lower.endswith(('.xyz', '.xyz.gz')):
        from frogchem.io import read_xyz
        from types import SimpleNamespace
        atoms_list, coords_list, _ = read_xyz(path)
        return SimpleNamespace(atoms=atoms_list[0], geom=coords_list[0])

    # .out, .log → use frogchem.read format detection and validate
    try:
        import frogchem as frog
        calc = frog.read(path)
        _ = calc.atoms
        return calc
    except Exception as e:
        raise ParseError(
            f"Cannot parse {path} with any supported parser: {e}"
        ) from e


def load_structures(path):
    """Load atoms and geometry from any supported format.

    Returns list of (atoms, geom) tuples — one per frame/conformer.
    For computation outputs (.out/.log), returns a single frame.
    For multiXYZ, returns all frames.
    """
    path = str(path)
    path_lower = path.lower()
    if path_lower.endswith(('.xyz', '.xyz.gz')):
        from frogchem.io import read_xyz
        atoms_list, coords_list, _ = read_xyz(path)
        return list(zip(atoms_list, coords_list))

    calc = load_calculation(path)
    return [(calc.atoms, calc.geom)]


def load_frames(path):
    """(atoms, coords, comment) for every structure; comments are kept for XYZ input."""
    from frogchem.io import read_xyz as _read_xyz
    path = str(path)
    if path.lower().endswith(('.xyz', '.xyz.gz')):
        return list(zip(*_read_xyz(path)))
    return [(atoms, geom, Path(path).name) for atoms, geom in load_structures(path)]
