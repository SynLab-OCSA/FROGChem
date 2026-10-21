import re
import numpy as np

# Row of an orientation table: center, atomic number, [atomic type,] x, y, z
# (the 'Atomic Type' column is missing in old Gaussian versions)
_ROW = re.compile(r'^\s*\d+\s+(\d+)\s+(?:-?\d+\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')


def parse_orientation(fl, reg):
    """Atomic numbers and coordinates (Å) of the last orientation table.

    The first header of reg['geom_headers'] present in the file is used
    ('Input orientation:' when printed, otherwise 'Standard orientation:', ...),
    so a file with only standard orientations (e.g. opt + freq jobs) is read too.
    Returns (numbers, coords); empty arrays when no table is found.
    """
    header = next((h for h in reg['geom_headers'] if h in fl), None)
    if header is None:
        return np.array([], dtype=int), np.array([])

    numbers, coords = [], []
    dashes_seen = 0
    for line in fl.split(header)[-1].splitlines():
        if line.strip().startswith('---'):
            dashes_seen += 1
            if dashes_seen >= 3:        # dashes: above the column titles, below them, end of table
                break
            continue
        if dashes_seen == 2:
            match = _ROW.match(line)
            if not match:
                break
            numbers.append(int(match.group(1)))
            coords.append([float(match.group(i)) for i in (2, 3, 4)])
    return np.array(numbers, dtype=int), np.array(coords, dtype=float)


def get_geom(fl, reg):
    """Coordinates (Å) of the last orientation table (see parse_orientation)."""
    return parse_orientation(fl, reg)[1]
