"""Gaussian frequency section: frequencies, IR intensities and normal-mode displacement vectors.

NOTE: get_freq, get_ir_intensities, get_normal_modes, align_normal_modes, to_displacements and the
private helpers are copied verbatim in ensemble_analyzer/parsers/vibrations.py (with a
``gaussian_`` prefix). Keep the two copies in sync.
"""
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Orientation-table row: center, atomic number, [atomic type,] x, y, z
_ORIENT_ROW = re.compile(r'^\s*\d+\s+(\d+)\s+(?:-?\d+\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')
# Row of a normal-mode block: atom index, atomic number, then x y z for each mode of the block
_MODE_ROW = re.compile(r'^\s*(\d+)\s+(\d+)((?:\s+[-+]?\d+\.\d+)+)\s*$')
_FLOAT = re.compile(r'[-+]?\d+\.\d+')


def _freq_block(fl, reg):
    # last frequency section (with freq=HPModes the standard 3-column section is printed last)
    return fl.split(reg['freq_section'])[-1].split(reg['freq_end'])[0]


def get_freq(fl, reg):
    if reg['freq_section'] not in fl:
        return np.array([])

    block = _freq_block(fl, reg)
    freq = np.array([float(x) for m in re.findall(reg['freq_pattern'], block) for x in m.split()])
    return freq


def get_ir_intensities(fl, reg):
    if reg['freq_section'] not in fl:
        return np.zeros(shape=(1, 2))

    block = _freq_block(fl, reg)
    freq = np.array([float(x) for m in re.findall(reg['freq_pattern'], block) for x in m.split()])
    ir = np.array([float(x) for m in re.findall(reg['ir_pattern'], block) for x in m.split()])
    if len(freq) == 0 or len(ir) == 0:
        return np.zeros(shape=(1, 2))
    return np.column_stack((freq, ir))


def get_normal_modes(fl, reg, N):
    """Cartesian displacement vectors of the normal modes, in Gaussian's *standard orientation*.

    Returns an (N, n_modes) matrix, N = 3 * n_atoms rows (x0, y0, z0, x1, ...), one column per
    vibrational mode in the order of get_freq (same layout as orca_parser.hessian.get_normal_modes,
    but Gaussian prints no translations/rotations). Read from the standard 3-modes-per-block table
    ('Atom  AN  X Y Z ...'). Use align_normal_modes to express them in the frame of the parsed
    geometry. Raises ValueError when the file has no frequency section.
    """
    if reg['freq_section'] not in fl:
        raise ValueError("Frequency section not found")

    columns = []        # one (N,) vector per mode
    block_modes = 0     # modes in the current block (from its 'Frequencies --' line)
    in_table = False
    for line in _freq_block(fl, reg).splitlines():
        freq_match = re.search(reg['freq_pattern'], line)
        if freq_match:
            block_modes = len(freq_match.group(1).split())
            block = [np.zeros(N) for _ in range(block_modes)]
            columns.extend(block)
            in_table = False
            continue
        if block_modes and line.split()[:2] == ['Atom', 'AN']:
            in_table = True
            continue
        if not in_table:
            continue
        row = _MODE_ROW.match(line)
        if row is None:
            in_table = False
            continue
        atom = int(row.group(1)) - 1
        values = [float(x) for x in _FLOAT.findall(row.group(3))]
        if len(values) != 3 * block_modes or 3 * atom + 2 >= N:
            logger.warning("Skipping malformed normal-mode row: %r", line)
            continue
        for k in range(block_modes):
            block[k][3 * atom:3 * atom + 3] = values[3 * k:3 * k + 3]

    if not columns:
        return np.zeros((N, 0))
    return np.column_stack(columns)


def _last_orientation(fl, header):
    """Coordinates (Å) of the last orientation table with the given header, or None."""
    if header not in fl:
        return None
    coords, dashes = [], 0
    for line in fl.split(header)[-1].splitlines():
        if line.strip().startswith('---'):
            dashes += 1
            if dashes >= 3:
                break
            continue
        if dashes == 2:
            match = _ORIENT_ROW.match(line)
            if not match:
                break
            coords.append([float(match.group(i)) for i in (2, 3, 4)])
    return np.array(coords, dtype=float) if coords else None


def _kabsch(P, Q):
    """Proper rotation R and RMSD (Å) such that (P - <P>) @ R ≈ (Q - <Q>)."""
    Pc, Qc = P - P.mean(axis=0), Q - Q.mean(axis=0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    rmsd = float(np.sqrt(np.mean(np.sum((Pc @ R - Qc) ** 2, axis=1))))
    return R, rmsd


def align_normal_modes(fl, normal_modes, target='Input orientation:', source='Standard orientation:', tol=1e-2):
    """Rotate (3N, n_modes) normal modes from the standard orientation to the `target` frame.

    Gaussian prints the normal coordinates in the standard orientation, while the parsers return
    the input-orientation geometry when it is printed. The rotation is the Kabsch fit of the last
    standard-orientation table onto the last `target` table; when one of the two tables is
    missing (e.g. NoSymm jobs print only the input orientation) the modes are returned unchanged.
    """
    std = _last_orientation(fl, source)
    tgt = _last_orientation(fl, target)
    if std is None or tgt is None or std.shape != tgt.shape or normal_modes.shape[0] != std.size:
        return normal_modes
    R, rmsd = _kabsch(std, tgt)
    if rmsd > tol:
        logger.warning("Standard and input orientations do not superimpose (RMSD %.3g Å): "
                       "normal modes left in the standard orientation", rmsd)
        return normal_modes
    n_atoms = std.shape[0]
    disp = normal_modes.T.reshape(-1, n_atoms, 3) @ R
    return np.ascontiguousarray(disp.reshape(-1, 3 * n_atoms).T)


def to_displacements(normal_modes, n_atoms):
    """(3N, n_modes) normal-mode matrix -> (n_modes, n_atoms, 3) displacement vectors."""
    normal_modes = np.asarray(normal_modes, dtype=np.float64)
    return np.ascontiguousarray(normal_modes[:3 * n_atoms].T).reshape(-1, n_atoms, 3)
