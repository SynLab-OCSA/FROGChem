import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from itertools import groupby

__all__ = [
    "read_xyz", "write_xyz", "write_crest_constraints",
    "compact_list_fragments", "parse_gaussian_cube", "write_gaussian_cube",
    "read_nbo_plot_37", "rdkit_mol", "open_text",
]


def open_text(path):
    """Text handle on a plain or .gz file (UTF-8, undecodable bytes replaced)."""
    import gzip
    opener = gzip.open if str(path).lower().endswith('.gz') else open
    return opener(path, 'rt', encoding='utf-8', errors='replace')


def _parse_xyz_block(block: List[str]) -> Tuple[List[str], np.ndarray, str]:
    """Parse a single XYZ block.
    
    Args:
        block (list[str]): List of strings representing one XYZ structure block
        
    Returns:
        Tuple of (atoms, positions, comment)
    """
    if len(block) < 3:
        raise ValueError("XYZ block must have at least 3 lines")
    
    comment = block[1].strip()
    atoms = []
    positions = []
    
    for line in block[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    
    return atoms, np.array(positions, dtype=float), comment


def read_xyz(fname: str) -> Tuple[List[List[str]], List[np.ndarray], List[str]]:
    """Read a multi-XYZ file and return atoms, positions and comments.

    Args:
        fname: XYZ filename

    Returns:
        Tuple of (atoms_list, positions_list, comments_list) for each geometry
    """
    try:
        with open_text(fname) as f:
            lines = [line.rstrip() for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"File {fname} not found")
    
    atoms_list = []
    positions_list = []
    comments_list = []

    # Each frame carries its own atom count, so frames may hold different molecules.
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            n_atoms = int(lines[i].split()[0])
        except ValueError:
            raise ValueError(
                f"Invalid XYZ file format: line {i + 1} of {fname} should contain the number of atoms"
            )
        block_size = n_atoms + 2
        if i + block_size > len(lines):
            import warnings
            warnings.warn(f"{fname}: incomplete last frame (line {i + 1}) ignored", stacklevel=2)
            break

        atoms, positions, comment = _parse_xyz_block(lines[i:i + block_size])
        atoms_list.append(atoms)
        positions_list.append(positions)
        comments_list.append(comment)
        i += block_size

    return atoms_list, positions_list, comments_list


def write_xyz(fname: str, 
              atoms_list: List[List[str]], 
              positions_list: List[np.ndarray], 
              comments_list: Optional[List[str]] = None) -> None:
    """Write a multi-XYZ file with multiple structures.

    Args:
        fname: XYZ filename to write
        atoms_list: List of lists containing atomic symbols for each structure
        positions_list: List of numpy arrays containing positions for each structure
        comments_list: Optional list of comments for each structure
        
    Raises:
        ValueError: If input lists have different lengths
    """
    if not atoms_list or not positions_list:
        raise ValueError("Atoms and positions lists cannot be empty")
    
    n = len(atoms_list)
    if n != len(positions_list):
        raise ValueError("Atoms and positions lists must have the same length")
    
    if comments_list is None:
        comments_list = [''] * n
    elif len(comments_list) != n:
        raise ValueError("Comments list must have the same length as atoms list")
    
    try:
        with open(fname, 'w') as f:
            for atoms, positions, comment in zip(atoms_list, positions_list, comments_list):
                if len(atoms) != len(positions):
                    raise ValueError("Number of atoms must match number of positions")
                
                # Write number of atoms
                f.write(f"{len(atoms)}\n")
                # Write comment
                f.write(f"{comment}\n")
                # Write atoms and positions
                for atom, pos in zip(atoms, positions):
                    f.write(f"{atom:>2s} {pos[0]:>12.6f} {pos[1]:>12.6f} {pos[2]:>12.6f}\n")
    except IOError as e:
        raise IOError(f"Error writing to file {fname}: {e}")

def write_crest_constraints(
    fname: str, 
    atom_indexes: List[int], 
    force_constant: float = 0.5, 
    reference_file: str = "coord.ref",
    metadyn_atoms: Optional[List[int]] = None,
    internals: Optional[List[Tuple[List[int], float]]] = None
) -> None:
    """Writes a CREST constraint file (.c).

    Generates the constraint block required by CREST, automatically compacting
    the atom indexes (e.g., "1, 2, 3" becomes "1-3").

    Args:
        fname (str): The output filename (usually 'constrain.c').
        atom_indexes (List[int]): List of 0-based atom indexes to constrain.
            They will be converted to 1-based for CREST automatically. If empty,
            no 'atoms:' and 'reference=' lines are written.
        force_constant (float, optional): The force constant for the constraint. 
            Defaults to 0.5.
        reference_file (str, optional): The name of the reference geometry file. 
            Defaults to "coord.ref".
        metadyn_atoms (List[int], optional): 0-based atoms included in the metadynamics
            bias ($metadyn block); the others are left out of it. None writes no block.
        internals (List[Tuple[List[int], float]], optional): (0-based atoms, value) pairs;
            2, 3 or 4 atoms give a distance (Å), angle or dihedral (degrees) constraint.
    """
    kinds = {2: 'distance', 3: 'angle', 4: 'dihedral'}

    with open(fname, 'w') as f:
        f.write('$constrain\n')
        if atom_indexes:
            compact_str = compact_list_fragments([i + 1 for i in atom_indexes], range_sep="-", sep=",")
            f.write(f'  atoms: {compact_str}\n')
        for idxs, value in internals or []:
            atoms_str = ', '.join(str(i + 1) for i in idxs)
            f.write(f'  {kinds[len(idxs)]}: {atoms_str}, {value:.4f}\n')
        f.write(f'  force constant={force_constant}\n')
        if atom_indexes:
            f.write(f'  reference={reference_file}\n')
        if metadyn_atoms is not None:
            mtd_str = compact_list_fragments([i + 1 for i in metadyn_atoms], range_sep="-", sep=",")
            f.write('$metadyn\n')
            f.write(f'  atoms: {mtd_str}\n')
        f.write('$end\n')


def compact_list_fragments(indices, range_sep: str = ":", sep: str = " "):
    """Compact a list of integers into ranges, e.g. [1, 2, 3, 5] -> "1:3 5".

    Use ``range_sep="-", sep=","`` for the xtb/CREST list syntax ("1-3,5").
    """
    if not indices:
        return ""
    if isinstance(indices[0], (list, tuple, set)):
        return sep.join(compact_list_fragments(list(sub), range_sep, sep) for sub in indices)
    tmp = []
    for _, group in groupby(enumerate(sorted(set(indices))), lambda x: x[1] - x[0]):
        g = list(group)
        start, end = g[0][1], g[-1][1]
        tmp.append(f"{start}" if start == end else f"{start}{range_sep}{end}")
    return sep.join(tmp)


def parse_gaussian_cube(filename):
    """
    Complete parsing of a Gaussian cube file
    
    Returns:
        dict: Dictionary with all cube file information
    """
    with open_text(filename) as f:
        lines = f.readlines()
    
    # Metadata
    title = lines[0].strip()
    comment = lines[1].strip()
    
    # Number of atoms and origin
    line3 = lines[2].split()
    natoms = int(line3[0])
    origin = np.array([float(x) for x in line3[1:4]])
    
    # Grid dimensions and voxel vectors
    nx = int(lines[3].split()[0])
    vx = np.array([float(x) for x in lines[3].split()[1:4]])
    
    ny = int(lines[4].split()[0])
    vy = np.array([float(x) for x in lines[4].split()[1:4]])
    
    nz = int(lines[5].split()[0])
    vz = np.array([float(x) for x in lines[5].split()[1:4]])
    
    # Determine units: positive = Bohr, negative = Angstrom
    units = "Bohr" if nx > 0 else "Angstrom"
    
    # Use absolute values for grid dimensions
    nx, ny, nz = abs(nx), abs(ny), abs(nz)
    
    voxel_vectors = np.array([vx, vy, vz])
    
    # Atoms
    atoms = []
    for i in range(abs(natoms)):
        line = lines[6 + i].split()
        atoms.append({
            'atomic_number': int(line[0]),
            'charge': float(line[1]),
            'coords': np.array([float(x) for x in line[2:5]])
        })
    
    # Multicube: negative natoms, then the orbital-id record "N id1 ... idN" (may wrap over lines)
    orbital_ids = None
    data_start = 6 + abs(natoms)
    if natoms < 0:
        first = lines[data_start].split()
        k = int(first[0])
        ids = first[1:]
        data_start += 1
        while len(ids) < k:
            ids += lines[data_start].split()
            data_start += 1
        orbital_ids = [int(i) for i in ids[:k]]

    # Volumetric data
    # ponytail: fast C-buffer stream conversion with np.fromstring avoids millions of Python float allocations
    data_text = " ".join(lines[data_start:])
    values = np.fromstring(data_text, sep=' ', dtype=float)
    if orbital_ids is None:
        data = values.reshape(nx, ny, nz)
    else:
        # one block per grid point: the N orbital values are contiguous (orbital index fastest)
        data = values.reshape(nx, ny, nz, len(orbital_ids))
    
    return {
        'title': title,
        'comment': comment,
        'natoms': natoms,
        'origin': origin,
        'grid_size': (nx, ny, nz),
        'voxel_vectors': voxel_vectors,
        'units': units,
        'atoms': atoms,
        'data': data,
        'data_range': (data.min(), data.max()),
        'orbital_ids': orbital_ids,
    }


def write_gaussian_cube(filename, cube_data):
    """
    Write a Gaussian cube file
    
    Args:
        filename: output filename
        cube_data: dictionary with cube file data (from parse_gaussian_cube or custom)
                  Required keys: 'title', 'comment', 'natoms', 'origin', 'grid_size',
                  'voxel_vectors', 'atoms', 'data', 'units' (optional, default='Bohr')
    """
    with open(filename, 'w') as f:
        # Header lines
        f.write(f"{cube_data['title']}\n")
        f.write(f"{cube_data['comment']}\n")
        
        # Number of atoms and origin
        orbital_ids = cube_data.get('orbital_ids')
        natoms = cube_data['natoms']
        if orbital_ids is not None:
            natoms = -abs(natoms)  # negative natoms announces the orbital-id record
        origin = cube_data['origin']
        f.write(f"{natoms:5d}{origin[0]:12.6f}{origin[1]:12.6f}{origin[2]:12.6f}\n")
        
        # Grid dimensions and voxel vectors
        nx, ny, nz = cube_data['grid_size']
        voxel_vectors = cube_data['voxel_vectors']
        units = cube_data.get('units', 'Bohr')
        
        # Apply sign convention: positive for Bohr, negative for Angstrom
        sign_factor = 1 if units == 'Bohr' else -1
        
        f.write(f"{nx * sign_factor:5d}{voxel_vectors[0,0]:12.6f}{voxel_vectors[0,1]:12.6f}{voxel_vectors[0,2]:12.6f}\n")
        f.write(f"{ny * sign_factor:5d}{voxel_vectors[1,0]:12.6f}{voxel_vectors[1,1]:12.6f}{voxel_vectors[1,2]:12.6f}\n")
        f.write(f"{nz * sign_factor:5d}{voxel_vectors[2,0]:12.6f}{voxel_vectors[2,1]:12.6f}{voxel_vectors[2,2]:12.6f}\n")
        
        # Atoms
        for atom in cube_data['atoms']:
            atomic_num = atom['atomic_number']
            charge = atom['charge']
            coords = atom['coords']
            f.write(f"{atomic_num:5d}{charge:12.6f}{coords[0]:12.6f}{coords[1]:12.6f}{coords[2]:12.6f}\n")
        
        # Multicube: orbital-id record after the atoms, Gaussian format 10I5: the count and the
        # 1-based MO numbers, 10 fields per line (count + 9 ids on the first line)
        if orbital_ids is not None:
            fields = [len(orbital_ids)] + [int(i) for i in orbital_ids]
            for start in range(0, len(fields), 10):
                f.write("".join(f"{v:5d}" for v in fields[start:start + 10]) + "\n")

        # Volumetric data: x, y, z order, orbital index fastest, 6 values per line
        _write_cube_values(f, np.asarray(cube_data['data'], dtype=float).reshape(-1))


def _write_cube_values(f, flat, per_line=6):
    """Write a flat array of cube values, per_line values per line in %13.5E format."""
    n_full = len(flat) - len(flat) % per_line
    if n_full:
        body = flat[:n_full].reshape(-1, per_line)
        f.write("\n".join("".join(f"{v:13.5E}" for v in row) for row in body) + "\n")
    rest = flat[n_full:]
    if len(rest):
        f.write("".join(f"{v:13.5E}" for v in rest) + "\n")


def read_nbo_plot_37(
    filepath: Union[str, Path],
    n_ao: Optional[int] = None,
    n_nbo: Optional[int] = None,
) -> np.ndarray:
    """Read AO-to-NBO transformation matrix from an NBO PLOT .37 file.

    In NBO PLOT output (.37 files), coefficients are written as sequential
    floating-point numbers mapping the AO basis to NBOs (column-major, Fortran order).

    Args:
        filepath: Path to the .37 file.
        n_ao: Number of AO basis functions (rows in the transformation matrix).
        n_nbo: Number of NBOs (columns in the transformation matrix).

    Returns:
        np.ndarray: 2D array of shape (n_ao, n_nbo) or 1D array if dimensions cannot be determined.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"NBO plot file not found: {path}")

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        tokens = f.read().split()

    vals = np.array([float(x) for x in tokens], dtype=np.float64)

    if n_ao is not None and n_nbo is not None:
        expected = n_ao * n_nbo
        if len(vals) >= expected:
            return vals[:expected].reshape((n_ao, n_nbo), order='F')
    elif n_ao is not None and n_ao > 0:
        n_cols = len(vals) // n_ao
        if n_cols > 0:
            return vals[:n_ao * n_cols].reshape((n_ao, n_cols), order='F')

    dim = int(np.round(np.sqrt(len(vals))))
    if dim * dim == len(vals) and dim > 0:
        return vals.reshape((dim, dim), order='F')

    return vals


def rdkit_mol(atoms, geom, charge: int = 0, strict: bool = True):
    """RDKit molecule with bonds perceived from the 3D coordinates.

    strict=False keeps the molecule without bonds when the perception fails.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (symbol, xyz) in enumerate(zip(atoms, geom)):
        mol.AddAtom(Chem.Atom(str(symbol)))
        conf.SetAtomPosition(i, tuple(float(c) for c in xyz))
    mol = mol.GetMol()
    mol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception:
        if strict:
            raise
    return mol
