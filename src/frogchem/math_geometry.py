import numpy as np
from scipy.spatial import distance_matrix
from typing import List, Tuple
from frogchem.chemistry import get_atomic_weights, BOHR_TO_ANGSTROM

def calculate_distance(idx1: np.ndarray, idx2: np.ndarray) -> float: 
    """
    Calculate the Euclidean distance between two points represented by their position.
    """
    return np.linalg.norm(idx1 - idx2)


def calculate_angle(idx1: np.ndarray, idx2: np.ndarray, idx3: np.ndarray) -> float:
    """
    Calculate the angle (degrees) formed by three points, with idx2 at the vertex.

    Uses atan2(|u x v|, u . v), which stays accurate near 0 and 180 degrees
    (the law-of-cosines/arccos form loses all precision there).

    Raises:
        ValueError: if one of the two bonds has zero length.
    """
    u = np.asarray(idx1, dtype=float) - np.asarray(idx2, dtype=float)
    v = np.asarray(idx3, dtype=float) - np.asarray(idx2, dtype=float)
    if np.linalg.norm(u) < 1e-12 or np.linalg.norm(v) < 1e-12:
        raise ValueError("Angle undefined: coincident points (zero-length bond)")
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


def calculate_dihedral(idx1: np.ndarray, idx2: np.ndarray, idx3: np.ndarray, idx4: np.ndarray) -> float:
    """
    Calculate the dihedral angle formed by four points represented by their position.
    """
    b0 = -1.0 * (idx2 - idx1).astype(float)
    b1 = (idx3 - idx2).astype(float)
    b2 = (idx4 - idx3).astype(float)

    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def calculate_centroids(cube_data, threshold_percentile=100.0):
    """
    Calculate centroids of positive and negative data using charge-weighted approach
    (Le Bahers, Adamo, Ciofini, JCTC 2011, 7, 2498: q_CT, R+ and R- integrate all
    positive / negative voxels of the density difference).

    Args:
        cube_data: parsed cube dict. Cube values are always in atomic units (e/bohr^3);
            'units' only refers to the grid (origin / voxel vectors), so for
            'Angstrom' grids the voxel volume is converted to bohr^3 before integrating.
        threshold_percentile: voxels whose |value| is above this percentile are
            discarded as outliers. Default 100 = no trimming (standard definition);
            values < 100 bias q_CT and the dipole low.

    Returns centroids in Angstrom, charges in e, the CT distance in Angstrom and the
    'dipole_moment' q+ * d_CT in e*Angstrom.
    """
    data = cube_data['data']
    origin = cube_data['origin']
    voxel_vectors = cube_data['voxel_vectors']
    nx, ny, nz = cube_data['grid_size']
    unit = cube_data['units']
    voxel_volume = abs(np.linalg.det(voxel_vectors))
    if unit != 'Bohr':
        # grid in Angstrom, data in e/bohr^3 -> voxel volume in bohr^3
        voxel_volume /= BOHR_TO_ANGSTROM ** 3

    pos_data_all = data[data > 0]
    neg_data_all = data[data < 0]

    pos_threshold = np.percentile(pos_data_all, threshold_percentile) if len(pos_data_all) > 0 else 0
    neg_threshold = np.percentile(np.abs(neg_data_all), threshold_percentile) if len(neg_data_all) > 0 else 0

    pos_mask = (data > 0) & (data <= pos_threshold)
    neg_mask = (data < 0) & (np.abs(data) <= neg_threshold)

    n_pos_filtered = np.sum((data > pos_threshold) & (data > 0))
    n_neg_filtered = np.sum((np.abs(data) > neg_threshold) & (data < 0))

    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)
    iv, jv, kv = np.meshgrid(i, j, k, indexing='ij', sparse=True)
    cart_coords = origin + iv[..., None] * voxel_vectors[0] + jv[..., None] * voxel_vectors[1] + kv[..., None] * voxel_vectors[2]

    if np.any(pos_mask):
        pos_data = data[pos_mask]
        pos_coords = cart_coords[pos_mask]
        pos_charges = pos_data * voxel_volume
        total_pos_charge = np.sum(pos_charges)
        pos_centroid = np.sum(pos_charges[:, np.newaxis] * pos_coords, axis=0) / total_pos_charge
    else:
        pos_centroid = None
        total_pos_charge = 0

    if np.any(neg_mask):
        neg_data = np.abs(data[neg_mask])
        neg_coords = cart_coords[neg_mask]
        neg_charges = neg_data * voxel_volume
        total_neg_charge = np.sum(neg_charges)
        neg_centroid = np.sum(neg_charges[:, np.newaxis] * neg_coords, axis=0) / total_neg_charge
    else:
        neg_centroid = None
        total_neg_charge = 0

    if unit == 'Bohr':
        if pos_centroid is not None: pos_centroid *= BOHR_TO_ANGSTROM
        if neg_centroid is not None: neg_centroid *= BOHR_TO_ANGSTROM

    ct_distance = None
    if pos_centroid is not None and neg_centroid is not None:
        ct_distance = np.linalg.norm(pos_centroid - neg_centroid)

    return {
        'data': cube_data,
        'positive_centroid': pos_centroid,
        'negative_centroid': neg_centroid,
        'positive_charge': total_pos_charge,
        'negative_charge': total_neg_charge,
        'charge_transfer_distance': ct_distance,
        'dipole_moment': (ct_distance * total_pos_charge) if ct_distance is not None else None,
        'thresholds': {
            'positive_threshold': pos_threshold,
            'negative_threshold': neg_threshold,
            'filtered_points': {'positive': n_pos_filtered, 'negative': n_neg_filtered}
        }
    }


def calc_distance_matrix(geom: np.ndarray, atoms: np.ndarray, exclude_H : bool = True) -> np.ndarray:
    """
    Calculate the Euclidean distance matrix for a geometry.
    """
    if exclude_H:
        mask = np.char.upper(np.asarray(atoms, dtype=str)) != 'H'
        geo = np.array(geom)[mask]
    else:
        geo = np.array(geom)

    return distance_matrix(geo, geo)


def calc_rmsd(eig1: np.ndarray, eig2: np.ndarray) -> float:
    """
    Calculate the RMSD based on eigenvalues of the distance matrix.
    
    Args:
        eig1 (np.ndarray): Eigenvalues of the reference structure.
        eig2 (np.ndarray): Eigenvalues of the structure to compare.
        
    Returns:
        float: The Root Mean Square Deviation.
    """
    return np.sqrt(np.mean((eig1 - eig2)**2))


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length, returning the original if norm < 1e-10."""
    norm = np.linalg.norm(v)
    return v if norm < 1e-10 else v / norm


def get_rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Generate a 3x3 rotation matrix for rotation around an axis by theta radians."""
    axis = normalize(axis)
    a, b, c = axis
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    m = 1 - cos_t
    return np.array([
        [a*a*m + cos_t, a*b*m - c*sin_t, a*c*m + b*sin_t],
        [b*a*m + c*sin_t, b*b*m + cos_t, b*c*m - a*sin_t],
        [c*a*m - b*sin_t, c*b*m + a*sin_t, c*c*m + cos_t]
    ])


def get_reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Generate a 3x3 Householder reflection matrix across the plane perpendicular to normal."""
    normal = normalize(normal)
    return np.eye(3) - 2 * np.outer(normal, normal)


def calculate_center_of_mass(symbols: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate center of mass and total molecular mass."""
    masses = np.array([get_atomic_weights(s) for s in symbols])
    total_mass = np.sum(masses)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    return com, total_mass


def _inertia_tensor_from_masses(masses: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Compute 3x3 moment of inertia tensor from masses and centered coordinates."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    Ixx = np.sum(masses * (y**2 + z**2))
    Iyy = np.sum(masses * (x**2 + z**2))
    Izz = np.sum(masses * (x**2 + y**2))
    Ixy = -np.sum(masses * x * y)
    Ixz = -np.sum(masses * x * z)
    Iyz = -np.sum(masses * y * z)
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def get_inertia_tensor(symbols: List[str], positions: np.ndarray) -> np.ndarray:
    """Compute eigenvalues (ascending) and orthonormal eigenvectors (columns) of the inertia tensor.

    Uses eigh: the tensor is real symmetric, and the general eig solver returns
    non-orthogonal eigenvectors for degenerate moments (symmetric/spherical tops).
    """
    pos = np.array(positions)
    com, _ = calculate_center_of_mass(symbols, pos)
    masses = np.array([get_atomic_weights(s) for s in symbols])
    tensor = _inertia_tensor_from_masses(masses, pos - com)
    return np.linalg.eigh(tensor)


def align_molecule(symbols: List[str], positions: np.ndarray) -> np.ndarray:
    """
    Align the molecule so that its principal axes of inertia coincide with the
    Cartesian axes (X, Y, Z).
    
    Args:
        symbols: List of atomic symbols.
        positions: Array of Cartesian coordinates.
        
    Returns:
        np.ndarray: The aligned coordinates centered at the COM.
    """
    pos = np.array(positions)
    com, _ = calculate_center_of_mass(symbols, pos)
    coords = pos - com
    
    # Calculate inertia tensor to find principal axes
    evals, evecs = get_inertia_tensor(symbols, coords)
    
    # Sort axes by moment of inertia (Low -> High)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]
    
    # Ensure a right-handed coordinate system (determinant +1)
    if np.linalg.det(evecs) < 0:
        evecs[:, 0] *= -1
        
    # Rotate coordinates: New = Old * RotationMatrix
    aligned_coords = np.dot(coords, evecs)
    
    return aligned_coords