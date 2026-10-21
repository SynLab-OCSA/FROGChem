"""Grid manipulation utilities for Gaussian Cube files.

This module provides functions to perform arithmetic operations on volumetric
data sets loaded from Gaussian Cube files.
"""

from typing import Dict, Any, Union
import numpy as np
from scipy.spatial import cKDTree

def _check_same_lattice(grid1, grid2, op):
    """Voxel-by-voxel arithmetic only makes sense on identical grids (size, origin, voxels)."""
    if tuple(grid1['grid_size']) != tuple(grid2['grid_size']):
        raise ValueError(f"Grid sizes must match for {op}.")
    for key in ('origin', 'voxel_vectors'):
        if key in grid1 and key in grid2 and not np.allclose(grid1[key], grid2[key], atol=1e-5):
            raise ValueError(f"Grid {key.replace('_', ' ')} must match for {op}.")


def add_grids(grid1, grid2):
    _check_same_lattice(grid1, grid2, 'addition')
    new_data = grid1['data'] + grid2['data']
    result = grid1.copy()
    result.update({'title': 'Addition', 'comment': 'Sum of grids', 'data': new_data,
                   'data_range': (new_data.min(), new_data.max())})
    return result

def subtract_grids(grid1, grid2):
    _check_same_lattice(grid1, grid2, 'subtraction')
    new_data = grid2['data'] - grid1['data']
    result = grid1.copy()
    result.update({'title': 'Difference', 'comment': 'Difference of grids', 'data': new_data,
                   'data_range': (new_data.min(), new_data.max())})
    return result

def multiply_grid(grid, constant):
    new_data = grid['data'] * constant
    result = grid.copy()
    result.update({'title': 'Multiplication', 'comment': f'Scaled by {constant}', 'data': new_data,
                   'data_range': (new_data.min(), new_data.max())})
    return result


def integrate_density_voronoi(cube_data: Dict[str, Any]) -> np.ndarray:
    """
    Integrates volumetric data assigning each voxel to the nearest atom (Voronoi partition).

    Args:
        cube_data (Dict[str, Any]): Dictionary returned by frogchem.io.parse_gaussian_cube

    Returns:
        np.ndarray: Integrated value ∫ρ dV for each atom (voxel sum times the voxel
        volume, in the cube's units: e.g. electrons for a density in e/bohr³).
    """
    # Extract data from the cube dictionary
    nx, ny, nz = cube_data['grid_size']
    origin = cube_data['origin']
    voxel_vectors = cube_data['voxel_vectors']
    atoms_list = cube_data['atoms']
    data = cube_data['data']

    n_atoms = len(atoms_list)
    atom_coords = np.array([atom['coords'] for atom in atoms_list])

    # 1. Generate Voxel Coordinates
    # We create indices grid and transform to Cartesian coordinates
    # i, j, k indices
    i, j, k = np.indices((nx, ny, nz))
    
    # Reshape for vectorized calculation (N_points, 3)
    # Coord = Origin + i*Vx + j*Vy + k*Vz
    # Using broadcasting for efficiency
    grid_coords = (origin + 
                   (i[..., None] * voxel_vectors[0]) + 
                   (j[..., None] * voxel_vectors[1]) + 
                   (k[..., None] * voxel_vectors[2])).reshape(-1, 3)

    # 2. Nearest Neighbor Assignment (Voronoi)
    tree = cKDTree(atom_coords)
    # query returns (distances, atom_indices)
    _, nearest_atom_indices = tree.query(grid_coords, k=1)

    # 3. Integration
    # Flatten data to match grid_coords
    flat_density = data.flatten()
    
    # Sum density for each atom index and multiply by the voxel volume |vx·(vy×vz)|,
    # so that the result is ∫ρ dV (e.g. electrons or unpaired spins), not a raw voxel sum.
    voxel_volume = abs(np.linalg.det(np.asarray(voxel_vectors, dtype=float)))
    atom_integrals = np.bincount(nearest_atom_indices, weights=flat_density, minlength=n_atoms)

    return atom_integrals * voxel_volume