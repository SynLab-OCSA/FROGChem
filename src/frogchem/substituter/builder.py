"""
Geometric fragment builder and clash resolution for molecular substitution.

Andrea Pellegrini, 2026
"""
from typing import Optional, Tuple, List, Dict
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from scipy.spatial.distance import cdist


# Canonical standard single bond lengths in Angstroms (experimental / standard DFT averages)
STANDARD_BOND_LENGTHS: Dict[Tuple[str, str], float] = {
    # Carbon - Carbon
    ("C", "C"): 1.54,
    # Carbon - Hydrogen
    ("C", "H"): 1.09,
    # Carbon - Halogens
    ("C", "F"): 1.35,
    ("C", "Cl"): 1.77,
    ("C", "Br"): 1.94,
    ("C", "I"): 2.14,
    # Carbon - Heteroatoms
    ("C", "O"): 1.43,
    ("C", "N"): 1.45,
    ("C", "P"): 1.84,
    ("C", "S"): 1.81,
    ("C", "Si"): 1.87,
    ("C", "B"): 1.56,
    # Phosphorus
    ("P", "P"): 2.21,
    ("P", "O"): 1.63,
    ("P", "H"): 1.42,
    ("P", "F"): 1.56,
    ("P", "Cl"): 2.04,
    ("P", "N"): 1.68,
    ("P", "C"): 1.84,
    # Silicon
    ("Si", "O"): 1.64,
    ("Si", "H"): 1.48,
    ("Si", "C"): 1.87,
    ("Si", "Si"): 2.34,
    # Boron
    ("B", "O"): 1.36,
    ("B", "N"): 1.40,
    ("B", "H"): 1.19,
    ("B", "C"): 1.56,
    # Nitrogen / Oxygen / Sulfur
    ("N", "H"): 1.01,
    ("O", "H"): 0.96,
    ("S", "H"): 1.34,
    ("N", "N"): 1.45,
    ("N", "O"): 1.40,
    ("O", "O"): 1.48,
    ("S", "S"): 2.05,
    ("S", "O"): 1.49,
}


def get_optimal_bond_length(sym1: str, sym2: str, default: Optional[float] = None) -> float:
    """Returns optimal standard single bond length for two elements in Angstroms."""
    pair = (sym1, sym2)
    inv_pair = (sym2, sym1)
    if pair in STANDARD_BOND_LENGTHS:
        return STANDARD_BOND_LENGTHS[pair]
    if inv_pair in STANDARD_BOND_LENGTHS:
        return STANDARD_BOND_LENGTHS[inv_pair]
    if default is not None:
        return default
    
    # Fallback to covalent radii sum
    try:
        from ase.data import atomic_numbers
        z1 = atomic_numbers.get(sym1, 6)
        z2 = atomic_numbers.get(sym2, 6)
        return covalent_radii[z1] + covalent_radii[z2]
    except Exception:
        return 1.50


def check_clash(mol: Atoms, frag: Atoms, threshold: float = 0.8) -> bool:
    """
    Fast clash detection using vectorized distance calculation.
    Returns True if NO clash exists, False if any clash is found.
    """
    pos_mol = mol.positions
    pos_frag = frag.positions
    
    radii_mol = np.array([covalent_radii[z] for z in mol.numbers])
    radii_frag = np.array([covalent_radii[z] for z in frag.numbers])

    dists = cdist(pos_frag, pos_mol)
    thresh_matrix = (radii_frag[:, None] + radii_mol[None, :]) * threshold

    return not np.any(dists < thresh_matrix)


def calculate_steric_score(mol: Atoms, frag: Atoms, threshold: float = 0.8) -> Tuple[int, float, float]:
    """
    Calculates detailed steric metrics between fragment and base molecule:
    - n_clashes: number of atom pairs closer than threshold * (r_i + r_j)
    - repulsion_energy: continuous soft repulsive potential penalizing close contacts
    - min_distance: absolute minimum interatomic distance (higher is better)
    """
    pos_mol = mol.positions
    pos_frag = frag.positions
    
    radii_mol = np.array([covalent_radii[z] for z in mol.numbers])
    radii_frag = np.array([covalent_radii[z] for z in frag.numbers])

    dists = cdist(pos_frag, pos_mol)
    thresh_matrix = (radii_frag[:, None] + radii_mol[None, :]) * threshold
    
    n_clashes = int(np.sum(dists < thresh_matrix))
    
    # Continuous soft-core repulsion potential:
    # vdw_sum / dist; values above 0.8 generate steep positive penalty
    vdw_sum = radii_frag[:, None] + radii_mol[None, :]
    overlap = np.maximum(0.0, (vdw_sum / np.maximum(dists, 0.2)) - 0.75)
    repulsion_energy = float(np.sum(overlap ** 4))
    
    min_distance = float(np.min(dists)) if dists.size > 0 else 999.0
    
    return n_clashes, repulsion_energy, min_distance


# Per-atom index arrays carried through attach_fragment; -1 marks atoms not in the base molecule
TRACKING_ARRAYS = ('orig_idx', 'original_id')


def track_atoms(mol: Atoms, indices) -> Atoms:
    """Store an 'original_id' array: the original index for `indices`, -1 for the other atoms."""
    original_id = np.full(len(mol), -1, dtype=int)
    idx = np.asarray(list(indices), dtype=int)
    original_id[idx] = idx
    mol.set_array('original_id', original_id)
    return mol


def attach_fragment(
    mol: Atoms, 
    frag_template: Atoms, 
    mol_pair: Tuple[int, int], 
    frag_pair: Tuple[int, int], 
    n_rotations: int = 36, 
    clash_thresh: float = 0.8,
    bond_length: Optional[float] = None,
    clash_handling: str = "best_effort"
) -> Tuple[Atoms, List[int], Tuple[int, int]]:
    """
    Attaches a single fragment to the molecule with intelligent dihedral search.
    Guarantees the core molecule atoms are completely untouched, and orients
    the fragment to minimize steric strain and maximize clearance.

    Args:
        mol (Atoms): Base molecule.
        frag_template (Atoms): Fragment to attach.
        mol_pair (Tuple[int, int]): (i_replace, i_anchor) on base molecule.
        frag_pair (Tuple[int, int]): (j_delete, j_attach) on fragment.
        n_rotations (int): Number of dihedral angles sampled around attachment axis (default 36, 10° steps).
        clash_thresh (float): Fraction of covalent radii sum considered a hard clash (default 0.8).
        bond_length (float, optional): Custom bond length in Angstroms. If None, uses optimal standard value.
        clash_handling (str): 'best_effort' (picks lowest clash angle and leaves rest to optimizer) or 'raise'.

    Returns:
        Tuple: (New Molecule, Indices of added atoms, (anchor_idx, attach_idx))
    """
    mol = mol.copy()
    frag = frag_template.copy()
    
    i_replace, i_anchor = mol_pair
    j_delete, j_attach = frag_pair

    # 1. Center and Orient Fragment
    frag.translate(-frag[j_attach].position)
    
    v_frag = frag[j_delete].position 
    v_target = mol[i_replace].position - mol[i_anchor].position
    norm_target = np.linalg.norm(v_target)
    if norm_target < 1e-6:
        raise ValueError(f"Target vector between replace ({i_replace}) and anchor ({i_anchor}) is zero length.")
    
    axis = v_target / norm_target
    
    # The fragment dummy (j_delete) stands in for the base anchor atom, so the
    # attach->dummy vector must point back to the anchor, i.e. anti-parallel to
    # anchor->replace (the fragment body then points away from the core).
    norm_v_frag = np.linalg.norm(v_frag)
    if norm_v_frag > 1e-6:
        frag.rotate(v_frag, -v_target, center=(0, 0, 0))

    # 2. Remove dummy atoms
    del frag[j_delete]
    del mol[i_replace]
    
    if i_replace < i_anchor: 
        i_anchor -= 1
    
    # 3. Position at optimal bond distance
    real_j_attach = j_attach if j_delete > j_attach else j_attach - 1
    
    if bond_length is None:
        anchor_sym = mol[i_anchor].symbol
        attach_sym = frag[real_j_attach].symbol
        bond_dist = get_optimal_bond_length(anchor_sym, attach_sym)
    else:
        bond_dist = float(bond_length)
    
    frag.translate(mol[i_anchor].position + (bond_dist * axis))

    # 4. Anti-clash & Optimal Dihedral Sampling
    best_frag = None
    best_clashes = float('inf')
    best_repulsion = float('inf')
    best_min_dist = -1.0

    # Sample full 360-degree rotation around attachment bond
    angles = np.linspace(0.0, 360.0, n_rotations, endpoint=False)
    
    for angle in angles:
        test_frag = frag.copy()
        test_frag.rotate(angle, axis, center=mol[i_anchor].position)
        
        n_clashes, repulsion, min_dist = calculate_steric_score(mol, test_frag, threshold=clash_thresh)
        
        # Priority:
        # 1. Zero hard clashes (clashes == 0)
        # 2. Lowest number of clashes
        # 3. Lowest continuous repulsion energy
        # 4. Highest minimum distance to core
        is_better = False
        if n_clashes < best_clashes:
            is_better = True
        elif n_clashes == best_clashes:
            if repulsion < best_repulsion:
                is_better = True
            elif np.isclose(repulsion, best_repulsion, atol=1e-3) and min_dist > best_min_dist:
                is_better = True
        
        if is_better:
            best_clashes = n_clashes
            best_repulsion = repulsion
            best_min_dist = min_dist
            best_frag = test_frag
            
            # If 0 clashes and repulsion is essentially 0, it's an ideal conformation
            if best_clashes == 0 and best_repulsion < 1e-4:
                break
            
    if best_frag is None or (best_clashes > 0 and clash_handling == "raise"):
        raise ValueError(f"Unresolvable clash for site {mol_pair} (minimum clashes found: {best_clashes})")

    # 5. Merge
    len_mol = len(mol)
    mol += best_frag
    added_indices = list(range(len_mol, len(mol)))
    # ASE pads arrays missing from the fragment with 0, which would alias original atom 0
    for key in TRACKING_ARRAYS:
        if key in mol.arrays and key not in best_frag.arrays:
            mol.arrays[key][len_mol:] = -1
    
    new_anchor_idx = i_anchor
    new_attach_idx = len_mol + real_j_attach
    
    return mol, added_indices, (new_anchor_idx, new_attach_idx)