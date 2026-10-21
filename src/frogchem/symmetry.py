"""Symmetry utilities using only NumPy.

Robust detection of Point Group and Equivalent Atoms.
Handles symmetric tops (degenerate inertia) by checking candidate axes derived from atoms.
Hydrogens are included by default (true molecular point group); excluding them
(opt-in) gives the symmetry of the heavy-atom framework, to tolerate geometric noise.
Proper (C_n) and improper (S_n, n = 4, 6, 8) axes, mirror planes and inversion are tested.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from scipy.spatial.distance import cdist
from frogchem.chemistry import get_atomic_weights
from frogchem.math_geometry import (
    normalize,
    get_rotation_matrix,
    get_reflection_matrix,
    calculate_center_of_mass,
    _inertia_tensor_from_masses,
    get_inertia_tensor,
    align_molecule,
)
from frogchem.graph_theory import DisjointSet

__all__ = [
    'PointGroupDetector',
    'detect_point_group',
    'get_absolute_configuration',
    'normalize',
    'get_rotation_matrix',
    'get_reflection_matrix',
    'calculate_center_of_mass',
    'get_inertia_tensor',
    'align_molecule',
    'DisjointSet',
]

class PointGroupDetector:
    """Point-group detector.

    exclude_h=False (default): the true molecular point group (all atoms).
    exclude_h=True: the point group of the heavy-atom framework, centred on the
    framework's own centre of mass (useful to tolerate noisy/rotameric H). If the
    framework has fewer than two atoms, all atoms are used instead.
    """
    def __init__(self, symbols: List[str], positions: np.ndarray, tolerance: float = 0.1, exclude_h: bool = False):
        self.symbols = np.array(symbols)
        positions = np.asarray(positions, dtype=float)
        self.tol = tolerance
        self.n_atoms = len(symbols)
        self.exclude_h = exclude_h

        if self.exclude_h:
            self.mask = np.array([s.upper() != 'H' for s in self.symbols])
            if np.sum(self.mask) < 2:
                self.mask = np.ones(self.n_atoms, dtype=bool)
        else:
            self.mask = np.ones(self.n_atoms, dtype=bool)

        # Symmetry elements pass through the centre of mass of the analysed atoms
        com, _ = calculate_center_of_mass(self.symbols[self.mask], positions[self.mask])
        self.coords = positions - com

        self.valid_ops = []
        self.found_axes = []
        self.found_planes = []
        self.found_improper = []
        self.has_inversion = False
        self.max_op_rmsd = 0.0

        self._analyze_inertia()
        self.candidate_axes = self._generate_candidate_axes()

    def _analyze_inertia(self):
        syms_active = self.symbols[self.mask]
        coords_active = self.coords[self.mask]
        masses = np.array([get_atomic_weights(s) for s in syms_active])
        tensor = _inertia_tensor_from_masses(masses, coords_active)
        evals, evecs = np.linalg.eigh(tensor)
        idx = np.argsort(evals)
        self.inertia_evals = evals[idx]
        self.inertia_evecs = evecs[:, idx]
        if np.linalg.det(self.inertia_evecs) < 0:
            self.inertia_evecs[:, 0] *= -1

    def _generate_candidate_axes(self) -> List[np.ndarray]:
        """Candidate rotation axes / plane normals.

        Atom-derived candidates come first (exact for symmetric geometries): atom
        directions, pair midpoints (C2 axes) and, for same-element pairs, the
        difference vectors r_i - r_j (mirror normals, perpendicular C2 axes) and
        the normal of the plane spanned by the pair and the origin.
        Principal inertia axes are appended last. Near-duplicates are removed with a
        tight threshold (|cos| > 0.9999, ~0.8 deg) so that an approximate axis can
        never shadow an exact one.
        """
        candidates = []
        active_indices = np.where(self.mask)[0]
        coords = self.coords
        syms = self.symbols

        for i in active_indices:
            if np.linalg.norm(coords[i]) > 0.1:
                candidates.append(normalize(coords[i]))

        if len(active_indices) < 50:
            for idx_i in range(len(active_indices)):
                i = active_indices[idx_i]
                for idx_j in range(idx_i + 1, len(active_indices)):
                    j = active_indices[idx_j]
                    mid = coords[i] + coords[j]
                    if np.linalg.norm(mid) > 0.1:
                        candidates.append(normalize(mid))
                    if syms[i] == syms[j]:
                        diff = coords[i] - coords[j]
                        if np.linalg.norm(diff) > 0.1:
                            candidates.append(normalize(diff))
                        # normal of the plane through the origin and the pair (C3/C4 axes of cubic groups)
                        cr = np.cross(coords[i], coords[j])
                        if np.linalg.norm(cr) > 0.1:
                            candidates.append(normalize(cr))

        for i in range(3):
            candidates.append(self.inertia_evecs[:, i])

        unique = []
        for c in candidates:
            if not any(abs(np.dot(c, u)) > 0.9999 for u in unique):
                unique.append(c)
        return unique

    def _check_op_mat(self, op_mat: np.ndarray) -> Tuple[bool, float]:
        coords_sub = self.coords[self.mask]
        syms_sub = self.symbols[self.mask]
        
        transformed = np.dot(coords_sub, op_mat.T)
        
        total_max_dist = 0.0
        
        for el in np.unique(syms_sub):
            mask_el = syms_sub == el
            orig_el = coords_sub[mask_el]
            trans_el = transformed[mask_el]
            
            dists = cdist(trans_el, orig_el)
            min_dists = np.min(dists, axis=1)
            current_max = np.max(min_dists)
            
            if current_max > total_max_dist:
                total_max_dist = current_max
                
            if current_max > self.tol:
                return False, current_max
                
        return True, total_max_dist

    def analyze(self):
        self.valid_ops = [np.eye(3)]
        self.max_op_rmsd = 0.0 
        
        valid, rmsd = self._check_op_mat(-np.eye(3))
        if valid:
            self.has_inversion = True
            self.valid_ops.append(-np.eye(3))
            self.max_op_rmsd = max(self.max_op_rmsd, rmsd)

        # Linear framework: every C_n about the molecular axis is valid; keep all orders
        # so that atoms mapped by C3/C5... (e.g. methyl H) are found equivalent.
        linear = self._is_linear()
        for axis in self.candidate_axes:
            for n in [8, 6, 5, 4, 3, 2]:
                mat = get_rotation_matrix(axis, 2*np.pi/n)
                valid, rmsd = self._check_op_mat(mat)
                if valid:
                    self.found_axes.append((axis, n))
                    self.max_op_rmsd = max(self.max_op_rmsd, rmsd)
                    for k in range(1, n):
                        mat_k = get_rotation_matrix(axis, k*2*np.pi/n)
                        self.valid_ops.append(mat_k)
                    if not linear:
                        break

        # Improper rotations S_n = sigma_h * C_n (S1 = sigma, S2 = i are handled separately;
        # S3/S5 imply C3h/C5h and are covered by C_n + sigma_h)
        for axis in self.candidate_axes:
            refl = get_reflection_matrix(axis)
            for n in [8, 6, 4]:
                mat = refl @ get_rotation_matrix(axis, 2*np.pi/n)
                valid, rmsd = self._check_op_mat(mat)
                if valid:
                    self.found_improper.append((axis, n))
                    self.max_op_rmsd = max(self.max_op_rmsd, rmsd)
                    op = np.eye(3)
                    for _ in range(1, n):
                        op = mat @ op
                        self.valid_ops.append(op.copy())
                    break

        for axis in self.candidate_axes:
            mat = get_reflection_matrix(axis)
            valid, rmsd = self._check_op_mat(mat)
            if valid:
                self.found_planes.append(axis)
                self.valid_ops.append(mat)
                self.max_op_rmsd = max(self.max_op_rmsd, rmsd)

    def _is_linear(self) -> bool:
        return self.inertia_evals[0] < 1e-4 and self.inertia_evals[1] >= 1e-4

    def _identify_group(self) -> str:
        if self.inertia_evals[1] < 1e-4:
            return "Kh"  # single atom
        if self._is_linear():
            return "D_inf_h" if self.has_inversion else "C_inf_v"

        if not self.found_axes:
            if self.found_planes: return "Cs"
            if self.has_inversion: return "Ci"
            return "C1"

        self.found_axes.sort(key=lambda x: x[1], reverse=True)
        n = self.found_axes[0][1]

        high_order = [x for x in self.found_axes if x[1] >= 3]
        unique_high = []
        for ax, order in high_order:
            if not any(abs(np.dot(ax, u[0])) > 0.9 for u in unique_high):
                unique_high.append((ax, order))

        if len(unique_high) > 1:
            orders = {order for _, order in unique_high}
            if 5 in orders: return "Ih" if self.has_inversion else "I"
            if 4 in orders: return "Oh" if self.has_inversion else "O"
            if self.has_inversion: return "Th"
            return "Td" if self.found_planes else "T"

        def coaxial_improper(ax, order):
            return any(m == order and abs(np.dot(ax, a)) > 0.9 for a, m in self.found_improper)

        # Principal axis: highest order; among equal orders (e.g. three C2 in D2d)
        # prefer the one carrying a coaxial S_2n.
        top = [ax for ax, order in self.found_axes if order == n]
        principal_axis = next((ax for ax in top if coaxial_improper(ax, 2 * n)), top[0])

        perp_c2s = 0
        for ax, order in self.found_axes:
            if order == 2 and abs(np.dot(ax, principal_axis)) < 0.1:
                perp_c2s += 1

        has_sigma_h = False
        has_sigma_v = False
        for normal in self.found_planes:
            dot = abs(np.dot(normal, principal_axis))
            if dot > 0.9: has_sigma_h = True
            if dot < 0.1: has_sigma_v = True

        if perp_c2s >= 1:
            if has_sigma_h: return f"D{n}h"
            if has_sigma_v: return f"D{n}d"
            return f"D{n}"
        else:
            if has_sigma_h: return f"C{n}h"
            if has_sigma_v: return f"C{n}v"
            if coaxial_improper(principal_axis, 2 * n): return f"S{2*n}"
            if self.has_inversion and n%2!=0: return f"S{2*n}"
            return f"C{n}"

    def get_equivalent_atoms(self, map_tolerance: float = 0.3) -> Dict[str, List[List[int]]]:
        if not self.valid_ops:
            self.analyze()

        dset = DisjointSet(self.n_atoms)

        for mat in self.valid_ops:
            if isinstance(mat, tuple):
                mat = mat[0]
            transformed = np.dot(self.coords, mat.T)

            for el in np.unique(self.symbols):
                indices = np.where(self.symbols == el)[0]
                if len(indices) == 0:
                    continue

                sub_orig = self.coords[indices]
                sub_trans = transformed[indices]

                dists = cdist(sub_orig, sub_trans)
                rows, cols = np.where(dists < map_tolerance)

                for r, c in zip(rows, cols):
                    dset.union(indices[r], indices[c])

        groups = dset.get_groups()
        output = defaultdict(list)
        for g in groups.values():
            el = self.symbols[g[0]]
            output[el].append(sorted(g))

        for k in output:
            output[k].sort(key=lambda x: x[0])

        return dict(output)

def detect_point_group(symbols: List[str], positions: np.ndarray, tolerance: float = 0.1, exclude_h: bool = False) -> Dict[str, Any]:
    """Schoenflies symbol and symmetry-equivalent atoms.

    exclude_h=False (default) gives the true molecular point group. exclude_h=True
    gives the group of the heavy-atom framework (centred on its own COM); hydrogens
    are then mapped onto equivalence classes with a 3x looser tolerance.
    """
    detector = PointGroupDetector(symbols, positions, tolerance, exclude_h=exclude_h)
    detector.analyze()
    label = detector._identify_group()
    map_tol = tolerance * 3 if exclude_h else tolerance
    equiv = detector.get_equivalent_atoms(map_tolerance=map_tol)
    
    return {
        'is_valid': True,
        'sch_symbol': label,
        'equivalencies': equiv,
        'error_rmsd': detector.max_op_rmsd
    }



def get_absolute_configuration(rdkit_mol) -> list[tuple[int, str]]:
    from rdkit import Chem
    # Assign chiral tags and find absolute configuration
    Chem.AssignAtomChiralTagsFromStructure(rdkit_mol)
    absolute_conf = Chem.FindMolChiralCenters(rdkit_mol, includeUnassigned=True)
    return absolute_conf