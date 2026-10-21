"""Graph theory utilities for molecular connectivity analysis.

This module provides functions to build connectivity graphs from atomic positions
and identify connected fragments (molecules) within a system.
"""

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict
from typing import List, Dict, Any

from frogchem.chemistry import get_atomic_radius


def create_graph_matrix(
    symbols: List[str],
    positions: np.ndarray,
    radii: Dict[str, float],
    buffer: float = 0.3
) -> np.ndarray:
    """Generates the connectivity graph matrix of the system.

    Two atoms are considered connected if their distance is less than the sum
    of their radii plus the buffer.

    Args:
        symbols (List[str]): List of atomic symbols.
        positions (np.ndarray): Array of shape (N, 3) with atomic coordinates.
        radii (Dict[str, float]): Dictionary mapping symbols to atomic radii.
        buffer (float, optional): Buffer distance in Angstrom. Defaults to 0.3.

    Returns:
        np.ndarray: Adjacency matrix (N, N) where 1 implies connection.
    """
    n_atoms = len(symbols)
    # ponytail: cdist replaces N*(N-1)/2 individual norm calls
    dist_matrix = cdist(positions, positions)

    r = np.array([radii.get(s, 1.5) for s in symbols])
    threshold = r[:, None] + r[None, :] + buffer

    graph = (dist_matrix < threshold).astype(int)
    np.fill_diagonal(graph, 0)
    return graph


def dfs(atom: int, graph: np.ndarray, visited: List[bool]) -> List[int]:
    """Performs Depth First Search to find connected component.

    Args:
        atom (int): Index of the starting atom.
        graph (np.ndarray): Connectivity graph.
        visited (List[bool]): List tracking visited atoms.

    Returns:
        List[int]: List of atom indexes in the connected component.
    """
    stack = [atom]
    fragment = []

    while stack:
        current_atom = stack.pop()
        if not visited[current_atom]:
            visited[current_atom] = True
            fragment.append(current_atom)
            # Find neighbors
            neighbors = np.where(graph[current_atom] == 1)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    stack.append(neighbor)

    return fragment


def find_fragments(graph: np.ndarray) -> List[List[int]]:
    """Identifies connected fragments in the graph.

    Args:
        graph (np.ndarray): Adjacency matrix.

    Returns:
        List[List[int]]: List of fragments, where each fragment is a sorted list of atom indexes.
    """
    n_atoms = graph.shape[0]
    visited = [False] * n_atoms
    fragments = []

    for atom in range(n_atoms):
        if not visited[atom]:
            fragment = dfs(atom, graph, visited)
            fragments.append(sorted(fragment))

    return fragments


def detect_fragments_from_geom(
    symbols: List[str], 
    positions: np.ndarray, 
    buffer: float = 0.3
) -> List[List[int]]:
    """High-level function to detect fragments from geometry.

    Args:
        symbols (List[str]): Atomic symbols.
        positions (np.ndarray): Atomic coordinates.
        buffer (float, optional): Bonding buffer. Defaults to 0.3.

    Returns:
        List[List[int]]: List of fragments (lists of atom indexes).
    """
    # Create radii dictionary on the fly for current atoms
    unique_syms = set(symbols)
    radii = {k: get_atomic_radius(k) for k in unique_syms}
    
    graph = create_graph_matrix(symbols, positions, radii, buffer)
    return find_fragments(graph)


class DisjointSet:
    """Disjoint-set (Union-Find) data structure with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        """Find the representative element of the set containing i with path compression."""
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """Merge the sets containing i and j. Returns True if merged, False if already in the same set."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            if self.rank[root_i] == self.rank[root_j]:
                self.rank[root_i] += 1
            return True
        return False

    def get_groups(self) -> Dict[int, List[int]]:
        """Return a dictionary mapping root representatives to member indices."""
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return dict(groups)