import numpy as np
import pytest
from frogchem.graph_theory import (
    create_graph_matrix,
    dfs,
    find_fragments,
    detect_fragments_from_geom,
    DisjointSet,
)


class TestCreateGraphMatrix:
    def test_two_connected_atoms(self):
        symbols = ["H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        radii = {"H": 0.25}
        graph = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        assert graph[0, 1] == 1
        assert graph[1, 0] == 1

    def test_two_distant_atoms(self):
        symbols = ["H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        radii = {"H": 0.25}
        graph = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        assert graph[0, 1] == 0

    def test_diagonal_is_zero(self):
        symbols = ["H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        radii = {"H": 0.25}
        graph = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        assert graph[0, 0] == 0
        assert graph[1, 1] == 0

    def test_three_atoms_chain(self):
        symbols = ["O", "H", "H"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.96, 0.0, 0.0],
        ])
        radii = {"O": 1.52, "H": 0.25}
        graph = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        assert graph[0, 1] == 1
        assert graph[0, 2] == 1
        assert graph[1, 2] == 0  # H-H not bonded

    def test_symmetry(self):
        symbols = ["H", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        radii = {"H": 0.25}
        graph = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        assert graph[0, 1] == graph[1, 0]


class TestDFS:
    def test_single_component(self):
        graph = np.array([[0, 1], [1, 0]])
        visited = [False, False]
        fragment = dfs(0, graph, visited)
        assert sorted(fragment) == [0, 1]

    def test_unvisited_preserved(self):
        graph = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        visited = [False, False, False]
        fragment = dfs(0, graph, visited)
        assert sorted(fragment) == [0, 1]
        assert visited[0] and visited[1]
        assert not visited[2]

    def test_isolated_atom(self):
        graph = np.array([[0]])
        visited = [False]
        fragment = dfs(0, graph, visited)
        assert fragment == [0]


class TestFindFragments:
    def test_two_isolated_pairs(self):
        graph = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        fragments = find_fragments(graph)
        assert len(fragments) == 2
        assert sorted(fragments[0]) == [0, 1]
        assert sorted(fragments[1]) == [2, 3]

    def test_single_fragment(self):
        graph = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        fragments = find_fragments(graph)
        assert len(fragments) == 1
        assert sorted(fragments[0]) == [0, 1, 2]

    def test_all_isolated(self):
        graph = np.zeros((4, 4), dtype=int)
        fragments = find_fragments(graph)
        assert len(fragments) == 4

    def test_empty_graph(self):
        graph = np.zeros((0, 0), dtype=int)
        fragments = find_fragments(graph)
        assert fragments == []


class TestDetectFragmentsFromGeom:
    def test_two_h2_molecules(self):
        symbols = ["H", "H", "H", "H"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.74],
            [10.0, 0.0, 0.0],
            [10.0, 0.0, 0.74],
        ])
        fragments = detect_fragments_from_geom(symbols, positions, buffer=0.3)
        assert len(fragments) == 2
        assert sorted(fragments[0]) == [0, 1]
        assert sorted(fragments[1]) == [2, 3]

    def test_single_h2(self):
        symbols = ["H", "H"]
        positions = np.array([[0, 0, 0], [0, 0, 0.74]])
        fragments = detect_fragments_from_geom(symbols, positions)
        assert len(fragments) == 1
        assert sorted(fragments[0]) == [0, 1]

    def test_water_as_one_fragment(self):
        symbols = ["O", "H", "H"]
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.96],
            [0.96, 0.0, 0.0],
        ])
        fragments = detect_fragments_from_geom(symbols, positions, buffer=0.3)
        assert len(fragments) == 1

    def test_known_element_noble_gas(self):
        symbols = ["He", "He"]
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        fragments = detect_fragments_from_geom(symbols, positions, buffer=0.3)
        assert len(fragments) == 2


class TestDisjointSet:
    def test_init_and_find(self):
        ds = DisjointSet(5)
        for i in range(5):
            assert ds.find(i) == i

    def test_union_and_connectivity(self):
        ds = DisjointSet(6)
        assert ds.union(0, 1) is True
        assert ds.union(1, 2) is True
        assert ds.find(0) == ds.find(2)
        assert ds.union(0, 2) is False  # already same set

        assert ds.union(3, 4) is True
        assert ds.find(0) != ds.find(3)

    def test_get_groups(self):
        ds = DisjointSet(4)
        ds.union(0, 2)
        ds.union(1, 3)
        groups = ds.get_groups()
        assert len(groups) == 2
        group_members = [sorted(m) for m in groups.values()]
        assert [0, 2] in group_members
        assert [1, 3] in group_members
