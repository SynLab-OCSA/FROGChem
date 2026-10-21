"""Unit tests for high-speed steric volume calculation."""

import numpy as np
import pytest
from frogchem.steric_maps import (
    compute_buried_volume, 
    orient_positions, 
    get_sambvca_radii
)

def test_fast_buried_volume_synthetic():
    # Center at origin (excluded), 1 atom at (1, 0, 0) with radius 1.5
    positions = np.array([[1.0, 0.0, 0.0]])
    radii = np.array([1.5])
    
    res = compute_buried_volume(positions, radii, radius=3.0, spacing=0.1)
    
    assert res["total"] > 0.0
    assert "quadrants" in res
    assert "octants" in res
    assert set(res["quadrants"].keys()) == {"NE", "NW", "SW", "SE"}
    assert len(res["octants"]) == 8

def test_quadrants_symmetry():
    # Symmetric 4 atoms placed on NE, NW, SW, SE
    d = 1.2
    positions = np.array([
        [d, d, 0.0],
        [-d, d, 0.0],
        [-d, -d, 0.0],
        [d, -d, 0.0]
    ])
    radii = np.array([1.0, 1.0, 1.0, 1.0])
    
    res = compute_buried_volume(positions, radii, radius=3.0, spacing=0.1)
    q = res["quadrants"]
    
    # Due to symmetry, all 4 quadrants should have nearly identical %V_bur (within grid discretization)
    assert abs(q["NE"] - q["NW"]) < 2.0
    assert abs(q["NE"] - q["SW"]) < 2.0
    assert abs(q["NE"] - q["SE"]) < 2.0

def test_octants_z_symmetry():
    # 2 atoms placed symmetrically at +Z and -Z
    positions = np.array([
        [0.0, 0.0, 1.5],
        [0.0, 0.0, -1.5]
    ])
    radii = np.array([1.0, 1.0])
    res = compute_buried_volume(positions, radii, radius=3.0, spacing=0.1)
    o = res["octants"]
    
    # Front NE should match Back NE (within grid discretization)
    assert abs(o["Front NE"] - o["Back NE"]) < 1.0
    assert abs(o["Front NW"] - o["Back NW"]) < 1.0
