"""Topographic Steric Map and Buried Volume Calculations.

Native implementation of the SambVca mathematical model to quantify 
steric bulk, heavily inspired by the architecture of StericRender.
Calculates volumetric quadrants, octants, and exact steric maps.
"""

import os
from pathlib import Path
import numpy as np
from frogchem.chemistry import get_atomic_radius

BONDI_RADII = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 
    'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 
    'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 
    'K': 2.75, 'Ca': 2.31, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39, 'Ga': 1.87, 
    'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 
    'Sr': 2.49, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17, 
    'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 
    'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07,
    'Po': 1.97, 'At': 2.02, 'Rn': 2.20, 'Fr': 3.48, 'Ra': 2.83, 'U': 1.86
}

def get_sambvca_radii(symbols):
    """Bondi radii scaled by 1.17 (SambVca). Elements missing from BONDI_RADII fall
    back to 1.5 x covalent radius (approximation) before scaling."""
    radii = []
    for s in symbols:
        s = str(s).strip().capitalize()
        r = BONDI_RADII.get(s)
        if r is None:
            r = get_atomic_radius(s) * 1.5
        radii.append(r * 1.17)
    return np.array(radii)

def orient_positions(positions, center_idx, z_axis_idx=None, x_axis_idx=None, flip_x=False, flip_z=False):
    """
    Translate and rotate molecular coordinates.
    
    Parameters:
    -----------
    positions : np.ndarray
        Nx3 array of coordinates
    center_idx : int
        Index of the origin atom (becomes 0,0,0)
    z_axis_idx : list of int, optional
        Atom indices defining the +Z axis.
    x_axis_idx : list of int, optional
        Atom indices defining the X axis (fixes roll).
    flip_x : bool, optional
        If True, align x_axis_idx to the -X axis instead of +X.
    flip_z : bool, optional
        If True, align the Z axis in the opposite direction (-Z).
    """
    coords = np.array(positions)
    center = coords[center_idx].copy()
    coords = coords - center
    
    if z_axis_idx is None:
        return coords
        
    # Align Z axis
    z_point = coords[z_axis_idx].mean(axis=0)
    z_norm = np.linalg.norm(z_point)
    if z_norm < 1e-8:
        raise ValueError("The centroid of the Z-axis atoms coincides with the center atom: Z axis undefined")
    z_vec = z_point / z_norm
    if flip_z:
        z_vec = -z_vec
        
    # Rotation matrix to align z_vec with [0,0,1]
    v = np.cross(z_vec, [0, 0, 1])
    s = np.linalg.norm(v)
    c = np.dot(z_vec, [0, 0, 1])
    
    R_z = np.eye(3)
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_z = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    elif c < -0.999999:
        R_z = -np.eye(3)
        R_z[0,0] = 1 
        
    coords = np.dot(coords, R_z.T)
    
    # Fix the roll around Z using the X-axis reference
    if x_axis_idx is not None:
        x_point = coords[x_axis_idx].mean(axis=0)
        x_proj = np.array([x_point[0], x_point[1], 0])
        n_proj = np.linalg.norm(x_proj)
        if n_proj > 1e-6:
            x_proj = x_proj / n_proj
            
            # If flip_x is True, we want to align x_proj to [-1, 0, 0], 
            # which is equivalent to aligning -x_proj to [1, 0, 0]
            if flip_x:
                x_proj = -x_proj
                
            cos_theta = x_proj[0]
            sin_theta = x_proj[1]
            R_roll = np.array([
                [cos_theta, sin_theta, 0],
                [-sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            coords = np.dot(coords, R_roll.T)
            
    return coords

def compute_buried_volume(positions, radii, radius=3.5, spacing=0.1):
    """
    Compute %V_bur analytically using an ultra-fast 3D bounding-box rasterizer,
    partitioned into quadrants and octants.

    The grid is symmetric about the origin (points k*spacing, k = -m..m) so that
    the x=0, y=0, z=0 planes are sampled exactly; points lying on a dividing plane
    count 1/2 in each adjacent half, so quadrants/octants carry no orientation bias.
    """
    m = int(np.floor(radius / spacing + 1e-9))
    coords_1d = spacing * np.arange(-m, m + 1)
    n_pts = len(coords_1d)
    coords_min = coords_1d[0]

    # Sphere mask
    cx2 = coords_1d ** 2
    r_sq = cx2[:, None, None] + cx2[None, :, None] + cx2[None, None, :]
    sphere_mask = r_sq <= radius ** 2
    total_voxels = int(sphere_mask.sum())

    # Fast 3D localized grid
    grid = np.zeros((n_pts, n_pts, n_pts), dtype=bool)
    for pos, rad in zip(positions, radii):
        # Skip atoms completely outside sphere influence
        if np.linalg.norm(pos) > radius + rad:
            continue

        i0 = max(0, int(np.floor((pos[0] - rad - coords_min) / spacing)))
        i1 = min(n_pts, int(np.ceil((pos[0] + rad - coords_min) / spacing)) + 1)
        j0 = max(0, int(np.floor((pos[1] - rad - coords_min) / spacing)))
        j1 = min(n_pts, int(np.ceil((pos[1] + rad - coords_min) / spacing)) + 1)
        k0 = max(0, int(np.floor((pos[2] - rad - coords_min) / spacing)))
        k1 = min(n_pts, int(np.ceil((pos[2] + rad - coords_min) / spacing)) + 1)

        sub_x = coords_1d[i0:i1, None, None]
        sub_y = coords_1d[None, j0:j1, None]
        sub_z = coords_1d[None, None, k0:k1]
        sub_dist2 = (sub_x - pos[0]) ** 2 + (sub_y - pos[1]) ** 2 + (sub_z - pos[2]) ** 2
        grid[i0:i1, j0:j1, k0:k1] |= (sub_dist2 <= rad ** 2)

    buried_in_sphere = grid & sphere_mask
    buried_voxels = int(buried_in_sphere.sum())
    voxel_vol = spacing ** 3
    buried_vol = float(buried_voxels * voxel_vol)
    total_vol = float(total_voxels * voxel_vol)
    free_vol = total_vol - buried_vol

    v_bur_total = 100.0 * buried_vol / total_vol if total_vol > 0 else 0.0

    # Half-space weights along one axis: 1 on the side, 1/2 on the dividing plane
    on_plane = np.abs(coords_1d) < 1e-9 * max(spacing, 1.0)
    w_pos = np.where(coords_1d > 0, 1.0, 0.0) + 0.5 * on_plane
    w_neg = np.where(coords_1d < 0, 1.0, 0.0) + 0.5 * on_plane
    w_all = np.ones(n_pts)

    slices = {
        "NE": (w_pos, w_pos, w_all),
        "NW": (w_neg, w_pos, w_all),
        "SW": (w_neg, w_neg, w_all),
        "SE": (w_pos, w_neg, w_all),
        "Front NE": (w_pos, w_pos, w_pos),
        "Front NW": (w_neg, w_pos, w_pos),
        "Front SW": (w_neg, w_neg, w_pos),
        "Front SE": (w_pos, w_neg, w_pos),
        "Back NE": (w_pos, w_pos, w_neg),
        "Back NW": (w_neg, w_pos, w_neg),
        "Back SW": (w_neg, w_neg, w_neg),
        "Back SE": (w_pos, w_neg, w_neg),
    }
    sphere_f = sphere_mask.astype(float)
    buried_f = buried_in_sphere.astype(float)

    results = {
        "total": v_bur_total,
        "quadrants": {},
        "octants": {},
        "raw": {
            "buried_volume": buried_vol,
            "sphere_volume": total_vol,
            "free_volume": free_vol,
            "voxel_volume": voxel_vol,
            "buried_voxels": buried_voxels,
            "total_voxels": total_voxels,
        },
    }

    for name, (wx, wy, wz) in slices.items():
        sub_tot = float(np.einsum('ijk,i,j,k->', sphere_f, wx, wy, wz))
        sub_bur = float(np.einsum('ijk,i,j,k->', buried_f, wx, wy, wz))
        val = 100.0 * sub_bur / sub_tot if sub_tot > 0 else 0.0
        if "Front" in name or "Back" in name:
            results["octants"][name] = val
        else:
            results["quadrants"][name] = val

    return results

def compute_steric_map(positions, radii, radius=3.5, spacing=0.1):
    """
    Compute the analytical 2.5D topographic steric map heights.
    """
    xy = np.arange(-radius, radius + spacing * 0.5, spacing)
    xx, yy = np.meshgrid(xy, xy)
    
    lateral2 = xx**2 + yy**2
    sphere_mask = lateral2 <= radius**2
    z_top = np.where(sphere_mask, np.sqrt(np.maximum(radius**2 - lateral2, 0.0)), np.nan)
    
    z_values = np.full_like(z_top, np.nan)
    
    for pos, rad in zip(positions, radii):
        dx = xx - pos[0]
        dy = yy - pos[1]
        dist2 = dx**2 + dy**2
        r2 = rad ** 2
        
        atom_mask = sphere_mask & (dist2 <= r2)
        if not atom_mask.any():
            continue
            
        dz = np.sqrt(np.maximum(r2 - dist2, 0.0))
        z_hit = np.minimum(pos[2] + dz, z_top)
        # the atom chord [z-dz, z+dz] must overlap the sphere column [-z_top, z_top]
        valid = atom_mask & (z_hit >= -z_top) & (pos[2] - dz <= z_top)
        z_values = np.fmax(z_values, np.where(valid, z_hit, np.nan))
        
    return xy, xy, z_values


def generate_svg_overlay(
    out_svg, 
    atoms, 
    oriented_coords, 
    radius, 
    vol_res, 
    map_x, 
    map_y, 
    map_z,
    excluded_indices=None,
    excluded_on_top=True
):
    """
    Generate the exact StericRender SVG overlay by interfacing with stericrender's API.
    Optionally renders excluded atoms on top of the steric map for high visual contrast.
    """
    try:
        from stericrender.overlay import write_xyzrender_overlay_svg, _insert_before_close
        from stericrender.maps import StericMapResult
        from stericrender.volume import BuriedVolumeResult
        import xyzrender
        from xyzrender import build_config, render, load
    except ImportError:
        raise ImportError("The 'stericrender' and 'xyzrender' packages are required for SVG overlays.")
        
    # Write a temporary XYZ file of the oriented molecule for xyzrender
    temp_xyz = str(Path(out_svg).with_suffix('.temp.xyz'))
    with open(temp_xyz, 'w') as f:
        f.write(f"{len(atoms)}\nGenerated by FROG\n")
        for sym, coord in zip(atoms, oriented_coords):
            f.write(f"{sym} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
            
    # Pack native FROG results into StericRender dataclasses
    sm_res = StericMapResult(x=map_x, y=map_y, z=map_z)
    
    bv_res = BuriedVolumeResult(
        percent_buried=vol_res["total"],
        buried_volume=vol_res["raw"]["buried_volume"],
        sphere_volume=vol_res["raw"]["sphere_volume"],
        free_volume=vol_res["raw"]["free_volume"],
        voxel_volume=vol_res["raw"]["voxel_volume"],
        buried_voxels=vol_res["raw"]["buried_voxels"],
        total_voxels=vol_res["raw"]["total_voxels"],
        quadrant_percent=vol_res["quadrants"],
        hemisphere_percent={"north": 0.0, "south": 0.0, "east": 0.0, "west": 0.0} # Unused visually but required by dataclass
    )
    
    # Generate the base SVG overlay
    write_xyzrender_overlay_svg(
        oriented_xyz=temp_xyz,
        output_svg=out_svg,
        steric_map=sm_res,
        volume=bv_res,
        sphere_radius=radius,
        canvas_size=800,
        show_contours=True,
        show_colorbar=True,
        show_vbur_label=True
    )
    
    # If requested, render excluded atoms ON TOP of the steric map
    if excluded_on_top and excluded_indices:
        import re
        cfg = build_config("default", canvas_size=800, orient=False, hy=False, bo=True)
        cfg.fixed_center = (0.0, 0.0)
        cfg.fixed_span = 2.0 * radius

        mol = load(temp_xyz, auto_align=False)
        spec = ",".join(str(idx + 1) for idx in excluded_indices)
        try:
            fg_svg = str(render(mol, config=cfg, orient=False, only=spec))

            defs_match = re.search(r"<defs>(.*?)</defs>", fg_svg, re.DOTALL)
            defs_content = defs_match.group(1) if defs_match else ""

            body_match = re.search(r"<svg\b[^>]*>(.*?)</svg>", fg_svg, re.DOTALL)
            if body_match:
                body = body_match.group(1)
                body = re.sub(r"<rect\b[^>]*/>", "", body)
                body = re.sub(r"<defs>.*?</defs>", "", body, flags=re.DOTALL)
                fg_layer = f'    <g id="stericrender-foreground-excluded">\n{defs_content}\n{body}\n    </g>\n'

                current_svg = Path(out_svg).read_text()
                updated_svg = _insert_before_close(current_svg, fg_layer)
                Path(out_svg).write_text(updated_svg)
        except Exception as e:
            # Fallback gracefully if foreground isolation encounters issues
            import logging
            logging.warning(f"Could not layer excluded atoms on top: {e}")

    # Cleanup temp file
    if os.path.exists(temp_xyz):
        os.remove(temp_xyz)
