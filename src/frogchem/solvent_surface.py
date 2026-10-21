"""Solvent Accessible Surface (SAS) and Solvent Excluded Surface (SES)
grids, output as Gaussian cube files.

VdW radii are Bondi/Mantina radii (frogchem.steric_maps.BONDI_RADII) + vdw_offset
(default 0); elements missing from the table fall back to 1.5 x covalent radius.
Grid spacings are always taken from the actual grid axes (they differ from
`spacing` when n_voxels is given).

SES computed via the EDTSurf algorithm (Xu & Zhang, PLoS ONE 2009):
  1. Build SAS solid (union of spheres with radius vdw + probe)
  2. EDT of SAS solid → distance from each interior point to SAS boundary
  3. SES solid = SAS points where distance > probe_radius
  4. SDF of SES solid
"""

import numpy as np
import periodictable as pt
from scipy.ndimage import distance_transform_edt

from frogchem.density_builder import generate_grid, _guess_atomic_number
from frogchem.io import write_gaussian_cube
from frogchem.chemistry import BOHR_TO_ANGSTROM
from frogchem.steric_maps import BONDI_RADII
import logging
from frogchem.printer import print_info

logger = logging.getLogger(__name__)

_AU = 1.0 / BOHR_TO_ANGSTROM  # Å → Bohr (CODATA 2018)

_PROBE_RADIUS = 1.4  # water probe radius in Å


def _get_vdw_radius(symbol, offset=0.0):
    """VdW radius (Å) = Bondi radius + offset (fallback: 1.5 x covalent radius)."""
    sym = str(symbol).strip().capitalize()
    r = BONDI_RADII.get(sym)
    if r is None:
        r = 1.5 * getattr(pt, sym).covalent_radius
    return r + offset


def _axes_spacing(axes):
    """Actual (hx, hy, hz) grid spacings in Å from the grid axes."""
    return tuple(float(a[1] - a[0]) if len(a) > 1 else 1.0 for a in axes)


def _sdf_from_boolean(inside, spacing):
    """Signed distance field: negative inside, positive outside, zero on surface.

    distance_transform_edt(mask) returns 0 for False (background) voxels and
    distance-to-nearest-background for True (foreground) voxels.
    So d_in = edt(inside) = boundary distance for True, 0 for False.
    And d_out = edt(~inside) = boundary distance for outside, 0 for inside.
    SDF = d_out - d_in → negative inside, positive outside.
    """
    sampling = list(spacing) if np.ndim(spacing) else [spacing] * 3
    d_in = distance_transform_edt(inside, sampling=sampling)
    d_out = distance_transform_edt(~inside, sampling=sampling)
    return d_out - d_in


def _build_sas_solid(atoms, coords, axes, vdw_offset, probe_radius):
    """Boolean grid: True where inside SAS (= union of vdw+probe spheres)."""
    ax, ay, az = axes
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing='ij')
    inside = np.zeros(gx.shape, dtype=bool)
    for i, (cx, cy, cz) in enumerate(coords):
        R = _get_vdw_radius(atoms[i], vdw_offset) + probe_radius
        inside |= (gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2 <= R**2
    return inside


def _write_solvent_cube(output, data_3d, axes, atoms, coords_ang, title, comment):
    """Write cube file. axes in Å, coords_ang in Å — converted to Bohr."""
    ax, ay, az = axes
    origin = np.array([ax[0], ay[0], az[0]]) * _AU
    vx = np.array([ax[1] - ax[0], 0.0, 0.0]) * _AU
    vy = np.array([0.0, ay[1] - ay[0], 0.0]) * _AU
    vz = np.array([0.0, 0.0, az[1] - az[0]]) * _AU

    atoms_data = []
    for i, el in enumerate(atoms):
        atoms_data.append({
            'atomic_number': _guess_atomic_number(el),
            'charge': 0.0,
            'coords': coords_ang[i] * _AU,
        })

    cube_data = {
        'title': title,
        'comment': comment,
        'natoms': len(atoms),
        'origin': origin,
        'grid_size': data_3d.shape,
        'voxel_vectors': np.array([vx, vy, vz]),
        'atoms': atoms_data,
        'data': data_3d,
        'units': 'Bohr',
    }
    write_gaussian_cube(output, cube_data)


def _grid_params(atoms, coords, spacing, padding, n_voxels, extra_pad=0.0):
    """Compute grid axes and dimensions."""
    max_r = extra_pad
    for el in atoms:
        max_r = max(max_r, _get_vdw_radius(el) + _PROBE_RADIUS)
    pad = max_r + padding
    _, axes, dims = generate_grid(coords, spacing=spacing, padding=pad,
                                  n_voxels=n_voxels)
    return axes, dims


def compute_sas(atoms, coords, spacing=0.1, padding=3.0, n_voxels=None,
                probe_radius=_PROBE_RADIUS, vdw_offset=0.0):
    """Compute SAS signed distance field (analytical, single-pass).

    phi_SAS(p) = min_i { ||p - c_i|| - (r_vdw_i + probe_radius) }
    Negative inside SAS, positive outside, zero on surface.
    """
    max_r = max(_get_vdw_radius(el, vdw_offset) + probe_radius for el in atoms)
    pad = max_r + padding

    _, axes, dims = generate_grid(coords, spacing=spacing, padding=pad,
                                  n_voxels=n_voxels)
    ax, ay, az = axes
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing='ij')

    phi = np.full(gx.shape, np.inf)
    for i, (cx, cy, cz) in enumerate(coords):
        R = _get_vdw_radius(atoms[i], vdw_offset) + probe_radius
        dist = np.sqrt((gx - cx)**2 + (gy - cy)**2 + (gz - cz)**2) - R
        phi = np.minimum(phi, dist)

    return phi, axes, dims


def compute_ses(atoms, coords, spacing=0.1, padding=3.0, n_voxels=None,
                probe_radius=_PROBE_RADIUS, vdw_offset=0.0):
    """Compute SES signed distance field via EDTSurf two-step EDT.

    1. Build SAS solid (union of vdw+probe spheres)
    2. EDT of SAS solid → distance from interior to SAS boundary
    3. SES solid = SAS points where EDT > probe_radius
    4. SDF of SES solid
    """
    max_r = max(_get_vdw_radius(el, vdw_offset) + probe_radius for el in atoms)
    pad = max_r + padding

    _, axes, dims = generate_grid(coords, spacing=spacing, padding=pad,
                                  n_voxels=n_voxels)

    # Step 1: SAS solid
    sas_inside = _build_sas_solid(atoms, coords, axes, vdw_offset, probe_radius)

    h = _axes_spacing(axes)  # actual spacing (differs from `spacing` with n_voxels)

    # Step 2: EDT of SAS solid → distance from each interior voxel to boundary
    edt_sas = distance_transform_edt(sas_inside, sampling=h)

    # Step 3: SES solid = interior points at least probe_radius away from boundary.
    # (Discretisation: the EDT measures to the nearest outside voxel centre, so the
    # SES volume is slightly over-estimated: ~+1 % at 0.1 Å, ~+10 % at 0.2 Å spacing.)
    ses_inside = sas_inside & (edt_sas > probe_radius)

    # Step 4: Signed distance field of SES solid
    phi = _sdf_from_boolean(ses_inside, h)

    return phi, axes, dims


def build_sas_cube(atoms, coords, output='sas.cube',
                   spacing=0.1, padding=3.0, n_voxels=None,
                   probe_radius=_PROBE_RADIUS, vdw_offset=0.0):
    """Compute SAS and write to cube file."""
    data, axes, dims = compute_sas(atoms, coords, spacing, padding, n_voxels,
                                   probe_radius, vdw_offset)
    vol = np.sum(data < 0) * np.prod(_axes_spacing(axes))
    print_info(f'SAS grid: {dims[0]}x{dims[1]}x{dims[2]} = {np.prod(dims):,} points')
    print_info(f'range: [{data.min():.4f}, {data.max():.4f}]')
    print_info(f'volume: {vol:.2f} A^3')
    _write_solvent_cube(output, data, axes, atoms, coords,
                        'Solvent Accessible Surface',
                        f'SAS grid: {dims[0]}x{dims[1]}x{dims[2]}')
    logger.debug(f'wrote {output}')


def build_ses_cube(atoms, coords, output='ses.cube',
                   spacing=0.1, padding=3.0, n_voxels=None,
                   probe_radius=_PROBE_RADIUS, vdw_offset=0.0):
    """Compute SES and write to cube file."""
    data, axes, dims = compute_ses(atoms, coords, spacing, padding, n_voxels,
                                   probe_radius, vdw_offset)
    vol = np.sum(data < 0) * np.prod(_axes_spacing(axes))
    print_info(f'SES grid: {dims[0]}x{dims[1]}x{dims[2]} = {np.prod(dims):,} points')
    print_info(f'range: [{data.min():.4f}, {data.max():.4f}]')
    print_info(f'volume: {vol:.2f} A^3')
    _write_solvent_cube(output, data, axes, atoms, coords,
                        'Solvent Excluded Surface',
                        f'SES grid: {dims[0]}x{dims[1]}x{dims[2]}')
    logger.debug(f'wrote {output}')
