import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
import periodictable as pt

from frogchem.orca_parser.orb_type import ORB_POLINOMIAL, CARTESIAN_CODES, CARTESIAN_EXPONENTS
from frogchem.io import write_gaussian_cube
import logging
from frogchem.printer import print_info, print_warning

logger = logging.getLogger(__name__)

_AU = 1.8897259886  # Å → Bohr

# Cartesian functions (Gaussian 6D/10F): code -> (a, b, c) exponents of x^a y^b z^c
_CART_EXP = {code: CARTESIAN_EXPONENTS[name] for name, code in CARTESIAN_CODES.items()}

def _orbital_letter(ang_code: int) -> str:
    name, *_ = ORB_POLINOMIAL[ang_code]
    return name[0]


def _solid_harmonic_poly(ang_code: int, dx, dy, dz) -> np.ndarray:
    if ang_code == 0:    # s
        return np.ones_like(dx)
    elif ang_code == 1:  # px
        return dx
    elif ang_code == 2:  # py
        return dy
    elif ang_code == 3:  # pz
        return dz
    elif ang_code == 4:  # dz2 = 2*z^2 - x^2 - y^2
        return 2.0*dz*dz - dx*dx - dy*dy
    elif ang_code == 5:  # dxz
        return dx * dz
    elif ang_code == 6:  # dyz
        return dy * dz
    elif ang_code == 7:  # dx2y2 = x^2 - y^2
        return dx*dx - dy*dy
    elif ang_code == 8:  # dxy
        return dx * dy
    elif ang_code == 9:  # f0 = z*(2*z^2 - 3*x^2 - 3*y^2)
        return dz * (2.0*dz*dz - 3.0*dx*dx - 3.0*dy*dy)
    elif ang_code == 10: # f+1 = x*(4*z^2 - x^2 - y^2)
        return dx * (4.0*dz*dz - dx*dx - dy*dy)
    elif ang_code == 11: # f-1 = y*(4*z^2 - x^2 - y^2)
        return dy * (4.0*dz*dz - dx*dx - dy*dy)
    elif ang_code == 12: # f+2 = z*(x^2 - y^2)
        return dz * (dx*dx - dy*dy)
    elif ang_code == 13: # f-2 = x*y*z
        return dx * dy * dz
    # ORCA's real f(+3)/f(-3) functions carry the opposite phase to the standard real solid
    # harmonics (same convention handled by orca_2mkl, Multiwfn, IOData). With the standard
    # sign the ORCA MO coefficients are not orthonormal (C^T S C != I); with this sign they are.
    elif ang_code == 14: # f+3 = -x*(x^2 - 3*y^2)   (ORCA phase)
        return -dx * (dx*dx - 3.0*dy*dy)
    elif ang_code == 15: # f-3 = -y*(3*x^2 - y^2)   (ORCA phase)
        return -dy * (3.0*dx*dx - dy*dy)
    elif ang_code in _CART_EXP:  # Cartesian x^a y^b z^c (Gaussian 6D/10F)
        a, b, c = _CART_EXP[ang_code]
        return dx**a * dy**b * dz**c
    else:
        return np.ones_like(dx)


# ORCA orders p-type basis functions as pz, px, py (types 3, 1, 2)
_ANG_PER_LETTER = {
    's': [0],
    'p': [3, 1, 2],
    'd': [4, 5, 6, 7, 8],
    'f': [9, 10, 11, 12, 13, 14, 15],
}


def build_basis_table(calc, geom=None) -> List[Dict[str, Any]]:
    if geom is None:
        geom = calc.geom * _AU
    n_bf = len(calc.mask_atoms)
    basis_table = []

    element_shell_keys: Dict[str, List[str]] = {}
    for el, shells in calc.basis_coeff.items():
        element_shell_keys[el] = list(shells.keys())

    prev_atom = -1
    shell_ptr: Dict[str, int] = defaultdict(int)

    i = 0
    while i < n_bf:
        a = int(calc.mask_atoms[i])
        ac = int(calc.mask_orbital_type[i])
        letter = _orbital_letter(ac)
        el = calc.atoms[a]
        center = geom[a]

        if a != prev_atom:
            shell_ptr.clear()
            prev_atom = a

        block_start = i
        while i < n_bf and int(calc.mask_atoms[i]) == a and _orbital_letter(int(calc.mask_orbital_type[i])) == letter:
            i += 1
        block_end = i

        if el not in element_shell_keys:
            continue

        keys = element_shell_keys[el]
        # Shell size: 6 Cartesian d / 10 Cartesian f (Gaussian 6D/10F), otherwise the pure size
        cart = int(calc.mask_orbital_type[block_start]) in _CART_EXP
        n_per_shell = {'d': 6, 'f': 10}[letter] if cart else len(_ANG_PER_LETTER[letter])
        block_size = block_end - block_start
        n_shells_in_block = block_size // n_per_shell

        for shell_in_block in range(n_shells_in_block):
            ptr = shell_ptr[letter]
            expected_key = None
            count = 0
            for k in keys:
                if _letter_from_key(k) == letter:
                    if count == ptr:
                        expected_key = k
                        break
                    count += 1

            if expected_key is None:
                print_warning(f'Shell mismatch for {el} atom {a}: letter {letter} ptr={ptr}')
                break

            shell_ptr[letter] += 1
            primitives = calc.basis_coeff[el][expected_key]

            # Function order inside the shell as printed by the program (ORCA: pz px py;
            # Gaussian: px py pz), so the table rows always match the MO coefficient rows
            first = block_start + shell_in_block * n_per_shell
            for ang_code in calc.mask_orbital_type[first:first + n_per_shell]:
                ang_code = int(ang_code)
                basis_table.append({
                    'atom_idx': a,
                    'center': center,
                    'element': el,
                    'ang_code': ang_code,
                    'primitives': primitives,
                })

    return basis_table


def _letter_from_key(key: str) -> str:
    m = re.search(r'[a-z]', key)
    return m.group() if m else ''


def _ang_momentum(ang_code: int) -> int:
    if ang_code in _CART_EXP:
        return sum(_CART_EXP[ang_code])
    return {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}.get(ang_code, 3)


def _cartesian_norm(a: int, b: int, c: int) -> float:
    """(2/π)^(3/4) 2^L / sqrt((2a-1)!!(2b-1)!!(2c-1)!!): normalises x^a y^b z^c exp(-αr²) with α^(L/2+3/4)."""
    dfact = {0: 1, 1: 1, 2: 3, 3: 15}
    return (2.0 / np.pi) ** 0.75 * 2.0 ** (a + b + c) / np.sqrt(dfact[a] * dfact[b] * dfact[c])


_NORM_CONST = {
    0:  0.712705,   # s
    1:  1.425411,   # px
    2:  1.425411,   # py
    3:  1.425411,   # pz
    4:  0.822961,   # dz2
    5:  2.850822,   # dxz
    6:  2.850822,   # dyz
    7:  1.425411,   # dx2y2
    8:  2.850822,   # dxy
    9:  0.736079,   # f0
    10: 0.901509,   # f+1
    11: 0.901509,   # f-1
    12: 2.850822,   # f+2
    13: 5.701644,   # f-2
    14: 1.163843,   # f+3
    15: 1.163843,   # f-3
}
_NORM_CONST.update({code: _cartesian_norm(*abc) for code, abc in _CART_EXP.items()})


def eval_basis_1(basis_func: Dict[str, Any], points: np.ndarray) -> np.ndarray:
    center = basis_func['center']
    ang_code = basis_func['ang_code']
    primitives = basis_func['primitives']
    l = _ang_momentum(ang_code)
    C = _NORM_CONST[ang_code]

    dr = points - center
    dx, dy, dz = dr[:, 0], dr[:, 1], dr[:, 2]
    r2 = dx*dx + dy*dy + dz*dz

    val = np.zeros(len(points), dtype=np.float64)
    for alpha, coeff in primitives:
        norm = C * alpha ** (0.5 * l + 0.75)
        val += coeff * norm * np.exp(-alpha * r2)

    val *= _solid_harmonic_poly(ang_code, dx, dy, dz)
    return val


def eval_all_basis(basis_table: List[Dict[str, Any]], points: np.ndarray) -> np.ndarray:
    n_bf = len(basis_table)
    n_pts = len(points)
    bf = np.zeros((n_bf, n_pts), dtype=np.float64)
    for idx, b in enumerate(basis_table):
        bf[idx, :] = eval_basis_1(b, points)
    return bf


def compute_density_on_points(
    calc,
    points: np.ndarray,
    basis_table: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    if basis_table is None:
        basis_table = build_basis_table(calc)

    bf = eval_all_basis(basis_table, points)
    rho = np.zeros(len(points), dtype=np.float64)

    # Check if we are dealing with an Open-Shell calculation (Unrestricted)
    if calc.mult != 1 and hasattr(calc, 'alpha_coeff') and hasattr(calc, 'beta_coeff'):
        # Sum Alpha density using Alpha dimensions
        for i in range(calc.alpha_coeff.shape[1]):
            if i < len(calc.alpha_occ) and calc.alpha_occ[i] > 1e-12:
                mo_i = bf.T @ calc.alpha_coeff[:, i]
                rho += calc.alpha_occ[i] * mo_i * mo_i
        
        # Sum Beta density using Beta dimensions to prevent IndexError
        for i in range(calc.beta_coeff.shape[1]):
            if i < len(calc.beta_occ) and calc.beta_occ[i] > 1e-12:
                mo_i = bf.T @ calc.beta_coeff[:, i]
                rho += calc.beta_occ[i] * mo_i * mo_i
    else:
        # Standard Closed-Shell (Restricted) path
        mo_coeff = calc.orbital_coefficients
        occ = calc.occupation
        n_mo_total = mo_coeff.shape[1]
        
        occ_padded = np.zeros(n_mo_total)
        occ_padded[:min(len(occ), n_mo_total)] = occ[:min(len(occ), n_mo_total)]

        for i in range(n_mo_total):
            occ_i = occ_padded[i]
            if occ_i < 1e-12:
                continue
            mo_i = bf.T @ mo_coeff[:, i]
            rho += occ_i * mo_i * mo_i

    return rho


def generate_grid(
    geom: np.ndarray,
    spacing: float = 0.15,
    padding: float = 5.0,
    n_voxels: Optional[int] = None,
):
    lo = geom.min(axis=0) - padding
    hi = geom.max(axis=0) + padding
    if n_voxels:
        axes = [np.linspace(lo[j], hi[j], n_voxels) for j in range(3)]
    else:
        axes = [np.arange(lo[j], hi[j] + spacing, spacing) for j in range(3)]
    ax, ay, az = axes
    nx, ny, nz = len(ax), len(ay), len(az)
    grid_3d = np.stack(np.meshgrid(ax, ay, az, indexing='ij'), axis=-1)
    return grid_3d, (ax, ay, az), (nx, ny, nz)


def _guess_atomic_number(el: str) -> int:
    return getattr(pt, el.capitalize(), pt.C).number


def _grid_and_basis(calc, spacing, padding, n_voxels):
    geom_bohr = calc.geom * _AU
    table = build_basis_table(calc, geom_bohr)
    print_info(f'basis functions: {len(table)}')
    print_info(f'MOs: {calc.orbital_coefficients.shape[1]}')
    print_info(f'atoms: {len(calc.atoms)}')

    grid_3d, axes, (nx, ny, nz) = generate_grid(geom_bohr, spacing * _AU, padding * _AU, n_voxels)
    points = grid_3d.reshape(-1, 3)
    print_info(f'grid: {nx} x {ny} x {nz} = {len(points):,} points')
    return table, points, axes, (nx, ny, nz)


def _make_cube_data(data_3d: np.ndarray, axes, calc, title: str = '', comment: str = '') -> dict:
    """Construct a cube_data dict from density grid data without writing to disk."""
    nx, ny, nz = data_3d.shape
    ax, ay, az = axes

    origin = np.array([ax[0], ay[0], az[0]])
    vx = np.array([ax[1] - ax[0], 0.0, 0.0])
    vy = np.array([0.0, ay[1] - ay[0], 0.0])
    vz = np.array([0.0, 0.0, az[1] - az[0]])

    atoms_data = []
    for i in range(len(calc.atoms)):
        z = _guess_atomic_number(calc.atoms[i])
        coords = calc.geom[i] * _AU
        atoms_data.append({
            'atomic_number': z,
            'charge': 0.0,
            'coords': coords
        })

    return {
        'title': title,
        'comment': comment,
        'natoms': len(calc.atoms),
        'origin': origin,
        'grid_size': (nx, ny, nz),
        'voxel_vectors': np.array([vx, vy, vz]),
        'atoms': atoms_data,
        'data': data_3d,
        'units': 'Bohr'
    }


def _finish_cube(output: str, data_3d: np.ndarray, axes, calc, title: str, comment: str) -> dict:
    cube_data = _make_cube_data(data_3d, axes, calc, title, comment)

    print_info(f'range: [{data_3d.min():.6f}, {data_3d.max():.6f}]')
    write_gaussian_cube(filename=output, cube_data=cube_data)
    logger.debug(f'wrote {output}')
    return cube_data


def build_density_cube(
    calc,
    output: str = 'density.cube',
    spacing: float = 0.15,
    padding: float = 5.0,
    n_voxels: Optional[int] = None,
) -> None:
    table, points, axes, (nx, ny, nz) = _grid_and_basis(calc, spacing, padding, n_voxels)
    print_info('evaluating density...')
    rho = compute_density_on_points(calc, points, table).reshape(nx, ny, nz)
    _finish_cube(output, rho, axes, calc,
                 f'Electron Density from {calc.fname}',
                 f'Grid: {nx}x{ny}x{nz} voxels')


def build_spin_density_cube(
    calc,
    output: str = 'spin_density.cube',
    spacing: float = 0.15,
    padding: float = 5.0,
    n_voxels: Optional[int] = None,
) -> None:
    if calc.mult == 1:
        print_warning('Closed-shell (mult=1): spin density is zero everywhere')
        return

    table, points, axes, (nx, ny, nz) = _grid_and_basis(calc, spacing, padding, n_voxels)
    print_info('evaluating spin density...')

    bf = eval_all_basis(table, points)
    rho = np.zeros(len(points))
    
    # Loop safely on Alpha
    for i in range(calc.alpha_coeff.shape[1]):
        if i < len(calc.alpha_occ):
            mo = bf.T @ calc.alpha_coeff[:, i]
            rho += calc.alpha_occ[i] * mo * mo
            
    # Loop safely on Beta
    for i in range(calc.beta_coeff.shape[1]):
        if i < len(calc.beta_occ):
            mo = bf.T @ calc.beta_coeff[:, i]
            rho -= calc.beta_occ[i] * mo * mo

    _finish_cube(output, rho.reshape(nx, ny, nz), axes, calc,
                 f'Spin Density from {calc.fname}',
                 f'Grid: {nx}x{ny}x{nz} voxels')
    
def build_orbital_cube(
    calc,
    orbital_index: int,
    output: str = 'orbital.cube',
    spacing: float = 0.15,
    padding: float = 5.0,
    n_voxels: Optional[int] = None,
    alpha: bool = True,
    beta: bool = False,
) -> None:
    table, points, axes, (nx, ny, nz) = _grid_and_basis(calc, spacing, padding, n_voxels)
    print_info(f'evaluating orbital {orbital_index}...')

    bf = eval_all_basis(table, points)
    
    if hasattr(calc, 'mult') and calc.mult != 1:
        if beta and hasattr(calc, 'beta_coeff'):
            coeff = calc.beta_coeff[:, orbital_index]
            label = 'beta'
        elif hasattr(calc, 'alpha_coeff'):
            coeff = calc.alpha_coeff[:, orbital_index]
            label = 'alpha'
        else:
            coeff = calc.orbital_coefficients[:, orbital_index]
            label = 'open-shell'
    else:
        coeff = calc.orbital_coefficients[:, orbital_index]
        label = 'closed-shell'
        
    mo = bf.T @ coeff

    _finish_cube(output, mo.reshape(nx, ny, nz), axes, calc,
                 f'Orbital {orbital_index} ({label}) from {calc.fname}',
                 f'Grid: {nx}x{ny}x{nz} voxels')


def compute_difference_density_on_points(
    calc,
    transitions: list,
    points: np.ndarray,
    basis_table: Optional[List[Dict[str, Any]]] = None,
    bf_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute excited-state difference density at given grid points.

    For each TD-DFT transition (i → a) with squared amplitude w_ia,
    the difference density is:

        Δρ(r) = Σ_ia  w_ia  [ |φ_a(r)|²  −  |φ_i(r)|² ]

    Parameters
    ----------
    calc : Calculation
        Parsed ORCA/Gaussian calculation with ``orbital_coefficients``.
    transitions : list of (int, int, float)
        Each tuple is ``(occ_orbital_idx, virt_orbital_idx, squared_weight)``
        as stored in ``calc.tddft_states[mult][state]['population']``.
    points : np.ndarray, shape (N, 3)
        Grid points in **Bohr**.
    basis_table : list of dict, optional
        Pre-built basis table (from ``build_basis_table``).
    bf_matrix : np.ndarray of shape (n_bf, N), optional
        Pre-evaluated basis-function values at *points*.
        Pass this when processing multiple states to avoid recomputation.

    Returns
    -------
    np.ndarray, shape (N,)
        Difference density value at each grid point.
    """
    if bf_matrix is None:
        if basis_table is None:
            basis_table = build_basis_table(calc)
        bf_matrix = eval_all_basis(basis_table, points)

    mo_coeff = calc.orbital_coefficients
    n_bf = mo_coeff.shape[0]
    delta_rho = np.zeros(len(points), dtype=np.float64)

    for item in transitions:
        orb_from, orb_to, weight = item[0], item[1], item[2]
        spin = item[3] if len(item) > 3 else ''
        if weight < 1e-6:
            continue

        if spin == 'b' and hasattr(calc, 'beta_coeff') and getattr(calc, 'mult', 1) != 1:
            coeffs = calc.beta_coeff
        elif hasattr(calc, 'alpha_coeff') and getattr(calc, 'mult', 1) != 1:
            coeffs = calc.alpha_coeff
        else:
            coeffs = mo_coeff

        if orb_from >= coeffs.shape[1] or orb_to >= coeffs.shape[1]:
            continue
        mo_occ = bf_matrix[:n_bf].T @ coeffs[:, orb_from]
        mo_virt = bf_matrix[:n_bf].T @ coeffs[:, orb_to]
        delta_rho += weight * (mo_virt * mo_virt - mo_occ * mo_occ)

    return delta_rho


def build_nbo_cube(
    calc,
    nbo_index: int,
    coeff_vector: np.ndarray,
    output: str = 'nbo.cube',
    spacing: float = 0.15,
    padding: float = 5.0,
    n_voxels: Optional[int] = None,
    label: str = '',
    occupancy: Optional[float] = None,
    energy: Optional[float] = None,
) -> str:
    """Build a 3D Gaussian cube file for a Natural Bond Orbital (NBO).

    Args:
        calc: Calculation object containing basis set and geometry.
        nbo_index: 1-based NBO index matching the NBO output listing.
        coeff_vector: 1D array of coefficients in the AO basis.
        output: Destination path for the .cube file.
        spacing: Grid spacing in Angstroms (default: 0.15).
        padding: Grid box padding around molecule in Angstroms (default: 5.0).
        n_voxels: Fixed voxel count along axes if desired.
        label: Description of the NBO (e.g. 'LP ( 1) O 17' or 'BD ( 1) C 1- C 2').
        occupancy: Orbital electron occupancy if known.
        energy: Orbital energy in a.u. if known.

    Returns:
        str: Path of the generated cube file.
    """
    table, points, axes, (nx, ny, nz) = _grid_and_basis(calc, spacing, padding, n_voxels)
    lbl_str = f" ({label})" if label else ""
    print_info(f'evaluating NBO {nbo_index}{lbl_str}...')

    bf = eval_all_basis(table, points)
    mo = bf.T @ coeff_vector

    title = f'NBO {nbo_index}: {label}' if label else f'NBO {nbo_index}'
    comment_parts = [f'Grid: {nx}x{ny}x{nz}']
    if occupancy is not None:
        comment_parts.append(f'Occ: {occupancy:.4f}')
    if energy is not None:
        comment_parts.append(f'E: {energy:.4f} a.u.')
    comment = ' | '.join(comment_parts)

    _finish_cube(output, mo.reshape(nx, ny, nz), axes, calc, title=title, comment=comment)
    return output


def generate_visualization_scripts(
    cube_files: List[str],
    output_dir: Union[str, Path] = '.',
    isovalue: float = 0.05,
) -> Dict[str, str]:
    """Generate ChimeraX (.cxc) and PyMOL (.pml) helper scripts to view generated cubes.

    Args:
        cube_files: List of file paths to generated .cube files.
        output_dir: Directory where scripts should be written.
        isovalue: Surface isovalue (default: 0.05 a.u.).

    Returns:
        Dict[str, str]: Paths to created 'view_chimerax.cxc' and 'view_pymol.pml'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cxc_path = out_dir / 'view_chimerax.cxc'
    pml_path = out_dir / 'view_pymol.pml'

    # ChimeraX script
    cxc_lines = [
        "# ChimeraX script to visualize NBO surfaces",
        "# Open with: chimerax view_chimerax.cxc",
        "set bgColor white",
    ]
    for idx, cf in enumerate(cube_files, start=1):
        rel_cf = Path(cf).name if Path(cf).parent.resolve() == out_dir.resolve() else str(cf)
        cxc_lines.append(f"open {rel_cf}")
        cxc_lines.append(f"volume #{idx} level {isovalue:.4f} color #3b82f6 transparency 0.3")
        cxc_lines.append(f"volume #{idx} level -{isovalue:.4f} color #ef4444 transparency 0.3")
    cxc_path.write_text('\n'.join(cxc_lines) + '\n', encoding='utf-8')

    # PyMOL script
    pml_lines = [
        "# PyMOL script to visualize NBO surfaces",
        "# Open with: pymol view_pymol.pml",
        "bg_color white",
    ]
    for idx, cf in enumerate(cube_files, start=1):
        rel_cf = Path(cf).name if Path(cf).parent.resolve() == out_dir.resolve() else str(cf)
        obj_name = f"nbo_{idx}"
        pml_lines.append(f"load {rel_cf}, map_{obj_name}")
        pml_lines.append(f"isosurface {obj_name}_pos, map_{obj_name}, {isovalue:.4f}")
        pml_lines.append(f"isosurface {obj_name}_neg, map_{obj_name}, -{isovalue:.4f}")
        pml_lines.append(f"color marine, {obj_name}_pos")
        pml_lines.append(f"color firebrick, {obj_name}_neg")
        pml_lines.append(f"set transparency, 0.3, {obj_name}_pos")
        pml_lines.append(f"set transparency, 0.3, {obj_name}_neg")
    pml_path.write_text('\n'.join(pml_lines) + '\n', encoding='utf-8')

    return {'chimerax': str(cxc_path), 'pymol': str(pml_path)}


