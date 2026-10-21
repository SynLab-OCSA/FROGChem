import logging
import numpy as np
import re

logger = logging.getLogger(__name__)

def get_freq(fl, reg):
    f_ = re.findall(reg['frequency'], fl)[-1]
    match = np.array(re.findall(r'(\d+):\s+(-?[\d.]+)\s+cm\*\*-1*', f_), dtype=np.float64)
    numbers_mode, freq = match[:, 0].astype(int), match[:, 1]
    return numbers_mode, freq.astype(np.float64)


def get_normal_modes(fl, reg, N):
    matches = re.findall(reg['normal_modes'], fl, re.DOTALL)
    if not matches:
        raise ValueError("Pattern 'normal_modes' non trovato nel file")

    target_text = matches[-1].split('\n\n')[-1]
    matrix = np.zeros((N, N))
    header_pattern = r'^\s*(\d+(?:\s+\d+)*)\s*$'
    data_pattern = r'^\s*(\d+)\s+((?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*)+)'

    lines = target_text.strip().split('\n')
    current_columns = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        header_match = re.match(header_pattern, line)
        if header_match:
            current_columns = [int(x) for x in header_match.group(1).split()]
            continue

        data_match = re.match(data_pattern, line)
        if data_match and current_columns:
            try:
                row_idx = int(data_match.group(1))
                coefficients_str = data_match.group(2)
                coefficients = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', coefficients_str)
                coefficients = [float(c) for c in coefficients]

                for i, coeff in enumerate(coefficients):
                    if i < len(current_columns) and row_idx < N and current_columns[i] < N:
                        matrix[row_idx, current_columns[i]] = coeff
            except (ValueError, IndexError) as e:
                logger.warning("Errore processando riga dati '%s': %s", line, e)
                continue

    return matrix


def get_hessian_graphs_impulses(fl, reg):
    d = {}
    ir = fl.split(reg['ir_graph'])[-1].split('\n\n')[0].splitlines()[3:]
    if len(ir) > 1:
        d['ir'] = [i.split()[reg['ir_imp_idx']] for i in ir if i]
    vcd = fl.split(reg['vcd_graph'])[-1].split('\n\n')[0].splitlines()[3:]
    if len(vcd) > 1:
        d['vcd'] = [i.split()[reg['vcd_imp_idx']] for i in vcd if i]
    return d


# ---------------------------------------------------------------------------
# Localization — operate on Calculation data passed explicitly
# ---------------------------------------------------------------------------

def displace_geometry(calc, mode, max_displ=1, ngeom=30, path=None):
    """Write a multi-frame XYZ of the geometry displaced along normal mode `mode`.

    Frames span -max_displ..+max_displ (Å) in `ngeom` steps. The file at `path`
    (default 'displ_geometry_{mode}.xyz' in the current directory) is overwritten.
    """
    fname = path if path is not None else f'displ_geometry_{mode}.xyz'
    steps = np.linspace(-max_displ, max_displ, ngeom)
    mode_vec = calc.normal_modes[:, mode].reshape(calc.geom.shape)

    with open(fname, 'w') as f:
        for step in steps:
            frame = calc.geom + step * mode_vec
            f.write(f'{len(calc.geom)}\nDisplaced by {step:.3f} Å along mode {mode}\n')
            f.write('\n'.join(
                f'{calc.atoms[a]} {j[0]:>10.6f} {j[1]:>10.6f} {j[2]:>10.6f}'
                for a, j in enumerate(frame)
            ))
            f.write('\n')
    return True


def localize_mode(calc, mode: int) -> np.ndarray:
    displ = calc.normal_modes[:, mode].reshape(calc.geom.shape)
    sq = np.sum(displ ** 2, axis=1)
    total = np.sum(sq)
    if total < 1e-14:
        return np.zeros(len(calc.atoms))
    return (sq / total) * 100.0


def localize_mode_fragment(
    calc,
    mode: int,
    fragments: dict[str, list[int]],
) -> dict[str, float]:
    atomic = localize_mode(calc, mode)
    return {
        name: float(np.sum(atomic[indices]))
        for name, indices in fragments.items()
    }


def inverse_participation_ratio(calc, mode: int) -> float:
    displ = calc.normal_modes[:, mode].reshape(calc.geom.shape)
    sq = np.sum(displ ** 2, axis=1)
    num = np.sum(sq ** 2)
    denom = np.sum(sq) ** 2
    if denom < 1e-28:
        return 0.0
    return float(num / denom)


def localize_all_modes(calc, fragments=None) -> dict:
    n_modes = calc.normal_modes.shape[1]
    n_atoms = len(calc.atoms)
    ipr = np.zeros(n_modes)
    atomic = np.zeros((n_modes, n_atoms))
    for k in range(n_modes):
        ipr[k] = inverse_participation_ratio(calc, k)
        atomic[k] = localize_mode(calc, k)
    result = {'ipr': ipr, 'atomic': atomic}
    if fragments is not None:
        frag_contrib = {name: np.zeros(n_modes) for name in fragments}
        for k in range(n_modes):
            for name, indices in fragments.items():
                frag_contrib[name][k] = float(np.sum(atomic[k, indices]))
        result['fragments'] = frag_contrib
    return result