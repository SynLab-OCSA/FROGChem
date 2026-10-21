import numpy as np

_HARTREE_TO_KCAL = 627.5096080305927


def _numeric_rows(block: str) -> np.ndarray:
    # Keep only data lines (path, distance, energy); headers such as
    # "Images: Distance  (Bohr), Energy (Eh)" or "Interp.: ..." are skipped.
    rows = []
    for line in block.splitlines():
        try:
            vals = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(vals) >= 3:
            rows.append(vals[:3])
    return np.array(rows, dtype=float).reshape(-1, 3)


def parse_neb_interp(filename: str) -> tuple[list, list]:
    # Parse ORCA neb.interp file into images and interpolations
    
    with open(filename, 'r') as f:
        file_content = f.read()

    iters = file_content.split('Iteration')[1:]
    imags, interpols = [], []
    
    for i in iters:
        if not i.strip(): 
            continue
        parts = i.split('\n\n')
        if len(parts) < 2:
            continue
            
        imag_data, interp_data = parts[0], parts[1]

        img = _numeric_rows(imag_data)
        imags.append(np.column_stack((img[:, 0], img[:, 2] * _HARTREE_TO_KCAL)))

        intr = _numeric_rows(interp_data)
        interpols.append(np.column_stack((intr[:, 0], intr[:, 2] * _HARTREE_TO_KCAL)))

    return imags, interpols