import numpy as np
import re
from collections import defaultdict

from frogchem.orca_parser.orb_type import ORB_TYPE


def get_number(str_val):
    match = re.search(r"\d+", str_val)
    if match:
        return match.group()
    return None


def create_mask(section):
    mask_atoms, maske_type_n, mask_type = [], [], []
    for i in section.split('              ')[0].splitlines():
        if not i:
            continue
        if '--' in i:
            continue
        parts = i.split()
        if len(parts) < 2:
            continue
        mask_atoms.append(get_number(parts[0].strip()))
        maske_type_n.append(parts[1].strip()[0])
        mask_type.append(ORB_TYPE[parts[1].strip()[1:]])
    return np.array(mask_atoms), np.array(maske_type_n), np.array(mask_type)


def check_shell_type(shell_type, element, basis_sets):
    filtered_keys = [key for key in basis_sets[element].keys() if str(key).endswith(shell_type)]
    if filtered_keys:
        return len(filtered_keys)
    return 0


def get_MOs_coefficients(fl, reg):
    matches = re.findall(reg['orbitals'], fl)
    if not matches:
        return None, None, None, None

    block = matches[-1]
    chunks = block.split('--------\n')

    if len(chunks) < 2:
        return None, None, None, None

    data_chunks = chunks[2:]

    mask_atoms, maske_type_n, mask_type = create_mask(data_chunks[1].strip())
    n_basis = len(mask_type)
    if n_basis == 0:
        return None, None, None, None

    cols_data = []

    for chunk in data_chunks:
        lines = chunk.splitlines()
        block_coeffs = []

        for line in lines:
            line = line.strip()
            if not line or '--' in line or 'SPIN' in line or 'MOLECULAR ORBITALS' in line:
                continue

            floats = re.findall(r"-?\d+\.\d+", line)

            if floats and any(orb_label in line for orb_label in ['s', 'p', 'd', 'f', 'g']):
                block_coeffs.append([float(x) for x in floats])

        if not block_coeffs:
            continue

        block_matrix = np.array(block_coeffs)

        if block_matrix.shape[0] == n_basis:
            cols_data.append(block_matrix)

    if not cols_data:
        return None, None, None, None

    final_coeff = np.hstack(cols_data)
    return mask_atoms, maske_type_n, mask_type, final_coeff


def get_MOs_energies(fl, reg):
    # Last block, consistent with get_MOs_coefficients (final geometry in multi-step jobs)
    fl = re.findall(reg['orbital_energies'], fl)[-1].splitlines()[4:]

    en = []
    oc = []

    for i in fl:
        if '--' in i or '*' in i or 'SPIN' in i or 'NO' in i or 'ORBITAL' in i:
            continue
        if not i.strip():
            continue
        parts = i.strip().split()
        if len(parts) >= 3:
            en.append(parts[2])
            oc.append(parts[1])

    return np.array(en, dtype=float), np.array(oc, dtype=float)


def get_basis_coefficients(fl, reg):
    basis_sets = defaultdict(lambda: defaultdict(list))
    fl = re.findall(reg['basis_coefficients'], fl)[0].splitlines()[:]

    element = None
    shell_type = None

    for line in fl:
        line = line.strip()

        if line.startswith("# Basis set for element :"):
            element = line.split(":")[-1].strip()
        elif line.startswith("NewGTO"):
            continue
        elif re.match(r'^[SPDFG] \d+', line):
            parts = line.split()
            shell_type = parts[0].lower()
            check_tmp = check_shell_type(shell_type, element, basis_sets)
            shell_type = f"{check_tmp + 1}{shell_type}"
        elif line.startswith("end;"):
            element = None
            shell_type = None
        elif element and shell_type:
            parts = line.split()
            if len(parts) == 3:
                _, exponent, coefficient = parts
                basis_sets[element][shell_type].append((float(exponent), float(coefficient)))

    for element in basis_sets:
        for shell in basis_sets[element]:
            basis_sets[element][shell] = np.array(basis_sets[element][shell])

    return dict(basis_sets)