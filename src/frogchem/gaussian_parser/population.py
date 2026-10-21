import re
import numpy as np

_CHARGE_ROW = re.compile(r'^\s*(\d+)\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?\s*$')


def parse_atomic_charges(fl, reg, kind):
    """Mulliken or Löwdin charges of the LAST block, as the ORCA parser: {idx (0-based): {'atom', 'charge'}}.

    Open-shell Mulliken blocks ('Mulliken charges and spin densities') also give 'spin'.
    Returns None if the block is not printed (Löwdin needs iop(6/80=1)).
    """
    matches = list(re.finditer(reg[f'{kind}_start'], fl))
    if not matches:
        return None
    block = fl[matches[-1].end():].split(reg[f'{kind}_end'])[0]
    charges = {}
    for line in block.splitlines():
        m = _CHARGE_ROW.match(line)
        if not m:
            continue
        entry = {'atom': m.group(2), 'charge': float(m.group(3))}
        if m.group(4) is not None:
            entry['spin'] = float(m.group(4))
        charges[int(m.group(1)) - 1] = entry
    return charges or None


def parse_npa_charges(fl, reg):
    if reg['npa_start'] not in fl:
        return None

    block = fl.split(reg['npa_start'])[-1].split(reg['npa_end'])[0]
    start = False
    charges = {}
    for line in block.splitlines():
        if 'Atom' in line and 'Charge' in line:
            start = True
            continue
        if start:
            match = re.search(reg['npa_pattern'], line)
            if match:
                idx = int(match.group(2)) - 1
                charges[idx] = {
                    'atom': match.group(1),
                    'charge': float(match.group(3)),
                    'core': float(match.group(4)),
                    'valence': float(match.group(5)),
                    'rydberg': float(match.group(6)),
                    'total_pop': float(match.group(7)),
                }
    return charges if charges else None
