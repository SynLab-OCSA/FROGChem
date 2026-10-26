import re
import numpy as np


def parse_uv(fl, reg):
    if reg['tddft_start'] not in fl:
        return np.zeros(shape=(1, 2))

    block = fl.split(reg['tddft_start'])[-1].split(reg['tddft_end'])[0]
    matches = re.findall(reg['tddft_state'], block)
    if not matches:
        return np.zeros(shape=(1, 2))
    data = np.array(matches, dtype=np.float64)
    return np.column_stack((data[:, 0], data[:, 1]))


def _strengths(block, reg):
    return np.array(re.findall(reg['ecd_pattern'], block, flags=re.MULTILINE), dtype=np.float64)


def parse_ecd(fl, reg):
    """(energy eV, R) per state, R = rotatory strength in 1e-40 cgs.

    Uses the length-gauge R (the 'R(length)' table), the gauge of ORCA's CD spectrum, so both programs
    give comparable numbers; falls back to the velocity-gauge table ('R(velocity)') if the length one
    is missing.
    """
    if reg['ecd_section_start'] not in fl:
        return np.zeros(shape=(1, 2))

    uv = parse_uv(fl, reg)
    tail = fl.split(reg['ecd_section_start'])[-1]
    velocity, _, length = tail.partition(reg['ecd_section_end'])
    length = length.split(reg['ecd_length_end'])[0].split(reg['tddft_start'])[0]
    for block in (length, velocity):
        strengths = _strengths(block, reg)
        if len(strengths) and len(strengths) == uv.shape[0]:
            return np.column_stack((uv[:, 0], strengths))
    return np.zeros(shape=(1, 2))


_MULT_NAMES = {'singlet': 1, 'doublet': 2, 'triplet': 3, 'quartet': 4, 'quintet': 5,
               'sextet': 6, 'septet': 7}
_STATE_RE = re.compile(
    r'Excited State\s+(\d+):\s+(\S+?)-\S*\s+([-+]?\d+\.\d+) eV.*?f=([-+]?\d+\.\d+)'
    r'(?:\s+<S\*\*2>=\s*([-+]?\d+\.\d+))?')
_TRANS_RE = re.compile(r'^\s*(\d+)([AB]?)\s*->\s*(\d+)([AB]?)\s+([-+]?\d+\.\d+)')


def _multiplicity(label: str) -> int:
    lab = label.lower()
    if lab in _MULT_NAMES:
        return _MULT_NAMES[lab]
    try:  # unrestricted runs print a spin estimate such as '2.034-A'
        return max(1, int(round(float(lab))))
    except ValueError:
        return 0


def parse_tddft_states(fl, reg) -> dict:
    """Build ``{mult: {state: {...}}}`` from the last Gaussian TD block (same layout as ORCA).

    Each state holds 'energy' (eV), 'fosc', 'S**2', 'population' (list of
    (from, to, weight[, spin]) with 0-based MO indices) and, when the rotatory strengths are
    printed, 'R' (length gauge, 1e-40 cgs, as returned by :func:`parse_ecd`). The weight is 2c² for
    restricted (closed-shell) excitations and c² for unrestricted (A/B labelled) ones;
    de-excitation ('<-') lines are ignored.

    States are numbered per multiplicity in order of energy (S1, S2, ... T1, T2, ... as in ORCA);
    Gaussian's own running number (singlets and triplets mixed with td=50-50) is kept in
    'gaussian_state'.
    """
    if reg['tddft_start'] not in fl:
        return {}
    block = fl.split(reg['tddft_start'])[-1].split(reg['tddft_end'])[0]

    states: dict = {}
    current = None
    for line in block.splitlines():
        m = _STATE_RE.search(line)
        if m:
            num, mult = int(m.group(1)), _multiplicity(m.group(2))
            current = {
                'energy': float(m.group(3)),
                'fosc': float(m.group(4)),
                'S**2': float(m.group(5)) if m.group(5) else 0.0,
                'population': [],
                'gaussian_state': num,
                '_order': len([1 for s in states.values() for _ in s]),
            }
            same_mult = states.setdefault(mult, {})
            same_mult[len(same_mult) + 1] = current
            continue
        t = _TRANS_RE.match(line)
        if t and current is not None:
            spin = t.group(2).lower()  # '' (restricted), 'a' or 'b'
            c = float(t.group(5))
            weight = c * c if spin else 2.0 * c * c
            current['population'].append((int(t.group(1)) - 1, int(t.group(3)) - 1, weight, spin))

    ecd = parse_ecd(fl, reg)
    n_states = sum(len(s) for s in states.values())
    has_r = ecd.shape[0] == n_states and np.any(ecd[:, 1] != 0) if n_states else False
    for s in states.values():
        for d in s.values():
            order = d.pop('_order')
            if has_r:
                d['R'] = float(ecd[order, 1])
    return states
