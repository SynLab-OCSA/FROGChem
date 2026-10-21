import re
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class Transition(tuple):
    """3-tuple (orb_from, orb_to, weight) with optional .spin attribute ('a' or 'b')."""
    def __new__(cls, orb_from: int, orb_to: int, weight: float, spin: str = ''):
        obj = super().__new__(cls, (orb_from, orb_to, weight))
        obj.spin = spin
        return obj

    def __getitem__(self, item):
        if item == 3:
            return self.spin
        return super().__getitem__(item)


def parse_tddft_states(fl, reg) -> dict:
    states = defaultdict(lambda: defaultdict(dict))

    singlets_re = re.compile(r'TD-DFT(?:/TDA)? EXCITED STATES \(SINGLETS\)')
    triplets_re = re.compile(r'TD-DFT(?:/TDA)? EXCITED STATES \(TRIPLETS\)')
    open_shell_re = re.compile(r'TD-DFT(?:/TDA)? EXCITED STATES(?:\s*\n|\r\n)')
    spectra_re = re.compile(r'TD-DFT(?:/TDA)?-EXCITATION SPECTRA')
    soc_re = re.compile(r'TD-DFT(?:/TDA)? SPIN-ORBIT COUPLING')

    s_match = singlets_re.search(fl)
    t_match = triplets_re.search(fl)
    o_match = open_shell_re.search(fl)
    spec_match = spectra_re.search(fl)
    soc_match = soc_re.search(fl)

    singlets_part = ""
    if s_match:
        s_start = s_match.end()
        if t_match and t_match.start() > s_start:
            s_end = t_match.start()
        elif spec_match and spec_match.start() > s_start:
            s_end = spec_match.start()
        else:
            s_end = len(fl)
        singlets_part = fl[s_start:s_end]

    triplets_part = ""
    if t_match:
        t_start = t_match.end()
        if soc_match and soc_match.start() > t_start:
            t_end = soc_match.start()
        elif spec_match and spec_match.start() > t_start:
            t_end = spec_match.start()
        else:
            t_end = len(fl)
        triplets_part = fl[t_start:t_end]
        
    open_shell_part = ""
    if o_match and not s_match and not t_match:
        o_start = o_match.end()
        if spec_match and spec_match.start() > o_start:
            o_end = spec_match.start()
        else:
            o_end = len(fl)
        open_shell_part = fl[o_start:o_end]

    f_ = singlets_part + "\n" + triplets_part + "\n" + open_shell_part

    state_pattern = re.compile(reg['state_pattern'])
    transition_pattern = re.compile(reg['transition_pattern'])

    for match in state_pattern.finditer(f_):
        state_num = int(match.group(1))
        energy = float(match.group(2))
        s2 = float(match.group(3))
        multiplicity = int(match.group(4))

        states[multiplicity][state_num] = {
            "energy": energy,
            "S**2": s2,
            "population": [],
        }

    current_state = None
    current_mult = None

    for line in f_.splitlines():
        trans_match = transition_pattern.search(line)
        if trans_match and current_state is not None:
            orb_from_raw, orb_to_raw, weight = trans_match.groups()
            spin = ''
            if open_shell_part or 'b' in f_.lower():
                if 'b' in orb_from_raw.lower():
                    spin = 'b'
                elif 'a' in orb_from_raw.lower():
                    spin = 'a'

            try:
                orb_from = int(re.sub(r'\D', '', orb_from_raw))
                orb_to = int(re.sub(r'\D', '', orb_to_raw))
            except (ValueError, AttributeError) as e:
                logger.warning("Could not parse orbital indices: %s", e)

            states[current_mult][current_state]["population"].append(
                Transition(orb_from, orb_to, float(weight), spin)
            )
        else:
            state_match = state_pattern.search(line)
            if state_match:
                current_state = int(state_match.group(1))
                current_mult = int(state_match.group(4))

    return states


def get_tddft_soc(fl, reg, tddft_states):
    soc_start = reg.get('soc_start', 'CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS')
    if soc_start not in fl:
        return np.zeros((0, 0))

    f_ = fl.split(soc_start)[-1]
    soc_end = reg.get('soc_end', 'SOC stabilization')
    if soc_end in f_:
        f_ = f_.split(soc_end)[0]

    pattern = re.compile(reg['pattern_soc'])
    matches = list(pattern.finditer(f_))
    if not matches:
        return np.zeros((0, 0))

    max_t = max(int(m.group(1)) for m in matches)
    max_s = max(int(m.group(2)) for m in matches)
    num_s = max(max_s + 1, len(tddft_states.get(1, {})) + 1 if tddft_states else 0)
    num_t = max(max_t, len(tddft_states.get(3, {})) if tddft_states else 0)

    results = np.zeros((num_s, num_t))
    for match in matches:
        triplet_state = int(match.group(1))
        singlet_state = int(match.group(2))

        re_z, im_z = float(match.group(3)), float(match.group(4))
        re_x, im_x = float(match.group(5)), float(match.group(6))
        re_y, im_y = float(match.group(7)), float(match.group(8))

        # True Euclidean norm of the (X, Y, Z) complex Cartesian components:
        norm = np.sqrt(re_x**2 + im_x**2 + re_y**2 + im_y**2 + re_z**2 + im_z**2)
        results[singlet_state][triplet_state - 1] = float(norm)

    return results


def parse_tddft_fosc(fl, tddft_states):
    # Match the uncorrected TD-DFT absorption spectrum (prefer non-SOC CORRECTED)
    abs_match = re.search(
        r"(?<!SOC CORRECTED )ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        fl
    )
    if not abs_match:
        abs_match = re.search(
            r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
            fl
        )

    if abs_match:
        spectrum_section = fl[abs_match.end():].split('\n\n')[0]
        fosc_pattern = re.compile(
            r"\s*(\d+)-(\d+(?:\.\d+)?)A\s*->\s*(\d+)-(\d+(?:\.\d+)?)A\s+([\d\.]+)\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)"
        )
        for match in fosc_pattern.finditer(spectrum_section):
            triplet_state, triplet_mult, singlet_state, singlet_mult, energy, fosc = match.groups()
            s_state = int(singlet_state)
            s_mult = int(float(singlet_mult))
            if s_mult in tddft_states and s_state in tddft_states[s_mult]:
                tddft_states[s_mult][s_state]["fosc"] = float(fosc)

    # Match the uncorrected TD-DFT CD spectrum (prefer non-SOC CORRECTED)
    cd_match = re.search(
        r"(?<!SOC CORRECTED )CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
        fl
    )
    if not cd_match:
        cd_match = re.search(
            r"CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS",
            fl
        )

    if cd_match:
        spectrum_section = fl[cd_match.end():].split('\n\n')[0]
        cd_pattern = re.compile(
            r"\s*(\d+)-(\d+(?:\.\d+)?)A\s*->\s*(\d+)-(\d+(?:\.\d+)?)A\s+([\d\.\-]+)\s+[\d\.\-]+\s+[\d\.\-]+\s+([\d\.\-]+)"
        )
        for match in cd_pattern.finditer(spectrum_section):
            triplet_state, triplet_mult, singlet_state, singlet_mult, energy, R = match.groups()
            s_state = int(singlet_state)
            s_mult = int(float(singlet_mult))
            if s_mult in tddft_states and s_state in tddft_states[s_mult]:
                tddft_states[s_mult][s_state]["R"] = float(R)


