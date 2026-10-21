import re
import numpy as np


def get_freq(fl, reg):
    if reg['freq_section'] not in fl:
        return np.array([])

    block = fl.split(reg['freq_section'])[-1].split(reg['freq_end'])[0]
    freq = np.array([float(x) for m in re.findall(reg['freq_pattern'], block) for x in m.split()])
    return freq


def get_ir_intensities(fl, reg):
    if reg['freq_section'] not in fl:
        return np.zeros(shape=(1, 2))

    block = fl.split(reg['freq_section'])[-1].split(reg['freq_end'])[0]
    freq = np.array([float(x) for m in re.findall(reg['freq_pattern'], block) for x in m.split()])
    ir = np.array([float(x) for m in re.findall(reg['ir_pattern'], block) for x in m.split()])
    if len(freq) == 0 or len(ir) == 0:
        return np.zeros(shape=(1, 2))
    return np.column_stack((freq, ir))
