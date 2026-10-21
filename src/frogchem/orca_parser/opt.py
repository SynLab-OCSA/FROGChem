import numpy as np
import re


def get_geom(fl, reg):
    pattern = reg['start_geom']
    xyz = re.findall(pattern, fl)[-1].strip().splitlines()[1:]
    as_, gs_ = [], []

    for j in xyz:
        a, *g = j.split()
        as_.append(a)
        gs_.append(g)
    return np.array(as_), np.array(gs_, dtype=float)


def get_energy_decomposition(fl, reg):
    d = {}
    fl_ = re.findall(reg['SCF_energy'], fl)[-1]

    for i in re.finditer(reg['decomp_energy'], fl_):
        k, v = i.groups()
        d[k.strip()] = float(v)

    fl_ = re.findall(reg['dispersion'], fl)
    if fl_:
        k, v = re.findall(reg['disp_energy'], fl_[-1])[-1]
        d[k.strip()] = float(v)
        d[reg['total_energy']] += float(v)

    return d
    