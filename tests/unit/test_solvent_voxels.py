"""Solvent surfaces must use the real grid spacing when n_voxels overrides `spacing`."""

import re

import numpy as np

from frogchem.solvent_surface import _get_vdw_radius, build_sas_cube, build_ses_cube, compute_ses

ATOMS = ['C']
COORDS = np.zeros((1, 3))


def _volume(capsys):
    captured = capsys.readouterr()            # progress messages go to stderr (print_info)
    return float(re.search(r'volume: ([\d.]+)', captured.out + captured.err).group(1))


def test_ses_single_atom_with_voxels():
    # For one atom the SES is its vdW sphere: phi at the centre is -r_vdw
    r = _get_vdw_radius('C')
    phi, axes, _ = compute_ses(ATOMS, COORDS, spacing=0.1, padding=1.0, n_voxels=41)
    h = axes[0][1] - axes[0][0]
    assert h > 0.15  # n_voxels really overrides spacing
    centre = tuple(int(np.argmin(np.abs(a))) for a in axes)
    assert abs(phi[centre] + r) < 1.5 * h


def test_printed_volumes_use_actual_spacing(tmp_path, capsys, monkeypatch):
    from frogchem import printer
    monkeypatch.setattr(printer, "_QUIET", False)    # other CLI tests may have run with -q
    r = _get_vdw_radius('C')
    for builder, radius in ((build_sas_cube, r + 1.4), (build_ses_cube, r)):
        builder(ATOMS, COORDS, output=str(tmp_path / 'x.cube'), spacing=0.1, padding=1.0, n_voxels=61)
        expected = 4.0 / 3.0 * np.pi * radius ** 3
        assert abs(_volume(capsys) - expected) / expected < 0.1
