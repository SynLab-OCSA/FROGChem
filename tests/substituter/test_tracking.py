"""Original-atom tracking through fragment attachment (frog-substitute --track)."""

import numpy as np
from ase import Atoms

from frogchem.substituter.builder import attach_fragment, track_atoms


def _water():
    return Atoms('OH2', positions=[[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])


def _methyl():
    # C attached at index 0, H at index 1 is the dummy that will be removed
    return Atoms('CH4', positions=[[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89],
                                   [-0.36, -0.51, -0.89]])


def test_track_atoms_marks_only_requested_indices():
    mol = _water()
    track_atoms(mol, [0, 2])
    assert mol.get_array('original_id').tolist() == [0, -1, 2]


def test_attached_atoms_are_untracked():
    mol = _water()
    mol.set_array('orig_idx', np.arange(len(mol)))
    track_atoms(mol, [0])

    # Replace H1 (anchor O0) with a methyl group
    new, added, _ = attach_fragment(mol, _methyl(), (1, 0), (1, 0), n_rotations=4,
                                    clash_handling="best_effort")

    assert len(added) == 4
    orig = new.get_array('orig_idx')
    tracked = new.get_array('original_id')
    # Remaining core atoms keep their original indices, new atoms are -1 (not 0)
    assert orig.tolist() == [0, 2, -1, -1, -1, -1]
    assert tracked.tolist() == [0, -1, -1, -1, -1, -1]
