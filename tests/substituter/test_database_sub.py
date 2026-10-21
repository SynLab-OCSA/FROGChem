"""Regression tests for FragmentDatabase add/rename/delete bookkeeping."""

import pytest

from frogchem.io import write_xyz
from frogchem.substituter.database_sub import FragmentDatabase


@pytest.fixture
def db(tmp_path):
    return FragmentDatabase(base_dir=tmp_path / "db", auto_init_standard=False)


@pytest.fixture
def h2_xyz(tmp_path):
    path = tmp_path / "h2.xyz"
    write_xyz(str(path), [["H", "H"]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]])
    return path


def test_add_with_existing_name_and_missing_structure(db, h2_xyz):
    db.add_fragment("frag", str(h2_xyz), delete_idx=1, attach_idx=0)
    (db.structures_dir / "frag.xyz").unlink()
    # The stored XYZ is gone: must not crash in samefile, picks a new name
    name = db.add_fragment("frag", str(h2_xyz), delete_idx=1, attach_idx=0)
    assert name == "frag_1"
    assert (db.structures_dir / "frag_1.xyz").exists()


def test_add_same_file_again_keeps_name(db, h2_xyz):
    db.add_fragment("frag", str(h2_xyz), delete_idx=1, attach_idx=0)
    stored = db.structures_dir / "frag.xyz"
    assert db.add_fragment("frag", str(stored), delete_idx=1, attach_idx=0) == "frag"


def test_rename_moves_structure_file(db, h2_xyz):
    db.add_fragment("old", str(h2_xyz), delete_idx=1, attach_idx=0)
    db.update_fragment("old", new_name="new")

    assert db.data["new"]["filename"] == "new.xyz"
    assert (db.structures_dir / "new.xyz").exists()
    assert not (db.structures_dir / "old.xyz").exists()
    path, pair = db.get_fragment("new")
    assert path.endswith("new.xyz") and pair == (1, 0)

    # Re-adding a fragment called 'old' must not collide with the renamed one
    assert db.add_fragment("old", str(h2_xyz), delete_idx=1, attach_idx=0) == "old"
    assert FragmentDatabase(base_dir=db.base_dir, auto_init_standard=False).data["new"]["filename"] == "new.xyz"


def test_delete_unknown_fragment_raises(db):
    with pytest.raises(ValueError):
        db.delete_fragment("does-not-exist")
