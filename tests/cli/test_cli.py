import pytest

from frogchem.cli.base import dispatch


def test_dispatch_unknown_exits():
    with pytest.raises(SystemExit) as exc_info:
        dispatch('nonexistent-tool')
    assert exc_info.value.code == 1


def test_dispatch_none_exits():
    with pytest.raises(SystemExit) as exc_info:
        dispatch(None)
    assert exc_info.value.code == 0
