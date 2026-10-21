"""Every CLI command documents why / what you need / how / what to expect, and TD-DFT states
are labelled consistently (S1, T1, D1 ...)."""
import pytest

from frogchem.cli.base import Guide, load_all_commands
from frogchem.cli.tddft_parser import (parse_dct_state_selection, prepare_soc_table,
                                     prepare_states_table, state_label)

COMMANDS = load_all_commands()


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_has_a_complete_guide(name):
    guide = COMMANDS[name].guide
    assert isinstance(guide, Guide), f"{name}: missing `guide = Guide(...)`"
    for field in ("why", "needs", "expect"):
        assert len(getattr(guide, field).split()) >= 15, f"{name}: Guide.{field} is too short"
    assert len(guide.how) >= 2, f"{name}: Guide.how needs at least two steps"
    for step in guide.how:
        assert "\n\n" not in step, f"{name}: a Guide.how step must be a single paragraph"


def test_guide_rejects_empty_fields():
    with pytest.raises(ValueError):
        Guide(why="", needs="x", how=["a"], expect="y")
    with pytest.raises(ValueError):
        Guide(why="x", needs="x", how=[], expect="y")


def test_state_labels_per_multiplicity():
    assert [state_label(1, 1), state_label(2, 3), state_label(3, 1), state_label(4, 2)] == \
        ["S1", "D3", "T1", "Q2"]
    assert state_label(5, 1) == "M5-1"


def test_states_table_uses_same_labels_as_soc_and_dct():
    states = {1: {1: {"energy": 2.78, "fosc": 0.08, "population": []}},
              3: {1: {"energy": 2.51, "fosc": 0.0, "population": []}}}
    rows = prepare_states_table(states, threshold=0.1)
    assert [r[0] for r in rows] == ["S1", "T1"]
    assert [r[1] for r in rows] == [1, 3]
    import numpy as np
    headers, soc_rows = prepare_soc_table(states, np.array([[6.857], [1.040]]))
    assert headers[1:] == ["T1"] and [r[0] for r in soc_rows] == ["S0", "S1"]
    # every label printed in the table is accepted back by --dct-states
    sel = parse_dct_state_selection(["S1", "T1", "D2", "Q1", "S1-S3"])
    assert sel[1] == {1, 2, 3} and sel[3] == {1} and sel[2] == {2} and sel[4] == {1}
