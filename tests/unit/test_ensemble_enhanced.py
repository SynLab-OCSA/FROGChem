import json
import tempfile
from pathlib import Path
import pytest

from frogchem.ensemble import extract_conf_and_protocol, load_checkpoint_populations, build_ensemble


def test_extract_enan_folder_structure():
    # Standard EnAn paths: .../conf_<ID>/protocol_<X>/*.(out|log)
    c_id, p_id = extract_conf_and_protocol("/home/user/project/conf_1/protocol_1/orca.out")
    assert c_id == 1
    assert p_id == 1

    c_id, p_id = extract_conf_and_protocol("results/conf_42/protocol_3/calc.log")
    assert c_id == 42
    assert p_id == 3

    # Mixed casing and hyphen
    c_id, p_id = extract_conf_and_protocol("test/CONF-5/Protocol_2/output.out")
    assert c_id == 5
    assert p_id == 2


def test_extract_fallback_patterns():
    # Filename pattern fallback
    c_id, p_id = extract_conf_and_protocol("conf_2_p1.out")
    assert c_id == 2
    assert p_id == 1

    c_id, p_id = extract_conf_and_protocol("1_p2_orca.out")
    assert c_id == 1
    assert p_id == 2

    # Frame label fallback
    c_id, p_id = extract_conf_and_protocol("trajectory.xyz - Conf 0")
    assert c_id == 1
    assert p_id is None

    c_id, p_id = extract_conf_and_protocol("trajectory.xyz - Conf 4")
    assert c_id == 5
    assert p_id is None


def test_load_checkpoint_enan_nested_format():
    data = {
        "conformers": {
            "1": {
                "number": 1,
                "energies": {
                    "1": {"Pop": 20.0},
                    "2": {"Pop": 80.0}
                }
            },
            "2": {
                "number": 2,
                "energies": {
                    "1": {"Pop": 80.0},
                    "2": {"Pop": 20.0}
                }
            }
        }
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(data, tmp)
        p = Path(tmp.name)

    try:
        # Default loads latest protocol (2)
        pops = load_checkpoint_populations(p)
        assert pops[1] == 80.0
        assert pops[2] == 20.0

        # Protocol 1 explicitly
        pops_p1 = load_checkpoint_populations(p, protocol=1)
        assert pops_p1[1] == 20.0
        assert pops_p1[2] == 80.0
    finally:
        if p.exists():
            p.unlink()


def test_build_ensemble_weights():
    data = {
        "1": {"energies": {"1": {"Pop": 10.0}}},
        "2": {"energies": {"1": {"Pop": 30.0}}},
    }
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tmp:
        json.dump(data, tmp)
        ckpt_path = Path(tmp.name)

    try:
        files = [
            "experiment/conf_1/protocol_1/orca.out",
            "experiment/conf_2/protocol_1/orca.out",
        ]
        ens = build_ensemble(files, ckpt_path, protocol=1)
        assert len(ens) == 2
        assert ens.entries[0].conf_id == 1
        assert ens.entries[0].proto_id == 1
        assert ens.entries[0].population == 10.0
        assert pytest.approx(ens.entries[0].weight) == 0.25

        assert ens.entries[1].conf_id == 2
        assert ens.entries[1].proto_id == 1
        assert ens.entries[1].population == 30.0
        assert pytest.approx(ens.entries[1].weight) == 0.75
    finally:
        if ckpt_path.exists():
            ckpt_path.unlink()
