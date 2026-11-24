"""Multicube cube files, orbital grids on the matching geometry, and the ChimeraX grid script."""
import json
from pathlib import Path

import numpy as np
import pytest

from frogchem.io import parse_gaussian_cube, write_gaussian_cube

ROOT = Path(__file__).resolve().parents[2]
TDDFT_ORCA = ROOT / "tests" / "data" / "gaussian_ref" / "orca" / "g05_methyloxirane_td.out"


def _cube(data, ids=None):
    nx, ny, nz = data.shape[:3]
    return dict(title="t", comment="c", natoms=2, origin=np.zeros(3), grid_size=(nx, ny, nz),
                voxel_vectors=np.eye(3) * 0.5, units="Bohr",
                atoms=[{"atomic_number": 8, "charge": 8.0, "coords": np.zeros(3)},
                       {"atomic_number": 1, "charge": 1.0, "coords": np.array([0.0, 0.0, 1.8])}],
                data=data, orbital_ids=ids)


class TestMulticubeIO:
    def test_round_trip_keeps_ids_and_values(self, tmp_path):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(4, 5, 7, 12))
        ids = list(range(40, 52))  # 12 ids: the record wraps over two lines
        path = tmp_path / "multi.cube"
        write_gaussian_cube(str(path), _cube(data, ids))
        back = parse_gaussian_cube(str(path))
        assert back["orbital_ids"] == ids
        assert back["natoms"] == -2  # negative natoms announces the id record
        assert back["data"].shape == (4, 5, 7, 12)
        np.testing.assert_allclose(back["data"], data, atol=1e-5)  # %13.5E precision


    def test_single_cube_has_no_id_record(self, tmp_path):
        data = np.arange(24, dtype=float).reshape(2, 3, 4)
        path = tmp_path / "single.cube"
        write_gaussian_cube(str(path), _cube(data, None))
        back = parse_gaussian_cube(str(path))
        assert back["natoms"] == 2
        assert back["orbital_ids"] is None
        assert back["data"].shape == (2, 3, 4)

    def test_values_per_line_is_six(self, tmp_path):
        data = np.ones((2, 2, 2, 2))
        path = tmp_path / "six.cube"
        write_gaussian_cube(str(path), _cube(data, [1, 2]))
        lines = path.read_text().splitlines()
        # header 6 lines + 2 atoms + id line; values start after
        value_lines = lines[6 + 2 + 1:]
        assert all(len(line.split()) <= 6 for line in value_lines)


class TestOrbitalGrid:
    @pytest.fixture(scope="class")
    def calc(self):
        if not TDDFT_ORCA.exists():
            pytest.skip("methyloxirane TD-DFT sample not available")
        from frogchem.filetypes import load_calculation
        return load_calculation(str(TDDFT_ORCA))

    def test_multicube_components_match_single_orbital_cubes(self, calc, tmp_path):
        from frogchem.density_builder import build_orbitals_cube
        multi = tmp_path / "multi.cube"
        single = tmp_path / "single.cube"
        build_orbitals_cube(calc, [14, 15, 16], str(multi), n_voxels=(12, 10, 8), padding=3.0)
        build_orbitals_cube(calc, [15], str(single), n_voxels=(12, 10, 8), padding=3.0)
        m = parse_gaussian_cube(str(multi))
        s = parse_gaussian_cube(str(single))
        assert m["orbital_ids"] == [15, 16, 17]  # 1-based cube ids
        np.testing.assert_allclose(m["data"][..., 1], s["data"], atol=1e-5)

    def test_phase_is_fixed_by_largest_lobe(self):
        from frogchem.density_builder import fix_orbital_phase
        psi = np.array([[1.0, -2.0], [-0.5, 1.0], [0.2, -1.0]])
        fixed = fix_orbital_phase(psi)
        assert np.sum(fixed ** 3, axis=0).min() > 0
        # sign flip only, magnitudes unchanged
        np.testing.assert_allclose(np.abs(fixed), np.abs(psi))

    def test_per_axis_voxels_give_that_grid(self, calc):
        from frogchem.density_builder import generate_grid
        geom = calc.geom * 1.8897259886
        _, _, shape = generate_grid(geom, n_voxels=(7, 5, 9))
        assert shape == (7, 5, 9)
        _, _, shape = generate_grid(geom, n_voxels=6)
        assert shape == (6, 6, 6)

    def test_index_tokens_and_frontier_span(self):
        from frogchem.cli.tddft_parser import _index_tokens
        from frogchem.cli.electronic_density import ElectronicDensity
        assert _index_tokens(["14-16", "20", "18,19"]) == [14, 15, 16, 18, 19, 20]
        assert ElectronicDensity._frontier_span(15, 16, 1) == [14, 15, 16, 17]
        assert ElectronicDensity._frontier_span(3, 4, 5) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


class TestGridScript:
    def test_manifest_and_viewer_are_written(self, tmp_path):
        from frogchem.chimerax_viewer import write_grid_script, panel, group
        manifest = {"title": "x", "geometry": "m.cube", "columns": 2, "turn": 25.0, "transparency": 0.25,
                    "global_style": True, "caption": ["cap \"quoted\" line"],
                    "groups": [group("Occupied", [panel("m.cube", "H (cube 16)", 0.02, 0)]),
                               group("Densities", [panel("d.cube", "S1", 0.002)])]}
        path = write_grid_script(tmp_path / "x_grid.py", manifest)
        text = path.read_text(encoding="utf-8")
        assert "MANIFEST = _json.loads(" in text
        assert "def run_grid(session, manifest):" in text
        assert text.rstrip().endswith("run_grid(session, MANIFEST)  # noqa: F821  (session is defined by ChimeraX)")
        # the manifest literal decodes back to the same data
        literal = text.split("MANIFEST = _json.loads(", 1)[1].split("\n", 1)[0].rstrip()
        decoded = json.loads(json.loads(literal[:-1]))
        assert decoded == json.loads(json.dumps(manifest))

    def test_viewer_source_is_valid_python(self):
        import ast
        from frogchem.chimerax_viewer import VIEWER_SOURCE
        ast.parse(VIEWER_SOURCE)


class TestTddftOrbitalGridCLI:
    def test_orbital_grid_writes_multicube_and_script(self, tmp_path, monkeypatch):
        if not TDDFT_ORCA.exists():
            pytest.skip("methyloxirane TD-DFT sample not available")
        from frogchem.cli.tddft_parser import TddftParser
        src = tmp_path / "tddft.out"
        src.write_text(TDDFT_ORCA.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        TddftParser().main([str(src), "--orbital-grid", "--dct", "--voxels", "10", "8", "6",
                            "--dct-states", "S1", "--grid-frontier", "0"])
        assert (tmp_path / "tddft_orbitals.cube").exists()
        assert (tmp_path / "tddft_S1_delta_rho.cube").exists()
        assert (tmp_path / "tddft_grid.py").exists()
        multi = parse_gaussian_cube(str(tmp_path / "tddft_orbitals.cube"))
        assert multi["orbital_ids"] is not None and len(multi["orbital_ids"]) >= 2
        script = (tmp_path / "tddft_grid.py").read_text(encoding="utf-8")
        assert "tddft_S1_delta_rho.cube" in script


class TestGridOptions:
    @pytest.mark.parametrize("argv", [
        ["x.out", "--spacing", "0.2", "--voxels", "10"],
        ["x.out", "--voxels", "10", "--spacing", "0.2"],
    ])
    def test_spacing_and_voxels_are_exclusive(self, argv):
        from frogchem.cli.electronic_density import ElectronicDensity
        with pytest.raises(SystemExit):
            ElectronicDensity().build_parser().parse_args(argv)

    def test_spacing_or_voxels_alone(self):
        from frogchem.cli.electronic_density import ElectronicDensity
        parser = ElectronicDensity().build_parser()
        assert parser.parse_args(["x.out", "--voxels", "20", "16", "12"]).voxels == (20, 16, 12)
        args = parser.parse_args(["x.out", "--spacing", "0.2"])
        assert args.spacing == 0.2 and args.voxels is None  # explicit spacing drops the default voxels
        assert parser.parse_args(["x.out"]).voxels == 80

    def test_beta_on_closed_shell_is_an_error(self, tmp_path):
        if not TDDFT_ORCA.exists():
            pytest.skip("methyloxirane TD-DFT sample not available")
        from frogchem.cli.electronic_density import ElectronicDensity
        with pytest.raises(SystemExit) as exc:
            ElectronicDensity().main([str(TDDFT_ORCA), "--homo", "--beta", "--voxels", "5",
                                      "--out-dir", str(tmp_path)])
        assert exc.value.code == 1
        assert not list(tmp_path.glob("*.cube"))


class TestGridConfirmation:
    def _args(self, yes=False):
        from types import SimpleNamespace
        return SimpleNamespace(yes=yes)

    def test_no_terminal_keeps_selection(self, monkeypatch):
        from frogchem.cli import tddft_parser as tp
        monkeypatch.setattr(tp.sys.stdin, "isatty", lambda: False, raising=False)
        assert tp._confirm_grid_orbitals([14, 15], 30, self._args()) == [14, 15]

    def test_answers_in_a_terminal(self, monkeypatch):
        from frogchem.cli import tddft_parser as tp

        class TTY:
            def isatty(self):
                return True
        monkeypatch.setattr(tp.sys, "stdin", TTY())
        monkeypatch.setattr(tp.sys, "stderr", type("E", (), {"isatty": lambda s: True, "write": lambda s, x: None,
                                                            "flush": lambda s: None})())
        answers = iter(["99", "14-16 20", ""])  # out of range, then a new selection, then Enter
        monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
        assert tp._confirm_grid_orbitals([15, 16], 30, self._args()) == [14, 15, 16, 20]
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        assert tp._confirm_grid_orbitals([15, 16], 30, self._args()) is None
        assert tp._confirm_grid_orbitals([15, 16], 30, self._args(yes=True)) == [15, 16]
