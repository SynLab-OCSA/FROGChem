import os
import json
from pathlib import Path
import pytest
import numpy as np

from frogchem.cli.calc_er import CalcEr
from frogchem.cli.split_xyz import SplitXyz
from frogchem.cli.mirror_ensemble import MirrorEnsemble
from frogchem.cli.create_crest_input import CreateCrestInput
from frogchem.cli.sort_diastero import SortDiastero
from frogchem.cli.detect_symmetry import DetectSymmetry
from frogchem.cli.internals import Internals
from frogchem.cli.calc_rmsd import CalcRmsd
from frogchem.cli.cube_manipulation import CubeManipulation
from frogchem.cli.manage_sub_db import ManageSubDb
from frogchem.cli.nics_geometry import NicsGeometry
from frogchem.cli.calc_loc_spin import parse_index_string
from frogchem.cli.calc_rrho import CalcRrho
from frogchem.cli.get_charges import GetCharges
from frogchem.cli.write_xyz import WriteXYZ
from frogchem.cli.fragment_detection import FragmentDetection
from frogchem.cli.solvent_surface import SolventSurface
from frogchem.cli.stereocenters import Stereocenters
from frogchem.cli.tddft_parser import TddftParser
from frogchem.cli.plot_neb import PlotNeb
from frogchem.io import write_xyz, read_xyz, write_gaussian_cube


@pytest.fixture
def sample_water_xyz(tmp_path):
    f = tmp_path / "water.xyz"
    atoms = [["O", "H", "H"]]
    coords = [np.array([
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469]
    ])]
    write_xyz(str(f), atoms, coords, ["Water molecule"])
    return str(f)


@pytest.fixture
def sample_ensemble_xyz(tmp_path):
    f = tmp_path / "ensemble.xyz"
    atoms = [["C", "C", "H", "H"], ["C", "C", "H", "H"]]
    coords = [
        np.array([[0, 0, 0], [1.5, 0, 0], [0, 1, 0], [1.5, 1, 0]], dtype=float),
        np.array([[0, 0, 0], [1.5, 0, 0], [0, 1, 0], [1.5, -1, 0]], dtype=float),
    ]
    write_xyz(str(f), atoms, coords, ["Conf 1", "Conf 2"])
    return str(f)


class TestCLIMissingFilesValidation:
    @pytest.mark.parametrize("cmd_class,args", [
        (CalcRmsd, ["nonexistent.xyz"]),
        (SplitXyz, ["nonexistent.xyz"]),
        (MirrorEnsemble, ["nonexistent.xyz"]),
        (CreateCrestInput, ["nonexistent.xyz", "-i", "0"]),
        (DetectSymmetry, ["nonexistent.xyz"]),
        (Internals, ["nonexistent.xyz", "-i", "0", "1"]),
    ])
    def test_missing_files_exit_code_2(self, cmd_class, args):
        cmd = cmd_class()
        with pytest.raises(SystemExit) as exc:
            cmd.main(args)
        assert exc.value.code in [1, 2]


class TestCalcErCLI:
    def test_calc_er_subcommand(self, capsys):
        cmd = CalcEr()
        cmd.main(["er", "1.42"])
        captured = capsys.readouterr()
        assert "Enantiomeric Ratio" in captured.out or "ee (%)" in captured.out

    def test_calc_er_json(self, capsys):
        cmd = CalcEr()
        cmd.main(["er", "1.42", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert data[0]["rows"][0]["dG (kcal/mol)"] == 1.42
        assert "ee (%)" in data[0]["columns"]

    def test_calc_dg_subcommand(self, capsys):
        cmd = CalcEr()
        cmd.main(["dg", "90", "10"])
        captured = capsys.readouterr()
        assert "ΔG from Ratio" in captured.out or "dG (kcal/mol)" in captured.out


class TestSplitAndMergeXYZCLI:
    def test_split_and_merge(self, sample_ensemble_xyz, tmp_path):
        out_dir = str(tmp_path / "split_out")
        split_cmd = SplitXyz()
        split_cmd.main([sample_ensemble_xyz, "-o", out_dir, "-f"])

        assert os.path.exists(os.path.join(out_dir, "conf_1", "geom.xyz"))
        assert os.path.exists(os.path.join(out_dir, "conf_2", "geom.xyz"))

        # Merge back
        c1 = os.path.join(out_dir, "conf_1", "geom.xyz")
        c2 = os.path.join(out_dir, "conf_2", "geom.xyz")
        merged_file = str(tmp_path / "merged.xyz")
        split_cmd.main(["--merge", c1, c2, "-o", merged_file, "-f"])

        assert os.path.exists(merged_file)
        atoms, coords, _ = read_xyz(merged_file)
        assert len(atoms) == 2


class TestMirrorEnsembleCLI:
    def test_mirror(self, sample_water_xyz, tmp_path):
        cmd = MirrorEnsemble()
        cmd.main([sample_water_xyz])

        mirror_file = sample_water_xyz.replace(".xyz", "_mirror.xyz")
        assert os.path.exists(mirror_file)
        orig_atoms, orig_coords, _ = read_xyz(sample_water_xyz)
        mir_atoms, mir_coords, _ = read_xyz(mirror_file)

        # X coords should be negated
        np.testing.assert_allclose(mir_coords[0][:, 0], -orig_coords[0][:, 0])
        np.testing.assert_allclose(mir_coords[0][:, 1:], orig_coords[0][:, 1:])


class TestCreateCrestInputCLI:
    def test_crest_input(self, sample_water_xyz, tmp_path):
        cmd = CreateCrestInput()
        out_inp = str(tmp_path / "constrain.inp")
        cmd.main([sample_water_xyz, "-i", "0", "1", "-o", out_inp])
        assert os.path.exists(out_inp)

    def test_crest_metal_center_sphere(self, tmp_path):
        f = tmp_path / "mc.xyz"
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.5, 0.0],
                           [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [11.5, 0.0, 0.0]])
        write_xyz(str(f), [["Pd", "P", "N", "C", "Cu", "Cl"]], [coords], ["test"])
        out_inp = tmp_path / "constrain.inp"
        CreateCrestInput().main([str(f), "--mc", "0", "4", "-r", "2.6", "-i", "3", "-o", str(out_inp)])
        # sphere of 0 -> 0 1 2, sphere of 4 -> 4 5, plus -i 3: all atoms, 1-based
        assert "atoms: 1-6\n" in out_inp.read_text()

        CreateCrestInput().main([str(f), "--metal-center", "0", "--radius", "2.1", "-o", str(out_inp)])
        assert "atoms: 1-2\n" in out_inp.read_text()
        assert "$metadyn" not in out_inp.read_text()

    def test_crest_exclude_mtd(self, tmp_path):
        f = tmp_path / "mc.xyz"
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        write_xyz(str(f), [["Pd", "P", "C", "C"]], [coords], ["test"])
        out_inp = tmp_path / "constrain.inp"
        CreateCrestInput().main([str(f), "--mc", "0", "-r", "2.1", "--exclude-mtd", "-o", str(out_inp)])
        text = out_inp.read_text()
        assert "$constrain\n  atoms: 1-2\n" in text
        assert "$metadyn\n  atoms: 3-4\n$end\n" in text

        with pytest.raises(SystemExit):
            CreateCrestInput().main([str(f), "-i", "0", "1", "2", "3", "--exclude-mtd", "-o", str(out_inp)])

    @pytest.mark.parametrize("extra", [["--mc", "0"], ["-r", "2.0"], []])
    def test_crest_metal_center_requires_radius(self, sample_water_xyz, tmp_path, extra):
        with pytest.raises(SystemExit):
            CreateCrestInput().main([sample_water_xyz, *extra, "-o", str(tmp_path / "c.inp")])


class TestSortDiasteroCLI:
    def test_sort_diastero(self, sample_ensemble_xyz, tmp_path):
        cmd = SortDiastero()
        cmd.main([sample_ensemble_xyz, "-i", "0", "1", "2", "3", "--format", "json"])


class TestDetectSymmetryCLI:
    def test_detect_symmetry_json(self, sample_water_xyz, capsys):
        cmd = DetectSymmetry()
        cmd.main([sample_water_xyz, "--include-h", "--format", "json"])
        captured = capsys.readouterr()
        assert "Point Group" in captured.out
        assert "C2v" in captured.out


class TestInternalsCLI:
    def test_internals_distance_and_angle(self, sample_water_xyz, capsys):
        cmd = Internals()
        cmd.main([sample_water_xyz, "-i", "0", "1", "-i", "1", "0", "2", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) >= 1


class TestCalcRmsdCLI:
    def test_calc_rmsd(self, sample_ensemble_xyz, capsys):
        cmd = CalcRmsd()
        cmd.main([sample_ensemble_xyz, "-r", sample_ensemble_xyz, "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data[0]["rows"]) == 2


class TestCubeManipulationCLI:
    @pytest.fixture
    def sample_cube_files(self, tmp_path):
        c1 = str(tmp_path / "c1.cube")
        c2 = str(tmp_path / "c2.cube")
        grid1 = {
            'title': 'Cube 1', 'comment': '', 'natoms': 1,
            'origin': np.zeros(3), 'grid_size': (2, 2, 2),
            'voxel_vectors': np.eye(3), 'units': 'Bohr',
            'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.zeros(3)}],
            'data': np.ones((2, 2, 2)) * 3.0,
        }
        grid2 = {
            'title': 'Cube 2', 'comment': '', 'natoms': 1,
            'origin': np.zeros(3), 'grid_size': (2, 2, 2),
            'voxel_vectors': np.eye(3), 'units': 'Bohr',
            'atoms': [{'atomic_number': 1, 'charge': 0.0, 'coords': np.zeros(3)}],
            'data': np.ones((2, 2, 2)) * 2.0,
        }
        write_gaussian_cube(c1, grid1)
        write_gaussian_cube(c2, grid2)
        return c1, c2

    def test_add_cubes(self, sample_cube_files, tmp_path):
        c1, c2 = sample_cube_files
        out_cube = str(tmp_path / "sum.cube")
        cmd = CubeManipulation()
        cmd.main(["-a", c1, c2, "-o", out_cube])
        assert os.path.exists(out_cube)

    def test_multiply_cube(self, sample_cube_files, tmp_path):
        c1, _ = sample_cube_files
        out_cube = str(tmp_path / "mult.cube")
        cmd = CubeManipulation()
        cmd.main(["-m", c1, "2.5", "-o", out_cube])
        assert os.path.exists(out_cube)

    def test_subtract_cube(self, sample_cube_files, tmp_path):
        c1, c2 = sample_cube_files
        out_cube = str(tmp_path / "diff.cube")
        cmd = CubeManipulation()
        cmd.main(["-s", c2, c1, "-o", out_cube])
        assert os.path.exists(out_cube)


class TestNicsGeometryCLI:
    def test_nics_geometry(self, sample_water_xyz, capsys):
        cmd = NicsGeometry()
        cmd.main([sample_water_xyz, "-i", "0", "1"])
        captured = capsys.readouterr()
        assert "NewGTO" in captured.out


class TestCalcLocSpinHelpers:
    def test_parse_atom_selection(self):
        # 0-based ranges and singles to sorted list
        res = parse_index_string("1-3, 5")
        assert res == [1, 2, 3, 5]

    def test_calc_loc_spin_interactive_cli(self, monkeypatch, capsys):
        from frogchem.cli.calc_loc_spin import CalcLocSpin
        import InquirerPy.inquirer
        actions = iter(["Add fragment", "Show results"])
        texts = iter(["Frag1", "0-1"])

        class MockPrompt:
            def __init__(self, val):
                self.val = val
            def execute(self):
                return self.val

        monkeypatch.setattr(InquirerPy.inquirer, "select", lambda **kw: MockPrompt(next(actions)))
        monkeypatch.setattr(InquirerPy.inquirer, "text", lambda **kw: MockPrompt(next(texts)))

        if os.path.exists("files/test.cube"):
            cmd = CalcLocSpin()
            cmd.main(["files/test.cube", "--interactive", "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            frag = [t for t in data if t["title"].startswith("Fragment")][0]["rows"]
            assert any(row.get("Fragment") == "Frag1" for row in frag)


class TestManageSubDbCLI:
    def test_db_list_empty(self):
        cmd = ManageSubDb()
        cmd.main(["list"])

    def test_db_crud(self, sample_water_xyz, tmp_path, monkeypatch, capsys):
        # Isolate database in temporary directory
        import frogchem.substituter.database_sub as db_mod
        orig_init = db_mod.FragmentDatabase.__init__

        def mock_init(self):
            self.base_dir = tmp_path
            self.structures_dir = tmp_path / "structures"
            self.db_file = tmp_path / "library.json"
            self.structures_dir.mkdir(parents=True, exist_ok=True)
            self.data = self._load()

        monkeypatch.setattr(db_mod.FragmentDatabase, "__init__", mock_init)

        cmd = ManageSubDb()
        # 1. Add fragment
        cmd.main(["add", "frag1", sample_water_xyz, "--indices", "1", "0", "--desc", "Water fragment"])
        captured = capsys.readouterr()
        assert "Added" in captured.err

        # 2. List fragments
        cmd.main(["--format", "json", "list"])
        captured = capsys.readouterr()
        assert "frag1" in captured.out

        # 3. Edit fragment
        cmd.main(["edit", "frag1", "--desc", "Updated desc", "--rename", "frag1_renamed"])
        captured = capsys.readouterr()

        # 4. Remove fragment
        cmd.main(["remove", "frag1_renamed"])
        captured = capsys.readouterr()


class TestMoreCLITools:
    def test_calc_rrho_cli(self, capsys):
        if os.path.exists("files/test.out"):
            cmd = CalcRrho()
            cmd.main(["files/test.out", "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) >= 1

    def test_get_charges_cli(self, capsys):
        if os.path.exists("files/test.out"):
            cmd = GetCharges()
            cmd.main(["files/test.out", "-i", "0", "1", "--type", "mulliken", "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) >= 1

    def test_write_xyz_cli(self, tmp_path):
        if os.path.exists("files/test.out"):
            cmd = WriteXYZ()
            # Run on files/test.out
            cmd.main(["files/test.out"])
            expected = "files/test_final.xyz"
            assert os.path.exists(expected)
            os.remove(expected)

    def test_fragment_detection_cli(self, sample_water_xyz, capsys):
        cmd = FragmentDetection()
        cmd.main([sample_water_xyz, "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) >= 1

    def test_distance_charge_transfer_cli(self, capsys):
        if os.path.exists("files/test.cube"):
            cmd = TddftParser()
            cmd.main(["files/test.cube", "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) >= 1
            assert "Δr (Å)" in data[0]["columns"]

    def test_distance_charge_transfer_tddft_out(self, capsys, tmp_path):
        tddft_file = "files/tddft_mos.out"
        if os.path.exists(tddft_file):
            cmd = TddftParser()
            out_dir = str(tmp_path / "dct_cubes")
            cmd.main([tddft_file, "--dct", "--voxels", "15", "--out-dir", out_dir, "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert len(data) >= 2
            dct_data = data[1]["rows"]
            assert len(dct_data) == 10
            assert dct_data[0]["State"] == "S1"
            assert "Δr (Å)" in dct_data[0]
            assert "Dipole Moment" in dct_data[0]
            # Verify cube files were saved
            s1_cube = os.path.join(out_dir, "tddft_mos_S1_delta_rho.cube")
            assert os.path.exists(s1_cube)
            from frogchem.io import parse_gaussian_cube
            cube_data = parse_gaussian_cube(s1_cube)
            assert cube_data["grid_size"] == (15, 15, 15)

    def test_solvent_surface_cli(self, sample_water_xyz, tmp_path):
        cmd = SolventSurface()
        sas_out = str(tmp_path / "water_sas.cube")
        ses_out = str(tmp_path / "water_ses.cube")
        cmd.main([sample_water_xyz, "--voxels", "10", "--out-dir", str(tmp_path)])
        assert os.path.exists(sas_out)
        assert os.path.exists(ses_out)

    def test_stereocenters_cli(self, sample_water_xyz, capsys):
        pytest.importorskip("rdkit", reason="rdkit needed for stereocenters")
        cmd = Stereocenters()
        cmd.main([sample_water_xyz])
        captured = capsys.readouterr()
        assert "stereocenters" in captured.out.lower() or "stereocenters" in captured.err.lower()

    def test_tddft_parser_json_cli(self, capsys):
        if os.path.exists("files/tddft.out"):
            cmd = TddftParser()
            cmd.main(["files/tddft.out", "--format", "json"])
            captured = capsys.readouterr()
            data = json.loads(captured.out)  # valid JSON even when the file has no excited states
            assert isinstance(data, list)

    def test_plot_neb_cli(self, tmp_path):
        interp_file = tmp_path / "neb.interp"
        content = """Iteration: 0
------------------------------------------------------------------
0   0.000000   -1.000000
1   1.000000   -0.980000

Interpolated Path
------------------------------------------------------------------
0.000000   0.000000   -1.000000
1.000000   1.000000   -0.980000

"""
        interp_file.write_text(content)
        cmd = PlotNeb()
        plot_png = str(tmp_path / "neb_plot.png")
        cmd.main([str(interp_file), "-o", plot_png])
        assert os.path.exists(plot_png)

    def test_electronic_density_orbital_cli(self, tmp_path):
        from frogchem.cli.electronic_density import ElectronicDensity
        if os.path.exists("files/orb.out"):
            input_path = os.path.abspath("files/orb.out")
            cmd = ElectronicDensity()
            cmd.main([input_path, "--orbital", "0", "--voxels", "3", "--out-dir", str(tmp_path)])
            assert (tmp_path / "orb_orbital_0.cube").exists()

    def test_hessian_localizer_cli(self, capsys):
        from frogchem.cli.hessian_localizer import HessianLocalizer, parse_fragments
        # Test helper
        frags = parse_fragments(["CO2:0,1", "invalid_format"])
        assert frags == {"CO2": [0, 1]}

        if os.path.exists("files/test.out"):
            cmd = HessianLocalizer()
            cmd.main(["files/test.out", "-m", "6", "--ipr", "--fragment", "F1:0,1", "--format", "json"])
            captured = capsys.readouterr()
            assert "Mode 6" in captured.out
            assert "IPR" in captured.out

    def test_reaction_profile_cli(self, tmp_path, monkeypatch):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        from frogchem.cli.reaction_profile import ReactionProfile
        prof_file = tmp_path / "pes.txt"
        prof_file.write_text("""--# PES
0.0; 10.0; 20.0
--# LABELS
A; B; C
--# GRAPHS
legend: false
title: Test CLI Plot
label: Path 1
""")
        cmd = ReactionProfile()
        cmd.main([str(prof_file)])

    def test_nbo_parser_cli(self, capsys):
        from frogchem.cli.nbo_parser import NboParser
        if os.path.exists("files/nbo.out"):
            cmd = NboParser()
            cmd.main(["files/nbo.out", "--npa", "--e2", "-t", "2.0", "--bonds", "--lp", "--analysis", "--wiberg", "--dtype", "LP", "-i", "0", "1", "--format", "json"])
            captured = capsys.readouterr()
            assert "Bond" in captured.out or "Charge" in captured.out
            assert "Donor Orbital" in captured.out

    def test_calc_rrho_boltz_order(self, capsys):
        from frogchem.cli.calc_rrho import CalcRrho
        if os.path.exists("files/test.out"):
            cmd = CalcRrho()
            cmd.main(["files/test.out", "--boltz", "--sort", "--format", "json"])
            captured = capsys.readouterr()
            assert "Boltzmann" in captured.out or "Boltz %" in captured.out or "E [Eh]" in captured.out

    def test_substitute_fragment_cli(self, tmp_path, monkeypatch):
        from ase import Atoms
        from ase.io import write
        from frogchem.cli.substitute_fragment import SubstituteFragment
        mol_f = str(tmp_path / "mol.xyz")
        frag_f = str(tmp_path / "frag.xyz")
        mol = Atoms('OH2', positions=[[0, 0, 0], [0, 0.7, 0.6], [0, -0.7, 0.6]])
        frag = Atoms('OH', positions=[[0, 0, 1.5], [0, 0, 2.5]])
        write(mol_f, mol)
        write(frag_f, frag)

        # --frag registers the fragment: keep the package library untouched
        import frogchem.substituter.database_sub as db_mod
        lib_dir = tmp_path / "lib"

        def mock_init(self, *a, **kw):
            self.base_dir = lib_dir
            self.structures_dir = lib_dir / "structures"
            self.db_file = lib_dir / "library.json"
            self.structures_dir.mkdir(parents=True, exist_ok=True)
            self.data = self._load()

        monkeypatch.setattr(db_mod.FragmentDatabase, "__init__", mock_init)

        cmd = SubstituteFragment()
        cmd.main(['--mol', mol_f, '--frag', frag_f, '--frag-pair', '1', '0', '--mol-pairs', '1', '0', '--track', '0', '2'])
        out_f = tmp_path / "mol_frag.xyz"
        assert out_f.exists()

    def test_mo_coeff_cli(self, monkeypatch, capsys):
        import matplotlib.pyplot as plt
        monkeypatch.setattr(plt, "show", lambda: None)
        from frogchem.cli.mo_coeff import MoCoeff
        if os.path.exists("files/orb.out"):
            cmd = MoCoeff()
            cmd.main(["files/orb.out", "-i", "0", "1", "-b", "2", "--format", "json"])
            captured = capsys.readouterr()
            assert "MO" in captured.out
            assert "Total_s" in captured.out

    def test_cie_plot_cli(self, tmp_path):
        from frogchem.cli.cie_plot import CiePlot
        csv_path = str(tmp_path / "spec.csv")
        with open(csv_path, "w") as f:
            f.write("wavelength,intensity\n")
            for wl in range(380, 781, 10):
                f.write(f"{wl},{np.exp(-((wl - 550) / 60) ** 2):.4f}\n")
        out_png = str(tmp_path / "cie.png")
        cmd = CiePlot()
        cmd.main([csv_path, "-o", out_png])
        assert os.path.exists(out_png)

    def test_md_analysis_thermo_and_plot(self, tmp_path):
        from frogchem.cli.md_analysis import parse_thermo, plot_combined_data
        mdout = tmp_path / "step.mdout"
        mdout.write_text(""" NSTEP =      100   TIME(PS) =       0.200  TEMP(K) =   298.15  PRESS =     1.0132
 Etot =     -1000.0000  EKtot =       300.0000  EPtot =     -1300.0000
 NSTEP =      200   TIME(PS) =       0.400  TEMP(K) =   299.15  PRESS =     1.0200
 Etot =     -1002.0000  EKtot =       301.0000  EPtot =     -1303.0000
 NSTEP =      300   TIME(PS) =       0.600  TEMP(K) =   298.50  PRESS =     1.0150
 Etot =     -1001.0000  EKtot =       300.5000  EPtot =     -1301.5000
""")
        df_thermo = parse_thermo([str(mdout)])
        assert df_thermo is not None
        assert len(df_thermo) >= 2
        fig = plot_combined_data(df_thermo, None)
        out_png = tmp_path / "md_res_summary.png"
        fig.savefig(out_png)
        assert out_png.exists()

    def test_tddft_parser_cli(self, tmp_path, monkeypatch):
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        from frogchem.cli.tddft_parser import TddftParser
        tddft_file = Path("files/tddft.out").resolve()
        if tddft_file.exists():
            monkeypatch.chdir(tmp_path)
            cmd = TddftParser()
            cmd.main([str(tddft_file), "--uv", "--ecd", "--save-spectra", "--threshold", "0.1",
                      "--out-dir", str(tmp_path)])
            assert any(tmp_path.glob("*.png"))
            assert any(tmp_path.glob("*.csv"))

    def test_electronic_density_cli(self, tmp_path, monkeypatch):
        from pathlib import Path
        from frogchem.cli.electronic_density import ElectronicDensity, get_frontier_orbitals
        orb_file = Path("files/orb.out").resolve()
        if orb_file.exists():
            monkeypatch.chdir(tmp_path)
            cmd = ElectronicDensity()
            # 1. Default non-interactive (no flags) with custom output name
            cmd.main([str(orb_file), "-o", "custom.cube", "--voxels", "5", "--padding", "1.0"])
            assert (tmp_path / "custom.cube").exists()

            # 2. HOMO and LUMO with prefix
            cmd.main([str(orb_file), "--homo", "--lumo", "-o", "frontier.cube", "--voxels", "5", "--padding", "1.0"])
            assert (tmp_path / "frontier_homo.cube").exists()
            assert (tmp_path / "frontier_lumo.cube").exists()

            # 3. Specific orbital, per-input name in --out-dir (orb.out is closed-shell: no spin density)
            cmd.main([str(orb_file), "--orbital", "0", "--voxels", "5", "--padding", "1.0",
                      "--out-dir", str(tmp_path)])
            assert (tmp_path / "orb_orbital_0.cube").exists()

        # 4. Calculation without basis set data
        test_file = Path("files/test.out").resolve()
        if test_file.exists():
            cmd = ElectronicDensity()
            cmd.main([str(test_file), "--density", "--voxels", "5", "--out-dir", str(tmp_path)])

        # 5. Helper unit check
        h, l = get_frontier_orbitals(None)
        assert h is None and l is None

    def test_detect_symmetry_modes(self, capsys):
        from frogchem.cli.detect_symmetry import DetectSymmetry
        if os.path.exists("files/water.xyz"):
            cmd = DetectSymmetry()
            cmd.main(["files/water.xyz", "-t", "0.2"])
            out1 = capsys.readouterr().out
            assert "Point Group" in out1
            cmd.main(["files/water.xyz", "-g", "C2v"])
            out2 = capsys.readouterr().out
            assert "C2v" in out2


class TestCLICoherenceAndAliases:
    """Verify that normalized flags and --table-file work consistently."""

    def test_out_file_json_and_csv(self, sample_water_xyz, tmp_path):
        from frogchem.cli.internals import Internals
        json_out = str(tmp_path / "internals.json")
        csv_out = str(tmp_path / "internals.csv")

        cmd = Internals()
        cmd.main([sample_water_xyz, "-i", "0", "1", "--format", "json", "--table-file", json_out])
        assert os.path.isfile(json_out)
        data = json.loads(Path(json_out).read_text())
        assert len(data) >= 1

        cmd.main([sample_water_xyz, "-i", "0", "1", "--format", "csv", "--table-file", csv_out])
        assert os.path.isfile(csv_out)
        content = Path(csv_out).read_text()
        assert "Structure" in content

    def test_get_charges_indexes_alias(self):
        from frogchem.cli.get_charges import GetCharges
        cmd = GetCharges()
        parser = cmd.build_parser()
        args1 = parser.parse_args(["f.out", "-i", "0", "1"])
        assert args1.indices == [0, 1]
        args3 = parser.parse_args(["f.out", "--indices", "0", "1"])
        assert args3.indices == [0, 1]

    def test_hessian_threshold_alias(self):
        from frogchem.cli.hessian_localizer import HessianLocalizer
        cmd = HessianLocalizer()
        parser = cmd.build_parser()
        assert parser.parse_args(["f.out", "-t", "10.0"]).threshold == 10.0
        assert parser.parse_args(["f.out", "--threshold", "10.0"]).threshold == 10.0

    def test_nbo_aliases(self):
        from frogchem.cli.nbo_parser import NboParser
        cmd = NboParser()
        parser = cmd.build_parser()
        assert parser.parse_args(["f.out", "-t", "3.5"]).threshold == 3.5
        assert parser.parse_args(["f.out", "--threshold", "3.5"]).threshold == 3.5
        assert parser.parse_args(["f.out", "-i", "0", "2"]).indices == [0, 2]

    def test_rrho_cutoff_alias(self):
        from frogchem.cli.calc_rrho import CalcRrho
        cmd = CalcRrho()
        parser = cmd.build_parser()
        assert parser.parse_args(["f.out", "--cutoff", "50"]).cut_off == 50.0
        assert parser.parse_args(["f.out", "--model", "truhlar"]).model == "truhlar"
        assert parser.parse_args(["f.out", "-m", "grimme"]).model == "grimme"
        assert parser.parse_args(["f.out", "--sigma", "2"]).sigma == 2
        assert parser.parse_args(["f.out", "-s", "3"]).sigma == 3
        assert parser.parse_args(["f.out", "-P", "100.0"]).pressure == 100.0

    def test_symmetry_aliases(self):
        from frogchem.cli.detect_symmetry import DetectSymmetry
        cmd = DetectSymmetry()
        parser = cmd.build_parser()
        assert parser.parse_args(["f.xyz", "-t", "0.3"]).threshold == 0.3
        assert parser.parse_args(["f.xyz", "--threshold", "0.3"]).threshold == 0.3
        assert parser.parse_args(["f.xyz", "-w"]).write is True
        assert parser.parse_args(["f.xyz", "--write"]).write is True

    def test_nics_flag_indexes(self, sample_water_xyz, capsys):
        from frogchem.cli.nics_geometry import NicsGeometry
        cmd = NicsGeometry()
        cmd.main([sample_water_xyz, "-i", "0", "1"])
        captured = capsys.readouterr()
        assert "NewGTO" in captured.out

    def test_argument_grouping_structure(self):
        from frogchem.cli.sort_ez import SortEZ
        from frogchem.cli.internals import Internals

        # Tool with thresholds, no ensemble support
        p_ez = SortEZ().build_parser()
        groups_ez = {g.title: [a.dest for a in g._group_actions if a.dest != 'help'] for g in p_ez._action_groups}
        assert "threshold" in groups_ez["thresholds"]
        assert "indices" in groups_ez["options"]
        assert "ensemble averaging" not in groups_ez
        assert {"format", "table_file", "write", "out_dir", "verbose", "quiet"} <= set(groups_ez["output"])

        # Ensemble tool without thresholds
        p_int = Internals().build_parser()
        groups_int = {g.title: [a.dest for a in g._group_actions if a.dest != 'help'] for g in p_int._action_groups}
        assert "thresholds" not in groups_int
        assert "checkpoint" in groups_int["ensemble averaging"]
        assert "format" in groups_int["output"]

