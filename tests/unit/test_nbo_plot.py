import pytest
import numpy as np
import os
from pathlib import Path
from frogchem.io import read_nbo_plot_37
from frogchem.density_builder import generate_visualization_scripts, build_nbo_cube
from frogchem.cli.nbo_parser import NboParser


def test_read_nbo_plot_37_file_not_found(tmp_path):
    fake_path = tmp_path / "nonexistent.37"
    with pytest.raises(FileNotFoundError):
        read_nbo_plot_37(fake_path)


def test_read_nbo_plot_37_with_explicit_dimensions(tmp_path):
    file_37 = tmp_path / "test.37"
    # Create 3 AOs x 2 NBOs = 6 coefficients in Fortran column order:
    # NBO 1: [1.0, 2.0, 3.0], NBO 2: [4.0, 5.0, 6.0]
    data = "1.000000 2.000000 3.000000 4.000000 5.000000 6.000000"
    file_37.write_text(data, encoding="utf-8")

    mat = read_nbo_plot_37(file_37, n_ao=3, n_nbo=2)
    assert mat.shape == (3, 2)
    np.testing.assert_allclose(mat[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(mat[:, 1], [4.0, 5.0, 6.0])


def test_read_nbo_plot_37_square_auto_detect(tmp_path):
    file_37 = tmp_path / "square.37"
    # 4 coefficients -> 2x2 square
    data = "1.0 2.0\n3.0 4.0"
    file_37.write_text(data, encoding="utf-8")

    mat = read_nbo_plot_37(file_37)
    assert mat.shape == (2, 2)
    # Fortran order: first column [1.0, 2.0], second column [3.0, 4.0]
    np.testing.assert_allclose(mat[:, 0], [1.0, 2.0])
    np.testing.assert_allclose(mat[:, 1], [3.0, 4.0])


def test_generate_visualization_scripts(tmp_path):
    cubes = [str(tmp_path / "nbo_1.cube"), str(tmp_path / "nbo_2.cube")]
    scripts = generate_visualization_scripts(cubes, output_dir=tmp_path, isovalue=0.04)

    cxc_file = Path(scripts['chimerax'])
    pml_file = Path(scripts['pymol'])

    assert cxc_file.is_file()
    assert pml_file.is_file()

    cxc_content = cxc_file.read_text()
    assert "open nbo_1.cube" in cxc_content
    assert "volume #1 level 0.0400" in cxc_content
    assert "volume #1 level -0.0400" in cxc_content

    pml_content = pml_file.read_text()
    assert "load nbo_1.cube, map_nbo_1" in pml_content
    assert "isosurface nbo_1_pos, map_nbo_1, 0.0400" in pml_content


def test_build_nbo_cube_with_calc(tmp_path):
    if not os.path.exists("files/nbo.out"):
        pytest.skip("files/nbo.out not found")

    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation("files/nbo.out")

    n_ao = len(calc.mask_atoms)
    dummy_coeff = np.zeros(n_ao)
    dummy_coeff[0] = 1.0  # Put weight on first AO

    out_cube = tmp_path / "test_nbo_1.cube"
    res = build_nbo_cube(
        calc=calc,
        nbo_index=1,
        coeff_vector=dummy_coeff,
        output=str(out_cube),
        n_voxels=3,
        label="CR ( 1) C 1",
        occupancy=1.9999,
        energy=-10.17,
    )

    assert out_cube.is_file()
    content = out_cube.read_text()
    assert "NBO 1: CR ( 1) C 1" in content
    assert "Occ: 1.9999" in content


def test_cli_summary_and_all(capsys):
    if not os.path.exists("files/nbo.out"):
        pytest.skip("files/nbo.out not found")

    parser = NboParser()
    parser.main(["files/nbo.out", "--summary"])
    captured = capsys.readouterr()
    assert "Natural Population Summary" in captured.out
    assert "Lewis Population" in captured.out
    assert "Top 5 E(2)" in captured.out


def test_cli_cube_generation_with_plot_file(tmp_path, capsys):
    if not os.path.exists("files/nbo.out"):
        pytest.skip("files/nbo.out not found")

    from frogchem.orca_parser.calculation import Calculation
    calc = Calculation("files/nbo.out")
    n_ao = len(calc.mask_atoms)

    # Create synthetic .37 file with 845 AOs x 2 NBOs
    mat = np.zeros((n_ao, 2))
    mat[0, 0] = 1.0
    mat[1, 1] = 1.0
    fake_37 = tmp_path / "nbo.37"
    fake_37.write_text(" ".join(f"{x:.6f}" for x in mat.flatten(order='F')), encoding="utf-8")

    out_dir = tmp_path / "cubes_out"

    parser = NboParser()
    parser.main([
        "files/nbo.out",
        "--cube", "1", "2",
        "--coeff-file", str(fake_37),
        "--out-dir", str(out_dir),
        "--spacing", "1.5",
    ])

    captured = capsys.readouterr()
    assert "NBO Cubes" in captured.out
    cube_dir = out_dir / "nbo_nbo_cubes"
    assert len(list(cube_dir.glob("nbo_nbo*.cube"))) == 2
    assert (cube_dir / "view_chimerax.cxc").is_file()
    assert (cube_dir / "view_pymol.pml").is_file()
