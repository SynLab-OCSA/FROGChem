import numpy as np
import pytest
from frogchem.chimera_commands import chimerax_add_sphere, chimerax_add_arrow


class TestChimeraXAddSphere:
    def test_basic_sphere(self):
        cmd = chimerax_add_sphere((0.0, 0.0, 0.0))
        assert cmd.startswith("shape sphere")
        assert "center 0.000,0.000,0.000" in cmd
        assert "radius 0.300" in cmd

    def test_custom_radius(self):
        cmd = chimerax_add_sphere((1.5, 2.5, 3.5), radius=0.5)
        assert "radius 0.500" in cmd

    def test_custom_color(self):
        cmd = chimerax_add_sphere((0, 0, 0), color="blue")
        assert "color blue" in cmd

    def test_custom_alpha(self):
        cmd = chimerax_add_sphere((0, 0, 0), alpha=0.5)
        assert "transparency 0.50" in cmd

    def test_opaque_alpha(self):
        cmd = chimerax_add_sphere((0, 0, 0), alpha=1.0)
        assert "transparency" not in cmd

    def test_named_sphere(self):
        cmd = chimerax_add_sphere((0, 0, 0), name="mysphere")
        assert "name mysphere" in cmd

    def test_all_options(self):
        cmd = chimerax_add_sphere((1.0, 2.0, 3.0), radius=0.8, color="green", alpha=0.3, name="test")
        assert "center 1.000,2.000,3.000" in cmd
        assert "radius 0.800" in cmd
        assert "color green" in cmd
        assert "transparency 0.70" in cmd
        assert "name test" in cmd


class TestChimeraXAddArrow:
    def test_basic_arrow(self):
        cmd = chimerax_add_arrow((0, 0, 0), (3, 0, 0))
        assert "shape cylinder" in cmd
        assert "shape cone" in cmd
        assert "fromPoint 0.000,0.000,0.000" in cmd

    def test_arrow_length(self):
        cmd = chimerax_add_arrow((0, 0, 0), (5, 0, 0), tip_length=1.0)
        assert "shape cylinder" in cmd
        assert "shape cone" in cmd

    def test_tip_too_long_raises(self):
        with pytest.raises(ValueError, match="Arrow head is longer than"):
            chimerax_add_arrow((0, 0, 0), (0.5, 0, 0), tip_length=1.0)

    def test_custom_radius(self):
        cmd = chimerax_add_arrow((0, 0, 0), (5, 0, 0), radius=0.5)
        assert "radius 0.500" in cmd

    def test_transparent_arrow(self):
        cmd = chimerax_add_arrow((0, 0, 0), (5, 0, 0), alpha=0.5)
        assert "transparency 0.50" in cmd

    def test_named_arrow(self):
        cmd = chimerax_add_arrow((0, 0, 0), (5, 0, 0), name="arrow1")
        assert "name arrow1_shaft" in cmd
        assert "name arrow1_head" in cmd

    def test_multiline_command(self):
        cmd = chimerax_add_arrow((0, 0, 0), (5, 0, 0))
        lines = cmd.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("shape cylinder")
        assert lines[1].startswith("shape cone")
