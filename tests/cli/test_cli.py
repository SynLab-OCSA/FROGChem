import pytest
import sys

# Import all CLI modules to populate the registry
import frogchem.cli.calc_rmsd
import frogchem.cli.calc_rrho
import frogchem.cli.create_crest_input
import frogchem.cli.cube_manipulation
import frogchem.cli.fragment_detection
import frogchem.cli.hessian_localizer
import frogchem.cli.internals
import frogchem.cli.mirror_ensemble
import frogchem.cli.reaction_profile
import frogchem.cli.sort_diastero
import frogchem.cli.sort_ez
import frogchem.cli.split_xyz
import frogchem.cli.tddft_parser
import frogchem.cli.detect_symmetry
import frogchem.cli.mo_coeff
import frogchem.cli.nics_geometry
import frogchem.cli.write_xyz
import frogchem.cli.calc_loc_spin
import frogchem.cli.get_charges
import frogchem.cli.nbo_parser
import frogchem.cli.substitute_fragment
import frogchem.cli.manage_sub_db
import frogchem.cli.cie_plot
import frogchem.cli.md_analysis
import frogchem.cli.calc_er
import frogchem.cli.plot_neb
import frogchem.cli.stereocenters
import frogchem.cli.electronic_density
import frogchem.cli.solvent_surface

from frogchem.cli.base import CLICommand, _REGISTRY, dispatch, main


class TestCLICommand:
    def test_all_tools_registered(self):
        expected = [
            'frog-rmsd', 'frog-rrho', 'frog-crest', 'frog-cube',
            'frog-fragments', 'frog-hessian', 'frog-internals',
            'frog-mirror', 'frog-profile', 'frog-sort', 'frog-sort-ez', 'frog-split',
            'frog-tddft', 'frog-sym', 'frog-mo', 'frog-nics',
            'frog-xyz', 'frog-spin', 'frog-charges', 'frog-nbo',
            'frog-substitute', 'frog-sub-db', 'frog-cie', 'frog-md',
            'frog-er', 'frog-plot-neb', 'frog-stereocenters',
            'frog-density', 'frog-solvent',
        ]
        for name in expected:
            assert name in _REGISTRY, f"{name} not in registry"

    def test_all_tools_are_cli_command_subclasses(self):
        for name, cls in _REGISTRY.items():
            assert issubclass(cls, CLICommand), f"{name} is not a CLICommand subclass"

    def test_all_tools_have_setup_args(self):
        for name, cls in _REGISTRY.items():
            assert hasattr(cls, 'setup_args'), f"{name} missing setup_args"

    def test_all_tools_have_run(self):
        for name, cls in _REGISTRY.items():
            assert hasattr(cls, 'run'), f"{name} missing run"

    def test_dispatch_unknown_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            dispatch('nonexistent-tool')
        assert exc_info.value.code == 1

    def test_dispatch_none_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            dispatch(None)
        assert exc_info.value.code == 0
