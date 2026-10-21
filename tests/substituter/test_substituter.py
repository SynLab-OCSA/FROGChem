import os
import shutil
import pytest
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from frogchem.substituter.builder import (
    check_clash, 
    attach_fragment, 
    get_optimal_bond_length, 
    calculate_steric_score,
    STANDARD_BOND_LENGTHS
)
from frogchem.substituter.trajectory import TrajectoryHandler, reset_trajectory_file
from frogchem.substituter.database_sub import FragmentDatabase, FragmentEntry, SCOPE_PRESETS
from frogchem.substituter.optimizer import TopologicalOptimizer, relax_with_rdkit


class DummyHarmonicCalculator(Calculator):
    """Simple harmonic toy calculator for testing optimizer."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, target_positions=None, **kwargs):
        super().__init__(**kwargs)
        self.target_positions = target_positions

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        if self.target_positions is None:
            self.target_positions = pos.copy()
        diff = pos - self.target_positions
        energy = 0.5 * np.sum(diff ** 2)
        forces = -diff
        self.results['energy'] = float(energy)
        self.results['forces'] = forces


class TestBuilder:
    def test_check_clash(self):
        # Two carbon atoms far apart
        mol1 = Atoms('C', positions=[[0, 0, 0]])
        mol2 = Atoms('C', positions=[[10, 0, 0]])
        assert check_clash(mol1, mol2) is True

        # Two carbon atoms on top of each other
        mol_clash = Atoms('C', positions=[[0.1, 0, 0]])
        assert check_clash(mol1, mol_clash) is False

    def test_get_optimal_bond_length(self):
        # Canonical single bond lengths
        assert get_optimal_bond_length("C", "C") == 1.54
        assert get_optimal_bond_length("C", "F") == 1.35
        assert get_optimal_bond_length("C", "H") == 1.09
        assert get_optimal_bond_length("P", "C") == 1.84
        # Symmetric check
        assert get_optimal_bond_length("F", "C") == 1.35
        # Fallback check
        val = get_optimal_bond_length("Xe", "He")
        assert val > 0.0

    def test_calculate_steric_score(self):
        mol = Atoms('C', positions=[[0, 0, 0]])
        frag_far = Atoms('C', positions=[[5.0, 0, 0]])
        n_clash, repulsion, min_dist = calculate_steric_score(mol, frag_far)
        assert n_clash == 0
        assert repulsion < 1e-4
        assert np.isclose(min_dist, 5.0)

        frag_close = Atoms('C', positions=[[0.5, 0, 0]])
        n_clash, repulsion, min_dist = calculate_steric_score(mol, frag_close)
        assert n_clash > 0
        assert repulsion > 10.0

    def test_attach_fragment_success(self):
        # Methane CH4: center C at 0,0,0; H at 1.09, 0, 0
        mol = Atoms(
            symbols=['C', 'H', 'H', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ]
        )
        # Fragment: Methyl radical with dummy H (attach C to anchor C, delete dummy H)
        frag = Atoms(
            symbols=['H', 'C', 'H', 'H', 'H'],
            positions=[
                [-1.09, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.36, 1.03, 0.0],
                [0.36, -0.51, 0.89],
                [0.36, -0.51, -0.89],
            ]
        )

        # Replace H(1) on mol, anchored to C(0)
        # Delete H(0) on frag, attach from C(1)
        new_mol, added_indices, (anchor, attach) = attach_fragment(
            mol=mol,
            frag_template=frag,
            mol_pair=(1, 0),
            frag_pair=(0, 1),
            n_rotations=8,
            clash_thresh=0.5
        )

        # Ethane has 8 atoms
        assert len(new_mol) == 8
        assert anchor == 0
        assert attach == 4
        assert len(added_indices) == 4
        # Verify core atoms didn't move
        assert np.allclose(new_mol.positions[0], mol.positions[0])


class TestTrajectoryHandler:
    def test_trajectory_write_and_reset(self, tmp_path):
        traj_file = str(tmp_path / "test.traj")
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])

        with TrajectoryHandler(traj_file, atoms, mode='w') as handler:
            handler.write()
            atoms.positions[1, 2] = 0.75
            handler.write()

        assert os.path.exists(traj_file)
        assert os.path.getsize(traj_file) > 0

        reset_trajectory_file(traj_file)
        assert not os.path.exists(traj_file)


class TestFragmentDatabase:
    @pytest.fixture
    def custom_db(self, tmp_path):
        db = FragmentDatabase(base_dir=tmp_path, auto_init_standard=False)
        return db

    def test_add_and_get_fragment(self, custom_db, tmp_path):
        src_xyz = tmp_path / "frag.xyz"
        atoms = Atoms('CH4', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]])
        from frogchem.io import write_xyz
        write_xyz(str(src_xyz), [atoms.get_chemical_symbols()], [atoms.get_positions()])

        name = custom_db.add_fragment("methyl", str(src_xyz), delete_idx=1, attach_idx=0, desc="Methyl group")
        assert name == "methyl"
        assert "methyl" in custom_db.list_fragments()

        path, pair = custom_db.get_fragment("methyl")
        assert os.path.exists(path)
        assert pair == (1, 0)
        assert custom_db.get_fragment("nonexistent") is None

    def test_add_from_smiles(self, custom_db):
        name = custom_db.add_from_smiles("phenyl_test", "*c1ccccc1", desc="Test phenyl")
        assert name == "phenyl_test"
        path, pair = custom_db.get_fragment("phenyl_test")
        assert os.path.exists(path)
        assert pair[0] == 0  # dummy was at 0
        assert pair[1] == 1  # attach is neighbor

    def test_scope_presets(self, custom_db):
        assert "halogens" in SCOPE_PRESETS
        assert "alkyls" in SCOPE_PRESETS
        assert "ligands" in SCOPE_PRESETS
        halogens = custom_db.resolve_scope("halogens")
        assert "fluoro" in halogens
        assert "chloro" in halogens

    def test_update_and_delete_fragment(self, custom_db, tmp_path):
        src_xyz = tmp_path / "frag.xyz"
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
        from frogchem.io import write_xyz
        write_xyz(str(src_xyz), [atoms.get_chemical_symbols()], [atoms.get_positions()])

        custom_db.add_fragment("diatom", str(src_xyz), delete_idx=1, attach_idx=0)
        custom_db.update_fragment("diatom", new_name="h2_frag", new_desc="Updated H2")

        assert "h2_frag" in custom_db.list_fragments()
        assert "diatom" not in custom_db.list_fragments()

        custom_db.delete_fragment("h2_frag")
        assert "h2_frag" not in custom_db.list_fragments()

    def test_find_fragment_by_structure(self, custom_db, tmp_path):
        src_xyz = tmp_path / "h2.xyz"
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        from frogchem.io import write_xyz
        write_xyz(str(src_xyz), [atoms.get_chemical_symbols()], [atoms.get_positions()])

        custom_db.add_fragment("hydrogen", str(src_xyz), delete_idx=1, attach_idx=0)

        query_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        match = custom_db.find_fragment_by_structure(query_atoms)
        assert match == "hydrogen"

        different_atoms = Atoms('O2', positions=[[0, 0, 0], [0, 0, 1.2]])
        assert custom_db.find_fragment_by_structure(different_atoms) is None


class TestTopologicalOptimizer:
    def test_optimizer_run(self, tmp_path):
        atoms = Atoms('H2', positions=[[0, 0, 0.0], [0, 0, 0.8]])
        calc = DummyHarmonicCalculator(target_positions=np.array([[0, 0, 0.0], [0, 0, 0.74]]))
        
        optimizer = TopologicalOptimizer(atoms, calc)
        optimizer.prepare(active_indices=[1], new_bond=(0, 1), buffer=0.5)

        traj_file = str(tmp_path / "opt.traj")
        converged = optimizer.run(fmax=0.05, steps=30, trajectory_file=traj_file)
        assert converged is True
        assert os.path.exists(traj_file)
        assert optimizer.get_core_rmsd() == 0.0

    def test_optimizer_unprepared_raises(self):
        atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
        optimizer = TopologicalOptimizer(atoms, None)
        with pytest.raises(RuntimeError, match="prepare"):
            optimizer.run()

    def test_rdkit_relaxation_fixed_core(self):
        # Ethane with one slightly stretched C-H
        atoms = Atoms('C2H6', positions=[
            [0.0, 0.0, 0.0], [1.54, 0.0, 0.0],
            [-0.36, 1.03, 0.0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89],
            [1.90, 1.03, 0.0], [1.90, -0.51, 0.89], [1.90, -0.51, -0.89]
        ])
        initial_core = atoms.positions[:5].copy()
        # Move H at index 7
        atoms.positions[7] += [0.2, 0.2, 0.0]
        
        ok = relax_with_rdkit(atoms, active_indices=[5, 6, 7], method='mmff')
        assert ok is True
        # Core atoms (0..4) MUST NOT MOVE AT ALL (0.000 A)
        core_drift = np.max(np.abs(atoms.positions[:5] - initial_core))
        assert core_drift < 1e-5


class TestCLIScopeAndMultiSite:
    def test_scope_substitution_cli(self, tmp_path):
        from ase.io import write
        from frogchem.cli.substitute_fragment import SubstituteFragment
        mol_f = str(tmp_path / "methane.xyz")
        mol = Atoms('CH4', positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89]
        ])
        write(mol_f, mol)

        out_dir = tmp_path / "derivatives"
        cmd = SubstituteFragment()
        cmd.main([
            '--mol', mol_f, 
            '--mol-pairs', '1', '0', 
            '--scope', 'halogens',
            '--out-dir', str(out_dir),
            '--combined-xyz', str(tmp_path / "combined.xyz")
        ])

        assert (out_dir / "methane_fluoro.xyz").exists()
        assert (out_dir / "methane_chloro.xyz").exists()
        assert (out_dir / "methane_bromo.xyz").exists()
        assert (out_dir / "methane_iodo.xyz").exists()
        assert (tmp_path / "combined.xyz").exists()

    def test_multi_site_substitution_cli(self, tmp_path):
        from ase.io import write
        from frogchem.cli.substitute_fragment import SubstituteFragment
        # Methane replacing H1 and H2
        mol_f = str(tmp_path / "methane.xyz")
        mol = Atoms('CH4', positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89]
        ])
        write(mol_f, mol)

        cmd = SubstituteFragment()
        cmd.main([
            '--mol', mol_f,
            '--mol-pairs', '1', '0', '2', '0',
            '--db-frag', 'fluoro'
        ])
        # Two H replaced by two F -> CH2F2 (5 atoms: C, 2 H, 2 F)
        out_f = tmp_path / "methane_fluoro.xyz"
        assert out_f.exists()
        from ase.io import read
        res_mol = read(str(out_f))
        assert len(res_mol) == 5
        syms = res_mol.get_chemical_symbols()
        assert syms.count('F') == 2
        assert syms.count('C') == 1
        assert syms.count('H') == 2

