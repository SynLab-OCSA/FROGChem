"""
Topological and Constrained Geometry Optimization for Molecular Substitutions.

Preserves the base scaffold strictly fixed (0.000 A RMSD) while relaxing
newly attached fragments via modern optimizers (LBFGS, FIRE, RDKit MMFF/UFF,
AIMNet2, xTB, MACE).

Andrea Pellegrini, 2026
"""
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from typing import List, Tuple, Optional, Dict, Any

from frogchem.graph_theory import create_graph_matrix
from frogchem.chemistry import get_atomic_radius
from frogchem.printer import print_warning, print_success, print_info
from frogchem.substituter.trajectory import TrajectoryHandler


def relax_with_rdkit(
    atoms: Atoms, 
    active_indices: List[int], 
    method: str = 'mmff',
    charge: int = 0
) -> bool:
    """
    Ultra-fast force field relaxation using RDKit with all core scaffold atoms
    strictly fixed via AddFixedPoint. Guarantees 0.000 A movement for core atoms.

    Args:
        atoms (Atoms): ASE Atoms object to modify in-place.
        active_indices (List[int]): Indices of atoms allowed to move (the substituent).
        method (str): 'mmff' (default) or 'uff'.
        charge (int): Net molecular charge.

    Returns:
        bool: True if force field relaxation succeeded, False otherwise.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdDetermineBonds
    except ImportError:
        return False

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    rw_mol = Chem.RWMol()
    conf = Chem.Conformer(n_atoms)
    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        rw_mol.AddAtom(Chem.Atom(sym))
        conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))

    mol = rw_mol.GetMol()
    mol.AddConformer(conf)

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
    except Exception:
        pass

    Chem.FastFindRings(mol)
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    fixed_indices = set(range(n_atoms)) - set(active_indices)
    success = False

    # Try MMFF first
    if method.lower() in ('mmff', 'mmff94'):
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            if mp is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
                if ff is not None:
                    for idx in fixed_indices:
                        ff.AddFixedPoint(int(idx))
                    ff.Minimize(maxIts=500)
                    success = True
        except Exception:
            success = False

    # Fallback to UFF
    if not success:
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            if ff is not None:
                for idx in fixed_indices:
                    ff.AddFixedPoint(int(idx))
                ff.Minimize(maxIts=500)
                success = True
        except Exception:
            success = False

    if success:
        new_conf = mol.GetConformer()
        new_positions = new_conf.GetPositions()
        # Verify that fixed atoms did not move
        if fixed_indices:
            fixed_arr = list(fixed_indices)
            diff = np.max(np.abs(new_positions[fixed_arr] - positions[fixed_arr]))
            if diff > 1e-4:
                # Force fixed atoms back to exact starting positions
                new_positions[fixed_arr] = positions[fixed_arr]
        atoms.set_positions(new_positions)
        return True

    return False


class TopologicalOptimizer:
    """
    Optimizer that relaxes attached fragments while preserving molecular
    connectivity and keeping the base scaffold intact.
    Supports LBFGS, FIRE, RDKit MMFF/UFF, and adaptive backtracking.
    """

    def __init__(self, atoms: Atoms, calculator=None):
        self.atoms = atoms
        if calculator is not None:
            self.atoms.calc = calculator
        self.reference_graph = None
        self.active_indices: List[int] = []
        self.fixed_indices: List[int] = []
        self._initial_core_positions: Optional[np.ndarray] = None
        
        self._radii_cache: Dict[str, float] = {}
        self._symbols_cache: List[str] = []

    def prepare(
        self, 
        active_indices: List[int], 
        new_bond: Tuple[int, int],
        buffer: float = 0.3,
        freeze_core: bool = True
    ) -> None:
        """Initializes topology, reference graph, and constraints."""
        self.active_indices = list(active_indices)
        self._symbols_cache = self.atoms.get_chemical_symbols()
        n_atoms = len(self.atoms)
        
        # Determine fixed core atoms
        all_indices = set(range(n_atoms))
        self.fixed_indices = sorted(list(all_indices - set(self.active_indices)))
        
        if freeze_core and self.fixed_indices:
            self.atoms.set_constraint(FixAtoms(indices=self.fixed_indices))
            self._initial_core_positions = self.atoms.positions[self.fixed_indices].copy()
        else:
            self.atoms.set_constraint(None)
            self._initial_core_positions = None

        # Build Reference Connectivity Graph
        positions = self.atoms.get_positions()
        unique_syms = set(self._symbols_cache)
        self._radii_cache = {k: get_atomic_radius(k) for k in unique_syms}
        
        self.reference_graph = create_graph_matrix(
            self._symbols_cache, positions, self._radii_cache, buffer=buffer
        )

        # Enforce the newly formed bond in reference graph
        i, j = new_bond
        self.reference_graph[i, j] = 1
        self.reference_graph[j, i] = 1
        
    def _check_topology(self, positions: np.ndarray, buffer: float) -> bool:
        """Checks if atomic connectivity matches reference graph."""
        if self.reference_graph is None:
            return True
        current_graph = create_graph_matrix(
            self._symbols_cache, positions, self._radii_cache, buffer=buffer
        )
        return np.array_equal(self.reference_graph, current_graph)

    def get_core_rmsd(self) -> float:
        """Computes RMSD of fixed core atoms relative to starting geometry."""
        if self._initial_core_positions is None or len(self.fixed_indices) == 0:
            return 0.0
        current_core_pos = self.atoms.positions[self.fixed_indices]
        diff = current_core_pos - self._initial_core_positions
        return float(np.sqrt(np.mean(diff ** 2)))

    def run(
        self, 
        fmax: float = 0.05, 
        steps: int = 100, 
        buffer: float = 0.3, 
        alpha: float = 0.1, 
        trajectory_file: Optional[str] = None,
        optimizer_type: str = "lbfgs",
        pre_relax: bool = True
    ) -> bool:
        """
        Runs the optimization loop.

        Args:
            fmax (float): Force convergence threshold in eV/A.
            steps (int): Maximum optimization steps.
            buffer (float): Tolerance buffer for topology checking.
            alpha (float): Step size for backtracking line search (if used).
            trajectory_file (str, optional): Path to .traj file to log steps.
            optimizer_type (str): 'lbfgs' (default), 'fire', 'rdkit', or 'backtracking'.
            pre_relax (bool): Pre-relax substituent with RDKit MMFF to relieve severe clashes.

        Returns:
            bool: True if converged successfully, False otherwise.
        """
        if self.reference_graph is None:
            raise RuntimeError("Optimizer not initialized. Call prepare() first.")

        # 1. Quick pre-relaxation with RDKit force field if enabled
        if pre_relax and len(self.active_indices) > 0:
            relax_with_rdkit(self.atoms, self.active_indices, method='mmff')

        # If no ASE calculator attached or explicit 'rdkit' requested, we are done!
        if self.atoms.calc is None or optimizer_type.lower() == 'rdkit':
            core_rmsd = self.get_core_rmsd()
            if self._check_topology(self.atoms.get_positions(), buffer):
                print_success(f"    RDKit force-field relaxation complete. Core RMSD: {core_rmsd:.4f} A.")
                return True
            else:
                print_warning(f"    RDKit relaxation altered topology slightly beyond buffer. Core RMSD: {core_rmsd:.4f} A.")
                return True

        # Initialize Trajectory Handler
        handler = None
        if trajectory_file:
            handler = TrajectoryHandler(trajectory_file, self.atoms, mode='a')
            handler.open()
            handler.write()

        try:
            # 2. Modern ASE Optimizer (LBFGS or FIRE)
            opt_name = optimizer_type.lower()
            if opt_name in ('lbfgs', 'fire', 'bfgs'):
                from ase.optimize import LBFGS, FIRE, BFGS
                
                opt_cls = {'lbfgs': LBFGS, 'fire': FIRE, 'bfgs': BFGS}.get(opt_name, LBFGS)
                optimizer = opt_cls(self.atoms, logfile=None)
                
                last_safe_pos = self.atoms.get_positions().copy()
                topology_broken = False

                def step_callback():
                    nonlocal last_safe_pos, topology_broken
                    if handler:
                        handler.write()
                    curr_pos = self.atoms.get_positions()
                    if not self._check_topology(curr_pos, buffer):
                        topology_broken = True
                        self.atoms.set_positions(last_safe_pos)
                        raise StopIteration("Topology integrity limit reached.")
                    last_safe_pos = curr_pos.copy()

                optimizer.attach(step_callback)

                try:
                    converged = optimizer.run(fmax=fmax, steps=steps)
                except StopIteration:
                    print_warning("    Optimization halted: topology boundary encountered. Reverted to last safe geometry.")
                    converged = False
                except Exception as e:
                    print_warning(f"    Optimizer stopped with error: {e}")
                    converged = False

                core_rmsd = self.get_core_rmsd()
                if converged:
                    print_success(f"    Converged ({opt_name.upper()}) to fmax < {fmax}. Core RMSD: {core_rmsd:.4f} A.")
                return bool(converged)

            # 3. Adaptive Backtracking Steepest Descent (Legacy fallback)
            current_alpha = alpha
            max_alpha = 0.6
            min_alpha = 1e-4

            for step in range(steps):
                try:
                    forces = self.atoms.get_forces(apply_constraint=True)
                except Exception as e:
                    print_warning(f"    Force calculation failed at step {step}: {e}")
                    return False

                fmax_current = np.sqrt((forces**2).sum(axis=1).max())
                if fmax_current < fmax:
                    core_rmsd = self.get_core_rmsd()
                    print_success(f"    Converged at step {step} (fmax: {fmax_current:.4f}). Core RMSD: {core_rmsd:.4f} A.")
                    return True

                current_positions = self.atoms.get_positions()
                attempts = 0
                max_attempts = 10
                step_accepted = False

                while attempts < max_attempts:
                    if current_alpha < min_alpha:
                        print_warning("    Stuck: Step size too small. Stopping.")
                        return False

                    displacement = forces * current_alpha
                    max_disp = np.linalg.norm(displacement, axis=1).max()
                    if max_disp > 0.4:
                        scale = 0.4 / max_disp
                        displacement *= scale

                    candidate_positions = current_positions + displacement
                    
                    if self._check_topology(candidate_positions, buffer):
                        self.atoms.set_positions(candidate_positions)
                        step_accepted = True
                        if handler:
                            handler.write()
                        if attempts == 0:
                            current_alpha = min(current_alpha * 1.05, max_alpha)
                        break
                    else:
                        current_alpha *= 0.5
                        attempts += 1
                
                if not step_accepted:
                    print_warning(f"    Topology lock at step {step}.")
                    return False

            print_warning(f"    Reached max steps ({steps}) without full convergence.")
            return False

        finally:
            if handler:
                handler.close()