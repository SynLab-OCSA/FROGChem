import os
from ase import Atoms
from ase.io.trajectory import Trajectory
from typing import Optional

class TrajectoryHandler:
    """
    Handles writing ASE optimization trajectories.
    Can be used as a Context Manager.
    """
    def __init__(self, filename: str, atoms: Atoms, mode: str = 'a'):
        self.filename = filename
        self.atoms = atoms
        self.mode = mode
        self.traj: Optional[Trajectory] = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self):
        """Opens the trajectory file."""
        self.traj = Trajectory(self.filename, self.mode, self.atoms)

    def write(self):
        """Writes the current atoms state to the trajectory."""
        if self.traj is not None:
            self.traj.write(self.atoms)

    def close(self):
        """Closes the trajectory file."""
        if self.traj is not None:
            self.traj.close()
            self.traj = None

def reset_trajectory_file(filename: str):
    """
    Helper to delete an existing trajectory file before starting a new run.
    Useful when processing a molecule from scratch.
    """
    if filename and os.path.exists(filename):
        os.remove(filename)