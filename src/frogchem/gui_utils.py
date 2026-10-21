import numpy as np
import sys
from typing import TYPE_CHECKING
from frogchem.printer import print_info, print_warning, print_error, print_success

if TYPE_CHECKING:
    from ase import Atoms
    from ase.gui.gui import GUI

def customize_gui_appearance(gui):
    gui.config['covalent_radii_scale'] = 0.7
    gui.show_bonds = True
    try:
        if hasattr(gui, 'set_colors'):
            gui.set_colors('jmol')
    except Exception as e:
        print_warning(f"Could not apply some GUI customizations: {e}")

def select_atoms_visually(atoms: Atoms, limit: None|int = None, pairs_mode: bool = False) -> list[int]:
    """
    Opens the ASE GUI to allow visual selection of atoms with a retry loop.
    
    The function opens a window where the user must select atoms. 
    It verifies the number of selected atoms matches 'limit'.
    If not, it asks the user to retry.

    Args:
        atoms: The ASE Atoms object to visualize.
        limit: The exact number of atoms expected to be selected.

    Returns:
        List[int]: A list of indices of the selected atoms. Returns empty list if aborted.
    """
    
    while True:
        print_info(f"🖥️  Opening GUI... Please select exactly {limit} atoms.")
        print("      > HOW TO SELECT: Hold 'Ctrl' and Click on the atoms.")
        print("      > HOW TO ROTATE MOLECULE: Right click and drag.")
        print("      > WHEN DONE: Close the visualization window to confirm.\n")

        # ponytail: lazy-load GUI libraries only when user requests visual selection
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        # Create a fresh Images object to ensure clean state
        images = Images([atoms])
        
        try:
            # Initialize and run GUI (blocks execution until window is closed)
            gui = GUI(images)
            customize_gui_appearance(gui)
            gui.run()
        except KeyboardInterrupt:
            print_warning("\nSelection interrupted by user.")
            return []
        except Exception as e:
            print_error(f"Failed to open GUI: {e}")
            return []

        # --- Retrieve Selection ---

        raw_indices = []
        
        # Try retrieving from GUI class first (newer ASE)
        if hasattr(gui, 'selected_atoms'):
            val = gui.selected_atoms
            if callable(val):
                raw_indices = list(val())
            else:
                raw_indices = list(val)

        print(raw_indices)

        if not raw_indices and hasattr(images, 'selected'):
            sel = images.selected
            # FIX: Handle dimensionality
            if hasattr(sel, 'shape'):
                if sel.ndim == 1:
                    # Case: (n_atoms,) -> Array of booleans for the single image
                    raw_indices = np.where(sel)[0].tolist()
                elif sel.ndim == 2:
                    # Case: (n_images, n_atoms) -> Take first image [0]
                    raw_indices = np.where(sel[0])[0].tolist()


        selected_indices = [int(idx) for idx in raw_indices]
        n_selected = len(selected_indices)

        # --- Validation ---
        valid = True
        error_msg = ""

        if limit is not None and n_selected != limit:
            valid = False
            error_msg = f"You selected {n_selected} atoms, but exactly {limit} are required."
        
        if pairs_mode:
            if n_selected == 0:
                valid = False
                error_msg = "No atoms selected."
            elif n_selected % 2 != 0:
                valid = False
                error_msg = f"You selected {n_selected} atoms (ODD). Please select PAIRS (Even number)."

        if valid:
            print_success(f"Successfully selected {n_selected} atoms: {selected_indices}")
            return selected_indices
        
        # --- Retry Logic ---
        print_warning(f"Selection Mismatch: {error_msg}")
        if n_selected == 0:
            print("      (Tip: Did you hold 'Ctrl' while clicking?)")

        try:
            user_input = input("   🔄 Selection incorrect. Try again? [Y/n] ").strip().lower()
            if user_input in ('n', 'no'):
                print_warning("Selection aborted by user.")
                return []
            print_info("   Restarting selection interface...\n")
        except KeyboardInterrupt:
            return []