"""
Substitute atoms of molecules with fragments, keeping the scaffold fixed.

Replaces atoms in base molecule(s) with functional fragments from the curated library,
a scope preset or a custom XYZ file. Bond lengths are set from atomic radii, the fragment
is rotated to avoid clashes and then relaxed with the scaffold frozen (core RMSD 0 Å).

Andrea Pellegrini, 2026
"""
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from frogchem.cli.base import CLICommand, Guide, Output, split_name
from frogchem.exceptions import InputError
from frogchem.printer import print_info, print_warning, print_error


def get_calculator(method: str, device: str = 'cpu', charge: int = 0):
    """Factory to initialize the requested calculator."""
    method = method.lower()

    if method in ('rdkit', 'mmff', 'uff'):
        # Handled natively by TopologicalOptimizer via RDKit
        return None

    elif method == 'aimnet':
        try:
            from aimnet.calculators.aimnet2ase import AIMNet2ASE
        except ImportError as e:
            raise ImportError(
                "AIMNet2 not found. Install via: pip install 'aimnet[ase] @ git+https://github.com/isayevlab/aimnetcentral.git'"
            ) from e

        print_info(f"Initializing AIMNet2 (model='aimnet2', charge={charge})")
        return AIMNet2ASE("aimnet2", charge=charge)

    elif method == 'mace':
        try:
            from mace.calculators import mace_off
        except ImportError as e:
            raise ImportError("MACE not found. Install via: pip install mace-torch") from e

        print_info(f"Initializing MACE-OFF on {device}")
        return mace_off(model="small", device=device, default_dtype="float32")

    elif method == 'xtb':
        import shutil
        if not shutil.which("xtb"):
            raise FileNotFoundError("xTB executable not found in PATH.")
        raise NotImplementedError("Standalone xTB via ASE requires xtb-python or tblite. Use --method aimnet or rdkit.")

    else:
        raise ValueError(f"Method '{method}' not supported. Options: rdkit, aimnet, mace, xtb")


class SubstituteFragment(CLICommand, name='frog-substitute'):
    """Substitute atoms of molecules with fragments, keeping the scaffold fixed.

Replaces atoms in base molecule(s) with functional fragments from the curated library
(see frog-sub-db), a scope preset or a custom XYZ file. Bond lengths are set from atomic
radii, the fragment is rotated to avoid clashes and then relaxed with the scaffold frozen
(core RMSD 0 Å).

Target sites are pairs 'REPLACE ANCHOR' of 0-based atom indices of the base molecule: the
REPLACE atom is removed and the fragment is bonded to ANCHOR. A custom fragment (--frag)
is registered in the fragment library unless an identical one is already there.
"""
    category = 'substituter'
    outputs = [
        Output('table', 'Fragment Substitution Summary',
               'One row per molecule and fragment: status, core RMSD (Å), number of atoms, output file'),
        Output('file', '<stem>_<fragment>.xyz',
               'Substituted molecule; <fragment> is the library name, or the --frag file stem'),
        Output('file', '<stem>_<fragment>_opt.traj', 'ASE optimization trajectory (ML calculators only)',
               when='--traj'),
        Output('file', 'PATH', 'All substituted molecules in one multi-structure XYZ', when='--combined-xyz'),
        Output('file', 'frogchem/substituter/data/library.json',
               'Fragment library, updated in place when a new custom fragment is registered', when='--frag'),
    ]
    guide = Guide(
        why="""
            Building a series of derivatives by hand in a viewer (replace this H by Cl, Br, OMe, CF3, ...)
            is slow and gives uneven starting geometries. This command does it automatically: it removes
            an atom of your molecule, attaches a substituent from a built-in library (or your own XYZ
            fragment) in its place, rotates it to avoid clashes and relaxes **only the new fragment** with
            a force field, leaving the rest of the molecule exactly where it was. Use it to prepare
            substituent screens (e.g. all halogens, all aryls) on a catalyst or substrate before the QM
            optimisations.
            """,
        needs="""
            * The base molecule as an `.xyz` file, ideally already optimised.
            * The target site as two 0-based atom numbers: the atom to **replace** (usually an H) and the
              atom it is bonded to (the **anchor**, which will carry the substituent). Look them up in a
              viewer (the first atom of the file is 0), or pick them with `--gui-pick`.
            * A fragment name from the library (`frog-sub-db list`, e.g. `phenyl`, `methoxy`), a scope
              preset (`frog-sub-db scopes`, e.g. `halogens`), or your own fragment XYZ with `--frag`.
            * The default relaxation uses the RDKit force field. `--method aimnet` (AIMNet2 machine-learned
              potential, installed with FROG) is closer to DFT and accepts charged systems (`--charge`), but
              covers main-group elements only (no transition metals);
              `--method mace` needs `pip install mace-torch`. `--method xtb` is **not implemented yet**.
              If a calculator cannot be started, a warning is printed and RDKit is used instead.
            """,
        how=[
            "Find the site: in `tol.xyz` (toluene) the para H is atom 10, bonded to the para carbon, atom 4.",
            "Substitute: `frog-substitute --mol tol.xyz --mol-pairs 10 4 --db-frag phenyl --out-dir subs` "
            "writes `subs/tol_phenyl.xyz` (without `--out-dir` it is written next to the input).",
            "Screen several fragments: `--db-frags methyl ethyl isopropyl` or `--scope halogens`; add "
            "`--combined-xyz subs/all.xyz` to get all derivatives in one file for quick viewing.",
            "Several sites at once: give more pairs, `--mol-pairs 10 4 7 0` (each pair REPLACE ANCHOR).",
            "Use your own fragment: `--frag OMe.xyz --frag-pair 4 0` (fragment atom 4 is the dummy/H that is "
            "removed, atom 0 is the atom that bonds to the molecule).",
            "Open the result in a viewer and use it as the start of a normal QM optimisation.",
        ],
        expect="""
            A *Fragment Substitution Summary* with one row per molecule and fragment: **Status**
            (SUCCESS/FAILED), **Core RMSD** (how much the original atoms moved, Å), number of atoms and the
            output file. Example: toluene + phenyl at atom 10/4 → SUCCESS, core RMSD 0.0000 Å, 25 atoms;
            `--scope halogens` gives `tol_fluoro.xyz` … `tol_iodo.xyz`, 15 atoms each.

            * Core RMSD is 0 by design (the scaffold is frozen); with `--no-freeze-core` it shows how much the
              scaffold relaxed.
            * The geometry is a force-field guess, **not** a minimum: always reoptimise it with your QM method.
              With `--no-opt` the fragment is only placed geometrically.
            * `Optimization incomplete` means the relaxation stopped early: inspect the structure for clashes.
            * A custom fragment given with `--frag` is **added to the fragment library** (the package data
              file) unless an identical one is already there; remove it later with `frog-sub-db remove`.
            * `--track 0 4` adds an `original_id` column (extended XYZ): the original index for the tracked
              atoms, -1 for all the others (including the fragment atoms).
            """,
    )
    examples = [
        ('frog-substitute --mol cat.xyz --mol-pairs 12 3 --db-frag phenyl',
         'Replace atom 12 (bonded to atom 3) with a phenyl group'),
        ('frog-substitute --mol cat.xyz --mol-pairs 12 3 --scope halogens --out-dir derivs '
         '--combined-xyz derivs/all.xyz',
         'Screen all halogens on one site; one file per derivative plus a combined XYZ'),
        ('frog-substitute --mol cat.xyz --mol-pairs 12 3 --frag OMe.xyz --frag-pair 4 0 --method aimnet',
         'Attach a custom fragment (delete atom 4, bond through atom 0) and relax it with AIMNet2'),
    ]

    def setup_args(self, parser):
        parser.add_argument("--mol", required=True, nargs='+', metavar='XYZ', help="Base molecule file(s) (XYZ)")
        parser.add_argument("--combined-xyz", metavar='PATH',
                            help="Also write all substituted molecules to this multi-structure XYZ")
        self.add_out_dir_arg(parser)

        frag_group = parser.add_argument_group("fragment selection")
        frag_group.add_argument("--db-frag", type=str, metavar='NAME', help="One fragment from the library, e.g. phenyl")
        frag_group.add_argument("--db-frags", type=str, nargs='+', metavar='NAME',
                                help="Several fragments from the library, e.g. methyl ethyl isopropyl")
        frag_group.add_argument("--scope", type=str,
                                help="Scope preset, category, tag, comma-separated names or 'all' "
                                     "(e.g. halogens, alkyls, aryls, ewg, edg)")
        frag_group.add_argument("--frag", metavar='XYZ', help="Custom fragment XYZ file (use with --frag-pair)")
        frag_group.add_argument("--frag-pair", type=int, nargs=2, metavar=('DELETE', 'ATTACH'),
                                help="Fragment atoms (0-based): atom deleted on attachment and attachment atom")

        target_group = parser.add_argument_group("target selection")
        target_group.add_argument("--mol-pairs", type=int, nargs='+', metavar='IDX',
                                  help="Target sites as flat pairs of 0-based indices: REPLACE ANCHOR [REPLACE ANCHOR ...]")
        target_group.add_argument("--gui-pick", action="store_true",
                                  help="Pick the target atom pairs in a 3D viewer")

        opt_group = parser.add_argument_group("optimization")
        opt_group.add_argument("--no-opt", dest="opt", action="store_false",
                               help="Skip the relaxation (geometric placement only)")
        opt_group.add_argument("--method", type=str, default="rdkit", choices=["rdkit", "aimnet", "mace", "xtb"],
                               help="Method used to relax the fragment (xtb is not implemented yet: "
                                    "it falls back to the RDKit force field)")
        opt_group.add_argument("--optimizer", type=str, default="lbfgs",
                               choices=["lbfgs", "fire", "rdkit", "backtracking"],
                               help="Optimizer used with ML calculators")
        opt_group.add_argument("--no-freeze-core", dest="freeze_core", action="store_false",
                               help="Let the scaffold atoms move during the relaxation")
        opt_group.add_argument("--traj", action="store_true", help="Write the optimization trajectory (.traj)")
        opt_group.add_argument("--device", type=str, default="cpu", help="Device for ML calculators (cpu, cuda)")
        opt_group.add_argument("--charge", type=int, default=0, help="Total charge of the system")

        parser.add_argument("--buffer", type=float, default=0.3, metavar='Å',
                            help="Bonding buffer in Å for the topology check")
        parser.add_argument("--clash", type=float, default=0.8, metavar='Å',
                            help="Clash distance in Å used while rotating the fragment")
        parser.add_argument("--rotations", type=int, default=36, metavar='N',
                            help="Number of dihedral angles sampled for clash avoidance")
        parser.add_argument("--track", type=int, nargs='+', metavar='IDX',
                            help="Original atom indices (0-based) to track: written to an 'original_id' column "
                                 "(extended XYZ), -1 for the other atoms")

    def _resolve_fragments(self, args, db, read) -> List[Tuple[str, str, Any, Tuple[int, int]]]:
        """(display name, file tag, Atoms, (delete, attach)) for every requested fragment."""
        jobs = []
        if args.scope:
            names = db.resolve_scope(args.scope)
            if not names:
                raise InputError(f"Scope '{args.scope}' did not match any fragment or preset")
            print_info(f"Resolved scope '{args.scope}' into {len(names)} fragments: "
                       f"{', '.join(names[:8])}{'...' if len(names) > 8 else ''}")
        elif args.db_frags:
            names = args.db_frags
        elif args.db_frag:
            names = [args.db_frag]
        elif args.frag and args.frag_pair:
            template = read(args.frag)
            pair = tuple(args.frag_pair)
            name = db.find_fragment_by_structure(template)
            if name:
                print_info(f"Fragment matches the library entry '{name}'")
            else:
                name = db.add_fragment(name=Path(args.frag).stem, source_xyz=args.frag,
                                       delete_idx=pair[0], attach_idx=pair[1],
                                       desc="Auto-registered via substitute_fragment")
                print_info(f"Registered fragment '{name}' in the library")
                self.saved(db.db_file, 'library')
            return [(name, Path(args.frag).stem, template, pair)]
        else:
            raise InputError("Specify the fragment with --db-frag, --db-frags, --scope, "
                             "or --frag together with --frag-pair")

        for name in names:
            res = db.get_fragment(name)
            if not res:
                if args.db_frag:
                    raise InputError(f"Fragment '{name}' not found in the library")
                print_warning(f"Fragment '{name}' not found in the library: skipped")
                continue
            fpath, fpair = res
            jobs.append((name, name, read(fpath), fpair))
        return jobs

    def run(self, args):
        self.require_files(args.mol)
        if args.frag:
            self.require_files([args.frag])
        if not (args.gui_pick or args.mol_pairs):
            raise InputError("Specify the target atoms with --mol-pairs or --gui-pick")
        if args.mol_pairs and len(args.mol_pairs) % 2 != 0:
            raise InputError("--mol-pairs needs an even number of indices (REPLACE ANCHOR pairs)")

        from ase.io import read, write
        from frogchem.substituter.database_sub import FragmentDatabase
        from frogchem.substituter import attach_fragment, TopologicalOptimizer, reset_trajectory_file
        from frogchem.substituter.builder import track_atoms

        db = FragmentDatabase()
        frag_jobs = self._resolve_fragments(args, db, read)
        if not frag_jobs:
            raise InputError("No valid fragments resolved for substitution")

        calc = None
        if args.opt and args.method.lower() != 'rdkit':
            try:
                calc = get_calculator(args.method, args.device, args.charge)
                print_info(f"Calculator initialized: {args.method.upper()} on {args.device}")
            except Exception as e:
                print_warning(f"Calculator initialization failed: {e}. Falling back to the RDKit force field.")
                calc = None

        combined_structures = []
        summary_rows = []
        unreadable = []

        for mol_path in args.mol:
            mol_path_obj = Path(mol_path)
            print_info(f"Processing base molecule: {mol_path_obj.name}")
            try:
                base_mol = read(str(mol_path_obj))
            except Exception as e:
                print_error(f"Could not read base molecule file '{mol_path}': {e}")
                unreadable.append(mol_path)
                continue

            if args.gui_pick:
                from frogchem.gui_utils import select_atoms_visually
                print_info("Opening the GUI to select target atom pairs...")
                selection = select_atoms_visually(base_mol, pairs_mode=True)
                if not selection or len(selection) % 2 != 0:
                    print_warning(f"Invalid GUI selection: {mol_path_obj.name} skipped")
                    continue
                target_pairs = []
                for k in range(0, len(selection), 2):
                    idx1, idx2 = selection[k], selection[k + 1]
                    # The lighter atom is replaced, the heavier one is the anchor
                    if base_mol[idx1].mass < base_mol[idx2].mass:
                        target_pairs.append((idx1, idx2))
                    else:
                        target_pairs.append((idx2, idx1))
            else:
                target_pairs = list(zip(args.mol_pairs[::2], args.mol_pairs[1::2]))

            for frag_name, frag_tag, frag_template, frag_pair in frag_jobs:
                print_info(f"Substituting with fragment '{frag_name}'...")

                current_mol = base_mol.copy()
                # Track original atom indices to survive index shifts after each attachment
                current_mol.set_array('orig_idx', np.arange(len(current_mol)))
                if args.track:
                    track_atoms(current_mol, args.track)

                core_rmsd = 0.0
                traj_path = None
                if args.opt and args.traj:
                    traj_path = self.derived_path(mol_path_obj, f"{frag_tag}_opt", '.traj')
                    reset_trajectory_file(str(traj_path))

                try:
                    for orig_replace, orig_anchor in target_pairs:
                        curr_orig = current_mol.get_array('orig_idx')
                        match_rep = np.where(curr_orig == orig_replace)[0]
                        match_anc = np.where(curr_orig == orig_anchor)[0]
                        if len(match_rep) == 0 or len(match_anc) == 0:
                            raise ValueError(f"Target site (replace={orig_replace}, anchor={orig_anchor}) "
                                             "could not be resolved")

                        current_mol, added_indices, bond_tuple = attach_fragment(
                            current_mol,
                            frag_template,
                            (int(match_rep[0]), int(match_anc[0])),
                            tuple(frag_pair),
                            n_rotations=args.rotations,
                            clash_thresh=args.clash,
                            clash_handling="best_effort",
                        )

                        # New fragment atoms get orig_idx = -1
                        orig_arr = current_mol.get_array('orig_idx')
                        if len(orig_arr) < len(current_mol):
                            new_arr = np.full(len(current_mol), -1, dtype=int)
                            new_arr[:len(orig_arr)] = orig_arr
                            current_mol.set_array('orig_idx', new_arr)

                        if args.opt:
                            optimizer = TopologicalOptimizer(current_mol, calc)
                            optimizer.prepare(active_indices=added_indices, new_bond=bond_tuple,
                                              buffer=args.buffer, freeze_core=args.freeze_core)
                            opt_ok = optimizer.run(
                                buffer=args.buffer,
                                trajectory_file=str(traj_path) if traj_path else None,
                                optimizer_type=args.optimizer if calc is not None else "rdkit",
                            )
                            core_rmsd = optimizer.get_core_rmsd()
                            if not opt_ok:
                                print_warning("Optimization incomplete (topology boundary or step limit reached)")

                except Exception as e:
                    print_warning(f"Failed substitution with '{frag_name}': {e}")
                    summary_rows.append([mol_path_obj.name, frag_name, "FAILED", None, None, None])
                    continue

                out_path = self.derived_path(mol_path_obj, frag_tag, split_name(mol_path_obj)[1] or '.xyz')
                save_mol = current_mol.copy()
                del save_mol.arrays['orig_idx']
                write(str(out_path), save_mol)
                self.saved(out_path, 'geometry')
                if traj_path and traj_path.exists():
                    self.saved(traj_path, 'trajectory')
                combined_structures.append(save_mol)
                summary_rows.append([mol_path_obj.name, frag_name, "SUCCESS", float(core_rmsd),
                                     len(save_mol), out_path.name])

        if args.combined_xyz and combined_structures:
            combined_path = Path(args.combined_xyz)
            combined_path.parent.mkdir(parents=True, exist_ok=True)
            write(str(combined_path), combined_structures)
            self.saved(combined_path, f'combined XYZ ({len(combined_structures)} structures)')

        self.output(summary_rows, headers=["Molecule", "Fragment", "Status", "Core RMSD (Å)", "Atoms", "Output File"],
                    title="Fragment Substitution Summary", floatfmt='.4f')

        if unreadable:
            raise InputError(f"{len(unreadable)} base molecule file(s) could not be read: {', '.join(unreadable)}")
        if not combined_structures:
            raise InputError("No substituted molecule was generated")


def parse_args():
    return SubstituteFragment().build_parser().parse_args()

def main():
    SubstituteFragment().main()


if __name__ == '__main__':
    main()
