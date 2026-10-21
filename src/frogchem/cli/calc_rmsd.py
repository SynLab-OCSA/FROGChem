"""
RMSD between the distance-matrix eigenvalue spectra of structures and a reference.

Andrea Pellegrini, 2025
"""

import numpy as np

from frogchem.exceptions import InputError
from frogchem.filetypes import load_structures
from frogchem.math_geometry import calc_distance_matrix, calc_rmsd
from frogchem.printer import print_info, print_warning
from frogchem.cli.base import CLICommand, Guide, Output, ENSEMBLE_WEIGHTS


class CalcRmsd(CLICommand, name='frog-rmsd'):
    """Compare structures with a reference via the RMSD of their distance-matrix eigenvalues.

    The eigenvalues of the interatomic distance matrix are invariant to translation,
    rotation, reflection and any permutation of the atoms, so no alignment is needed.
    Element identity is not encoded and enantiomers give 0: this is not a Kabsch RMSD.
    Without -r/--ref the first structure of the first file is the reference.
    """
    category = 'geometry'
    supports_ensemble = True
    outputs = [
        Output('table', 'λ-RMSD (reference: <ref>)',
               "One row per structure ('<file> - Conf <n>', n 0-based in file order): "
               'eigenvalue RMSD to the reference (Å)'),
        ENSEMBLE_WEIGHTS,
    ]
    guide = Guide(
        why="""
            Tell how different two geometries are with one number: how far each conformer of an ensemble is
            from the lowest one, whether an optimised structure still resembles the X-ray structure, or
            whether two "different" conformers are in fact the same. This command uses a measure that needs
            **no alignment and no matching atom order**: it compares the eigenvalues (λ, a set of numbers that
            characterises the shape) of the matrix of all interatomic distances.
            """,
        needs="""
            * The structures to compare: `.xyz` files (multi-structure allowed), ORCA or Gaussian outputs
              (last geometry). All must have the **same number of atoms** (of heavy atoms with
              `--no-hydrogen`); the atom order may differ.
            * Optionally a reference (`-r`): XYZ, ORCA or Gaussian; the first structure is used. Without `-r`
              the first structure of the first file is the reference.
            * Optionally (`-c`) a checkpoint with Boltzmann populations to get the population-weighted
              average RMSD.
            """,
        how=[
            "Compare all conformers with the first one: `frog-rmsd ensemble.xyz`.",
            "Compare with an experimental or other reference structure: `frog-rmsd conf_*/geom.xyz -r xray.xyz`.",
            "Add `--no-hydrogen` when hydrogens are missing or unreliable in the reference (X-ray structures, "
            "methyl rotations) — it is the usual choice to compare with crystal structures.",
            "Add `-c checkpoint.json` to append a *⟨Ensemble Average⟩* row weighted by the Boltzmann populations.",
        ],
        expect="""
            One row per structure (`<file> - Conf <n>`, n 0-based) with the λ-RMSD in Å; the reference itself
            gives 0.000. Example (18-conformer Ag complex `ensemble.xyz`, 168 atoms, reference conformer 0):
            0.504 Å (conformer 4) to 5.051 Å (conformer 15); with populations 40/25/15/10/10 % on the first
            five conformers the weighted average is 0.564 Å.

            * The values are **not** the usual (Kabsch) RMSD of atomic positions printed by viewers or by CREST,
              and are not on the same scale: use them to rank and compare structures, not as "atoms moved by
              x Å". As a rough guide, values close to 0 (a few hundredths of Å) mean the same structure; clearly
              larger values mean a different conformer.
            * **Enantiomers give 0** (a mirror image has the same distances), and element identity is not
              encoded: this measure cannot distinguish R from S or two structures that differ only by swapping
              atoms of different elements. Use `frog-stereocenters` for configurations.
            * `has N atoms, the reference has M` means the two files do not contain the same molecule (or one
              has hydrogens removed: use `--no-hydrogen` on both).
            """,
    )
    examples = [
        ('frog-rmsd ensemble.xyz', 'RMSD of every conformer to the first one'),
        ('frog-rmsd conf_*/protocol_1/geom.xyz -r xray.xyz --no-hydrogen',
         'Heavy-atom RMSD of optimized conformers to an X-ray structure'),
        ('frog-rmsd conf_*/protocol_1/geom.xyz -r xray.xyz -c checkpoint.json',
         'Add the Boltzmann-weighted average RMSD to the reference'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ (multi-structure allowed), ORCA or Gaussian files')
        parser.add_argument('-r', '--ref', metavar='FILE',
                            help='Reference structure (XYZ, ORCA or Gaussian; first structure is used)')
        parser.add_argument('--no-hydrogen', action='store_true', help='Exclude hydrogen atoms')

    def _eigenvalues(self, atoms, geom):
        dm = calc_distance_matrix(geom=np.asarray(geom), atoms=np.asarray(atoms), exclude_H=self.args.no_hydrogen)
        # The distance matrix is real symmetric: eigvalsh returns real, sorted eigenvalues
        return np.linalg.eigvalsh(dm)

    def run(self, args):
        files = self.require_files(args.file)
        if args.ref:
            self.require_files([args.ref])

        eig_ref, ref_label = None, None
        if args.ref:
            structs = load_structures(args.ref)
            if not structs:
                raise InputError(f"No structure found in {args.ref}")
            if len(structs) > 1:
                print_warning(f"{args.ref} contains {len(structs)} structures: using the first as reference")
            eig_ref, ref_label = self._eigenvalues(*structs[0]), args.ref

        results = []
        for p in files:
            f = str(p)
            print_info(f"Processing {f}")
            structs = load_structures(f)
            if not structs:
                raise InputError(f"No structure found in {f}")

            if eig_ref is None:
                print_info(f"Using the first structure of {f} as reference")
                eig_ref, ref_label = self._eigenvalues(*structs[0]), f"{f} - Conf 0"

            for idx, (atoms, geom) in enumerate(structs):
                eig = self._eigenvalues(atoms, geom)
                if len(eig) != len(eig_ref):
                    raise InputError(f"{f} (conf {idx}) has {len(eig)} atoms, the reference has {len(eig_ref)}"
                                     f"{' (hydrogens excluded)' if args.no_hydrogen else ''}")
                results.append([f'{f} - Conf {idx}', float(calc_rmsd(eig_ref, eig))])

        self.output(results, headers=['Structure', 'λ-RMSD (Å)'], title=f"λ-RMSD (reference: {ref_label})")


# backward compat
def parse_args():
    return CalcRmsd().build_parser().parse_args()


def main():
    CalcRmsd().main()


if __name__ == '__main__':
    main()
