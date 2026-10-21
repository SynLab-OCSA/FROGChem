"""Find stereocenters and their CIP absolute configuration (R/S) from 3D geometries."""

from frogchem.exceptions import InputError
from frogchem.filetypes import load_structures
from frogchem.symmetry import get_absolute_configuration
from frogchem.printer import print_info
from frogchem.cli.base import CLICommand, Guide, Output


def _rdkit_mol(atoms, geom, charge: int = 0):
    """RDKit molecule with bonds perceived from the 3D coordinates."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds
    except ImportError as e:
        raise InputError("frog-stereocenters requires RDKit (pip install rdkit)") from e

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, (symbol, xyz) in enumerate(zip(atoms, geom)):
        mol.AddAtom(Chem.Atom(str(symbol)))
        conf.SetAtomPosition(i, tuple(float(c) for c in xyz))
    mol = mol.GetMol()
    mol.AddConformer(conf, assignId=True)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


class Stereocenters(CLICommand, name='frog-stereocenters'):
    """Find stereocenters and their CIP absolute configuration (R/S) from 3D geometries.

    Bonds are perceived from the coordinates with RDKit (rdDetermineBonds); every
    structure of a multi-XYZ file is analysed. Unassignable centres are reported as '?'.
    """
    category = 'geometry'
    outputs = [
        Output('table', 'Stereocenters (<file>)',
               'One row per stereocenter: atom index (0-based), element, R/S '
               '(plus a 0-based Conf column for multi-structure files)'),
    ]
    guide = Guide(
        why="""
            Check the absolute configuration (*R*/*S*, CIP rules) of every stereocentre directly from 3D
            coordinates. Use it to make sure that a conformer search, an optimisation or a substitution did
            not invert a centre, that every conformer of an ensemble is the same stereoisomer, or that the
            structure you computed is the enantiomer you think it is (e.g. before comparing a computed ECD or
            optical rotation with experiment).
            """,
        needs="""
            * Geometries: `.xyz` (multi-structure allowed: every conformer is checked), ORCA or Gaussian outputs
              (last geometry).
            * The **total charge** of the molecule if it is not neutral (`--charge`): bonds and bond orders are
              deduced from the coordinates (RDKit), and this needs the right number of electrons.
            * RDKit installed (`pip install rdkit`).
            * Works for **organic molecules**. Structures containing metals (e.g. Ag, Ru, Pd complexes) usually
              fail with *"has no valences defined"*, because bond orders cannot be assigned to them.
            """,
        how=[
            "Run `frog-stereocenters opt.xyz` (add `--charge -1`, `--charge 1`, ... for ions).",
            "Read the table: atom index (**0-based**), element and configuration (R, S, or `?`).",
            "For an ensemble (`frog-stereocenters crest_conformers.xyz`) a *Conf* column is added: check that "
            "every conformer has the same labels; `--format csv` makes this easy to filter.",
        ],
        expect="""
            One table per file with one row per stereocentre. Example: `nmr_iPr.xyz` gives C0 = **S** and
            C2 = **R**; its mirror image (`frog-mirror`) gives C0 = R and C2 = S. A molecule without
            stereocentres prints *"No stereocenters found in ..."* and no table.

            * `?` means RDKit recognises a potential stereocentre but cannot assign it (e.g. two substituents it
              cannot rank, or a centre that is only stereogenic because of another one).
            * *"Bond perception failed ... (check --charge)"* followed by an RDKit message about the molecular
              charge or *"could not find valid bond ordering"*: the charge is wrong (or the geometry is
              distorted) — give the right `--charge`. With *"has no valences defined"* the structure contains a
              metal and cannot be analysed with this command; analyse the free ligand instead (delete the metal
              in a viewer), keeping in mind that its CIP priorities may differ from those in the complex.
            * Only stereocentres (atoms) are reported: E/Z double bonds and axial/planar chirality are not.
              Labels are CIP descriptors, so replacing a substituent can change R into S without any change
              in the 3D arrangement.
            """,
    )
    examples = [
        ('frog-stereocenters opt.xyz', 'R/S labels of the stereocenters of one structure'),
        ('frog-stereocenters crest_conformers.xyz --format csv', 'Check every conformer, CSV on stdout'),
    ]

    def setup_args(self, parser):
        parser.add_argument('file', nargs='+', help='XYZ (multi-structure allowed), ORCA or Gaussian files')
        parser.add_argument('--charge', type=int, default=0,
                            help='Total molecular charge, used for bond perception')

    def run(self, args):
        self.require_files(args.file)
        for f in args.file:
            structs = load_structures(f)
            if not structs:
                raise InputError(f"No structures found in {f}")
            multi = len(structs) > 1
            rows = []
            for n, (atoms, geom) in enumerate(structs):
                try:
                    mol = _rdkit_mol(atoms, geom, args.charge)
                except InputError:
                    raise
                except Exception as e:
                    where = f"{f} (conf {n})" if multi else f
                    raise InputError(f"Bond perception failed for {where}: {e} (check --charge)") from e
                for atom_idx, label in get_absolute_configuration(mol):
                    row = [atom_idx, mol.GetAtomWithIdx(atom_idx).GetSymbol(), label]
                    rows.append([n, *row] if multi else row)

            if not rows:
                print_info(f"No stereocenters found in {f}")
                continue
            headers = ['Atom', 'Element', 'Configuration']
            self.output(rows, headers=['Conf', *headers] if multi else headers, title=f"Stereocenters ({f})")


def parse_args():
    return Stereocenters().build_parser().parse_args()


def main():
    Stereocenters().main()


if __name__ == '__main__':
    main()
