"""
Fragment Database for Molecular Substitution and High-Throughput Scopes.

Manages curated 3D fragment structures with anchor and deletion indices,
supports auto-registration from SMILES, and provides scope presets
(halogens, alkyls, aryls, ligands, EDG, EWG, steric series).

Andrea Pellegrini, 2026
"""
import json
import os
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Optional, List, Any
import numpy as np
from ase import Atoms
from ase.io import read

from frogchem.graph_theory import create_graph_matrix
from frogchem.chemistry import get_atomic_radius


@dataclass
class FragmentEntry:
    name: str
    filename: str
    frag_pair: Tuple[int, int]
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    smiles: str = ""


# Pre-curated library of standard chemical fragments (SMILES with '*' as attachment dummy atom)
STANDARD_LIBRARY: Dict[str, Dict[str, Any]] = {
    # Halogens
    'fluoro': {'smiles': '*F', 'category': 'halogens', 'tags': ['halogen', 'ewg'], 'desc': 'Fluoro group (-F)'},
    'chloro': {'smiles': '*Cl', 'category': 'halogens', 'tags': ['halogen', 'ewg'], 'desc': 'Chloro group (-Cl)'},
    'bromo': {'smiles': '*Br', 'category': 'halogens', 'tags': ['halogen', 'ewg'], 'desc': 'Bromo group (-Br)'},
    'iodo': {'smiles': '*I', 'category': 'halogens', 'tags': ['halogen'], 'desc': 'Iodo group (-I)'},
    
    # Alkyls
    'methyl': {'smiles': '*C', 'category': 'alkyls', 'tags': ['alkyl', 'edg'], 'desc': 'Methyl group (-Me)'},
    'ethyl': {'smiles': '*CC', 'category': 'alkyls', 'tags': ['alkyl', 'edg'], 'desc': 'Ethyl group (-Et)'},
    'n-propyl': {'smiles': '*CCC', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'n-Propyl group (-nPr)'},
    'isopropyl': {'smiles': '*C(C)C', 'category': 'alkyls', 'tags': ['alkyl', 'bulky', 'edg'], 'desc': 'Isopropyl group (-iPr)'},
    'cyclopropyl': {'smiles': '*C1CC1', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'Cyclopropyl group (-cPr)'},
    'n-butyl': {'smiles': '*CCCC', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'n-Butyl group (-nBu)'},
    'isobutyl': {'smiles': '*CC(C)C', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'Isobutyl group (-iBu)'},
    'sec-butyl': {'smiles': '*C(C)CC', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'sec-Butyl group (-sBu)'},
    'tert-butyl': {'smiles': '*C(C)(C)C', 'category': 'alkyls', 'tags': ['alkyl', 'bulky', 'edg'], 'desc': 'tert-Butyl group (-tBu)'},
    'cyclohexyl': {'smiles': '*C1CCCCC1', 'category': 'alkyls', 'tags': ['alkyl', 'bulky'], 'desc': 'Cyclohexyl group (-Cy)'},
    'cyclopentyl': {'smiles': '*C1CCCC1', 'category': 'alkyls', 'tags': ['alkyl'], 'desc': 'Cyclopentyl group (-Cp)'},
    'adamantyl': {'smiles': '*C12CC3CC(CC(C3)C1)C2', 'category': 'alkyls', 'tags': ['alkyl', 'bulky'], 'desc': '1-Adamantyl group (-1-Ad)'},
    'trifluoromethyl': {'smiles': '*C(F)(F)F', 'category': 'haloalkyls', 'tags': ['haloalkyl', 'ewg', 'fluorinated'], 'desc': 'Trifluoromethyl group (-CF3)'},
    'difluoromethyl': {'smiles': '*C(F)F', 'category': 'haloalkyls', 'tags': ['haloalkyl', 'ewg', 'fluorinated'], 'desc': 'Difluoromethyl group (-CHF2)'},
    'trichloromethyl': {'smiles': '*C(Cl)(Cl)Cl', 'category': 'haloalkyls', 'tags': ['haloalkyl', 'ewg'], 'desc': 'Trichloromethyl group (-CCl3)'},

    # Aryls
    'phenyl': {'smiles': '*c1ccccc1', 'category': 'aryls', 'tags': ['aryl', 'aromatic'], 'desc': 'Phenyl group (-Ph)'},
    'p-tolyl': {'smiles': '*c1ccc(C)cc1', 'category': 'aryls', 'tags': ['aryl', 'edg', 'para'], 'desc': 'p-Tolyl group (-4-Me-Ph)'},
    'o-tolyl': {'smiles': '*c1ccccc1C', 'category': 'aryls', 'tags': ['aryl', 'ortho', 'bulky'], 'desc': 'o-Tolyl group (-2-Me-Ph)'},
    'm-tolyl': {'smiles': '*c1cccc(C)c1', 'category': 'aryls', 'tags': ['aryl', 'meta'], 'desc': 'm-Tolyl group (-3-Me-Ph)'},
    'xylyl': {'smiles': '*c1c(C)cccc1C', 'category': 'aryls', 'tags': ['aryl', 'ortho', 'bulky', '2,6-disubstituted'], 'desc': '2,6-Xylyl group (-2,6-Me2-Ph)'},
    'mesityl': {'smiles': '*c1c(C)cc(C)cc1C', 'category': 'aryls', 'tags': ['aryl', 'bulky'], 'desc': 'Mesityl group (-2,4,6-Me3-Ph)'},
    'Dipp': {'smiles': '*c1c(C(C)C)cccc1C(C)C', 'category': 'aryls', 'tags': ['aryl', 'bulky', 'ligand', '2,6-disubstituted'], 'desc': '2,6-Diisopropylphenyl group (-Dipp)'},
    'p-anisyl': {'smiles': '*c1ccc(OC)cc1', 'category': 'aryls', 'tags': ['aryl', 'edg', 'para'], 'desc': 'p-Anisyl group (-4-OMe-Ph)'},
    'p-fluorophenyl': {'smiles': '*c1ccc(F)cc1', 'category': 'aryls', 'tags': ['aryl', 'para', 'fluorinated'], 'desc': '4-Fluorophenyl group (-4-F-Ph)'},
    'p-chlorophenyl': {'smiles': '*c1ccc(Cl)cc1', 'category': 'aryls', 'tags': ['aryl', 'para'], 'desc': '4-Chlorophenyl group (-4-Cl-Ph)'},
    'p-bromophenyl': {'smiles': '*c1ccc(Br)cc1', 'category': 'aryls', 'tags': ['aryl', 'para'], 'desc': '4-Bromophenyl group (-4-Br-Ph)'},
    'p-CF3-Ph': {'smiles': '*c1ccc(C(F)(F)F)cc1', 'category': 'aryls', 'tags': ['aryl', 'ewg', 'para', 'fluorinated'], 'desc': '4-Trifluoromethylphenyl group'},
    'p-cyanophenyl': {'smiles': '*c1ccc(C#N)cc1', 'category': 'aryls', 'tags': ['aryl', 'ewg', 'para'], 'desc': '4-Cyanophenyl group (-4-CN-Ph)'},
    'p-nitrophenyl': {'smiles': '*c1ccc([N+](=O)[O-])cc1', 'category': 'aryls', 'tags': ['aryl', 'ewg', 'para'], 'desc': '4-Nitrophenyl group (-4-NO2-Ph)'},
    '3,5-bis(CF3)phenyl': {'smiles': '*c1cc(C(F)(F)F)cc(C(F)(F)F)c1', 'category': 'aryls', 'tags': ['aryl', 'ewg', 'bulky', 'fluorinated'], 'desc': '3,5-Bis(trifluoromethyl)phenyl (ArF)'},
    '1-naphthyl': {'smiles': '*c1cccc2ccccc12', 'category': 'aryls', 'tags': ['aryl', 'bulky'], 'desc': '1-Naphthyl group'},
    '2-naphthyl': {'smiles': '*c1ccc2ccccc2c1', 'category': 'aryls', 'tags': ['aryl'], 'desc': '2-Naphthyl group'},

    # Heteroaryls
    '2-pyridyl': {'smiles': '*c1ccccn1', 'category': 'heteroaryls', 'tags': ['heteroaryl', 'ligand', 'directing_group'], 'desc': '2-Pyridyl group'},
    '3-pyridyl': {'smiles': '*c1cccnc1', 'category': 'heteroaryls', 'tags': ['heteroaryl'], 'desc': '3-Pyridyl group'},
    '4-pyridyl': {'smiles': '*c1ccncc1', 'category': 'heteroaryls', 'tags': ['heteroaryl'], 'desc': '4-Pyridyl group'},
    '2-thienyl': {'smiles': '*c1sccc1', 'category': 'heteroaryls', 'tags': ['heteroaryl'], 'desc': '2-Thienyl group'},
    '2-furyl': {'smiles': '*c1occc1', 'category': 'heteroaryls', 'tags': ['heteroaryl'], 'desc': '2-Furyl group'},

    # Oxygen / Sulfur
    'hydroxy': {'smiles': '*O', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'edg', 'protic'], 'desc': 'Hydroxy group (-OH)'},
    'methoxy': {'smiles': '*OC', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'edg'], 'desc': 'Methoxy group (-OMe)'},
    'ethoxy': {'smiles': '*OCC', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'edg'], 'desc': 'Ethoxy group (-OEt)'},
    'isopropoxy': {'smiles': '*OC(C)C', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'edg'], 'desc': 'Isopropoxy group (-OiPr)'},
    'tert-butoxy': {'smiles': '*OC(C)(C)C', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'bulky', 'edg'], 'desc': 'tert-Butoxy group (-OtBu)'},
    'trifluoromethoxy': {'smiles': '*OC(F)(F)F', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'ewg', 'fluorinated'], 'desc': 'Trifluoromethoxy group (-OCF3)'},
    'phenoxy': {'smiles': '*Oc1ccccc1', 'category': 'oxygen_sulfur', 'tags': ['oxygen', 'aryl'], 'desc': 'Phenoxy group (-OPh)'},
    'mercapto': {'smiles': '*S', 'category': 'oxygen_sulfur', 'tags': ['sulfur'], 'desc': 'Mercapto / Thiol group (-SH)'},
    'thiomethyl': {'smiles': '*SC', 'category': 'oxygen_sulfur', 'tags': ['sulfur'], 'desc': 'Thiomethyl group (-SMe)'},

    # Nitrogen
    'amino': {'smiles': '*N', 'category': 'nitrogen', 'tags': ['nitrogen', 'edg', 'protic'], 'desc': 'Amino group (-NH2)'},
    'dimethylamino': {'smiles': '*N(C)C', 'category': 'nitrogen', 'tags': ['nitrogen', 'edg'], 'desc': 'Dimethylamino group (-NMe2)'},
    'diethylamino': {'smiles': '*N(CC)CC', 'category': 'nitrogen', 'tags': ['nitrogen', 'edg'], 'desc': 'Diethylamino group (-NEt2)'},
    'nitro': {'smiles': '*[N+](=O)[O-]', 'category': 'nitrogen', 'tags': ['nitrogen', 'ewg'], 'desc': 'Nitro group (-NO2)'},
    'acetylamino': {'smiles': '*NC(=O)C', 'category': 'nitrogen', 'tags': ['nitrogen', 'amide'], 'desc': 'Acetylamino group (-NHAc)'},

    # Carbonyl / Carboxy / Nitrile
    'cyano': {'smiles': '*C#N', 'category': 'carbonyl_nitrile', 'tags': ['nitrile', 'ewg'], 'desc': 'Cyano group (-CN)'},
    'formyl': {'smiles': '*C=O', 'category': 'carbonyl_nitrile', 'tags': ['carbonyl', 'ewg'], 'desc': 'Formyl group (-CHO)'},
    'acetyl': {'smiles': '*C(=O)C', 'category': 'carbonyl_nitrile', 'tags': ['carbonyl', 'ewg'], 'desc': 'Acetyl group (-Ac)'},
    'carboxy': {'smiles': '*C(=O)O', 'category': 'carbonyl_nitrile', 'tags': ['carboxy', 'ewg', 'protic'], 'desc': 'Carboxy group (-COOH)'},
    'methoxycarbonyl': {'smiles': '*C(=O)OC', 'category': 'carbonyl_nitrile', 'tags': ['ester', 'ewg'], 'desc': 'Methoxycarbonyl group (-COOMe)'},

    # Phosphorus / Ligands
    'PPh2': {'smiles': '*P(c1ccccc1)c2ccccc2', 'category': 'phosphorus', 'tags': ['phosphine', 'ligand', 'bulky', 'aryl'], 'desc': 'Diphenylphosphino group (-PPh2)'},
    'PtBu2': {'smiles': '*P(C(C)(C)C)C(C)(C)C', 'category': 'phosphorus', 'tags': ['phosphine', 'ligand', 'bulky', 'alkyl'], 'desc': 'Di-tert-butylphosphino group (-PtBu2)'},
    'PCy2': {'smiles': '*P(C1CCCCC1)C2CCCCC2', 'category': 'phosphorus', 'tags': ['phosphine', 'ligand', 'bulky', 'alkyl'], 'desc': 'Dicyclohexylphosphino group (-PCy2)'},
    'PMe2': {'smiles': '*P(C)C', 'category': 'phosphorus', 'tags': ['phosphine', 'ligand', 'alkyl'], 'desc': 'Dimethylphosphino group (-PMe2)'},
    'PEt2': {'smiles': '*P(CC)CC', 'category': 'phosphorus', 'tags': ['phosphine', 'ligand', 'alkyl'], 'desc': 'Diethylphosphino group (-PEt2)'},

    # Silicon & Boron
    'TMS': {'smiles': '*[Si](C)(C)C', 'category': 'silicon', 'tags': ['silyl', 'bulky', 'protecting_group'], 'desc': 'Trimethylsilyl group (-TMS)'},
    'TIPS': {'smiles': '*[Si](C(C)C)(C(C)C)C(C)C', 'category': 'silicon', 'tags': ['silyl', 'bulky', 'protecting_group'], 'desc': 'Triisopropylsilyl group (-TIPS)'},
    'Bpin': {'smiles': '*B1OC(C)(C)C(C)(C)O1', 'category': 'boron', 'tags': ['boryl', 'ligand', 'coupling'], 'desc': 'Pinacolatoboryl group (-Bpin)'},
}

# Pre-defined scope sets for high-throughput screening
SCOPE_PRESETS: Dict[str, List[str]] = {
    "halogens": ["fluoro", "chloro", "bromo", "iodo"],
    "alkyls": ["methyl", "ethyl", "n-propyl", "isopropyl", "cyclopropyl", "n-butyl", "isobutyl", "sec-butyl", "tert-butyl", "cyclohexyl", "cyclopentyl", "adamantyl"],
    "haloalkyls": ["trifluoromethyl", "difluoromethyl", "trichloromethyl"],
    "aryls": ["phenyl", "p-tolyl", "o-tolyl", "m-tolyl", "xylyl", "mesityl", "Dipp", "p-anisyl", "p-fluorophenyl", "p-chlorophenyl", "p-bromophenyl", "p-CF3-Ph", "p-cyanophenyl", "p-nitrophenyl", "3,5-bis(CF3)phenyl", "1-naphthyl", "2-naphthyl"],
    "heteroaryls": ["2-pyridyl", "3-pyridyl", "4-pyridyl", "2-thienyl", "2-furyl"],
    "oxygen": ["hydroxy", "methoxy", "ethoxy", "isopropoxy", "tert-butoxy", "trifluoromethoxy", "phenoxy"],
    "sulfur": ["mercapto", "thiomethyl"],
    "nitrogen": ["amino", "dimethylamino", "diethylamino", "nitro", "acetylamino"],
    "carbonyl": ["formyl", "acetyl", "carboxy", "methoxycarbonyl", "cyano"],
    "phosphines": ["PPh2", "PtBu2", "PCy2", "PMe2", "PEt2"],
    "ligands": ["PPh2", "PtBu2", "PCy2", "PMe2", "PEt2", "2-pyridyl", "TMS", "TIPS", "Bpin"],
    "edg": ["methyl", "ethyl", "isopropyl", "tert-butyl", "methoxy", "hydroxy", "dimethylamino", "diethylamino", "p-tolyl", "p-anisyl"],
    "ewg": ["fluoro", "chloro", "bromo", "trifluoromethyl", "nitro", "cyano", "formyl", "acetyl", "carboxy", "methoxycarbonyl", "p-CF3-Ph", "p-nitrophenyl", "p-cyanophenyl", "3,5-bis(CF3)phenyl"],
    "steric": ["methyl", "isopropyl", "tert-butyl", "cyclohexyl", "adamantyl", "xylyl", "mesityl", "Dipp", "3,5-bis(CF3)phenyl", "PPh2", "PtBu2", "PCy2", "TIPS"],
}


class FragmentDatabase:
    """
    Database of fragments for molecular functionalization and scope studies.
    """

    def __init__(self, base_dir: Optional[Path] = None, auto_init_standard: bool = True):
        if base_dir is None:
            self.base_dir = Path(__file__).parent / "data"
        else:
            self.base_dir = Path(base_dir)
            
        self.structures_dir = self.base_dir / "structures"
        self.db_file = self.base_dir / "library.json"

        self.structures_dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict[str, dict] = self._load()
        
        # Populate standard library if empty or new
        if auto_init_standard and (len(self.data) < len(STANDARD_LIBRARY)):
            self.ensure_standard_library()

    def _load(self) -> dict:
        if self.db_file.exists():
            try:
                with open(self.db_file, 'r') as f:
                    loaded = json.load(f)
                    for k, v in loaded.items():
                        v.setdefault('category', 'general')
                        v.setdefault('tags', [])
                        v.setdefault('smiles', '')
                    return loaded
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def _get_graph_spectrum(self, atoms: Atoms) -> np.ndarray:
        """Calculates rotation-invariant graph spectrum."""
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        unique_syms = set(symbols)
        radii = {k: get_atomic_radius(k) for k in unique_syms}
        adj_matrix = create_graph_matrix(symbols, positions, radii, buffer=0.3)
        return np.sort(np.linalg.eigvalsh(adj_matrix))

    def find_fragment_by_structure(self, query_atoms: Atoms) -> Optional[str]:
        """Checks if structure exists in DB."""
        query_formula = query_atoms.get_chemical_formula()
        query_spectrum = self._get_graph_spectrum(query_atoms)

        for name, entry in self.data.items():
            try:
                candidate_path = self.structures_dir / entry['filename']
                if not candidate_path.exists(): 
                    continue
                candidate_atoms = read(candidate_path)
            except Exception:
                continue

            if candidate_atoms.get_chemical_formula() != query_formula:
                continue

            candidate_spectrum = self._get_graph_spectrum(candidate_atoms)
            if np.allclose(query_spectrum, candidate_spectrum, atol=1e-4):
                return name
        return None

    def add_fragment(
        self, 
        name: str, 
        source_xyz: str, 
        delete_idx: int, 
        attach_idx: int, 
        desc: str = "",
        category: str = "general",
        tags: Optional[List[str]] = None,
        smiles: str = ""
    ) -> str:
        """Adds a fragment from an existing XYZ file."""
        src_path = Path(source_xyz)
        if not src_path.exists():
            raise FileNotFoundError(f"Source file {source_xyz} not found.")

        base_name = name
        counter = 1
        while name in self.data and not self._is_stored_file(name, src_path):
            name = f"{base_name}_{counter}"
            counter += 1

        dest_filename = f"{name}.xyz"
        dest_path = self.structures_dir / dest_filename
        
        # Avoid copying to itself if src and dest are identical
        if src_path.resolve() != dest_path.resolve():
            shutil.copy2(src_path, dest_path)

        entry = FragmentEntry(
            name=name,
            filename=dest_filename,
            frag_pair=(delete_idx, attach_idx),
            description=desc,
            category=category,
            tags=tags or [],
            smiles=smiles
        )
        
        self.data[name] = asdict(entry)
        self._save()
        return name

    def _is_stored_file(self, name: str, path: Path) -> bool:
        """True if `path` is the XYZ file already stored for fragment `name`."""
        stored = self.structures_dir / self.data[name].get('filename', f"{name}.xyz")
        return stored.exists() and stored.samefile(path)

    def add_from_smiles(
        self,
        name: str,
        smiles: str,
        desc: str = "",
        category: str = "general",
        tags: Optional[List[str]] = None,
        overwrite: bool = True
    ) -> str:
        """
        Creates and registers a fragment directly from SMILES containing a dummy atom '*'
        as the attachment handle (e.g. '*c1ccccc1' for phenyl).
        Generates 3D conformer with RDKit ETKDGv3 and MMFF/UFF minimization.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from frogchem.io import write_xyz

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}'")

        dummy_idx = None
        attach_idx = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0 or atom.GetSymbol() == '*':
                dummy_idx = atom.GetIdx()
                neighbors = atom.GetNeighbors()
                if not neighbors:
                    raise ValueError(f"Dummy atom in '{smiles}' has no neighbors to attach to.")
                attach_idx = neighbors[0].GetIdx()
                atom.SetAtomicNum(1)  # temporarily set to H for 3D embedding
                break

        if dummy_idx is None:
            raise ValueError("No dummy atom '*' found in SMILES. Specify attachment site with '*' (e.g. '*C(=O)C').")

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except Exception:
                pass

        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        dest_filename = f"{name}.xyz"
        dest_path = self.structures_dir / dest_filename
        write_xyz(str(dest_path), [symbols], [positions])

        entry = FragmentEntry(
            name=name,
            filename=dest_filename,
            frag_pair=(dummy_idx, attach_idx),
            description=desc,
            category=category,
            tags=tags or [],
            smiles=smiles
        )

        self.data[name] = asdict(entry)
        self._save()
        return name

    def ensure_standard_library(self):
        """Builds and caches all curated standard fragments in the database."""
        for name, spec in STANDARD_LIBRARY.items():
            xyz_path = self.structures_dir / f"{name}.xyz"
            if name not in self.data or not xyz_path.exists():
                try:
                    self.add_from_smiles(
                        name=name,
                        smiles=spec['smiles'],
                        desc=spec['desc'],
                        category=spec['category'],
                        tags=spec['tags']
                    )
                except Exception:
                    pass

    def update_fragment(
        self, 
        name: str, 
        new_name: Optional[str] = None, 
        new_pair: Optional[Tuple[int, int]] = None, 
        new_desc: Optional[str] = None,
        new_category: Optional[str] = None,
        new_tags: Optional[List[str]] = None
    ):
        """Updates an existing fragment entry."""
        if name not in self.data:
            raise ValueError(f"Fragment '{name}' not found.")

        entry_data = self.data[name]
        updated = False

        if new_desc is not None:
            entry_data['description'] = new_desc
            updated = True

        if new_pair is not None:
            entry_data['frag_pair'] = new_pair
            updated = True

        if new_category is not None:
            entry_data['category'] = new_category
            updated = True

        if new_tags is not None:
            entry_data['tags'] = new_tags
            updated = True

        if new_name and new_name != name:
            if new_name in self.data:
                raise ValueError(f"Cannot rename: Fragment '{new_name}' already exists.")
            old_path = self.structures_dir / entry_data['filename']
            new_filename = f"{new_name}.xyz"
            new_path = self.structures_dir / new_filename
            if old_path.exists():
                if new_path.exists() and not new_path.samefile(old_path):
                    raise ValueError(f"Cannot rename: file {new_path} already exists.")
                old_path.rename(new_path)
            entry_data['filename'] = new_filename
            entry_data['name'] = new_name
            self.data[new_name] = entry_data
            del self.data[name]
            updated = True

        if updated:
            self._save()

    def get_fragment(self, name: str) -> Optional[Tuple[str, Tuple[int, int]]]:
        """Returns (filepath, (delete_idx, attach_idx)) for a given fragment name."""
        if name not in self.data:
            # Check case-insensitive match
            for k in self.data:
                if k.lower() == name.lower():
                    name = k
                    break
            else:
                return None
        
        entry = self.data[name]
        full_path = self.structures_dir / entry['filename']
        
        if not full_path.exists():
            # If structure is in STANDARD_LIBRARY, regenerate it
            if name in STANDARD_LIBRARY:
                self.add_from_smiles(
                    name=name,
                    smiles=STANDARD_LIBRARY[name]['smiles'],
                    desc=STANDARD_LIBRARY[name]['desc'],
                    category=STANDARD_LIBRARY[name]['category'],
                    tags=STANDARD_LIBRARY[name]['tags']
                )
            else:
                raise FileNotFoundError(f"XYZ file for '{name}' missing at {self.structures_dir}")

        return str(full_path), tuple(entry['frag_pair'])

    def list_fragments(self, category: Optional[str] = None, tag: Optional[str] = None) -> List[str]:
        """Lists available fragment names, optionally filtered by category or tag."""
        names = []
        for name, entry in self.data.items():
            if category and entry.get('category') != category:
                continue
            if tag and tag not in entry.get('tags', []):
                continue
            names.append(name)
        return names

    def get_scope_presets(self) -> Dict[str, List[str]]:
        """Returns the dictionary of predefined scope presets."""
        return SCOPE_PRESETS

    def resolve_scope(self, scope_name_or_list: str) -> List[str]:
        """
        Resolves a scope string into a list of fragment names.
        Examples:
            'halogens' -> ['fluoro', 'chloro', 'bromo', 'iodo']
            'alkyls' -> list of alkyl names
            'all' -> all fragments in DB
            'methyl,ethyl,phenyl' -> ['methyl', 'ethyl', 'phenyl']
        """
        key = scope_name_or_list.lower().strip()
        if key == "all":
            return self.list_fragments()
        if key in SCOPE_PRESETS:
            return SCOPE_PRESETS[key]
        if "," in scope_name_or_list:
            return [x.strip() for x in scope_name_or_list.split(",") if x.strip()]
        # Check if it matches a category
        by_cat = self.list_fragments(category=key)
        if by_cat:
            return by_cat
        # Check if it matches a tag
        by_tag = self.list_fragments(tag=key)
        if by_tag:
            return by_tag
        return [scope_name_or_list]

    def delete_fragment(self, name: str):
        if name not in self.data:
            raise ValueError(f"Fragment '{name}' not found.")
        file_path = self.structures_dir / self.data[name]['filename']
        if file_path.exists():
            os.remove(file_path)
        del self.data[name]
        self._save()
