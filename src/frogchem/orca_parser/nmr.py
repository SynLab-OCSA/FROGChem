"""
NMR Analysis module for ORCA output files.

Parses Chemical Shieldings (ppm) and Spin-Spin Couplings (J, Hz),
and computes interatomic distance matrices for NOE estimation.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Any


class NMRAnalysis:
    """
    Parses NMR Chemical Shieldings and Spin-Spin Couplings from ORCA output files.
    
    Attributes:
        shifts: Dict[int, Dict[str, Any]]
            Mapping of 0-based atom index to {'element': str, 'iso': float, 'aniso': float}.
        j_couplings: Dict[Tuple[int, int], float]
            Mapping of (idx1, idx2) 0-based pairs to J-coupling in Hz.
        atoms: List[str]
            Atom element symbols.
        geom: np.ndarray
            Cartesian coordinates of atoms in Angstrom.
    """

    def __init__(self, fl: str, geom: Any, atoms: List[str]):
        self.fl = fl
        self.geom = np.array(geom, dtype=float) if geom is not None else np.array([])
        self.atoms = atoms if atoms is not None else []
        self.shifts: Dict[int, Dict[str, Any]] = {}
        self.j_couplings: Dict[Tuple[int, int], float] = {}
        self.atom_isotopes: Dict[int, str] = {}
        self.atom_gn: Dict[int, float] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse chemical shifts and spin-spin couplings from file content."""
        # 1. Parse Chemical Shielding Summary
        shift_section = re.search(
            r'CHEMICAL SHIELDING SUMMARY \(ppm\).*?Nucleus\s+Element\s+Isotropic\s+Anisotropy\n.*?\n(.*?)(\n\n|\Z)',
            self.fl,
            re.DOTALL
        )
        if shift_section:
            lines = shift_section.group(1).strip().split('\n')
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # ORCA prints 0-based nucleus index in this section matching internal atom numbering
                        idx = int(parts[0])
                        el = parts[1]
                        iso = float(parts[2])
                        aniso = float(parts[3])
                        self.shifts[idx] = {
                            'element': el,
                            'iso': iso,
                            'aniso': aniso
                        }
                    except (ValueError, IndexError):
                        continue

        # 2. Parse Spin-Spin Coupling (J) and nuclear isotopes / g-factors
        # NUCLEUS A = C    0 NUCLEUS B = C    1 
        #  ( 13C  gnA =  1.405  13C  gnB =  1.405) r(AB) =     1.3906
        iso_pattern = r'NUCLEUS A =\s*[a-zA-Z]+\s*(\d+)\s*NUCLEUS B =\s*[a-zA-Z]+\s*(\d+)\s*\n\s*\(\s*(\d*[a-zA-Z]+)\s+gnA\s*=\s*([-\d\.]+)\s+(\d*[a-zA-Z]+)\s+gnB\s*=\s*([-\d\.]+)\)'
        iso_matches = re.findall(iso_pattern, self.fl)
        for m in iso_matches:
            try:
                idx1 = int(m[0])
                idx2 = int(m[1])
                self.atom_isotopes[idx1] = m[2]
                self.atom_gn[idx1] = float(m[3])
                self.atom_isotopes[idx2] = m[4]
                self.atom_gn[idx2] = float(m[5])
            except (ValueError, IndexError):
                continue

        # Enrich shift dict with detected isotope if available
        for idx in self.shifts:
            if idx in self.atom_isotopes:
                self.shifts[idx]['isotope'] = self.atom_isotopes[idx]

        j_pattern = r'NUCLEUS A =\s*[a-zA-Z]+\s*(\d+)\s*NUCLEUS B =\s*[a-zA-Z]+\s*(\d+)[\s\S]*?J\[\1,\2\]\(Total\)[\s\S]*?iso=\s*(-?\d+\.\d+)'
        j_matches = re.findall(j_pattern, self.fl)
        for m in j_matches:
            try:
                idx1 = int(m[0])
                idx2 = int(m[1])
                val = float(m[2])
                self.j_couplings[(idx1, idx2)] = val
                self.j_couplings[(idx2, idx1)] = val
            except (ValueError, IndexError):
                continue

