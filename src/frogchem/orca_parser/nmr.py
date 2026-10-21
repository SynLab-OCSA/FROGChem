"""
NMR Analysis module for ORCA output files.

Parses Chemical Shieldings (ppm) and Spin-Spin Couplings (J, Hz),
and computes interatomic distance matrices for NOE estimation.
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


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

    def get_isotopes(self) -> Dict[int, str]:
        """Return dictionary mapping 0-based atom indices to detected isotopes (e.g. {0: '13C', 9: '17O'})."""
        return self.atom_isotopes

    def get_gn(self) -> Dict[int, float]:
        """Return dictionary mapping 0-based atom indices to nuclear g-factors."""
        return self.atom_gn

    @property
    def shifts_df(self) -> pd.DataFrame:
        """DataFrame representation of all chemical shifts."""
        return self.get_shifts_df()

    def get_shifts_df(self, atoms: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Return a pandas DataFrame of chemical shifts.
        
        Args:
            atoms: List of 0-based atom indices to filter.
        """
        selected = sorted(self.shifts.keys()) if atoms is None else [a for a in atoms if a in self.shifts]
        data = []
        for idx in selected:
            d = self.shifts[idx]
            data.append({
                'Index_0': idx,
                'Index_1': idx + 1,
                'Element': d['element'],
                'Isotropic': d['iso'],
                'Anisotropy': d['aniso']
            })
        return pd.DataFrame(data)

    def get_shifts_data(self, atoms: Optional[List[int]] = None) -> Tuple[List[List[Any]], List[str]]:
        """
        Get table data and headers for chemical shifts.
        Output uses 1-based indexing for display (like NBO).
        """
        selected = sorted(self.shifts.keys()) if atoms is None else [a for a in atoms if a in self.shifts]
        headers = ['Atom', 'Idx (1-based)', 'Element', 'Isotropic (ppm)', 'Anisotropy (ppm)']
        rows = []
        for idx in selected:
            d = self.shifts[idx]
            el = d['element']
            rows.append([f"{el}{idx+1}", idx + 1, el, d['iso'], d['aniso']])
        return rows, headers

    def get_shifts_table(self, atoms: Optional[List[int]] = None) -> str:
        """Return formatted string table of chemical shifts."""
        rows, headers = self.get_shifts_data(atoms)
        if not rows:
            return "No chemical shifts found."
        col_w = [12, 14, 10, 18, 18]
        header_line = " | ".join(f"{h:<{col_w[i]}}" for i, h in enumerate(headers))
        sep = "-" * len(header_line)
        lines = [header_line, sep]
        for r in rows:
            lines.append(f"{r[0]:<{col_w[0]}} | {r[1]:<{col_w[1]}} | {r[2]:<{col_w[2]}} | {r[3]:<{col_w[3]}.3f} | {r[4]:<{col_w[4]}.3f}")
        return "\n".join(lines)

    def get_j_df(self, atoms: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Return a pandas DataFrame representing the square J-coupling matrix.
        Index and column labels are 1-based strings (e.g. 'H24', 'H25').
        """
        if atoms is None:
            raw_atoms = sorted(list(set(k[0] for k in self.j_couplings.keys())))
        else:
            raw_atoms = list(atoms)

        labels = []
        for idx in raw_atoms:
            el = self.atoms[idx] if idx < len(self.atoms) else self.shifts.get(idx, {}).get('element', 'X')
            labels.append(f"{el}{idx+1}")

        matrix = np.zeros((len(raw_atoms), len(raw_atoms)), dtype=float)
        for i, idx1 in enumerate(raw_atoms):
            for j, idx2 in enumerate(raw_atoms):
                if idx1 == idx2:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = self.j_couplings.get((idx1, idx2), 0.0)

        return pd.DataFrame(matrix, index=labels, columns=labels)

    def get_j_data(self, atoms: Optional[List[int]] = None) -> Tuple[List[List[Any]], List[str]]:
        """
        Get table data and headers for the J-coupling matrix.
        Headers and row labels are 1-based.
        """
        df = self.get_j_df(atoms)
        headers = ['Atom'] + list(df.columns)
        rows = []
        for idx, row in df.iterrows():
            rows.append([str(idx)] + [float(v) for v in row.values])
        return rows, headers

    def get_j_matrix(self, atoms: Optional[List[int]] = None) -> str:
        """Return formatted string matrix of J-couplings."""
        rows, headers = self.get_j_data(atoms)
        if len(headers) <= 1:
            return "No J-couplings found."
        col_width = 10
        lines = ["".join(f"{h:>{col_width}}" for h in headers)]
        for r in rows:
            line_str = f"{r[0]:>{col_width}}" + "".join(f"{v:>{col_width}.3f}" for v in r[1:])
            lines.append(line_str)
        return "\n".join(lines)

    def get_distances_df(self, atoms: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Return a pandas DataFrame representing the mutual distances (in Angstrom).
        Useful for NOE estimation.
        Labels are 1-based (e.g. 'H24').
        """
        if atoms is None:
            raw_atoms = list(range(len(self.atoms)))
        else:
            raw_atoms = list(atoms)

        labels = []
        for idx in raw_atoms:
            el = self.atoms[idx] if idx < len(self.atoms) else self.shifts.get(idx, {}).get('element', 'X')
            labels.append(f"{el}{idx+1}")

        n = len(raw_atoms)
        matrix = np.zeros((n, n), dtype=float)
        for i, idx1 in enumerate(raw_atoms):
            for j, idx2 in enumerate(raw_atoms):
                if idx1 == idx2:
                    matrix[i, j] = 0.0
                elif len(self.geom) > max(idx1, idx2):
                    dist = float(np.linalg.norm(self.geom[idx1] - self.geom[idx2]))
                    matrix[i, j] = dist
        return pd.DataFrame(matrix, index=labels, columns=labels)

    def get_distance_data(self, atoms: Optional[List[int]] = None) -> Tuple[List[List[Any]], List[str]]:
        """Get table data and headers for mutual distances in Angstrom."""
        df = self.get_distances_df(atoms)
        headers = ['Atom'] + list(df.columns)
        rows = []
        for idx, row in df.iterrows():
            rows.append([str(idx)] + [float(v) for v in row.values])
        return rows, headers

    def get_distances(self, atoms: Optional[List[int]] = None) -> str:
        """Return formatted string matrix of mutual distances in Angstrom."""
        rows, headers = self.get_distance_data(atoms)
        if len(headers) <= 1:
            return "No atomic distances available."
        col_width = 10
        lines = ["".join(f"{h:>{col_width}}" for h in headers)]
        for r in rows:
            line_str = f"{r[0]:>{col_width}}" + "".join(f"{v:>{col_width}.3f}" for v in r[1:])
            lines.append(line_str)
        return "\n".join(lines)
