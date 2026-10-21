import re
import logging
from typing import Optional, Union, Sequence, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NBOAnalysis:
    def __init__(self, output_text: str, cycle: int = -1):
        """Parse NBO analysis from quantum chemical output text.

        Args:
            output_text: Output file text containing NBO calculation.
            cycle: NBO cycle to parse if multiple are present in text (e.g. geometry optimization).
                   Defaults to -1 (last, converged cycle). Set to 0 for first cycle.
        """
        self.raw_text = output_text
        self.cycle = cycle
        self.block_text = self._get_nbo_block(output_text, cycle=cycle)

        self.npa_charges = self._parse_npa(self.block_text)
        self.nbo_details = self._parse_nbo_details(self.block_text)
        self.perturbation_theory = self._parse_perturbation(self.block_text)
        self.wiberg_matrix, self.wiberg_df, self.wiberg_totals = self._parse_wiberg(self.block_text)

    @staticmethod
    def _get_nbo_block(text: str, cycle: int = -1) -> str:
        """Extract a specific NBO block from calculation text."""
        marker = "NATURAL BOND ORBITAL ANALYSIS:"
        parts = list(re.finditer(re.escape(marker), text))
        if not parts:
            return text
        if cycle < -len(parts) or cycle >= len(parts):
            cycle = -1

        norm_cycle = cycle if cycle >= 0 else len(parts) + cycle
        target = parts[norm_cycle]
        start_pos = target.start()

        # Delimit backward search window by preceding cycle boundary or start of file
        search_start = parts[norm_cycle - 1].end() if norm_cycle > 0 else 0
        npa_matches = list(re.finditer(r'Summary of Natural Population Analysis:', text[search_start:start_pos]))
        block_start = (search_start + npa_matches[-1].start()) if npa_matches else start_pos

        wib_matches = list(re.finditer(r'Wiberg bond index matrix in the NAO basis:', text[search_start:start_pos]))
        if wib_matches and (search_start + wib_matches[-1].start()) < block_start:
            block_start = search_start + wib_matches[-1].start()

        block_end = parts[norm_cycle + 1].start() if norm_cycle < len(parts) - 1 else len(text)
        return text[block_start:block_end]

    @staticmethod
    def _extract_atom_indices(text: str) -> list[int]:
        """Extract 0-based atom indices from atom label or orbital description."""
        clean_str = re.sub(r'\([^\)]*\)', '', text)
        matches = re.findall(r'[A-Z][a-z]?\s*(\d+)', clean_str)
        return [int(idx) - 1 for idx in matches if int(idx) > 0]

    _extract_atom_indices_from_label = _extract_atom_indices

    @staticmethod
    def _clean_label(raw_label: str) -> str:
        """Clean orbital label by removing hybridization and trailing numbers."""
        cleaned = re.split(r'[spdf]\(', raw_label)[0]
        return re.sub(r'\s+-?\d+\.\d+\s*$', '', cleaned).strip()

    @staticmethod
    def _is_coefficient_line(line: str) -> bool:
        """Check if line contains purely numerical expansion coefficients (no parentheses)."""
        stripped = line.strip()
        if not stripped or "(" in stripped or ")" in stripped:
            return False
        try:
            float(stripped.split()[0])
            return True
        except (ValueError, IndexError):
            return False

    def _parse_composition_line(self, line: str):
        """
        Parsa l'inizio di un blocco di composizione per un atomo.
        Es: ( 51.37%)   0.7167* C  1 s( 27.37%)p 2.65( 72.55%)...
        """
        # Deve iniziare con la percentuale di contributo totale tra parentesi
        if not re.search(r'^\s*\(\s*\d+\.\d+%\)', line):
            return None

        # Regex base per catturare: %contributo, atomo, indice, %s ('0.7289*Br 33': no space after '*')
        base_pattern = r'\(\s*([0-9.]+)%\)\s+[-0-9.]+\*?\s*([A-Z][a-z]?)\s+(\d+)\s+s\(\s*([0-9.]+)%\)'
        match = re.search(base_pattern, line)
        if not match:
            return None

        comp = {
            'contribution_percent': float(match.group(1)),
            'atom_symbol': match.group(2),
            'atom_index': int(match.group(3)),
            's_percent': float(match.group(4))
        }

        # Cerca altri orbitali sulla stessa linea (p, d, f)
        self._extract_orbitals_from_line(line, comp)
        return comp

    def _extract_orbitals_from_line(self, line: str, comp_dict):
        """Helper per estrarre p, d, f da una linea e aggiornare il dizionario."""
        # Fix 2: Regex case-insensitive (opzionale, dipende dal formato ORCA)
        # Cerca pattern tipo: p 2.65( 72.55%), f 0.00( 0.02%) o p99.99( 99.92%)
        # Cattura: tipo (p/d/f), coefficiente (ignorato), percentuale
        matches = re.findall(r'([pdfPDF])\s*[-0-9.]+\(\s*([0-9.]+)%\)', line)
        for orb_type, percent in matches:
            comp_dict[f'{orb_type.lower()}_percent'] = float(percent)


    def _parse_npa(self, text):
        """Parsa la sezione Summary of Natural Population Analysis"""
        data = []

        # Atom No    Charge        Core      Valence    Rydberg      Total
        #   C  1    0.12345      1.999      3.800      0.010      5.809
        pattern = r'^\s*([A-Z][a-z]?)\s+(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
        
        section_start = text.find('Summary of Natural Population Analysis')
        if section_start == -1:
            logger.warning("NPA section not found")
            return None
            
        # ponytail: splitlines() avoids trailing \r and is faster than split('\n')
        lines = text[section_start:].splitlines()
        start_parsing = False
        
        for line in lines:
            if 'Atom' in line and 'Charge' in line:
                start_parsing = True
                continue
            if '==================' in line and start_parsing:
                break
                
            if start_parsing:
                match = re.search(pattern, line)
                if match:
                    data.append({
                        'Atom': match.group(1),
                        'Index': int(match.group(2)) - 1, # 0-based index
                        'Charge': float(match.group(3)),
                        'Core': float(match.group(4)),
                        'Valence': float(match.group(5)),
                        'Rydberg': float(match.group(6)),
                        'Total_Pop': float(match.group(7))
                    })
        
        return pd.DataFrame(data) if data else None


    def _extract_type(self, description_str):
        """Estrae il tipo di orbitale (BD, BD*, LP, ecc.)"""
        # Cerca stringhe come "BD ( 1)" o "BD*( 2)"
        match = re.search(r'([A-Z]+\*?)\s*\(', description_str)
        return match.group(1) if match else "???"

    @staticmethod
    def _parse_donor_acceptor_pair(text_part: str) -> Tuple[Optional[int], str, Optional[int], str]:
        """Extract donor and acceptor (index, desc) from perturbation line."""
        if '/' in text_part:
            parts = text_part.split('/', 1)
            dm = re.match(r'^(\d+)\.\s+(.*)', parts[0].strip())
            am = re.match(r'^(\d+)\.\s+(.*)', parts[1].strip())
            if dm and am:
                return int(dm.group(1)), dm.group(2).strip(), int(am.group(1)), am.group(2).strip()
        else:
            indices_matches = list(re.finditer(r'(\d+)\.', text_part))
            if len(indices_matches) >= 2:
                dm = re.match(r'^(\d+)\.\s+(.*)', text_part[indices_matches[0].start():indices_matches[1].start()].strip())
                am = re.match(r'^(\d+)\.\s+(.*)', text_part[indices_matches[1].start():].strip())
                if dm and am:
                    return int(dm.group(1)), dm.group(2).strip(), int(am.group(1)), am.group(2).strip()
        return None, "", None, ""

    def _parse_perturbation(self, text):
        """
        Parsa la Second Order Perturbation Theory Analysis (E2).
        Gestisce formati standard a colonne e formati con '/'
        """
        data = []
        # NBO 7 (ORCA) prints the title in upper case, NBO 3.1 (Gaussian) in mixed case
        section = re.search(r'Second Order Perturbation Theory Analysis', text, re.IGNORECASE)
        if section is None:
            logger.warning("E(2) perturbation section not found")
            return None

        lines = text[section.start():].splitlines()
        start_parsing = False
        nums_pattern = r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$'

        for line in lines:
            if 'within' in line:
                start_parsing = True
                continue
            if start_parsing and "Natural Bond Orbitals" in line:
                break

            if start_parsing:
                line = line.strip()
                if not line:
                    continue

                num_match = re.search(nums_pattern, line)
                if not num_match:
                    continue

                try:
                    e2_val = float(num_match.group(1))
                    e_diff = float(num_match.group(2))
                    f_ij = float(num_match.group(3))
                except ValueError as e:
                    logger.debug(f"Errore conversione numeri E(2): {e}")
                    continue

                text_part = line[:num_match.start()].strip()
                donor_idx, donor_desc, acc_idx, acc_desc = self._parse_donor_acceptor_pair(text_part)

                if donor_idx is not None:
                    d_type = self._extract_type(donor_desc)
                    a_type = self._extract_type(acc_desc)
                    d_atoms_lbl = re.sub(r'^[A-Z]+\*?\s*\(\s*\d+\)', '', donor_desc).strip()
                    a_atoms_lbl = re.sub(r'^[A-Z]+\*?\s*\(\s*\d+\)', '', acc_desc).strip()

                    data.append({
                        'Donor_Index': donor_idx,
                        'Donor_Type': d_type,
                        'Donor_Label': d_atoms_lbl,
                        'Donor_Atoms': self._extract_atom_indices(donor_desc),
                        'Acceptor_Index': acc_idx,
                        'Acceptor_Type': a_type,
                        'Acceptor_Label': a_atoms_lbl,
                        'Acceptor_Atoms': self._extract_atom_indices(acc_desc),
                        'E(2)': e2_val,
                        'E_diff': e_diff,
                        'F_ij': f_ij
                    })

        return pd.DataFrame(data) if data else None

    def _parse_nbo_details(self, text):
        """Parsa la sezione 'NATURAL BOND ORBITAL ANALYSIS'."""
        start_marker = "NATURAL BOND ORBITAL ANALYSIS:"
        section_start = text.find(start_marker)
        if section_start == -1:
            logger.warning("Sezione NBO Details non trovata")
            return None

        raw_section = text[section_start:]
        # NBO 7 (ORCA) prints these titles in upper case, NBO 3.1 (Gaussian) in mixed case; without
        # the stop the NHO, E(2) and summary rows of NBO 3.1 would be read as further NBOs
        stop = re.search(r'NHO Directionality|Natural Atomic Orbitals|Second Order Perturbation'
                         r'|Natural Bond Orbitals \(Summary\)', raw_section, re.IGNORECASE)
        end_idx = stop.start() if stop else len(raw_section)

        lines = raw_section[:end_idx].splitlines()
        fmt_a = r'^\s+(\d+)\.\s+\((-?\d+\.\d+)\)\s+([A-Z]+\*?)\s*\(\s*\d+\)\s*(.*)'
        fmt_b = r'^\s+(\d+)\.\s+([A-Z]+\*?)\s*\(\s*\d+\)\s*(.*)'

        nbos = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line or "---" in line or "NATURAL BOND" in line:
                i += 1
                continue

            match_a = re.search(fmt_a, lines[i])
            match_b = re.search(fmt_b, lines[i])

            if match_a:
                index, occupancy, nbo_type, raw_label = int(match_a.group(1)), float(match_a.group(2)), match_a.group(3), match_a.group(4)
                energy = 0.0
                e_match = re.search(r'\s+(-\d+\.\d+)\s*$', raw_label)
                if e_match:
                    try:
                        energy = float(e_match.group(1))
                        raw_label = raw_label[:e_match.start()]
                    except ValueError:
                        pass
            elif match_b:
                index, nbo_type, raw_content = int(match_b.group(1)), match_b.group(2), match_b.group(3)
                metrics = re.search(r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$', raw_content)
                if metrics:
                    occupancy, energy = float(metrics.group(1)), float(metrics.group(2))
                    raw_label = raw_content[:metrics.start()]
                else:
                    occupancy, energy = 0.0, 0.0
                    raw_label = raw_content
            else:
                i += 1
                continue

            current_nbo = {
                'Index': index,
                'Type': nbo_type,
                'Label': self._clean_label(raw_label),
                'Atom_Indices': self._extract_atom_indices_from_label(raw_label),
                'Occupancy': occupancy,
                'Energy': energy,
                'Composition': []
            }

            # Inline composition
            # One-centre NBO: "LP ( 1) O  1   s(  0.00%)p 1.00( 99.92%)d 0.00(  0.08%)"
            inline_pattern = r'([A-Z][a-z]?)\s+(\d+)\s+s\(\s*([0-9.]+)%\)'
            inline_match = re.search(inline_pattern, lines[i])
            if inline_match:
                comp = {
                    'contribution_percent': 100.0,
                    'atom_symbol': inline_match.group(1),
                    'atom_index': int(inline_match.group(2)),
                    's_percent': float(inline_match.group(3))
                }
                self._extract_orbitals_from_line(lines[i][inline_match.end():], comp)
                current_nbo['Composition'].append(comp)

            # Composition lines following
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                # Next NBO header ("5. (1.99869) BD ..." or "5. BD ..."); a bare r'^\d+\.' would
                # also match coefficient rows such as "0.0001 -0.4922 ..." and drop the second atom
                if re.search(r'^\d+\.\s+(\(|[A-Z])', next_line):
                    break
                if self._is_coefficient_line(next_line):
                    i += 1
                    continue
                # Hybrid wrapped onto its own row ("f 0.00(  0.02%)", NBO 7 with f functions)
                if current_nbo['Composition'] and re.match(r'^[pdf]\s*[-0-9.]+\(', next_line):
                    self._extract_orbitals_from_line(next_line, current_nbo['Composition'][-1])
                    i += 1
                    continue

                comp = self._parse_composition_line(next_line)
                if comp:
                    current_nbo['Composition'].append(comp)
                    i += 1
                else:
                    if not next_line:
                        i += 1
                        continue
                    break

            nbos.append(current_nbo)

        # Post-processing per formattazione stringa composizione
        for nbo in nbos:
            comp_strs = []
            for comp in nbo['Composition']:
                atom_str = f"{comp['atom_symbol']} {comp['atom_index']} ({comp['contribution_percent']:.2f}%)"
                hyb_parts = [f"{orb}({comp[f'{orb}_percent']:.2f}%)" for orb in ('s', 'p', 'd', 'f') if f'{orb}_percent' in comp]
                hyb_str = ' | '.join(hyb_parts)
                comp_strs.append(f"{atom_str} {hyb_str}".strip())
            nbo['Composition'] = comp_strs

        # Orbital energies are only printed in the summary table
        summary = self._parse_nbo_summary(text)
        for nbo in nbos:
            if nbo['Index'] in summary:
                occ, energy = summary[nbo['Index']]
                nbo['Energy'] = energy
                if not nbo['Occupancy']:
                    nbo['Occupancy'] = occ

        return pd.DataFrame(nbos) if nbos else None

    @staticmethod
    def _parse_nbo_summary(text: str) -> Dict[int, Tuple[float, float]]:
        """Map NBO index -> (occupancy, energy) from 'NATURAL BOND ORBITALS (Summary)'.

        NBO 3.1 (Gaussian) prints the title as 'Natural Bond Orbitals (Summary)'.
        For open-shell outputs (alpha and beta summaries) the first occurrence is kept.
        """
        start = re.search(r'NATURAL BOND ORBITALS \(Summary\)', text, re.IGNORECASE)
        if start is None:
            return {}
        row = re.compile(r'^\s*(\d+)\.\s+[A-Z]+\*?\s*\(\s*\d+\)[^.]*?\s(-?\d+\.\d+)\s+(-?\d+\.\d+)')
        summary = {}
        for line in text[start.start():].splitlines():
            m = row.match(line)
            if m and int(m.group(1)) not in summary:
                summary[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        return summary

    def _parse_wiberg(
        self, text: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[Dict[int, float]]]:
        """Parse Wiberg bond index matrix and atom totals from NBO text."""
        marker = "Wiberg bond index matrix in the NAO basis:"
        idx = text.rfind(marker)
        if idx == -1:
            return None, None, None

        section = text[idx + len(marker):]
        lines = section.splitlines()

        matrix_data: Dict[Tuple[int, int], float] = {}
        totals: Dict[int, float] = {}
        col_indices: list[int] = []
        max_atom = 0
        parsing_totals = False

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            if "Totals by atom:" in line_str:
                parsing_totals = True
                continue

            if parsing_totals:
                if "==================" in line_str or line_str.startswith("---"):
                    continue
                m_tot = re.match(r'^\s*(\d+)\.\s+([A-Z][a-z]?)\s+(-?\d+\.\d+)', line)
                if m_tot:
                    a_idx = int(m_tot.group(1)) - 1
                    totals[a_idx] = float(m_tot.group(3))
                    max_atom = max(max_atom, a_idx + 1)
                elif (
                    line_str.startswith("NATURAL")
                    or line_str.startswith("NHO")
                    or line_str.startswith("SECOND ORDER")
                    or line_str.startswith("*******")
                ):
                    break
                continue

            if line_str.startswith("Atom"):
                parts = line_str.split()[1:]
                col_indices = [int(p) - 1 for p in parts if p.isdigit()]
                continue

            if line_str.startswith("----"):
                continue

            if line_str.startswith("NATURAL") or line_str.startswith("NHO"):
                break

            m_row = re.match(r'^\s*(\d+)\.\s+([A-Z][a-z]?)\s+(.*)', line)
            if m_row and col_indices:
                row_idx = int(m_row.group(1)) - 1
                max_atom = max(max_atom, row_idx + 1)
                raw_vals = m_row.group(3).split()
                for c_idx, val_str in zip(col_indices, raw_vals):
                    try:
                        matrix_data[(row_idx, c_idx)] = float(val_str)
                        max_atom = max(max_atom, c_idx + 1)
                    except ValueError:
                        pass

        if max_atom == 0:
            return None, None, None

        mat = np.zeros((max_atom, max_atom), dtype=float)
        for (r, c), v in matrix_data.items():
            mat[r, c] = v
            mat[c, r] = v

        df = pd.DataFrame(mat, index=range(max_atom), columns=range(max_atom))
        return mat, df, totals

    def get_bonds(
        self,
        atom1: Optional[int] = None,
        atom2: Optional[int] = None,
        bond_type: Optional[str] = None,
        min_occ: float = 0.0,
    ) -> pd.DataFrame:
        """Filter and return bonding/antibonding NBO orbitals.

        Args:
            atom1: 0-based atom index. If specified alone, matches any bond involving atom1.
            atom2: 0-based atom index. If specified with atom1, matches bonds strictly between atom1 and atom2.
            bond_type: Optional filter ('BD', 'BD*', 'sigma', 'pi').
            min_occ: Minimum occupancy threshold (default: 0.0).

        Returns:
            pandas.DataFrame of matching bond orbitals.
        """
        if self.nbo_details is None or self.nbo_details.empty:
            return pd.DataFrame()

        df = self.nbo_details[self.nbo_details['Type'].isin(['BD', 'BD*'])].copy()

        if min_occ > 0:
            df = df[df['Occupancy'] >= min_occ]

        if bond_type:
            bt = bond_type.strip().upper()
            if bt == 'SIGMA':
                df = df[df['Label'].str.contains(r'\( 1\)|BD\s*\( 1\)|BD\*\s*\( 1\)')]
            elif bt == 'PI':
                df = df[df['Label'].str.contains(r'\(\s*[23]\)|BD\s*\(\s*[23]\)|BD\*\s*\(\s*[23]\)')]
            else:
                df = df[df['Type'].str.upper() == bt]

        if atom1 is not None and atom2 is not None:
            df = df[df['Atom_Indices'].apply(lambda idxs: atom1 in idxs and atom2 in idxs)]
        elif atom1 is not None:
            df = df[df['Atom_Indices'].apply(lambda idxs: atom1 in idxs)]
        elif atom2 is not None:
            df = df[df['Atom_Indices'].apply(lambda idxs: atom2 in idxs)]

        return df

    def get_lone_pairs(
        self,
        atom: Optional[int] = None,
        min_occ: float = 0.0,
    ) -> pd.DataFrame:
        """Filter and return lone pairs (LP, LP*, LV).

        Args:
            atom: Optional 0-based atom index.
            min_occ: Minimum occupancy threshold (default: 0.0).

        Returns:
            pandas.DataFrame of matching lone pairs.
        """
        if self.nbo_details is None or self.nbo_details.empty:
            return pd.DataFrame()

        df = self.nbo_details[self.nbo_details['Type'].isin(['LP', 'LP*', 'LV'])].copy()

        if min_occ > 0:
            df = df[df['Occupancy'] >= min_occ]

        if atom is not None:
            df = df[df['Atom_Indices'].apply(lambda idxs: atom in idxs)]

        return df

    def get_interactions(
        self,
        donor: Optional[Union[int, Sequence[int]]] = None,
        acceptor: Optional[Union[int, Sequence[int]]] = None,
        donor_type: Optional[str] = None,
        acceptor_type: Optional[str] = None,
        min_e2: float = 2.0,
    ) -> pd.DataFrame:
        """Query donor-acceptor interactions from second-order perturbation theory E(2).

        Args:
            donor: 0-based atom index or list of atom indices in donor orbital.
            acceptor: 0-based atom index or list of atom indices in acceptor orbital.
            donor_type: Filter by donor type (e.g. 'LP', 'BD').
            acceptor_type: Filter by acceptor type (e.g. 'BD*', 'RY*').
            min_e2: Minimum perturbation energy threshold in kcal/mol (default: 2.0).

        Returns:
            pandas.DataFrame sorted descending by E(2).
        """
        if self.perturbation_theory is None or self.perturbation_theory.empty:
            return pd.DataFrame()

        df = self.perturbation_theory[self.perturbation_theory['E(2)'] >= min_e2].copy()

        if donor_type:
            df = df[df['Donor_Type'].str.upper().str.contains(donor_type.upper())]

        if acceptor_type:
            df = df[df['Acceptor_Type'].str.upper().str.contains(acceptor_type.upper())]

        if donor is not None:
            d_set = {donor} if isinstance(donor, int) else set(donor)
            df = df[df['Donor_Atoms'].apply(lambda a: not set(a).isdisjoint(d_set))]

        if acceptor is not None:
            a_set = {acceptor} if isinstance(acceptor, int) else set(acceptor)
            df = df[df['Acceptor_Atoms'].apply(lambda a: not set(a).isdisjoint(a_set))]

        return df.sort_values(by='E(2)', ascending=False)

    def get_wiberg_bond_order(self, atom1: int, atom2: int) -> float:
        """Return Wiberg bond order between two atoms (0-based indices)."""
        if self.wiberg_matrix is None:
            raise ValueError(
                "Wiberg bond index matrix not found. "
                "Ensure NBO was executed with $NBO BNDIDX $END."
            )
        n = self.wiberg_matrix.shape[0]
        if not (0 <= atom1 < n and 0 <= atom2 < n):
            raise IndexError(f"Atom indices ({atom1}, {atom2}) out of range (0..{n - 1})")
        return float(self.wiberg_matrix[atom1, atom2])

    def get_wiberg_total(self, atom: int) -> float:
        """Return total Wiberg valence for an atom (0-based index)."""
        if self.wiberg_totals is not None and atom in self.wiberg_totals:
            return float(self.wiberg_totals[atom])
        if self.wiberg_matrix is not None:
            n = self.wiberg_matrix.shape[0]
            if not (0 <= atom < n):
                raise IndexError(f"Atom index {atom} out of range (0..{n - 1})")
            diag = self.wiberg_matrix[atom, atom]
            return float(np.sum(self.wiberg_matrix[atom, :]) - diag)
        raise ValueError("Wiberg bond index data not found in NBO output.")

    def get_npa_charge(self, atom: int) -> float:
        """Return the NPA partial charge of an atom (0-based index)."""
        if self.npa_charges is None or self.npa_charges.empty:
            raise ValueError("NPA charges not found in NBO output.")
        match = self.npa_charges[self.npa_charges['Index'] == atom]
        if match.empty:
            raise IndexError(f"Atom index {atom} not found in NPA charges.")
        return float(match.iloc[0]['Charge'])

    def summary(self) -> dict:
        """Return a structured summary dictionary of the NBO analysis."""
        res: Dict[str, Any] = {
            "n_atoms": len(self.npa_charges) if self.npa_charges is not None else 0,
            "has_npa": self.npa_charges is not None and not self.npa_charges.empty,
            "has_details": self.nbo_details is not None and not self.nbo_details.empty,
            "has_e2": self.perturbation_theory is not None and not self.perturbation_theory.empty,
            "has_wiberg": self.wiberg_matrix is not None,
        }
        if self.nbo_details is not None and not self.nbo_details.empty:
            res["orbital_counts"] = self.nbo_details['Type'].value_counts().to_dict()
        if self.perturbation_theory is not None and not self.perturbation_theory.empty:
            top_e2 = self.perturbation_theory.sort_values(by='E(2)', ascending=False).iloc[0]
            res["strongest_interaction"] = {
                "donor": f"{top_e2['Donor_Type']} {top_e2['Donor_Label']}",
                "acceptor": f"{top_e2['Acceptor_Type']} {top_e2['Acceptor_Label']}",
                "e2": float(top_e2['E(2)']),
            }
        return res