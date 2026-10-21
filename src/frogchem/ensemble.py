"""
Utilities for conformer ensembles and Boltzmann population weighting.

Completely standalone: parses standard JSON and CSV files with zero dependencies
on external ensemble packages (no import or dependency on ensemble_analyzer).
"""

import re
import csv
import json
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.constants import R
from frogchem.thermo import free_gibbs_energy
from frogchem.chemistry import HARTREE_TO_KCAL, get_atomic_weights
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence


def extract_conf_and_protocol(file_path: Union[str, Path]) -> Tuple[Optional[int], Optional[int]]:
    """
    Extracts conformer number and protocol number from a file path or label.
    
    Prioritizes EnAn directory structure:
        .../conf_<ID>/protocol_<X>/*.(out|log)
    
    Also supports:
        - Trajectory / frame labels: '... - Conf <X>'
        - Filename patterns: 'conf_1_p2.out', '1_p2.out'
    
    Returns:
        Tuple of (conf_number, protocol_number) as ints, or (None, None).
    """
    s = str(file_path).strip()
    p = Path(s)
    parts = p.parts
    
    conf_id: Optional[int] = None
    proto_id: Optional[int] = None

    # 1. EnAn directory parts: .../conf_<ID>/protocol_<X>/...
    for part in reversed(parts[:-1]):
        if proto_id is None:
            m_proto = re.match(r'^protocol[_-]?(\d+)$', part, re.IGNORECASE)
            if m_proto:
                proto_id = int(m_proto.group(1))
        if conf_id is None:
            m_conf = re.match(r'^conf[_-]?(\d+)$', part, re.IGNORECASE)
            if m_conf:
                conf_id = int(m_conf.group(1))

    # 2. Check filename as fallback or supplement
    fname = p.name
    if conf_id is None or proto_id is None:
        # Match pattern like: 1_p2_orca.out or conf_1_p2.out
        m_file = re.search(r'(?:conf[_-]?)?(\d+)_p(\d+)', fname, re.IGNORECASE)
        if m_file:
            if conf_id is None:
                conf_id = int(m_file.group(1))
            if proto_id is None:
                proto_id = int(m_file.group(2))

    if conf_id is None:
        # Match standalone conf number in filename: conf_1.out, conf1.out
        m_conf_file = re.search(r'conf[_-]?(\d+)', fname, re.IGNORECASE)
        if m_conf_file:
            conf_id = int(m_conf_file.group(1))

    # 3. Check for '- Conf X' frame label (e.g. from internals or rmsd)
    if conf_id is None:
        m_frame = re.search(r'-\s*Conf\s*(\d+)', s, re.IGNORECASE)
        if m_frame:
            conf_id = int(m_frame.group(1)) + 1

    return conf_id, proto_id


def _energy_from_record(conf: dict, rec: dict, energy_term: str, temp: float,
                        recompute: bool) -> Tuple[float, str]:
    """Return (energy [Eh], term actually used) of one checkpoint record.

    With ``recompute`` and stored frequencies, H/G are recomputed (mRRHO) from E, Freq and
    B_vec. Otherwise stored values are used; a missing G falls back to E and is tagged 'E'
    so that the caller can refuse to mix terms.
    """
    nan = float('nan')
    e = rec.get('E', nan)
    e = nan if e is None else float(e)
    freqs = rec.get('Freq', None)
    if recompute and freqs and not np.isnan(e):
        atoms = conf.get('atoms', [])
        try:
            mw = sum(get_atomic_weights(a) for a in atoms) if atoms else 100.0
        except AttributeError:
            mw = 100.0  # unknown symbol: mw only shifts G of all conformers equally
        mult = int(rec.get('mult', conf.get('mult', 1)) or 1)
        try:
            G_calc, _, H_calc, _ = free_gibbs_energy(
                SCF=e, T=temp, freq=np.asarray(freqs, dtype=float), mw=mw,
                B=np.asarray(rec['B_vec'], dtype=float), m=mult)
            return (H_calc, 'H') if energy_term == 'H' else (G_calc, 'G')
        except Exception:
            pass  # fall back to the stored values below

    def _get(key):
        x = rec.get(key, nan)
        return nan if x is None else float(x)

    if energy_term == 'E':
        return e, 'E'
    if energy_term == 'EZPVE':
        return e + _get('zpve'), 'EZPVE'
    if energy_term == 'H':
        return e + _get('H'), 'H'
    g = _get('G')
    if np.isnan(g):
        return e, 'E'
    return g, 'G'


def load_checkpoint_populations(
    checkpoint_path: Union[str, Path],
    protocol: Optional[Union[str, int]] = None,
    energy_term: str = 'G',
    temp: float = 298.15
) -> Dict[int, float]:
    """
    Load Boltzmann populations from a JSON or CSV/TSV file without requiring any external package.
    
    Supported formats:
    1. Checkpoint JSON (from EnAn / ensemble_analyzer):
       { "1": { "energies": { "2": { "Pop": 75.0 } } } }
       or with top-level "conformers" key:
       { "conformers": { "1": { "energies": { "2": { "Pop": 75.0 } } } } }
    2. Simple dictionary JSON:
       { "1": 75.0, "2": 25.0 }
    3. CSV / TSV file:
       conf,pop
       1, 75.0
       2, 25.0
    
    Args:
        checkpoint_path: Path to the population or checkpoint file.
        protocol: Specific protocol number to read Boltzmann populations from
                  when using a nested checkpoint file.
                  
    Returns:
        Dict mapping conformer number (int) -> Boltzmann population (percentage).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Population file not found: {checkpoint_path}")

    populations: Dict[int, float] = {}

    # Format A: CSV / TSV
    if path.suffix.lower() in ('.csv', '.tsv', '.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            f.seek(0)
            delimiter = '\t' if '\t' in sample and ',' not in sample else ','
            reader = csv.reader(f, delimiter=delimiter)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                try:
                    conf_idx = int(re.sub(r'[^\d]', '', row[0]))
                    pop_val = float(row[1].strip().replace('%', ''))
                    populations[conf_idx] = pop_val
                except (ValueError, TypeError):
                    continue
        return populations

    # Format B: JSON
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return populations

    # Check for top-level wrapper keys from EnAn
    if "conformers" in data and isinstance(data["conformers"], dict):
        data = data["conformers"]
    elif "data" in data and isinstance(data["data"], dict):
        data = data["data"]

    raw_energies = {}
    fallback_populations = {}
    records = {}

    for key, val in data.items():
        try:
            num = int(val.get('number', key) if isinstance(val, dict) else key)
        except (ValueError, TypeError):
            continue

        if isinstance(val, (int, float)):
            fallback_populations[num] = float(val)
            continue

        if not isinstance(val, dict):
            continue

        energies = val.get('energies', {})
        if not isinstance(energies, dict):
            pop_direct = val.get('Pop', val.get('population', val.get('pop', np.nan)))
            if pop_direct is not None and not np.isnan(pop_direct):
                fallback_populations[num] = float(pop_direct)
            continue

        rec = None
        if protocol is not None:
            proto_candidates = [str(protocol), f"protocol_{protocol}", f"p{protocol}"]
            for pk in proto_candidates:
                if pk in energies:
                    rec = energies[pk]
                    break
        else:
            sorted_keys = sorted(
                energies.keys(),
                key=lambda x: int(re.sub(r'[^\d]', '', str(x))) if re.search(r'\d+', str(x)) else str(x),
                reverse=True
            )
            for k in sorted_keys:
                if isinstance(energies[k], dict):
                    rec = energies[k]
                    break

        if isinstance(rec, dict):
            records[num] = (val, rec)

            # Also store fallback Pop in case we can't calculate Boltzmann
            v = rec.get('Pop', rec.get('pop', rec.get('population', np.nan)))
            if v is not None and not np.isnan(v):
                fallback_populations[num] = float(v)

    # mRRHO H/G are recomputed from the stored frequencies only if EVERY conformer that has
    # frequencies also has its rotational constants: a default B for some conformers only
    # biases their relative G by several kcal/mol.
    with_freq = [r for _, r in records.values() if r.get('Freq')]
    recompute = (energy_term in ('H', 'G') and bool(with_freq)
                 and all(r.get('B_vec') is not None for r in with_freq))
    if energy_term in ('H', 'G') and with_freq and not recompute:
        warnings.warn("Some conformers have frequencies but no 'B_vec': mRRHO thermochemistry is not "
                      "recomputed, the stored energies are used", RuntimeWarning)

    terms = {}
    for num, (val, rec) in records.items():
        val_e, term = _energy_from_record(val, rec, energy_term, temp, recompute)
        if not np.isnan(val_e):
            raw_energies[num] = val_e
            terms[num] = term

    # Never mix energy terms (e.g. G of some conformers with E of others) in one Boltzmann pool.
    if len(set(terms.values())) > 1:
        dropped = sorted(n for n, t in terms.items() if t != energy_term)
        warnings.warn(f"Conformers {dropped} lack the '{energy_term}' energy and are excluded from the "
                      "Boltzmann populations (energy terms cannot be mixed)", RuntimeWarning)
        raw_energies = {n: e for n, e in raw_energies.items() if terms[n] == energy_term}

    if raw_energies:
        # Calculate Boltzmann populations on the fly (CODATA Eh -> kcal/mol, 4184 J/kcal)
        min_e = min(raw_energies.values())

        weights = {}
        sum_w = 0.0
        for num, e in raw_energies.items():
            rel_kcal = (e - min_e) * HARTREE_TO_KCAL
            exponent = -(rel_kcal * 4184.0) / (R * temp)
            w = math.exp(exponent)
            weights[num] = w
            sum_w += w

        if sum_w > 0:
            for num, w in weights.items():
                populations[num] = (w / sum_w) * 100.0
            return populations

    # Fallback to pre-calculated populations if energies were missing
    for num, p in fallback_populations.items():
        populations[num] = p

    return populations


class EnsembleEntry:
    """Represents a matched conformer entry in an ensemble."""
    def __init__(self, item: Any, file_path: str, conf_id: int, proto_id: Optional[int], population: float, weight: float):
        self.item = item
        self.file_path = file_path
        self.conf_id = conf_id
        self.proto_id = proto_id
        self.population = population
        self.weight = weight

    def __repr__(self) -> str:
        return f"<EnsembleEntry conf_{self.conf_id} P{self.proto_id} pop={self.population:.2f}% w={self.weight:.4f}>"


class EnsembleData:
    """Holds the complete ensemble matching and weighting results."""
    def __init__(
        self,
        entries: List[EnsembleEntry],
        populations: Dict[int, float],
        protocol_used: Optional[Union[str, int]] = None,
    ):
        self.entries = entries
        self.populations = populations
        self.protocol_used = protocol_used

        proto_info = f" (Protocol {protocol_used})" if protocol_used else ""
        self.summary_headers = ['Conformer', 'Protocol', 'File', f'Boltzmann Pop (%){proto_info}', 'Normalized Weight (%)']
        self.summary_rows = []
        for entry in self.entries:
            c_lbl = f"conf_{entry.conf_id}" if entry.conf_id is not None else "N/A"
            p_lbl = f"P{entry.proto_id}" if entry.proto_id is not None else "N/A"
            fname = Path(entry.file_path).name if entry.file_path else "N/A"
            self.summary_rows.append([
                c_lbl,
                p_lbl,
                fname,
                float(entry.population),
                float(entry.weight * 100)
            ])

    @property
    def weights(self) -> List[float]:
        return [e.weight for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)


def build_ensemble(
    items: Sequence[Any],
    checkpoint_path: Union[str, Path],
    protocol: Optional[Union[str, int]] = None,
    path_extractor: Optional[Any] = None,
    energy_term: str = 'G',
    temp: float = 298.15
) -> EnsembleData:
    """
    Build an ensemble weighting object by matching items to an EnAn checkpoint file.
    
    Conformer and protocol IDs are determined via extract_conf_and_protocol
    (prioritizing EnAn folder structure .../conf_<ID>/protocol_<X>/*.(out|log)).
    If conf_id is not found, fallback to 1-based sequential index.
    
    Args:
        items: List of file paths or objects.
        checkpoint_path: Path to checkpoint JSON or CSV file.
        protocol: Optional protocol number/name to read populations from.
        path_extractor: Optional callable taking an item and returning its file path/string.
        
    Returns:
        EnsembleData object.
    """
    populations = load_checkpoint_populations(checkpoint_path, protocol=protocol, energy_term=energy_term, temp=temp)

    entries = []
    raw_pops = []

    for idx, it in enumerate(items):
        f_path = str(path_extractor(it) if path_extractor else it)
        c_id, p_id = extract_conf_and_protocol(f_path)
        if c_id is None:
            c_id = idx + 1
        pop = populations.get(c_id, 0.0)
        raw_pops.append(pop)
        entries.append({
            'item': it,
            'file_path': f_path,
            'conf_id': c_id,
            'proto_id': p_id,
            'pop': pop,
        })

    tot_pop = sum(raw_pops)
    if tot_pop <= 0:
        weights = [1.0 / len(entries)] * len(entries) if entries else []
    else:
        weights = [p / tot_pop for p in raw_pops]

    ensemble_entries = [
        EnsembleEntry(
            item=d['item'],
            file_path=d['file_path'],
            conf_id=d['conf_id'],
            proto_id=d['proto_id'],
            population=d['pop'],
            weight=w,
        )
        for d, w in zip(entries, weights)
    ]

    return EnsembleData(
        entries=ensemble_entries,
        populations=populations,
        protocol_used=protocol,
    )
