import re
import logging
import numpy as np

from frogchem.parser_base import BaseParser
from frogchem.exceptions import ParseError
from frogchem.chemistry import get_atomic_weights
from frogchem.orca_parser.regex_ORCA import REGEXS
from frogchem.orca_parser.orbitals import get_MOs_coefficients, get_MOs_energies, get_basis_coefficients
from frogchem.orca_parser.hessian import get_freq, get_normal_modes, get_hessian_graphs_impulses
from frogchem.orca_parser.opt import get_geom, get_energy_decomposition
from frogchem.orca_parser.tddft import parse_tddft_states, get_tddft_soc, parse_tddft_fosc
from frogchem.orca_parser.population import parse_charges
from frogchem.orca_parser.nbo import NBOAnalysis
from frogchem.orca_parser.nmr import NMRAnalysis

logger = logging.getLogger(__name__)


class Calculation(BaseParser):

    def _parse_file(self):
        version_match = re.findall(r'Program Version (\d).\d.\d', self.fl)
        if not version_match:
            raise ParseError(f"Cannot determine ORCA version in {self.fname}")
        self.version = version_match[0]
        if self.version not in REGEXS:
            raise ParseError(
                f"Unsupported ORCA version '{self.version}' in {self.fname}. "
                f"Supported versions: {list(REGEXS.keys())}"
            )
        self.reg = REGEXS[self.version]
        self._parse_input()

        self.opt_sections = re.findall(self.reg['opt_section'], self.fl)
        self.atoms, self.geom = get_geom(self.fl, self.reg)

        # ponytail: safe extraction for single-atom calculations or partial outputs without rotational constants
        rot_matches = re.findall(self.reg['rot_const'], self.fl)
        self.B = np.array(rot_matches[-1], dtype=float) if rot_matches else None
        self.mw = sum(get_atomic_weights(i) for i in self.atoms)

        # SCF energy decomposition only (Eh, keys as printed by ORCA, plus dispersion);
        # ZPE/H/G are not parsed here: compute them with frogchem.thermo.free_gibbs_energy
        self.energy_terms = get_energy_decomposition(self.fl, self.reg)
        self.scf_energy = self.energy_terms.get(self.reg.get('total_energy', ''), 0.0) if self.energy_terms else 0.0
        # ponytail: safe fallback if final single point energy marker is missing (e.g. uncompleted run)
        fsp_matches = re.findall(self.reg['final_single_point_energy'], self.fl)
        self.final_energy = float(fsp_matches[-1]) if fsp_matches else self.scf_energy

        try:
            self.mulliken = parse_charges(self.fl, 'mulliken', self.reg)
            self.loewdin = parse_charges(self.fl, 'loewdin', self.reg)
        except Exception as e:
            raise ParseError(f"Failed to parse atomic charges in {self.fname}: {e}") from e

        if 'opt' in self.__dict__:
            try:
                self.converged = 'THE OPTIMIZATION HAS CONVERGED' in self.fl
                self.trajectory = self._get_trajectory()
                self.energy_trajectory = self._energy_trajectory()
            except Exception as e:
                raise ParseError(f"Failed to parse optimization data in {self.fname}: {e}") from e

        if 'printbasis' in self.__dict__:
            try:
                self.basis_coeff = get_basis_coefficients(self.fl, self.reg)
            except Exception as e:
                raise ParseError(f"Failed to parse basis coefficients in {self.fname}: {e}") from e

        if 'printmos' in self.__dict__:
            try:
                self.orbital_energies, self.occupation = get_MOs_energies(self.fl, self.reg)
                self.mask_atoms, self.mask_type_n, self.mask_orbital_type, self.orbital_coefficients = get_MOs_coefficients(self.fl, self.reg)

                if self.mult != 1:
                    idx = self.orbital_coefficients.shape[0]
                    self.alpha_occ, self.beta_occ = self.occupation[:idx], self.occupation[idx:]
                    self.alpha_coeff, self.beta_coeff = self.orbital_coefficients[:, :idx], self.orbital_coefficients[:, idx:]
            except Exception as e:
                raise ParseError(f"Failed to parse molecular orbitals in {self.fname}: {e}") from e

        if 'freq' in self.__dict__:
            try:
                self.number_modes, self.freq = get_freq(self.fl, self.reg)
                self.normal_modes = get_normal_modes(self.fl, self.reg, N=len(self.freq))
                self.vib_impulses = get_hessian_graphs_impulses(self.fl, self.reg)
            except Exception as e:
                raise ParseError(f"Failed to parse frequencies in {self.fname}: {e}") from e

        if 'tddft' in self.__dict__:
            try:
                self.tddft_states = parse_tddft_states(self.fl, self.reg)
                parse_tddft_fosc(self.fl, self.tddft_states)
                self.soc = False
                soc_start = self.reg.get('soc_start', 'CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS')
                if soc_start in self.fl or "soc" in getattr(self, 'input', '').lower():
                    if soc_start in self.fl:
                        self.soc = True
                        self.tddft_soc = get_tddft_soc(self.fl, self.reg, self.tddft_states)
            except Exception as e:
                raise ParseError(f"Failed to parse TDDFT data in {self.fname}: {e}") from e

        if 'nbo' in self.__dict__ or self.has_nbo():
            try:
                self.nbo = NBOAnalysis(self.fl, cycle=-1)
            except Exception as e:
                self.errors.add("nbo", f"Failed to parse NBO analysis: {e}")
                self.nbo = None

        if 'eprnmr' in self.__dict__ or self.has_nmr():
            try:
                self.nmr = NMRAnalysis(self.fl, self.geom, self.atoms)
            except Exception as e:
                self.errors.add("nmr", f"Failed to parse NMR analysis: {e}")
                self.nmr = None

    def has_nmr(self) -> bool:
        """Check if output file contains NMR analysis."""
        return 'CHEMICAL SHIELDING SUMMARY (ppm)' in self.fl or 'NMR SPIN-SPIN COUPLING CONSTANTS' in self.fl

    def has_nbo(self) -> bool:
        """Check if output file contains NBO analysis."""
        return 'NATURAL BOND ORBITAL ANALYSIS:' in self.fl or 'Summary of Natural Population Analysis:' in self.fl

    def _get_trajectory(self):
        trj = []
        for i in self.opt_sections:
            _, gs_ = get_geom(i, self.reg)
            trj.append(gs_)
        return np.array(trj, dtype=float)

    def _energy_trajectory(self):
        trj = []
        for i in self.opt_sections:
            trj.append(get_energy_decomposition(i, self.reg))
        return trj

    def _parse_input(self):
        try:
            self.basis_set = re.findall(self.reg['basis_set'], self.fl)[0]
            raw_input = re.findall(self.reg['input'], self.fl)[0]
            self.input = '\n'.join(re.findall(r'> ([\S\s]*?)\n', raw_input))

            commands = r' '.join(re.findall(r'!([\S\s]*?)\n', raw_input)).lower().split()
            re_blocks = rf"%\s*({'|'.join(self.reg['blocks'])})\b([^%]*?)end\b"
            self.blocks = re.findall(re_blocks, self.input.replace('\r\n', '\n'))
            for block_name, _ in self.blocks:
                setattr(self, block_name.strip().lower(), True)

            theory = []
            for i in commands:
                if i in self.reg['commands']:
                    setattr(self, i, True)
                elif i not in self.basis_set.lower() and 'cpcm' not in i and 'smd' not in i:
                    theory.append(i)

            self.functional = ' '.join(theory)
            self.theory = f'{self.functional}/{self.basis_set}'

            geom_input = re.findall(self.reg['chrg_mult'], raw_input.lower())[0]
            self.charge, self.mult = int(geom_input[2]), int(geom_input[3])

        except (IndexError, KeyError, AttributeError) as e:
            raise ParseError(f"Failed to parse input block in {self.fname}: {e}") from e

    @property
    def homo(self):
        """Return the 0-based index of the Highest Occupied Molecular Orbital (HOMO), or None."""
        self._ensure_parsed()
        if hasattr(self, 'mult') and self.mult != 1 and hasattr(self, 'alpha_occ') and self.alpha_occ is not None:
            occ = np.asarray(self.alpha_occ)
            occ_idx = np.where(occ > 0.001)[0]
            if len(occ_idx) > 0:
                return int(occ_idx[-1])
        elif hasattr(self, 'occupation') and self.occupation is not None and len(self.occupation) > 0:
            occ = np.asarray(self.occupation)
            occ_idx = np.where(occ > 0.001)[0]
            if len(occ_idx) > 0:
                return int(occ_idx[-1])

        # Parse from ORBITAL ENERGIES block if available
        pos = self.fl.rfind('ORBITAL ENERGIES')
        if pos != -1:
            chunk = self.fl[pos:pos + 100000]
            occ_matches = re.findall(r'^\s*(\d+)\s+([0-9.]+)\s+[-+]?[0-9.]+', chunk, re.M)
            occupied = [int(m[0]) for m in occ_matches if float(m[1]) > 0.001]
            if occupied:
                return occupied[-1]

        # Parse from N(Alpha)
        nalpha = re.findall(r'N\(Alpha\)\s*:\s*([0-9.]+)', self.fl)
        if nalpha:
            return int(round(float(nalpha[-1]))) - 1

        # Parse from NEL
        nel = re.findall(r'Number of Electrons\s+NEL\s+\.\.\.\.\s+(\d+)', self.fl)
        if nel:
            return (int(nel[-1]) // 2) - 1

        return None

    @property
    def beta_homo(self):
        """Return the 0-based index of the Highest Occupied Beta Molecular Orbital, or None."""
        self._ensure_parsed()
        if hasattr(self, 'mult') and self.mult != 1 and hasattr(self, 'beta_occ') and self.beta_occ is not None:
            occ = np.asarray(self.beta_occ)
            occ_idx = np.where(occ > 0.001)[0]
            if len(occ_idx) > 0:
                return int(occ_idx[-1])
        return None

    @property
    def lumo(self):
        """Return the 0-based index of the Lowest Unoccupied Molecular Orbital (LUMO), or None."""
        homo_idx = self.homo
        return homo_idx + 1 if homo_idx is not None else None

    def to_ase(self):
        """Convert calculation geometry to an ASE Atoms object."""
        from ase import Atoms
        return Atoms(symbols=self.atoms, positions=self.geom)

    def to_rdkit(self, charge=None):
        """Convert calculation geometry to an RDKit Mol object."""
        from rdkit import Chem
        from rdkit.Chem import rdDetermineBonds

        mol = Chem.RWMol()
        conf = Chem.Conformer()
        for i, (sym, pos) in enumerate(zip(self.atoms, self.geom)):
            mol.AddAtom(Chem.Atom(str(sym)))
            conf.SetAtomPosition(i, (float(pos[0]), float(pos[1]), float(pos[2])))
        mol = mol.GetMol()
        mol.AddConformer(conf)

        chg = 0
        if charge is not None:
            chg = charge
        elif hasattr(self, 'charge') and self.charge is not None:
            try:
                chg = int(self.charge)
            except (ValueError, TypeError):
                chg = 0

        try:
            rdDetermineBonds.DetermineBonds(mol, charge=chg)
        except Exception as e:
            logger.debug("Failed to determine bonds with RDKit: %s", e)
        return mol
