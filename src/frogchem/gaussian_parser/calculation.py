import re
import logging
import numpy as np

from frogchem.parser_base import BaseParser
from frogchem.exceptions import ParseError
from frogchem.chemistry import get_atomic_weights, atomic_number_to_symbol
from frogchem.gaussian_parser.regex_gaussian import REGEXS
from frogchem.gaussian_parser.opt import parse_orientation
from frogchem.gaussian_parser.hessian import get_freq, get_ir_intensities
from frogchem.gaussian_parser.tddft import parse_uv, parse_ecd, parse_tddft_states
from frogchem.gaussian_parser.population import parse_npa_charges, parse_atomic_charges
from frogchem.gaussian_parser.orbitals import get_MOs, get_overlap, get_basis_coefficients

logger = logging.getLogger(__name__)

CONVERT_B = {'GHz': 29.9792458}  # 1 cm-1 = c[cm/s] * 1e-9 GHz = 29.9792458 GHz (exact)


class GaussianCalculation(BaseParser):

    def __init__(self, fname):
        super().__init__(fname)
        self.reg = REGEXS

    def _parse_file(self):
        self.atoms, self.geom = self.parse_geom()
        self.charge, self.mult = self.get_charge_mult()
        self.B = self.get_B()
        self.m = self.get_dipole()
        self.mw = self.calc_mw()
        self.final_energy = self.get_final_energy()

        if self.has_freq():
            self.freq = get_freq(self.fl, self.reg)
            self.ir_spectrum = get_ir_intensities(self.fl, self.reg)

        if self.has_tddft():
            self.uv = parse_uv(self.fl, self.reg)
            self.ecd = parse_ecd(self.fl, self.reg)
            # Same interface as the ORCA parser, used by frog-tddft
            self.tddft_states = parse_tddft_states(self.fl, self.reg)
            self.tddft = bool(self.tddft_states)

        self.npa_charges = parse_npa_charges(self.fl, self.reg)
        # Same attributes as the ORCA parser ({idx: {'atom', 'charge'[, 'spin']}}, None if not printed)
        self.mulliken = parse_atomic_charges(self.fl, self.reg, 'mulliken')
        self.loewdin = parse_atomic_charges(self.fl, self.reg, 'loewdin')

        self._parse_orbitals()

    def _parse_orbitals(self):
        """MO coefficients (pop=full), basis set (gfinput) and AO overlap (iop(3/33=1)), ORCA layout.

        orbital_coefficients is n_bf x n_MO (alpha then beta columns for open shell), with
        occupation / orbital_energies (Eh) in the same order and alpha_/beta_coeff, alpha_/beta_occ
        for open-shell runs, as in orca_parser.Calculation. A block FROG cannot use (e.g. g
        functions) is reported in self.errors and leaves the attributes unset.
        """
        try:
            mos = get_MOs(self.fl)
        except ParseError as e:
            self.errors.add(str(self.fname), 'orbitals', f'MO coefficients not usable: {e}', section='orbitals')
            logger.warning(f'{self.fname}: MO coefficients not usable: {e}')
            return
        if mos is None:
            return

        n_alpha, n_beta = self.n_electrons
        self.mask_atoms, self.mask_type_n = mos['mask_atoms'], mos['mask_type_n']
        self.mask_orbital_type = mos['mask_orbital_type']
        alpha_coeff, alpha_en = mos['alpha']
        alpha_occ = (np.arange(alpha_coeff.shape[1]) < n_alpha).astype(float)
        if 'beta' in mos or n_alpha != n_beta:
            # Unrestricted (or restricted open shell, whose single MO set serves both spins)
            beta_coeff, beta_en = mos.get('beta', mos['alpha'])
            beta_occ = (np.arange(beta_coeff.shape[1]) < n_beta).astype(float)
            self.orbital_coefficients = np.hstack((alpha_coeff, beta_coeff))
            self.orbital_energies = np.concatenate((alpha_en, beta_en))
            self.occupation = np.concatenate((alpha_occ, beta_occ))
            if self.mult != 1:
                self.alpha_coeff, self.beta_coeff = alpha_coeff, beta_coeff
                self.alpha_occ, self.beta_occ = alpha_occ, beta_occ
        else:
            self.orbital_coefficients = alpha_coeff
            self.orbital_energies = alpha_en
            self.occupation = 2.0 * alpha_occ

        self.overlap = get_overlap(self.fl, mos['signs'])
        try:
            self.basis_coeff = get_basis_coefficients(self.fl, self.atoms)
        except ParseError as e:
            self.errors.add(str(self.fname), 'basis', str(e), section='orbitals')
            logger.warning(f'{self.fname}: {e}')

    @property
    def n_electrons(self):
        """(n_alpha, n_beta) from 'N alpha electrons  M beta electrons' (last job)."""
        self._ensure_parsed()
        match = re.findall(r'(\d+)\s+alpha electrons\s+(\d+)\s+beta electrons', self.fl)
        if not match:
            raise ParseError(f"No 'alpha electrons ... beta electrons' line in {self.fname}")
        return int(match[-1][0]), int(match[-1][1])

    def parse_geom(self):
        """Element symbols and coordinates (Å) of the last geometry printed in the file."""
        numbers, coords = parse_orientation(self.fl, self.reg)
        if len(coords) == 0:
            raise ParseError(f"No geometry found in {self.fname} "
                             f"(none of: {', '.join(self.reg['geom_headers'])})")
        return np.array([atomic_number_to_symbol(int(z)) for z in numbers]), coords

    def get_charge_mult(self):
        """Total charge and spin multiplicity of the last job (same attributes as the ORCA parser)."""
        match = re.findall(self.reg['charge_mult'], self.fl)
        if not match:
            logger.warning(f"No 'Charge = ... Multiplicity = ...' line in {self.fname}: assuming 0 and 1")
            return 0, 1
        charge, mult = match[-1]
        return int(charge), int(mult)

    def get_final_energy(self):
        match = re.findall(self.reg['energy'], self.fl)
        if not match:
            return 0.0
        return float(match[-1])

    def get_B(self):
        match = re.findall(self.reg['rot_const'], self.fl)
        if not match:
            # Same convention as the ORCA parser: no rotational constants -> None
            # (thermo.free_gibbs_energy warns instead of silently using a fake rotor).
            return None
        B = np.array(match[-1], dtype=float)
        B /= CONVERT_B.get(self.reg['units_B'], 1.0)
        return B

    def get_dipole(self):
        if self.reg['dipole_start'] not in self.fl:
            return np.array([1.0, 0.0, 0.0])
        block = self.fl.split(self.reg['dipole_start'])[-1]
        block = block.split(self.reg['dipole_end'])[0]
        match = re.findall(self.reg['dipole_pattern'], block)
        if not match:
            return np.array([1.0, 0.0, 0.0])
        return np.array(match[-1], dtype=float)

    def calc_mw(self):
        if len(self.atoms) == 0:
            raise ParseError(f"No atoms to calculate molecular weight in {self.fname}")
        return sum(get_atomic_weights(i) for i in self.atoms)

    def has_freq(self):
        return self.reg['freq_section'] in self.fl

    def has_tddft(self):
        return self.reg['tddft_start'] in self.fl

    def opt_done(self):
        return len(re.findall(self.reg['opt_done'], self.fl)) >= 1

    def normal_termination(self):
        return len(re.findall(self.reg['finish'], self.fl)) >= 1

    def to_ase(self):
        """Convert calculation geometry to an ASE Atoms object."""
        from ase import Atoms
        return Atoms(symbols=self.atoms, positions=self.geom)

    def to_rdkit(self, charge: int = 0):
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

        try:
            rdDetermineBonds.DetermineBonds(mol, charge=charge)
        except Exception:
            pass
        return mol

    def has_nbo(self) -> bool:
        """Check if output file contains NBO analysis."""
        return 'NATURAL BOND ORBITAL ANALYSIS:' in self.fl or 'Summary of Natural Population Analysis:' in self.fl

    @property
    def homo(self):
        """Return the 0-based index of the Highest Occupied Molecular Orbital (HOMO), or None."""
        self._ensure_parsed()
        alpha_match = re.findall(r'(\d+)\s+alpha electrons', self.fl)
        if alpha_match:
            return int(alpha_match[-1]) - 1
        nalpha = re.findall(r'NAlpha\s*=\s*(\d+)', self.fl)
        if nalpha:
            return int(nalpha[-1]) - 1
        return None

    @property
    def beta_homo(self):
        """Return the 0-based index of the highest occupied beta MO (open shell), or None."""
        self._ensure_parsed()
        if self.__dict__.get('beta_occ') is None:
            return None
        occ_idx = np.where(np.asarray(self.beta_occ) > 0.001)[0]
        return int(occ_idx[-1]) if len(occ_idx) else None

    @property
    def lumo(self):
        """Return the 0-based index of the Lowest Unoccupied Molecular Orbital (LUMO), or None."""
        h = self.homo
        return h + 1 if h is not None else None

    @property
    def nbo(self):
        """Lazy access to NBO analysis if present in the output."""
        if not hasattr(self, '_nbo_obj'):
            self._ensure_parsed()
            if self.has_nbo():
                from frogchem.orca_parser.nbo import NBOAnalysis
                try:
                    self._nbo_obj = NBOAnalysis(self.fl, cycle=-1)
                except Exception as e:
                    logger.warning(f"Failed to parse NBO in {self.fname}: {e}")
                    self._nbo_obj = None
            else:
                self._nbo_obj = None
        return self._nbo_obj
