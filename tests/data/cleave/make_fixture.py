"""Regenerate water.orbitals.json (needs cleave + PySCF; not run by the test suite).

Water, B3LYP/def2-TZVP (f functions on O), three sets written with cleave's own writer:
``scf`` (RKS, default), ``cation`` (UKS doublet, α/β channels) and ``almo`` (the RKS occupied
space mixed by a non-orthogonal transformation: same density, overlap σ ≠ 1). The ``check``
block holds PySCF reference values (Bohr) for the frogchem tests.

    python tests/data/cleave/make_fixture.py
"""
from pathlib import Path

import numpy as np
from pyscf import dft, gto

from almoeda import Method
from almoeda import orbitals as orb

mol = gto.M(atom='O 0 0 0.12; H 0.76 0.05 -0.48; H -0.74 0 -0.5', basis='def2-tzvp', verbose=0)
rks = dft.RKS(mol, xc='b3lyp').run(conv_tol=1e-10)
cat = mol.copy()
cat.charge, cat.spin = 1, 1
cat.build()
uks = dft.UKS(cat, xc='b3lyp').run(conv_tol=1e-10)

s = mol.intor_symmetric('int1e_ovlp')
occ = rks.mo_coeff[:, rks.mo_occ > 0]
t = np.eye(occ.shape[1]) + 0.3 * np.random.default_rng(7).standard_normal((occ.shape[1],) * 2)
almo = occ @ t
sets = {
    'scf': orb.orbital_set('canonical', 'RKS water', orb._canonical(rks)),
    'cation': orb.orbital_set('canonical', 'UKS water cation', orb._canonical(uks)),
    'almo': orb.orbital_set('almo', 'occupied RKS orbitals, non-orthogonal mix',
                            [orb.channel(almo, np.full(almo.shape[1], 2.0), s=s)],
                            orthonormal=False),
}
data = orb.payload(mol, sets, program='test', source='water', method=Method('b3lyp', 'def2-tzvp'),
                   default_set='scf')

pts = np.random.default_rng(0).normal(scale=1.5, size=(20, 3)) + mol.atom_coords().mean(axis=0)
ao = mol.eval_gto('GTOval_sph', pts)
rho = lambda dm: dft.numint.eval_rho(mol, ao, dm)          # noqa: E731
da, db = uks.make_rdm1()
data['check'] = {
    'points_bohr': pts.tolist(),
    'scf_orbitals_0_4_5': (ao @ rks.mo_coeff[:, [0, 4, 5]]).T.tolist(),
    'scf_density': rho(rks.make_rdm1()).tolist(),
    'cation_density': rho(da + db).tolist(),
    'cation_spin_density': rho(da - db).tolist(),
}
out = Path(__file__).with_name('water.orbitals.json')
orb.write_orbitals(out, data)
print(f'{out}: {out.stat().st_size / 1024:.0f} KB')
