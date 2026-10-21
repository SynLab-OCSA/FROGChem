
ORB_TYPE = {
    's': 0,
    'px': 1,
    'py': 2,
    'pz': 3,
    'dz2': 4,
    'dxz': 5,
    'dyz': 6,
    'dx2y2': 7,
    'dxy': 8,
    "f0": 9,
    "f+1": 10,
    "f-1": 11,
    "f+2": 12,
    "f-2": 13,
    "f+3": 14,
    "f-3": 15
}

# Mappa che associa i tipi numerici di funzione ai nomi e parametri angolari
ORB_POLINOMIAL = {
    #Orbitale s
    0: ('s', 0, 0, 0),  # s

    # Orbitali p
    1: ('px', 1, 0, 0),  # px 
    2: ('py', 0, 1, 0),  # py 
    3: ('pz', 0, 0, 1),  # pz 

    # Orbitali d
    4: ('dz2', 0, 0, 2),      # dz^2
    5: ('dxz', 1, 0, 1),      # dxz
    6: ('dyz', 0, 1, 1),      # dyz
    7: ('dx2y2', 2, -2, 0),   # dx^2-y^2
    8: ('dxy', 1, 1, 0),      # dxy

    # Orbitali f
    9: ('f0', 3, 0, 0),       # f0 
    10: ('f+1', 2, 1, 0),     # f+1 
    11: ('f-1', 1, 2, 0),     # f-1 
    12: ('f+2', 3, -1, 0),    # f+2 
    13: ('f-2', 0, 3, -1),    # f-2 
    14: ('f+3', 3, 0, -3),    # f+3
    15: ('f-3', 0, -3, +3)    # f-3
}

# Cartesian d/f functions (Gaussian 6D/10F), in Gaussian's order. Each one is the plain monomial
# x^a y^b z^c, normalised on its own (the diagonal of the Gaussian overlap matrix is 1).
CARTESIAN_EXPONENTS = {
    'd_xx': (2, 0, 0), 'd_yy': (0, 2, 0), 'd_zz': (0, 0, 2),
    'd_xy': (1, 1, 0), 'd_xz': (1, 0, 1), 'd_yz': (0, 1, 1),
    'f_xxx': (3, 0, 0), 'f_yyy': (0, 3, 0), 'f_zzz': (0, 0, 3), 'f_xyy': (1, 2, 0), 'f_xxy': (2, 1, 0),
    'f_xxz': (2, 0, 1), 'f_xzz': (1, 0, 2), 'f_yzz': (0, 1, 2), 'f_yyz': (0, 2, 1), 'f_xyz': (1, 1, 1),
}
CARTESIAN_CODES = {name: 16 + i for i, name in enumerate(CARTESIAN_EXPONENTS)}
ORB_TYPE.update(CARTESIAN_CODES)
ORB_POLINOMIAL.update({code: (name, *CARTESIAN_EXPONENTS[name]) for name, code in CARTESIAN_CODES.items()})