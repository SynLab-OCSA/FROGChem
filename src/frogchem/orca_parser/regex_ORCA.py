
REGEXS = {

    '6': {
        # basic
        'input': r'INPUT FILE([\S\s]*?)END OF INPUT',
        'commands': ['opt', 'freq', 'irc', 'neb', 'cpcm', 'smd', 'optts', 'printmos', 'printbasis', 'nbo'],
        'blocks': ['geom', 'freq', 'irc', 'neb', 'tddft', 'pal', 'cpcm', 'basis'],
        'basis_set': r'Your calculation utilizes the basis: ([\S\s]*?)\n',
        'opt_section': r"GEOMETRY OPTIMIZATION CYCLE([\S\s]*?)ORCA GEOMETRY RELAXATION STEP",
        'chrg_mult' : r'(?:\*(\s){0,}(xyz|int|xyzfile)\s+)(-?\d+)\s+(\d+)',
        'rot_const' : r'Rotational constants in cm-1:\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)',

        # opt
        'start_geom' : r'CARTESIAN COORDINATES \(ANGSTROEM\)([\S\s]*?)\n\n',
        "SCF_energy": r"TOTAL SCF ENERGY\n----------------\n([\S\s]*?)---",
        "decomp_energy" : r'(.*)\s*:\s+(-?\d+\.\d+)',
        "disp_energy" : r'(Dispersion correction)\s*\s+(-?\d+.\d+)',
        "total_energy": r'Total Energy',
        "dispersion" : r"DFT DISPERSION CORRECTION([\S\s]*?)\n\n\n",
        "final_single_point_energy": r'FINAL SINGLE POINT ENERGY\s*(-?\d*.\d*)',

        # orbitals
        # the block ends at the auxiliary basis (RI jobs) or at the first double blank line (NoRI jobs
        # print no auxiliary basis); elements inside the block are separated by a single blank line
        'basis_coefficients': r'BASIS SET IN INPUT FORMAT([\S\s]*?)(?:AUXILIARY|\n[ \t]*\n[ \t]*\n)',
        'orbital_energies': r'ORBITAL ENERGIES([\S\s]*?)MOLECULAR ORBITALS', 
        'orbitals': r'MOLECULAR ORBITALS (?:\(RHF, ROHF\)|\(UHF\))([\S\s]*?)\n\n\n',
        
        # hessian
        'frequency': r'VIBRATIONAL FREQUENCIES([\S\s]*?)\n\n\n',
        'normal_modes': r'NORMAL MODES([\S\s]*?)\n\n\n', 
        'ir_graph' : ' Mode   freq       eps      Int      T**2         TX        TY        TZ',
        'vcd_graph' : 'Mode   Freq    VCD-Intensity',
        'ir_imp_idx' : 3,
        'vcd_imp_idx' : 2,

        # tddft
        'tddft_singlets_start': 'TD-DFT EXCITED STATES (SINGLETS)',
        'tddft_singlets_end': 'TD-DFT-EXCITATION SPECTRA',  
        'tddft_singlets_end_if_soc': '********************************',  
        'tddft_triplets_start': 'TD-DFT EXCITED STATES (TRIPLETS)', 
        'tddft_triplets_end': 'TD-DFT SPIN-ORBIT COUPLING', 
        'state_pattern' : r"STATE\s+(\d+):\s+E=\s+[-+]?\d*\.\d+ au\s+([-+]?\d+\.\d+) eV\s+.*?<S\*\*2>\s+=\s+([-+]?\d+\.\d+)\s+Mult (\d+)",
        'transition_pattern': r"(\d+[a-zA-Z]?)\s*->\s*(\d+[a-zA-Z]?)\s*:\s*([-+]?\d+\.\d+)",
        'soc_start' : "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS",
        'soc_end' : "SOC stabilization",
        'pattern_soc': r"\s+(\d+)\s+(\d+)" +
        r"\s+\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)" +
        r"\s+\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)" +
        r"\s+\(\s*([-+]?\d+\.\d+)\s*,\s*([-+]?\d+\.\d+)\s*\)", 

        # Population Analysis
        'mulliken': r'MULLIKEN ATOMIC CHARGES([\S\s]*?)Sum of atomic charges',
        # header (optionally '... AND SPIN POPULATIONS'), its dashed underline, then the table up to the first blank line
        'loewdin': r'LOEWDIN ATOMIC CHARGES[^\n]*\n-+\n([\S\s]*?)\n[ \t]*\n',
    },
    
}
