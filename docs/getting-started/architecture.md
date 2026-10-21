# Architecture & Design Principles

FROG is engineered to be lightweight, modular, and fast on both personal workstations and high-performance computing (HPC) clusters.

---

## 1. Lazy Loading & Fast Startup

Computational chemistry scripts and command-line utilities often require quick startup times (e.g., extracting a single value or piping JSON in shell scripts). Heavy scientific packages like `matplotlib`, `pandas`, `seaborn`, `torch`, or `rdkit` can take seconds to import.

To ensure instant CLI execution:

* Top-level modules and classes in FROG are resolved dynamically via `__getattr__` in `src/frogchem/__init__.py`.
* Heavy visualization or parsing submodules are only loaded when explicitly referenced.
* CLI tools parse arguments first before importing domain-specific libraries.

```text
[User Script / CLI Call]
          │
          ▼
    import frogchem  (Instant startup, minimal imports)
          │
          ├──> __getattr__ dynamic dispatch
          │       │
          │       ├──> frogchem.orca_parser.calculation (on-demand)
          │       ├──> frogchem.thermo (on-demand)
          │       ├──> frogchem.nmr / frogchem.nmr_sim (on-demand)
          │       └──> frogchem.reaction_plotter (on-demand)
```

---

## 2. Calculation Object Abstraction

The central abstraction for quantum chemistry output parsing is the `Calculation` class (for ORCA) and `GaussianCalculation` class (for Gaussian). Both provide standardized, cached properties:

* **Energy**: `calc.final_energy`, `calc.scf_energy`, `calc.dispersion_correction`.
* **Geometry**: `calc.atoms` (chemical symbols), `calc.coordinates` (numpy array in Å), `calc.initial_coordinates`.
* **Vibrational analysis**: `calc.frequencies`, `calc.rotational_constants`, `calc.molecular_weight`, `calc.zero_point_energy`.
* **Electronic properties**: `calc.mulliken_charges`, `calc.loewdin_charges`, `calc.spin_densities`, `calc.dipole_moment`.
* **Excited states**: `calc.transitions`, `calc.absorption_spectrum`.
* **NMR data**: `calc.nmr_shieldings`, `calc.nmr_couplings`.

Properties are computed lazily and cached upon first access, avoiding redundant file scans.

---

## 3. Universal Interoperability (ASE & RDKit)

Rather than inventing proprietary molecular data structures, FROG natively converts parsed geometries and electronic properties to industry-standard objects:

```python
import frogchem as frog

calc = frog.read("opt.out")

# Convert to Atomic Simulation Environment (ASE) Atoms
atoms = calc.to_ase()
print(f"ASE formula: {atoms.get_chemical_formula()}")

# Convert to RDKit Mol with 3D Conformer & Bonds
mol = calc.to_rdkit()
print(f"RDKit atoms: {mol.GetNumAtoms()}")
```

**Output atteso:**
```text
ASE formula: C7H8O
RDKit atoms: 16
```

---

## 4. Standalone Ensemble & Boltzmann Weighting

All ensemble analysis routines in `frogchem.ensemble` are completely standalone and do not depend on external ensemble packages. They can parse:

1. Standard JSON checkpoints (such as from `ensemble_analyzer`).
2. Flat conformer weight dictionaries (`{"1": 0.45, "2": 0.55}`).
3. Two-column CSV/TSV tables (`conf_id, population`).

Every CLI tool that extracts atomic or molecular properties supports automatic Boltzmann population averaging via the `-c / --checkpoint` flag.

---

## 5. Standardized CLI Interface

All 33 CLI tools in FROG share a uniform argument pattern:

* **Positional inputs**: One or more input files or patterns (e.g. `*.out`, `*.log.gz`).
* **Ensemble flags**: `-c / --checkpoint`, `--read-boltzmann`, `--individual`.
* **Formatting**: `--format {table,json,csv}` for automated scripting and Unix pipelines.
* **Verbosity**: `-v / --verbose` (debug) and `-q / --quiet` (suppress headers).
* **Output redirection**: `-o / --output` or `--out-file`.
