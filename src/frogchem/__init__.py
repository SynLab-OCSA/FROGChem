import re
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

# Version is defined exclusively in pyproject.toml
try:
    __version__ = version("frogchem")
except PackageNotFoundError:
    _pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if _pyproject.exists():
        try:
            import tomllib
            with open(_pyproject, "rb") as _f:
                __version__ = tomllib.load(_f).get("project", {}).get("version", "0.0.0")
        except Exception:
            try:
                _text = _pyproject.read_text(encoding="utf-8")
                _match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', _text)
                __version__ = _match.group(1) if _match else "0.0.0"
            except Exception:
                __version__ = "0.0.0"
    else:
        __version__ = "0.0.0"

# Lazy top-level classes: avoid importing matplotlib, pandas, seaborn on cold start
_TOP_LEVEL_MAP = {
    'Calculation': ('frogchem.orca_parser.calculation', 'Calculation'),
    'OrcaCalculation': ('frogchem.orca_parser.calculation', 'Calculation'),
    'GaussianCalculation': ('frogchem.gaussian_parser.calculation', 'GaussianCalculation'),
    'ReactionPlotter': ('frogchem.reaction_plotter.plotter', 'ReactionPlotter'),
    'ReactionProfileAnalyzer': ('frogchem.reaction_plotter.profile', 'ReactionProfileAnalyzer'),
    'NMRAnalysis': ('frogchem.orca_parser.nmr', 'NMRAnalysis'),
    'get_nmr_equivalencies': ('frogchem.nmr', 'get_nmr_equivalencies'),
    'convert_shielding_to_shift': ('frogchem.nmr', 'convert_shielding_to_shift'),
    'resolve_reference_shielding': ('frogchem.nmr', 'resolve_reference_shielding'),
    'extract_reference_shieldings': ('frogchem.nmr', 'extract_reference_shieldings'),
    'symmetrize_nmr_data': ('frogchem.nmr', 'symmetrize_nmr_data'),
    'simulate_quantum_transitions': ('frogchem.nmr_sim', 'simulate_quantum_transitions'),
    'simulate_1d_spectrum': ('frogchem.nmr_sim', 'simulate_1d_spectrum'),
    'simulate_fid': ('frogchem.nmr_sim', 'simulate_fid'),
    'simulate_1d_from_fid': ('frogchem.nmr_sim', 'simulate_1d_from_fid'),
    'simulate_cosy': ('frogchem.nmr_sim', 'simulate_cosy'),
    'simulate_noesy': ('frogchem.nmr_sim', 'simulate_noesy'),
}


def read_orca(filename):
    """Parse an ORCA output file (.out, .out.gz)."""
    from frogchem.orca_parser.calculation import Calculation
    return Calculation(filename)


def read_gaussian(filename):
    """Parse a Gaussian output file (.log, .out, .log.gz)."""
    from frogchem.gaussian_parser.calculation import GaussianCalculation
    return GaussianCalculation(filename)


def read(filename):
    """Auto-detect quantum chemistry output format (ORCA or Gaussian) and parse.

    Args:
        filename: Path to output or log file (.out, .log, .out.gz, etc.)

    Returns:
        OrcaCalculation or GaussianCalculation
    """
    path_str = str(filename).lower()
    p = Path(filename)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    header = ""
    try:
        if path_str.endswith(".gz"):
            import gzip
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                header = "".join(f.readline() for _ in range(100))
        else:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                header = "".join(f.readline() for _ in range(100))
    except Exception:
        pass

    if "O   R   C   A" in header or "Program Version" in header:
        return read_orca(filename)
    elif "Gaussian" in header or "Entering Gaussian System" in header:
        return read_gaussian(filename)

    if path_str.endswith((".log", ".log.gz")):
        return read_gaussian(filename)
    return read_orca(filename)


# Lazy submodule access via __getattr__ avoids cold-start penalty
# while enabling `from frogchem import thermo` etc.
_SUBMODULES = [
    'chemistry', 'chimera_commands', 'density_builder', 'exceptions',
    'graph_theory', 'grid', 'gui_utils', 'io', 'math_geometry',
    'parser_base', 'printer', 'solvent_surface', 'substituter',
    'symmetry', 'thermo', 'ensemble', 'nmr', 'nmr_sim',
]

__all__ = _SUBMODULES + [
    '__version__',
    'Calculation',
    'OrcaCalculation',
    'GaussianCalculation',
    'ReactionPlotter',
    'ReactionProfileAnalyzer',
    'read',
    'read_orca',
    'read_gaussian',
    'NMRAnalysis',
    'get_nmr_equivalencies',
    'convert_shielding_to_shift',
    'symmetrize_nmr_data',
    'simulate_quantum_transitions',
    'simulate_1d_spectrum',
    'simulate_cosy',
    'simulate_noesy',
    'orca_parser',
    'gaussian_parser',
    'reaction_plotter',
]


def __getattr__(name: str):
    if name in _TOP_LEVEL_MAP:
        import importlib
        mod_name, attr_name = _TOP_LEVEL_MAP[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    if name in _SUBMODULES:
        import importlib
        mod = importlib.import_module(f'frogchem.{name}')
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'frogchem' has no attribute {name!r}")
