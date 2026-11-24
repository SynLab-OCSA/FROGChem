#!/usr/bin/env python3
"""Validate DFT level of theory (functional, basis set coverage on XYZ, and dispersion models)."""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import basis_set_exchange as bse
import periodictable as pt
import pylibxc
from dftd3.interface import (
    ModifiedRationalDampingParam as D3MBJParam,
    ModifiedZeroDampingParam as D3MZeroParam,
    RationalDampingParam as D3BJParam,
    ZeroDampingParam as D3ZeroParam,
)
from dftd4.interface import DampingParam as D4Param

from frogchem.basis_exchange import resolve_basis_name
from frogchem.filetypes import load_structures
from frogchem.printer import print_error, print_info, print_table, print_title


def extract_unique_elements(xyz_path: str | Path) -> List[str]:
    """Load all frames from an XYZ (or .out/.log) file and return unique element symbols."""
    structures = load_structures(xyz_path)
    unique_syms: Set[str] = set()
    for atoms, _ in structures:
        for atom in atoms:
            unique_syms.add(atom.capitalize())
    return sorted(unique_syms)


def check_functional_libxc(func_name: str) -> Tuple[bool, List[str]]:
    """Verify if a functional name matches an exact Libxc string or substring."""
    available = pylibxc.util.xc_available_functional_names()
    target = func_name.strip().lower().replace("-", "_")

    if target in available:
        return True, [target]

    # Match standard Libxc naming convention (e.g., 'b3lyp' -> 'hyb_gga_xc_b3lyp')
    pattern = re.compile(rf"(^|_){re.escape(target)}($|_)")
    matches = sorted([name for name in available if pattern.search(name)])
    return len(matches) > 0, matches


def check_basis_set(basis_name: str, elements: List[str]) -> Tuple[bool, Optional[str], List[str]]:
    """Verify if a basis set exists in BSE and covers all given elements."""
    resolved = resolve_basis_name(basis_name.strip())

    if resolved is None:
        return False, None, elements

    # Check each element directly against BSE so schema differences do not cause false negatives
    missing: List[str] = []
    for sym in elements:
        el = getattr(pt, sym, None)
        if el is None:
            missing.append(sym)
            continue
        try:
            bse.get_basis(resolved, elements=[el.number])
        except Exception:
            missing.append(sym)

    return True, resolved, missing


def _format_damping_params(param_obj) -> str:
    """Format damping parameters safely across dftd3 and dftd4 Python bindings."""
    if isinstance(param_obj, dict):
        keys = [k for k in ("s6", "s8", "a1", "a2", "s9") if k in param_obj]
        if keys:
            return "Supported (" + ", ".join(f"{k}={param_obj[k]:.3f}" for k in keys) + ")"
    for attr in ("s6", "s8", "a1", "a2", "s9"):
        if hasattr(param_obj, attr):
            val = getattr(param_obj, attr)
            if isinstance(val, (int, float)):
                return f"Supported ({attr}={val:.3f})"
    return "Supported"


def check_dispersions(func_name: str, libxc_matches: List[str]) -> Dict[str, str]:
    """Check dispersion parameter availability for a functional across D3, D4, and VV10/NL."""
    clean_name = re.sub(
        r"[-_](d2|d3|d3bj|d3zero|d3m|d3mbj|d3mzero|d4|nl|v)$",
        "",
        func_name.strip().lower(),
    )
    results: Dict[str, str] = {
        "D2": "Not available in simple-dftd3/dftd4 (legacy)"
    }

    d3_models = {
        "D3(0)": D3ZeroParam,
        "D3(BJ)": D3BJParam,
        "D3M(0)": D3MZeroParam,
        "D3M(BJ)": D3MBJParam,
    }
    for label, param_cls in d3_models.items():
        try:
            p = param_cls(method=clean_name)
            results[label] = _format_damping_params(p)
        except Exception as exc:
            results[label] = f"Not parametrized ({exc})"

    try:
        p4 = D4Param(method=clean_name)
        results["D4"] = _format_damping_params(p4)
    except Exception as exc:
        results["D4"] = f"Not parametrized ({exc})"

    vv10_built_in = any(("vv10" in m) or m.endswith("_v") for m in libxc_matches)
    results["VV10 / NL"] = (
        "Built-in (Libxc NL functional)"
        if vv10_built_in
        else "External/Software-specific (no standalone Python API)"
    )

    return results

def validate_method(xyz_path: str, functional: str, basis: str) -> None:
    """Run full validation and print results using FROG printer conventions."""
    print_title("DFT Method & Basis Set Validator")

    elements = extract_unique_elements(xyz_path)
    print_info(f"Structure: {xyz_path} | Unique elements: {', '.join(elements)}")

    # 1. Check Basis Set
    b_exists, resolved_b, missing_el = check_basis_set(basis, elements)
    if not b_exists:
        b_status = f"Not found in BSE ('{basis}')"
    elif missing_el:
        b_status = f"Found ({resolved_b}), MISSING elements: {', '.join(missing_el)}"
    else:
        b_status = f"Valid ({resolved_b}) — covers all {len(elements)} elements"

    # 2. Check Functional in Libxc
    f_exists, f_matches = check_functional_libxc(functional)
    if f_exists:
        preview = ", ".join(f_matches[:4]) + (" ..." if len(f_matches) > 4 else "")
        f_status = f"Found ({len(f_matches)} match): {preview}"
    else:
        f_status = f"Not found in Libxc ('{functional}')"

    # 3. Check Dispersions
    disp_map = check_dispersions(functional, f_matches)

    rows = [
        ["Basis Set", basis, b_status],
        ["XC Functional (Libxc)", functional, f_status],
    ]
    for disp_name, disp_status in disp_map.items():
        rows.append([f"Dispersion: {disp_name}", functional, disp_status])

    print_table(rows, headers=["Component", "Query", "Status / Details"], title="Validation Summary")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DFT functional, basis set, and dispersion.")
    parser.add_argument("xyz", help="Path to XYZ or calculation output file")
    parser.add_argument("-f", "--functional", required=True, help="DFT functional name (e.g., b3lyp, pbe0, r2scan)")
    parser.add_argument("-b", "--basis", required=True, help="Basis set name (e.g., def2-TZVP, cc-pVTZ)")
    args = parser.parse_args()

    if not Path(args.xyz).exists():
        print_error(f"File not found: {args.xyz}", exit_code=1)

    validate_method(args.xyz, args.functional, args.basis)


if __name__ == "__main__":
    main()