"""Configuration manager for FROG Reaction Profile Designer.

Handles persistent loading, concurrent-safe writing, and querying of
GUI configurations (Presets, Palettes, Themes) in gui_config.yaml.
"""

from __future__ import annotations

import contextlib
import copy
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

# Path to the primary configuration file
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "gui_config.yaml"

# Default fallbacks if YAML is corrupted or missing
DEFAULT_PALETTES: dict[str, list[str]] = {
    "Pacific Sunset (Custom)": ["#D53302", "#FCC560", "#A0CBAD", "#8FB1D0", "#000145"],
    "JACS / ACS Blue & Crimson": ["#0B2545", "#134074", "#8DA9C4", "#C0392B", "#D35400", "#7D6608"],
    "Colorblind-Safe (Wong/Tol)": ["#0072B2", "#D55E00", "#009E73", "#F0E442", "#CC79A7", "#56B4E9"],
    "Nordic Muted Harmony": ["#2E3440", "#5E81AC", "#88C0D0", "#A3BE8C", "#EBCB8B", "#BF616A"],
    "Nature / Science Classic": ["#103E84", "#1F56A3", "#58A4E0", "#E05858", "#3DAA65", "#9B59B6"],
}

DEFAULT_THEMES: dict[str, dict[str, Any]] = {
    "Custom (Keep current)": {},
    "JACS / ACS Classic": {
        "point_type": "bar",
        "curviness": 0.35,
        "fig_w": 8.5,
        "fig_h": 5.5,
        "palette": "JACS / ACS Blue & Crimson",
    },
    "Nature / Springer": {
        "point_type": "hollow",
        "curviness": 0.42,
        "fig_w": 10.5,
        "fig_h": 6.5,
        "palette": "Pacific Sunset (Custom)",
    },
    "ChemDraw Clean": {
        "point_type": "bar",
        "curviness": 0.0,
        "fig_w": 9.0,
        "fig_h": 5.5,
        "palette": "Nordic Muted Harmony",
    },
    "Slide / Thesis (High-Contrast)": {
        "point_type": "hollow",
        "curviness": 0.45,
        "fig_w": 12.0,
        "fig_h": 7.0,
        "palette": "Colorblind-Safe (Wong/Tol)",
    },
}

DEFAULT_PRESETS: dict[str, dict[str, Any]] = {
    "General Photochemical (S0 -> S1 -> Branches)": {
        "title": "General Photochemical Mechanism",
        "units": "kcal/mol",
        "point_type": "hollow",
        "has_photo": True,
        "photo_step": 0,
        "show_xticks": False,
        "show_legend": True,
        "fig_w": 12.0,
        "fig_h": 7.0,
        "curviness": 0.42,
        "font_size": 11,
        "steps": [
            r"Reactant $\mathbf{S_0}$",
            r"Reactant $\mathbf{S_1}$",
            r"$\mathbf{Int_1}$",
            r"$\mathbf{Int_2}$",
            r"$\mathbf{TS_1}$",
            r"$\mathbf{Int_3}$",
            r"$\mathbf{MECP}$",
            r"$\mathbf{Int_4}$",
            r"$\mathbf{TS_2}$",
            r"Product",
        ],
        "label_energy": [False, False, False, False, True, False, True, False, True, False],
        "is_ts": [False, False, False, False, True, False, True, False, True, False],
        "pathways": [
            {
                "name": "Main Trunk",
                "color": "#103E84",
                "style": "solid",
                "energies": [0.0, 70.0, 68.0, 69.5, 73.0, 40.0, None, None, None, None],
            },
            {
                "name": "Branch A",
                "color": "#1F56A3",
                "style": "solid",
                "energies": [None, None, None, None, None, 40.0, 45.0, 15.0, 18.0, -20.0],
            },
            {
                "name": "Branch B",
                "color": "#58A4E0",
                "style": "dashed",
                "energies": [None, None, None, None, None, 40.0, 46.5, 38.0, 15.5, None],
            },
        ],
    },
    "Catalytic Cycle (R -> TS1 -> Int -> TS2 -> P)": {
        "title": "Catalytic Cycle Energy Profile",
        "units": "kcal/mol",
        "point_type": "bar",
        "has_photo": False,
        "photo_step": 0,
        "show_xticks": True,
        "show_legend": True,
        "fig_w": 10.0,
        "fig_h": 6.0,
        "curviness": 0.42,
        "font_size": 11,
        "steps": ["Reactants", "TS1", "Intermediate", "TS2", "Products"],
        "label_energy": [False, True, False, True, False],
        "is_ts": [False, True, False, True, False],
        "pathways": [
            {
                "name": "Catalytic Pathway",
                "color": "#103E84",
                "style": "solid",
                "energies": [0.0, 22.4, 8.2, 19.5, -14.2],
            }
        ],
    },
    "Competitive Pathways (Major vs Minor TS)": {
        "title": "Bifurcating Reaction Profile",
        "units": "kcal/mol",
        "point_type": "hollow",
        "has_photo": False,
        "photo_step": 0,
        "show_xticks": True,
        "show_legend": True,
        "fig_w": 10.5,
        "fig_h": 6.5,
        "curviness": 0.42,
        "font_size": 11,
        "steps": ["Reactants", "Intermediate", "TS-Major / TS-Minor", "Products"],
        "label_energy": [False, False, True, False],
        "is_ts": [False, False, True, False],
        "pathways": [
            {
                "name": "Major Pathway",
                "color": "#103E84",
                "style": "solid",
                "energies": [0.0, 5.2, 16.4, -12.0],
            },
            {
                "name": "Minor Pathway",
                "color": "#E05858",
                "style": "dashed",
                "energies": [0.0, 5.2, 21.8, -4.5],
            },
        ],
    },
    "Simple 1-Step (R -> TS -> P)": {
        "title": "Single Step Activation Barrier",
        "units": "kcal/mol",
        "point_type": "bar",
        "has_photo": False,
        "photo_step": 0,
        "show_xticks": True,
        "show_legend": True,
        "fig_w": 8.0,
        "fig_h": 5.5,
        "curviness": 0.42,
        "font_size": 11,
        "steps": ["Reactants", "TS", "Products"],
        "label_energy": [False, True, False],
        "is_ts": [False, True, False],
        "pathways": [
            {
                "name": "Reaction Path",
                "color": "#103E84",
                "style": "solid",
                "energies": [0.0, 18.5, -12.3],
            }
        ],
    },
    "Blank / Custom": {
        "title": "Reaction Energy Profile",
        "units": "kcal/mol",
        "point_type": "hollow",
        "has_photo": False,
        "photo_step": 0,
        "show_xticks": True,
        "show_legend": True,
        "fig_w": 10.0,
        "fig_h": 6.0,
        "curviness": 0.42,
        "font_size": 11,
        "steps": ["Step 0", "Step 1", "Step 2"],
        "label_energy": [False, False, False],
        "is_ts": [False, False, False],
        "pathways": [
            {
                "name": "Path 1",
                "color": "#103E84",
                "style": "solid",
                "energies": [0.0, 10.0, 0.0],
            }
        ],
    },
}


@contextlib.contextmanager
def file_lock(lock_path: Path):
    """Context manager acquiring an exclusive flock on Unix/macOS."""
    lock_file = lock_path.with_suffix(".lock")
    try:
        import fcntl

        f = open(lock_file, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(f, fcntl.LOCK_UN)
            except OSError:
                pass
            f.close()
    except (ImportError, OSError):
        # Fallback if fcntl unavailable
        yield


def _resolve_config_path(path: Path | str | None) -> Path:
    return Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration dictionary from YAML file, with fallback defaults."""
    path = _resolve_config_path(config_path)
    if not path.exists():
        # Initialize default file
        cfg = {
            "PALETTES": copy.deepcopy(DEFAULT_PALETTES),
            "THEMES": copy.deepcopy(DEFAULT_THEMES),
            "PRESETS": copy.deepcopy(DEFAULT_PRESETS),
        }
        try:
            save_config(cfg, path)
        except OSError:
            pass
        return cfg

    with file_lock(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

    result: dict[str, Any] = {
        "PALETTES": data.get("PALETTES", copy.deepcopy(DEFAULT_PALETTES)),
        "THEMES": data.get("THEMES", copy.deepcopy(DEFAULT_THEMES)),
        "PRESETS": data.get("PRESETS", copy.deepcopy(DEFAULT_PRESETS)),
    }
    return result


def save_config(config: dict[str, Any], config_path: Path | str | None = None) -> None:
    """Save full configuration to YAML safely using atomic replace and file lock."""
    path = _resolve_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with file_lock(path):
        # Write to temporary file in same folder, then atomic rename
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", suffix=".tmp") as tmp:
            yaml.dump(config, tmp, sort_keys=False, allow_unicode=True)
            tmp_name = tmp.name
        os.replace(tmp_name, path)


def get_presets(config_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Retrieve all available presets."""
    cfg = load_config(config_path)
    presets = cfg.get("PRESETS", {})
    return presets if isinstance(presets, dict) else copy.deepcopy(DEFAULT_PRESETS)


def save_preset(name: str, preset_data: dict[str, Any], config_path: Path | str | None = None) -> None:
    """Add or overwrite a template preset under PRESETS and persist to YAML."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Preset name cannot be empty.")
    cfg = load_config(config_path)
    if "PRESETS" not in cfg or not isinstance(cfg["PRESETS"], dict):
        cfg["PRESETS"] = copy.deepcopy(DEFAULT_PRESETS)
    cfg["PRESETS"][clean_name] = copy.deepcopy(preset_data)
    save_config(cfg, config_path)


def delete_preset(name: str, config_path: Path | str | None = None) -> bool:
    """Remove a preset by name. Returns True if found and deleted."""
    cfg = load_config(config_path)
    presets = cfg.get("PRESETS", {})
    if isinstance(presets, dict) and name in presets:
        del presets[name]
        cfg["PRESETS"] = presets
        save_config(cfg, config_path)
        return True
    return False


def get_palettes(config_path: Path | str | None = None) -> dict[str, list[str]]:
    """Retrieve all available palettes."""
    cfg = load_config(config_path)
    palettes = cfg.get("PALETTES", {})
    return palettes if isinstance(palettes, dict) else copy.deepcopy(DEFAULT_PALETTES)


def save_palette(name: str, colors: list[str], config_path: Path | str | None = None) -> None:
    """Add or overwrite a palette under PALETTES and persist to YAML."""
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Palette name cannot be empty.")
    if not colors:
        raise ValueError("Palette must contain at least one color.")
    cfg = load_config(config_path)
    if "PALETTES" not in cfg or not isinstance(cfg["PALETTES"], dict):
        cfg["PALETTES"] = copy.deepcopy(DEFAULT_PALETTES)
    cfg["PALETTES"][clean_name] = [str(c).strip() for c in colors]
    save_config(cfg, config_path)


def delete_palette(name: str, config_path: Path | str | None = None) -> bool:
    """Remove a palette by name. Returns True if found and deleted."""
    cfg = load_config(config_path)
    palettes = cfg.get("PALETTES", {})
    if isinstance(palettes, dict) and name in palettes:
        del palettes[name]
        cfg["PALETTES"] = palettes
        save_config(cfg, config_path)
        return True
    return False


def get_themes(config_path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Retrieve all available journal themes."""
    cfg = load_config(config_path)
    themes = cfg.get("THEMES", {})
    return themes if isinstance(themes, dict) else copy.deepcopy(DEFAULT_THEMES)
