"""Helper and utility functions for the FROG Reaction Profile Designer Web GUI.

Contains parsing, coordinate mapping, peak detection, template extraction,
CSV serialization, and centralized session state initialization.
"""

from __future__ import annotations

import re
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer


def safe_rerun() -> None:
    """Trigger a rerun compatible with both newer and older Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def compute_x_coords(n_steps: int, has_photo: bool, photo_step: int = 0) -> list[int]:
    """Compute X coordinates, sharing X between photo_step and photo_step + 1 if has_photo."""
    if not has_photo or n_steps <= 1:
        return list(range(n_steps))
    photo_step = max(0, min(photo_step, n_steps - 2))
    xs: list[int] = []
    cur = 0
    for i in range(n_steps):
        if i == photo_step + 1:
            xs.append(xs[-1])
        else:
            if i > 0 and i != photo_step + 1:
                cur += 1
            xs.append(cur)
    return xs


def parse_pasted_numbers(text: str) -> list[float | None]:
    """Parse string with numbers separated by commas, semicolons, tabs, newlines or spaces."""
    cleaned = re.split(r"[,;\s\t\n]+", text.strip())
    vals: list[float | None] = []
    for token in cleaned:
        if not token:
            continue
        tl = token.lower()
        if tl in ("none", "nan", "-", "null", "nil"):
            vals.append(None)
        else:
            try:
                vals.append(float(token))
            except ValueError:
                vals.append(None)
    return vals


TS_REGEX = re.compile(r"(?:\b|_)(?:TS\d*|MECP\d*|TRANS(?:ITION)?)\b|[‡*]", re.IGNORECASE)


def auto_detect_ts(df: pd.DataFrame, path_cols: list[str]) -> list[bool]:
    """Auto-detect TS / MECP states based on local energy maxima AND step names."""
    n_rows = len(df)
    is_ts = [False] * n_rows
    for col in path_cols:
        series = pd.to_numeric(df[col], errors="coerce").values
        for i in range(n_rows):
            cur = series[i]
            if np.isnan(cur):
                continue
            # Look left
            left_val = None
            for li in range(i - 1, -1, -1):
                if not np.isnan(series[li]):
                    left_val = series[li]
                    break
            # Look right
            right_val = None
            for ri in range(i + 1, n_rows):
                if not np.isnan(series[ri]):
                    right_val = series[ri]
                    break
            if left_val is not None and right_val is not None:
                if cur > left_val and cur >= right_val:
                    is_ts[i] = True

    # Also detect from name
    for i, s in enumerate(df["Step Name"]):
        s_str = str(s) if pd.notna(s) else ""
        if TS_REGEX.search(s_str):
            is_ts[i] = True

    return is_ts


auto_detect_peaks = auto_detect_ts


def build_dataframe_from_preset(preset: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Construct DataFrame and pathway metadata from preset dict."""
    steps = list(preset["steps"])
    label_energy = list(preset.get("label_energy", []))
    n_steps = len(steps)
    if len(label_energy) < n_steps:
        label_energy.extend([False] * (n_steps - len(label_energy)))

    is_ts = list(preset.get("is_ts", []))
    if len(is_ts) < n_steps:
        derived_ts = [
            bool(TS_REGEX.search(str(steps[i])))
            for i in range(len(is_ts), n_steps)
        ]
        is_ts.extend(derived_ts)

    data: dict[str, Any] = {
        "Step Name": steps,
        "Is TS?": is_ts,
        "Show Energy?": label_energy,
    }

    path_meta: list[dict[str, Any]] = []
    for p in preset.get("pathways", []):
        col_name = p["name"]
        energies = list(p.get("energies", []))
        if len(energies) < n_steps:
            energies.extend([None] * (n_steps - len(energies)))
        elif len(energies) > n_steps:
            energies = energies[:n_steps]
        float_energies = [
            float(e) if (e is not None and pd.notna(e)) else np.nan
            for e in energies
        ]
        data[col_name] = pd.Series(float_energies, dtype="float64")
        path_meta.append({
            "col_name": col_name,
            "display_name": col_name,
            "color": p.get("color", "#103E84"),
            "style": p.get("style", "solid"),
        })

    df = pd.DataFrame(data)
    return df, path_meta


def load_file_into_preset(content_str: str, filename: str = "input.txt") -> dict[str, Any]:
    """Convert an existing input (.txt or .yaml) content into preset dict format."""
    suffix = ".yaml" if filename.lower().endswith((".yaml", ".yml")) else ".txt"
    with tempfile.NamedTemporaryFile("w+", suffix=suffix, delete=False) as tmp:
        tmp.write(content_str)
        tmp_path = tmp.name

    analyzer = ReactionProfileAnalyzer(tmp_path, auto_plot=False)
    path_names = list(analyzer.energy_data.keys())
    max_len = max(len(v) for v in analyzer.energy_data.values()) if analyzer.energy_data else 1

    steps: list[str] = []
    for i in range(max_len):
        unique_labels_at_i = []
        for pname in path_names:
            p_lbls = analyzer.point_labels.get(pname, [])
            if i < len(p_lbls) and p_lbls[i]:
                val = analyzer.energy_data[pname][i] if i < len(analyzer.energy_data[pname]) else None
                if val is not None and not np.isnan(val):
                    lbl = str(p_lbls[i]).strip()
                    if lbl and lbl not in unique_labels_at_i:
                        unique_labels_at_i.append(lbl)
        if unique_labels_at_i:
            steps.append(" / ".join(unique_labels_at_i))
        else:
            steps.append("")

    # Label energy (labelpoint)
    label_energy = [False] * max_len
    if analyzer.config.labelpoint:
        for idx in analyzer.config.labelpoint:
            if isinstance(idx, int) and 0 <= idx < max_len:
                label_energy[idx] = True

    is_ts = [
        bool(any(kw in str(steps[i]).upper() for kw in ("TS", "MECP", "TRANS", "‡", "*")))
        for i in range(max_len)
    ]

    # Pathways
    colors = analyzer.config.colors or ["#103E84", "#1F56A3", "#58A4E0", "#E05858"]
    styles = analyzer.config.style or ["solid"]

    pathways = []
    for idx, name in enumerate(path_names):
        energies = [
            round(e, 2) if e is not None and not np.isnan(e) else None
            for e in analyzer.energy_data[name]
        ]
        pathways.append({
            "name": name,
            "color": colors[idx % len(colors)],
            "style": styles[idx % len(styles)],
            "energies": energies,
        })

    # X coordinates check for photo
    has_photo = False
    photo_step = 0
    if analyzer.config.x and len(analyzer.config.x) >= 2:
        for idx in range(len(analyzer.config.x) - 1):
            if analyzer.config.x[idx] == analyzer.config.x[idx + 1]:
                has_photo = True
                photo_step = idx
                break

    sections_list = []
    if analyzer.config.sections and analyzer.config.sections_start and analyzer.config.sections_end:
        for lbl, s_idx, e_idx in zip(analyzer.config.sections, analyzer.config.sections_start, analyzer.config.sections_end):
            sections_list.append({"label": str(lbl), "start": int(s_idx), "end": int(e_idx)})

    return {
        "title": analyzer.config.title or "Reaction Energy Profile",
        "units": getattr(analyzer.config, "units", "kcal/mol") or "kcal/mol",
        "point_type": getattr(analyzer.config, "point_type", None) or "hollow",
        "has_photo": has_photo,
        "photo_step": photo_step,
        "show_xticks": getattr(analyzer.config, "show_xticks", True),
        "show_legend": getattr(analyzer.config, "legend", True) if getattr(analyzer.config, "legend", None) is not None else True,
        "fig_w": analyzer.config.figsize[0],
        "fig_h": analyzer.config.figsize[1],
        "curviness": getattr(analyzer.config, "curviness", None) or 0.42,
        "font_size": getattr(analyzer.config, "font_size", 11),
        "axes_style": getattr(analyzer.config, "axes", "none") or "none",
        "sections_list": sections_list,
        "sections_shift": float(getattr(analyzer.config, "sections_shift", 0.05)),
        "transparent_bg": bool(getattr(analyzer.config, "transparent", False)),
        "bg_color": getattr(analyzer.config, "bg_color", None) or "#FFFFFF",
        "steps": steps,
        "is_ts": is_ts,
        "label_energy": label_energy,
        "pathways": pathways,
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export current DataFrame to CSV format using pandas native method."""
    return df.to_csv(index=False).encode("utf-8")


def extract_current_template(
    df: pd.DataFrame,
    path_meta: list[dict[str, Any]],
    state: Any = None,
) -> dict[str, Any]:
    """Extract complete template preset structure from current session state and DataFrame."""
    s = state if state is not None else st.session_state
    steps = [str(x) if pd.notna(x) else "" for x in df["Step Name"]]
    is_ts = [bool(x) for x in df.get("Is TS?", [False] * len(df))]
    label_energy = [bool(x) for x in df.get("Show Energy?", [False] * len(df))]

    path_cols = [c for c in df.columns if c not in ("Step Name", "Is TS?", "Show Energy?")]
    meta_map = {m["col_name"]: m for m in path_meta}

    pathways = []
    for col in path_cols:
        meta = meta_map.get(col, {})
        energies = [
            round(float(v), 2) if (pd.notna(v) and v is not None) else None
            for v in df[col]
        ]
        pathways.append({
            "name": meta.get("display_name", col),
            "color": meta.get("color", "#103E84"),
            "style": meta.get("style", "solid"),
            "energies": energies,
        })

    template: dict[str, Any] = {
        "title": str(getattr(s, "title", "Reaction Energy Profile")),
        "units": str(getattr(s, "units", "kcal/mol")),
        "point_type": str(getattr(s, "point_type", "hollow")),
        "has_photo": bool(getattr(s, "has_photo", False)),
        "photo_step": int(getattr(s, "photo_step", 0)),
        "show_xticks": bool(getattr(s, "show_xticks", True)),
        "show_legend": bool(getattr(s, "show_legend", True)),
        "fig_w": float(getattr(s, "fig_w", 10.0)),
        "fig_h": float(getattr(s, "fig_h", 6.0)),
        "curviness": float(getattr(s, "curviness", 0.42)),
        "font_size": int(getattr(s, "font_size", 11)),
        "axes_style": str(getattr(s, "axes_style", "none")),
        "sections_list": [dict(sec) for sec in getattr(s, "sections_list", [])],
        "sections_shift": float(getattr(s, "sections_shift", 0.05)),
        "layout_preset": str(getattr(s, "layout_preset", "Custom (Manual sliders)")),
        "transparent_bg": bool(getattr(s, "transparent_bg", False)),
        "bg_color": str(getattr(s, "bg_color", "#FFFFFF")),
        "steps": steps,
        "is_ts": is_ts,
        "label_energy": label_energy,
        "pathways": pathways,
    }
    return template


def init_session_state(
    preset: dict[str, Any] | None = None,
    preset_name: str | None = None,
    force_reset: bool = False,
    state: Any = None,
) -> None:
    """Consolidated initialization and total reset of all session state variables."""
    s = state if state is not None else st.session_state

    # If force_reset or not yet initialized
    needs_init = force_reset or ("df" not in s) or ("path_meta" not in s)

    if needs_init and preset is not None:
        df, meta = build_dataframe_from_preset(preset)
        s.df = df
        s.path_meta = meta
        s.current_preset_name = preset_name or preset.get("title", "Custom")
        s.title = preset.get("title", "Reaction Energy Profile")
        s.units = preset.get("units", "kcal/mol")
        s.point_type = preset.get("point_type", "hollow")
        s.has_photo = bool(preset.get("has_photo", False))
        s.photo_step = int(preset.get("photo_step", 0))
        s.show_xticks = bool(preset.get("show_xticks", True))
        s.show_legend = bool(preset.get("show_legend", True))
        s.fig_w = float(preset.get("fig_w", 10.0))
        s.fig_h = float(preset.get("fig_h", 6.0))
        s.curviness = float(preset.get("curviness", 0.42))
        s.font_size = int(preset.get("font_size", 11))
        s.axes_style = str(preset.get("axes_style", "none"))
        s.layout_preset = str(preset.get("layout_preset", "Custom (Manual sliders)"))
        s.transparent_bg = bool(preset.get("transparent_bg", preset.get("transparent", False)))
        s.bg_color = str(preset.get("bg_color", "#FFFFFF"))
        s.ddg_items = []
        s.sections_list = [dict(sec) for sec in preset.get("sections_list", [])]
        s.sections_shift = float(preset.get("sections_shift", 0.05))
        s.plot_result = None
        s.render_plot = False

    # Ensure all individual keys are present in session state with safe fallbacks
    def _set_default(key: str, val: Any) -> None:
        if hasattr(s, "setdefault"):
            s.setdefault(key, val)
        elif isinstance(s, dict):
            s.setdefault(key, val)
        else:
            if not hasattr(s, key):
                setattr(s, key, val)

    _set_default("title", "Reaction Energy Profile")
    _set_default("units", "kcal/mol")
    _set_default("point_type", "hollow")
    _set_default("has_photo", False)
    _set_default("photo_step", 0)
    _set_default("show_xticks", True)
    _set_default("show_legend", True)
    _set_default("fig_w", 10.0)
    _set_default("fig_h", 6.0)
    _set_default("curviness", 0.42)
    _set_default("font_size", 11)
    _set_default("axes_style", "none")
    _set_default("layout_preset", "Custom (Manual sliders)")
    _set_default("transparent_bg", False)
    _set_default("bg_color", "#FFFFFF")
    _set_default("ddg_items", [])
    _set_default("sections_list", [])
    _set_default("sections_shift", 0.05)
    _set_default("plot_result", None)
    _set_default("render_plot", False)
