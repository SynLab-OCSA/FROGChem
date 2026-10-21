"""Interactive Streamlit Web GUI for FROG reaction profile designer.

Provides an Excel-like interactive spreadsheet, 1-click presets,
smart TS/MECP peak auto-detection, copy-paste import, and manual plot generation
with high-res export (PNG, SVG, PDF) and input.txt generation.
"""

from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from frogchem.reaction_plotter.config_manager import (
    DEFAULT_PRESETS,
    get_palettes,
    get_presets,
    get_themes,
    save_palette,
    save_preset,
)
from frogchem.reaction_plotter.gui_utils import (
    auto_detect_peaks,
    auto_detect_ts,
    build_dataframe_from_preset,
    compute_x_coords,
    dataframe_to_csv_bytes,
    extract_current_template,
    init_session_state,
    load_file_into_preset,
    parse_pasted_numbers,
    safe_rerun,
)
from frogchem.reaction_plotter.plotter import ReactionProfilePlotter
from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer


# ── Main Application ──────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="FROG — Reaction Profile Designer",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS for clean UI ───────────────────────────────────────────────
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .stButton>button { border-radius: 8px; font-weight: 500; transition: all 0.15s ease-in-out; }
        .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 3px 8px rgba(0,0,0,0.08); }
        [data-testid="stSidebar"] { padding-top: 1rem; }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] { gap: 2px; }
        [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] { padding: 6px 8px; font-size: 13px; font-weight: 500; }
        .main .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; padding: 8px 16px; }
        div[data-testid="stMetricValue"] { font-family: monospace; font-size: 1.4rem; }
        button[kind="primary"] { background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%) !important; border: none !important; font-weight: 600 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧪 FROG — Reaction Profile Designer")
    st.caption("Interactively build, branch, and preview publication-quality reaction energy profiles.")

    # Load persistent configurations
    PRESETS = get_presets()
    PALETTES = get_palettes()
    THEMES = get_themes()

    # ── Consolidated Session State Initialization ──────────────────────────────
    if "df" not in st.session_state or "path_meta" not in st.session_state:
        default_preset_name = "Blank / Custom"
        if default_preset_name not in PRESETS and PRESETS:
            default_preset_name = list(PRESETS.keys())[0]
        init_preset = PRESETS.get(default_preset_name, list(PRESETS.values())[0] if PRESETS else DEFAULT_PRESETS["Blank / Custom"])
        init_session_state(preset=init_preset, preset_name=default_preset_name, force_reset=True)
    else:
        init_session_state(force_reset=False)

    # Migration for active browser sessions to ensure separate Is TS? and Show Energy?
    if "df" in st.session_state and isinstance(st.session_state.df, pd.DataFrame):
        df_cols = list(st.session_state.df.columns)
        if "Label Energy? (TS/MECP)" in df_cols:
            old_vals = st.session_state.df["Label Energy? (TS/MECP)"]
            if "Show Energy?" not in df_cols:
                st.session_state.df["Show Energy?"] = old_vals
            if "Is TS?" not in df_cols:
                st.session_state.df["Is TS?"] = [
                    bool(any(kw in str(s).upper() for kw in ("TS", "MECP", "TRANS", "‡", "*")))
                    for s in st.session_state.df["Step Name"]
                ]
            st.session_state.df.drop(columns=["Label Energy? (TS/MECP)"], inplace=True)
        if "Is TS?" not in st.session_state.df.columns:
            st.session_state.df["Is TS?"] = [False] * len(st.session_state.df)
        if "Show Energy?" not in st.session_state.df.columns:
            st.session_state.df["Show Energy?"] = [False] * len(st.session_state.df)
        # Ensure all existing pathway columns have float64 dtype
        mig_path_cols = [c for c in st.session_state.df.columns if c not in ("Step Name", "Is TS?", "Show Energy?")]
        for c in mig_path_cols:
            st.session_state.df[c] = pd.to_numeric(st.session_state.df[c], errors="coerce").astype("float64")

    # ── Top Toolbar: Quick Presets, Save Template, File Upload, Clear All ──────
    c_preset, c_save_tmpl, c_upload, c_clear = st.columns([3, 2.2, 2.2, 1])

    with c_preset:
        preset_options = list(PRESETS.keys())
        cur_preset = st.session_state.get("current_preset_name", preset_options[0] if preset_options else "Custom")
        idx_preset = preset_options.index(cur_preset) if cur_preset in preset_options else 0
        chosen_preset = st.selectbox(
            "🚀 Quick Template / Preset:",
            options=preset_options,
            index=idx_preset,
            help="Select a mechanism template to instantly populate the table and options.",
        )
        if chosen_preset != st.session_state.current_preset_name and chosen_preset in PRESETS:
            st.session_state.current_preset_name = chosen_preset
            p = PRESETS[chosen_preset]
            init_session_state(preset=p, preset_name=chosen_preset, force_reset=True)
            safe_rerun()

    with c_save_tmpl:
        st.markdown("<div style='margin-top: 4px;'></div>", unsafe_allow_html=True)
        new_tmpl_name = st.text_input(
            "Save Template as:",
            placeholder="Template name...",
            key="input_save_template_name",
            label_visibility="collapsed",
            help="Save the current mechanism table and options to gui_config.yaml.",
        )
        if st.button("💾 Save Template", key="btn_save_template", use_container_width=True):
            clean_tmpl = new_tmpl_name.strip()
            if not clean_tmpl:
                st.warning("Please enter a template name.")
            else:
                tmpl_data = extract_current_template(st.session_state.df, st.session_state.path_meta, st.session_state)
                save_preset(clean_tmpl, tmpl_data)
                st.session_state.current_preset_name = clean_tmpl
                st.success(f"Template '{clean_tmpl}' saved!")
                safe_rerun()

    with c_upload:
        uploaded_file = st.file_uploader(
            "📂 Load existing file (TXT or YAML):",
            type=["txt", "yaml", "yml"],
            help="Upload an existing FROG profile file (.txt, .yaml) to view, edit, or customize it.",
        )
        if uploaded_file is not None:
            if "last_uploaded_name" not in st.session_state or st.session_state.last_uploaded_name != uploaded_file.name:
                try:
                    content_str = uploaded_file.getvalue().decode("utf-8")
                    parsed_preset = load_file_into_preset(content_str, filename=uploaded_file.name)
                    st.session_state.last_uploaded_name = uploaded_file.name
                    init_session_state(preset=parsed_preset, preset_name="Blank / Custom", force_reset=True)
                    st.success(f"Loaded '{uploaded_file.name}'!")
                    safe_rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    with c_clear:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear All", help="Reset everything to a blank template."):
            blank_preset = PRESETS.get("Blank / Custom", DEFAULT_PRESETS["Blank / Custom"])
            init_session_state(preset=blank_preset, preset_name="Blank / Custom", force_reset=True)
            safe_rerun()

    # ── Sidebar Styling Controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Appearance & Setup")

        tab_sb_style, tab_sb_layout, tab_sb_axes = st.tabs([
            "🎨 Style",
            "📐 Layout",
            "⚙️ Axes",
        ])

        with tab_sb_style:
            st.markdown("##### 🏛️ Publication Preset")
            chosen_theme = st.selectbox(
                "Theme Preset:",
                options=list(THEMES.keys()),
                index=0,
                help="Apply instant formatting tailored for journals like JACS, Nature, or ChemDraw.",
                key="chosen_theme_preset",
            )
            if chosen_theme != "Custom (Keep current)":
                th_conf = THEMES[chosen_theme]
                if "point_type" in th_conf:
                    st.session_state.point_type = th_conf["point_type"]
                if "curviness" in th_conf:
                    st.session_state.curviness = th_conf["curviness"]
                if "fig_w" in th_conf:
                    st.session_state.fig_w = th_conf["fig_w"]
                if "fig_h" in th_conf:
                    st.session_state.fig_h = th_conf["fig_h"]
                if "palette" in th_conf:
                    st.session_state.chosen_palette = th_conf["palette"]
                    pal_colors = PALETTES[th_conf["palette"]]
                    for i, pm in enumerate(st.session_state.path_meta):
                        pm["color"] = pal_colors[i % len(pal_colors)]

            st.markdown("##### 🎨 Color Palette")
            chosen_pal = st.selectbox(
                "Palette Selection:",
                options=list(PALETTES.keys()),
                index=list(PALETTES.keys()).index(st.session_state.get("chosen_palette", "Pacific Sunset (Custom)"))
                if st.session_state.get("chosen_palette") in PALETTES
                else 0,
                help="Select a color scheme to automatically apply across all reaction pathways.",
                key="chosen_color_palette_select",
            )
            
            # Swatches preview with explicit HEX labels
            pal_colors = PALETTES[chosen_pal]
            swatches_html = "<div style='display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;'>" + "".join([
                f"<div style='display:inline-flex; align-items:center; background:#f6f8fa; border:1px solid #d0d7de; border-radius:5px; padding:2px 6px; font-size:11px; font-family:monospace;'>"
                f"<span style='display:inline-block; width:12px; height:12px; background-color:{c}; border-radius:3px; border:1px solid #777; margin-right:5px;'></span>"
                f"<span style='color:#1f2328; font-weight:600;'>{c}</span>"
                f"</div>"
                for c in pal_colors
            ]) + "</div>"
            st.markdown(swatches_html, unsafe_allow_html=True)

            if chosen_pal != st.session_state.get("chosen_palette"):
                st.session_state.chosen_palette = chosen_pal
                for i, pm in enumerate(st.session_state.path_meta):
                    pm["color"] = pal_colors[i % len(pal_colors)]
                safe_rerun()

            if st.button("🔄 Apply to All Pathways", key="btn_reapply_palette", help="Instantly apply this palette across all reaction pathways.", use_container_width=True):
                for i, pm in enumerate(st.session_state.path_meta):
                    pm["color"] = pal_colors[i % len(pal_colors)]
                st.success(f"Applied '{chosen_pal}' to all pathways!")
                safe_rerun()

            # Save Custom Palette expander
            with st.expander("💾 Save as New Palette", expanded=False):
                new_pal_name = st.text_input("Palette Name:", placeholder="e.g. My Paper Palette", key="input_save_palette_name")
                if st.button("Save Palette", key="btn_save_palette", use_container_width=True):
                    clean_pal = new_pal_name.strip()
                    if not clean_pal:
                        st.warning("Please enter a palette name.")
                    else:
                        current_colors = [pm.get("color") for pm in st.session_state.path_meta if pm.get("color")]
                        if not current_colors:
                            current_colors = pal_colors
                        save_palette(clean_pal, current_colors)
                        st.session_state.chosen_palette = clean_pal
                        st.success(f"Palette '{clean_pal}' saved!")
                        safe_rerun()

            st.markdown("---")
            st.session_state.point_type = st.selectbox(
                "Marker Style",
                options=["hollow", "bar", "dot", "none"],
                index=["hollow", "bar", "dot", "none"].index(st.session_state.point_type)
                if st.session_state.point_type in ["hollow", "bar", "dot", "none"]
                else 0,
                help="'hollow': white-centered circles; 'bar': horizontal levels (ChemDraw style); 'dot': filled points.",
            )

            st.session_state.curviness = st.slider(
                "Curviness (Bezier Tension)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.curviness),
                step=0.02,
                help="0.0 = straight lines, 0.42 = standard smooth curves.",
            )

        with tab_sb_layout:
            st.session_state.title = st.text_input("Plot Title", value=st.session_state.title)
            st.session_state.units = st.selectbox(
                "Energy Units",
                options=["kcal/mol", "kJ/mol", "eV", "Hartree"],
                index=["kcal/mol", "kJ/mol", "eV", "Hartree"].index(st.session_state.units)
                if st.session_state.units in ["kcal/mol", "kJ/mol", "eV", "Hartree"]
                else 0,
            )

            st.markdown("##### 📄 A4 Layout & Dimensions")
            layout_presets = [
                "Custom (Manual sliders)",
                "Single Column (A4 — 8.5 cm / 3.35 in)",
                "1.5 Column (A4 — 12.0 cm / 4.75 in)",
                "Double Column (A4 — 17.5 cm / 6.90 in)",
            ]
            cur_layout = st.session_state.get("layout_preset", "Custom (Manual sliders)")
            if cur_layout == "Single Column (A4 — 8.5 cm / 3.3 in)":
                cur_layout = "Single Column (A4 — 8.5 cm / 3.35 in)"
            elif cur_layout == "Double Column (A4 — 17.5 cm / 6.9 in)":
                cur_layout = "Double Column (A4 — 17.5 cm / 6.90 in)"
            idx_layout = layout_presets.index(cur_layout) if cur_layout in layout_presets else 0
            chosen_layout = st.selectbox(
                "Column / Size Preset (A4):",
                options=layout_presets,
                index=idx_layout,
                help="Automatically set standard print dimensions for 1-column, 1.5-column, or 2-column scientific articles on A4.",
                key="sel_layout_preset",
            )
            if chosen_layout != cur_layout:
                st.session_state.layout_preset = chosen_layout
                if chosen_layout == "Single Column (A4 — 8.5 cm / 3.35 in)":
                    st.session_state.fig_w = 3.35
                    st.session_state.fig_h = 2.05
                    st.session_state.font_size = 10
                elif chosen_layout == "1.5 Column (A4 — 12.0 cm / 4.75 in)":
                    st.session_state.fig_w = 4.75
                    st.session_state.fig_h = 2.75
                    st.session_state.font_size = 12
                elif chosen_layout == "Double Column (A4 — 17.5 cm / 6.90 in)":
                    st.session_state.fig_w = 6.90
                    st.session_state.fig_h = 3.70
                    st.session_state.font_size = 14
                safe_rerun()

            col_w, col_h = st.columns(2)
            with col_w:
                w_val = float(st.session_state.fig_w)
                new_w = st.slider("Width (in)", min_value=2.0, max_value=22.0, value=w_val, step=0.05)
                if abs(new_w - w_val) > 1e-4:
                    st.session_state.fig_w = new_w
                    st.session_state.layout_preset = "Custom (Manual sliders)"
            with col_h:
                h_val = float(st.session_state.fig_h)
                new_h = st.slider("Height (in)", min_value=1.5, max_value=14.0, value=h_val, step=0.05)
                if abs(new_h - h_val) > 1e-4:
                    st.session_state.fig_h = new_h
                    st.session_state.layout_preset = "Custom (Manual sliders)"

            st.session_state.font_size = st.slider(
                "Font Size",
                min_value=6,
                max_value=24,
                value=int(st.session_state.get("font_size", 11)),
                step=1,
                help="Base font size for labels, ticks, and text on the plot.",
            )

            st.session_state.show_legend = st.checkbox(
                "Show Legend",
                value=bool(st.session_state.get("show_legend", True)),
                help="Check to show or hide the pathway legend on the plot.",
            )

            st.markdown("---")
            st.markdown("##### 🖼️ Background & Transparency")
            st.session_state.transparent_bg = st.checkbox(
                "Transparent Background (No background)",
                value=bool(st.session_state.get("transparent_bg", False)),
                help="Export and preview without background (alpha=0), ideal for Word, PowerPoint, and papers.",
                key="chk_transparent_bg",
            )
            if not st.session_state.transparent_bg:
                bg_col1, bg_col2 = st.columns([2, 1])
                with bg_col1:
                    st.session_state.bg_color = st.color_picker(
                        "Background Color:",
                        value=st.session_state.get("bg_color", "#FFFFFF"),
                        key="pick_bg_color",
                    )
                with bg_col2:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    if st.button("White", key="btn_bg_white", use_container_width=True):
                        st.session_state.bg_color = "#FFFFFF"
                        safe_rerun()
                q_bgs = [
                    ("White", "#FFFFFF"),
                    ("Paper", "#FAFAFA"),
                    ("Soft Gray", "#F4F4F5"),
                    ("Dark Slide", "#1E1E1E"),
                ]
                sw_cols = st.columns(len(q_bgs))
                for sw_idx, (b_lbl, b_hex) in enumerate(q_bgs):
                    with sw_cols[sw_idx]:
                        if st.button(b_lbl, key=f"quick_bg_{sw_idx}", use_container_width=True):
                            st.session_state.bg_color = b_hex
                            safe_rerun()

        with tab_sb_axes:
            _axes_options = ["none", "x", "y", "both", "box"]
            _axes_labels  = ["None (clean)", "Bottom axis only", "Left axis only", "Both axes", "Full box"]
            _cur_axes = st.session_state.get("axes_style", "none")
            if _cur_axes not in _axes_options:
                _cur_axes = "none"
            st.session_state.axes_style = st.selectbox(
                "Plot Frame / Axes",
                options=_axes_options,
                index=_axes_options.index(_cur_axes),
                format_func=lambda v: _axes_labels[_axes_options.index(v)],
                help="Controls which axis lines are drawn around the plot. 'None' = clean frameless look.",
            )

            st.session_state.show_xticks = st.checkbox(
                "Show X-Axis Tick Marks",
                value=bool(st.session_state.show_xticks),
                help="If unchecked, hides the numerical/reaction coordinate ticks for a clean publication look.",
            )

            st.markdown("---")
            st.session_state.has_photo = st.checkbox(
                "Vertical Excitation (Absorption Arrow)",
                value=bool(st.session_state.has_photo),
                help="Shares an X coordinate between two steps and draws an absorption arrow from the lower to the higher state.",
            )
            if st.session_state.has_photo:
                n_avail = max(1, len(st.session_state.df) - 1)
                step_opts = list(range(n_avail))

                def _fmt_photo_step(idx: int) -> str:
                    if idx < len(st.session_state.df):
                        s_a = st.session_state.df["Step Name"].iloc[idx]
                        s_b = st.session_state.df["Step Name"].iloc[idx + 1] if idx + 1 < len(st.session_state.df) else ""
                        return f"Step {idx} → {idx + 1} ({s_a} → {s_b})"
                    return f"Step {idx} → {idx + 1}"

                cur_p_step = int(st.session_state.get("photo_step", 0))
                st.session_state.photo_step = st.selectbox(
                    "Excitation Step (shares X):",
                    options=step_opts,
                    index=min(cur_p_step, len(step_opts) - 1),
                    format_func=_fmt_photo_step,
                    help="Choose which step undergoes vertical excitation (e.g. Step 0 for S0->S1, or any intermediate absorbing a photon).",
                )

    # ── Table Data Editor Section ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Interactive Data Spreadsheet")
    st.caption("Edit values below. Leave blank for unpopulated steps (they automatically become `None`). You can paste directly from Excel or Google Sheets!  \n💡 **Distinct pathway names at the same coordinate:** If two or more pathways have different intermediates or TSs at the same step, write the names separated by `/` (e.g., `MECP-SET / TS-HAT`). They will be automatically assigned to the respective active pathways in order!")

    # Toolbar buttons above table
    btn_c1, btn_c2, btn_c3, btn_csv, btn_c4 = st.columns([1.6, 1.4, 1.4, 1.3, 2.0])

    non_path_cols = ("Step Name", "Is TS?", "Show Energy?", "Label Energy? (TS/MECP)")
    path_cols = [c for c in st.session_state.df.columns if c not in non_path_cols]

    with btn_c1:
        if st.button("⚡ Auto-detect TS", help="Scan curves for local energy maxima and keywords (TS, MECP, ‡, *) and mark them as TS."):
            peaks = auto_detect_ts(st.session_state.df, path_cols)
            st.session_state.df["Is TS?"] = peaks
            # Also enable Show Energy? for detected transition states
            cur_shows = list(st.session_state.df.get("Show Energy?", [False] * len(peaks)))
            st.session_state.df["Show Energy?"] = [
                bool(cur or is_p) for cur, is_p in zip(cur_shows, peaks)
            ]
            safe_rerun()

    with btn_c2:
        if st.button("🏷️ Show All", help="Show numerical energy labels for all steps on the plot."):
            st.session_state.df["Show Energy?"] = True
            safe_rerun()

    with btn_c3:
        if st.button("❌ Hide All", help="Hide numerical energy labels for all steps on the plot."):
            st.session_state.df["Show Energy?"] = False
            safe_rerun()

    with btn_csv:
        csv_bytes = dataframe_to_csv_bytes(st.session_state.df)
        st.download_button(
            "📥 CSV",
            data=csv_bytes,
            file_name="reaction_data.csv",
            mime="text/csv",
            help="Direct CSV download of the active table using pandas .to_csv().",
            use_container_width=True,
        )

    with btn_c4:
        new_path_name = st.text_input("New Pathway Name", value="", placeholder="+ Add Pathway...", label_visibility="collapsed")
        if new_path_name:
            if new_path_name not in st.session_state.df.columns:
                st.session_state.df[new_path_name] = pd.Series([np.nan] * len(st.session_state.df), dtype="float64")
                cur_pal = PALETTES.get(st.session_state.get("chosen_palette", "Pacific Sunset (Custom)"), ["#103E84"])
                c_idx = len(st.session_state.path_meta) % len(cur_pal)
                st.session_state.path_meta.append({
                    "col_name": new_path_name,
                    "display_name": new_path_name,
                    "color": cur_pal[c_idx],
                    "style": "solid",
                })
                safe_rerun()

    # Ensure all pathway columns are float64 with np.nan for empty cells
    for col in path_cols:
        if col in st.session_state.df.columns:
            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors="coerce").astype("float64")

    # Column configuration for st.data_editor
    col_config = {
        "Step Name": st.column_config.TextColumn(
            "Step / Species Name",
            help="Reaction state name (e.g., TS-I, MECP). For branched pathways with different intermediates at the same X, separate them with ' / ' (e.g., 'MECP-SET / TS-HAT'). Supports LaTeX such as $\\mathbf{TS\\text{-}I}$.",
            required=True,
        ),
        "Is TS?": st.column_config.CheckboxColumn(
            "Is TS?",
            help="Check if this state represents a transition state or MECP (used for kinetic analysis and ΔΔG calculation).",
            default=False,
        ),
        "Show Energy?": st.column_config.CheckboxColumn(
            "Show Energy?",
            help="Check to display the numerical energy label on the plot above/below the point.",
            default=False,
        ),
    }
    for col in path_cols:
        col_config[col] = st.column_config.NumberColumn(
            col,
            help=f"Energy values for {col} in {st.session_state.units}. Blank = None.",
            format="%.2f",
            step=0.01,
        )

    edited_df = st.data_editor(
        st.session_state.df,
        column_config=col_config,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        key="data_editor",
    )
    st.session_state.df = edited_df
    # Preserve float64 dtype on all pathway columns after user edits
    for col in path_cols:
        if col in st.session_state.df.columns:
            st.session_state.df[col] = pd.to_numeric(st.session_state.df[col], errors="coerce").astype("float64")

    # ── Advanced Profile Controls (Tabs instead of collapsed accordions) ──────
    st.markdown("---")
    if "sections_list" not in st.session_state:
        st.session_state.sections_list = []  # list of {"label": str, "start": int, "end": int}
    _n_steps_for_sec = len(st.session_state.df)

    tab_styles, tab_sections, tab_ddg, tab_tools = st.tabs([
        "🎨 Pathway Styles & Colors",
        "📐 Sections & Brackets",
        "⚡ ΔΔG Barrier Comparison",
        "🛠️ Data Helpers",
    ])

    # ── Tab 1: Pathway Styles & Colors ────────────────────────────────────────
    with tab_styles:
        leg_c1, leg_c2 = st.columns([3, 2])
        with leg_c1:
            is_leg = st.session_state.get("show_legend", True)
            st.markdown(f"**Legend on plot:** {'✅ Active / Visible' if is_leg else '❌ Disabled / Hidden'}")
        with leg_c2:
            if st.button("❌ Hide Legend" if is_leg else "✅ Show Legend", key="btn_toggle_legend_tab"):
                st.session_state.show_legend = not is_leg
                safe_rerun()
        st.markdown("---")
        chosen_pal_name = st.session_state.get("chosen_palette", list(PALETTES.keys())[0])
        active_pal_colors = PALETTES.get(chosen_pal_name, list(PALETTES.values())[0])

        st.caption(f"🎨 Active Palette: **{chosen_pal_name}**")
        p_chips_html = "<div style='display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px;'>" + "".join([
            f"<div style='display:inline-flex; align-items:center; background:#f6f8fa; border:1px solid #d0d7de; border-radius:5px; padding:2px 7px; font-size:11px; font-family:monospace;'>"
            f"<span style='display:inline-block; width:12px; height:12px; background-color:{c}; border-radius:3px; border:1px solid #777; margin-right:5px;'></span>"
            f"<span style='color:#1f2328; font-weight:600;'>{c}</span>"
            f"</div>"
            for c in active_pal_colors
        ]) + "</div>"
        st.markdown(p_chips_html, unsafe_allow_html=True)

        for m in st.session_state.path_meta:
            col_id = m["col_name"]
            if col_id not in path_cols:
                continue
            sc1, sc2, sc3, sc4 = st.columns([2.6, 3.4, 1.8, 0.6])
            with sc1:
                m["display_name"] = st.text_input(
                    f"Legend Name: {col_id}",
                    value=m.get("display_name", col_id),
                    help="Supports LaTeX math, e.g. $\\mathit{SET\\ pathway}$.",
                    key=f"name_{col_id}",
                )
            with sc2:
                cur_color = m.get("color", "#103E84")
                pal_choices = list(active_pal_colors)
                if cur_color not in pal_choices:
                    pal_choices.append(cur_color)

                def _fmt_pref(c: str) -> str:
                    if c in active_pal_colors:
                        idx = active_pal_colors.index(c) + 1
                        return f"★ {c} (Palette #{idx})"
                    return f"🎨 {c} (Custom)"

                col_sel, col_pick = st.columns([2.3, 1.0])
                with col_sel:
                    idx_choice = pal_choices.index(cur_color) if cur_color in pal_choices else 0
                    chosen_pref = st.selectbox(
                        f"Color ({col_id}):",
                        options=pal_choices,
                        index=idx_choice,
                        format_func=_fmt_pref,
                        key=f"sel_pref_{col_id}",
                        help="Choose directly from active palette favorites, or customize via the color picker.",
                    )
                with col_pick:
                    picked_c = st.color_picker(
                        "Picker:",
                        value=cur_color,
                        key=f"color_{col_id}",
                        help="Pick or type any custom HEX color.",
                    )

                if chosen_pref != cur_color:
                    m["color"] = chosen_pref
                    safe_rerun()
                elif picked_c != cur_color:
                    m["color"] = picked_c
                    safe_rerun()

            with sc3:
                m["style"] = st.selectbox(
                    "Line Style",
                    options=["solid", "dashed", "dotted", "dashdot"],
                    index=["solid", "dashed", "dotted", "dashdot"].index(m.get("style", "solid")),
                    key=f"style_{col_id}",
                )
            with sc4:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if len(path_cols) > 1 and st.button("🗑️", key=f"del_{col_id}", help=f"Remove pathway {col_id}"):
                    st.session_state.df.drop(columns=[col_id], inplace=True)
                    st.session_state.path_meta = [pm for pm in st.session_state.path_meta if pm["col_name"] != col_id]
                    safe_rerun()

        # Pathway-Specific Names (Visual Editor)
        multi_active_rows = []
        for r_i in range(len(st.session_state.df)):
            active_paths = [
                c for c in path_cols
                if pd.notna(st.session_state.df.at[r_i, c]) and st.session_state.df.at[r_i, c] is not None
            ]
            if len(active_paths) > 1:
                multi_active_rows.append((r_i, active_paths))

        if multi_active_rows:
            st.markdown("---")
            st.markdown("##### 🏷️ Distinct Pathway Names at Shared Steps")
            st.caption(
                "If two or more pathways have different intermediates/TSs at the same step (same X coordinate), "
                "edit their specific labels here:"
            )
            changed_names = False
            for r_i, a_paths in multi_active_rows:
                s_current = str(st.session_state.df.at[r_i, "Step Name"]).strip()
                tokens = [t.strip() for t in s_current.split("/") if t.strip()] if "/" in s_current else [s_current]

                st.markdown(f"**Step {r_i}** (active on: *{', '.join(a_paths)}*)")
                n_cols = st.columns(len(a_paths))
                new_tokens = []
                for p_idx, (p_col, col_box) in enumerate(zip(a_paths, n_cols)):
                    initial_val = tokens[p_idx] if p_idx < len(tokens) else (tokens[0] if tokens else f"Step {r_i}")
                    with col_box:
                        new_val = st.text_input(
                            f"{p_col}:",
                            value=initial_val,
                            key=f"pname_{r_i}_{p_col}",
                        )
                        new_tokens.append(new_val.strip())

                combined_name = " / ".join(new_tokens)
                if combined_name != s_current:
                    st.session_state.df.at[r_i, "Step Name"] = combined_name
                    changed_names = True

            if changed_names:
                safe_rerun()

    # ── Tab 2: Sections & Brackets ────────────────────────────────────────────
    with tab_sections:
        st.caption(
            "Add horizontal annotations that delimit step groups on the plot. "
            "Each section appears as a bidirectional arrow with a label below (or above) the points."
        )

        to_del_sec = []
        for s_idx, sec in enumerate(st.session_state.sections_list):
            sc1, sc2, sc3, sc4 = st.columns([3, 2, 2, 1])
            with sc1:
                sec["label"] = st.text_input(
                    "Section Label",
                    value=sec.get("label", f"Section {s_idx + 1}"),
                    key=f"sec_lbl_{s_idx}",
                )
            with sc2:
                sec["start"] = st.number_input(
                    "Start Step (index)",
                    min_value=0,
                    max_value=max(0, _n_steps_for_sec - 1) if _n_steps_for_sec else 0,
                    value=int(sec.get("start", 0)),
                    step=1,
                    key=f"sec_start_{s_idx}",
                )
            with sc3:
                sec["end"] = st.number_input(
                    "End Step (index)",
                    min_value=0,
                    max_value=max(0, _n_steps_for_sec - 1) if _n_steps_for_sec else 0,
                    value=int(sec.get("end", min(1, _n_steps_for_sec - 1) if _n_steps_for_sec > 1 else 0)),
                    step=1,
                    key=f"sec_end_{s_idx}",
                )
            with sc4:
                st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"del_sec_{s_idx}", help="Remove section"):
                    to_del_sec.append(s_idx)

        if to_del_sec:
            for d_i in sorted(to_del_sec, reverse=True):
                st.session_state.sections_list.pop(d_i)
            safe_rerun()

        if st.button("➕ Add Section", key="btn_add_section"):
            n_existing = len(st.session_state.sections_list)
            st.session_state.sections_list.append({
                "label": f"Section {n_existing + 1}",
                "start": 0,
                "end": min(1, _n_steps_for_sec - 1) if _n_steps_for_sec > 1 else 0,
            })
            safe_rerun()

        if st.session_state.sections_list:
            st.markdown("---")
            st.session_state.sections_shift = st.slider(
                "↕ Vertical sections offset",
                min_value=-0.5,
                max_value=0.5,
                value=float(st.session_state.get("sections_shift", 0.05)),
                step=0.01,
                help="Distance of section arrows from the bottom edge of the plot (fraction of total plot height).",
                key="slider_sections_shift",
            )

    # ── Tab 3: ΔΔG Barrier Comparison ─────────────────────────────────────────
    with tab_ddg:
        st.markdown(
            r"Quickly compare the energy of two states or transition states to calculate **$\Delta\Delta G$**. "
            "You can compute it relative to a common zero or by specifying a common starting intermediate."
        )
        valid_points_list: list[dict[str, Any]] = []
        for r_i in range(len(st.session_state.df)):
            s_name = str(st.session_state.df.at[r_i, "Step Name"]).strip()
            toks = [t.strip() for t in s_name.split("/") if t.strip()]
            act_cols = [c for c in path_cols if pd.notna(st.session_state.df.at[r_i, c]) and st.session_state.df.at[r_i, c] is not None]
            is_ts_row = bool(st.session_state.df.at[r_i, "Is TS?"]) if "Is TS?" in st.session_state.df.columns else False
            ts_badge = "⚡ [TS] " if is_ts_row else ""
            for p_i, col in enumerate(act_cols):
                val = float(st.session_state.df.at[r_i, col])
                raw_tok = toks[p_i] if p_i < len(toks) else s_name
                p_label = raw_tok if raw_tok and raw_tok.lower() not in ("none", "nan", "-") else ""
                lbl_suffix = f": {p_label}" if p_label else ""
                valid_points_list.append({
                    "step_idx": r_i,
                    "path": col,
                    "label": p_label,
                    "energy": val,
                    "is_ts": is_ts_row,
                    "display": f"{ts_badge}Step {r_i} [{col}]{lbl_suffix} ({val:.1f} {st.session_state.units})",
                })

        if len(valid_points_list) < 2:
            st.info(r"Please enter at least two energy points in the table to calculate $\Delta\Delta G$.")
        else:
            dd_c1, dd_c2, dd_c3 = st.columns([3, 3, 3])
            with dd_c1:
                state_1_idx = st.selectbox(
                    "State 1 (TS₁ / Step A):",
                    options=list(range(len(valid_points_list))),
                    format_func=lambda i: valid_points_list[i]["display"],
                    index=0,
                    key="ddg_state_1",
                )
            with dd_c2:
                default_s2 = min(1, len(valid_points_list) - 1)
                state_2_idx = st.selectbox(
                    "State 2 (TS₂ / Step B):",
                    options=list(range(len(valid_points_list))),
                    format_func=lambda i: valid_points_list[i]["display"],
                    index=default_s2,
                    key="ddg_state_2",
                )
            with dd_c3:
                ref_mode = st.radio(
                    "Reference:",
                    options=["Common zero (G₂ − G₁)", "Relative to common intermediate"],
                    index=0,
                    key="ddg_ref_mode",
                )

            pt1 = valid_points_list[state_1_idx]
            pt2 = valid_points_list[state_2_idx]

            g_ref = 0.0
            ref_label = "common zero"
            if ref_mode == "Relative to common intermediate":
                ref_idx = st.selectbox(
                    "Common reference intermediate (Ground State):",
                    options=list(range(len(valid_points_list))),
                    format_func=lambda i: valid_points_list[i]["display"],
                    key="ddg_ref_intermediate",
                )
                g_ref = valid_points_list[ref_idx]["energy"]
                ref_label = valid_points_list[ref_idx]["label"]

            barr_1 = pt1["energy"] - g_ref
            barr_2 = pt2["energy"] - g_ref
            ddg = barr_2 - barr_1

            res_c1, res_c2 = st.columns([2, 3])
            with res_c1:
                st.metric(
                    label=r"$\Delta\Delta G$ (State 2 − State 1)",
                    value=f"{ddg:+.2f} {st.session_state.units}",
                    delta=f"{abs(ddg):.2f} {st.session_state.units} diff",
                    delta_color="off",
                )
            with res_c2:
                if abs(ddg) < 1e-4:
                    st.info(r"The two states have exactly the same energy level ($\Delta\Delta G = 0.00$).")
                elif ddg > 0:
                    st.success(
                        f"🏆 **{pt1['label']}** ({pt1['path']}) is thermodynamically/kinetically favored by **{ddg:.2f} {st.session_state.units}** over **{pt2['label']}** ({pt2['path']})."
                    )
                else:
                    st.success(
                        f"🏆 **{pt2['label']}** ({pt2['path']}) is thermodynamically/kinetically favored by **{abs(ddg):.2f} {st.session_state.units}** over **{pt1['label']}** ({pt1['path']})."
                    )

            add_c1, add_c2 = st.columns([2, 3])
            with add_c1:
                if st.button("➕ Add this value to plot", key="btn_add_ddg_quota"):
                    if "ddg_items" not in st.session_state:
                        st.session_state.ddg_items = []
                    cur_pal = PALETTES.get(st.session_state.get("chosen_palette", "Pacific Sunset (Custom)"), ["#D53302", "#103E84", "#000145"])
                    color_item = cur_pal[len(st.session_state.ddg_items) % len(cur_pal)]
                    st.session_state.ddg_items.append({
                        "pt1": pt1,
                        "pt2": pt2,
                        "ddg": ddg,
                        "color": color_item,
                        "x_offset": 0.35,
                        "show_on_plot": True,
                        "title": f"{pt1['label']} vs {pt2['label']} (ΔΔG = {abs(ddg):.1f})",
                    })
                    st.success("Value added to plot!")
                    safe_rerun()
            with add_c2:
                st.session_state.show_ddg_indicator = st.checkbox(
                    r"Quick preview of this selection on plot",
                    value=st.session_state.get("show_ddg_indicator", False),
                    help="Instantly show the selected value above on the plot without adding it to the permanent list.",
                    key="chk_show_ddg_indicator",
                )
                st.session_state.ddg_plot_data = {
                    "pt1": pt1,
                    "pt2": pt2,
                    "ddg": ddg,
                }

            # Display list of added DDG items if any
            if st.session_state.get("ddg_items"):
                st.markdown("---")
                st.markdown(r"##### 📌 Saved $\Delta\Delta G$ markers on plot:")
                to_delete = []
                for idx, item in enumerate(st.session_state.ddg_items):
                    ic1, ic2, ic3, ic4, ic5 = st.columns([1, 4, 2, 2, 2])
                    with ic1:
                        if st.button("🗑️", key=f"del_ddg_item_{idx}", help="Remove this value"):
                            to_delete.append(idx)
                    with ic2:
                        st.markdown(f"**{item.get('title', 'Value')}**  \n*{item['pt1']['path']}* vs *{item['pt2']['path']}*")
                    with ic3:
                        item["show_on_plot"] = st.checkbox("Show", value=item.get("show_on_plot", True), key=f"vis_ddg_item_{idx}")
                    with ic4:
                        item["x_offset"] = st.number_input("X Offset", value=float(item.get("x_offset", 0.35)), step=0.05, key=f"off_ddg_item_{idx}")
                    with ic5:
                        item["color"] = st.color_picker("Color", value=item.get("color", "#D53302"), key=f"col_ddg_item_{idx}")

                if to_delete:
                    for d_idx in sorted(to_delete, reverse=True):
                        st.session_state.ddg_items.pop(d_idx)
                    safe_rerun()

    # ── Tab 4: Data Helpers ───────────────────────────────────────────────────
    with tab_tools:
        tool_col1, tool_col2 = st.columns([1, 1], gap="large")
        with tool_col1:
            st.markdown("##### 🌿 Branch Assistant")
            st.caption("Create a new branch starting from an existing intermediate. Pre-branch cells are filled with `None`.")
            step_opts = list(range(len(st.session_state.df)))
            def _step_format(idx: int) -> str:
                s_name = st.session_state.df['Step Name'].iloc[idx]
                return f"Step {idx}: {s_name}"
            branch_step = st.selectbox(
                "Starting Intermediate (Junction):",
                options=step_opts,
                format_func=_step_format,
                key="branch_junction_step",
            )
            new_br_name = st.text_input("New pathway name:", value="Branch Pathway", key="new_branch_name_input")
            br_style = st.selectbox("Line style:", options=["dashed", "solid", "dotted", "dashdot"], index=0, key="new_branch_style")
            br_energies_raw = st.text_input(
                "Branch energies (space/comma-separated):",
                placeholder="e.g. 46.9  39.2  16.6",
                key="new_branch_energies_raw",
                help="Enter energy levels subsequent to the starting point.",
            )
            if st.button("➕ Create & Insert Branch", key="btn_apply_branch", use_container_width=True):
                parsed_vals = parse_pasted_numbers(br_energies_raw)
                if not new_br_name.strip():
                    st.error("Please enter a valid pathway name!")
                elif new_br_name.strip() in st.session_state.df.columns:
                    st.error(f"Pathway '{new_br_name}' already exists!")
                else:
                    new_br_clean = new_br_name.strip()
                    # Find junction energy from first active pathway at branch_step
                    junction_energy = None
                    for c in path_cols:
                        v = st.session_state.df.at[branch_step, c]
                        if pd.notna(v) and v is not None:
                            junction_energy = float(v)
                            break
                    if junction_energy is None:
                        junction_energy = 0.0

                    total_points = [junction_energy] + (parsed_vals or [])
                    needed_len = branch_step + len(total_points)
                    cur_len = len(st.session_state.df)

                    if needed_len > cur_len:
                        extra = needed_len - cur_len
                        new_rows_data = {
                            "Step Name": ["" for _ in range(extra)],
                            "Is TS?": [False] * extra,
                            "Show Energy?": [False] * extra,
                        }
                        for c in path_cols:
                            new_rows_data[c] = [np.nan] * extra
                        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_rows_data)], ignore_index=True)

                    new_col_vals = [np.nan] * len(st.session_state.df)
                    for idx, val in enumerate(total_points):
                        new_col_vals[branch_step + idx] = float(val) if (val is not None and pd.notna(val)) else np.nan
                    st.session_state.df[new_br_clean] = pd.Series(new_col_vals, dtype="float64")

                    cur_pal = PALETTES.get(st.session_state.get("chosen_palette", "Pacific Sunset (Custom)"), ["#103E84"])
                    color_idx = len(st.session_state.path_meta) % len(cur_pal)
                    st.session_state.path_meta.append({
                        "col_name": new_br_clean,
                        "display_name": new_br_clean,
                        "color": cur_pal[color_idx],
                        "style": br_style,
                    })
                    st.success(f"Branched pathway '{new_br_clean}' successfully created!")
                    safe_rerun()

        with tool_col2:
            st.markdown("##### 📋 Quick Paste Helper")
            st.caption("Paste numbers copied from Excel or logs (separated by spaces, tabs, commas, or newlines).")
            raw_paste_text = st.text_area(
                "Paste numbers here:",
                placeholder="e.g. 42.0  47.1  13.9  16.6  -24.1",
                height=105,
            )
            target_path = st.selectbox("Target Pathway:", options=path_cols)
            start_step = st.number_input(
                "Start at step index:",
                min_value=0,
                max_value=max(0, len(st.session_state.df) - 1),
                value=0,
                help="Previous steps will be left blank without having to count them.",
            )
            if st.button("📥 Apply numbers to table", key="btn_apply_paste", use_container_width=True):
                parsed_vals = parse_pasted_numbers(raw_paste_text)
                if parsed_vals:
                    needed_rows = start_step + len(parsed_vals)
                    cur_len = len(st.session_state.df)
                    if needed_rows > cur_len:
                        extra_rows = needed_rows - cur_len
                        new_data = {
                            "Step Name": ["" for _ in range(extra_rows)],
                            "Is TS?": [False] * extra_rows,
                            "Show Energy?": [False] * extra_rows,
                        }
                        for c in path_cols:
                            new_data[c] = [np.nan] * extra_rows
                        st.session_state.df = pd.concat([st.session_state.df, pd.DataFrame(new_data)], ignore_index=True)

                    for idx, val in enumerate(parsed_vals):
                        st.session_state.df.at[start_step + idx, target_path] = float(val) if (val is not None and pd.notna(val)) else np.nan
                    st.session_state.df[target_path] = pd.to_numeric(st.session_state.df[target_path], errors="coerce").astype("float64")
                    st.success(f"Pasted {len(parsed_vals)} values into '{target_path}' starting from step {start_step}!")
                    safe_rerun()


    # ── Construct Profile and Manual Plot Trigger ─────────────────────────────
    st.markdown("---")
    
    st.subheader("📊 Live Reaction Profile")
    btn_generate = st.button("📈 Generate Plot", type="primary", use_container_width=True)

    if btn_generate:
        n_steps = len(st.session_state.df)
        active_path_cols = [c for c in st.session_state.df.columns if c not in non_path_cols]

        if n_steps == 0 or not active_path_cols:
            st.warning("Please define at least one step and one pathway in the table above.")
        else:
            with st.spinner("Generating publication plot..."):
                # Extract X coordinates
                cur_p_step = int(st.session_state.get("photo_step", 0)) if st.session_state.has_photo else 0
                x_coords = compute_x_coords(n_steps, st.session_state.has_photo, cur_p_step)

                # Extract labelpoint indices
                show_energy_col = list(st.session_state.df["Show Energy?"]) if "Show Energy?" in st.session_state.df.columns else list(st.session_state.df.get("Label Energy? (TS/MECP)", [False] * n_steps))
                labelpoint_indices = [
                    i for i, checked in enumerate(show_energy_col) if checked
                ]

                # Meta lookup
                meta_lookup = {m["col_name"]: m for m in st.session_state.path_meta}

                per_path_step_labels: dict[str, list[Optional[str]]] = {col: [] for col in active_path_cols}
                base_clean_steps: list[str] = []

                for r_idx, raw_s in enumerate(st.session_state.df["Step Name"]):
                    raw_str = str(raw_s).strip() if pd.notna(raw_s) else ""
                    s_text = "" if not raw_str or raw_str.lower() in ("none", "null", "nan", "-") else raw_str
                    row_active = [
                        col for col in active_path_cols
                        if pd.notna(st.session_state.df.at[r_idx, col]) and st.session_state.df.at[r_idx, col] is not None
                    ]
                    sep = "/" if "/" in s_text else ("|" if "|" in s_text else None)
                    if sep and len(row_active) > 1 and s_text:
                        tokens = [t.strip() for t in s_text.split(sep) if t.strip()]
                        if len(tokens) > 1:
                            for idx, p_col in enumerate(row_active):
                                tok = tokens[idx] if idx < len(tokens) else tokens[-1]
                                per_path_step_labels[p_col].append(tok)
                            for p_col in active_path_cols:
                                if p_col not in row_active:
                                    per_path_step_labels[p_col].append(None)
                            base_clean_steps.append(tokens[0])
                            continue

                    base_clean_steps.append(s_text)
                    for p_col in active_path_cols:
                        per_path_step_labels[p_col].append(s_text if (p_col in row_active and s_text) else None)

                # Construct input.txt content
                lines = ["--# PES"]
                for col in active_path_cols:
                    vals = []
                    for v in st.session_state.df[col]:
                        if pd.isna(v) or v is None:
                            vals.append("None")
                        else:
                            vals.append(str(round(float(v), 2)))
                    lines.append("; ".join(vals))

                lines.append("\n--# LABELS")
                for col in active_path_cols:
                    col_lbls = [str(l) if l is not None else "None" for l in per_path_step_labels[col]]
                    lines.append("; ".join(col_lbls))

                # Section brackets
                valid_sections = []
                for s in st.session_state.get("sections_list", []):
                    lbl = s.get("label", "")
                    try:
                        s_start = int(s.get("start", 0))
                        s_end = int(s.get("end", 0))
                    except (TypeError, ValueError):
                        continue
                    if lbl and s_end > s_start:
                        valid_sections.append({"label": lbl, "start": s_start, "end": s_end})

                lines.append("\n--# GRAPHS")
                lines.append(f"title : {st.session_state.title}")
                lines.append(f"label : {'; '.join(meta_lookup.get(col, {}).get('display_name', col) for col in active_path_cols)}")
                lines.append(f"x : {'; '.join(str(x) for x in x_coords)}")
                lines.append(f"colors : {'; '.join(meta_lookup.get(col, {}).get('color', '#103E84') for col in active_path_cols)}")
                lines.append(f"style : {'; '.join(meta_lookup.get(col, {}).get('style', 'solid') for col in active_path_cols)}")
                lines.append(f"figsize : {st.session_state.fig_w}; {st.session_state.fig_h}")
                lines.append(f"font_size : {st.session_state.font_size}")
                lines.append(f"units : {st.session_state.units}")
                lines.append(f"marker : {st.session_state.point_type}")
                lines.append(f"curviness : {float(round(st.session_state.curviness, 2))}")
                lines.append(f"axes : {st.session_state.axes_style}")
                lines.append("labels : true")
                if labelpoint_indices:
                    lines.append(f"labelpoint : {'; '.join(str(i) for i in labelpoint_indices)}")
                if not st.session_state.show_xticks:
                    lines.append("xticks : false")
                if not st.session_state.get("show_legend", True):
                    lines.append("legend : false")
                if st.session_state.has_photo:
                    lines.append(f"excitation : {cur_p_step}")
                if valid_sections:
                    lines.append(f"sections : {'; '.join(s['label'] for s in valid_sections)}")
                    lines.append(f"sections_start : {'; '.join(str(s['start']) for s in valid_sections)}")
                    lines.append(f"sections_end : {'; '.join(str(s['end']) for s in valid_sections)}")
                    lines.append(f"sections_shift : {float(round(st.session_state.get('sections_shift', 0.05), 3))}")
                if st.session_state.get("transparent_bg", False):
                    lines.append("transparent : true")
                elif st.session_state.get("bg_color"):
                    lines.append(f"bg_color : {st.session_state.bg_color}")
                lines.append("save : false\n")

                txt_content = "\n".join(lines)

                # Construct YAML content
                yaml_dict: dict[str, Any] = {
                    "title": st.session_state.title,
                    "units": st.session_state.units,
                    "marker": st.session_state.point_type,
                    "curviness": float(round(st.session_state.curviness, 2)),
                    "font_size": int(st.session_state.font_size),
                    "figsize": [float(st.session_state.fig_w), float(st.session_state.fig_h)],
                    "xticks": bool(st.session_state.show_xticks),
                    "legend": bool(st.session_state.get("show_legend", True)),
                }
                if st.session_state.get("transparent_bg", False):
                    yaml_dict["transparent"] = True
                elif st.session_state.get("bg_color"):
                    yaml_dict["bg_color"] = st.session_state.bg_color
                if st.session_state.has_photo:
                    yaml_dict["excitation"] = int(cur_p_step)

                yaml_dict["steps"] = base_clean_steps

                selected_labels = []
                for idx, (sname, checked) in enumerate(zip(base_clean_steps, show_energy_col)):
                    if checked:
                        for col in active_path_cols:
                            l_val = per_path_step_labels[col][idx]
                            if l_val and l_val not in selected_labels:
                                selected_labels.append(l_val)
                if selected_labels:
                    yaml_dict["label_points"] = selected_labels

                if "Is TS?" in st.session_state.df.columns:
                    ts_labels = []
                    for idx, (sname, is_ts) in enumerate(zip(base_clean_steps, st.session_state.df["Is TS?"])):
                        if is_ts:
                            for col in active_path_cols:
                                l_val = per_path_step_labels[col][idx]
                                if l_val and l_val not in ts_labels:
                                    ts_labels.append(l_val)
                    if ts_labels:
                        yaml_dict["transition_states"] = ts_labels

                paths_out = []
                for col in active_path_cols:
                    meta = meta_lookup.get(col, {})
                    p_info: dict[str, Any] = {
                        "name": meta.get("display_name", col),
                        "color": meta.get("color", "#103E84"),
                        "style": meta.get("style", "solid"),
                    }
                    raw_vals = list(st.session_state.df[col])
                    first_idx = next((i for i, v in enumerate(raw_vals) if pd.notna(v) and v is not None), 0)
                    last_idx = len(raw_vals) - 1
                    while last_idx >= 0 and (pd.isna(raw_vals[last_idx]) or raw_vals[last_idx] is None):
                        last_idx -= 1

                    if first_idx > 0 and base_clean_steps[first_idx] and base_clean_steps.count(base_clean_steps[first_idx]) == 1:
                        p_info["branch_at"] = base_clean_steps[first_idx]
                        sub_vals = [
                            round(float(v), 2) if pd.notna(v) and v is not None else None
                            for v in raw_vals[first_idx : last_idx + 1]
                        ]
                        p_info["energies"] = sub_vals
                    else:
                        p_info["energies"] = [
                            round(float(v), 2) if pd.notna(v) and v is not None else None
                            for v in raw_vals[: last_idx + 1]
                        ]

                    # Check for pathway-specific overrides in this range
                    my_lbls = per_path_step_labels[col]
                    overrides = {}
                    for r_i in range(first_idx, last_idx + 1):
                        if my_lbls[r_i] and my_lbls[r_i] != base_clean_steps[r_i]:
                            overrides[base_clean_steps[r_i]] = my_lbls[r_i]
                    if overrides:
                        p_info["labels"] = overrides

                    paths_out.append(p_info)
                yaml_dict["pathways"] = paths_out

                # Axes frame style
                _axes_val = st.session_state.get("axes_style", "none")
                if _axes_val and _axes_val != "none":
                    yaml_dict["axes"] = _axes_val
                else:
                    yaml_dict["axes"] = "none"

                # Section brackets
                if valid_sections:
                    yaml_dict["sections"] = [s["label"] for s in valid_sections]
                    yaml_dict["sections_start"] = [s["start"] for s in valid_sections]
                    yaml_dict["sections_end"] = [s["end"] for s in valid_sections]
                    yaml_dict["sections_shift"] = float(round(st.session_state.get("sections_shift", 0.05), 3))

                yaml_content = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True)

                try:
                    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=False) as tmp:
                        tmp.write(yaml_content)
                        tmp_file = tmp.name

                    analyzer = ReactionProfileAnalyzer(tmp_file, auto_plot=False)
                    is_trans = bool(st.session_state.get("transparent_bg", False))
                    bg_c = st.session_state.get("bg_color", "#FFFFFF")

                    fig, ax = analyzer.plot(
                        dpi=200,
                        font_size=int(st.session_state.font_size),
                        transparent=is_trans,
                        bg_color=bg_c if not is_trans else None,
                    )

                    if is_trans:
                        fig.patch.set_alpha(0.0)
                        ax.patch.set_alpha(0.0)
                    elif bg_c:
                        fig.patch.set_facecolor(bg_c)
                        ax.set_facecolor(bg_c)

                    box_bg = "white" if is_trans else bg_c

                    # Draw all configured DeltaDeltaG indicators on plot
                    rendered_pairs = set()
                    for item in st.session_state.get("ddg_items", []):
                        if not item.get("show_on_plot", True):
                            continue
                        p1, p2, val_ddg = item["pt1"], item["pt2"], item["ddg"]
                        pair_key = (p1["step_idx"], p1["path"], p2["step_idx"], p2["path"])
                        rendered_pairs.add(pair_key)

                        col_box = item.get("color", "#D53302")
                        x_off = float(item.get("x_offset", 0.35))
                        x1 = x_coords[p1["step_idx"]] if p1["step_idx"] < len(x_coords) else p1["step_idx"]
                        x2 = x_coords[p2["step_idx"]] if p2["step_idx"] < len(x_coords) else p2["step_idx"]
                        y1 = p1["energy"]
                        y2 = p2["energy"]
                        x_mid = ((x1 + x2) / 2 if x1 != x2 else x1) + x_off

                        ax.annotate(
                            "",
                            xy=(x_mid, y1),
                            xytext=(x_mid, y2),
                            arrowprops=dict(arrowstyle="<->", color=col_box, lw=1.4, ls="--"),
                        )
                        ax.text(
                            x_mid + 0.08,
                            (y1 + y2) / 2,
                            rf"$\Delta\Delta G = {abs(val_ddg):.1f}$",
                            color=col_box,
                            fontweight="bold",
                            va="center",
                            ha="left",
                            fontsize=max(6, int(st.session_state.font_size) - 1),
                            bbox=dict(boxstyle="round,pad=0.2", facecolor=box_bg, edgecolor=col_box, lw=0.8, alpha=0.9),
                        )

                    # If user checked single quick preview and it wasn't already in ddg_items
                    if st.session_state.get("show_ddg_indicator") and st.session_state.get("ddg_plot_data"):
                        dd_d = st.session_state.ddg_plot_data
                        p1, p2, val_ddg = dd_d["pt1"], dd_d["pt2"], dd_d["ddg"]
                        pair_key = (p1["step_idx"], p1["path"], p2["step_idx"], p2["path"])
                        if pair_key not in rendered_pairs:
                            x1 = x_coords[p1["step_idx"]] if p1["step_idx"] < len(x_coords) else p1["step_idx"]
                            x2 = x_coords[p2["step_idx"]] if p2["step_idx"] < len(x_coords) else p2["step_idx"]
                            y1 = p1["energy"]
                            y2 = p2["energy"]
                            x_mid = ((x1 + x2) / 2 if x1 != x2 else x1) + 0.35

                            ax.annotate(
                                "",
                                xy=(x_mid, y1),
                                xytext=(x_mid, y2),
                                arrowprops=dict(arrowstyle="<->", color="#D53302", lw=1.4, ls="--"),
                            )
                            ax.text(
                                x_mid + 0.08,
                                (y1 + y2) / 2,
                                rf"$\Delta\Delta G = {abs(val_ddg):.1f}$",
                                color="#D53302",
                                fontweight="bold",
                                va="center",
                                ha="left",
                                fontsize=max(6, int(st.session_state.font_size) - 1),
                                bbox=dict(boxstyle="round,pad=0.2", facecolor=box_bg, edgecolor="#D53302", lw=0.8, alpha=0.9),
                            )

                    # High-res export buffers
                    save_fc = "none" if is_trans else bg_c
                    preview_buf = io.BytesIO()
                    fig.savefig(preview_buf, format="png", dpi=200, bbox_inches="tight", transparent=is_trans, facecolor=save_fc)
                    preview_buf.seek(0)

                    png_buf = io.BytesIO()
                    fig.savefig(png_buf, format="png", dpi=600, bbox_inches="tight", transparent=is_trans, facecolor=save_fc)
                    png_buf.seek(0)

                    svg_buf = io.BytesIO()
                    fig.savefig(svg_buf, format="svg", bbox_inches="tight", transparent=is_trans, facecolor=save_fc)
                    svg_buf.seek(0)

                    pdf_buf = io.BytesIO()
                    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight", transparent=is_trans, facecolor=save_fc)
                    pdf_buf.seek(0)

                    plt.close(fig)

                    st.session_state.plot_result = {
                        "preview_png": preview_buf.getvalue(),
                        "png": png_buf.getvalue(),
                        "svg": svg_buf.getvalue(),
                        "pdf": pdf_buf.getvalue(),
                        "yaml": yaml_content,
                        "txt": txt_content,
                    }

                except Exception as err:
                    st.error(f"Error rendering plot preview: {err}")

    if not st.session_state.get("plot_result"):
        st.info("💡 Click **'📈 Generate Plot'** above to render the visualization.")
        return

    # Render Preview and Export Controls from stored plot_result
    res = st.session_state.plot_result
    col_plot, col_export = st.columns([3, 2])

    with col_plot:
        try:
            st.image(res["preview_png"], use_container_width=True)
        except TypeError:
            st.image(res["preview_png"], use_column_width=True)

        # Export row
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        with exp_c1:
            st.download_button(
                label="📥 PNG (600 DPI)",
                data=res["png"],
                file_name="reaction_profile.png",
                mime="image/png",
                use_container_width=True,
            )
        with exp_c2:
            st.download_button(
                label="📥 Vector SVG",
                data=res["svg"],
                file_name="reaction_profile.svg",
                mime="image/svg+xml",
                use_container_width=True,
            )
        with exp_c3:
            st.download_button(
                label="📥 Vector PDF",
                data=res["pdf"],
                file_name="reaction_profile.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col_export:
        tab_yaml, tab_txt, tab_csv = st.tabs(["✨ YAML (Recommended)", "📄 Classic TXT", "📊 CSV Data"])
        with tab_yaml:
            st.caption("Clean, human-readable modern input format.")
            st.code(res["yaml"], language="yaml")
            st.download_button(
                label="💾 Download profile.yaml",
                data=res["yaml"],
                file_name="profile.yaml",
                mime="application/x-yaml",
                use_container_width=True,
            )
        with tab_txt:
            st.caption("Legacy section-based input format.")
            st.code(res["txt"], language="ini")
            st.download_button(
                label="💾 Download input.txt",
                data=res["txt"],
                file_name="input.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with tab_csv:
            st.caption("Native CSV format exported from pandas DataFrame.")
            csv_export_bytes = dataframe_to_csv_bytes(st.session_state.df)
            st.code(st.session_state.df.to_csv(index=False), language="csv")
            st.download_button(
                label="💾 Download reaction_data.csv",
                data=csv_export_bytes,
                file_name="reaction_data.csv",
                mime="text/csv",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()