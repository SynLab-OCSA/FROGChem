"""Unit tests for config_manager and gui_utils in reaction_plotter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frogchem.reaction_plotter.config_manager import (
    DEFAULT_PALETTES,
    DEFAULT_PRESETS,
    DEFAULT_THEMES,
    delete_palette,
    delete_preset,
    get_palettes,
    get_presets,
    get_themes,
    load_config,
    save_config,
    save_palette,
    save_preset,
)
from frogchem.reaction_plotter.gui_utils import (
    auto_detect_ts,
    build_dataframe_from_preset,
    compute_x_coords,
    dataframe_to_csv_bytes,
    extract_current_template,
    init_session_state,
    parse_pasted_numbers,
)


def test_config_manager_defaults():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_cfg_path = Path(tmp_dir) / "gui_config.yaml"
        # Not existing yet, should load defaults and save
        cfg = load_config(tmp_cfg_path)
        assert "PALETTES" in cfg
        assert "THEMES" in cfg
        assert "PRESETS" in cfg
        assert tmp_cfg_path.exists()

        # Preset operations
        presets = get_presets(tmp_cfg_path)
        assert len(presets) > 0

        dummy_preset = {
            "title": "Test Preset",
            "units": "kcal/mol",
            "steps": ["A", "B"],
            "pathways": [{"name": "P1", "energies": [0.0, 10.0]}],
        }
        save_preset("Test Custom Preset", dummy_preset, config_path=tmp_cfg_path)
        updated_presets = get_presets(tmp_cfg_path)
        assert "Test Custom Preset" in updated_presets
        assert updated_presets["Test Custom Preset"]["title"] == "Test Preset"

        # Palette operations
        palettes = get_palettes(tmp_cfg_path)
        assert len(palettes) > 0

        save_palette("My Palette", ["#111111", "#222222"], config_path=tmp_cfg_path)
        updated_palettes = get_palettes(tmp_cfg_path)
        assert "My Palette" in updated_palettes
        assert updated_palettes["My Palette"] == ["#111111", "#222222"]

        # Deletions
        assert delete_preset("Test Custom Preset", config_path=tmp_cfg_path) is True
        assert "Test Custom Preset" not in get_presets(tmp_cfg_path)

        assert delete_palette("My Palette", config_path=tmp_cfg_path) is True
        assert "My Palette" not in get_palettes(tmp_cfg_path)

        # Themes
        themes = get_themes(tmp_cfg_path)
        assert "JACS / ACS Classic" in themes


def test_compute_x_coords():
    # Without photo
    assert compute_x_coords(4, has_photo=False) == [0, 1, 2, 3]

    # With photo at step 0: step 0 and 1 share x=0, next is 1, next is 2
    assert compute_x_coords(4, has_photo=True, photo_step=0) == [0, 0, 1, 2]

    # With photo at step 1: step 0 has 0, step 1 has 1, step 2 has 1, step 3 has 2
    assert compute_x_coords(4, has_photo=True, photo_step=1) == [0, 1, 1, 2]


def test_parse_pasted_numbers():
    text = "0.0,  15.5; None\t-4.2 \n 20.1 nan"
    vals = parse_pasted_numbers(text)
    assert vals == [0.0, 15.5, None, -4.2, 20.1, None]


def test_auto_detect_ts():
    df = pd.DataFrame({
        "Step Name": ["Reactants", "TS1", "Int", "MECP", "Products"],
        "Path 1": [0.0, 22.0, 10.0, 18.0, -5.0],
    })
    is_ts = auto_detect_ts(df, ["Path 1"])
    assert is_ts == [False, True, False, True, False]


def test_build_dataframe_from_preset_and_csv():
    preset = {
        "title": "Sample",
        "steps": ["R", "TS", "P"],
        "label_energy": [False, True, False],
        "is_ts": [False, True, False],
        "pathways": [
            {"name": "Path A", "color": "#000000", "style": "solid", "energies": [0.0, 15.0, -10.0]}
        ],
    }
    df, meta = build_dataframe_from_preset(preset)
    assert len(df) == 3
    assert "Path A" in df.columns
    assert meta[0]["col_name"] == "Path A"

    # CSV bytes export
    csv_bytes = dataframe_to_csv_bytes(df)
    assert isinstance(csv_bytes, bytes)
    csv_str = csv_bytes.decode("utf-8")
    assert "Step Name" in csv_str
    assert "Path A" in csv_str
    assert "15.0" in csv_str


def test_extract_current_template_and_init_session_state():
    df = pd.DataFrame({
        "Step Name": ["A", "B"],
        "Is TS?": [False, True],
        "Show Energy?": [False, True],
        "Path 1": [0.0, 10.0],
    })
    path_meta = [{"col_name": "Path 1", "display_name": "Path 1", "color": "#123456", "style": "solid"}]

    state = SimpleNamespace(
        title="My Reaction",
        units="kcal/mol",
        point_type="hollow",
        has_photo=False,
        photo_step=0,
        show_xticks=True,
        show_legend=True,
        fig_w=10.0,
        fig_h=6.0,
        curviness=0.42,
        font_size=12,
        axes_style="box",
        layout_preset="Double Column (A4 — 17.5 cm / 6.9 in)",
        transparent_bg=True,
        bg_color="#1E1E1E",
    )

    tmpl = extract_current_template(df, path_meta, state=state)
    assert tmpl["title"] == "My Reaction"
    assert tmpl["font_size"] == 12
    assert tmpl["axes_style"] == "box"
    assert tmpl["steps"] == ["A", "B"]
    assert tmpl["pathways"][0]["name"] == "Path 1"
    assert tmpl["transparent_bg"] is True
    assert tmpl["bg_color"] == "#1E1E1E"
    assert tmpl["layout_preset"] == "Double Column (A4 — 17.5 cm / 6.9 in)"

    # Test init_session_state
    new_state = SimpleNamespace()
    init_session_state(preset=tmpl, preset_name="My Reaction", force_reset=True, state=new_state)
    assert hasattr(new_state, "df")
    assert hasattr(new_state, "font_size")
    assert new_state.font_size == 12
    assert new_state.transparent_bg is True
    assert new_state.bg_color == "#1E1E1E"
    assert new_state.layout_preset == "Double Column (A4 — 17.5 cm / 6.9 in)"

    # Test 1.5 Column preset template
    state_1_5 = SimpleNamespace(
        title="Intermediate Reaction",
        units="kcal/mol",
        point_type="hollow",
        has_photo=False,
        photo_step=0,
        show_xticks=True,
        show_legend=True,
        fig_w=4.75,
        fig_h=2.75,
        curviness=0.42,
        font_size=12,
        axes_style="none",
        layout_preset="1.5 Column (A4 — 12.0 cm / 4.75 in)",
        transparent_bg=False,
        bg_color="#FFFFFF",
    )
    tmpl_1_5 = extract_current_template(df, path_meta, state=state_1_5)
    assert tmpl_1_5["layout_preset"] == "1.5 Column (A4 — 12.0 cm / 4.75 in)"
    assert tmpl_1_5["fig_w"] == 4.75
    assert tmpl_1_5["fig_h"] == 2.75
    tm_state = SimpleNamespace()
    init_session_state(preset=tmpl_1_5, preset_name="Intermediate", force_reset=True, state=tm_state)
    assert tm_state.layout_preset == "1.5 Column (A4 — 12.0 cm / 4.75 in)"
    assert tm_state.font_size == 12
    assert tm_state.fig_w == 4.75
    assert tm_state.fig_h == 2.75
