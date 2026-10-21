"""Reaction profile plotting.

High-quality 2D reaction coordinate diagrams with cubic Bezier curves,
customizable point markers, multi-step annotations, and style presets.
Modelled after plotprofile.
"""

from __future__ import annotations

import colorsys
import inspect
import logging
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.colors as mpc
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# Default style presets
DEFAULT_STYLES: dict[str, dict[str, Any]] = {
    "default": {
        "figsize": (5, 4.5),
        "point_type": "hollow",
        "curviness": 0.42,
        "desaturate": True,
        "desaturate_factor": 1.2,
        "linestyle": None,
        "dash_spacing": 2.5,
        "labels": True,
        "show_legend": True,
        "line_width": 2.0,
        "bar_width": 3.0,
        "bar_length": 0.3,
        "marker_size": 7,
        "font_size": 12,
        "font_family": "Arial",
        "font_weight": "bold",
        "font_style": "normal",
        "axis_linewidth": 1.0,
        "buffer_factor": 0.06,
        "axes": "box",
        "colors": ["darkcyan", "maroon", "midnightblue", "darkmagenta", "darkgreen", "saddlebrown"],
        "arrow_color": "xkcd:dark grey",
        "annotation_color": "maroon",
        "annotation_size": 11,
        "energy": "G",
        "units": "kcal",
        "annotation_below_arrow": False,
        "annotation_space": 0.01,
        "annotation_buffer": 0.0,
        "arrow_width": 1.5,
        "sig_figs": 1,
        "point_label_color": "black",
        "connect_bar_ends": True,
        "stagger_labels": True,
        "labelpoint": None,
    },
    "presentation": {
        "figsize": (8, 5),
        "font_size": 14,
        "annotation_size": 14,
        "marker_size": 8,
        "line_width": 3.0,
        "bar_width": 3.5,
        "annotation_space": 0.025,
        "arrow_width": 2.5,
    },
    "straight": {
        "curviness": 0.0,
    },
}

_SERIES_KEYS = (
    "colors",
    "curviness",
    "line_width",
    "marker_size",
    "point_type",
    "bar_length",
    "bar_width",
    "connect_bar_ends",
    "desaturate",
    "desaturate_factor",
    "linestyle",
    "dash_spacing",
)

_AXIS_KEYS = (*_SERIES_KEYS, "label")
_LABEL_GAP_POINTS = 2.0

_STYLE_KEYS = frozenset(
    (
        *_SERIES_KEYS,
        "annotation_below_arrow",
        "annotation_buffer",
        "annotation_color",
        "annotation_size",
        "annotation_space",
        "annotation_style",
        "annotation_weight",
        "arrow_color",
        "arrow_width",
        "axes",
        "axis_linewidth",
        "bg_color",
        "buffer_factor",
        "energy",
        "excitation_arrows",
        "figsize",
        "font_family",
        "font_size",
        "font_style",
        "font_weight",
        "labelpoint",
        "labels",
        "legend",
        "point_label_color",
        "show_legend",
        "sig_figs",
        "square",
        "stagger_labels",
        "transparent",
        "units",
        "x_indices",
        "x_label",
        "y1",
        "y2",
        "y_label",
    )
)


def _load_style(style_name: str) -> dict[str, Any]:
    base = dict(DEFAULT_STYLES.get("default", {}))
    overlay = DEFAULT_STYLES.get(style_name, {})
    base.update(overlay)
    return base


def _desaturate_color(color: Any, factor: float = 1.2) -> tuple[float, float, float]:
    """Desaturate a colour, for the fill/line under a curve."""
    rgb = mpc.to_rgb(color)
    hls = colorsys.rgb_to_hls(*rgb)
    hls_new = (hls[0], 1.0 - (0.4 * factor), 0.3 * factor)
    return colorsys.hls_to_rgb(*hls_new)


desaturate_colour = _desaturate_color


def generate_coordinates(energies: Sequence[Optional[float]]) -> tuple[list[float], list[float]]:
    """Map a list of energies to (x, y) points, skipping None/NaN and centring repeats."""
    x_coords: list[float] = []
    y_coords: list[float] = []
    i = 0
    while i < len(energies):
        if energies[i] is None or (isinstance(energies[i], (float, np.floating)) and np.isnan(energies[i])):
            i += 1
            continue

        start_idx = i
        current_energy = energies[i]

        j = i + 1
        while (
            j < len(energies)
            and energies[j] == current_energy
            and energies[j] is not None
            and not (isinstance(energies[j], (float, np.floating)) and np.isnan(energies[j]))
        ):
            j += 1

        if j - start_idx > 1:
            midpoint = (start_idx + j - 1) / 2.0
            x_coords.append(midpoint)
            y_coords.append(cast(float, current_energy))
        else:
            x_coords.append(float(i))
            y_coords.append(cast(float, current_energy))

        i = j
    return x_coords, y_coords


_generate_coordinates = generate_coordinates


def cubic_bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, num: int = 500) -> np.ndarray:
    """Sample a cubic bezier defined by control points p0-p3."""
    t = np.linspace(0, 1, num)
    return (
        ((1 - t) ** 3)[:, None] * p0
        + 3 * ((1 - t) ** 2)[:, None] * t[:, None] * p1
        + 3 * (1 - t)[:, None] * (t**2)[:, None] * p2
        + (t**3)[:, None] * p3
    )


_cubic_bezier_points = cubic_bezier_points


# ── PlotConfig (backward-compatible dataclass) ───────────────────────────────

@dataclass
class PlotConfig:
    label: List[Optional[str]] = field(default_factory=list)
    colors: List[Optional[str]] = field(default_factory=lambda: [None])
    legend: bool = False
    save: bool = False
    relative: bool = False
    labelpoint: Optional[List[int]] = None
    zero: List[int] = field(default_factory=lambda: [0])
    title: str = "Reaction Energy Path"
    style: List[str] = field(default_factory=lambda: ['-'])
    linewidth: List[float] = field(default_factory=lambda: [1.5])
    points: List[str] = field(default_factory=list)
    pointscolor: List[Optional[str]] = field(default_factory=list)
    pointsstyle: List[str] = field(default_factory=list)
    ypad: List[float] = field(default_factory=lambda: [0.0])
    horizontalalignment: List[str] = field(default_factory=lambda: ['left'])
    y_limit: Optional[float] = None
    decimal_digits: int = 1
    directory: str = "reaction"
    sections: List[str] = field(default_factory=list)
    sections_start: List[int] = field(default_factory=list)
    sections_end: List[int] = field(default_factory=list)
    colors_sections: List[str] = field(default_factory=lambda: ['black'])
    position_sections: str = 'bottom'
    figsize: Tuple[float, float] = (10.0, 7.0)
    fontsize_title: int = 12
    fontsize_label: int = 12
    fontsize_ticks: int = 10
    desaturate: bool = False
    show: bool = True
    legend_barriers: Optional[bool] = None
    x: Optional[List[float]] = None
    excitation_arrows: bool = True
    labels: Optional[bool] = None
    show_xticks: bool = True
    xticks: bool = True
    point_type: Optional[str] = None
    curviness: Optional[float] = None
    font_size: int = 11
    axes: str = "none"
    sections_shift: float = 0.05
    units: str = "kcal/mol"
    transparent: bool = False
    bg_color: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> 'PlotConfig':
        config = cls()
        data = dict(data)
        if 'color' in data and 'colors' not in data:
            data['colors'] = data['color']
        if 'xticks' in data and 'show_xticks' not in data:
            data['show_xticks'] = data['xticks']
        if 'show_xticks' in data and 'xticks' not in data:
            data['xticks'] = data['show_xticks']
        if 'marker' in data and 'point_type' not in data:
            data['point_type'] = data['marker']
        if 'font_size' not in data and 'fontsize' in data:
            data['font_size'] = data['fontsize']
        if 'transparent' in data and 'transparent_bg' not in data:
            data['transparent_bg'] = data['transparent']
        if 'transparent_bg' in data and 'transparent' not in data:
            data['transparent'] = data['transparent_bg']

        type_conversions = {
            'legend': lambda x: x[0].lower() == 'true' if x else False,
            'labels': lambda x: x[0].lower() == 'true' if x else False,
            'show_xticks': lambda x: x[0].lower() == 'true' if x else True,
            'xticks': lambda x: x[0].lower() == 'true' if x else True,
            'point_type': lambda x: x[0] if x else None,
            'curviness': lambda x: float(x[0]) if x else None,
            'font_size': lambda x: int(x[0]) if x else 11,
            'axes': lambda x: x[0] if x else "none",
            'sections_shift': lambda x: float(x[0]) if x else 0.05,
            'units': lambda x: x[0] if x else "kcal/mol",
            'transparent': lambda x: x[0].lower() in ('true', '1') if x else False,
            'transparent_bg': lambda x: x[0].lower() in ('true', '1') if x else False,
            'bg_color': lambda x: x[0] if x and x[0].lower() not in ('none', '') else None,
            'legend_barriers': lambda x: x[0].lower() == 'true' if x else False,
            'excitation_arrows': lambda x: x[0].lower() == 'true' if x else False,
            'x': lambda x: [float(i) for i in x if i.strip()],
            'save': lambda x: x[0].lower() == 'true' if x else False,
            'show': lambda x: x[0].lower() == 'true' if x else True,
            'relative': lambda x: x[0].lower() == 'true' if x else False,
            'desaturate': lambda x: x[0].lower() == 'true' if x else False,
            'labelpoint': lambda x: [int(i) if str(i).strip().lstrip('-').isdigit() else str(i).strip() for i in x if str(i).strip()] if x and x != ['None'] else None,
            'zero': lambda x: [int(i) for i in x if i.strip()],
            'title': lambda x: x[0] if x else "Reaction Energy Path",
            'linewidth': lambda x: [float(i) for i in x if i.strip()],
            'ypad': lambda x: [float(i) for i in x if i.strip()],
            'y_limit': lambda x: float(x[0]) if x and x[0] != 'None' else None,
            'decimal_digits': lambda x: int(x[0]) if x else 1,
            'directory': lambda x: x[0] if x else "reaction",
            'sections_start': lambda x: [int(i) for i in x if i.strip()],
            'sections_end': lambda x: [int(i) for i in x if i.strip()],
            'position_sections': lambda x: x[0] if x else "bottom",
            'figsize': lambda x: tuple(float(i) for i in x) if x else (10.0, 7.0),
            'fontsize_title': lambda x: int(x[0]) if x else 12,
            'fontsize_label': lambda x: int(x[0]) if x else 12,
            'fontsize_ticks': lambda x: int(x[0]) if x else 10,
        }
        for key, value in data.items():
            if hasattr(config, key):
                if key in type_conversions:
                    try:
                        setattr(config, key, type_conversions[key](value))
                    except (ValueError, IndexError):
                        import warnings
                        warnings.warn(f"Invalid value for {key}: {value}. Using default.")
                else:
                    setattr(config, key, value or getattr(config, key))
        return config


# ── ReactionProfilePlotter / ReactionPlotter ────────────────────────────────

class ReactionProfilePlotter:
    """Plot reaction energy profiles with plotprofile styling and capabilities."""

    _default_colors = [
        'darkcyan', 'maroon', 'midnightblue',
        'darkmagenta', 'darkgreen', 'saddlebrown',
    ]

    def __init__(self, style: str = "default", **kwargs: Any) -> None:
        # Check for legacy constructor arguments (such as figsize, desaturate)
        unknown = sorted(set(kwargs) - _STYLE_KEYS)
        if unknown:
            # Allow common matplotlib / legacy kwargs with a warning instead of hard error
            for k in list(unknown):
                if k in ("analyzer", "point_label_color", "ylabel", "xlabel"):
                    unknown.remove(k)
            if unknown:
                logger.debug(f"Ignoring unrecognized style keys: {unknown}")

        style_dict = _load_style(style)
        style_dict.update({k: v for k, v in kwargs.items() if k in _STYLE_KEYS})

        self.style_name = style
        self.figsize = tuple(style_dict.get("figsize", (5, 4.5)))
        self.point_type = style_dict.get("point_type", "hollow")
        self.curviness = float(style_dict.get("curviness", 0.42))
        self.desaturate = bool(style_dict.get("desaturate", True))
        self.desaturate_factor = float(style_dict.get("desaturate_factor", 1.2))
        self.linestyle = style_dict.get("linestyle", None)
        self.labels = bool(style_dict.get("labels", True))
        self.show_legend = bool(style_dict.get("show_legend", True))
        self.line_width = float(style_dict.get("line_width", 2.5))
        self.bar_width = float(style_dict.get("bar_width", 3.0))
        self.bar_length = float(style_dict.get("bar_length", 0.3))
        self.marker_size = float(style_dict.get("marker_size", 6))
        self.font_size = int(style_dict.get("font_size", 12))
        self.axes = style_dict.get("axes", "box")
        self.axis_linewidth = float(style_dict.get("axis_linewidth", 2.0))
        self.colors = style_dict.get("colors", self._default_colors)
        self.arrow_color = style_dict.get("arrow_color", "xkcd:dark grey")
        self.annotation_color = style_dict.get("annotation_color", "maroon")
        self.buffer_factor = float(style_dict.get("buffer_factor", 0.06))
        self.energy = style_dict.get("energy", "G")
        self.units = style_dict.get("units", "kcal")
        self.annotation_space = float(style_dict.get("annotation_space", 0.01))
        self.arrow_width = float(style_dict.get("arrow_width", 1.5))
        self.annotation_buffer = float(style_dict.get("annotation_buffer", 0.0))
        self.sig_figs = int(style_dict.get("sig_figs", 1))
        self.point_label_color = style_dict.get("point_label_color", "black")
        self.annotation_below_arrow = bool(style_dict.get("annotation_below_arrow", False))
        self.connect_bar_ends = bool(style_dict.get("connect_bar_ends", True))
        self.dash_spacing = float(style_dict.get("dash_spacing", 2.5))
        self.x_label = style_dict.get("x_label", None)
        self.y_label = style_dict.get("y_label", None)
        self.x_indices = bool(style_dict.get("x_indices", False))
        self.square = bool(style_dict.get("square", False))
        self.stagger_labels = bool(style_dict.get("stagger_labels", True))
        self.labelpoint = style_dict.get("labelpoint", None)
        self.excitation_arrows = bool(style_dict.get("excitation_arrows", True))
        self.transparent = bool(style_dict.get("transparent", False))
        self.bg_color = style_dict.get("bg_color", None)

        self.legend = self._sub_style("legend", style_dict.get("legend"), allowed=None)
        self.y1 = self._sub_style("y1", style_dict.get("y1"), _AXIS_KEYS)
        self.y2 = self._sub_style("y2", style_dict.get("y2"), _AXIS_KEYS)

        self.font_properties = self._get_font_properties(style_dict)
        font_family = self.font_properties.get_family()
        font_family = font_family[0] if font_family else "sans-serif"
        self.annotation_kwargs: dict[str, Any] = {
            "fontsize": style_dict.get("annotation_size", max(6, self.font_size)),
            "fontfamily": font_family,
            "fontweight": style_dict.get("annotation_weight", "semibold"),
            "fontstyle": style_dict.get("annotation_style", "italic"),
        }

        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self._legend_handles: list[Line2D] = []

    @staticmethod
    def _sub_style(name: str, value: Any, allowed: Optional[Sequence[str]]) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"'{name}' must be a dict of style overrides, got {type(value).__name__}")
        if allowed is not None:
            unknown = [key for key in value if key not in allowed]
            if unknown:
                raise TypeError(
                    f"Unknown {name} key(s): {', '.join(map(repr, unknown))}. Valid keys: {', '.join(sorted(allowed))}."
                )
        return dict(value)

    def _axis_style(self, overrides: Optional[dict[str, Any]] = None) -> SimpleNamespace:
        style = {key: getattr(self, key) for key in _SERIES_KEYS}
        style["label"] = self.y_label
        style.update(overrides or {})
        return SimpleNamespace(**style)

    @staticmethod
    def _line_kwargs(st: SimpleNamespace, label: str) -> dict[str, Any]:
        spec = st.linestyle
        if isinstance(spec, dict):
            spec = spec.get(label)
        base = {"dash_capstyle": "round"}
        if spec is None or spec in ("solid", "-"):
            return {**base, "linestyle": "solid", "dashes": (st.line_width, 0)}
        if spec in ("dashed", "--"):
            return {**base, "linestyle": "dashed", "dashes": (st.line_width, st.dash_spacing)}
        return {**base, "linestyle": spec}

    @staticmethod
    def _text_extents(fig: plt.Figure, texts: list[Any]) -> list[Any]:
        if not texts:
            return []
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        return [(text.get_bbox_patch() or text).get_window_extent(renderer) for text in texts]

    def _text_height_points(self, fig: plt.Figure, texts: list[Any]) -> float:
        extents = self._text_extents(fig, texts)
        if not extents:
            return 0.0
        return max(e.height for e in extents) * 72.0 / fig.dpi

    @staticmethod
    def _preferred_above(series_ys: list[float], i: int, neighbours: Sequence[float] = ()) -> bool:
        y = series_ys[i]
        at_an_end = len(series_ys) == 1 or i in (0, len(series_ys) - 1)
        if at_an_end:
            others = [other for other in neighbours if not np.isclose(other, y)]
            if others:
                if y > max(others):
                    return True
                if y < min(others):
                    return False
        if len(series_ys) == 1:
            return True
        if i == 0:
            return series_ys[1] < series_ys[0]
        if i == len(series_ys) - 1:
            return series_ys[-2] < series_ys[-1]
        curvature = series_ys[i + 1] - 2 * series_ys[i] + series_ys[i - 1]
        if curvature == 0:
            return series_ys[i] > series_ys[i - 1]
        return curvature < 0

    def _fit_axes_to_text(self, fig: plt.Figure, ax: plt.Axes, texts: list[Any], pad: float, passes: int = 4) -> None:
        if not texts:
            return
        fig.canvas.draw()
        for _ in range(passes):
            extents = self._text_extents(fig, texts)
            if not extents:
                return
            inverse = ax.transData.inverted()
            lows = [inverse.transform((e.x0, e.y0))[1] for e in extents]
            highs = [inverse.transform((e.x1, e.y1))[1] for e in extents]
            y_min, y_max = ax.get_ylim()
            effective_pad = max(pad, 1.5)
            need_expand_top = max(highs) + effective_pad > y_max
            need_expand_bottom = min(lows) - effective_pad < y_min
            new_ymin = min(y_min, min(lows) - effective_pad) if need_expand_bottom else y_min
            new_ymax = max(y_max, max(highs) + effective_pad) if need_expand_top else y_max

            if (new_ymin, new_ymax) == (y_min, y_max):
                return
            ax.set_ylim(new_ymin, new_ymax)
            fig.canvas.draw()

    def _legend_font(self, opts: dict[str, Any]) -> FontProperties:
        font = self.font_properties.copy()
        font.set_size(self.font_size)
        prop = opts.pop("prop", None)
        if isinstance(prop, FontProperties):
            font = prop.copy()
        elif isinstance(prop, dict):
            for key, value in prop.items():
                setter = getattr(font, f"set_{key}", None)
                if setter is not None:
                    setter(value)
        if "fontsize" in opts:
            font.set_size(opts.pop("fontsize"))
        return font

    @staticmethod
    def _x_positions(indices: list[float], x_values: Optional[list[float]]) -> list[float]:
        if x_values is None:
            return indices
        return list(np.interp(indices, np.arange(len(x_values)), x_values))

    @staticmethod
    def _validate_x(x: Any, n_points: int) -> Optional[list[float]]:
        if x is None:
            return None
        try:
            x_vals = [float(v) for v in x]
        except (TypeError, ValueError) as e:
            raise TypeError(f"'x' must be a sequence of numbers: {e}") from e
        if len(x_vals) < n_points:
            raise ValueError(f"'x' has {len(x_vals)} values but the longest series has {n_points} points.")
        return x_vals

    def _series_curve(self, points: list[tuple[float, float]], st: SimpleNamespace) -> Optional[np.ndarray]:
        points = [(float(x), float(y)) for x, y in points]
        if not points:
            return None

        # Partition points into contiguous sub-paths separated whenever two consecutive points
        # share the same X-coordinate (superimposed points, e.g. photochemical vertical excitation)
        sub_paths: list[list[tuple[float, float]]] = []
        curr_sub: list[tuple[float, float]] = [points[0]]
        for pt in points[1:]:
            if abs(pt[0] - curr_sub[-1][0]) < 1e-4:
                sub_paths.append(curr_sub)
                curr_sub = [pt]
            else:
                curr_sub.append(pt)
        if curr_sub:
            sub_paths.append(curr_sub)

        all_segments = []
        for sub in sub_paths:
            if len(sub) < 2:
                continue
            sub_pts = sub
            if st.point_type == "bar" and st.connect_bar_ends:
                half = st.bar_length / 2
                last = len(sub_pts) - 1
                adjusted = []
                for j, (x_pt, y_pt) in enumerate(sub_pts):
                    if j == 0:
                        adjusted.append((x_pt + half, y_pt))
                    elif j == last:
                        adjusted.append((x_pt - half, y_pt))
                    else:
                        adjusted.append((x_pt - half, y_pt))
                        adjusted.append((x_pt + half, y_pt))
                sub_pts = adjusted

            for (x0, y0), (x1, y1) in pairwise(sub_pts):
                p0 = np.array([x0, y0])
                p1 = np.array([x0 + st.curviness * (x1 - x0), y0])
                p2 = np.array([x1 - st.curviness * (x1 - x0), y1])
                p3 = np.array([x1, y1])
                all_segments.append(cubic_bezier_points(p0, p1, p2, p3))
            all_segments.append(np.array([[np.nan, np.nan]]))

        if not all_segments:
            return None
        if np.isnan(all_segments[-1][0, 0]):
            all_segments.pop()
        return np.vstack(all_segments) if all_segments else None

    def _draw_points(self, ax: plt.Axes, xs: list[float], ys: list[float], st: SimpleNamespace, color: Any) -> None:
        for x_pt, y_pt in zip(xs, ys, strict=True):
            if np.isnan(y_pt):
                continue
            if st.point_type == "bar":
                ax.plot(
                    [x_pt - st.bar_length / 2, x_pt + st.bar_length / 2],
                    [y_pt, y_pt],
                    color="black",
                    lw=st.bar_width,
                )
            elif st.point_type in ("dot", "."):
                ax.plot(x_pt, y_pt, "o", markersize=st.marker_size, color=color)
            elif st.point_type in ("hollow", "o"):
                ax.plot(
                    x_pt,
                    y_pt,
                    marker="o",
                    markersize=st.marker_size,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=st.line_width,
                )

    def _get_font_properties(self, font_dict: dict[str, Any]) -> FontProperties:
        requested_family = font_dict.get("font_family", "sans-serif")
        available_fonts = {f.name for f in fontManager.ttflist}
        if requested_family not in available_fonts:
            requested_family = "DejaVu Sans"

        return FontProperties(
            family=requested_family,
            weight=font_dict.get("font_weight", "bold"),
            style=font_dict.get("font_style", "normal"),
            size=font_dict.get("font_size", 12),
        )

    def _resolve_colors(self, setting: Any, num_colors: int) -> list[Any]:
        try:
            if isinstance(setting, str):
                try:
                    return list(sns.color_palette(setting, num_colors))
                except ValueError:
                    cmap = plt.get_cmap(setting)
                    return [cmap(i / max(num_colors, 1)) for i in range(num_colors)]
            elif isinstance(setting, (list, tuple)):
                if len(setting) < num_colors:
                    repeats = (num_colors + len(setting) - 1) // len(setting)
                    return list((list(setting) * repeats)[:num_colors])
                return list(setting[:num_colors])
            elif callable(setting):
                return [setting(i / max(num_colors, 1)) for i in range(num_colors)]
        except Exception:
            pass
        fallback = plt.get_cmap("viridis")
        return [fallback(i / max(num_colors, 1)) for i in range(num_colors)]

    def _validate_energy_list(self, lst: Any, label: Optional[str] = None) -> list[Optional[float]]:
        if not isinstance(lst, list):
            raise TypeError(f"Energy profile '{label}' must be a list." if label else "Energy profile must be a list.")
        valid_list: list[Optional[float]] = []
        for i, val in enumerate(lst):
            if val is None:
                valid_list.append(None)
            elif isinstance(val, (int, float)):
                valid_list.append(float(val))
            elif isinstance(val, str):
                try:
                    valid_list.append(float(val))
                except ValueError as e:
                    raise ValueError(f"Could not convert string to float at index {i} in '{label}': '{val}'") from e
            else:
                raise TypeError(f"Invalid energy value at index {i} in '{label}': {val} (type {type(val)})")
        return valid_list

    def _draw_secondary(
        self, ax: plt.Axes, secondary: dict[str, list[Any]], exclude_from_legend: list[str], x: Optional[list[float]] = None
    ) -> tuple[plt.Axes, list[Line2D]]:
        st = self._axis_style({"colors": "plasma", "linestyle": "--", **self.y2})
        ax2 = ax.twinx()
        labels = list(secondary.keys())
        colors = self._resolve_colors(st.colors, len(labels))
        light = [desaturate_colour(c, st.desaturate_factor) for c in colors] if st.desaturate else colors
        handles = []

        for i, label in enumerate(labels):
            values = [np.nan if v is None else float(v) for v in secondary[label]]
            xs, ys = generate_coordinates(values)
            if len(xs) < 2:
                continue
            xs = self._x_positions(xs, x)
            line = self._line_kwargs(st, label)
            curve = self._series_curve(list(zip(xs, ys, strict=True)), st)
            if curve is not None:
                ax2.plot(curve[:, 0], curve[:, 1], color=light[i], linewidth=st.line_width, zorder=1, **line)
            self._draw_points(ax2, xs, ys, st, colors[i])
            sec_marker_kw: dict[str, Any] = {}
            if st.point_type in ("dot", "."):
                sec_marker_kw = {
                    "marker": "o",
                    "markersize": st.marker_size,
                    "markerfacecolor": colors[i],
                    "markeredgecolor": colors[i],
                }
            elif st.point_type in ("hollow", "o"):
                sec_marker_kw = {
                    "marker": "o",
                    "markersize": st.marker_size,
                    "markerfacecolor": "white",
                    "markeredgecolor": colors[i],
                    "markeredgewidth": st.line_width,
                }
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=light[i],
                    linewidth=st.line_width,
                    linestyle=line["linestyle"],
                    label=label,
                    dash_capstyle="round",
                    **sec_marker_kw,
                )
            )

        for spine in ax2.spines.values():
            spine.set_visible(False)
        if self.axes in ("y", "both", "box"):
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_linewidth(self.axis_linewidth)
            ax2.tick_params(axis="y", which="both", right=True, labelright=True)
            ax2.tick_params(axis="y", labelsize=self.font_size, width=self.axis_linewidth)
            if st.label:
                ax2.set_ylabel(st.label, fontproperties=self.font_properties)
        else:
            ax2.tick_params(axis="y", which="both", right=False, labelright=False)
            ax2.set_ylabel("")

        ax2.margins(y=0.15)
        return ax2, [h for h in handles if h.get_label() not in exclude_from_legend]

    def plot(
        self,
        energy_data: Union[dict[str, list[Any]], list[list[Any]], list[Any]],
        filename: Optional[str] = None,
        annotations: Optional[dict[str, Tuple[float, float]]] = None,
        point_labels: Optional[Union[dict[str, list[Optional[str]]], list[Any]]] = None,
        file_format: str = "png",
        dpi: int = 600,
        include_keys: Optional[list[str]] = None,
        exclude_from_legend: Optional[list[str]] = None,
        secondary: Optional[dict[str, list[Any]]] = None,
        x: Optional[Sequence[float]] = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Draw a reaction energy profile and return (fig, ax)."""
        exclude_from_legend = [] if exclude_from_legend is None else exclude_from_legend

        processed_dict: dict[str, list[Optional[float]]] = {}
        if isinstance(energy_data, dict):
            for label, values in energy_data.items():
                processed_dict[str(label)] = self._validate_energy_list(values, label=str(label))
        elif isinstance(energy_data, list):
            if all(isinstance(sublist, list) for sublist in energy_data):
                for i, sublist in enumerate(energy_data):
                    processed_dict[f"Path {i + 1}"] = self._validate_energy_list(sublist, label=f"Path {i + 1}")
            else:
                processed_dict["Path 1"] = self._validate_energy_list(energy_data)
        else:
            raise TypeError("Data input must be a dict, list of lists, or a single list.")

        if annotations is not None and isinstance(annotations, dict):
            clean_annotations = {}
            for label, val in annotations.items():
                if isinstance(val, (tuple, list)) and len(val) == 2:
                    clean_annotations[str(label)] = (float(val[0]), float(val[1]))
            annotations = clean_annotations
        self.annotations = annotations

        if include_keys is not None:
            processed_dict = {k: v for k, v in processed_dict.items() if k in include_keys}

        labels = list(processed_dict.keys())
        energy_sets = [[e if e is not None else np.nan for e in processed_dict[k]] for k in labels]

        n_points = max((len(v) for v in processed_dict.values()), default=0)
        if secondary:
            n_points = max(n_points, *(len(v) for v in secondary.values()))
        x_values = self._validate_x(x, n_points)

        processed_point_labels: Optional[dict[str, list[Optional[str]]]] = None
        if point_labels is not None:
            if isinstance(point_labels, dict):
                processed_point_labels = {}
                for lbl, vals in point_labels.items():
                    if lbl in processed_dict:
                        processed_point_labels[lbl] = [str(v) if v is not None else None for v in vals]
            elif isinstance(point_labels, list):
                processed_point_labels = {}
                if all(isinstance(sublist, list) for sublist in point_labels):
                    for i, sublist in enumerate(cast("list[list[Any]]", point_labels)):
                        lbl = labels[i] if i < len(labels) else f"Path {i+1}"
                        processed_point_labels[lbl] = [str(v) if v is not None else None for v in sublist]
                else:
                    lbl = labels[0] if labels else "Path 1"
                    processed_point_labels[lbl] = [str(v) if v is not None else None for v in point_labels]

        coords = [generate_coordinates(e) for e in energy_sets]
        coords = [(self._x_positions(xs, x_values), ys) for xs, ys in coords]
        all_energies = [e for _, ys in coords for e in ys if not np.isnan(e)]
        if not all_energies:
            raise ValueError("No energies to plot: every series is empty or all None.")

        energy_span = max(all_energies) - min(all_energies)
        if energy_span == 0:
            energy_span = 1.0
        buffer_space = self.buffer_factor * energy_span

        primary = self._axis_style(self.y1)
        colors = self._resolve_colors(primary.colors, len(energy_sets))
        colors = colors[::-1]
        light_colors = [desaturate_colour(c, primary.desaturate_factor) for c in colors] if primary.desaturate else colors

        fig, ax = plt.subplots(figsize=self.figsize)
        self.fig, self.ax = fig, ax

        if self.transparent:
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
        elif self.bg_color:
            fig.patch.set_facecolor(self.bg_color)
            ax.set_facecolor(self.bg_color)

        if self.labels:
            ax.margins(x=0.08, y=0.1)

        # Draw curves
        for i, (xs, y) in enumerate(reversed(coords)):
            valid_points = [(xi, yi) for xi, yi in zip(xs, y, strict=True) if not np.isnan(yi)]
            if len(valid_points) < 2:
                continue
            label = labels[len(coords) - 1 - i]
            line = self._line_kwargs(primary, label)
            all_points = self._series_curve(valid_points, primary)
            if all_points is None:
                continue
            ax.plot(all_points[:, 0], all_points[:, 1], color=light_colors[i], linewidth=primary.line_width, **line)
            if label not in exclude_from_legend:
                marker_kw: dict[str, Any] = {}
                if primary.point_type in ("dot", "."):
                    marker_kw = {
                        "marker": "o",
                        "markersize": primary.marker_size,
                        "markerfacecolor": colors[i],
                        "markeredgecolor": colors[i],
                    }
                elif primary.point_type in ("hollow", "o"):
                    marker_kw = {
                        "marker": "o",
                        "markersize": primary.marker_size,
                        "markerfacecolor": "white",
                        "markeredgecolor": colors[i],
                        "markeredgewidth": primary.line_width,
                    }
                legend_line = Line2D(
                    [0],
                    [0],
                    color=light_colors[i],
                    linewidth=primary.line_width,
                    linestyle=line["linestyle"],
                    label=label,
                    dash_capstyle="round",
                    **marker_kw,
                )
                ax.add_line(legend_line)

        # Draw points
        for i, (xs, y) in enumerate(reversed(coords)):
            self._draw_points(ax, xs, y, primary, colors[i])

        # Track vertical transition endpoints to place labels correctly (above/below arrow)
        arrow_up_targets: set[tuple[float, float]] = set()
        arrow_up_origins: set[tuple[float, float]] = set()
        arrow_down_targets: set[tuple[float, float]] = set()
        arrow_down_origins: set[tuple[float, float]] = set()

        for i, (xs, y) in enumerate(reversed(coords)):
            valid = [(xi, yi) for xi, yi in zip(xs, y, strict=True) if not np.isnan(yi)]
            for (x0, y0), (x1, y1) in pairwise(valid):
                if abs(x1 - x0) < 1e-4 and abs(y1 - y0) > 1e-4:
                    if y1 > y0:
                        arrow_up_targets.add((round(x1, 3), round(y1, 3)))
                        arrow_up_origins.add((round(x0, 3), round(y0, 3)))
                    else:
                        arrow_down_targets.add((round(x1, 3), round(y1, 3)))
                        arrow_down_origins.add((round(x0, 3), round(y0, 3)))

        # Draw excitation arrows for vertical transitions (photochemical excitation)
        if self.excitation_arrows:
            for i, (xs, y) in enumerate(reversed(coords)):
                valid = [(xi, yi) for xi, yi in zip(xs, y, strict=True) if not np.isnan(yi)]
                for (x0, y0), (x1, y1) in pairwise(valid):
                    if abs(x1 - x0) < 1e-4 and abs(y1 - y0) > 1e-4:
                        y_span = max(all_energies) - min(all_energies)
                        inset = 0.035 * y_span if primary.point_type in ("hollow", "dot") else 0.015 * y_span
                        if y1 > y0:
                            start_xy = (x0, y0 + inset)
                            end_xy = (x1, y1 - inset)
                        else:
                            start_xy = (x0, y0 - inset)
                            end_xy = (x1, y1 + inset)
                        ax.annotate(
                            "",
                            xy=end_xy,
                            xytext=start_xy,
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color=colors[i],
                                lw=primary.line_width,
                                mutation_scale=16,
                            ),
                        )

        # Draw labels
        label_texts = []
        should_draw_numeric = self.labels or (self.labelpoint is not None)
        if should_draw_numeric or processed_point_labels:
            labeled = []
            labeled_set = set()
            series_points = [[(px, py) for px, py in zip(xs_, ys_, strict=True) if not np.isnan(py)] for xs_, ys_ in coords]
            at_x: dict[float, list[float]] = {}
            for points in series_points:
                for px, py in points:
                    at_x.setdefault(round(px, 3), []).append(py)

            point_label_map: dict[tuple[float, float], str] = {}
            if processed_point_labels is not None:
                for i, (_x, y) in enumerate(reversed(coords)):
                    profile_label = labels[len(coords) - 1 - i]
                    if profile_label not in processed_point_labels:
                        continue
                    point_label_list = processed_point_labels[profile_label]
                    energy_profile = processed_dict[profile_label]
                    valid_indices = [idx for idx, e in enumerate(energy_profile) if e is not None]
                    valid_coords = [(px, py) for px, py in zip(_x, y) if not np.isnan(py)]
                    for pt_idx, (px, py) in enumerate(valid_coords):
                        if pt_idx < len(valid_indices):
                            orig_idx = valid_indices[pt_idx]
                            if orig_idx < len(point_label_list) and point_label_list[orig_idx] is not None:
                                raw_lbl = point_label_list[orig_idx]
                                text_val = str(raw_lbl).strip() if raw_lbl is not None else ""
                                if not text_val or text_val.lower() in ("none", "nan", "null"):
                                    continue
                                key = (round(px, 3), round(py, 3))
                                if key in point_label_map:
                                    existing = point_label_map[key]
                                    if existing != text_val and text_val not in existing.split("/"):
                                        point_label_map[key] = f"{existing}/{text_val}"
                                else:
                                    point_label_map[key] = text_val

            if should_draw_numeric:
                # Group points by X coordinate
                points_by_x: dict[float, list[dict[str, Any]]] = {}
                for rev_i, (xs_, ys_) in enumerate(reversed(coords)):
                    col = colors[rev_i]
                    series_lbl = labels[len(coords) - 1 - rev_i]
                    energy_profile = processed_dict.get(series_lbl, [])
                    valid_indices = [idx for idx, e in enumerate(energy_profile) if e is not None]
                    valid_pts = []
                    for valid_idx, (xi, yi) in enumerate([(x, y) for x, y in zip(xs_, ys_, strict=True) if not np.isnan(y)]):
                        orig_idx = valid_indices[valid_idx] if valid_idx < len(valid_indices) else valid_idx
                        valid_pts.append((xi, yi, valid_idx, orig_idx))
                    series_ys = [yi for _, yi, _, _ in valid_pts]
                    for xi, yi, step_k, orig_k in valid_pts:
                        qualifies = False
                        if self.labelpoint is not None:
                            target_k = orig_k if len(energy_profile) > len(valid_pts) else step_k
                            if target_k in self.labelpoint:
                                qualifies = True
                        elif self.labels:
                            qualifies = True

                        if qualifies:
                            points_by_x.setdefault(round(xi, 3), []).append({
                                "x": xi,
                                "y": yi,
                                "color": col,
                                "step_idx": step_k,
                                "series_label": series_lbl,
                                "series_ys": series_ys,
                            })

                all_x_coords = [xi for xs_, _ in coords for xi in xs_]
                min_x_data = min(all_x_coords) if all_x_coords else 0.0
                max_x_data = max(all_x_coords) if all_x_coords else 1.0
                span_x = max(max_x_data - min_x_data, 1e-3)

                all_y_coords = [yi for _, ys_ in coords for yi in ys_ if not np.isnan(yi)]
                min_y_data = min(all_y_coords) if all_y_coords else 0.0
                max_y_data = max(all_y_coords) if all_y_coords else 1.0
                span_y = max(max_y_data - min_y_data, 1e-3)
                collision_thresh_y = 0.06 * span_y

                for x_key in sorted(points_by_x.keys()):
                    cluster = points_by_x[x_key]
                    unique_cluster = []
                    seen_ys = set()
                    for pt in cluster:
                        y_rounded = round(pt["y"], 3)
                        if y_rounded not in seen_ys:
                            seen_ys.add(y_rounded)
                            unique_cluster.append(pt)
                    cluster = unique_cluster
                    if len(cluster) == 1:
                        pt = cluster[0]
                        pt_key = (round(pt["x"], 3), round(pt["y"], 3))
                        if pt_key in arrow_up_targets or pt_key in arrow_down_origins:
                            above = True
                        elif pt_key in arrow_up_origins or pt_key in arrow_down_targets:
                            above = False
                        else:
                            above = self._preferred_above(pt["series_ys"], pt["step_idx"], at_x.get(round(pt["x"], 3), []))
                        preferred_y = pt["y"] + buffer_space if above else pt["y"] - buffer_space
                        label_text = f"{pt['y']:.{self.sig_figs}f}".replace("-", "−")
                        txt_color = pt["color"] if len(labels) > 1 and self.point_label_color == "black" else self.point_label_color
                        value_text = ax.text(
                            pt["x"],
                            preferred_y,
                            label_text,
                            ha="center",
                            va="center",
                            fontproperties=self.font_properties,
                            fontweight="bold" if len(labels) > 1 else "normal",
                            color=txt_color,
                        )
                        label_texts.append(value_text)
                        labeled.append((pt["x"], preferred_y, above, pt["y"]))
                    else:
                        sorted_cluster = sorted(cluster, key=lambda p: p["y"], reverse=True)
                        has_collision = any(
                            (sorted_cluster[i]["y"] - sorted_cluster[i + 1]["y"]) < collision_thresh_y
                            for i in range(len(sorted_cluster) - 1)
                        )
                        is_vert_arrow = any(
                            (round(p["x"], 3), round(p["y"], 3)) in arrow_up_targets
                            or (round(p["x"], 3), round(p["y"], 3)) in arrow_up_origins
                            or (round(p["x"], 3), round(p["y"], 3)) in arrow_down_targets
                            or (round(p["x"], 3), round(p["y"], 3)) in arrow_down_origins
                            for p in sorted_cluster
                        )
                        if is_vert_arrow or len(sorted_cluster) == 2:
                            use_stagger = False
                        else:
                            use_stagger = self.stagger_labels and has_collision and len(sorted_cluster) > 2

                        if not use_stagger:
                            if len(sorted_cluster) == 2:
                                # Smart 2-point vertical split: higher above, lower below
                                pt_high = sorted_cluster[0]
                                pt_low = sorted_cluster[1]
                                for pt, above in [(pt_high, True), (pt_low, False)]:
                                    preferred_y = pt["y"] + buffer_space if above else pt["y"] - buffer_space
                                    label_text = f"{pt['y']:.{self.sig_figs}f}".replace("-", "−")
                                    txt_color = pt["color"] if len(labels) > 1 and self.point_label_color == "black" else self.point_label_color
                                    value_text = ax.text(
                                        pt["x"],
                                        preferred_y,
                                        label_text,
                                        ha="center",
                                        va="center",
                                        fontproperties=self.font_properties,
                                        fontweight="bold" if len(labels) > 1 else "normal",
                                        color=txt_color,
                                    )
                                    label_texts.append(value_text)
                                    labeled.append((pt["x"], preferred_y, above, pt["y"]))
                            else:
                                for pt in cluster:
                                    pt_key = (round(pt["x"], 3), round(pt["y"], 3))
                                    if pt_key in arrow_up_targets or pt_key in arrow_down_origins:
                                        above = True
                                    elif pt_key in arrow_up_origins or pt_key in arrow_down_targets:
                                        above = False
                                    else:
                                        above = self._preferred_above(pt["series_ys"], pt["step_idx"], at_x.get(round(pt["x"], 3), []))
                                    preferred_y = pt["y"] + buffer_space if above else pt["y"] - buffer_space
                                    label_text = f"{pt['y']:.{self.sig_figs}f}".replace("-", "−")
                                    txt_color = pt["color"] if len(labels) > 1 and self.point_label_color == "black" else self.point_label_color
                                    value_text = ax.text(
                                        pt["x"],
                                        preferred_y,
                                        label_text,
                                        ha="center",
                                        va="center",
                                        fontproperties=self.font_properties,
                                        fontweight="bold" if len(labels) > 1 else "normal",
                                        color=txt_color,
                                    )
                                    label_texts.append(value_text)
                                    labeled.append((pt["x"], preferred_y, above, pt["y"]))
                        else:
                            # Automatic Staggering / Decluttering
                            k = len(sorted_cluster)
                            x_pt = sorted_cluster[0]["x"]

                            can_go_left = (x_pt - min_x_data) >= 0.15 * span_x
                            can_go_right = (max_x_data - x_pt) >= 0.15 * span_x

                            for j, pt in enumerate(sorted_cluster):
                                label_text = f"{pt['y']:.{self.sig_figs}f}".replace("-", "−")
                                col = pt["color"]

                                if can_go_left and can_go_right:
                                    if j == 0:
                                        dx, dy, ha = 22, 20, "left"
                                    elif j == 1:
                                        dx, dy, ha = -26, -18, "right"
                                    elif j == 2:
                                        dx, dy, ha = 38, 8, "left"
                                    elif j == 3:
                                        dx, dy, ha = -16, -18, "right"
                                    else:
                                        side = 1 if j % 2 == 0 else -1
                                        dx = side * (24 + (j // 2) * 12)
                                        dy = -10 - 8 * (j // 2)
                                        ha = "left" if side > 0 else "right"
                                elif can_go_right:
                                    dx = 24
                                    ha = "left"
                                    dy = int((k - 1 - 2 * j) * 8)
                                else:
                                    dx = -26
                                    ha = "right"
                                    dy = int((k - 1 - 2 * j) * 8)

                                txt_obj = ax.annotate(
                                    label_text,
                                    xy=(pt["x"], pt["y"]),
                                    xytext=(dx, dy),
                                    textcoords="offset points",
                                    ha=ha,
                                    va="center",
                                    fontproperties=self.font_properties,
                                    fontsize=max(5, self.font_size - 1),
                                    fontweight="bold",
                                    color=col,
                                    bbox=dict(
                                        boxstyle="round,pad=0.25",
                                        facecolor="white",
                                        edgecolor=col,
                                        linewidth=0.75,
                                        alpha=0.95,
                                    ),
                                    arrowprops=dict(
                                        arrowstyle="-",
                                        color=col,
                                        lw=0.8,
                                        shrinkA=3,
                                        shrinkB=3,
                                    ),
                                )
                                label_texts.append(txt_obj)
                                labeled.append((pt["x"], pt["y"], dy > 0, pt["y"]))

            if processed_point_labels is not None:
                offset = self._text_height_points(fig, label_texts) / 2 + _LABEL_GAP_POINTS
                annotated_keys = set()
                for x_pt, y, above, energy in labeled:
                    point_label = next(
                        (text for (px, py), text in point_label_map.items() if abs(px - x_pt) < 1e-3 and abs(py - energy) < 1e-3),
                        None,
                    )
                    if point_label and point_label.strip() and point_label.strip().lower() not in ("none", "nan", "null"):
                        annotated_keys.add((round(x_pt, 3), round(energy, 3)))
                        pt_key = (round(x_pt, 3), round(energy, 3))
                        eff_above = above
                        if pt_key in arrow_up_targets or pt_key in arrow_down_origins:
                            eff_above = True
                        elif pt_key in arrow_up_origins or pt_key in arrow_down_targets:
                            eff_above = False
                        label_texts.append(
                            ax.annotate(
                                point_label,
                                xy=(x_pt, y),
                                xytext=(0, offset if eff_above else -offset),
                                textcoords="offset points",
                                ha="center",
                                va="bottom" if eff_above else "top",
                                fontproperties=self.font_properties,
                                fontsize=self.font_size,
                                color=self.point_label_color,
                            )
                        )
                # Annotate any remaining points in point_label_map that didn't have numeric labels
                remaining_by_x: dict[float, list[tuple[float, float, str]]] = {}
                for (px, py), text in point_label_map.items():
                    if (round(px, 3), round(py, 3)) not in annotated_keys and text and text.strip() and text.strip().lower() not in ("none", "nan", "null"):
                        remaining_by_x.setdefault(round(px, 3), []).append((px, py, text))

                all_y_vals = [yi for _, ys_ in coords for yi in ys_ if not np.isnan(yi)]
                span_y_rem = max((max(all_y_vals) - min(all_y_vals)), 1e-3) if all_y_vals else 1.0
                thresh_y_rem = 0.06 * span_y_rem

                for r_x, pts in remaining_by_x.items():
                    if len(pts) == 1:
                        px, py, text = pts[0]
                        pt_key = (round(px, 3), round(py, 3))
                        is_above = False
                        if pt_key in arrow_up_targets or pt_key in arrow_down_origins:
                            is_above = True
                        elif pt_key in arrow_up_origins or pt_key in arrow_down_targets:
                            is_above = False
                        label_texts.append(
                            ax.annotate(
                                text,
                                xy=(px, py),
                                xytext=(0, offset if is_above else -offset),
                                textcoords="offset points",
                                ha="center",
                                va="bottom" if is_above else "top",
                                fontproperties=self.font_properties,
                                fontsize=self.font_size,
                                color=self.point_label_color,
                            )
                        )
                    else:
                        pts_sorted = sorted(pts, key=lambda p: p[1], reverse=True)
                        for j, (px, py, text) in enumerate(pts_sorted):
                            pt_key = (round(px, 3), round(py, 3))
                            if pt_key in arrow_up_targets or pt_key in arrow_down_origins:
                                is_above = True
                            elif pt_key in arrow_up_origins or pt_key in arrow_down_targets:
                                is_above = False
                            elif j == 0:
                                is_above = True
                            elif j == len(pts_sorted) - 1:
                                is_above = False
                            else:
                                is_above = False
                            label_texts.append(
                                ax.annotate(
                                    text,
                                    xy=(px, py),
                                    xytext=(0, offset if is_above else -offset),
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom" if is_above else "top",
                                    fontproperties=self.font_properties,
                                    fontsize=self.font_size,
                                    color=self.point_label_color,
                                )
                            )

            self._fit_axes_to_text(fig, ax, label_texts, buffer_space)

        # Secondary axis
        secondary_handles: list[Line2D] = []
        if secondary:
            _, secondary_handles = self._draw_secondary(ax, secondary, exclude_from_legend, x_values)

        # Legend
        if self.show_legend:
            handles, labels_ = [], []
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                if label and not label.startswith("_unlabeled_"):
                    handles.append(handle)
                    labels_.append(label)
            handles, labels_ = handles[::-1], labels_[::-1]
            handles += secondary_handles
            labels_ += [h.get_label() for h in secondary_handles]
            if handles:
                opts = dict(self.legend)
                outside = bool(opts.pop("outside", False))
                anchor = opts.pop("anchor", 1.02)
                font = self._legend_font(opts)
                if outside:
                    opts.setdefault("loc", "center left")
                    opts.setdefault("bbox_to_anchor", (anchor, 0.5))
                opts.setdefault("frameon", not outside)
                ax.legend(handles, labels_, prop=font, **opts)

        # Segment annotations
        annotation_texts = []
        if self.annotations:
            y_min, _ = ax.get_ylim()
            y_arrow = y_min - self.annotation_buffer * energy_span
            for label, span in self.annotations.items():
                x_start, x_end = self._x_positions(list(span), x_values)
                ax.annotate(
                    "",
                    xy=(x_end, y_arrow),
                    xytext=(x_start, y_arrow),
                    arrowprops={
                        "arrowstyle": "<->",
                        "color": self.arrow_color,
                        "lw": self.arrow_width,
                        "shrinkA": 0.5,
                        "shrinkB": 0.5,
                    },
                    annotation_clip=False,
                )
                x_center = (x_start + x_end) / 2.0
                if not self.annotation_below_arrow:
                    bbox_props = {
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "edgecolor": "none",
                    }
                    text = ax.annotate(
                        label,
                        xy=(x_center, y_arrow),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        color=self.annotation_color,
                        bbox=bbox_props,
                        **self.annotation_kwargs,
                    )
                else:
                    text = ax.text(
                        x_center,
                        y_arrow - 0.3,
                        label,
                        ha="center",
                        va="top",
                        color=self.annotation_color,
                        **self.annotation_kwargs,
                    )
                annotation_texts.append(text)

        # Labels & Spines
        if primary.label is not None:
            ax.set_ylabel(primary.label, fontproperties=self.font_properties)
        else:
            units = "kJ/mol" if self.units.lower() == "kj" else "kcal/mol"
            energy_term = self.energy.upper()
            ax.set_ylabel(f"Δ{energy_term} ({units})", fontproperties=self.font_properties)

        ax.set_xlabel(self.x_label if self.x_label is not None else "Reaction Coordinate", fontproperties=self.font_properties)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if not self.x_indices:
            ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False, labelright=False)

        if self.axes == "x":
            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_linewidth(self.axis_linewidth)
            ax.set_ylabel("")
            if self.x_indices:
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, width=self.line_width, length=5, labelsize=self.font_size)
        elif self.axes == "y":
            ax.spines["left"].set_visible(True)
            ax.spines["left"].set_linewidth(self.axis_linewidth)
            ax.set_xlabel("")
            ax.tick_params(axis="y", which="both", left=True, labelleft=True, width=self.line_width, length=5, labelsize=self.font_size)
        elif self.axes == "both":
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["bottom"].set_linewidth(self.axis_linewidth)
            ax.spines["left"].set_linewidth(self.axis_linewidth)
            if self.x_indices:
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, width=self.line_width, length=5, labelsize=self.font_size)
            ax.tick_params(axis="y", which="both", left=True, labelleft=True, width=self.line_width, length=5, labelsize=self.font_size)
        elif self.axes == "box":
            for spine_name in ["bottom", "top", "left", "right"]:
                ax.spines[spine_name].set_visible(True)
                ax.spines[spine_name].set_linewidth(self.axis_linewidth)
            if self.x_indices:
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True, width=self.line_width, length=5, labelsize=self.font_size)
            ax.tick_params(axis="y", which="both", left=True, labelleft=True, width=self.line_width, length=5, labelsize=self.font_size)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")

        if self.square:
            ax.set_box_aspect(1)

        fig.tight_layout()
        self._fit_axes_to_text(fig, ax, label_texts, buffer_space)
        self._fit_axes_to_text(fig, ax, annotation_texts, self.annotation_space * energy_span)

        if filename:
            fig.savefig(
                f"{filename}.{file_format}",
                format=file_format,
                dpi=dpi,
                bbox_inches="tight",
                transparent=self.transparent,
                facecolor=self.bg_color if not self.transparent else "none",
            )

        return fig, ax

    # ── Legacy compatibility methods (used by older FROG scripts) ───────────

    def plot_reaction_path(
        self,
        data: np.ndarray,
        x_positions: list[float],
        color: Optional[str] = None,
        label: Optional[str] = None,
        linestyle: str = '-',
        linewidth: float = 1.5,
        labelpoint: Optional[list[int]] = None,
        zero_idx: int = 0,
        ypad: float = 0.0,
        horizontalalignment: Optional[list[str]] = None,
        decimals: int = 1,
        index: int = 1,
        path_index: int = 0,
        total_paths: int = 1,
        path_label: Optional[str] = None,
    ) -> None:
        """Plot a single reaction path (legacy interface)."""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=self.figsize)

        if len(data) != len(x_positions):
            raise ValueError("data and x_positions must have same length")

        col = color or self._default_colors[0]
        light = _desaturate_color(col, self.desaturate_factor) if self.desaturate else col
        points = list(zip(x_positions, data))

        # Bezier curve
        verts = [[points[0][0], points[0][1]]]
        codes = [mpath.Path.MOVETO]
        for j in range(len(points) - 1):
            x0, y0 = points[j]
            x1, y1 = points[j + 1]
            dx = self.curviness * (x1 - x0)
            verts.append([x0 + dx, y0])
            verts.append([x1 - dx, y1])
            verts.append([x1, y1])
            codes += [mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4]

        path = mpath.Path(verts, codes)
        ls = 'dashed' if linestyle == 'dashed' else 'solid'
        patch = mpatches.PathPatch(
            path, fc='none', color=light, label=label,
            linestyle=ls, linewidth=linewidth,
            transform=self.ax.transData,
        )
        self.ax.add_patch(patch)

        if label:
            h = Line2D([], [], color=light, linewidth=linewidth, linestyle=ls, label=label)
            self._legend_handles.append(h)

        # Points
        for x, y in points:
            if self.point_type in ('dot', '.'):
                self.ax.plot(x, y, 'o', markersize=self.marker_size, color=col)
            elif self.point_type == 'bar':
                self.ax.plot([x - self.bar_length / 2, x + self.bar_length / 2], [y, y], color='black', lw=self.bar_width)
            else:
                self.ax.plot(x, y, marker='o', markerfacecolor='white', markeredgecolor=col, markeredgewidth=self.line_width)

        if labelpoint:
            ha_list = horizontalalignment or ['left'] * len(labelpoint)
            for idx, pt_idx in enumerate(labelpoint):
                if pt_idx >= len(data):
                    continue
                ha = ha_list[idx] if idx < len(ha_list) else 'left'
                ha = 'right' if pt_idx == len(data) - 1 else ha
                x_off = 0.2 if ha != 'right' else -0.2
                diff = np.round(data[pt_idx] - data[zero_idx], decimals)
                self.ax.text(
                    x_positions[pt_idx] + x_off, data[pt_idx] + ypad,
                    f"$\\Delta G^\\ddagger$ = {diff}\nkcal/mol",
                    verticalalignment='center', horizontalalignment=ha,
                    color=col,
                )

    def add_custom_tick_labels(
        self,
        x_positions: list[float],
        labels: list[str],
        color: Any,
        path_index: int,
        total_paths: int,
        path_label: Optional[str] = None,
    ) -> None:
        if not x_positions or not labels or self.ax is None:
            return
        y_min, y_max = self.ax.get_ylim()
        base_offset = 0.06
        offset = -base_offset * ((path_index + 1) if total_paths > 1 else 1)
        y_span = y_max - y_min
        y_pos = y_min + offset * y_span
        for pos, lbl in zip(x_positions, labels):
            self.ax.text(pos, y_pos, lbl, color=color, ha='center', va='top')
        if path_label:
            self.ax.text(min(x_positions) - 0.5, y_pos, path_label, color=color, ha='right', va='top')

    def add_section_arrows(
        self,
        x_positions: list[float],
        sections: list[str],
        sections_start: list[int],
        sections_end: list[int],
        colors: list[str],
        position: str,
        shift: float = 0.02,
    ) -> None:
        import itertools
        if not sections or not sections_start or not sections_end or self.ax is None:
            return
        fallback_colors = colors if colors else ["black"]
        y_min, y_max = self.ax.get_ylim()
        y_span = y_max - y_min
        arrow_y = y_min - shift * y_span if position != 'top' else y_max + shift * y_span
        for title, start, end, c in zip(sections, sections_start, sections_end, itertools.cycle(fallback_colors)):
            if start >= len(x_positions) or end >= len(x_positions) or start >= end:
                continue
            sx, ex = x_positions[start], x_positions[end]
            mx = (sx + ex) / 2
            self.ax.annotate(
                '', xy=(ex, arrow_y), xytext=(sx, arrow_y),
                arrowprops=dict(arrowstyle='<->', color=c, lw=self.arrow_width),
                annotation_clip=False,
            )
            self.ax.text(
                mx, arrow_y, title,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=c, linewidth=0.5),
                fontsize=max(6, self.font_size - 2), zorder=3,
            )


    def add_additional_points(self, points: list[str], colors: list[Any], styles: list[str]) -> None:
        if not points or self.ax is None:
            return
        if not styles:
            styles = ['-'] * len(points)
        for point_str, color, style in zip(points, colors, styles):
            try:
                coords = np.array(point_str.strip('()').split(','), dtype=float)
                if len(coords) == 2:
                    x, y = coords
                    self.ax.scatter(x, y, color=color, alpha=0.6, s=30)
                    self.ax.hlines(y, x - 0.1, x + 0.1, color=color, alpha=0.7, linestyle=style)
            except Exception:
                pass

    def finalize_plot(
        self,
        config: PlotConfig,
        show_legend: bool,
        show_standard_ticks: bool,
        x_positions: Optional[list[float]] = None,
        x_labels: Optional[list[str]] = None,
        show_sections: bool = False,
    ) -> None:
        if self.ax is None:
            return
        self.ax.set_title(config.title, fontsize=config.fontsize_title)
        self.ax.set_ylabel(r"$\Delta G$ (kcal/mol)", fontsize=config.fontsize_label)
        self.ax.tick_params(axis='both', labelsize=config.fontsize_ticks)
        if show_standard_ticks and x_positions and x_labels:
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(x_labels, fontsize=config.fontsize_ticks)
            self.ax.spines['bottom'].set_visible(True)
        else:
            self.ax.set_xticks([])
        if show_sections and x_positions and config.sections:
            self.add_section_arrows(
                x_positions, config.sections, config.sections_start, config.sections_end,
                config.colors_sections, config.position_sections,
            )
        if show_legend and self._legend_handles:
            labs = [h.get_label() for h in self._legend_handles]
            self.ax.legend(self._legend_handles, labs, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
        if config.y_limit is not None:
            self.ax.set_ylim(top=config.y_limit)
        if self.fig is not None:
            self.fig.tight_layout()

    def save_and_show(self, config: PlotConfig, file_content: str = '', show: bool = False,
                      dpi: Optional[int] = None, fig_format: str = 'png') -> None:
        """Legacy output: with `save: true`, write <directory>/Rxn_profile.<fig_format> and input.txt.

        Saving goes through frogchem.plotter.save_figure (shared --dpi, reported on stderr);
        the figure is shown only when `show` is True and is always closed afterwards.
        """
        if self.fig is None:
            return
        from frogchem.plotter import DEFAULT_DPI, save_figure
        if config.save:
            output_dir = Path(config.directory)
            output_dir.mkdir(exist_ok=True, parents=True)
            if file_content:
                from frogchem.printer import print_saved
                (output_dir / 'input.txt').write_text(file_content)
                print_saved(output_dir / 'input.txt', 'input copy')
            save_figure(self.fig, output_dir / f'Rxn_profile.{fig_format}',
                        dpi=dpi or DEFAULT_DPI, show=show)
        else:
            if show:
                plt.show()
            plt.close(self.fig)


# Aliases
ReactionPlotter = ReactionProfilePlotter

_PLOT_ARGS = frozenset(inspect.signature(ReactionProfilePlotter.plot).parameters) - {"self", "energy_data"}


def plot_reaction_profile(energy_data: Any, style: str = "default", **kwargs: Any) -> tuple[plt.Figure, plt.Axes]:
    """Plot a reaction profile in one call, returning (fig, ax)."""
    plot_kwargs = {k: v for k, v in kwargs.items() if k in _PLOT_ARGS}
    style_kwargs = {k: v for k, v in kwargs.items() if k not in _PLOT_ARGS}
    plotter = ReactionProfilePlotter(style=style, **style_kwargs)
    return plotter.plot(energy_data, **plot_kwargs)
