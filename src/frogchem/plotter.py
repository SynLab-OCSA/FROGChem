"""
Modulo centralizzato per i grafici (matplotlib), come frogchem.printer per il terminale.

Every FROG figure is drawn through this module: one style, one set of plot options
(the shared CLI flags --title, --colors, --linewidth, ...) and one way to save figures.

* `PlotOptions` holds the user-tunable properties; `opts.with_defaults(...)` merges the
  command's own defaults (CLI flag > command default > STYLE).
* `style(opts)` applies STYLE as a context, never touching matplotlib's global rcParams.
* Primitives (`line`, `scatter`, `bar`, `hist`, `heatmap`, `contour`) draw on a new figure or
  on `ax=`, then `finish()` applies title, labels, limits, legend and grid. Keyword arguments
  of a primitive are the call's matplotlib defaults: options set in `opts` win over them.

See docs/cli/conventions.md, section "Plots".
"""
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from frogchem.printer import print_saved

__all__ = [
    "DEFAULT_DPI", "DEFAULT_FIGSIZE", "FIG_FORMATS", "LINESTYLES", "STYLE",
    "PlotOptions", "rc_params", "style", "new_figure", "finish", "series_colors", "get_cmap",
    "energy_label", "line", "scatter", "label_points", "bar", "hist", "heatmap", "contour", "colorbar",
    "tight_layout", "save_figure", "resolve_path", "show_figures", "close",
]

DEFAULT_DPI = 300
DEFAULT_FIGSIZE = (8, 5)
FIG_FORMATS = ("png", "svg", "pdf", "eps")
#: CLI names of the line styles (--linestyle) → matplotlib codes
LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "figure.figsize": DEFAULT_FIGSIZE,
    "figure.dpi": 100,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
}


# ------------------------------------------------------------------ options
@dataclass(frozen=True)
class PlotOptions:
    """User-tunable plot properties. None means "not set": the command default or STYLE applies."""
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    colors: Optional[Tuple[str, ...]] = None
    cmap: Optional[str] = None
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None
    marker: Optional[str] = None
    alpha: Optional[float] = None
    legend: Optional[bool] = None
    legend_loc: Optional[str] = None
    grid: Optional[bool] = None
    figsize: Optional[Tuple[float, float]] = None
    font_size: Optional[float] = None

    def __post_init__(self):
        # ponytail: argparse gives lists (nargs); keep the dataclass hashable and comparable
        for name in ("xlim", "ylim", "colors", "figsize"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, tuple):
                object.__setattr__(self, name, tuple(value))

    @classmethod
    def from_args(cls, args) -> "PlotOptions":
        """Options given on the command line (attributes missing from `args` stay unset)."""
        return cls(**{f.name: getattr(args, f.name, None) for f in fields(cls)})

    def with_defaults(self, **defaults) -> "PlotOptions":
        """Fill the unset options with `defaults` (the command's own choices)."""
        unknown = set(defaults) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f"Unknown plot options: {', '.join(sorted(unknown))}")
        return replace(self, **{k: v for k, v in defaults.items() if getattr(self, k) is None})


def _opts(opts: Optional[PlotOptions]) -> PlotOptions:
    return opts if opts is not None else PlotOptions()


# -------------------------------------------------------------------- style
def rc_params(opts: Optional[PlotOptions] = None) -> dict:
    """STYLE plus the rcParams implied by `opts` (font size, line width/style, colors, grid, size)."""
    rc = dict(STYLE)
    if opts is None:
        return rc
    if opts.font_size is not None:
        fs = float(opts.font_size)
        rc.update({"font.size": fs, "axes.titlesize": fs + 2, "axes.labelsize": fs + 1,
                   "xtick.labelsize": fs - 1, "ytick.labelsize": fs - 1, "legend.fontsize": fs - 1})
    if opts.linewidth is not None:
        rc["lines.linewidth"] = opts.linewidth
    if opts.linestyle:
        rc["lines.linestyle"] = LINESTYLES.get(opts.linestyle, opts.linestyle)
    if opts.colors:
        from cycler import cycler
        rc["axes.prop_cycle"] = cycler(color=list(opts.colors))
    if opts.grid is not None:
        rc["axes.grid"] = opts.grid
    if opts.figsize:
        rc["figure.figsize"] = opts.figsize
    return rc


@contextmanager
def style(opts: Optional[PlotOptions] = None):
    """Apply the FROG style (and `opts`) inside the block; rcParams are restored afterwards."""
    import matplotlib as mpl
    with mpl.rc_context(rc_params(opts)):
        yield


def new_figure(nrows: int = 1, ncols: int = 1, figsize=None, opts: Optional[PlotOptions] = None, **kwargs):
    """Create a styled figure. Returns (fig, ax_or_axes).

    Size: opts.figsize, else `figsize`, else DEFAULT_FIGSIZE.
    """
    import matplotlib.pyplot as plt
    size = (opts.figsize if opts is not None and opts.figsize else None) or figsize or DEFAULT_FIGSIZE
    with style(opts):
        return plt.subplots(nrows, ncols, figsize=size, **kwargs)


def _axes(ax, opts: PlotOptions):
    if ax is None:
        return new_figure(opts=opts)
    return ax.figure, ax


def series_colors(opts: Optional[PlotOptions], n: int) -> list:
    """`n` colors: opts.colors repeated in order, else matplotlib's color cycle."""
    colors = list(_opts(opts).colors or [])
    if not colors:
        import matplotlib as mpl
        colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    return [colors[i % len(colors)] for i in range(n)]


def get_cmap(name: str):
    """Matplotlib colormap by name; seaborn colormaps (rocket, mako, ...) are registered on demand."""
    import matplotlib as mpl
    try:
        return mpl.colormaps[name]
    except KeyError:
        pass
    try:
        import seaborn  # noqa: F401  (registers its colormaps in matplotlib)
    except ImportError:
        pass
    try:
        return mpl.colormaps[name]
    except KeyError:
        raise ValueError(f"Unknown colormap {name!r}") from None


def energy_label(quantity: str = "Energy", unit: str = "kcal/mol", delta: bool = False) -> str:
    """Axis label for an energy: energy_label('G', delta=True) → 'ΔG (kcal/mol)'."""
    return f"{'Δ' if delta else ''}{quantity} ({unit})"


# ------------------------------------------------------------------- finish
def finish(ax, opts: Optional[PlotOptions] = None, handles=None, **legend_kwargs):
    """Apply title, axis labels, limits, legend and grid from `opts` to `ax`.

    Legend: opts.legend True/False forces it; unset shows it when there are labelled artists
    (or explicit `handles`). Limits keep an inverted axis inverted (NMR ppm axes).
    """
    opts = _opts(opts)
    if opts.title is not None:
        ax.set_title(opts.title)
    if opts.xlabel is not None:
        ax.set_xlabel(opts.xlabel)
    if opts.ylabel is not None:
        ax.set_ylabel(opts.ylabel)
    if opts.xlim:
        lo, hi = sorted(opts.xlim)
        ax.set_xlim((hi, lo) if ax.xaxis_inverted() else (lo, hi))
    if opts.ylim:
        lo, hi = sorted(opts.ylim)
        ax.set_ylim((hi, lo) if ax.yaxis_inverted() else (lo, hi))
    if opts.grid is not None:
        ax.grid(opts.grid)

    if handles is None:
        handles = ax.get_legend_handles_labels()[0]
    show_legend = opts.legend if opts.legend is not None else bool(handles)
    if show_legend and handles:
        legend_kwargs.setdefault("loc", opts.legend_loc or "best")
        ax.legend(handles=handles, **legend_kwargs)
    elif not show_legend and ax.get_legend() is not None:
        ax.get_legend().remove()
    return ax


# --------------------------------------------------------------- primitives
_ALIASES = {"c": "color", "lw": "linewidth", "ls": "linestyle", "ec": "edgecolor", "fc": "facecolor"}


def _kwargs(opts: PlotOptions, defaults: Mapping[str, Any], color=None, lines: bool = True,
            marker: bool = True) -> dict:
    """Call defaults overridden by the options set in `opts`."""
    kw = {_ALIASES.get(k, k): v for k, v in defaults.items()}
    if color is not None:
        kw["color"] = color
    if lines and opts.linewidth is not None:
        kw["linewidth"] = opts.linewidth
    if lines and opts.linestyle:
        kw["linestyle"] = LINESTYLES.get(opts.linestyle, opts.linestyle)
    if marker and opts.marker:
        kw["marker"] = opts.marker
    if opts.alpha is not None:
        kw["alpha"] = opts.alpha
    return kw


def _is_multi(values) -> bool:
    """True for several series: a 2-D array, or a list/tuple of arrays (pandas Series count as one)."""
    if hasattr(values, "ndim"):
        return values.ndim > 1
    return bool(len(values)) and np.ndim(values[0]) > 0


def _series(ys, labels) -> list:
    """Normalize ys (array, sequence of arrays, or {label: array}) to [(label, y), ...]."""
    if isinstance(ys, Mapping):
        return list(ys.items())
    if _is_multi(ys):
        ys = list(ys)
        labels = list(labels) if labels is not None else [None] * len(ys)
        return list(zip(labels, ys))
    label = labels if isinstance(labels, str) else (labels[0] if labels else None)
    return [(label, ys)]


def _xs(x, n: int) -> list:
    """One x per series: a shared axis, or one array per series."""
    if n > 1 and _is_multi(x) and len(x) == n:
        return list(x)
    return [x] * n


def _color_list(opts: PlotOptions, n: int, defaults: Mapping[str, Any]) -> list:
    """Per-series colors: opts.colors, else the call's color, else None (matplotlib's cycle)."""
    if opts.colors:
        return series_colors(opts, n)
    return [defaults.get("color", defaults.get("c"))] * n


def line(x, ys, labels=None, fill: bool = False, fill_alpha: float = 0.15, invert_x: bool = False,
         vertical: bool = False, ax=None, opts: Optional[PlotOptions] = None, **kwargs):
    """Line plot of one or more series. Returns (fig, ax).

    ys: one array, a sequence of arrays, or {label: array}; x is shared or one array per series.
    fill: shade the area under each curve. vertical: draw y along the horizontal axis
    (side traces). invert_x: higher x on the left (NMR ppm convention).
    """
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        series = _series(ys, labels)
        colors = _color_list(opts, len(series), kwargs)
        for (label, y), xi, color in zip(series, _xs(x, len(series)), colors):
            kw = _kwargs(opts, kwargs, color=color)
            if vertical:
                drawn, = ax.plot(y, xi, label=label, **kw)
                if fill:
                    ax.fill_betweenx(xi, 0, y, color=drawn.get_color(), alpha=fill_alpha, linewidth=0)
            else:
                drawn, = ax.plot(xi, y, label=label, **kw)
                if fill:
                    ax.fill_between(xi, y, color=drawn.get_color(), alpha=fill_alpha, linewidth=0)
        if invert_x and not ax.xaxis_inverted():
            ax.invert_xaxis()
        finish(ax, opts)
    return fig, ax


def scatter(x, y, labels=None, annotate: Optional[Sequence[str]] = None, size=None, ax=None,
            opts: Optional[PlotOptions] = None, **kwargs):
    """Scatter plot of one or more series (same input shapes as `line`). Returns (fig, ax).

    annotate: one text per point (single series only), drawn next to each point.
    """
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        series = _series(y, labels)
        colors = _color_list(opts, len(series), kwargs)
        for (label, yi), xi, color in zip(series, _xs(x, len(series)), colors):
            kw = _kwargs(opts, kwargs, color=color, lines=False)
            if size is not None:
                kw["s"] = size
            ax.scatter(xi, yi, label=label, **kw)
        if annotate is not None:
            for text, xi, yi in zip(annotate, np.atleast_1d(x), np.atleast_1d(y)):
                ax.annotate(text, (xi, yi), xytext=(5, 4), textcoords="offset points",
                            fontsize="small", weight="bold",
                            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))
        finish(ax, opts)
    return fig, ax


# label offsets tried by label_points: directions in order of preference, distances in points
_LABEL_DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
_LABEL_RADII = (6, 14, 24, 36, 50, 66, 84, 104)


def _overlap(a, b) -> float:
    """Overlap area of two (x0, y0, x1, y1) boxes."""
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def label_points(ax, x, y, texts: Sequence[str], fontsize="medium", marker_size=4.0):
    """Write `texts` next to the points (x, y) without labels overlapping each other or the markers.

    Each label goes to the nearest free spot around its point (8 directions, growing distance),
    in the given order (earlier points get the closest spots); a thin leader line joins a label
    moved away from its point. Placement is redone whenever the view changes, so labels stay apart
    while zooming in the --show window. Returns the annotations.
    marker_size: half-size (points) of the area kept free around each point (one value or one per point).
    """
    from matplotlib.text import Text

    xy = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)
    half = np.broadcast_to(np.asarray(marker_size, dtype=float), len(xy))
    anns = [ax.annotate(text, tuple(p), xytext=(0, 0), textcoords="offset points", fontsize=fontsize,
                        weight="bold", zorder=11,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85,
                                  edgecolor="0.3", linewidth=0.5),
                        arrowprops=dict(arrowstyle="-", color="0.15", linewidth=0.7, shrinkA=0, shrinkB=2))
            for text, p in zip(texts, xy)]

    def place(*_):
        fig = ax.figure
        renderer = fig.canvas.get_renderer()
        ax.apply_aspect()                                    # fixed aspect moves the axes only at draw time
        pt = fig.dpi / 72                                    # pixels per point
        frame = ax.get_window_extent(renderer).extents
        pix = ax.transData.transform(xy)
        taken = [(px - r * pt, py - r * pt, px + r * pt, py + r * pt) for (px, py), r in zip(pix, half)]
        pad = 2 * pt
        for ann, (px, py), r in zip(anns, pix, half):
            if not (frame[0] <= px <= frame[2] and frame[1] <= py <= frame[3]):
                continue                                     # point out of view: clipped anyway
            ext = Text.get_window_extent(ann, renderer)      # text only: Annotation's adds the arrow
            w, h = ext.width + 2 * pad, ext.height + 2 * pad
            best = None
            for dist in _LABEL_RADII:
                dist = max(dist, r + 2)
                d = dist * pt
                for dx, dy in _LABEL_DIRECTIONS:
                    x0 = px + d if dx > 0 else px - d - w if dx < 0 else px - w / 2
                    y0 = py + d if dy > 0 else py - d - h if dy < 0 else py - h / 2
                    box = (x0, y0, x0 + w, y0 + h)
                    outside = w * h - _overlap(box, frame)
                    if outside < 0.5:                        # < half a pixel²: rounding, i.e. inside
                        outside = 0.0
                    score = sum(_overlap(box, t) for t in taken) + outside
                    if best is None or score < best[0]:
                        best = (score, dist, dx, dy, box)
                    if score == 0:
                        break
                if best[0] == 0:
                    break
            _, dist, dx, dy, box = best
            ann.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
            ann.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
            ann.xyann = (dx * (dist + 2), dy * (dist + 2))
            ann.arrow_patch.set_visible(dist > max(_LABEL_RADII[0], r + 2))
            taken.append((box[0] - pt, box[1] - pt, box[2] + pt, box[3] + pt))   # 1 pt gap between labels

    place()
    ax.callbacks.connect("xlim_changed", place)
    ax.callbacks.connect("ylim_changed", place)
    ax.figure.canvas.mpl_connect("resize_event", place)
    _after_layout(ax.figure, place)                          # tight layout in save_figure moves the axes
    return anns


def bar(categories, values, labels=None, horizontal: bool = False, width: float = 0.8, ax=None,
        opts: Optional[PlotOptions] = None, **kwargs):
    """Bar chart; several series ({label: values} or a sequence) are grouped side by side.

    Returns (fig, ax).
    """
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        series = _series(values, labels)
        colors = _color_list(opts, len(series), kwargs)
        pos = np.arange(len(categories), dtype=float)
        w = width / len(series)
        for i, ((label, v), color) in enumerate(zip(series, colors)):
            kw = _kwargs(opts, kwargs, color=color, lines=False, marker=False)
            offset = pos - width / 2 + w * (i + 0.5)
            if horizontal:
                ax.barh(offset, v, height=w, label=label, **kw)
            else:
                ax.bar(offset, v, width=w, label=label, **kw)
        if horizontal:
            ax.set_yticks(pos, [str(c) for c in categories])
        else:
            ax.set_xticks(pos, [str(c) for c in categories])
        finish(ax, opts)
    return fig, ax


def hist(values, bins="auto", density: bool = False, labels=None, ax=None,
         opts: Optional[PlotOptions] = None, **kwargs):
    """Histogram of one or more data sets (several are overlaid, semi-transparent). Returns (fig, ax)."""
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        series = _series(values, labels)
        colors = _color_list(opts, len(series), kwargs)
        if len(series) > 1:
            kwargs.setdefault("alpha", 0.6)
        for (label, v), color in zip(series, colors):
            kw = _kwargs(opts, kwargs, color=color, marker=False)
            ax.hist(np.asarray(v, dtype=float), bins=bins, density=density, label=label, **kw)
        finish(ax, opts)
    return fig, ax


def colorbar(mappable, ax=None, cax=None, label: Optional[str] = None):
    """Colorbar next to `ax` (or in `cax`). Returns the Colorbar."""
    fig = (cax if cax is not None else ax).figure
    if cax is not None:
        return fig.colorbar(mappable, cax=cax, label=label)
    return fig.colorbar(mappable, ax=ax, label=label, fraction=0.046, pad=0.04)


def _tick_labels(labels, max_ticks: Optional[int]):
    labels = [str(t) for t in labels]
    if not max_ticks or len(labels) <= max_ticks:
        return labels
    step = max(1, len(labels) // 20)
    return [t if i % step == 0 else "" for i, t in enumerate(labels)]


def heatmap(matrix, xticklabels=None, yticklabels=None, cbar_label: Optional[str] = None,
            vmin=None, vmax=None, max_xticks: Optional[int] = 50, max_yticks: Optional[int] = None,
            ax=None, opts: Optional[PlotOptions] = None, **kwargs):
    """Cell heatmap with a colorbar; row 0 at the top, cell (i, j) spans [j, j+1] × [i, i+1].

    Tick labels beyond max_xticks / max_yticks are thinned to about 20. Returns (fig, ax).
    """
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        data = np.asarray(matrix, dtype=float)
        cmap = get_cmap(opts.cmap or kwargs.pop("cmap", "viridis"))
        mesh = ax.pcolormesh(data, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
        ax.set_xlim(0, data.shape[1])
        ax.set_ylim(data.shape[0], 0)
        if xticklabels is not None:
            ax.set_xticks(np.arange(data.shape[1]) + 0.5,
                          _tick_labels(xticklabels, max_xticks), rotation=90, fontsize="small")
        if yticklabels is not None:
            ax.set_yticks(np.arange(data.shape[0]) + 0.5,
                          _tick_labels(yticklabels, max_yticks), rotation=0, fontsize="small")
        colorbar(mesh, ax=ax, label=cbar_label)
        finish(ax, opts)
    return fig, ax


def contour(x, y, z, levels=10, filled: bool = False, colors=None, cbar_label: Optional[str] = None,
            ax=None, opts: Optional[PlotOptions] = None, **kwargs):
    """Contour map of z(y, x): explicit `colors` per level, otherwise opts.cmap.

    A colorbar is added when `cbar_label` is given. Returns (fig, ax).
    """
    opts = _opts(opts)
    with style(opts):
        fig, ax = _axes(ax, opts)
        kw = {_ALIASES.get(k, k): v for k, v in kwargs.items()}
        if not filled:
            if opts.linewidth is not None:
                kw["linewidths"] = opts.linewidth
            if opts.linestyle:
                kw["linestyles"] = LINESTYLES.get(opts.linestyle, opts.linestyle)
        if colors is not None:
            kw["colors"] = colors
        else:
            kw["cmap"] = get_cmap(opts.cmap or kw.pop("cmap", "viridis"))
        draw = ax.contourf if filled else ax.contour
        cs = draw(x, y, z, levels=levels, **kw)
        if cbar_label is not None:
            colorbar(cs, ax=ax, label=cbar_label)
        finish(ax, opts)
    return fig, ax


# ------------------------------------------------------------------- output
def resolve_path(
    user_path: Optional[Union[str, Path]],
    default: Union[str, Path],
    ext: str,
    tag: Optional[str] = None,
    known_exts=None,
) -> Path:
    """Build an output path following the FROG naming rules.

    * `user_path` (from -o/--plot) wins over `default`; without either the default is used.
    * A known extension already in the path wins over `ext`.
    * `tag` is appended to the stem: `{stem}_{tag}{ext}` (prefix semantics for multi-output).
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    known = {e if e.startswith(".") else f".{e}" for e in (known_exts or FIG_FORMATS)}
    known.add(ext)
    base = Path(user_path) if user_path else Path(default)
    if base.suffix.lower() in known:
        ext = base.suffix
        base = base.with_suffix("")
    if tag:
        base = base.with_name(f"{base.name}_{tag}")
    return base.with_name(base.name + ext)


def _after_layout(fig, callback) -> None:
    """Run `callback()` after save_figure() has laid out `fig` (e.g. to re-place labels)."""
    fig._frog_after_layout = [*getattr(fig, "_frog_after_layout", ()), callback]


def tight_layout(fig) -> None:
    """Matplotlib's tight layout (titles, labels and an outside legend fit in the figure), then the
    callbacks registered with _after_layout. Figures with their own layout engine (layout='constrained')
    are left alone."""
    import warnings

    from matplotlib.layout_engine import PlaceHolderLayoutEngine
    engine = fig.get_layout_engine()
    if engine is not None and not isinstance(engine, PlaceHolderLayoutEngine):
        return
    with warnings.catch_warnings():
        # axes tight layout cannot handle are left as they are
        warnings.filterwarnings("ignore", message=".*[Tt]ight.?layout not applied")
        fig.tight_layout()
    for callback in getattr(fig, "_frog_after_layout", ()):
        callback()


def save_figure(fig, path: Union[str, Path], dpi: int = DEFAULT_DPI, show: bool = False,
                kind: str = "plot", defer_show: bool = False, layout: bool = True, **savefig_kwargs) -> Path:
    """Save `fig` to `path`, report it on stderr, then show (if requested) or close it.

    layout=True (default) applies tight_layout(fig) first, so the saved and shown figure use the
    same layout. With defer_show=True the figure stays open and the caller calls show_figures()
    once (CLICommand does this after run(), so --show opens all figures together).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    with style():
        if layout:
            tight_layout(fig)
        fig.savefig(path, dpi=dpi, **savefig_kwargs)
    print_saved(path, kind)
    if not show:
        close(fig)
    elif not defer_show:
        show_figures()
    return path


def show_figures() -> None:
    """Open every open figure in an interactive window (blocks until they are closed)."""
    import matplotlib.pyplot as plt
    if plt.get_fignums():
        plt.show()


def close(fig) -> None:
    """Close a figure without showing it."""
    import matplotlib.pyplot as plt
    plt.close(fig)
