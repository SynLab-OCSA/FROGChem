"""frogchem.plotter: options precedence, primitives, non-global style, saving, shared CLI flags."""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from frogchem import plotter  # noqa: E402
from frogchem.plotter import PlotOptions  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rgba(color):
    return mcolors.to_rgba(color)


# ------------------------------------------------------------------ options
def test_cli_option_beats_command_default():
    opts = PlotOptions(title="Mine").with_defaults(title="Command", xlabel="x (Å)")
    assert opts.title == "Mine"
    assert opts.xlabel == "x (Å)"


def test_empty_title_from_cli_is_kept():
    assert PlotOptions(title="").with_defaults(title="Command").title == ""


def test_unknown_default_is_an_error():
    with pytest.raises(TypeError):
        PlotOptions().with_defaults(colour="red")


def test_from_args_reads_cli_flags_and_tuples_lists():
    args = argparse.Namespace(title="T", colors=["red", "blue"], figsize=[4, 3], legend=False, dpi=300)
    opts = PlotOptions.from_args(args)
    assert opts.title == "T"
    assert opts.colors == ("red", "blue")
    assert opts.figsize == (4, 3)
    assert opts.legend is False
    assert opts.linewidth is None


def test_series_colors_cycle_user_colors():
    assert plotter.series_colors(PlotOptions(colors=("red", "blue")), 3) == ["red", "blue", "red"]


def test_energy_label():
    assert plotter.energy_label("G", delta=True) == "ΔG (kcal/mol)"
    assert plotter.energy_label() == "Energy (kcal/mol)"


# -------------------------------------------------------------------- style
def test_primitives_and_saving_leave_global_rcparams_untouched(tmp_path):
    before = dict(mpl.rcParams)
    fig, _ = plotter.line([0, 1, 2], [0, 1, 4], opts=PlotOptions(font_size=20, linewidth=4, grid=True))
    plotter.save_figure(fig, tmp_path / "a.png")
    assert dict(mpl.rcParams) == before


def test_style_applies_inside_the_block_only():
    before = mpl.rcParams["font.size"]
    with plotter.style(PlotOptions(font_size=20)):
        assert mpl.rcParams["font.size"] == 20
        assert mpl.rcParams["axes.titlesize"] == 22
    assert mpl.rcParams["font.size"] == before


def test_new_figure_size_from_options():
    fig, _ = plotter.new_figure(figsize=(3, 2), opts=PlotOptions(figsize=(6, 4)))
    assert tuple(fig.get_size_inches()) == (6, 4)
    fig, _ = plotter.new_figure(figsize=(3, 2))
    assert tuple(fig.get_size_inches()) == (3, 2)


# --------------------------------------------------------------- primitives
def test_line_applies_options():
    opts = PlotOptions(title="Spectrum", xlabel="x", ylabel="y", colors=("red", "blue"),
                       linewidth=3, linestyle="dashed", xlim=(0, 2))
    _, ax = plotter.line([0, 1, 2], {"a": [0, 1, 2], "b": [2, 1, 0]}, opts=opts)
    lines = ax.get_lines()
    assert len(lines) == 2
    assert [_rgba(ln.get_color()) for ln in lines] == [_rgba("red"), _rgba("blue")]
    assert all(ln.get_linewidth() == 3 and ln.get_linestyle() == "--" for ln in lines)
    assert ax.get_title() == "Spectrum" and ax.get_xlabel() == "x" and ax.get_ylabel() == "y"
    assert ax.get_xlim() == (0, 2)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


def test_option_wins_over_call_default_but_call_default_wins_over_style():
    _, ax = plotter.line([0, 1], [0, 1], opts=PlotOptions(), linewidth=0.5, color="gray")
    assert ax.get_lines()[0].get_linewidth() == 0.5
    _, ax = plotter.line([0, 1], [0, 1], opts=PlotOptions(linewidth=2, colors=("red",)),
                         linewidth=0.5, color="gray")
    assert ax.get_lines()[0].get_linewidth() == 2
    assert _rgba(ax.get_lines()[0].get_color()) == _rgba("red")


def test_legend_is_automatic_and_can_be_forced_off():
    _, ax = plotter.line([0, 1], [0, 1])
    assert ax.get_legend() is None                          # nothing labelled
    _, ax = plotter.line([0, 1], {"a": [0, 1]})
    assert ax.get_legend() is not None
    _, ax = plotter.line([0, 1], {"a": [0, 1]}, opts=PlotOptions(legend=False))
    assert ax.get_legend() is None


def test_line_fill_invert_and_limits_keep_inversion():
    _, ax = plotter.line([1, 2, 3], [0, 1, 0], fill=True, invert_x=True, opts=PlotOptions(xlim=(1, 3)))
    assert len(ax.collections) == 1                          # shaded area
    assert ax.xaxis_inverted()
    assert ax.get_xlim() == (3, 1)


def test_line_accepts_pandas_series_with_non_zero_index():
    pd = pytest.importorskip("pandas")
    s = pd.Series([1.0, 2.0, 3.0], index=[5, 6, 7])
    _, ax = plotter.line(s.index, s)
    assert len(ax.get_lines()) == 1


def test_line_series_with_own_x():
    _, ax = plotter.line([[0, 1], [0, 1, 2]], [[0, 1], [1, 2, 3]], labels=["a", "b"])
    assert [len(ln.get_xdata()) for ln in ax.get_lines()] == [2, 3]


def test_scatter_with_annotations():
    _, ax = plotter.scatter([0.1, 0.2], [0.3, 0.4], annotate=["p1", "p2"], size=50,
                            opts=PlotOptions(colors=("green",)))
    assert len(ax.collections) == 1
    assert [t.get_text() for t in ax.texts] == ["p1", "p2"]
    assert _rgba(ax.collections[0].get_facecolor()[0]) == _rgba("green")


def test_grouped_bars():
    _, ax = plotter.bar(["C0", "C1", "C2"], {"Mulliken": [0.1, -0.2, 0.3], "Löwdin": [0.0, -0.1, 0.2]},
                        opts=PlotOptions(colors=("red", "blue")))
    assert len(ax.patches) == 6
    assert [t.get_text() for t in ax.get_xticklabels()] == ["C0", "C1", "C2"]
    assert _rgba(ax.patches[0].get_facecolor()) == _rgba("red")
    assert _rgba(ax.patches[3].get_facecolor()) == _rgba("blue")


def test_hist():
    _, ax = plotter.hist(np.random.default_rng(0).normal(size=200), bins=10, opts=PlotOptions(title="H"))
    assert len(ax.patches) == 10
    assert ax.get_title() == "H"


def test_heatmap_orientation_ticks_and_colorbar():
    data = np.arange(6).reshape(2, 3)
    fig, ax = plotter.heatmap(data, xticklabels=["a", "b", "c"], yticklabels=["r0", "r1"], cbar_label="%",
                              opts=PlotOptions(cmap="magma"))
    assert ax.get_ylim() == (2, 0)                           # row 0 on top
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]
    assert len(fig.axes) == 2                                # colorbar
    assert fig.axes[1].get_ylabel() == "%"
    assert ax.collections[0].get_cmap().name == "magma"


def test_heatmap_thins_many_tick_labels():
    _, ax = plotter.heatmap(np.zeros((1, 100)), xticklabels=range(100), max_xticks=50)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert 0 < sum(bool(t) for t in labels) <= 20


def test_contour_with_colorbar_and_linewidth():
    x = y = np.linspace(-1, 1, 20)
    z = np.exp(-(x[None, :] ** 2 + y[:, None] ** 2))
    fig, ax = plotter.contour(x, y, z, levels=5, cbar_label="I", opts=PlotOptions(linewidth=2.5))
    assert len(fig.axes) == 2
    assert all(lw == 2.5 for lw in ax.collections[0].get_linewidths())


def test_get_cmap_matplotlib_and_seaborn():
    assert plotter.get_cmap("viridis").name == "viridis"
    with pytest.raises(ValueError):
        plotter.get_cmap("not-a-colormap")
    pytest.importorskip("seaborn")
    assert plotter.get_cmap("rocket_r").name == "rocket_r"


# ------------------------------------------------------------------- output
def test_save_figure_writes_reports_and_closes(tmp_path, capsys, monkeypatch):
    from frogchem import printer
    monkeypatch.setattr(printer, "_QUIET", False)           # other CLI tests may have run with -q
    fig, _ = plotter.line([0, 1], [0, 1])
    path = plotter.save_figure(fig, tmp_path / "sub" / "plot.svg")
    assert path.exists() and path.read_text().startswith("<?xml")
    assert "Saved plot" in capsys.readouterr().err
    assert not plt.fignum_exists(fig.number)


def test_save_figure_deferred_show_keeps_figure_open(tmp_path):
    fig, _ = plotter.line([0, 1], [0, 1])
    plotter.save_figure(fig, tmp_path / "plot.png", show=True, defer_show=True)
    assert plt.fignum_exists(fig.number)


@pytest.mark.parametrize("user, default, ext, tag, expected", [
    (None, "neb", "png", None, "neb.png"),
    ("out.pdf", "neb", "png", None, "out.pdf"),          # extension in the path wins
    ("out", "neb", "svg", None, "out.svg"),
    ("res/out.png", "neb", "png", "UV", "res/out_UV.png"),
    (None, "spectra", "png", "ECD_superimposed", "spectra_ECD_superimposed.png"),
])
def test_resolve_path(user, default, ext, tag, expected):
    assert str(plotter.resolve_path(user, default, ext, tag)) == expected


# -------------------------------------------------------------- CLI flags
NEB_INTERP = """Iteration: 0
------------------------------------------------------------------
0   0.000000   -1.000000
1   1.000000   -0.980000

Interpolated Path
------------------------------------------------------------------
0.000000   0.000000   -1.000000
1.000000   1.000000   -0.980000

Iteration: 1
------------------------------------------------------------------
0   0.000000   -1.000000
1   1.000000   -0.990000

Interpolated Path
------------------------------------------------------------------
0.000000   0.000000   -1.000000
1.000000   1.000000   -0.990000

"""


def _run_neb(tmp_path, monkeypatch, *flags):
    from frogchem.cli.plot_neb import PlotNeb
    captured = {}
    real_save = plotter.save_figure

    def spy(fig, path, **kwargs):
        captured["fig"] = fig
        kwargs["show"] = True                                 # keep the figure open for inspection
        return real_save(fig, path, **kwargs)

    monkeypatch.setattr(plotter, "save_figure", spy)
    interp = tmp_path / "neb.interp"
    interp.write_text(NEB_INTERP)
    out = tmp_path / "neb.png"
    PlotNeb().main([str(interp), "-o", str(out), *flags])
    assert out.exists()
    return captured["fig"].axes[0]


def test_neb_defaults(tmp_path, monkeypatch):
    ax = _run_neb(tmp_path, monkeypatch)
    assert ax.get_ylabel() == "Energy (kcal/mol)"
    assert ax.get_title() == ""
    assert _rgba(ax.get_lines()[0].get_color()) == _rgba("black")
    assert _rgba(ax.get_lines()[-1].get_color()) == _rgba("red")


def test_neb_shared_cli_flags(tmp_path, monkeypatch):
    ax = _run_neb(tmp_path, monkeypatch, "--colors", "navy", "lightgray", "crimson", "--title", "NEB",
                  "--linewidth", "2.5", "--linestyle", "dotted", "--ylabel", "E", "--ylim", "-700", "-600")
    assert ax.get_title() == "NEB"
    assert ax.get_ylabel() == "E"
    assert ax.get_ylim() == (-700, -600)
    lines = ax.get_lines()
    assert _rgba(lines[0].get_color()) == _rgba("navy")
    assert _rgba(lines[-1].get_color()) == _rgba("crimson")
    assert all(ln.get_linewidth() == 2.5 and ln.get_linestyle() == ":" for ln in lines)


def test_cli_run_restores_rcparams(tmp_path, monkeypatch):
    before = dict(mpl.rcParams)
    _run_neb(tmp_path, monkeypatch, "--font-size", "18")
    assert dict(mpl.rcParams) == before
