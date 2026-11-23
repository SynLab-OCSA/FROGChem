"""Orbital transition diagram of TD-DFT excited states (ORCA or Gaussian), drawn with frogchem.plotter.

The orbitals involved in the chosen states are drawn as energy levels, one column for a closed-shell
(restricted) calculation, two (alpha | beta) for an open-shell one, with their occupation and one
arrow per main excitation (weight ≥ wmin): colour = state, width ∝ weight, dashed = dark state.
Level spacing is schematic (ordered by energy, energies printed). Orbital indices are 0-based.

Used by ``frog-tddft --transition-diagram``.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from frogchem.chemistry import HARTREE_TO_EV
from frogchem.orbital_character import SpinOrbitals, orbitals_of, transition_spin

MINUS, ARROW = "−", "→"
#: Okabe-Ito palette (colour-blind safe), default colours of the states
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999")
INK, MUTED = "#222222", "#666666"
ROW, GAP, LANE = 1.0, 0.7, 0.55     # layout units (1 unit = SCALE inches)
SCALE = 0.32


# --------------------------------------------------------------- energies
_GAUSS_EIG = re.compile(r"(Alpha|Beta)\s+(?:occ\.|virt\.) eigenvalues --(.*)")
_GAUSS_NUM = re.compile(r"-?\d+\.\d{5}")
_ORCA_ROW = re.compile(r"^\s*\d+\s+\d+\.\d+\s+-?\d+\.\d+\s+(-?\d+\.\d+)\s*$")


def _gaussian_levels(fl: str) -> Dict[str, List[float]]:
    """Last block of 'Alpha/Beta occ./virt. eigenvalues' (Eh) → eV."""
    blocks, cur, prev = [], None, False
    for line in fl.splitlines():
        m = _GAUSS_EIG.search(line)
        if m:
            if not prev:
                cur = {"a": [], "b": []}
                blocks.append(cur)
            cur["a" if m.group(1) == "Alpha" else "b"].extend(
                float(x) * HARTREE_TO_EV for x in _GAUSS_NUM.findall(m.group(2)))
            prev = True
        else:
            prev = False
    return blocks[-1] if blocks else {}


def _orca_levels(fl: str) -> Dict[str, List[float]]:
    """Last 'ORBITAL ENERGIES' block (E(eV) column; SPIN UP / SPIN DOWN for UKS)."""
    pos = fl.rfind("ORBITAL ENERGIES")
    if pos < 0:
        return {}
    out, spin, started = {"a": [], "b": []}, "a", False
    for line in fl[pos:].splitlines()[1:]:
        m = _ORCA_ROW.match(line)
        if m:
            out[spin].append(float(m.group(1)))
            started = True
            continue
        text = line.strip()
        if "SPIN DOWN" in text:
            spin = "b"
        elif not text or text.startswith(("-", "*", "NO ", "SPIN UP")):
            continue
        elif started:
            break
    return out


def orbital_levels(calc, open_shell: bool) -> Dict[str, Optional[np.ndarray]]:
    """Orbital energies in eV per column ('' closed shell, 'a'/'b' open shell); None when not printed."""
    en = getattr(calc, "orbital_energies", None)
    found: Dict[str, List[float]] = {}
    if en is not None and len(en):
        en = np.asarray(en, dtype=float) * HARTREE_TO_EV
        alpha = getattr(calc, "alpha_coeff", None)
        if alpha is not None:
            n = alpha.shape[1]
            found = {"a": en[:n], "b": en[n:]}
        else:
            found = {"a": en, "b": en}
    else:
        fl = calc.__dict__.get("fl") or ""
        found = _gaussian_levels(fl) if "eigenvalues --" in fl else _orca_levels(fl)
        if found and not found.get("b"):
            found["b"] = found.get("a", [])
    get = lambda s: np.asarray(found[s], dtype=float) if found.get(s) is not None and len(found[s]) else None
    if open_shell:
        return {"a": get("a"), "b": get("b")}
    return {"": get("a")}


# ------------------------------------------------------------------- data
@dataclass
class DiagramState:
    label: str
    energy: float
    fosc: float
    s2: Optional[float]
    contribs: List[Tuple[str, int, int, float]]        # (column, from, to, weight), by weight
    character: Optional[dict] = None


@dataclass
class DiagramData:
    title: str
    open_shell: bool
    columns: List[str]                                 # [''] or ['a', 'b']
    homo: Dict[str, int]
    states: List[DiagramState]
    arrows: List[Tuple[int, str, int, int, float]]     # (state index, column, from, to, weight)
    levels: Dict[str, List[int]]
    energies: Dict[str, Optional[np.ndarray]]
    analysis: Optional[Dict[str, SpinOrbitals]] = None
    n_fragments: int = 0
    wmin: float = 0.10
    fmin: float = 0.01
    labels: Dict[Tuple[str, int], str] = field(default_factory=dict)
    weight_def: str = "weights as printed"


def weight_definition(calc, open_shell: bool) -> str:
    """What the transition weights are: Gaussian prints coefficients c (the parser stores 2c² closed shell,
    c² open shell); ORCA prints the weight itself, already normalised, and it is used as is."""
    from frogchem.gaussian_parser.calculation import GaussianCalculation
    if isinstance(calc, GaussianCalculation):
        return "weight = c\u00b2" if open_shell else "weight = 2c\u00b2"
    return "weights as printed by ORCA"


def is_open_shell(calc, states: Sequence[dict] = ()) -> bool:
    if any(transition_spin(t) in ("a", "b") for st in states for t in st.get("population", [])):
        return True
    beta = getattr(calc, "beta_homo", None)
    return beta is not None and beta != calc.homo


def collect(calc, chosen: Sequence[Tuple[str, dict, Optional[dict]]], wmin: float = 0.10,
            fmin: float = 0.01, analysis: Optional[Dict[str, SpinOrbitals]] = None,
            labels: Optional[Dict[Tuple[str, int], str]] = None, title: str = "") -> DiagramData:
    """Levels and arrows of the chosen states [(label, state dict, character or None), ...]."""
    open_shell = is_open_shell(calc, [st for _, st, _ in chosen])
    columns = ["a", "b"] if open_shell else [""]
    homo_a = int(calc.homo)
    beta = getattr(calc, "beta_homo", None)
    homo = {"": homo_a, "a": homo_a, "b": int(beta) if beta is not None else homo_a}

    states, arrows = [], []
    for k, (label, st, char) in enumerate(chosen):
        contribs = []
        for t in st.get("population", []):
            spin = transition_spin(t)
            col = (spin or "a") if open_shell else ""
            contribs.append((col, int(t[0]), int(t[1]), float(t[2])))
        contribs.sort(key=lambda c: -c[3])
        s2 = st.get("S**2")
        states.append(DiagramState(label, float(st.get("energy", 0.0)), float(st.get("fosc", 0.0)),
                                   float(s2) if s2 is not None else None, contribs, char))
        arrows += [(k, c, i, a, w) for c, i, a, w in contribs if w >= wmin]

    levels = {c: set() for c in columns}
    for _, c, i, a, _ in arrows:
        levels[c] |= {i, a}
    if open_shell:
        for mo in range(homo["b"] + 1, homo["a"] + 1):     # singly occupied alpha and their empty beta partners
            levels["a"].add(mo)
            levels["b"].add(mo)
    else:
        levels[""] |= {homo_a, homo_a + 1}
    n_frag = 0
    if analysis:
        frag = next(iter(analysis.values())).fragments
        n_frag = 0 if frag is None else len(frag)
    return DiagramData(title=title, open_shell=open_shell, columns=columns, homo=homo, states=states,
                       arrows=arrows, levels={c: sorted(v) for c, v in levels.items()},
                       energies=orbital_levels(calc, open_shell), analysis=analysis, n_fragments=n_frag,
                       wmin=wmin, fmin=fmin, labels=dict(labels or {}),
                       weight_def=weight_definition(calc, open_shell))


# ------------------------------------------------------------------ names
def orbital_name(mo: int, col: str, data: DiagramData, short: bool = False) -> str:
    """H, H−1, L, L+2 (HOMO/LUMO with short=False) or SOMO, SOMO−1 for singly occupied alpha MOs."""
    homo_a, homo_b = data.homo["a"], data.homo["b"]
    if data.open_shell and col == "a" and homo_b < mo <= homo_a:
        return "SOMO" if mo == homo_a else f"SOMO{MINUS}{homo_a - mo}"
    r = mo - data.homo[col]
    h, l = ("H", "L") if short else ("HOMO", "LUMO")
    if r <= 0:
        return h if r == 0 else f"{h}{MINUS}{-r}"
    return l if r == 1 else f"{l}+{r - 1}"


def parse_level_labels(spec: str) -> Dict[Tuple[str, int], str]:
    """'a51=π*(C=S),b48=n(S),50=π' → {('a', 51): 'π*(C=S)', ('b', 48): 'n(S)', ('', 50): 'π', ...}.

    Indices are 0-based; a bare index applies to the closed-shell column and to both spins.
    """
    out = {}
    for item in [s for s in (spec or "").split(",") if "=" in s]:
        key, val = item.split("=", 1)
        key = key.strip().lower()
        cols = {"a": ["a"], "b": ["b"]}.get(key[:1], ["", "a", "b"])
        try:
            idx = int(key.lstrip("ab"))
        except ValueError:
            from frogchem.exceptions import InputError
            raise InputError(f"Level label not understood: {item!r} (use e.g. a51=pi*)") from None
        for c in cols:
            out[(c, idx)] = val.strip()
    return out


def _fmt_e(e: float) -> str:
    return f"{e:.2f}".replace("-", MINUS)


def level_text(data: DiagramData, col: str, mo: int) -> str:
    parts = [f"{mo} {orbital_name(mo, col, data)}"]
    if data.analysis:
        orb = orbitals_of(data.analysis, col)
        if mo < len(orb.types):
            parts.append(orb.types[mo])
            if orb.fragments is not None:
                names = "AB" if len(orb.fragments) == 2 else "A"
                parts.append(" ".join(f"{n} {orb.fragments[j][mo] * 100:.0f}%" for j, n in enumerate(names)))
    if (col, mo) in data.labels:
        parts.append(data.labels[(col, mo)])
    en = data.energies.get(col)
    if en is not None and mo < len(en):
        parts.append(f"{_fmt_e(en[mo])} eV")
    return " · ".join(parts)


def character_text(char: Optional[dict]) -> str:
    """' [CT A→B 0.92, π→π* 1.00]': the type with its share first, then the other shares ≥ 0.05."""
    if not char:
        return ""
    key = {"CT A→B": "A→B", "CT B→A": "B→A"}.get(char["type"], char["type"])
    shares = [(k, v) for k, v in char.items() if k not in ("type", "A loc", "B loc") and v >= 0.05]
    shares.sort(key=lambda kv: kv[0] != key)
    text = ", ".join(f"{char['type'] if k == key else k} {v:.2f}" for k, v in shares)
    if char["type"] == "mixed":
        return f" [mixed{': ' + text if text else ''}]"
    return f" [{text}]"


def legend_text(data: DiagramData, k: int) -> str:
    st = data.states[k]
    parts = []
    for col, i, a, w in st.contribs[:3]:
        if w < data.wmin:
            continue
        pre = {"a": "α ", "b": "β "}.get(col, "")
        parts.append(f"{pre}{orbital_name(i, col, data, True)}{ARROW}{orbital_name(a, col, data, True)} "
                     f"{w * 100:.0f}%")
    f = st.fosc
    fs = f"{f:.2f}" if f >= 0.1 else (f"{f:.3f}" if f >= 0.01 else f"{f:.4f}")
    nm = f"{1239.84193 / st.energy:.0f} nm, " if st.energy > 0 else ""
    s2 = f", <S²> {st.s2:.2f}" if data.open_shell and st.s2 is not None else ""
    return (f"{st.label}  {st.energy:.2f} eV, {nm}f {fs}{s2}{character_text(st.character)}: "
            + ", ".join(parts))


# ---------------------------------------------------------------- drawing
def draw(data: DiagramData, opts=None):
    """The transition diagram as a matplotlib figure (frogchem.plotter style and options)."""
    from frogchem import plotter

    opts = (opts or plotter.PlotOptions()).with_defaults(colors=PALETTE, title=data.title)
    fs = float(opts.font_size or 9)
    colors = plotter.series_colors(opts, max(1, len(data.states)))
    cols = data.columns
    homo = data.homo

    occ = {c: sorted([m for m in data.levels[c] if m <= homo[c]], reverse=True) for c in cols}
    vir = {c: sorted([m for m in data.levels[c] if m > homo[c]]) for c in cols}
    n_vir = max([len(vir[c]) for c in cols] + [1])
    n_occ = max([len(occ[c]) for c in cols] + [1])
    ypos = {}
    for c in cols:
        for j, m in enumerate(vir[c]):
            ypos[(c, m)] = GAP / 2 + (j + 0.5) * ROW
        for j, m in enumerate(occ[c]):
            ypos[(c, m)] = -(GAP / 2 + (j + 0.5) * ROW)
    y_top, y_bot = GAP / 2 + n_vir * ROW, -(GAP / 2 + n_occ * ROW)

    lanes = {c: [ar for ar in data.arrows if ar[1] == c] for c in cols}
    lw = {c: max(2.6, 1.5 + LANE * len(lanes[c])) for c in cols}
    x0 = {cols[0]: 0.0}
    if len(cols) == 2:
        x0["b"] = lw["a"] + 2.2
    x_left, x_right = 0.0, x0[cols[-1]] + lw[cols[-1]]

    n_leg = len(data.states) + 1
    y_leg = y_bot - 1.2
    y_min = y_leg - 0.55 * n_leg - 0.3
    y_max = y_top + 2.4
    size = opts.figsize or ((x_right - x_left + 1.0) * SCALE, (y_max - y_min) * SCALE)

    with plotter.style(opts):
        fig, ax = plotter.new_figure(figsize=size, opts=opts)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(x_left - 0.5, x_right + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axis("off")

        def text(x, y, s, size=fs, ha="left", color=INK, weight="normal", **kw):
            ax.text(x, y, s, fontsize=size, ha=ha, va="center", color=color, weight=weight, **kw)

        def arrow(x, y1, y2, color, width, dashed=False, head=6.0):
            ax.annotate("", xy=(x, y2), xytext=(x, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=width, shrinkA=0, shrinkB=0,
                                        mutation_scale=head * 1.6, linestyle="--" if dashed else "-"))

        if opts.title:
            text(x_left, y_top + 1.9, opts.title, fs + 3, weight="bold")
        if data.open_shell:
            kind = (f"open shell: {homo['a'] + 1} α / {homo['b'] + 1} β electrons "
                    "(indices 0-based)")
        else:
            kind = f"closed shell: {homo[''] + 1} doubly occupied orbitals (indices 0-based)"
        text(x_left, y_top + 1.35, kind, fs, color=MUTED)
        for c in cols:
            head = {"": "orbitals", "a": "spin α", "b": "spin β"}[c]
            text(x0[c] + lw[c] / 2, y_top + 0.55, head, fs + 1, ha="center", weight="bold")

        ax.plot([x_left - 0.3, x_right + 0.3], [0, 0], color=MUTED, lw=0.6, ls=(0, (4, 4)))
        if len(cols) == 2:
            xm = (x0["a"] + lw["a"] + x0["b"]) / 2
            text(xm, 0.22, "empty", fs - 1, ha="center", color=MUTED)
            text(xm, -0.22, "occupied", fs - 1, ha="center", color=MUTED)
        else:
            text(x_left - 0.4, 0.22, "empty", fs - 1, ha="right", color=MUTED)
            text(x_left - 0.4, -0.22, "occupied", fs - 1, ha="right", color=MUTED)

        for c in cols:
            xa = x0[c]
            cx = xa + 0.55
            right = c != "a"
            for m in data.levels[c]:
                y = ypos[(c, m)]
                ax.plot([xa, xa + lw[c]], [y, y], color=INK, lw=2.0, solid_capstyle="butt")
                text(xa + lw[c] + 0.25 if right else xa - 0.25, y, level_text(data, c, m), fs,
                     ha="left" if right else "right")
                if m <= homo[c]:
                    if c == "a":
                        arrow(cx, y - 0.3, y + 0.32, INK, 1.0, head=4)
                    elif c == "b":
                        arrow(cx, y + 0.3, y - 0.32, INK, 1.0, head=4)
                    else:
                        arrow(cx - 0.15, y - 0.3, y + 0.32, INK, 1.0, head=4)
                        arrow(cx + 0.15, y + 0.3, y - 0.32, INK, 1.0, head=4)
                elif c == "b" and homo["b"] < m <= homo["a"]:
                    ax.plot([cx], [y], marker="o", ms=5, mfc="none", mec=INK, mew=1.0)

        for c in cols:
            for idx, (k, _, i, a, w) in enumerate(lanes[c]):
                st = data.states[k]
                col = colors[k]
                x = x0[c] + 1.3 + LANE * idx
                y1, y2 = ypos[(c, i)], ypos[(c, a)]
                ax.plot([x], [y1], marker="o", ms=3.5, color=col)
                arrow(x, y1, y2, col, 0.8 + 2.4 * w, dashed=st.fosc < data.fmin, head=7)
                text(x, y2 + (0.28 if y2 > y1 else -0.28), st.label, fs - 2, ha="center", color=col,
                     weight="bold", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

        text(x_left - 0.3, y_leg, f"Excited states (contributions ≥ {data.wmin * 100:.0f}%, {data.weight_def}; "
             f"arrow width ~ weight; dashed: f < {data.fmin:g}; level spacing schematic)", fs - 1, color=MUTED)
        for k in range(len(data.states)):
            y = y_leg - 0.55 * (k + 1)
            ax.plot([x_left - 0.3, x_left + 0.5], [y, y], color=colors[k], lw=2.2,
                    ls="--" if data.states[k].fosc < data.fmin else "-")
            text(x_left + 0.7, y, legend_text(data, k), fs)
    return fig
