"""Orbital / difference-density grid for ChimeraX, written as a self-contained script.

``write_grid_script`` writes ``<stem>_grid.py``: a manifest (cube files, panels, labels, caption,
isovalues) followed by the viewer code below. Opening that file in ChimeraX (``open file.py``)
lays out one copy of the geometry per panel, in an MO-diagram arrangement: virtual orbitals on
top, occupied below, and the difference densities of the states at the bottom.

The viewer reads the cube files itself: ChimeraX's own cube reader misreads multicube files
with ten or more orbitals. It needs only numpy and ChimeraX; it does not import frogchem.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

__all__ = ["write_grid_script", "panel", "group", "VIEWER_SOURCE"]


VIEWER_SOURCE = r"""
# ---------------------------------------------------------------- viewer (runs inside ChimeraX)
import os
import tempfile
import numpy as np

BOHR_TO_ANG = 0.52917721067
POS_COLOR, NEG_COLOR, INK = "#2166ac", "#b2182b", "#0b0b0b"
ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge "
            "As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
            "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
PRESET_GLOBAL = ["lighting simple", "lighting shadows false", "set bgColor white"]
PRESET_ATOMS = ["style {s} ball", "size {s} stickRadius 0.06", "size {s} ballScale 0.25",
                "color {s} & C #8f8f8f target a", "color {s} #2b2b2b target b",
                "setattr {s} bonds halfbond false", "size {s} & H atomRadius 0.7",
                "size {s} & C atomRadius 1.21", "size {s} & ~H & ~C atomRadius 1.3"]


def _here():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()


def read_cube(path):
    '''Origin, axis steps (both in Angstrom), atoms (Z, Angstrom) and data (nx, ny, nz, n_orb).'''
    with open(path) as f:
        f.readline()
        f.readline()
        f3 = f.readline().split()
        na = int(f3[0])
        origin = np.array([float(x) for x in f3[1:4]])
        n, axes = [], []
        for _ in range(3):
            p = f.readline().split()
            n.append(int(p[0]))
            axes.append([float(x) for x in p[1:4]])
        unit = BOHR_TO_ANG if n[0] > 0 else 1.0
        Z, X = [], []
        for _ in range(abs(na)):
            p = f.readline().split()
            Z.append(int(float(p[0])))
            X.append([float(x) for x in p[2:5]])
        ids = None
        if na < 0:
            first = f.readline().split()
            k = int(first[0])
            ids = first[1:]
            while len(ids) < k:
                ids += f.readline().split()
            ids = [int(i) for i in ids[:k]]
        values = np.fromstring(f.read(), sep=" ", dtype=float)
    shape = [abs(v) for v in n]
    n_orb = len(ids) if ids else 1
    data = values.reshape(shape + [n_orb])
    return {"origin": origin * unit, "steps": np.array(axes) * unit, "Z": Z,
            "X": np.array(X) * unit, "ids": ids, "data": data}


def _xyz_file(Z, X):
    fd, tmp = tempfile.mkstemp(prefix="orbitalgrid_", suffix=".xyz")
    with os.fdopen(fd, "w") as f:
        f.write(f"{len(Z)}\nframogrid geometry\n")
        for z, x in zip(Z, X):
            f.write(f"{ELEMENTS[z]:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}\n")
    return tmp


def _layout(groups, ncol, W, H, gap=1.0):
    '''Panels of each group in rows of ncol, groups stacked from the top; centred on y = 0.'''
    placed = []
    y0 = 0.0
    for panels in groups:
        for k, p in enumerate(panels):
            r, c = divmod(k, ncol)
            placed.append((p, (c - (ncol - 1) / 2) * W, y0 - r * H))
        rows = -(-len(panels) // ncol)
        y0 -= rows * H + gap * H
    if placed:
        mid = (max(y for _, _, y in placed) + min(y for _, _, y in placed)) / 2
        placed = [(p, x, y - mid) for p, x, y in placed]
    return placed


def run_grid(session, manifest):
    from chimerax.core.commands import run
    from chimerax.core.models import Model
    from chimerax.geometry import Place, rotation, translation
    from chimerax.map import volume_from_grid_data
    from chimerax.map.volume import set_initial_region_and_style
    from chimerax.map_data import ArrayGridData

    base = _here()
    cache = {}

    def cube(name):
        if name not in cache:
            cache[name] = read_cube(os.path.join(base, name))
        return cache[name]

    geo = cube(manifest["geometry"])
    Z, X = geo["Z"], geo["X"]
    c = X.mean(0)
    w, v = np.linalg.eigh((X - c).T @ (X - c))
    ez = v[:, 0]
    heavy = max(range(len(Z)), key=lambda i: (Z[i], np.linalg.norm(X[i] - c)))
    up = X[heavy] - c
    up -= up.dot(ez) * ez
    ey = up / np.linalg.norm(up) if np.linalg.norm(up) > 0.5 else v[:, 2]
    ex = np.cross(ey, ez)
    turn = float(manifest.get("turn", 25.0))
    R = rotation(ey, turn).matrix[:, :3]
    Xr = (X - c) @ R.T
    P = np.c_[Xr @ ex, Xr @ ey]
    W = (P[:, 0].max() - P[:, 0].min()) + 6.0
    H = (P[:, 1].max() - P[:, 1].min()) + 7.5
    low = int(np.argmin(P[:, 1]))

    groups = manifest["groups"]
    layout = _layout([g["panels"] for g in groups], int(manifest.get("columns", 3)), W, H)

    tmp = _xyz_file(Z, X)
    root = Model(manifest.get("title", "Orbital grid"), session)
    session.models.add([root])
    if manifest.get("global_style", True):
        for cmd in PRESET_GLOBAL:
            run(session, cmd, log=False)

    for group in groups:
        gm = Model(group["name"], session)
        root.add([gm])
        for p, dx, dy in [t for t in layout if t[0] in group["panels"]]:
            g = cube(p["file"])
            comp = g["data"][..., p["component"]] if p.get("component") is not None else g["data"][..., 0]
            arr = np.ascontiguousarray(comp.transpose(2, 1, 0), dtype=np.float32)
            grid = ArrayGridData(arr, origin=tuple(g["origin"]), step=tuple(np.diag(g["steps"])),
                                 name=p["label"])
            grid.polar_values = True
            vol = volume_from_grid_data(grid, session, style="surface", open_model=False, show_dialog=False)
            set_initial_region_and_style(vol)
            vol.name = p["label"]
            mols, _ = session.open_command.open_data(tmp, format="xyz")
            mol = mols[0]
            mol.name = f"geometry for {p['label']}"
            pair = Model(p["label"], session)
            pair.add([vol, mol])
            pair.position = translation(dx * ex + dy * ey) * rotation(ey, turn, center=c)
            gm.add([pair])
            iso = float(p["iso"])
            run(session, f"volume #{vol.id_string} style surface level {-iso} color {NEG_COLOR} "
                         f"level {iso} color {POS_COLOR} transparency {manifest.get('transparency', 0.25)}",
                log=False)
            height = 0.9
            ox = -0.47 * height * len(p["label"]) / 2 - P[low, 0]
            run(session, f'label #{mol.id_string}@@serial_number={low + 1} atoms text "{p["label"]}" '
                         f"height {height} color {INK} offset {ox:.2f},-3.2,0", log=False)

    centre = c + np.mean([dx * ex + dy * ey for _, dx, dy in layout], axis=0)
    session.main_view.camera.position = Place(axes=[ex, ey, ez], origin=centre + 150 * ez)
    run(session, f"view #{root.id_string} pad 0.05", log=False)

    for i, t in enumerate(manifest.get("caption", [])):
        run(session, f'2dlabels text "{t}" xpos 0.01 ypos {0.975 - 0.027 * i:.3f} size 13 color {INK}',
            log=False)
    session.logger.info(f"orbitalgrid: {sum(len(g['panels']) for g in groups)} panels in "
                        f"#{root.id_string}")
    try:
        os.remove(tmp)
    except OSError:
        pass
"""


def write_grid_script(path, manifest: Dict[str, Any]) -> Path:
    """Write the self-contained ChimeraX script ``path`` for the given manifest.

    manifest keys:
      title, geometry (cube file with the atoms), columns, turn, transparency, global_style,
      caption (list of lines), groups: [{"name": str, "panels": [{"file", "component" (int or None),
      "label", "iso"}]}].
    All file names are relative to the script's folder.
    """
    path = Path(path)
    header = [
        "# FROG orbital / difference-density grid for ChimeraX.",
        "# Open in ChimeraX with:  open " + path.name,
        "import json as _json",
        "MANIFEST = _json.loads(" + json.dumps(json.dumps(manifest, ensure_ascii=False)) + ")",
    ]
    footer = [
        "",
        "run_grid(session, MANIFEST)  # noqa: F821  (session is defined by ChimeraX)",
    ]
    path.write_text("\n".join(header) + "\n" + VIEWER_SOURCE + "\n".join(footer) + "\n", encoding="utf-8")
    return path


def panel(file: str, label: str, iso: float, component: "int | None" = None) -> Dict[str, Any]:
    """One panel of a grid: a cube file (or one component of a multicube) with its label."""
    return {"file": file, "component": component, "label": label, "iso": iso}


def group(name: str, panels: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"name": name, "panels": panels}
