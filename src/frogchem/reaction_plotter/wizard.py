"""Interactive TUI Wizard for building FROG reaction profile input files.

Features 1-click presets, automatic branch padding (no manual None counting),
support for any delimiter (space, comma, tab, semicolon), and selective TS labeling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from InquirerPy import inquirer
from InquirerPy.validator import EmptyInputValidator
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def _parse_tokens(text: str) -> list[str]:
    """Split input string on semicolons, commas, tabs, spaces or newlines."""
    if ";" in text:
        parts = [p.strip() for p in text.split(";") if p.strip()]
    elif "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in text.split() if p.strip()]
    return parts


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


def run_tui_wizard() -> Optional[Path]:
    """Run interactive terminal wizard to create and optionally preview a reaction profile."""
    console.print(
        Panel.fit(
            "[bold cyan]🧪 FROG Reaction Profile Builder (TUI Wizard)[/bold cyan]\n"
            "[dim]Easily construct multi-pathway PES, branches, labels, and styling[/dim]",
            border_style="cyan",
        )
    )

    # 1. Preset or Custom
    start_mode = inquirer.select(
        message="Choose a starting template:",
        choices=[
            {"name": "🆕 Custom / Blank profile", "value": "custom"},
            {"name": "☀️ Photochemical Mechanism (S0 -> S1, MECP, HAT/SET branches)", "value": "photo"},
            {"name": "🔄 Catalytic Cycle (R -> TS1 -> Int -> TS2 -> P)", "value": "catalytic"},
            {"name": "🌿 Competitive Pathways (Major vs Minor TS)", "value": "competitive"},
        ],
        default="custom",
    ).execute()

    if start_mode == "photo":
        title = "Computed Photochemical Mechanism"
        units = "kcal/mol"
        has_photo = True
        show_xticks = False
        fig_w, fig_h = 12.0, 7.0
        pt_type = "hollow"
        pathways = [
            {
                "name": "Trunk",
                "pes": ["0.0", "71.7", "69.4", "70.5", "74.5", "42.0", "None", "None", "None", "None"],
                "labels": [
                    r"$\mathbf{1a}$-Br $\mathbf{S_0}$",
                    r"$\mathbf{1a}$-Br $\mathbf{S_1}$",
                    r"$\mathbf{I}$",
                    r"$\mathbf{I'}$",
                    r"$\mathbf{TS\text{-}I}$",
                    r"$\mathbf{II}$",
                    r"$\mathbf{MECP\text{-}SET}$",
                    r"$\mathbf{IV}$",
                    r"$\mathbf{TS\text{-}IV}$",
                    r"$\mathbf{2a\text{-}HBr}$",
                ],
                "style": "solid",
                "color": "#103E84",
            },
            {
                "name": r"$\mathit{SET\ pathway}$",
                "pes": ["None", "None", "None", "None", "None", "42.0", "47.1", "13.9", "16.6", "-24.1"],
                "labels": [
                    "None", "None", "None", "None", "None", "None",
                    r"$\mathbf{MECP\text{-}SET}$", "None", "None", "None",
                ],
                "style": "solid",
                "color": "#1F56A3",
            },
            {
                "name": r"$\mathit{HAT\ pathway}$",
                "pes": ["None", "None", "None", "None", "None", "42.0", "46.9", "39.2", "16.6", "None"],
                "labels": [
                    "None", "None", "None", "None", "None", "None",
                    r"$\mathbf{TS\text{-}II\text{-}HAT}$", r"$\mathbf{III}$", "None", "None",
                ],
                "style": "dashed",
                "color": "#58A4E0",
            },
        ]
    elif start_mode == "catalytic":
        title = "Catalytic Cycle Energy Profile"
        units = "kcal/mol"
        has_photo = False
        show_xticks = True
        fig_w, fig_h = 10.0, 6.0
        pt_type = "bar"
        pathways = [
            {
                "name": "Catalytic Pathway",
                "pes": ["0.0", "22.4", "8.2", "19.5", "-14.2"],
                "labels": ["Reactants", "TS1", "Intermediate", "TS2", "Products"],
                "style": "solid",
                "color": "#103E84",
            }
        ]
    elif start_mode == "competitive":
        title = "Bifurcating Reaction Profile"
        units = "kcal/mol"
        has_photo = False
        show_xticks = True
        fig_w, fig_h = 10.5, 6.5
        pt_type = "hollow"
        pathways = [
            {
                "name": "Major Pathway",
                "pes": ["0.0", "5.2", "16.4", "-12.0"],
                "labels": ["Reactants", "Intermediate", "TS-Major", "Products"],
                "style": "solid",
                "color": "#103E84",
            },
            {
                "name": "Minor Pathway",
                "pes": ["0.0", "5.2", "21.8", "-4.5"],
                "labels": ["Reactants", "Intermediate", "TS-Minor", "Products"],
                "style": "dashed",
                "color": "#E05858",
            },
        ]
    else:
        # Custom setup
        title = inquirer.text(
            message="Plot title:",
            default="Reaction Energy Profile",
        ).execute()

        units = inquirer.select(
            message="Energy units:",
            choices=["kcal/mol", "kJ/mol", "eV", "Hartree"],
            default="kcal/mol",
        ).execute()

        has_photo = inquirer.confirm(
            message="Include photochemical vertical excitation (two points sharing x=0)?",
            default=False,
        ).execute()

        show_xticks = inquirer.confirm(
            message="Show X-axis tick marks?",
            default=False if has_photo else True,
        ).execute()

        pt_type = inquirer.select(
            message="Point marker style:",
            choices=["hollow", "bar", "dot", "none"],
            default="hollow",
        ).execute()

        default_colors = ["#103E84", "#1F56A3", "#58A4E0", "#E05858", "#3DAA65", "#9B59B6"]
        default_styles = ["solid", "dashed", "dotted", "dashdot"]
        pathways = []

        n_paths = int(
            inquirer.number(
                message="Number of pathways / branches to define:",
                default=1,
                min_allowed=1,
                max_allowed=10,
            ).execute()
        )

        for i in range(n_paths):
            console.print(f"\n[bold yellow]── Pathway {i + 1} of {n_paths} ──[/bold yellow]")
            name = inquirer.text(
                message=f"Name of Pathway {i + 1} (supports $\\mathit{{...}}$):",
                default=f"Pathway {i + 1}" if n_paths > 1 else "Main Path",
                validate=EmptyInputValidator("Name cannot be empty"),
            ).execute()

            branch_offset = 0
            if i > 0 and len(pathways) > 0:
                max_prev_len = max(len(p["pes"]) for p in pathways)
                is_branch = inquirer.confirm(
                    message=f"Does '{name}' branch from a previous step? (auto-pads 'None')",
                    default=True,
                ).execute()
                if is_branch:
                    branch_offset = int(
                        inquirer.number(
                            message=f"Step index where branch begins (0 to {max_prev_len - 1}):",
                            default=min(5, max_prev_len - 1),
                            min_allowed=0,
                            max_allowed=max_prev_len - 1,
                        ).execute()
                    )

            raw_pes = inquirer.text(
                message=f"Energies for '{name}' (paste separated by space, comma, tab, or semicolon):",
                instruction="e.g. 42.0  47.1  13.9  16.6  -24.1",
            ).execute()

            pes_tokens = _parse_tokens(raw_pes)
            cleaned_pes = [
                "None" if t.lower() in ("none", "nan", "-", "_") else str(round(float(t), 2))
                for t in pes_tokens
            ]

            # Prepend branch offset if specified
            if branch_offset > 0:
                cleaned_pes = ["None"] * branch_offset + cleaned_pes

            raw_labels = inquirer.text(
                message=f"Species labels for '{name}' (optional, separated by ';'):",
                instruction="Leave empty to use generic step names",
                default="",
            ).execute()

            if raw_labels.strip():
                lbl_tokens = [t.strip() for t in raw_labels.split(";")]
                cleaned_lbls = ["None" if (not t or t.lower() in ("none", "-")) else t for t in lbl_tokens]
            else:
                cleaned_lbls = ["None"] * len(cleaned_pes)

            if len(cleaned_lbls) < len(cleaned_pes):
                cleaned_lbls.extend(["None"] * (len(cleaned_pes) - len(cleaned_lbls)))

            p_style = inquirer.select(
                message=f"Line style for '{name}':",
                choices=["solid", "dashed", "dotted", "dashdot"],
                default=default_styles[i % len(default_styles)],
            ).execute()

            p_color = inquirer.text(
                message=f"Color (hex, name) for '{name}':",
                default=default_colors[i % len(default_colors)],
            ).execute()

            pathways.append({
                "name": name,
                "pes": cleaned_pes,
                "labels": cleaned_lbls,
                "style": p_style,
                "color": p_color,
            })

        fig_w = float(inquirer.text(message="Figure width:", default="12.0").execute())
        fig_h = float(inquirer.text(message="Figure height:", default="7.0").execute())

    # Align max length
    max_len = max(len(p["pes"]) for p in pathways)
    for p in pathways:
        if len(p["pes"]) < max_len:
            p["pes"].extend(["None"] * (max_len - len(p["pes"])))
        if len(p["labels"]) < max_len:
            p["labels"].extend(["None"] * (max_len - len(p["labels"])))

    # Coordinates
    photo_step = 0
    if has_photo:
        if max_len > 2 and start_mode == "custom":
            photo_step = int(
                inquirer.number(
                    message=f"Step index where vertical excitation begins (0 to {max_len - 2}):",
                    default=0,
                    min_allowed=0,
                    max_allowed=max_len - 2,
                ).execute()
            )
    x_vals = compute_x_coords(max_len, has_photo, photo_step)

    # 4. Selective TS / MECP Labeling
    console.print("\n[bold yellow]── Selective TS / MECP Labeling ──[/bold yellow]")
    pt_choices = []
    default_checked = []
    for step_idx in range(max_len):
        names_at_step = []
        energies_at_step = []
        for p in pathways:
            val = p["pes"][step_idx]
            lbl = p["labels"][step_idx]
            if val != "None":
                energies_at_step.append(val)
            if lbl != "None":
                names_at_step.append(lbl)
        desc_name = "/".join(names_at_step) if names_at_step else f"Step {step_idx}"
        desc_val = "/".join(energies_at_step) if energies_at_step else "None"
        x_coord = x_vals[step_idx] if step_idx < len(x_vals) else step_idx

        # Auto-detect if name has TS or MECP
        is_ts_name = "ts" in desc_name.lower() or "mecp" in desc_name.lower()
        if is_ts_name:
            default_checked.append(step_idx)

        pt_choices.append({
            "name": f"Step {step_idx} (x={x_coord}): {desc_name} ({desc_val} {units})",
            "value": step_idx,
            "enabled": is_ts_name,
        })

    chosen_steps = inquirer.checkbox(
        message="Select which points should show numerical energy labels (Space to toggle, Enter to confirm):",
        choices=pt_choices,
    ).execute()

    # 5. Build Content
    lines = ["--# PES"]
    for p in pathways:
        lines.append("; ".join(p["pes"]))

    lines.append("\n--# LABELS")
    for p in pathways:
        lines.append("; ".join(p["labels"]))

    lines.append("\n--# GRAPHS")
    lines.append(f"title : {title}")
    lines.append(f"label : {'; '.join(p['name'] for p in pathways)}")
    lines.append(f"x : {'; '.join(str(x) for x in x_vals)}")
    lines.append(f"colors : {'; '.join(p['color'] for p in pathways)}")
    lines.append(f"style : {'; '.join(p['style'] for p in pathways)}")
    lines.append(f"figsize : {fig_w}; {fig_h}")
    lines.append(f"units : {units}")
    lines.append("labels : true")
    if chosen_steps:
        lines.append(f"labelpoint : {'; '.join(str(idx) for idx in chosen_steps)}")
    if not show_xticks:
        lines.append("xticks : false")
    lines.append("save : false\n")

    content = "\n".join(lines)

    # 6. Syntax Preview & Save
    console.print("\n[bold green]✔ Input File Generated Successfully![/bold green]")
    console.print(Panel(Syntax(content, "ini", theme="monokai", line_numbers=True), title="input.txt Preview"))

    save_path_str = inquirer.text(
        message="Save path for input file:",
        default="input.txt",
    ).execute()

    save_path = Path(save_path_str).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(content, encoding="utf-8")
    console.print(f"[bold green]Saved to: {save_path}[/bold green]")

    plot_now = inquirer.confirm(
        message="Generate and preview plot right now?",
        default=True,
    ).execute()

    if plot_now:
        from frogchem.reaction_plotter.profile import ReactionProfileAnalyzer
        analyzer = ReactionProfileAnalyzer(str(save_path), auto_plot=False)
        out_img = save_path.with_suffix(".png")
        analyzer.plot(
            filename=str(save_path.with_suffix("")),
            file_format="png",
            dpi=300,
            point_type=pt_type,
        )
        console.print(f"[bold cyan]Plot image saved to: {out_img}[/bold cyan]")
        import subprocess
        try:
            subprocess.run(["open", str(out_img)], check=False)
        except Exception:
            pass

    return save_path


if __name__ == "__main__":
    run_tui_wizard()
