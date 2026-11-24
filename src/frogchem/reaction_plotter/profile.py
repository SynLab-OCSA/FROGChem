"""Parse and plot reaction profiles from YAML input (see the Reaction Profiles user guide)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from frogchem.reaction_plotter.plotter import PlotConfig, ReactionProfilePlotter

logger = logging.getLogger(__name__)


class ReactionProfileAnalyzer:
    """Parse a YAML reaction profile (file or string) and plot it with ``plot()``."""

    def __init__(
        self,
        filename: Optional[Union[str, Path]] = None,
        content: Optional[str] = None,
    ):
        self.filename: Optional[Path] = Path(filename) if filename is not None else None
        self.file_content: str = content or ""
        self.pes_data: List[np.ndarray] = []
        self.labels: List[List[str]] = []
        self.config: PlotConfig = PlotConfig()

        # High-level structures for plotprofile engine
        self.pathways: Dict[str, Any] = {}
        self.energy_data: Dict[str, List[Optional[float]]] = {}
        self.point_labels: Dict[str, List[Optional[str]]] = {}

        if self.filename is not None:
            if not self.filename.exists():
                raise FileNotFoundError(f"Input file '{self.filename}' not found")
            self.file_content = self.filename.read_text(encoding="utf-8")
            self._process_content()
        elif self.file_content:
            self._process_content()

    # ── Processing & Parsing ────────────────────────────────────────────────

    def _process_content(self) -> None:
        """Parse the content as YAML (JSON following the same schema also loads)."""
        if not self.file_content.strip():
            raise ValueError("Input content is empty")
        try:
            data = yaml.safe_load(self.file_content)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Input must be a YAML mapping (steps:, pathways:, ...)")
        self._parse_yaml_dict(data)

    def _parse_yaml_dict(self, data: dict[str, Any]) -> None:
        """Parse structured YAML/dictionary configuration into profile data structures."""
        # 1. Title, units, styles, etc.
        title = data.get("title", "Reaction Energy Profile")
        units = data.get("units", "kcal/mol")
        point_type = data.get("point_type", data.get("marker", "hollow"))
        figsize = data.get("figsize", (10.0, 7.0))
        if isinstance(figsize, (list, tuple)) and len(figsize) == 2:
            figsize = (float(figsize[0]), float(figsize[1]))
        curviness = data.get("curviness", 0.42)
        show_xticks = data.get("show_xticks", data.get("xticks", True))

        self.config.title = str(title)
        self.config.units = str(units)
        self.config.point_type = str(point_type)
        self.config.figsize = figsize
        self.config.curviness = float(curviness) if curviness is not None else 0.42
        self.config.show_xticks = bool(show_xticks)
        self.config.xticks = bool(show_xticks)
        show_point_labels = data.get("show_point_labels", None)
        if show_point_labels is not None:
            self.config.show_point_labels = bool(show_point_labels)
        font_size = data.get("font_size", data.get("fontsize", None))
        if font_size is not None:
            try:
                self.config.font_size = int(font_size)
            except (ValueError, TypeError):
                pass

        # 2. Steps list
        global_steps = data.get("steps", [])
        if not isinstance(global_steps, list):
            global_steps = []
        global_steps = [str(s) if s is not None else None for s in global_steps]

        # 3. Pathways
        raw_pathways = data.get("pathways", {})
        pathway_items: list[tuple[str, dict[str, Any]]] = []
        if isinstance(raw_pathways, dict):
            for p_name, p_val in raw_pathways.items():
                if isinstance(p_val, dict):
                    pathway_items.append((str(p_name), p_val))
                elif isinstance(p_val, list):
                    pathway_items.append((str(p_name), {"energies": p_val}))
        elif isinstance(raw_pathways, list):
            for p_item in raw_pathways:
                if isinstance(p_item, dict):
                    p_name = p_item.get("name", f"Pathway {len(pathway_items) + 1}")
                    pathway_items.append((str(p_name), p_item))

        if not pathway_items and "energies" in data and isinstance(data["energies"], dict):
            for p_name, p_vals in data["energies"].items():
                pathway_items.append((str(p_name), {"energies": p_vals}))

        self.energy_data = {}
        self.point_labels = {}
        colors_list = []
        styles_list = []

        default_palette = ["#103E84", "#1F56A3", "#58A4E0", "#E05858", "#3DAA65", "#9B59B6", "#E67E22"]

        for idx, (p_name, p_info) in enumerate(pathway_items):
            raw_energies = list(p_info.get("energies", []))

            # Handle branch_at / branch_from
            branch_at = p_info.get("branch_at", p_info.get("branch_from", None))
            offset = 0
            if branch_at is not None:
                if isinstance(branch_at, int):
                    offset = max(0, branch_at)
                elif isinstance(branch_at, str):
                    try:
                        offset = global_steps.index(branch_at)
                    except ValueError:
                        matched = False
                        for s_i, s_name in enumerate(global_steps):
                            if s_name and (branch_at.lower() == str(s_name).lower() or branch_at in str(s_name)):
                                offset = s_i
                                matched = True
                                break
                        if not matched:
                            offset = 0

            energies: list[Optional[float]] = [None] * offset + [
                float(e) if e is not None and str(e).lower() not in ("none", "nan", "-") else None
                for e in raw_energies
            ]

            # Pathway-specific labels or overrides
            p_labels = p_info.get("labels", p_info.get("steps", None))
            if p_labels is None and isinstance(data.get("steps"), dict):
                p_labels = data["steps"].get(p_name, None)

            if p_labels is None:
                labels = [None] * offset + list(global_steps[offset:])
            elif isinstance(p_labels, dict):
                labels = [None] * offset + list(global_steps[offset:])
                for k, v in p_labels.items():
                    val_str = str(v) if v is not None else None
                    if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                        k_idx = int(k)
                        if 0 <= k_idx < len(labels):
                            labels[k_idx] = val_str
                    elif isinstance(k, str):
                        matched_k = False
                        for g_i, g_name in enumerate(global_steps):
                            if g_name and (k == str(g_name).strip() or k.lower() == str(g_name).lower().strip() or k in str(g_name)):
                                labels[g_i] = val_str
                                matched_k = True
                                break
                        if not matched_k:
                            logger.warning("Pathway '%s': label key '%s' matches no step; ignored", p_name, k)
            elif isinstance(p_labels, list):
                raw_l = [str(l) if l is not None else None for l in p_labels]
                if len(raw_l) == len(raw_energies):
                    labels = [None] * offset + raw_l
                elif len(raw_l) >= len(global_steps):
                    labels = raw_l
                else:
                    labels = [None] * offset + raw_l

            self.energy_data[p_name] = energies
            self.point_labels[p_name] = labels
            colors_list.append(p_info.get("color", default_palette[idx % len(default_palette)]))
            styles_list.append(p_info.get("style", "solid"))

        # Align lengths
        max_len = max((len(e) for e in self.energy_data.values()), default=len(global_steps))
        if len(global_steps) < max_len:
            global_steps.extend([None] * (max_len - len(global_steps)))

        for p_name in self.energy_data:
            e_len = len(self.energy_data[p_name])
            if e_len < max_len:
                self.energy_data[p_name].extend([None] * (max_len - e_len))
            l_len = len(self.point_labels[p_name])
            if l_len < max_len:
                self.point_labels[p_name].extend([None] * (max_len - l_len))

        # 4. Excitation & X coordinates
        excitation = data.get("excitation", None)
        if "x" in data and data["x"] is not None:
            self.config.x = [float(x) for x in data["x"]]
        elif excitation is not None:
            photo_step = 0
            if isinstance(excitation, int):
                photo_step = excitation
            elif isinstance(excitation, str):
                try:
                    photo_step = global_steps.index(excitation)
                except ValueError:
                    photo_step = 0
            photo_step = max(0, min(photo_step, max_len - 2)) if max_len > 1 else 0
            xs = []
            cur = 0
            for i in range(max_len):
                if i == photo_step + 1:
                    xs.append(xs[-1])
                else:
                    if i > 0 and i != photo_step + 1:
                        cur += 1
                    xs.append(cur)
            self.config.x = xs
            if "show_xticks" not in data and "xticks" not in data:
                self.config.show_xticks = False
                self.config.xticks = False
        else:
            self.config.x = list(range(max_len))

        # 5. Label points
        raw_labelpoints = data.get("label_points", data.get("labelpoint", None))
        if raw_labelpoints is not None:
            if isinstance(raw_labelpoints, (int, str)):
                raw_labelpoints = [raw_labelpoints]
            resolved_lp = []
            for lp in raw_labelpoints:
                if isinstance(lp, int):
                    resolved_lp.append(lp)
                elif isinstance(lp, str):
                    if lp.isdigit():
                        resolved_lp.append(int(lp))
                    else:
                        for s_i, s_name in enumerate(global_steps):
                            if s_name and (lp == s_name or lp.lower() == str(s_name).lower() or lp in str(s_name)):
                                resolved_lp.append(s_i)
                                break
            self.config.labelpoint = resolved_lp

        # 6. Axes frame style
        raw_axes = data.get("axes", None)
        if raw_axes is not None:
            self.config.axes = str(raw_axes)

        # 7. Section brackets
        raw_sections = data.get("sections", None)
        raw_sec_start = data.get("sections_start", None)
        raw_sec_end = data.get("sections_end", None)
        if raw_sections is not None and raw_sec_start is not None and raw_sec_end is not None:
            if isinstance(raw_sections, list) and isinstance(raw_sec_start, list) and isinstance(raw_sec_end, list):
                if len(raw_sections) == len(raw_sec_start) == len(raw_sec_end):
                    self.config.sections = [str(s) for s in raw_sections]
                    self.config.sections_start = [int(i) for i in raw_sec_start]
                    self.config.sections_end = [int(i) for i in raw_sec_end]

        raw_sec_shift = data.get("sections_shift", None)
        if raw_sec_shift is not None:
            try:
                self.config.sections_shift = float(raw_sec_shift)
            except (TypeError, ValueError):
                pass

        # 8. Finalize config & arrays
        self.config.label = list(self.energy_data.keys())
        self.config.colors = colors_list
        self.config.style = styles_list
        if "legend" in data:
            self.config.legend = bool(data["legend"])
        elif "show_legend" in data:
            self.config.legend = bool(data["show_legend"])
        else:
            self.config.legend = len(self.energy_data) > 1

        self.pes_data = []
        self.labels = []
        for p_name in self.config.label:
            arr = np.array([e if e is not None else np.nan for e in self.energy_data[p_name]], dtype=float)
            self.pes_data.append(arr)
            self.labels.append([str(l) if l is not None else "" for l in self.point_labels[p_name]])

        self.pathways = {}
        for p_name in self.config.label:
            steps_list = []
            for s_idx, (e_val, l_val) in enumerate(zip(self.energy_data[p_name], self.point_labels[p_name])):
                step_name = l_val or (global_steps[s_idx] if s_idx < len(global_steps) and global_steps[s_idx] else "")
                steps_list.append({"name": step_name, "energy": e_val})
            self.pathways[p_name] = steps_list

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot(
        self,
        filename: Optional[str] = None,
        file_format: str = "png",
        dpi: int = 600,
        style: str = "default",
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot the reaction energy profile using ReactionProfilePlotter."""
        plotter_kwargs: dict[str, Any] = {
            "style": style,
        }
        if self.config.colors and self.config.colors != [None]:
            plotter_kwargs["colors"] = self.config.colors
        if self.config.style and self.config.label and len(self.config.style) == len(self.config.label):
            plotter_kwargs["linestyle"] = {lbl: st for lbl, st in zip(self.config.label, self.config.style)}
        if hasattr(self.config, 'decimal_digits') and self.config.decimal_digits is not None:
            plotter_kwargs["sig_figs"] = self.config.decimal_digits
        if hasattr(self.config, 'desaturate') and self.config.desaturate is not None:
            plotter_kwargs["desaturate"] = self.config.desaturate
        if hasattr(self.config, 'labelpoint') and self.config.labelpoint is not None:
            plotter_kwargs["labelpoint"] = self.config.labelpoint
        if hasattr(self.config, 'labels') and self.config.labels is not None:
            plotter_kwargs["labels"] = self.config.labels
        if hasattr(self.config, 'figsize') and self.config.figsize is not None:
            plotter_kwargs["figsize"] = self.config.figsize
        if getattr(self.config, 'point_type', None) is not None:
            plotter_kwargs.setdefault("point_type", self.config.point_type)
        if getattr(self.config, 'curviness', None) is not None:
            plotter_kwargs.setdefault("curviness", self.config.curviness)
        if getattr(self.config, 'axes', None) is not None:
            plotter_kwargs.setdefault("axes", self.config.axes)
        if getattr(self.config, 'legend', None) is not None:
            plotter_kwargs.setdefault("show_legend", bool(self.config.legend))
        if getattr(self.config, 'font_size', None) is not None:
            plotter_kwargs.setdefault("font_size", int(self.config.font_size))

        # Plot arguments
        has_common_labels = False
        if "point_labels" not in kwargs and self.labels and any(self.labels):
            pt_labels = self.point_labels
            if all(lbls == self.labels[0] for lbls in self.labels):
                has_common_labels = True
        else:
            pt_labels = kwargs.pop("point_labels", self.point_labels if any(self.point_labels.values()) else None)

        ann = kwargs.pop("annotations", None)

        # Merge style kwargs into plotter_kwargs
        from frogchem.reaction_plotter.plotter import _STYLE_KEYS
        for k in list(kwargs.keys()):
            if k in _STYLE_KEYS:
                plotter_kwargs[k] = kwargs.pop(k)

        if "transparent" not in plotter_kwargs and getattr(self.config, "transparent", False):
            plotter_kwargs["transparent"] = self.config.transparent
        if "bg_color" not in plotter_kwargs and getattr(self.config, "bg_color", None) is not None:
            plotter_kwargs["bg_color"] = self.config.bg_color

        legend_barriers = kwargs.pop(
            "legend_barriers",
            getattr(self.config, "legend_barriers", False),
        )
        if legend_barriers is None:
            legend_barriers = False

        energy_data_to_plot = self.energy_data
        if legend_barriers:
            sig_figs = getattr(self.config, "decimal_digits", 1)
            sig_figs = 1 if sig_figs is None else int(sig_figs)
            updated_data = {}
            name_mapping = {}
            for name, energies in self.energy_data.items():
                valid = [e for e in energies if e is not None]
                if len(valid) >= 3:
                    peak = max(valid[1:-1])
                    val_str = f"{peak:.{sig_figs}f}".replace("-", "−")
                    new_name = f"{name}  ({val_str})"
                    name_mapping[name] = new_name
                    updated_data[new_name] = energies
                else:
                    updated_data[name] = energies
            energy_data_to_plot = updated_data
            if "linestyle" in plotter_kwargs and isinstance(plotter_kwargs["linestyle"], dict):
                plotter_kwargs["linestyle"] = {
                    name_mapping.get(k, k): v for k, v in plotter_kwargs["linestyle"].items()
                }
            if isinstance(pt_labels, dict):
                pt_labels = {
                    name_mapping.get(k, k): v for k, v in pt_labels.items()
                }

        # Default labels to False if multiple series, common labels, or legend_barriers exist
        if "labels" not in plotter_kwargs:
            if legend_barriers or has_common_labels or len(self.energy_data) > 1:
                plotter_kwargs["labels"] = False

        if "x" not in kwargs and getattr(self.config, "x", None) is not None:
            kwargs["x"] = self.config.x

        show_xticks = kwargs.pop("show_xticks", kwargs.pop("xticks", None))
        if show_xticks is None:
            show_xticks = bool(getattr(self.config, "show_xticks", True) and getattr(self.config, "xticks", True))

        plot_point_labels = pt_labels
        show_point_labels = kwargs.pop("show_point_labels", getattr(self.config, "show_point_labels", None))
        if show_point_labels is False:
            plot_point_labels = None
        elif show_point_labels is True:
            plot_point_labels = pt_labels
        elif "point_labels" not in kwargs and show_xticks and (self.labels and self.labels[0] and any(self.labels[0])):
            plot_point_labels = None

        plotter = ReactionProfilePlotter(**plotter_kwargs)

        fig, ax = plotter.plot(
            energy_data=energy_data_to_plot,
            filename=None,
            annotations=ann,
            point_labels=plot_point_labels,
            file_format=file_format,
            dpi=dpi,
            **kwargs,
        )

        if not show_xticks or not (self.labels and self.labels[0] and any(self.labels[0])):
            if ax is not None:
                ax.set_xticks([])
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        elif kwargs.get("x") is not None and ax is not None and self.labels and any(self.labels):
            x_vals = kwargs["x"]
            unique_xs = sorted(set(x_vals))
            lbl_by_x: dict[float, list[str]] = {}
            for p_lbls in self.labels:
                for xi, lbl in zip(x_vals, p_lbls):
                    if lbl and str(lbl).strip() and str(lbl).strip().lower() not in ("none", ""):
                        s_lbl = str(lbl).strip()
                        if s_lbl not in lbl_by_x.setdefault(xi, []):
                            lbl_by_x[xi].append(s_lbl)
            tick_labels = [" / ".join(lbl_by_x[xi]) if (xi in lbl_by_x and lbl_by_x[xi]) else "" for xi in unique_xs]
            ax.set_xticks(unique_xs)
            cfg_fsize = getattr(self.config, 'font_size', 11)
            eff_fsize = kwargs.get("fontsize_ticks") or getattr(self.config, 'fontsize_ticks', cfg_fsize)
            if "fontsize_ticks" not in kwargs and eff_fsize == 10 and cfg_fsize != 11:
                eff_fsize = cfg_fsize
            font_wt = "bold" if eff_fsize > 8 else "normal"
            ax.set_xticklabels(tick_labels, fontsize=eff_fsize, fontweight=font_wt)
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, length=4, width=1.0)
            ax.spines['bottom'].set_visible(True)
        elif has_common_labels and ax is not None and self.labels:
            max_len = len(self.labels[0])
            ax.set_xticks(list(range(max_len)))
            cfg_fsize = getattr(self.config, 'font_size', 12)
            eff_fsize = kwargs.get("fontsize_ticks") or getattr(self.config, 'fontsize_ticks', cfg_fsize)
            if "fontsize_ticks" not in kwargs and eff_fsize == 10 and cfg_fsize != 11:
                eff_fsize = cfg_fsize
            font_wt = "bold" if eff_fsize > 8 else "normal"
            ax.set_xticklabels(self.labels[0], fontsize=eff_fsize, fontweight=font_wt)
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, length=4, width=1.0)
            ax.spines['bottom'].set_visible(True)

        if self.config.title and ax is not None:
            cfg_fsize = getattr(self.config, 'font_size', 11)
            t_fsize = kwargs.get("fontsize_title") or getattr(self.config, 'fontsize_title', cfg_fsize + 2)
            if "fontsize_title" not in kwargs and t_fsize == 12 and cfg_fsize != 11:
                t_fsize = cfg_fsize + 2
            ax.set_title(self.config.title, fontsize=t_fsize)

        # Draw section brackets (if configured via YAML sections/sections_start/sections_end)
        if ax is not None and self.config.sections and self.config.sections_start and self.config.sections_end:
            x_pos_list = list(getattr(self.config, "x", None) or range(
                max(len(v) for v in self.energy_data.values()) if self.energy_data else 0
            ))
            if x_pos_list:
                plotter.ax = ax
                plotter.add_section_arrows(
                    x_positions=x_pos_list,
                    sections=self.config.sections,
                    sections_start=self.config.sections_start,
                    sections_end=self.config.sections_end,
                    colors=self.config.colors_sections or ["black"],
                    position=getattr(self.config, "position_sections", "bottom"),
                    shift=float(getattr(self.config, "sections_shift", 0.05)),
                )

        # Background options to pass to savefig (also used by the CLI's shared save_figure)
        self.savefig_kwargs = {
            "transparent": plotter.transparent,
            "facecolor": "none" if plotter.transparent else plotter.bg_color,
        }
        if filename and fig is not None:
            fig.savefig(
                f"{filename}.{file_format}",
                format=file_format,
                dpi=dpi,
                bbox_inches="tight",
                **self.savefig_kwargs,
            )

        return fig, ax
