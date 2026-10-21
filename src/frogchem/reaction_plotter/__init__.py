# ponytail: lazy-load plotting classes to avoid eager matplotlib import
_EXPORT_MAP = {
    "ReactionPlotter": ("frogchem.reaction_plotter.plotter", "ReactionPlotter"),
    "ReactionProfilePlotter": ("frogchem.reaction_plotter.plotter", "ReactionProfilePlotter"),
    "PlotConfig": ("frogchem.reaction_plotter.plotter", "PlotConfig"),
    "ReactionProfileAnalyzer": ("frogchem.reaction_plotter.profile", "ReactionProfileAnalyzer"),
    "plot_reaction_profile": ("frogchem.reaction_plotter.plotter", "plot_reaction_profile"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        import importlib
        mod_name, attr_name = _EXPORT_MAP[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
