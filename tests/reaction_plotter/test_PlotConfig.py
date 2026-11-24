from frogchem.reaction_plotter.plotter import PlotConfig


class TestPlotConfigDefaults:
    def test_default_values(self):
        config = PlotConfig()
        assert config.label == []
        assert config.colors == [None]
        assert config.legend is False
        assert config.labelpoint is None
        assert config.title == "Reaction Energy Path"
        assert config.style == ['-']
        assert config.decimal_digits == 1
        assert config.position_sections == 'bottom'
        assert config.desaturate is False
        assert config.transparent is False
        assert config.bg_color is None
