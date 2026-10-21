import pytest
from frogchem.reaction_plotter.plotter import PlotConfig


class TestPlotConfigDefaults:
    def test_default_values(self):
        config = PlotConfig()
        assert config.label == []
        assert config.colors == [None]
        assert config.legend is False
        assert config.save is False
        assert config.relative is False
        assert config.labelpoint is None
        assert config.zero == [0]
        assert config.title == "Reaction Energy Path"
        assert config.style == ['-']
        assert config.linewidth == [1.5]
        assert config.ypad == [0.0]
        assert config.y_limit is None
        assert config.directory == "reaction"
        assert config.decimal_digits == 1


class TestPlotConfigFromDict:
    def test_bool_conversion(self):
        config = PlotConfig.from_dict({'legend': ['true'], 'save': ['true']})
        assert config.legend is True
        assert config.save is True

    def test_bool_false(self):
        config = PlotConfig.from_dict({'legend': ['false']})
        assert config.legend is False

    def test_relative_energies(self):
        config = PlotConfig.from_dict({'relative': ['true']})
        assert config.relative is True

    def test_labelpoint(self):
        config = PlotConfig.from_dict({'labelpoint': ['1', '3', '5']})
        assert config.labelpoint == [1, 3, 5]

    def test_labelpoint_none(self):
        config = PlotConfig.from_dict({'labelpoint': ['None']})
        assert config.labelpoint is None

    def test_zero_list(self):
        config = PlotConfig.from_dict({'zero': ['0', '1']})
        assert config.zero == [0, 1]

    def test_title(self):
        config = PlotConfig.from_dict({'title': ['Custom Title']})
        assert config.title == "Custom Title"

    def test_title_default(self):
        config = PlotConfig.from_dict({})
        assert config.title == "Reaction Energy Path"

    def test_linewidth(self):
        config = PlotConfig.from_dict({'linewidth': ['2.0', '1.5']})
        assert config.linewidth == [2.0, 1.5]

    def test_ypad(self):
        config = PlotConfig.from_dict({'ypad': ['1.0', '2.0']})
        assert config.ypad == [1.0, 2.0]

    def test_y_limit(self):
        config = PlotConfig.from_dict({'y_limit': ['10.0']})
        assert config.y_limit == 10.0

    def test_y_limit_none(self):
        config = PlotConfig.from_dict({'y_limit': ['None']})
        assert config.y_limit is None

    def test_decimal_digits(self):
        config = PlotConfig.from_dict({'decimal_digits': ['3']})
        assert config.decimal_digits == 3

    def test_decimal_digits_default(self):
        config = PlotConfig.from_dict({})
        assert config.decimal_digits == 1

    def test_color_string(self):
        config = PlotConfig.from_dict({'colors': ['red', 'blue']})
        assert config.colors == ['red', 'blue']

    def test_style_list(self):
        config = PlotConfig.from_dict({'style': ['--', ':']})
        assert config.style == ['--', ':']

    def test_invalid_value_warns(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = PlotConfig.from_dict({'decimal_digits': ['not_a_number']})
            assert config.decimal_digits == 1

    def test_position_sections_default(self):
        config = PlotConfig.from_dict({})
        assert config.position_sections == 'bottom'

    def test_empty_data(self):
        config = PlotConfig.from_dict({})
        assert isinstance(config, PlotConfig)

    def test_color_singular_alias(self):
        config = PlotConfig.from_dict({'color': ['#353535']})
        assert config.colors == ['#353535']

    def test_desaturate(self):
        config = PlotConfig.from_dict({'desaturate': ['true']})
        assert config.desaturate is True
        config_default = PlotConfig.from_dict({})
        assert config_default.desaturate is False

    def test_transparent_and_bg_color(self):
        config = PlotConfig.from_dict({'transparent': ['true'], 'bg_color': ['#1E1E1E']})
        assert config.transparent is True
        assert config.bg_color == '#1E1E1E'

        config_alias = PlotConfig.from_dict({'transparent_bg': ['true']})
        assert config_alias.transparent is True

        config_def = PlotConfig()
        assert config_def.transparent is False
        assert config_def.bg_color is None
