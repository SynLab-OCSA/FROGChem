import pytest
from frogchem.exceptions import FrogError, ParseError, InputError, ConfigError, report


class TestExceptionHierarchy:
    def test_parse_error_is_frog_error(self):
        assert issubclass(ParseError, FrogError)

    def test_input_error_is_frog_error(self):
        assert issubclass(InputError, FrogError)

    def test_config_error_is_frog_error(self):
        assert issubclass(ConfigError, FrogError)

    def test_frog_error_is_exception(self):
        assert issubclass(FrogError, Exception)

    def test_parse_error_message(self):
        err = ParseError("bad file out.out")
        assert str(err) == "bad file out.out"

    def test_exception_cause_chain(self):
        with pytest.raises(ParseError) as exc_info:
            try:
                raise ValueError("original")
            except ValueError as e:
                raise ParseError("wrapped") from e
        assert exc_info.value.__cause__.__class__ is ValueError


class TestReport:
    def test_report_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            report(FrogError("test error"))
        assert exc_info.value.code == 1

    def test_report_generic_exception(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            report(RuntimeError("something broke"))
        assert exc_info.value.code == 1
