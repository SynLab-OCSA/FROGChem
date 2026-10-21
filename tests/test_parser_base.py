import pytest
import tempfile
import os
from frogchem.parser_base import BaseParser
from frogchem.exceptions import ParseError


class DummyParser(BaseParser):
    """Test parser that sets an attribute in _parse_file."""

    def _parse_file(self):
        self.parsed = True
        self.data = "hello"


class FailingParser(BaseParser):
    """Test parser that raises in _parse_file."""

    def _parse_file(self):
        raise ValueError("bad format")


class TestBaseParser:
    def test_lazy_parse(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = DummyParser(str(f))
        assert p._parsed is False
        assert p.parsed is True
        assert p.data == "hello"
        assert p._parsed is True

    def test_no_double_parse(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = DummyParser(str(f))
        _ = p.parsed
        _ = p.parsed  # second access should not re-parse

    def test_missing_attribute(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = DummyParser(str(f))
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = p.nonexistent

    def test_private_attrs_not_parsed(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = DummyParser(str(f))
        with pytest.raises(AttributeError):
            _ = p._private

    def test_fname_accessible(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = DummyParser(str(f))
        assert p.fname == str(f)

    def test_parse_error_on_failure(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("some content")
        p = FailingParser(str(f))
        with pytest.raises(ParseError, match="Failed to parse"):
            _ = p.data

    def test_file_contents_available(self, tmp_path):
        f = tmp_path / "test.out"
        f.write_text("line1\nline2\n")
        p = DummyParser(str(f))
        _ = p.parsed
        assert "line1" in p.fl
        assert "line2" in p.fl
