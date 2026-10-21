import logging

from frogchem.exceptions import ParseError, ErrorManifest

logger = logging.getLogger(__name__)


class BaseParser:
    """Base class for lazy-parsing output files.

    Subclasses must implement _parse_file() which is called once on first
    attribute access.
    """

    def __init__(self, fname):
        self.fname = fname
        self._parsed = False
        self.errors = ErrorManifest()

    def __getattr__(self, name):
        if name.startswith('_') or name in ('fname', 'reg'):
            raise AttributeError(name)
        self._ensure_parsed()
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f'{type(self).__name__} has no attribute {name!r}')

    def _ensure_parsed(self):
        if self._parsed:
            return
        self._parsed = True

        fname_str = str(self.fname)
        if fname_str.endswith('.gz'):
            import gzip
            with gzip.open(self.fname, 'rt', encoding='utf-8', errors='replace') as f:
                self.fl = f.read()
        else:
            with open(self.fname, 'r', encoding='utf-8', errors='replace') as f:
                self.fl = f.read()

        try:
            self._parse_file()
        except ParseError:
            raise
        except Exception as e:
            raise ParseError(f"Failed to parse {self.fname}: {e}") from e

    def _parse_file(self):
        raise NotImplementedError
