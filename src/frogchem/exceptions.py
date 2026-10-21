from dataclasses import dataclass, asdict
from typing import Optional


class FrogError(Exception):
    """Base exception for all FROG errors."""


#: Former name (before the rename paclab -> FROG), kept so existing ``except PaclabError`` still works.
PaclabError = FrogError


class ParseError(FrogError):
    """Raised when parsing an output file fails."""


class InputError(FrogError):
    """Raised when user input is invalid."""


class ConfigError(FrogError):
    """Raised when configuration is invalid or missing."""


@dataclass
class ErrorEntry:
    """A single non-fatal error collected during parsing."""
    file: str
    error_type: str
    message: str
    section: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class ErrorManifest:
    """Collects ErrorEntry instances during a parse run. Dump as JSON for batch summaries."""

    def __init__(self):
        self.entries: list[ErrorEntry] = []

    def add(self, file: str, error_type: str, message: str, section: str = None):
        self.entries.append(ErrorEntry(file, error_type, message, section))

    def has_errors(self):
        return len(self.entries) > 0

    def to_dicts(self):
        return [e.to_dict() for e in self.entries]


def report(err: Exception, verbose: bool = False) -> None:
    """Unified error reporting. Replaces print/logger/print_error inconsistency."""
    from frogchem.printer import print_error
    if isinstance(err, FrogError):
        print_error(str(err), exit_code=1)
    else:
        print_error(f"{type(err).__name__}: {err}", exit_code=1)
