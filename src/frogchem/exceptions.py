class FrogError(Exception):
    """Base exception for all FROG errors."""


class ParseError(FrogError):
    """Raised when parsing an output file fails."""


class InputError(FrogError):
    """Raised when user input is invalid."""


def report(err: Exception, verbose: bool = False) -> None:
    """Unified error reporting. Replaces print/logger/print_error inconsistency."""
    from frogchem.printer import print_error
    if isinstance(err, FrogError):
        print_error(str(err), exit_code=1)
    else:
        print_error(f"{type(err).__name__}: {err}", exit_code=1)
