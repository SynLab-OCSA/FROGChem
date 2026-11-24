"""
Modulo centralizzato per l'output su terminale usando Rich.

stdout carries data only (tables, JSON, CSV); banners, progress, saved-file notices,
warnings and errors go to stderr. See docs/cli/conventions.md.
"""
import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import List, Any, Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

__all__ = [
    "console", "err_console", "print_title", "print_info", "print_success",
    "print_saved", "print_warning", "print_error", "print_table",
    "set_quiet", "TableResult", "tables_to_json", "tables_to_csv", "tables_to_text",
    "write_tables",
]


console = Console(highlight=False)
# ponytail: errors, warnings, and info belong on stderr to keep stdout pipeline-clean for json/csv
err_console = Console(stderr=True, highlight=False)


# ponytail: quiet is an explicit switch (set by CLICommand from -q), not inferred from logging,
# so messages still appear when FROG is used as a library
_QUIET = False


def set_quiet(quiet: bool) -> None:
    """Silence banner, info, success and saved-file messages (warnings and errors stay)."""
    global _QUIET
    _QUIET = bool(quiet)


def print_title(text: str) -> None:
    """Print title in a colored box (stderr, suppressed in quiet mode)."""
    if _QUIET:
        return
    err_console.print()
    err_console.print(Panel(Text(text.upper(), justify="center", style="bold white"), style="blue"))
    err_console.print()

def print_info(text: str) -> None:
    """Print info message (cyan) to stderr, suppressed in quiet mode."""
    if not _QUIET:
        err_console.print(f"[bold cyan]ℹ[/]  {text}")

def print_success(text: str) -> None:
    """Print success message (green) to stderr, suppressed in quiet mode."""
    if not _QUIET:
        err_console.print(f"[bold green]✔[/]  {text}")

def print_saved(path, kind: str = "file") -> None:
    """Report a written file on stderr: '✔ Saved <kind>: <path>'."""
    print_success(f"Saved {kind}: [bold]{path}[/]")

def print_warning(text: str) -> None:
    """Print yellow warning on stderr."""
    err_console.print(f"[bold yellow]⚠  WARNING:[/] {text}")

def print_error(text: str, exit_code: Optional[int] = None) -> None:
    """Print error in red to stderr and exit (optional)."""
    err_console.print(f"[bold red]✖  ERROR:[/] {text}")
    if exit_code is not None:
        sys.exit(exit_code)


def _is_np_scalar(item, kinds: str) -> bool:
    return type(item).__module__ == "numpy" and hasattr(item, "dtype") and item.dtype.kind in kinds


def _is_number(item) -> bool:
    if isinstance(item, bool):
        return False
    return isinstance(item, (int, float)) or _is_np_scalar(item, "iuf")


def _format_cell(item, floatfmt):
    if isinstance(item, bool):
        return str(item)
    if isinstance(item, float) or _is_np_scalar(item, "f"):
        if math.isnan(float(item)):
            return "-"
        return f"{float(item):{floatfmt}}"
    return "-" if item is None else str(item)


def _plain(value: Any) -> Any:
    """Machine-readable cell: numpy → Python (arrays → lists), NaN/±inf → None.

    Strings are literal (no Rich markup). Non-finite floats become null because JSON has no
    representation for them.
    """
    if type(value).__module__ == "numpy":
        value = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _plain_header(h) -> str:
    return " ".join(str(h).split())


def _build_rich_table(data, headers, title, floatfmt) -> Table:
    # ponytail: headers, titles and cells are literal text, so units like "[eV]" survive
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED,
                  title=Text(str(title)) if title else None)
    for col, h in enumerate(headers):
        values = [r[col] for r in data if col < len(r) and r[col] is not None]
        numeric = bool(values) and all(_is_number(v) for v in values)
        table.add_column(Text(str(h)), justify="right" if numeric else "left")
    for row in data:
        table.add_row(*[Text(_format_cell(item, _col_fmt(floatfmt, i))) for i, item in enumerate(row)])
    return table


def _col_fmt(floatfmt, col: int) -> str:
    """floatfmt is one format for every column, or a sequence with one format per column."""
    if isinstance(floatfmt, str):
        return floatfmt
    return floatfmt[col] if col < len(floatfmt) else ".3f"


def print_table(
    data: List[List[Any]],
    headers: List[str],
    title: str = "",
    floatfmt: str = ".3f",
) -> None:
    """Print a table on stdout as a Rich box."""
    if not data:
        print_warning(f"No data to show{f' for {title!r}' if title else ''}.")
        return
    console.print(_build_rich_table(data, headers, title, floatfmt))


class TableResult:
    """One result table of a CLI run."""

    def __init__(self, data: Sequence[Sequence[Any]], headers: Sequence[str], title: str = "", floatfmt: str = ".3f"):
        self.data = [list(r) for r in data]
        self.headers = list(headers)
        self.title = title
        self.floatfmt = floatfmt

    def to_dict(self) -> dict:
        columns = [_plain_header(h) for h in self.headers]
        rows = [dict(zip(columns, (_plain(v) for v in row))) for row in self.data]
        return {"title": _plain_header(self.title), "columns": columns, "rows": rows}


def tables_to_json(tables: Sequence[TableResult]) -> str:
    return json.dumps([t.to_dict() for t in tables], indent=2, default=str, ensure_ascii=False,
                      allow_nan=False)


def tables_to_csv(tables: Sequence[TableResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for i, t in enumerate(tables):
        if i:
            buf.write("\n")
        if t.title:
            buf.write(f"# {_plain_header(t.title)}\n")
        writer.writerow([_plain_header(h) for h in t.headers])
        writer.writerows([[_plain(v) for v in row] for row in t.data])
    return buf.getvalue()


def tables_to_text(tables: Sequence[TableResult], width: int = 200) -> str:
    buf = io.StringIO()
    file_console = Console(file=buf, force_terminal=False, no_color=True, width=width)
    for t in tables:
        file_console.print(_build_rich_table(t.data, t.headers, t.title, t.floatfmt))
    return buf.getvalue()


TABLE_FILE_FORMATS = {".json": "json", ".csv": "csv", ".txt": "table", ".dat": "table"}


def write_tables(tables: Sequence[TableResult], path, fmt: str = "table") -> Path:
    """Write all tables to one file. The extension (.json/.csv/.txt) wins over `fmt`."""
    path = Path(path)
    fmt = TABLE_FILE_FORMATS.get(path.suffix.lower(), fmt)
    if fmt == "json":
        text = tables_to_json(tables) + "\n"
    elif fmt == "csv":
        text = tables_to_csv(tables)
    else:
        text = tables_to_text(tables)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


