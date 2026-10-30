from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class Span:
    start: int | None = None
    end: int | None = None
    line: int = 1
    column: int = 1
    end_line: int | None = None
    end_column: int | None = None

    @classmethod
    def from_meta(cls, meta) -> "Span":
        return cls(
            start=meta.start_pos,
            end=meta.end_pos,
            line=meta.line,
            column=meta.column,
            end_line=meta.end_line,
            end_column=meta.end_column,
        )

    @classmethod
    def point(cls, offset: int, line: int, column: int) -> "Span":
        return cls(offset, offset + 1, line, column, line, column + 1)

    @classmethod
    def at(cls, line: int, column: int, width: int = 1) -> "Span":
        width = max(1, width)
        return cls(None, None, line, column, line, column + width)

    @property
    def width(self) -> int:
        if self.end_line == self.line and self.end_column is not None:
            return max(1, self.end_column - self.column)
        if self.start is not None and self.end is not None and self.end > self.start:
            return self.end - self.start
        return 1


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str

    @classmethod
    def read(cls, path: str | Path) -> "SourceFile":
        path = Path(path)
        return cls(str(path), path.read_text(encoding="utf-8"))

    def line_text(self, line: int) -> str:
        lines = self.text.splitlines()
        if line < 1 or line > len(lines):
            return ""
        return lines[line - 1]

    def line_start(self, line: int) -> int:
        offset = 0
        for current, value in enumerate(self.text.splitlines(keepends=True), 1):
            if current == line:
                return offset
            offset += len(value)
        return len(self.text)


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    HELP = "help"


@dataclass(frozen=True)
class Label:
    span: Span
    message: str = ""
    primary: bool = False


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: Severity = Severity.ERROR
    labels: tuple[Label, ...] = ()
    notes: tuple[str, ...] = ()

    @classmethod
    def error(
        cls,
        code: str,
        message: str,
        span: Span | None = None,
        label: str = "",
        notes: tuple[str, ...] = (),
    ) -> "Diagnostic":
        labels = (Label(span, label, True),) if span is not None else ()
        return cls(code, message, Severity.ERROR, labels, notes)


@dataclass
class DiagnosticBag:
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def add(self, diagnostic: Diagnostic) -> None:
        self.diagnostics.append(diagnostic)

    def error(self, code: str, message: str, span: Span | None = None, label: str = "") -> None:
        self.add(Diagnostic.error(code, message, span, label))

    def extend(self, diagnostics) -> None:
        self.diagnostics.extend(diagnostics)

    def __iter__(self):
        return iter(self.diagnostics)

    def __len__(self) -> int:
        return len(self.diagnostics)

    @property
    def has_errors(self) -> bool:
        return any(diagnostic.severity == Severity.ERROR for diagnostic in self.diagnostics)


def gutter(width: int) -> str:
    return f" {' ' * width} |"


class DiagnosticRenderer:
    _COLORS = {
        Severity.ERROR: "\033[31;1m",
        Severity.WARNING: "\033[33;1m",
        Severity.NOTE: "\033[36;1m",
        Severity.HELP: "\033[32;1m",
    }
    _RESET = "\033[0m"

    def __init__(self, color: bool = False):
        self.color = color

    def render(self, diagnostic: Diagnostic, source: SourceFile | None = None) -> str:
        severity = diagnostic.severity.value
        heading = f"{severity}[{diagnostic.code}]: {diagnostic.message}"
        if self.color:
            heading = f"{self._COLORS[diagnostic.severity]}{heading}{self._RESET}"
        lines = [heading]

        if source is None:
            for label in diagnostic.labels:
                lines.append(f"  --> {label.span.line}:{label.span.column}")
                if label.message:
                    lines.append(f"   = {label.message}")
        elif diagnostic.labels:
            width = max(len(str(label.span.line)) for label in diagnostic.labels)
            first = diagnostic.labels[0].span
            lines.append(f"  --> {source.path}:{first.line}:{first.column}")
            lines.append(gutter(width))
            for label in diagnostic.labels:
                lines.extend(self._render_label(source, label, width))

        for note in diagnostic.notes:
            lines.append(f"  = note: {note}")
        return "\n".join(lines)

    @staticmethod
    def _render_label(source: SourceFile, label: Label, width: int) -> list[str]:
        span = label.span
        text = source.line_text(span.line)
        underline_start = max(0, span.column - 1)
        marker = "^" if label.primary else "-"
        caption = marker * span.width
        if label.message:
            caption += f" {label.message}"
        return [
            f" {span.line:>{width}} | {text}",
            gutter(width),
            f"{gutter(width)} {' ' * underline_start}{caption}",
        ]
