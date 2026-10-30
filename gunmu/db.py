"""The Gunmu codebase: definitions stored by content, names stored separately.

Layout mirrors Unison: the source of truth is a SQLite database under
``~/.gunmu``, definitions are keyed by the hash of their canonical form, and the
namespace table is the only thing that knows about names. Consequences:

* importing the same definition twice is a no-op (the hash is the identity);
* renaming touches only the namespace rows, never a definition;
* editing a dependency changes its hash, and therefore the hash of everything
  that references it — visible as a `dep` edge and a changed hash.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from .ast import Module
from .canonical import CanonicalDefinition, canonicalize_module
from .hashing import (
    FORMAT_VERSION,
    definition_hash,
    group_hash,
    group_member_hash,
    is_hash,
    resolve_prefix,
)
from .parser import parse
from .pretty import render_definition
from .sema import SemanticError, analyze
from .diagnostics import Diagnostic

SCHEMA_VERSION = 1
DEFAULT_ROOT = Path(os.environ.get("GUNMU_HOME", Path.home() / ".gunmu"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS def (
    hash        TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    format      INTEGER NOT NULL,
    module      TEXT NOT NULL,
    local_name  TEXT NOT NULL,
    canonical   TEXT NOT NULL,
    source      TEXT NOT NULL,
    group_hash  TEXT,
    group_index INTEGER,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS code_group (
    group_hash TEXT PRIMARY KEY,
    format     INTEGER NOT NULL,
    blob       TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dep (
    from_hash TEXT NOT NULL,
    to_hash   TEXT NOT NULL,
    PRIMARY KEY (from_hash, to_hash)
);
CREATE INDEX IF NOT EXISTS dep_reverse ON dep (to_hash);
CREATE TABLE IF NOT EXISTS namespace (
    name       TEXT PRIMARY KEY,
    hash       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS namespace_history (
    name       TEXT NOT NULL,
    hash       TEXT NOT NULL,
    changed_at TEXT NOT NULL,
    PRIMARY KEY (name, changed_at)
);
"""


class CodebaseError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.diagnostic = Diagnostic.error(code, message)


@dataclass(frozen=True)
class DefinitionRecord:
    hash: str
    kind: str
    module: str
    local_name: str
    canonical: str
    source: str
    group_hash: str | None = None
    group_index: int | None = None
    names: tuple[str, ...] = ()

    @property
    def primary_name(self) -> str:
        return self.names[0] if self.names else f"{self.module}.{self.local_name}"


@dataclass
class ImportResult:
    added: list[CanonicalDefinition] = field(default_factory=list)
    unchanged: list[CanonicalDefinition] = field(default_factory=list)
    renamed: list[tuple[str, str, str]] = field(default_factory=list)
    groups: int = 0

    @property
    def names(self) -> dict[str, str]:
        return {definition.name: definition.hash for definition in self.added + self.unchanged}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Codebase:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root is not None else DEFAULT_ROOT
        self.path = self.root / "code.db"

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init(self) -> None:
        with self.connect() as connection:
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('format_version', ?)",
                (str(FORMAT_VERSION),),
            )

    def exists(self) -> bool:
        return self.path.exists()

    def _require(self) -> sqlite3.Connection:
        if not self.exists():
            raise CodebaseError(
                "E3001",
                f"no codebase at {self.path}; run `gunmu db init` first",
            )
        connection = self.connect()
        connection.executescript(_SCHEMA)
        return connection

    # -- importing -----------------------------------------------------------

    def import_module(self, module: Module) -> ImportResult:
        canonical = canonicalize_module(module)
        items = {item.name: item for item in module.items}
        result = ImportResult(groups=len(canonical.groups))
        stamp = _now()

        with self._require() as connection:
            for group in canonical.groups:
                connection.execute(
                    "INSERT OR IGNORE INTO code_group (group_hash, format, blob, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (group.hash, FORMAT_VERSION, group.blob, stamp),
                )
            for definition in canonical.definitions:
                source = render_definition(items[definition.local_name], definition.module)
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO def"
                    " (hash, kind, format, module, local_name, canonical, source,"
                    "  group_hash, group_index, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        definition.hash,
                        definition.kind,
                        FORMAT_VERSION,
                        definition.module,
                        definition.local_name,
                        definition.canonical,
                        source,
                        definition.group_hash,
                        definition.group_index,
                        stamp,
                    ),
                )
                if cursor.rowcount:
                    result.added.append(definition)
                else:
                    result.unchanged.append(definition)
                for dependency in definition.dependencies:
                    connection.execute(
                        "INSERT OR IGNORE INTO dep (from_hash, to_hash) VALUES (?, ?)",
                        (definition.hash, dependency),
                    )
                previous = connection.execute(
                    "SELECT hash FROM namespace WHERE name = ?", (definition.name,)
                ).fetchone()
                if previous is None or previous["hash"] != definition.hash:
                    connection.execute(
                        "INSERT OR REPLACE INTO namespace (name, hash, updated_at) VALUES (?, ?, ?)",
                        (definition.name, definition.hash, stamp),
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO namespace_history (name, hash, changed_at)"
                        " VALUES (?, ?, ?)",
                        (definition.name, definition.hash, stamp),
                    )
                    if previous is not None:
                        result.renamed.append((definition.name, previous["hash"], definition.hash))
        return result

    def import_source(self, source: str) -> ImportResult:
        module = parse(source)
        analyze(module)
        return self.import_module(module)

    def import_file(self, path: Path | str) -> ImportResult:
        return self.import_source(Path(path).read_text(encoding="utf-8"))

    # -- resolving -----------------------------------------------------------

    def _all_hashes(self, connection: sqlite3.Connection) -> list[str]:
        return [row["hash"] for row in connection.execute("SELECT hash FROM def")]

    def resolve_hash(self, reference: str) -> str:
        with self._require() as connection:
            matches = resolve_prefix(reference, self._all_hashes(connection))
        if not matches:
            raise CodebaseError("E3002", f"unknown definition hash `{reference}`")
        if len(matches) > 1:
            raise CodebaseError(
                "E3003",
                f"ambiguous hash prefix `{reference}` matches {len(matches)} definitions",
            )
        return matches[0]

    def resolve_reference(self, reference: str) -> str:
        """A namespace name, a bare suffix, or a `#hash` prefix."""
        if is_hash(reference):
            return self.resolve_hash(reference)
        with self._require() as connection:
            row = connection.execute(
                "SELECT hash FROM namespace WHERE name = ?", (reference,)
            ).fetchone()
            if row is not None:
                return row["hash"]
            suffix = f"%.{reference}"
            rows = connection.execute(
                "SELECT name, hash FROM namespace WHERE name LIKE ? ORDER BY name", (suffix,)
            ).fetchall()
        if not rows:
            raise CodebaseError("E3004", f"unknown definition `{reference}`")
        if len(rows) > 1:
            names = ", ".join(row["name"] for row in rows)
            raise CodebaseError("E3003", f"ambiguous name `{reference}` matches: {names}")
        return rows[0]["hash"]

    def names_of(self, hash_value: str) -> tuple[str, ...]:
        with self._require() as connection:
            rows = connection.execute(
                "SELECT name FROM namespace WHERE hash = ? ORDER BY name", (hash_value,)
            ).fetchall()
        return tuple(row["name"] for row in rows)

    def get(self, reference: str) -> DefinitionRecord:
        hash_value = self.resolve_reference(reference)
        with self._require() as connection:
            row = connection.execute("SELECT * FROM def WHERE hash = ?", (hash_value,)).fetchone()
            if row is None:
                raise CodebaseError("E3002", f"unknown definition hash `{hash_value}`")
            names = tuple(
                item["name"]
                for item in connection.execute(
                    "SELECT name FROM namespace WHERE hash = ? ORDER BY name", (hash_value,)
                )
            )
        return DefinitionRecord(
            hash=row["hash"],
            kind=row["kind"],
            module=row["module"],
            local_name=row["local_name"],
            canonical=row["canonical"],
            source=row["source"],
            group_hash=row["group_hash"],
            group_index=row["group_index"],
            names=names,
        )

    # -- listing and traversal -----------------------------------------------

    def entries(self, prefix: str = "") -> list[tuple[str, str, str]]:
        with self._require() as connection:
            if prefix:
                rows = connection.execute(
                    "SELECT n.name AS name, n.hash AS hash, d.kind AS kind"
                    " FROM namespace n JOIN def d ON d.hash = n.hash"
                    " WHERE n.name = ? OR n.name LIKE ? ORDER BY n.name",
                    (prefix, f"{prefix}.%"),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT n.name AS name, n.hash AS hash, d.kind AS kind"
                    " FROM namespace n JOIN def d ON d.hash = n.hash ORDER BY n.name"
                ).fetchall()
        return [(row["name"], row["hash"], row["kind"]) for row in rows]

    def _direct(self, connection: sqlite3.Connection, hash_value: str, reverse: bool) -> list[str]:
        if reverse:
            rows = connection.execute(
                "SELECT from_hash AS other FROM dep WHERE to_hash = ?", (hash_value,)
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT to_hash AS other FROM dep WHERE from_hash = ?", (hash_value,)
            ).fetchall()
        return sorted(row["other"] for row in rows)

    def dependencies(self, reference: str, reverse: bool = False, transitive: bool = False) -> list[str]:
        root = self.resolve_reference(reference)
        with self._require() as connection:
            seen: set[str] = set()
            frontier = [root]
            collected: list[str] = []
            while frontier:
                current = frontier.pop(0)
                for other in self._direct(connection, current, reverse):
                    if other in seen:
                        continue
                    seen.add(other)
                    collected.append(other)
                    frontier.append(other)
                if not transitive:
                    break
        return collected

    def _closure(self, connection: sqlite3.Connection, root: str) -> list[str]:
        """Dependencies first, so the result can be rendered as a compilable file."""
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(node: str) -> None:
            if state.get(node):
                return
            state[node] = 1
            for dependency in sorted(self._direct(connection, node, reverse=False)):
                visit(dependency)
            state[node] = 2
            order.append(node)

        visit(root)
        return order

    def export(self, reference: str) -> str:
        root = self.resolve_reference(reference)
        with self._require() as connection:
            order = self._closure(connection, root)
            rows = {
                row["hash"]: row
                for row in connection.execute(
                    f"SELECT * FROM def WHERE hash IN ({','.join('?' * len(order))})", order
                )
            }
        module = rows[root]["module"] or "generated"
        chunks = [f"module {module}", ""]
        for hash_value in order:
            chunks.append(rows[hash_value]["source"])
            chunks.append("")
        return "\n".join(chunks).rstrip() + "\n"

    # -- names ---------------------------------------------------------------

    def rename(self, old: str, new: str, force: bool = False) -> tuple[str, str]:
        if is_hash(old) or is_hash(new):
            raise CodebaseError("E3005", "rename takes namespace names, not hashes")
        stamp = _now()
        with self._require() as connection:
            row = connection.execute("SELECT hash FROM namespace WHERE name = ?", (old,)).fetchone()
            if row is None:
                raise CodebaseError("E3004", f"unknown definition `{old}`")
            hash_value = row["hash"]
            existing = connection.execute(
                "SELECT hash FROM namespace WHERE name = ?", (new,)
            ).fetchone()
            if existing is not None and existing["hash"] != hash_value and not force:
                raise CodebaseError(
                    "E3006",
                    f"`{new}` already names a different definition; pass force to overwrite",
                )
            connection.execute(
                "INSERT OR REPLACE INTO namespace (name, hash, updated_at) VALUES (?, ?, ?)",
                (new, hash_value, stamp),
            )
            connection.execute(
                "INSERT OR REPLACE INTO namespace_history (name, hash, changed_at) VALUES (?, ?, ?)",
                (new, hash_value, stamp),
            )
            connection.execute("DELETE FROM namespace WHERE name = ?", (old,))
        return old, new

    # -- verification --------------------------------------------------------

    def verify(self, deep: bool = True) -> list[str]:
        problems: list[str] = []
        with self._require() as connection:
            groups = {
                row["group_hash"]: row["blob"]
                for row in connection.execute("SELECT group_hash, blob FROM code_group")
            }
            for group, blob in groups.items():
                if group_hash(blob) != group:
                    problems.append(f"group {group}: blob does not hash to its key")
            definitions = connection.execute("SELECT * FROM def").fetchall()
            for row in definitions:
                expected = (
                    definition_hash(row["kind"], row["canonical"], row["format"])
                    if row["group_hash"] is None
                    else group_member_hash(row["group_hash"], row["group_index"], row["format"])
                )
                if expected != row["hash"]:
                    problems.append(f"{row['hash']}: recomputed hash is {expected}")
                if row["group_hash"] is not None and row["group_hash"] not in groups:
                    problems.append(f"{row['hash']}: missing group {row['group_hash']}")

            # dangling namespace entries and dependency edges
            for row in connection.execute(
                "SELECT n.name AS name, n.hash AS hash FROM namespace n"
                " LEFT JOIN def d ON d.hash = n.hash WHERE d.hash IS NULL"
            ):
                problems.append(f"namespace {row['name']}: points at missing {row['hash']}")
            for row in connection.execute(
                "SELECT DISTINCT to_hash AS other FROM dep WHERE to_hash NOT IN (SELECT hash FROM def)"
            ):
                problems.append(f"dependency edge points at missing {row['other']}")
            for row in connection.execute(
                "SELECT DISTINCT from_hash AS other FROM dep WHERE from_hash NOT IN (SELECT hash FROM def)"
            ):
                problems.append(f"dependency edge starts at missing {row['other']}")

        if deep:
            problems.extend(self._verify_roundtrip())
        return problems

    def _verify_roundtrip(self) -> list[str]:
        """Re-parse stored sources and confirm they hash back to the same values.

        Only definitions currently reachable from the namespace are reconstructed.
        Superseded versions stay in `def` (the store is append-only) and share a
        `local_name` with the version that replaced them, so including them would
        concatenate two definitions of the same name into one module. They are
        still covered by the hash comparison in `verify`.
        """
        problems: list[str] = []
        with self._require() as connection:
            live = {row["hash"] for row in connection.execute("SELECT DISTINCT hash FROM namespace")}
            modules: dict[str, dict[str, sqlite3.Row]] = {}
            for row in connection.execute("SELECT * FROM def ORDER BY module, local_name"):
                if row["hash"] not in live:
                    continue
                modules.setdefault(row["module"], {})[row["local_name"]] = row

        for module_name, by_name in modules.items():
            rows = list(by_name.values())
            chunks = [f"module {module_name}", ""]
            for row in rows:
                chunks.append(row["source"])
                chunks.append("")
            text = "\n".join(chunks)
            try:
                reparsed = parse(text)
                recomputed = canonicalize_module(reparsed)
            except (ValueError, SemanticError) as error:
                problems.append(f"module {module_name}: stored source does not re-parse: {error}")
                continue
            actual = {definition.local_name: definition.hash for definition in recomputed.definitions}
            for row in rows:
                expected = actual.get(row["local_name"])
                if expected != row["hash"]:
                    problems.append(
                        f"{row['hash']} ({module_name}.{row['local_name']}):"
                        f" round-trip produced {expected}"
                    )
        return problems
