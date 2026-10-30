from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .ast import Theorem
from .db import DEFAULT_ROOT, Codebase, CodebaseError
from .diagnostics import Diagnostic, DiagnosticRenderer, Severity, SourceFile
from .hashing import short_hash
from .parser import ParseError, parse_source_file
from .proof import ProofStatus, check_theorem
from .sema import SemanticError, analyze


def _use_color(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gunmu", description="Gunmu language tools")
    commands = parser.add_subparsers(dest="command", required=True)

    proof = commands.add_parser("proof", help="check theorem proofs in a source file")
    proof.add_argument("path", type=Path, help="path to a .gunmu source file")
    proof.add_argument("--color", dest="color", action="store_true", default=None, help="force colored output")
    proof.add_argument("--no-color", dest="color", action="store_false", help="disable colored output")
    proof.set_defaults(handler=run_proof)

    database = commands.add_parser("db", help="content-addressed codebase commands")
    database.add_argument(
        "--codebase",
        type=Path,
        default=None,
        help=f"codebase root (default: {DEFAULT_ROOT})",
    )
    db_commands = database.add_subparsers(dest="db_command", required=True)

    db_init = db_commands.add_parser("init", help="create the codebase")
    db_init.set_defaults(db_handler=db_init_command)

    db_add = db_commands.add_parser("add", help="import source files")
    db_add.add_argument("paths", type=Path, nargs="+")
    db_add.set_defaults(db_handler=db_add_command)

    db_ls = db_commands.add_parser("ls", help="list names")
    db_ls.add_argument("prefix", nargs="?", default="")
    db_ls.add_argument("--json", action="store_true")
    db_ls.set_defaults(db_handler=db_ls_command)

    db_show = db_commands.add_parser("show", help="print a definition")
    db_show.add_argument("reference")
    db_show.add_argument("--metadata", action="store_true", help="include hash and namespace info")
    db_show.set_defaults(db_handler=db_show_command)

    db_deps = db_commands.add_parser("deps", help="show dependency edges")
    db_deps.add_argument("reference")
    db_deps.add_argument("--reverse", action="store_true", help="show dependents instead")
    db_deps.add_argument("--transitive", action="store_true")
    db_deps.add_argument("--json", action="store_true")
    db_deps.set_defaults(db_handler=db_deps_command)

    db_rename = db_commands.add_parser("rename", help="move a name, leaving hashes untouched")
    db_rename.add_argument("old")
    db_rename.add_argument("new")
    db_rename.add_argument("--force", action="store_true")
    db_rename.set_defaults(db_handler=db_rename_command)

    db_export = db_commands.add_parser("export", help="render a definition and its dependencies")
    db_export.add_argument("reference")
    db_export.set_defaults(db_handler=db_export_command)

    db_verify = db_commands.add_parser("verify", help="recheck stored hashes and round-trips")
    db_verify.add_argument("--shallow", action="store_true", help="skip the re-parse round-trip")
    db_verify.set_defaults(db_handler=db_verify_command)

    db_diff = db_commands.add_parser("diff", help="compare two stored definitions")
    db_diff.add_argument("left")
    db_diff.add_argument("right")
    db_diff.set_defaults(db_handler=db_diff_command)

    return parser


# --------------------------------------------------------------------- proof


def _proof_failure(theorem: Theorem, result) -> Diagnostic:
    if result.status == ProofStatus.DISPROVED:
        code, message = "E2001", f"theorem `{theorem.name}` is false"
    elif result.status == ProofStatus.UNSUPPORTED:
        code, message = "E2002", f"theorem `{theorem.name}` is outside the supported fragment"
    else:
        code, message = "E2003", f"theorem `{theorem.name}` could not be decided"
    labels = ()
    if theorem.span is not None:
        from .diagnostics import Label

        labels = (Label(theorem.span, result.message, True),)
    notes = []
    if result.model:
        notes.append(f"counterexample: {result.model}")
    return Diagnostic(code, message, Severity.ERROR, labels, tuple(notes))


def run_proof(path: Path, color: bool | None = None) -> int:
    try:
        source_file = SourceFile.read(path)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    renderer = DiagnosticRenderer(color=_use_color(color))

    try:
        module = parse_source_file(source_file)
        analyze(module)
    except ParseError as error:
        print(renderer.render(error.diagnostic, source_file), file=sys.stderr)
        return 1
    except SemanticError as error:
        print(renderer.render(error.diagnostic, source_file), file=sys.stderr)
        return 1

    theorems = [item for item in module.items if isinstance(item, Theorem)]
    if not theorems:
        print(f"{source_file.path}: no theorems found")
        return 0

    failures: list[tuple[Theorem, object]] = []
    width = max(len(theorem.name) for theorem in theorems)
    for theorem in theorems:
        result = check_theorem(theorem)
        status = result.status.value
        marker = "ok" if result.status == ProofStatus.PROVED else "FAIL"
        print(f"{theorem.name:<{width}}  {status:<11} {marker}")
        if result.status != ProofStatus.PROVED:
            failures.append((theorem, result))

    for theorem, result in failures:
        print(file=sys.stderr)
        print(renderer.render(_proof_failure(theorem, result), source_file), file=sys.stderr)

    return 1 if failures else 0


# ------------------------------------------------------------------------ db


def _open_codebase(arguments) -> Codebase:
    return Codebase(arguments.codebase)


def db_init_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    codebase.init()
    print(f"initialised codebase at {codebase.path}")
    return 0


def db_add_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    exit_code = 0
    for path in arguments.paths:
        try:
            source_file = SourceFile.read(path)
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            exit_code = 1
            continue
        try:
            module = parse_source_file(source_file)
            analyze(module)
            result = codebase.import_module(module)
        except ParseError as error:
            print(DiagnosticRenderer().render(error.diagnostic, source_file), file=sys.stderr)
            exit_code = 1
            continue
        except SemanticError as error:
            print(DiagnosticRenderer().render(error.diagnostic, source_file), file=sys.stderr)
            exit_code = 1
            continue
        except CodebaseError as error:
            print(f"error: {error.message}", file=sys.stderr)
            exit_code = 1
            continue
        width = max((len(d.local_name) for d in result.added + result.unchanged), default=0)
        for definition in result.added:
            print(f"+ {definition.local_name:<{width}}  {short_hash(definition.hash)}")
        for definition in result.unchanged:
            print(f"= {definition.local_name:<{width}}  {short_hash(definition.hash)}  (already stored)")
    return exit_code


def db_ls_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    entries = codebase.entries(arguments.prefix)
    if arguments.json:
        print(json.dumps(
            [{"name": name, "hash": hash_value, "kind": kind} for name, hash_value, kind in entries],
            indent=2,
        ))
        return 0
    if not entries:
        print("(empty)")
        return 0
    width = max(len(name) for name, _, _ in entries)
    for name, hash_value, kind in entries:
        print(f"{kind:<8} {name:<{width}}  {short_hash(hash_value)}")
    return 0


def db_show_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    record = codebase.get(arguments.reference)
    if arguments.metadata:
        print(f"hash:   {record.hash}")
        print(f"kind:   {record.kind}")
        print(f"module: {record.module}")
        print(f"names:  {', '.join(record.names) if record.names else '(none)'}")
        if record.group_hash:
            print(f"group:  {short_hash(record.group_hash)} #{record.group_index}")
        print()
    print(record.source)
    return 0


def db_deps_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    hashes = codebase.dependencies(
        arguments.reference, reverse=arguments.reverse, transitive=arguments.transitive
    )
    records = [codebase.get(hash_value) for hash_value in hashes]
    if arguments.json:
        print(json.dumps(
            [
                {
                    "hash": record.hash,
                    "name": record.primary_name,
                    "kind": record.kind,
                    "names": list(record.names),
                }
                for record in records
            ],
            indent=2,
        ))
        return 0
    if not records:
        print("(none)")
        return 0
    width = max(len(record.primary_name) for record in records)
    for record in records:
        print(f"{record.kind:<8} {record.primary_name:<{width}}  {short_hash(record.hash)}")
    return 0


def db_rename_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    old, new = codebase.rename(arguments.old, arguments.new, force=arguments.force)
    record = codebase.get(new)
    print(f"{old} -> {new}")
    print(f"hash unchanged: {short_hash(record.hash)}")
    return 0


def db_export_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    print(codebase.export(arguments.reference), end="")
    return 0


def db_verify_command(arguments) -> int:
    codebase = _open_codebase(arguments)
    problems = codebase.verify(deep=not arguments.shallow)
    if problems:
        for problem in problems:
            print(f"error[E3007]: {problem}", file=sys.stderr)
        return 1
    print("ok: every stored definition hashes back to its key")
    return 0


def db_diff_command(arguments) -> int:
    import difflib

    codebase = _open_codebase(arguments)
    left = codebase.get(arguments.left)
    right = codebase.get(arguments.right)
    if left.hash == right.hash:
        print(f"identical: both are {short_hash(left.hash)}")
        return 0
    print(f"--- {left.primary_name} ({short_hash(left.hash)})")
    print(f"+++ {right.primary_name} ({short_hash(right.hash)})")
    difference = difflib.unified_diff(
        left.source.splitlines(),
        right.source.splitlines(),
        lineterm="",
        n=1,
    )
    for line in difference:
        if line.startswith(("---", "+++")):
            continue
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "db":
        try:
            return arguments.db_handler(arguments)
        except CodebaseError as error:
            print(f"error: {error.message}", file=sys.stderr)
            return 1
    return arguments.handler(arguments.path, color=arguments.color)


if __name__ == "__main__":
    raise SystemExit(main())
