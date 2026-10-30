"""AST -> Gunmu source.

The database stores definitions, so reading code back out needs a printer. The
contract is round-trip fidelity: ``parse(pretty_module(m))`` must reproduce
``m`` modulo spans, otherwise a definition retrieved from the database would
not hash back to the value it was stored under.
"""

from __future__ import annotations

import json

from .ast import (
    AssertStmt, Binary, BindingPattern, BoolLiteral, BreakStmt, Call, ContinueStmt,
    Expr, ExprStmt, FieldAccess, Function, IntLiteral, LetStmt,
    LiteralPattern, LoopStmt, MatchStmt, Module, Name, Pattern, PrintStmt,
    Proposition, Record, RecordPattern, RecordValue, Requires, ReturnStmt,
    StringLiteral, Theorem, TypeRef, Unary, UnitLiteral, WildcardPattern,
)


class PrettyError(ValueError):
    pass


_PRECEDENCE = {
    "implies": 1,
    "or": 2,
    "and": 3,
    "==": 4, "!=": 4, "<": 4, "<=": 4, ">": 4, ">=": 4,
    "+": 5, "-": 5,
    "*": 6, "/": 6, "%": 6,
}
_UNARY_PRECEDENCE = 7
_ATOM_PRECEDENCE = 8
_RIGHT_ASSOCIATIVE = {"implies"}


def render_type(reference: TypeRef) -> str:
    return reference.name


def render_expr(value: Expr, parent_precedence: int = 0) -> str:
    if isinstance(value, IntLiteral):
        return str(value.value)
    if isinstance(value, StringLiteral):
        return json.dumps(value.value, ensure_ascii=False)
    if isinstance(value, BoolLiteral):
        return "true" if value.value else "false"
    if isinstance(value, UnitLiteral):
        return "unit"
    if isinstance(value, Name):
        return value.value
    if isinstance(value, Unary):
        operand = render_expr(value.operand, _UNARY_PRECEDENCE)
        text = f"-{operand}" if value.op == "-" else f"not {operand}"
        if _UNARY_PRECEDENCE < parent_precedence:
            return f"({text})"
        return text
    if isinstance(value, Binary):
        precedence = _PRECEDENCE.get(value.op)
        if precedence is None:
            raise PrettyError(f"unknown binary operator {value.op}")
        if value.op in _RIGHT_ASSOCIATIVE:
            left = render_expr(value.left, precedence + 1)
            right = render_expr(value.right, precedence)
        else:
            left = render_expr(value.left, precedence)
            right = render_expr(value.right, precedence + 1)
        text = f"{left} {value.op} {right}"
        return f"({text})" if precedence < parent_precedence else text
    if isinstance(value, Call):
        name = value.function.value if isinstance(value.function, Name) else render_expr(value.function)
        arguments = ", ".join(render_expr(argument) for argument in value.arguments)
        return f"{name}({arguments})"
    if isinstance(value, FieldAccess):
        return f"{render_expr(value.value, _ATOM_PRECEDENCE)}.{value.field}"
    if isinstance(value, RecordValue):
        inner = ", ".join(f"{name}: {render_expr(value.fields[name])}" for name in value.fields)
        return f"{value.type_name} {{ {inner} }}" if inner else f"{value.type_name} {{ }}"
    if isinstance(value, Proposition):
        return render_expr(value.expression, parent_precedence)
    raise PrettyError(f"cannot print expression {type(value).__name__}")


def render_proposition(value) -> str:
    return render_expr(value.expression if isinstance(value, Proposition) else value)


def render_pattern(value: Pattern) -> str:
    if isinstance(value, WildcardPattern):
        return "_"
    if isinstance(value, BindingPattern):
        return value.name
    if isinstance(value, LiteralPattern):
        return render_expr(value.value)
    if isinstance(value, RecordPattern):
        parts = [f"{name}: {render_pattern(value.fields[name])}" for name in value.fields]
        if value.rest:
            parts.append("..")
        inner = ", ".join(parts)
        return f"{value.type_name} {{ {inner} }}"
    raise PrettyError(f"cannot print pattern {type(value).__name__}")


class _Printer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.depth = 0

    def emit(self, text: str) -> None:
        self.lines.append("    " * self.depth + text)

    def open(self, text: str) -> None:
        self.emit(text)
        self.depth += 1

    def close(self, text: str = "}") -> None:
        self.depth -= 1
        self.emit(text)

    # -- statements ----------------------------------------------------------

    def block(self, body: list) -> None:
        for statement in body:
            self.statement(statement)

    def statement(self, value) -> None:
        if isinstance(value, LetStmt):
            annotation = f": {render_type(value.type)}" if value.type is not None else ""
            self.emit(f"let {value.name}{annotation} = {render_expr(value.value)}")
        elif isinstance(value, PrintStmt):
            self.emit(f"print {render_expr(value.value)}")
        elif isinstance(value, AssertStmt):
            self.emit(f"assert {render_proposition(value.proposition)}")
        elif isinstance(value, ReturnStmt):
            self.emit("return" if value.value is None else f"return {render_expr(value.value)}")
        elif isinstance(value, BreakStmt):
            self.emit("break" if value.value is None else f"break {render_expr(value.value)}")
        elif isinstance(value, ContinueStmt):
            if value.bindings:
                inner = ", ".join(f"{name} = {render_expr(v)}" for name, v in value.bindings.items())
                self.emit(f"continue ({inner})")
            else:
                self.emit("continue")
        elif isinstance(value, ExprStmt):
            self.emit(render_expr(value.value))
        elif isinstance(value, MatchStmt):
            self.open(f"match {render_expr(value.value)} {{")
            for arm in value.arms:
                guard = f" if {render_proposition(arm.guard)}" if arm.guard is not None else ""
                self.open(f"{render_pattern(arm.pattern)}{guard} => {{")
                self.block(arm.body)
                self.close()
            self.close()
        elif isinstance(value, LoopStmt):
            if not value.bindings:
                raise PrettyError("a loop must declare at least one state binding")
            bindings = ", ".join(f"{name} = {render_expr(v)}" for name, v in value.bindings.items())
            if value.specs:
                self.emit(f"loop ({bindings})")
                self.open("with {")
                for spec in value.specs:
                    if spec.kind == "invariant":
                        self.emit(f"invariant {render_proposition(spec.value)}")
                    else:
                        self.emit(f"variant {render_expr(spec.value)}")
                self.close("} {")
                self.depth += 1
                self.block(value.body)
                self.close()
            else:
                self.open(f"loop ({bindings}) {{")
                self.block(value.body)
                self.close()
        else:
            raise PrettyError(f"cannot print statement {type(value).__name__}")


def render_definition(item, module_name: str = "generated") -> str:
    printer = _Printer()
    _render_item(printer, item)
    return "\n".join(printer.lines)


def _render_item(printer: _Printer, item) -> None:
    if isinstance(item, Record):
        printer.open(f"record {item.name} {{")
        for field in item.fields:
            printer.emit(f"{field.name}: {render_type(field.type)}")
        printer.close()
    elif isinstance(item, Function):
        parameters = ", ".join(f"{p.name}: {render_type(p.type)}" for p in item.params)
        signature = f"fn {item.name}({parameters}) -> {render_type(item.return_type)}"
        if item.specs:
            printer.emit(signature)
            for spec in item.specs:
                keyword = "requires" if isinstance(spec, Requires) else "ensures"
                printer.emit(f"{keyword} {render_proposition(spec.proposition)}")
            printer.open("{")
        else:
            printer.open(f"{signature} {{")
        printer.block(item.body)
        printer.close()
    elif isinstance(item, Theorem):
        parameters = ", ".join(f"{p.name}: {render_type(p.type)}" for p in item.params)
        signature = f"theorem {item.name}({parameters})"
        goal = render_proposition(item.proposition)
        if item.requires:
            printer.emit(signature)
            for requirement in item.requires:
                printer.emit(f"requires {render_proposition(requirement)}")
            printer.emit(f": {goal}")
        else:
            printer.emit(f"{signature}: {goal}")
        printer.open("proof {")
        for command in item.proof:
            if command.kind == "assert" and command.value is not None:
                printer.emit(f"assert {render_proposition(command.value)}")
            elif command.kind == "exact" and isinstance(command.value, str):
                printer.emit(f"exact {command.value}")
            else:
                printer.emit(command.kind)
        printer.close()
    else:
        raise PrettyError(f"cannot print item {type(item).__name__}")


def pretty_module(module: Module) -> str:
    printer = _Printer()
    if module.name:
        printer.emit(f"module {module.name}")
        printer.emit("")
    for index, item in enumerate(module.items):
        if index:
            printer.emit("")
        _render_item(printer, item)
    return "\n".join(printer.lines) + "\n"
