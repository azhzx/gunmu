from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .ast import (
    AssertStmt, Binary, BindingPattern, BoolLiteral, BreakStmt, Call, ContinueStmt,
    Ensures, Expr, ExprStmt, FieldAccess, Function, IntLiteral, LetStmt,
    LiteralPattern, LoopStmt, MatchStmt, Module, Name, PrintStmt, Proposition,
    Record, RecordPattern, RecordValue, ReturnStmt, StringLiteral,
    Theorem, TypeRef, Unary, UnitLiteral, WildcardPattern,
)
from .diagnostics import Diagnostic, Span


class SemanticError(ValueError):
    """A semantic failure carrying a stable code and source diagnostic."""

    def __init__(
        self,
        message: str,
        code: str = "E1000",
        span: Span | None = None,
        label: str = "",
        notes: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.span = span
        self.label = label
        self.notes = notes
        self.diagnostic = Diagnostic.error(code, message, span, label, notes)

    def __str__(self) -> str:
        if self.span is not None:
            return f"{self.span.line}:{self.span.column}: {self.message}"
        return self.message


@dataclass(frozen=True)
class FunctionSig:
    name: str
    params: tuple[TypeRef, ...]
    return_type: TypeRef


INT = TypeRef("Int")
BOOL = TypeRef("Bool")
STRING = TypeRef("String")
UNIT = TypeRef("Unit")
PROP = TypeRef("Prop")

BUILTIN_TYPES = {"Int", "Bool", "String", "Unit", "Prop"}
SCALAR_TYPES = {"Int", "Bool", "String", "Unit"}


def span_of(node: Any) -> Span | None:
    return getattr(node, "span", None)


class SemanticAnalyzer:
    def __init__(self) -> None:
        self.records: dict[str, Record] = {}
        self.functions: dict[str, FunctionSig] = {}
        self.theorems: dict[str, Theorem] = {}

    # ------------------------------------------------------------------ errors

    @staticmethod
    def _fail(code: str, message: str, node: Any = None, label: str = "", notes: tuple[str, ...] = ()) -> None:
        raise SemanticError(message, code, span_of(node), label, notes)

    def analyze(self, module: Module) -> Module:
        self._collect_items(module)
        for item in module.items:
            if isinstance(item, Function):
                self._check_function(item)
            elif isinstance(item, Theorem):
                self._check_theorem(item)
        return module

    # ------------------------------------------------------------ declaration

    def _collect_items(self, module: Module) -> None:
        for item in module.items:
            if isinstance(item, Record):
                if item.name in self.records or item.name in self.functions:
                    self._fail("E1001", f"duplicate declaration: {item.name}", item, "redeclared here")
                self._validate_type_names(field.type for field in item.fields)
                names: set[str] = set()
                for field in item.fields:
                    if field.name in names:
                        self._fail(
                            "E1003",
                            f"duplicate field `{field.name}` in record `{item.name}`",
                            field,
                            "duplicate field",
                        )
                    names.add(field.name)
                self.records[item.name] = item
            elif isinstance(item, Function):
                if item.name in self.functions or item.name in self.records:
                    self._fail("E1001", f"duplicate declaration: {item.name}", item, "redeclared here")
                self._validate_type_names([param.type for param in item.params])
                self._validate_type_names([item.return_type])
                self.functions[item.name] = FunctionSig(
                    item.name, tuple(param.type for param in item.params), item.return_type
                )
            elif isinstance(item, Theorem):
                if item.name in self.theorems or item.name in self.functions or item.name in self.records:
                    self._fail("E1001", f"duplicate declaration: {item.name}", item, "redeclared here")
                self._validate_type_names(param.type for param in item.params)
                self.theorems[item.name] = item

    def _validate_type_names(self, types: Iterable[TypeRef]) -> None:
        for type_ref in types:
            if type_ref.name not in BUILTIN_TYPES and type_ref.name not in self.records:
                self._fail(
                    "E1002",
                    f"unknown type: {type_ref.name}",
                    type_ref,
                    "not a known type",
                    ("built-in types are Int, Bool, String, Unit, Prop",),
                )

    # --------------------------------------------------------------- checking

    def _check_function(self, function: Function) -> None:
        env: dict[str, TypeRef] = {}
        for param in function.params:
            if param.name in env:
                self._fail("E1004", f"duplicate parameter: {param.name}", param, "duplicate parameter")
            env[param.name] = param.type
        for spec in function.specs:
            spec_env = dict(env)
            if isinstance(spec, Ensures):
                spec_env["result"] = function.return_type
            self._check_proposition(spec.proposition, spec_env)
        self._check_block(function.body, env, function.return_type, loop=None)

    def _check_theorem(self, theorem: Theorem) -> None:
        env = {param.name: param.type for param in theorem.params}
        for proposition in theorem.requires:
            self._check_proposition(proposition, env)
        self._check_proposition(theorem.proposition, env)
        for command in theorem.proof:
            if command.kind == "exact":
                name = command.value
                if not isinstance(name, str) or name not in self.theorems:
                    self._fail(
                        "E1025",
                        f"unknown theorem: {name}",
                        command,
                        "not a theorem",
                        ("`exact` refers to a theorem declared in this module",),
                    )

    def _check_block(
        self,
        statements: list,
        env: dict[str, TypeRef],
        return_type: TypeRef,
        loop: LoopStmt | None,
    ) -> None:
        for statement in statements:
            self._check_statement(statement, env, return_type, loop)

    def _check_statement(self, statement, env: dict[str, TypeRef], return_type: TypeRef, loop: LoopStmt | None) -> None:
        if isinstance(statement, LetStmt):
            if statement.name in env:
                self._fail("E1005", f"duplicate binding: {statement.name}", statement, "already bound")
            value_type = self._check_expr(statement.value, env)
            if not self._is_let_value(statement.value):
                # Defensive: the grammar already enforces the one-operator rule, so
                # this only fires for ASTs built programmatically. Keeping it makes
                # the invariant explicit for any future AST producer.
                self._fail(
                    "E1008",
                    "let right-hand side must be a single operator application over atomic operands",
                    statement.value,
                    "not a single operation",
                    (
                        "operands of an operator must be literals or names",
                        "bind intermediate results with their own `let`",
                    ),
                )
            if statement.type is not None and statement.type != value_type:
                self._fail(
                    "E1007",
                    f"let `{statement.name}` expects {statement.type.name}, got {value_type.name}",
                    statement.value,
                    f"this is {value_type.name}",
                )
            env[statement.name] = statement.type or value_type
        elif isinstance(statement, PrintStmt):
            value_type = self._check_expr(statement.value, env)
            if value_type.name not in SCALAR_TYPES:
                self._fail(
                    "E1009",
                    f"print does not support type {value_type.name}",
                    statement.value,
                    f"this is {value_type.name}",
                    ("print supports Int, Bool, String and Unit",),
                )
        elif isinstance(statement, AssertStmt):
            self._check_proposition(statement.proposition, env)
        elif isinstance(statement, ReturnStmt):
            actual = UNIT if statement.value is None else self._check_expr(statement.value, env)
            self._expect(actual, return_type, "return", statement)
        elif isinstance(statement, BreakStmt):
            if loop is None:
                self._fail("E1010", "break is only valid inside loop", statement, "no enclosing loop")
            if statement.value is not None:
                self._check_expr(statement.value, env)
        elif isinstance(statement, ContinueStmt):
            if loop is None:
                self._fail("E1011", "continue is only valid inside loop", statement, "no enclosing loop")
            if set(statement.bindings) != set(loop.bindings):
                expected = ", ".join(sorted(loop.bindings)) or "no state"
                actual = ", ".join(sorted(statement.bindings)) or "no state"
                self._fail(
                    "E1012",
                    "continue must provide exactly the loop state bindings",
                    statement,
                    f"loop state is {expected}, got {actual}",
                )
            for name, value in statement.bindings.items():
                self._expect(
                    self._check_expr(value, env),
                    self._check_expr(loop.bindings[name], env),
                    f"loop state `{name}`",
                    value,
                )
        elif isinstance(statement, ExprStmt):
            self._check_expr(statement.value, env)
        elif isinstance(statement, MatchStmt):
            value_type = self._check_expr(statement.value, env)
            for arm in statement.arms:
                arm_env = dict(env)
                self._check_pattern(arm.pattern, value_type, arm_env)
                if arm.guard is not None:
                    self._check_proposition(arm.guard, arm_env)
                self._check_block(arm.body, arm_env, return_type, loop)
        elif isinstance(statement, LoopStmt):
            loop_env = dict(env)
            for name, value in statement.bindings.items():
                if name in loop_env:
                    self._fail("E1020", f"loop state shadows existing binding: {name}", statement, "shadowing here")
                loop_env[name] = self._check_expr(value, env)
            for spec in statement.specs:
                if spec.kind == "invariant":
                    self._check_proposition(spec.value, loop_env)
                else:
                    variant_type = self._check_expr(spec.value, loop_env)
                    self._expect(variant_type, INT, "loop variant", spec.value)
            self._check_block(statement.body, loop_env, return_type, statement)
        else:
            self._fail("E1022", f"unsupported statement: {type(statement).__name__}", statement)

    def _check_pattern(self, pattern, expected: TypeRef, env: dict[str, TypeRef]) -> None:
        if isinstance(pattern, WildcardPattern):
            return
        if isinstance(pattern, BindingPattern):
            if pattern.name in env:
                self._fail("E1017", f"pattern binding shadows existing name: {pattern.name}", pattern, "shadowing here")
            env[pattern.name] = expected
        elif isinstance(pattern, LiteralPattern):
            self._expect(self._check_expr(pattern.value, env), expected, "pattern", pattern)
        elif isinstance(pattern, RecordPattern):
            if pattern.type_name != expected.name:
                self._fail(
                    "E1018",
                    f"pattern expects {pattern.type_name}, got {expected.name}",
                    pattern,
                    f"matched value is {expected.name}",
                )
            record = self.records.get(pattern.type_name)
            if record is None:
                self._fail("E1019", f"unknown record: {pattern.type_name}", pattern, "not a known record")
            fields = {field.name: field.type for field in record.fields}
            for name, nested in pattern.fields.items():
                if name not in fields:
                    self._fail(
                        "E1013",
                        f"unknown field `{name}` in pattern `{pattern.type_name}`",
                        nested,
                        "no such field",
                    )
                self._check_pattern(nested, fields[name], env)

    def _check_proposition(self, proposition: Proposition | Expr, env: dict[str, TypeRef]) -> None:
        expression = proposition.expression if isinstance(proposition, Proposition) else proposition
        node = proposition if isinstance(proposition, Proposition) else expression
        self._expect(self._check_expr(expression, env), BOOL, "proposition", node)

    def _check_expr(self, expr: Expr, env: dict[str, TypeRef]) -> TypeRef:
        if isinstance(expr, Name):
            if expr.value not in env:
                self._fail(
                    "E1006",
                    f"undefined name: {expr.value}",
                    expr,
                    "not defined",
                    ("declare it with `let` before use",),
                )
            return env[expr.value]
        if isinstance(expr, IntLiteral):
            return INT
        if isinstance(expr, BoolLiteral):
            return BOOL
        if isinstance(expr, StringLiteral):
            return STRING
        if isinstance(expr, UnitLiteral):
            return UNIT
        if isinstance(expr, Unary):
            operand = self._check_expr(expr.operand, env)
            if expr.op == "-":
                self._expect(operand, INT, "unary -", expr.operand)
                return INT
            if expr.op == "not":
                self._expect(operand, BOOL, "not", expr.operand)
                return BOOL
            self._fail("E1021", f"unknown unary operator: {expr.op}", expr)
        if isinstance(expr, Binary):
            left = self._check_expr(expr.left, env)
            right = self._check_expr(expr.right, env)
            if expr.op in {"+", "-", "*", "/", "%"}:
                self._expect(left, INT, expr.op, expr.left)
                self._expect(right, INT, expr.op, expr.right)
                return INT
            if expr.op in {"<", "<=", ">", ">=", "==", "!="}:
                self._expect(right, left, expr.op, expr.right)
                return BOOL
            if expr.op in {"and", "or", "=>"}:
                self._expect(left, BOOL, expr.op, expr.left)
                self._expect(right, BOOL, expr.op, expr.right)
                return BOOL
            self._fail("E1021", f"unknown binary operator: {expr.op}", expr)
        if isinstance(expr, Call):
            if not isinstance(expr.function, Name) or expr.function.value not in self.functions:
                self._fail("E1016", "only declared functions can be called", expr.function, "not a function")
            signature = self.functions[expr.function.value]
            if len(expr.arguments) != len(signature.params):
                expected_count = len(signature.params)
                noun = "argument" if expected_count == 1 else "arguments"
                self._fail(
                    "E1015",
                    f"function `{signature.name}` expects {expected_count} {noun}, got {len(expr.arguments)}",
                    expr,
                    "wrong number of arguments",
                )
            for argument, expected in zip(expr.arguments, signature.params):
                self._expect(self._check_expr(argument, env), expected, f"argument to `{signature.name}`", argument)
            return signature.return_type
        if isinstance(expr, FieldAccess):
            value_type = self._check_expr(expr.value, env)
            record = self.records.get(value_type.name)
            if record is None:
                self._fail(
                    "E1024",
                    f"field access requires record, got {value_type.name}",
                    expr.value,
                    f"this is {value_type.name}",
                )
            for field in record.fields:
                if field.name == expr.field:
                    return field.type
            self._fail(
                "E1013",
                f"record `{record.name}` has no field `{expr.field}`",
                expr,
                "no such field",
                ("available fields: " + ", ".join(field.name for field in record.fields),),
            )
        if isinstance(expr, RecordValue):
            record = self.records.get(expr.type_name)
            if record is None:
                self._fail("E1019", f"unknown record: {expr.type_name}", expr, "not a known record")
            expected = {field.name: field.type for field in record.fields}
            if set(expr.fields) != set(expected):
                missing = sorted(set(expected) - set(expr.fields))
                extra = sorted(set(expr.fields) - set(expected))
                detail = []
                if missing:
                    detail.append("missing " + ", ".join(missing))
                if extra:
                    detail.append("unknown " + ", ".join(extra))
                self._fail(
                    "E1014",
                    f"record `{expr.type_name}` fields do not match declaration",
                    expr,
                    "; ".join(detail) or "field mismatch",
                )
            for name, value in expr.fields.items():
                self._expect(self._check_expr(value, env), expected[name], f"field `{name}`", value)
            return TypeRef(expr.type_name)
        self._fail("E1023", f"unsupported expression: {type(expr).__name__}", expr)

    @staticmethod
    def _is_leaf(expr: Expr) -> bool:
        """An operand of an operator: a literal or a plain name."""
        return isinstance(expr, (IntLiteral, BoolLiteral, StringLiteral, UnitLiteral, Name))

    @classmethod
    def _is_let_value(cls, expr: Expr) -> bool:
        """A bound value is exactly one operator application.

        Operators, calls and record builds never nest: `a + b` and `f(a)` are
        fine, while `a + b + c`, `f(a) + b`, `g(f(a))`, `f(a + 1)` and
        `Point { x: a + 1 }` are not.
        """
        if cls._is_leaf(expr):
            return True
        if isinstance(expr, Unary):
            return cls._is_leaf(expr.operand)
        if isinstance(expr, Binary):
            return cls._is_leaf(expr.left) and cls._is_leaf(expr.right)
        if isinstance(expr, FieldAccess):
            return True
        if isinstance(expr, Call):
            return all(cls._is_leaf(argument) for argument in expr.arguments)
        if isinstance(expr, RecordValue):
            return all(cls._is_leaf(value) for value in expr.fields.values())
        return False

    @staticmethod
    def _expect(actual: TypeRef, expected: TypeRef, context: str, node: Any = None) -> None:
        if actual != expected:
            SemanticAnalyzer._fail(
                "E1007",
                f"{context} expects {expected.name}, got {actual.name}",
                node,
                f"expected {expected.name}, found {actual.name}",
            )


def analyze(module: Module) -> Module:
    return SemanticAnalyzer().analyze(module)
