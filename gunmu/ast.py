from dataclasses import dataclass, field
from typing import Optional, Union

from .diagnostics import Span


@dataclass(frozen=True)
class TypeRef:
    name: str
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Field:
    name: str
    type: TypeRef
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Param:
    name: str
    type: TypeRef
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Expr:
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Name(Expr):
    value: str


@dataclass(frozen=True)
class IntLiteral(Expr):
    value: int


@dataclass(frozen=True)
class StringLiteral(Expr):
    value: str


@dataclass(frozen=True)
class BoolLiteral(Expr):
    value: bool


@dataclass(frozen=True)
class UnitLiteral(Expr):
    pass


@dataclass(frozen=True)
class Unary(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class Binary(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass(frozen=True)
class Call(Expr):
    function: Expr
    arguments: list[Expr]


@dataclass(frozen=True)
class FieldAccess(Expr):
    value: Expr
    field: str


@dataclass(frozen=True)
class RecordValue(Expr):
    type_name: str
    fields: dict[str, Expr]


@dataclass(frozen=True)
class Pattern:
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class WildcardPattern(Pattern):
    pass


@dataclass(frozen=True)
class BindingPattern(Pattern):
    name: str


@dataclass(frozen=True)
class LiteralPattern(Pattern):
    value: Expr


@dataclass(frozen=True)
class RecordPattern(Pattern):
    type_name: str
    fields: dict[str, Pattern]
    rest: bool = False


@dataclass(frozen=True)
class Proposition:
    expression: Expr
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Stmt:
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class LetStmt(Stmt):
    name: str
    type: Optional[TypeRef]
    value: Expr


@dataclass(frozen=True)
class PrintStmt(Stmt):
    value: Expr


@dataclass(frozen=True)
class AssertStmt(Stmt):
    proposition: Proposition


@dataclass(frozen=True)
class ReturnStmt(Stmt):
    value: Optional[Expr]


@dataclass(frozen=True)
class BreakStmt(Stmt):
    value: Optional[Expr]


@dataclass(frozen=True)
class ContinueStmt(Stmt):
    bindings: dict[str, Expr]


@dataclass(frozen=True)
class ExprStmt(Stmt):
    value: Expr


@dataclass(frozen=True)
class MatchArm:
    pattern: Pattern
    guard: Optional[Proposition]
    body: list[Stmt]
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class MatchStmt(Stmt):
    value: Expr
    arms: list[MatchArm]


@dataclass(frozen=True)
class LoopSpec:
    kind: str
    value: Union[Proposition, Expr]
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class LoopStmt(Stmt):
    bindings: dict[str, Expr]
    specs: list[LoopSpec]
    body: list[Stmt]


@dataclass(frozen=True)
class Requires:
    proposition: Proposition
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Ensures:
    proposition: Proposition
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Function:
    name: str
    params: list[Param]
    return_type: TypeRef
    specs: list[Union[Requires, Ensures]]
    body: list[Stmt]
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Record:
    name: str
    fields: list[Field]
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class ProofCommand:
    kind: str
    value: Optional[Union[str, Proposition]] = None
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Theorem:
    name: str
    params: list[Param]
    requires: list[Proposition]
    proposition: Proposition
    proof: list[ProofCommand]
    span: Span | None = field(default=None, kw_only=True, compare=False)


@dataclass(frozen=True)
class Module:
    name: str
    items: list[Union[Record, Function, Theorem]] = field(default_factory=list)
    span: Span | None = field(default=None, kw_only=True, compare=False)
