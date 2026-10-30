from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from lark import Lark, Transformer, UnexpectedInput, Token
from lark.exceptions import UnexpectedCharacters, UnexpectedToken

from .ast import (
    AssertStmt, Binary, BindingPattern, BoolLiteral, BreakStmt, Call,
    ContinueStmt, Ensures, Expr, ExprStmt, Field, FieldAccess, Function,
    IntLiteral, LetStmt, LiteralPattern, LoopSpec, LoopStmt, MatchArm,
    MatchStmt, Module, Name, Param, PrintStmt, Proposition, Record,
    RecordPattern, RecordValue, ReturnStmt, StringLiteral, Theorem, TypeRef,
    Unary, UnitLiteral, WildcardPattern, Requires, ProofCommand,
)
from .diagnostics import Diagnostic, SourceFile, Span


GRAMMAR = r"""
start: module_decl item*

module_decl: MODULE module_name
module_name: IDENTIFIER (DOT IDENTIFIER)*

?item: record_decl | fn_decl | theorem_decl
record_decl: RECORD IDENTIFIER LBRACE field_decl* RBRACE
field_decl: IDENTIFIER COLON type

fn_decl: FN IDENTIFIER LPAREN [param_list] RPAREN ARROW type function_spec* block
param_list: param (COMMA param)*
param: IDENTIFIER COLON type
function_spec: requires_clause | ensures_clause
requires_clause: REQUIRES proposition
ensures_clause: ENSURES proposition

theorem_decl: THEOREM IDENTIFIER LPAREN [param_list] RPAREN theorem_requires* COLON proposition proof_block
theorem_requires: REQUIRES proposition -> theorem_requires_clause
proof_block: PROOF LBRACE proof_command* RBRACE
?proof_command: SIMPLE | OMEGA -> omega_command
              | ASSERT proposition -> proof_assert
              | EXACT IDENTIFIER -> exact_command

block: LBRACE statement* RBRACE
?statement: let_stmt | print_stmt | assert_stmt | return_stmt | break_stmt
          | continue_stmt | match_stmt | loop_stmt | expr_stmt
let_stmt: LET IDENTIFIER [COLON type] EQUAL let_value
print_stmt: PRINT let_value
assert_stmt: ASSERT proposition
return_stmt: RETURN [let_value]
break_stmt: BREAK [let_value]
continue_stmt: CONTINUE [LPAREN [state_binding_list] RPAREN]
state_binding_list: state_binding (COMMA state_binding)*
state_binding: IDENTIFIER EQUAL state_expr
state_expr: state_term ((PLUS|MINUS) state_term)*
state_term: atom_expr
expr_stmt: call_expr

// A bound value is exactly one operator application: an operator together with
// one or two atomic operands, or a single call / field path / record build.
// Operators never nest, and neither do calls or record fields, so `a + b` is a
// valid binding while `a + b + c`, `f(a) + b` and `f(a + 1)` are not.
?let_value: let_leaf
          | let_unary_op let_leaf -> value_unary
          | let_leaf let_binary_op let_leaf -> value_binary
          | field_access
          | let_call
          | let_record
?let_leaf: literal
         | IDENTIFIER -> name_expr
?let_unary_op: MINUS | NOT
?let_binary_op: PLUS | MINUS | STAR | SLASH | PERCENT | COMP_OP | AND | OR | IMPLIES
let_call: IDENTIFIER LPAREN [let_leaf_list] RPAREN
let_leaf_list: let_leaf (COMMA let_leaf)*
let_record: IDENTIFIER LBRACE [let_field (COMMA let_field)*] [COMMA] RBRACE
let_field: IDENTIFIER COLON let_leaf

match_stmt: MATCH match_value LBRACE match_arm+ RBRACE
?match_value: literal
            | IDENTIFIER -> name_expr
            | call_expr
            | field_access
match_arm: pattern [IF proposition] FAT_ARROW block

loop_stmt: LOOP LPAREN binding_list RPAREN loop_with_clause? block
loop_with_clause: WITH LBRACE loop_spec* RBRACE
binding_list: binding (COMMA binding)*
binding: IDENTIFIER EQUAL let_value
loop_spec: INVARIANT proposition -> invariant_spec
         | VARIANT sum -> variant_spec

?pattern: UNDERSCORE -> wildcard_pattern
        | literal -> literal_pattern
        | IDENTIFIER -> binding_pattern
# `RecordPattern` is modelled, checked, hashed and printed, but no arm can
# reach this rule yet: hooking it up needs the field list to accept commas.
record_pattern: IDENTIFIER LBRACE pattern_field* [DOT DOT] RBRACE
pattern_field: IDENTIFIER COLON pattern
             | IDENTIFIER -> shorthand_pattern

?proposition: implication
?implication: disjunction [IMPLIES implication]
?disjunction: conjunction (OR conjunction)*
?conjunction: comparison (AND comparison)*
?comparison: sum (COMP_OP sum)*
?sum: product ((PLUS|MINUS) product)*
?product: unary ((STAR|SLASH|PERCENT) unary)*
?unary: NOT unary -> unary_expr
      | MINUS unary -> unary_expr
      | atom_expr

?atom_expr: literal
          | IDENTIFIER -> name_expr
          | call_expr
          | field_access
          | LPAREN proposition RPAREN
call_expr: IDENTIFIER LPAREN [argument_list] RPAREN
field_access: IDENTIFIER (DOT IDENTIFIER)+
// `record_expr` is deliberately absent from `atom_expr`. Including it made
// LALR shift a following `{` into a record value, so `ensures result > x {`
// failed to parse even though `x` cannot start a record. Nothing is lost: the
// only propositions that could hold a record are comparisons and sums, which
// need an Int or a Bool there. Binding positions spell records with
// `let_record`, which is a separate rule.
record_expr: IDENTIFIER LBRACE [record_value_field (COMMA record_value_field)*] [COMMA] RBRACE
record_value_field: IDENTIFIER COLON proposition
argument_list: proposition (COMMA proposition)*

?literal: INT -> int_literal
         | STRING -> string_literal
         | TRUE -> true_literal
         | FALSE -> false_literal
         | UNIT -> unit_literal

type: IDENTIFIER

WITH.10: "with"
MODULE.10: "module"
RECORD.10: "record"
FN.10: "fn"
THEOREM.10: "theorem"
PROOF.10: "proof"
LET.10: "let"
MATCH.10: "match"
LOOP.10: "loop"
BREAK.10: "break"
CONTINUE.10: "continue"
RETURN.10: "return"
REQUIRES.10: "requires"
ENSURES.10: "ensures"
ASSERT.10: "assert"
INVARIANT.10: "invariant"
VARIANT.10: "variant"
PRINT.10: "print"
SIMPLE.10: "simp"
OMEGA.10: "omega"
EXACT.10: "exact"
TRUE.10: "true"
FALSE.10: "false"
UNIT.10: "unit"
AND.10: "and"
OR.10: "or"
NOT.10: "not"
IF.10: "if"
IMPLIES.10: "implies"
FAT_ARROW: "=>"
ARROW: "->"
COMP_OP: "==" | "!=" | "<=" | ">=" | "<" | ">"
PLUS: "+"
MINUS: "-"
STAR: "*"
SLASH: "/"
PERCENT: "%"
EQUAL: "="
COLON: ":"
DOT: "."
COMMA: ","
LBRACE: "{"
RBRACE: "}"
LPAREN: "("
RPAREN: ")"
UNDERSCORE: "_"

IDENTIFIER: /(?!with\\b|module\\b|record\\b|fn\\b|theorem\\b|proof\\b|let\\b|match\\b|loop\\b|break\\b|continue\\b|return\\b|requires\\b|ensures\\b|assert\\b|invariant\\b|variant\\b|print\\b|simp\\b|omega\\b|exact\\b|true\\b|false\\b|unit\\b|and\\b|or\\b|not\\b|if\\b|implies\\b)[a-zA-Z_][a-zA-Z0-9_]*/
%import common.INT
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
%ignore /\/\/[^\n]*/
"""


class AstTransformer(Transformer[Any, Any]):
    def _call_userfunc(self, tree, new_children=None):
        result = super()._call_userfunc(tree, new_children)
        return _attach_span(tree, result)

    def start(self, items: list[Any]) -> Module:
        return Module(name=items[0], items=list(items[1:]))

    def module_decl(self, items: list[Any]) -> str:
        return str(items[-1])

    def module_name(self, items: list[Any]) -> str:
        return ".".join(str(item) for item in items if str(item) != ".")

    @staticmethod
    def _tokens(items: list[Any]) -> list[str]:
        return [str(item) for item in items if isinstance(item, Token) and item.type == "IDENTIFIER"]

    @staticmethod
    def _values(items: list[Any]) -> list[Any]:
        syntax_tokens = {
            "MODULE", "RECORD", "FN", "THEOREM", "PROOF", "LET", "MATCH", "LOOP",
            "BREAK", "CONTINUE", "RETURN", "REQUIRES", "ENSURES", "ASSERT", "INVARIANT",
            "VARIANT", "PRINT", "SIMPLE", "OMEGA", "EXACT", "WITH", "IF", "FAT_ARROW", "ARROW",
            "COLON", "EQUAL", "DOT", "COMMA", "LBRACE", "RBRACE", "LPAREN", "RPAREN",
            "AND", "OR", "NOT", "IMPLIES", "COMP_OP", "PLUS", "MINUS", "STAR", "SLASH",
            "PERCENT", "UNDERSCORE",
        }
        return [
            item for item in items
            if item is not None and not (isinstance(item, Token) and item.type in syntax_tokens)
        ]

    def record_decl(self, items: list[Any]) -> Record:
        values = self._values(items)
        return Record(name=str(values[0]), fields=list(values[1:]))

    def field_decl(self, items: list[Any]) -> Field:
        values = self._values(items)
        return Field(name=str(values[0]), type=values[1])

    def fn_decl(self, items: list[Any]) -> Function:
        values = self._values(items)
        name = str(values[0])
        index = 1
        params: list[Param] = []
        if index < len(values) and isinstance(values[index], list):
            params = values[index]
            index += 1
        return_type = values[index]
        index += 1
        specs = []
        while index < len(values) - 1:
            if values[index] is not None:
                specs.append(values[index])
            index += 1
        return Function(name, params, return_type, specs, values[index])

    def theorem_requires_clause(self, items: list[Any]) -> tuple[str, Proposition]:
        values = self._values(items)
        return ("requires", values[0])

    def theorem_decl(self, items: list[Any]) -> Theorem:
        values = self._values(items)
        name = str(values[0])
        index = 1
        params: list[Param] = []
        if index < len(values) and isinstance(values[index], list):
            params = values[index]
            index += 1
        requires: list[Proposition] = []
        while index < len(values) and isinstance(values[index], tuple):
            requires.append(values[index][1])
            index += 1
        proposition = values[index]
        proof_value = values[index + 1]
        proof = [item for item in proof_value if isinstance(item, ProofCommand)]
        return Theorem(name, params, requires, proposition, proof)

    def param_list(self, items: list[Any]) -> list[Param]:
        return [item for item in items if isinstance(item, Param)]

    def param(self, items: list[Any]) -> Param:
        values = self._values(items)
        return Param(str(values[0]), values[1])

    def type(self, items: list[Any]) -> TypeRef:
        return TypeRef(str(items[0]))

    def function_spec(self, items: list[Any]) -> Any:
        return self._values(items)[0]

    def requires_clause(self, items: list[Any]) -> Requires:
        values = self._values(items)
        return Requires(values[0])

    def ensures_clause(self, items: list[Any]) -> Ensures:
        values = self._values(items)
        return Ensures(values[0])

    def block(self, items: list[Any]) -> list[Any]:
        return self._values(items)

    def match_stmt(self, items: list[Any]) -> MatchStmt:
        values = self._values(items)
        return MatchStmt(values[0], list(values[1:]))

    def match_arm(self, items: list[Any]) -> MatchArm:
        values = self._values(items)
        if len(values) == 2:
            return MatchArm(values[0], None, values[1])
        return MatchArm(values[0], values[1], values[2])
    def let_stmt(self, items: list[Any]) -> LetStmt:
        values = self._values(items)
        name = str(values[0])
        if len(values) == 2:
            return LetStmt(name, None, values[1])
        return LetStmt(name, values[1], values[2])

    def print_stmt(self, items: list[Any]) -> PrintStmt:
        return PrintStmt(self._values(items)[0])

    def assert_stmt(self, items: list[Any]) -> AssertStmt:
        return AssertStmt(self._values(items)[0])

    def return_stmt(self, items: list[Any]) -> ReturnStmt:
        values = self._values(items)
        return ReturnStmt(values[0] if values else None)

    def break_stmt(self, items: list[Any]) -> BreakStmt:
        values = self._values(items)
        return BreakStmt(values[0] if values else None)

    def continue_stmt(self, items: list[Any]) -> ContinueStmt:
        values = self._values(items)
        return ContinueStmt(values[0] if values else {})

    def expr_stmt(self, items: list[Any]) -> ExprStmt:
        return ExprStmt(self._values(items)[0])

    def value_unary(self, items: list[Any]) -> Unary:
        return Unary(str(items[0]), items[1])

    def value_binary(self, items: list[Any]) -> Binary:
        return Binary(items[0], str(items[1]), items[2])

    def let_call(self, items: list[Any]) -> Call:
        values = self._values(items)
        arguments = next((value for value in values[1:] if isinstance(value, list)), [])
        return Call(Name(str(values[0])), arguments)

    def let_leaf_list(self, items: list[Any]) -> list[Expr]:
        return self._values(items)

    def let_record(self, items: list[Any]) -> RecordValue:
        values = self._values(items)
        fields = [value for value in values[1:] if isinstance(value, tuple)]
        return RecordValue(str(values[0]), dict(fields))

    def let_field(self, items: list[Any]) -> tuple[str, Expr]:
        values = self._values(items)
        return str(values[0]), values[1]

    def state_expr(self, items: list[Any]) -> Expr:
        return self._fold(list(items), None)

    def state_term(self, items: list[Any]) -> Expr:
        return self._values(items)[0]

    def state_binding_list(self, items: list[Any]) -> dict[str, Expr]:
        return dict(self._values(items))

    def state_binding(self, items: list[Any]) -> tuple[str, Expr]:
        values = self._values(items)
        return str(values[0]), values[1]

    def loop_with_clause(self, items: list[Any]) -> list[LoopSpec]:
        return [item for item in self._values(items) if isinstance(item, LoopSpec)]

    def binding_list(self, items: list[Any]) -> dict[str, Expr]:
        return dict(self._values(items))

    def binding(self, items: list[Any]) -> tuple[str, Expr]:
        values = self._values(items)
        return str(values[0]), values[1]

    def loop_stmt(self, items: list[Any]) -> LoopStmt:
        values = self._values(items)
        bindings = values[0]
        if len(values) == 2:
            return LoopStmt(bindings, [], values[1])
        return LoopStmt(bindings, values[1], values[2])

    def invariant_spec(self, items: list[Any]) -> LoopSpec:
        return LoopSpec("invariant", self._values(items)[0])

    def variant_spec(self, items: list[Any]) -> LoopSpec:
        return LoopSpec("variant", self._values(items)[0])

    def wildcard_pattern(self, _: list[Any]) -> WildcardPattern:
        return WildcardPattern()

    def binding_pattern(self, items: list[Any]) -> BindingPattern:
        return BindingPattern(str(items[0]))

    def literal_pattern(self, items: list[Any]) -> LiteralPattern:
        return LiteralPattern(items[0])

    def record_pattern(self, items: list[Any]) -> RecordPattern:
        fields = dict(items[1:-1] if items and isinstance(items[-1], bool) else items[1:])
        rest = bool(items[-1]) if items and isinstance(items[-1], bool) else False
        return RecordPattern(str(items[0]), fields, rest)

    def pattern_field(self, items: list[Any]) -> tuple[str, Any]:
        if len(items) == 1:
            return str(items[0]), BindingPattern(str(items[0]))
        return str(items[0]), items[1]

    def proposition(self, items: list[Any]) -> Proposition:
        return Proposition(items[0])

    def implication(self, items: list[Any]) -> Expr:
        values = self._values(items)
        if len(values) == 1:
            return values[0]
        return Binary(values[0], "=>", values[1])

    def disjunction(self, items: list[Any]) -> Expr:
        return self._fold(items, "or")

    def conjunction(self, items: list[Any]) -> Expr:
        return self._fold(items, "and")

    def comparison(self, items: list[Any]) -> Expr:
        return self._fold(items, None)

    def sum(self, items: list[Any]) -> Expr:
        return self._fold(items, None)

    def product(self, items: list[Any]) -> Expr:
        return self._fold(items, None)

    @staticmethod
    def _fold(items: list[Any], default_op: str | None) -> Expr:
        value = items[0]
        for index in range(1, len(items), 2):
            operator = str(items[index]) if default_op is None else default_op
            value = Binary(value, operator, items[index + 1])
        return value

    def unary_expr(self, items: list[Any]) -> Unary:
        return Unary(str(items[0]), items[1])

    def name_expr(self, items: list[Any]) -> Name:
        return Name(str(items[0]))

    def call_expr(self, items: list[Any]) -> Call:
        values = self._values(items)
        arguments = next((value for value in values[1:] if isinstance(value, list)), [])
        return Call(Name(str(values[0])), arguments)

    def field_access(self, items: list[Any]) -> FieldAccess:
        names = [str(item) for item in items if isinstance(item, Token) and item.type == "IDENTIFIER"]
        value: Expr = Name(names[0])
        for field in names[1:]:
            value = FieldAccess(value, field)
        return value

    def record_expr(self, items: list[Any]) -> RecordValue:
        values = self._values(items)
        fields = [value for value in values[1:] if isinstance(value, tuple)]
        return RecordValue(str(values[0]), dict(fields))

    def record_value_field(self, items: list[Any]) -> tuple[str, Expr]:
        values = self._values(items)
        return str(values[0]), values[1]

    def argument_list(self, items: list[Any]) -> list[Expr]:
        return self._values(items)

    def int_literal(self, items: list[Any]) -> IntLiteral:
        return IntLiteral(int(items[0]))

    def string_literal(self, items: list[Any]) -> StringLiteral:
        import json
        return StringLiteral(json.loads(str(items[0])))

    def true_literal(self, _: list[Any]) -> BoolLiteral:
        return BoolLiteral(True)

    def false_literal(self, _: list[Any]) -> BoolLiteral:
        return BoolLiteral(False)

    def unit_literal(self, _: list[Any]) -> UnitLiteral:
        return UnitLiteral()

    def proof_block(self, items: list[Any]) -> list[ProofCommand]:
        return list(items)

    def proof_assert(self, items: list[Any]) -> ProofCommand:
        values = self._values(items)
        return ProofCommand("assert", values[0])

    def omega_command(self, _: list[Any]) -> ProofCommand:
        return ProofCommand("omega")

    def exact_command(self, items: list[Any]) -> ProofCommand:
        # `EXACT` is a named terminal, so it survives into the children as a
        # token. Read past it or the referenced name comes back as "exact".
        values = self._values(items)
        return ProofCommand("exact", str(values[0]))

    def SIMPLE(self, _: list[Any]) -> ProofCommand:
        return ProofCommand("simp")


def _attach_span(tree, result: Any) -> Any:
    """Copy parse-tree position metadata onto span-bearing AST dataclasses."""
    if isinstance(result, list) or not hasattr(result, "__dataclass_fields__"):
        return result
    if "span" not in result.__dataclass_fields__ or getattr(result, "span", None) is not None:
        return result
    meta = getattr(tree, "meta", None)
    if meta is None or getattr(meta, "empty", True):
        return result
    try:
        span = Span.from_meta(meta)
    except AttributeError:
        return result
    result = replace(result, span=span)
    if isinstance(result, FieldAccess):
        base = result
        while isinstance(base, FieldAccess):
            base = base.value
        if isinstance(base, Name) and base.span is None:
            result = replace(result, value=replace(base, span=span))
    return result


class ParseError(ValueError):
    """A parser-facing error carrying a stable diagnostic with source position."""

    code = "E0001"

    def __init__(
        self,
        message: str,
        span: Span | None = None,
        code: str | None = None,
        notes: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.message = message
        self.span = span
        self.code = code or self.code
        # Backwards-compatible attributes.
        self.line = span.line if span is not None else None
        self.column = span.column if span is not None else None
        self.diagnostic = Diagnostic.error(
            self.code,
            message,
            span,
            label="unexpected here",
            notes=notes,
        )


_TERMINAL_DISPLAY: dict[str, str] = {
    "LBRACE": "'{'",
    "RBRACE": "'}'",
    "LPAREN": "'('",
    "RPAREN": "')'",
    "COMMA": "','",
    "COLON": "':'",
    "DOT": "'.'",
    "EQUAL": "'='",
    "FAT_ARROW": "'=>'",
    "ARROW": "'->'",
    "COMP_OP": "comparison operator",
    "PLUS": "'+'",
    "MINUS": "'-'",
    "STAR": "'*'",
    "SLASH": "'/'",
    "PERCENT": "'%'",
    "UNDERSCORE": "'_'",
    "IDENTIFIER": "identifier",
    "INT": "integer",
    "STRING": "string",
    "MODULE": "'module'",
    "RECORD": "'record'",
    "FN": "'fn'",
    "THEOREM": "'theorem'",
    "PROOF": "'proof'",
    "LET": "'let'",
    "MATCH": "'match'",
    "LOOP": "'loop'",
    "BREAK": "'break'",
    "CONTINUE": "'continue'",
    "RETURN": "'return'",
    "REQUIRES": "'requires'",
    "ENSURES": "'ensures'",
    "ASSERT": "'assert'",
    "INVARIANT": "'invariant'",
    "VARIANT": "'variant'",
    "PRINT": "'print'",
    "WITH": "'with'",
    "IF": "'if'",
    "AND": "'and'",
    "OR": "'or'",
    "NOT": "'not'",
    "IMPLIES": "'implies'",
    "SIMPLE": "'simp'",
    "OMEGA": "'omega'",
    "EXACT": "'exact'",
    "TRUE": "'true'",
    "FALSE": "'false'",
    "UNIT": "'unit'",
    "COMMENT": "comment",
    "$END": "end of file",
}


_STRUCTURAL = ("RBRACE", "RPAREN", "LBRACE", "LPAREN", "COMMA", "COLON", "EQUAL", "FAT_ARROW", "ARROW")


def _describe_expected(expected: set[str] | None, limit: int = 5) -> str:
    if not expected:
        return "valid syntax"
    ordered = [name for name in _STRUCTURAL if name in expected]
    ordered += sorted(name for name in expected if name not in _STRUCTURAL)
    names = [_TERMINAL_DISPLAY.get(name, name) for name in ordered]
    if len(names) > limit:
        remaining = len(names) - limit
        return ", ".join(names[:limit]) + f" (and {remaining} more)"
    return ", ".join(names)


_ARITHMETIC_TOKENS = {"PLUS", "MINUS", "STAR", "SLASH", "PERCENT", "COMP_OP", "AND", "OR"}


def _line_text(source: str, line: int) -> str:
    lines = source.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def _context_notes(error: UnexpectedInput, source: str) -> tuple[str, ...]:
    """Add targeted hints for common agent mistakes."""
    token = getattr(error, "token", None)
    if token is None or getattr(token, "type", None) not in _ARITHMETIC_TOKENS:
        return ()
    line = getattr(token, "line", None)
    column = getattr(token, "column", None)
    if line is None or column is None:
        return ()
    text = _line_text(source, line)
    before = text[: column - 1]
    if "let" in before and "=" in before:
        return (
            "a bound value is a single operator application",
            "operators never nest: bind each intermediate result with its own `let`",
        )
    return ()


def _error_span(error: UnexpectedInput) -> Span:
    token = getattr(error, "token", None)
    if token is not None:
        return Span(
            start=getattr(token, "start_pos", None),
            end=getattr(token, "end_pos", None),
            line=token.line,
            column=token.column,
            end_line=getattr(token, "end_line", None),
            end_column=getattr(token, "end_column", None),
        )
    line = getattr(error, "line", 1) or 1
    column = getattr(error, "column", 1) or 1
    return Span.at(line, column)


def _parse_error(error: UnexpectedInput, source: str = "") -> ParseError:
    span = _error_span(error)
    if isinstance(error, UnexpectedToken):
        found = getattr(error, "token", None)
        text = str(found) if found is not None else "input"
        if getattr(found, "type", None) == "$END":
            message = f"unexpected end of file; expected {_describe_expected(getattr(error, 'expected', None))}"
        else:
            message = (
                f"unexpected {text!r}; expected {_describe_expected(getattr(error, 'expected', None))}"
            )
        code = "E0001"
    elif isinstance(error, UnexpectedCharacters):
        text = error.char
        message = f"unexpected character {text!r}; expected {_describe_expected(getattr(error, 'allowed', None))}"
        code = "E0002"
    else:
        message = "could not parse input"
        code = "E0003"
    notes = _context_notes(error, source) or ("check for a missing `{`, `}`, `(`, `)` or `,`",)
    return ParseError(message, span, code, notes)


_parser = Lark(GRAMMAR, parser="lalr", lexer="contextual", start="start", propagate_positions=True)


def parse(source: str) -> Module:
    try:
        tree = _parser.parse(source)
    except UnexpectedInput as error:
        raise _parse_error(error, source) from error
    return AstTransformer().transform(tree)


def load(path: str | Path) -> SourceFile:
    return SourceFile.read(path)


def parse_source_file(source_file: SourceFile) -> Module:
    return parse(source_file.text)


def parse_file(path: str | Path) -> Module:
    return parse(SourceFile.read(path).text)
