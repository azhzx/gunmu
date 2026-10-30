"""Discharging theorem statements with Z3.

A `theorem` is proved by refutation: the goal is negated and the solver is asked
for a model. A model is a counterexample, so the engine only ever claims
`PROVED` when the negation is unsatisfiable under the theorem's `requires`.

Two things this module deliberately refuses to do:

* It never guesses the sort of a free name. Callers pass the declared type of
  every parameter, so a `Bool` parameter is a Boolean and a `String` parameter
  is a sequence. A name the caller did not declare falls back to `Int`, which is
  what the standalone `prove(expr)` entry point wants.
* It never hands a goal to the solver under a name that promises more than the
  solver is being asked for. `omega` decides quantifier-free linear integer
  arithmetic, so `omega` first checks the goal really is in that fragment and
  reports the reason when it is not; `simp` is the unrestricted escape hatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping

import z3

from .ast import (
    Binary, BoolLiteral, Call, Expr, FieldAccess, IntLiteral, Name, Proposition,
    RecordValue, StringLiteral, Unary, UnitLiteral, Theorem,
)


class ProofStatus(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


#: How badly a result went, used to aggregate several obligations into one status.
_SEVERITY = {
    ProofStatus.PROVED: 0,
    ProofStatus.UNSUPPORTED: 1,
    ProofStatus.UNKNOWN: 2,
    ProofStatus.DISPROVED: 3,
}


@dataclass(frozen=True)
class ProofResult:
    status: ProofStatus
    message: str = ""
    model: str | None = None


def worst(results: Iterable[ProofResult]) -> ProofStatus:
    """The status of a group of obligations: the most severe one wins."""
    statuses = [result.status for result in results]
    if not statuses:
        return ProofStatus.PROVED
    return max(statuses, key=lambda status: _SEVERITY[status])


class ProofError(ValueError):
    pass


def expression_of(proposition: Proposition | Expr) -> Expr:
    """The expression inside a proposition.

    `Proposition` is a wrapper the parser mostly elides, so specs and goals are
    frequently bare expressions. Accept both rather than making every caller
    care which one it is holding.
    """
    return proposition.expression if isinstance(proposition, Proposition) else proposition


#: Types with a first-order encoding. `Unit` and `Prop` have no inhabitants worth
#: reasoning about and records are nominal, so neither is given a sort.
_SORTS = {
    "Int": z3.IntSort(),
    "Bool": z3.BoolSort(),
    "String": z3.StringSort(),
}


def sort_of(type_name: str) -> z3.SortRef:
    try:
        return _SORTS[type_name]
    except KeyError:
        raise ProofError(
            f"no logical encoding for type `{type_name}`: "
            f"only {', '.join(sorted(_SORTS))} can be reasoned about"
        ) from None


class Z3Translator:
    """Translate propositions into Z3 terms.

    `types` gives the declared type of each free name, `bindings` pins a name to
    an already-translated term. Bindings take priority, which is how a call
    site substitutes actual arguments for a callee's parameters. `calls`, when
    given, decides what a call means instead of rejecting it; that is how the
    verifier models a callee's contract.
    """

    def __init__(
        self,
        types: Mapping[str, str] | None = None,
        bindings: Mapping[str, z3.ExprRef] | None = None,
        calls: "Callable[[Call], z3.ExprRef] | None" = None,
    ) -> None:
        self.symbols: dict[str, z3.ExprRef] = {}
        self.types: dict[str, str] = dict(types or {})
        self.bindings: dict[str, z3.ExprRef] = dict(bindings or {})
        self.calls = calls

    def translate(self, expr: Expr) -> z3.ExprRef:
        if isinstance(expr, IntLiteral):
            return z3.IntVal(expr.value)
        if isinstance(expr, BoolLiteral):
            return z3.BoolVal(expr.value)
        if isinstance(expr, StringLiteral):
            return z3.StringVal(expr.value)
        if isinstance(expr, UnitLiteral):
            raise ProofError("Unit expressions have no logical encoding")
        if isinstance(expr, Name):
            return self._symbol(expr.value)
        if isinstance(expr, Unary):
            operand = self.translate(expr.operand)
            if expr.op == "-":
                return -operand
            if expr.op == "not":
                return z3.Not(self._bool(operand))
            raise ProofError(f"unsupported unary operator: {expr.op}")
        if isinstance(expr, Binary):
            left = self.translate(expr.left)
            right = self.translate(expr.right)
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "/":
                # Z3's integer division is total, so `10 / x * x == 10` "fails"
                # at x = 0. Gunmu has no defined meaning for `/`, so agreeing
                # with the solver here would be answering a different question.
                raise ProofError("division has no defined semantics in Gunmu")
            if expr.op == "%":
                raise ProofError("remainder has no defined semantics in Gunmu")
            if expr.op == "==":
                return left == right
            if expr.op == "!=":
                return left != right
            if expr.op == "<":
                return left < right
            if expr.op == "<=":
                return left <= right
            if expr.op == ">":
                return left > right
            if expr.op == ">=":
                return left >= right
            if expr.op == "and":
                return z3.And(self._bool(left), self._bool(right))
            if expr.op == "or":
                return z3.Or(self._bool(left), self._bool(right))
            if expr.op == "=>":
                return z3.Implies(self._bool(left), self._bool(right))
            raise ProofError(f"unsupported binary operator: {expr.op}")
        if isinstance(expr, Call):
            if self.calls is not None:
                return self.calls(expr)
            raise ProofError(f"function calls are not supported in propositions: `{_call_name(expr)}`")
        if isinstance(expr, FieldAccess):
            raise ProofError("record fields have no logical encoding")
        if isinstance(expr, RecordValue):
            raise ProofError("record values have no logical encoding")
        raise ProofError(f"expression is not supported by Z3: {type(expr).__name__}")

    def _symbol(self, name: str) -> z3.ExprRef:
        bound = self.bindings.get(name)
        if bound is not None:
            return bound
        symbol = self.symbols.get(name)
        if symbol is None:
            symbol = z3.Const(name, sort_of(self.types.get(name, "Int")))
            self.symbols[name] = symbol
        return symbol

    @staticmethod
    def _bool(value: z3.ExprRef) -> z3.BoolRef:
        if not z3.is_bool(value):
            raise ProofError("expected a boolean proposition")
        return value


def _call_name(call: Call) -> str:
    if isinstance(call.function, Name):
        return call.function.value
    return "<expression>"


# --------------------------------------------------------------- omega fragment

_ARITHMETIC = {"+", "-"}
_COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
_CONNECTIVES = {"and", "or", "=>"}


class _NotLinear(Exception):
    """Why a goal is not linear integer arithmetic. The message is the reason."""


def presburger_violation(
    expressions: Iterable[Expr],
    types: Mapping[str, str],
) -> str | None:
    """Why `expressions` are not linear integer arithmetic, or None if they are.

    `omega` is the omega test, which decides quantifier-free linear integer
    arithmetic over `Int`. Anything outside that fragment would be decided by
    the solver's general machinery while the source said `omega`, so it is
    reported instead. Multiplication by a literal stays linear, which is what
    makes goals like `2 * x >= 0` decidable by an omega test at all.
    """
    for expression in expressions:
        try:
            _require_linear(expression, types)
        except _NotLinear as reason:
            return str(reason)
    return None


def _require_linear(expr: Expr, types: Mapping[str, str]) -> None:
    """Raise `_NotLinear` unless `expr` is a linear integer expression."""
    if isinstance(expr, (IntLiteral, BoolLiteral)):
        return
    if isinstance(expr, Name):
        declared = types.get(expr.value, "Int")
        if declared != "Int":
            raise _NotLinear(f"`{expr.value}` is {declared}, not Int")
        return
    if isinstance(expr, Unary):
        if expr.op == "-":
            _require_linear(expr.operand, types)
            return
        if expr.op == "not":
            _require_boolean(expr.operand, types)
            return
        raise _NotLinear(f"unsupported unary operator `{expr.op}`")
    if isinstance(expr, Binary):
        if expr.op in _ARITHMETIC:
            _require_linear(expr.left, types)
            _require_linear(expr.right, types)
            return
        if expr.op == "*":
            _require_linear(expr.left, types)
            _require_linear(expr.right, types)
            if not (_is_integer_literal(expr.left) or _is_integer_literal(expr.right)):
                raise _NotLinear("multiplying two variables is not linear")
            return
        if expr.op in {"/", "%"}:
            raise _NotLinear(f"`{expr.op}` is not linear")
        if expr.op in _COMPARISONS:
            _require_linear(expr.left, types)
            _require_linear(expr.right, types)
            return
        if expr.op in _CONNECTIVES:
            _require_boolean(expr.left, types)
            _require_boolean(expr.right, types)
            return
        raise _NotLinear(f"unsupported operator `{expr.op}`")
    if isinstance(expr, StringLiteral):
        raise _NotLinear("strings are outside linear integer arithmetic")
    if isinstance(expr, Call):
        raise _NotLinear(f"call to `{_call_name(expr)}` is outside linear integer arithmetic")
    if isinstance(expr, (FieldAccess, RecordValue)):
        raise _NotLinear("records are outside linear integer arithmetic")
    if isinstance(expr, UnitLiteral):
        raise _NotLinear("Unit has no logical encoding")
    raise _NotLinear(f"unsupported expression: {type(expr).__name__}")


def _require_boolean(expr: Expr, types: Mapping[str, str]) -> None:
    """Raise `_NotLinear` unless `expr` is a Boolean over linear integers."""
    if isinstance(expr, BoolLiteral):
        return
    if isinstance(expr, Name):
        declared = types.get(expr.value, "Int")
        if declared != "Bool":
            raise _NotLinear(f"`{expr.value}` is {declared}, not Bool")
        return
    if isinstance(expr, Unary) and expr.op == "not":
        _require_boolean(expr.operand, types)
        return
    if isinstance(expr, Binary):
        if expr.op in _COMPARISONS:
            _require_linear(expr.left, types)
            _require_linear(expr.right, types)
            return
        if expr.op in _CONNECTIVES:
            _require_boolean(expr.left, types)
            _require_boolean(expr.right, types)
            return
    if isinstance(expr, (StringLiteral, Call, FieldAccess, RecordValue, UnitLiteral)):
        # Not a Boolean, but a specific reason beats a generic one.
        _require_linear(expr, types)
    raise _NotLinear("expected a Boolean over linear integers")


def _is_integer_literal(expr: Expr) -> bool:
    return isinstance(expr, IntLiteral) or (
        isinstance(expr, Unary) and expr.op == "-" and isinstance(expr.operand, IntLiteral)
    )


class Prover:
    """Discharges one goal at a time, or a whole theorem's proof block."""

    def __init__(self, timeout_ms: int = 2000) -> None:
        self.timeout_ms = timeout_ms

    # ------------------------------------------------------------------ single

    def prove(
        self,
        proposition: Proposition | Expr,
        assumptions: Iterable[Proposition | Expr] = (),
        types: Mapping[str, str] | None = None,
    ) -> ProofResult:
        """Check one goal under `assumptions` by refutation.

        `types` declares the sort of each free name; without it every name is
        assumed to be an `Int`.
        """
        translator = Z3Translator(types)
        try:
            goal = translator.translate(expression_of(proposition))
            if not z3.is_bool(goal):
                return ProofResult(ProofStatus.UNSUPPORTED, "proof target is not Bool")
            context = [
                self._as_bool(translator.translate(expression_of(item)))
                for item in assumptions
            ]
            return self._refute(goal, context, translator.symbols)
        except (ProofError, z3.Z3Exception) as error:
            return ProofResult(ProofStatus.UNSUPPORTED, str(error))

    def check_implication(
        self,
        context: z3.BoolRef,
        goal: z3.BoolRef,
        symbols: Mapping[str, z3.ExprRef] | None = None,
    ) -> ProofResult:
        """Check `context -> goal` where both sides are already solver terms.

        This is the entry point for generated verification conditions, which are
        built as terms rather than parsed from source. `symbols` names the free
        variables a counterexample may report.
        """
        try:
            if not z3.is_bool(context):
                return ProofResult(ProofStatus.UNSUPPORTED, "verification context is not Bool")
            if not z3.is_bool(goal):
                return ProofResult(ProofStatus.UNSUPPORTED, "verification condition is not Bool")
            assumptions = [] if z3.is_true(context) else [context]
            return self._refute(goal, assumptions, symbols or {})
        except (ProofError, z3.Z3Exception) as error:
            return ProofResult(ProofStatus.UNSUPPORTED, str(error))

    def _refute(
        self,
        goal: z3.BoolRef,
        context: list[z3.BoolRef],
        symbols: Mapping[str, z3.ExprRef],
    ) -> ProofResult:
        solver = z3.Solver()
        solver.set(timeout=self.timeout_ms)
        solver.add(*context)
        solver.push()
        solver.add(z3.Not(goal))
        try:
            verdict = solver.check()
            # The model is only readable inside the pushed scope.
            counterexample = solver.model() if verdict == z3.sat else None
        finally:
            solver.pop()
        if verdict == z3.sat:
            model = self._format_model(counterexample, symbols)
            return ProofResult(ProofStatus.DISPROVED, "counterexample found", model)
        if verdict == z3.unsat:
            return ProofResult(ProofStatus.PROVED, "proved")
        return ProofResult(ProofStatus.UNKNOWN, "solver returned unknown")

    @staticmethod
    def _as_bool(value: z3.ExprRef) -> z3.BoolRef:
        if not z3.is_bool(value):
            raise ProofError("expected a boolean proposition")
        return value

    @staticmethod
    def _format_model(model: z3.ModelRef, symbols: Mapping[str, z3.ExprRef]) -> str | None:
        """Render the counterexample.

        Only symbols the model actually pins down are reported. Filling in the
        rest would invent values and present a guess as the witness.
        """
        entries = []
        for name in sorted(symbols):
            try:
                interpretation = model[symbols[name]]
            except z3.Z3Exception:
                continue
            if interpretation is None:
                continue
            entries.append(f"{name} = {z3.simplify(interpretation)}")
        return ", ".join(entries) if entries else None

    # ----------------------------------------------------------------- theorem

    def check_theorem(
        self,
        theorem: Theorem,
        theorems: Mapping[str, Theorem] | None = None,
    ) -> ProofResult:
        """Discharge a theorem's proof block.

        `theorems` is the module's theorem table, needed only by `exact`. The
        goal is checked once: a command that discharges it ends the block, so
        `simp` costs one solver call rather than two.
        """
        return self._check(theorem, theorems or {}, frozenset({theorem.name}))

    def _check(
        self,
        theorem: Theorem,
        theorems: Mapping[str, Theorem],
        visiting: frozenset[str],
    ) -> ProofResult:
        types = {param.name: param.type.name for param in theorem.params}
        assumptions = list(theorem.requires)
        discharged = False

        for command in theorem.proof:
            if command.kind in {"simp", "omega"}:
                result = self._discharge(theorem.proposition, assumptions, types, command.kind)
                if result.status != ProofStatus.PROVED:
                    return result
                discharged = True
            elif command.kind == "assert":
                goal = expression_of(command.value) if command.value is not None else None
                if goal is None:
                    return ProofResult(ProofStatus.UNSUPPORTED, "assert requires a proposition")
                result = self._discharge(goal, assumptions, types, None)
                if result.status != ProofStatus.PROVED:
                    return result
                assumptions.append(goal)
            elif command.kind == "exact":
                result = self._exact(theorem, command, theorems, visiting)
                if result.status != ProofStatus.PROVED:
                    return result
                discharged = True
            else:
                return ProofResult(ProofStatus.UNSUPPORTED, f"unknown proof command: `{command.kind}`")

        if discharged:
            return ProofResult(ProofStatus.PROVED, "proved")
        # An empty proof block is not a proof, so the goal still has to be closed.
        return self._discharge(theorem.proposition, assumptions, types, None)

    def _discharge(
        self,
        goal: Proposition | Expr,
        assumptions: list,
        types: Mapping[str, str],
        tactic: str | None,
    ) -> ProofResult:
        if tactic == "omega":
            reason = presburger_violation([*assumptions, expression_of(goal)], types)
            if reason is not None:
                return ProofResult(
                    ProofStatus.UNSUPPORTED,
                    f"`omega` decides linear integer goals only: {reason}",
                )
        return self.prove(goal, assumptions, types)

    def _exact(
        self,
        theorem: Theorem,
        command,
        theorems: Mapping[str, Theorem],
        visiting: frozenset[str],
    ) -> ProofResult:
        """`exact <name>`: the goal is the statement of an already proved theorem.

        Only an identical statement qualifies. Anything looser would need a
        unifier the language does not have, and quietly accepting a weaker goal
        is how a proof block starts lying.
        """
        name = command.value
        if not isinstance(name, str) or not name:
            return ProofResult(ProofStatus.UNSUPPORTED, "`exact` expects the name of a theorem")
        if name not in theorems:
            return ProofResult(ProofStatus.UNSUPPORTED, f"unknown theorem: `{name}`")
        if name in visiting:
            return ProofResult(
                ProofStatus.UNSUPPORTED,
                f"`{name}` is proved in terms of itself",
            )
        target = theorems[name]
        result = self._check(target, theorems, visiting | {name})
        if result.status != ProofStatus.PROVED:
            return ProofResult(
                result.status,
                f"`{name}` does not hold: {result.message}",
                result.model,
            )
        if expression_of(theorem.proposition) != expression_of(target.proposition):
            return ProofResult(
                ProofStatus.DISPROVED,
                f"the goal is not the statement of `{name}`",
            )
        return ProofResult(ProofStatus.PROVED, f"goal is the statement of `{name}`")


def prove(
    proposition: Proposition | Expr,
    assumptions: Iterable[Proposition | Expr] = (),
    types: Mapping[str, str] | None = None,
) -> ProofResult:
    return Prover().prove(proposition, assumptions, types)


def check_theorem(
    theorem: Theorem,
    theorems: Mapping[str, Theorem] | None = None,
) -> ProofResult:
    return Prover().check_theorem(theorem, theorems)
