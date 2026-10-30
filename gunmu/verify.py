"""Verification conditions for function contracts.

`proof.py` discharges `theorem` statements. This module does the rest of the
job: it turns a function's `requires`/`ensures`, the `assert`s in its body, the
preconditions of the calls it makes and its loop `invariant`/`variant`
annotations into verification conditions and hands them to the same solver.

The encoding is symbolic execution over solver terms. Each `let` extends the
environment with the symbolic value of its right-hand side, so a later goal
mentions that term rather than the variable. Every obligation is checked as
`path condition -> goal`, and a path whose condition is unsatisfiable is dropped
before it can produce a failure that no execution could reach.

Two rules keep the results honest:

* A callee's contract is only assumed if that callee's own contract was proved,
  or if the callee is in the same strongly connected component -- that is, the
  assumption is this function's own contract, discharged inductively through the
  loop variant. Otherwise the precondition cannot be discharged and says so.
* A loop needs an `invariant` to mean anything. Without one there is nothing to
  re-establish at `continue` and nothing to hold on entry, so the loop is
  reported rather than silently accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import z3

from .ast import (
    AssertStmt, Binary, BindingPattern, BreakStmt, Call, ContinueStmt, Ensures, Expr,
    ExprStmt, FieldAccess, Function, LetStmt, LiteralPattern, LoopStmt, MatchArm,
    MatchStmt, Module, Name, Pattern, PrintStmt, Proposition, RecordPattern,
    RecordValue, Requires, ReturnStmt, Unary, UnitLiteral, WildcardPattern,
)
from .canonical import strongly_connected_components
from .diagnostics import Span
from .pretty import render_proposition
from .proof import (
    ProofError, ProofResult, ProofStatus, Prover, Z3Translator, expression_of,
    sort_of, worst,
)


@dataclass(frozen=True)
class Obligation:
    """One verification condition, its verdict, and where it came from."""

    description: str
    result: ProofResult
    span: Span | None = None


@dataclass(frozen=True)
class FunctionReport:
    """Every obligation generated for one function, and the worst verdict."""

    name: str
    obligations: tuple[Obligation, ...]

    @property
    def status(self) -> ProofStatus:
        return worst([obligation.result for obligation in self.obligations])

    @property
    def proved(self) -> bool:
        return self.status == ProofStatus.PROVED

    @property
    def failures(self) -> tuple[Obligation, ...]:
        return tuple(
            obligation for obligation in self.obligations
            if obligation.result.status != ProofStatus.PROVED
        )


# ------------------------------------------------------------------- contracts


@dataclass(frozen=True)
class Contract:
    """The part of a function that callers, and the verifier, may rely on."""

    name: str
    params: tuple[tuple[str, str], ...]
    return_type: str
    requires: tuple[Expr, ...] = ()
    ensures: tuple[Expr, ...] = ()

    @property
    def has_post(self) -> bool:
        return bool(self.ensures)


def contract_of(function: Function) -> Contract:
    """The contract a call to `function` may assume."""
    return Contract(
        name=function.name,
        params=tuple((param.name, param.type.name) for param in function.params),
        return_type=function.return_type.name,
        requires=tuple(
            expression_of(spec.proposition) for spec in function.specs
            if isinstance(spec, Requires)
        ),
        ensures=tuple(
            expression_of(spec.proposition) for spec in function.specs
            if isinstance(spec, Ensures)
        ),
    )


# ------------------------------------------------------------ symbolic states


@dataclass(frozen=True)
class _Exit:
    """How a path left a block: the function, a loop, or a loop iteration."""

    kind: str
    value: z3.ExprRef | None = None
    bindings: tuple[tuple[str, z3.ExprRef], ...] = ()


@dataclass
class _State:
    """One symbolic path: the environment it sees and what must hold to reach it."""

    env: dict[str, z3.ExprRef]
    path: z3.BoolRef
    exit: _Exit | None = None


@dataclass
class _Pending:
    """An obligation before it is discharged. A `None` goal means unsupported."""

    description: str
    goal: z3.BoolRef | None
    reason: str = ""
    span: Span | None = None


def _conjunction(conditions: Iterable[z3.BoolRef]) -> z3.BoolRef:
    collected = list(conditions)
    return z3.And(*collected) if collected else z3.BoolVal(True)


def _span_of(node: object) -> Span | None:
    return getattr(node, "span", None)


def _callee_name(call: Call) -> str:
    return call.function.value if isinstance(call.function, Name) else "expression"


def _describe(expr: Expr) -> str:
    if isinstance(expr, Name):
        return f"`{expr.value}`"
    if isinstance(expr, Call):
        return f"call to `{_callee_name(expr)}`"
    return "expression"


def _children(expr: Expr) -> list[Expr]:
    """The sub-expressions of `expr`, for walking an expression tree."""
    if isinstance(expr, Unary):
        return [expr.operand]
    if isinstance(expr, Binary):
        return [expr.left, expr.right]
    if isinstance(expr, Call):
        return [expr.function, *expr.arguments]
    if isinstance(expr, FieldAccess):
        return [expr.value]
    if isinstance(expr, RecordValue):
        return list(expr.fields.values())
    return []


def _mentions(expr: Expr, name: str) -> bool:
    if isinstance(expr, Name):
        return expr.value == name
    return any(_mentions(child, name) for child in _children(expr))


# ------------------------------------------------------------------- verifier


class _FunctionVerifier:
    """Symbolically executes one function and collects its obligations."""

    def __init__(
        self,
        prover: Prover,
        contracts: Mapping[str, Contract],
        usable: frozenset[str],
    ) -> None:
        self.prover = prover
        self.contracts = contracts
        #: Contracts a call site may assume: either already proved elsewhere in
        #: this run, or this component's own contract for a recursive call.
        self.usable = usable
        self.obligations: list[Obligation] = []
        self.symbols: dict[str, z3.ExprRef] = {}
        self.post: tuple[Expr, ...] = ()
        self.name = ""
        self._counter = 0
        self._feasibility: dict[str, bool] = {}

    # ------------------------------------------------------------------ entry

    def verify(self, function: Function) -> FunctionReport:
        self.name = function.name
        self.obligations = []
        self.symbols = {}
        self.post = contract_of(function).ensures
        self._counter = 0
        self._feasibility = {}
        try:
            self._run(function)
        except ProofError as error:
            self._record(_Pending(
                f"`{function.name}` cannot be encoded", None, str(error), function.span
            ))
        return FunctionReport(function.name, tuple(self.obligations))

    def _run(self, function: Function) -> None:
        contract = contract_of(function)
        try:
            env = {}
            for param in function.params:
                symbol = z3.Const(param.name, sort_of(param.type.name))
                env[param.name] = symbol
                self.symbols[param.name] = symbol
            # The precondition seeds every path condition, so each obligation
            # below is checked under the whole contract in force at that point.
            entry = _conjunction(
                Z3Translator(bindings=env).translate(requirement)
                for requirement in contract.requires
            )
        except ProofError as error:
            self._record(_Pending(
                f"signature of `{function.name}` cannot be encoded", None, str(error), function.span
            ))
            return

        if contract.has_post and contract.return_type not in {"Int", "Bool", "String"}:
            self._record(_Pending(
                f"postcondition of `{function.name}` cannot be encoded", None,
                f"`result` would have type {contract.return_type}, which has no logical encoding",
                function.span,
            ))
            return

        self._exec_block(function.body, [_State(env=env, path=entry)], loop=None)

    # -------------------------------------------------------------- execution

    def _exec_block(
        self,
        statements: list,
        states: list[_State],
        loop: LoopStmt | None,
    ) -> list[_State]:
        """Run `statements` over `states`, collecting obligations on the way.

        Every resulting path comes back, tagged with how it left the block. A
        path that returned or jumped to a loop is inert for whatever follows, so
        it is set aside after each statement rather than only on entry -- without
        that, statements after an unconditional `return` would be checked on a
        path that can never reach them.
        """
        settled = [state for state in states if state.exit is not None]
        live = [state for state in states if state.exit is None]
        for statement in statements:
            if not live:
                break
            results = self._exec_statement(statement, live, loop)
            live = [state for state in results if state.exit is None]
            settled.extend(state for state in results if state.exit is not None)
        return settled + live

    def _exec_statement(
        self,
        statement,
        states: list[_State],
        loop: LoopStmt | None,
    ) -> list[_State]:
        if isinstance(statement, LetStmt):
            return self._exec_let(statement, states)
        if isinstance(statement, PrintStmt):
            # Printing has no precondition and no effect on the store.
            return states
        if isinstance(statement, AssertStmt):
            return self._exec_assert(statement, states)
        if isinstance(statement, ReturnStmt):
            return self._exec_return(statement, states)
        if isinstance(statement, BreakStmt):
            return self._exec_break(statement, states)
        if isinstance(statement, ContinueStmt):
            return self._exec_continue(statement, states)
        if isinstance(statement, ExprStmt):
            # The result is discarded, so only the call's precondition matters.
            for state in states:
                self._check_all(self._eval(statement.value, state)[2], state.path)
            return states
        if isinstance(statement, MatchStmt):
            return self._exec_match(statement, states, loop)
        if isinstance(statement, LoopStmt):
            return self._exec_loop(statement, states)
        return states

    def _exec_let(self, statement: LetStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            term, gains, pending = self._eval(statement.value, state)
            self._check_all(pending, state.path)
            if term is None:
                # Nothing downstream that mentions the binding can be encoded,
                # so this path stops here rather than reporting nonsense.
                continue
            env = dict(state.env)
            env[statement.name] = term
            results.append(_State(env, _conjunction([state.path, *gains])))
        return self._live(results)

    def _exec_assert(self, statement: AssertStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            goal = self._translate(statement.proposition, state)
            if goal is None:
                results.append(state)
                continue
            self._record(_Pending(
                f"asserted proposition holds in `{self.name}`", goal, span=_span_of(statement),
            ), state.path)
            # A proved assertion is available to everything after it.
            results.append(_State(dict(state.env), _conjunction([state.path, goal])))
        return self._live(results)

    def _exec_return(self, statement: ReturnStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            if statement.value is None:
                term, gains, pending = None, [], []
            else:
                term, gains, pending = self._eval(statement.value, state)
            self._check_all(pending, state.path)
            path = _conjunction([state.path, *gains])
            if self.post:
                goal = self._postcondition(state, path, term)
                if goal is not None:
                    self._record(_Pending(
                        f"postcondition of `{self.name}` holds where it returns",
                        goal, span=_span_of(statement),
                    ), path)
            results.append(_State(dict(state.env), path, _Exit("return", term)))
        return results

    def _exec_break(self, statement: BreakStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            if statement.value is None:
                term, gains, pending = None, [], []
            else:
                term, gains, pending = self._eval(statement.value, state)
            self._check_all(pending, state.path)
            results.append(_State(
                dict(state.env),
                _conjunction([state.path, *gains]),
                _Exit("break", term),
            ))
        return results

    def _exec_continue(self, statement: ContinueStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            env = dict(state.env)
            bindings = []
            gains: list[z3.BoolRef] = []
            pending: list[_Pending] = []
            for name, value in statement.bindings.items():
                term, found_gains, found = self._eval(value, state)
                pending.extend(found)
                gains.extend(found_gains)
                if term is None:
                    continue
                env[name] = term
                bindings.append((name, term))
            self._check_all(pending, state.path)
            results.append(_State(
                env,
                _conjunction([state.path, *gains]),
                _Exit("continue", bindings=tuple(bindings)),
            ))
        return results

    def _exec_match(
        self,
        statement: MatchStmt,
        states: list[_State],
        loop: LoopStmt | None,
    ) -> list[_State]:
        results = []
        for state in states:
            scrutinee, gains, pending = self._eval(statement.value, state)
            self._check_all(pending, state.path)
            if scrutinee is None:
                continue
            base = _conjunction([state.path, *gains])
            # Reaching an arm means every earlier unguarded literal arm missed.
            excluded: list[z3.BoolRef] = []
            for arm in statement.arms:
                arm_state = self._enter_arm(arm, scrutinee, base, excluded, state)
                if arm_state is not None:
                    results.extend(self._exec_block(arm.body, [arm_state], loop))
                if arm.guard is None and isinstance(arm.pattern, LiteralPattern):
                    literal = self._term(arm.pattern.value, state, what="pattern literal")
                    if literal is not None:
                        excluded.append(scrutinee == literal)
        return self._live(results)

    def _enter_arm(
        self,
        arm: MatchArm,
        scrutinee: z3.ExprRef,
        base: z3.BoolRef,
        excluded: list[z3.BoolRef],
        state: _State,
    ) -> _State | None:
        env = dict(state.env)
        conditions: list[z3.BoolRef] = [base]
        if excluded:
            conditions.append(z3.Not(_conjunction(excluded)))
        pattern: Pattern = arm.pattern
        if isinstance(pattern, BindingPattern):
            env[pattern.name] = scrutinee
        elif isinstance(pattern, LiteralPattern):
            literal = self._term(pattern.value, state, what="pattern literal")
            if literal is None:
                return None
            conditions.append(scrutinee == literal)
        elif isinstance(pattern, RecordPattern):
            # A record pattern would have to narrow the scrutinee field by field,
            # and records have no logical encoding to narrow it with.
            self._record(_Pending(
                f"pattern in `{self.name}` cannot be encoded", None,
                "record patterns are not supported by the verifier",
                _span_of(pattern),
            ), state.path)
            return None
        elif not isinstance(pattern, WildcardPattern):
            self._record(_Pending(
                f"pattern in `{self.name}` cannot be encoded", None,
                f"unsupported pattern: {type(pattern).__name__}",
                _span_of(pattern),
            ), state.path)
            return None
        if arm.guard is not None:
            guard = self._translate(arm.guard, state)
            if guard is None:
                return None
            conditions.append(guard)
        path = _conjunction(conditions)
        return _State(env, path) if self._feasible(path) else None

    # ------------------------------------------------------------------ loops

    def _exec_loop(self, statement: LoopStmt, states: list[_State]) -> list[_State]:
        results = []
        for state in states:
            results.extend(self._exec_one_loop(statement, state))
        return self._live(results)

    def _exec_one_loop(self, statement: LoopStmt, state: _State) -> list[_State]:
        invariants = [spec for spec in statement.specs if spec.kind == "invariant"]
        variants = [spec for spec in statement.specs if spec.kind == "variant"]

        initial: dict[str, z3.ExprRef] = {}
        for name, value in statement.bindings.items():
            term, gains, pending = self._eval(value, state)
            self._check_all(pending, state.path)
            if term is not None:
                initial[name] = term
        if len(initial) != len(statement.bindings):
            return []

        # The state at the head of an arbitrary iteration, tied to the entry
        # state by the bindings the loop started from.
        head_env = dict(state.env)
        entry: list[z3.BoolRef] = []
        for name, term in initial.items():
            symbol = z3.Int(f"head_{name}_{self._fresh()}")
            head_env[name] = symbol
            entry.append(symbol == term)
        entry_path = _conjunction([state.path, *entry])

        if not invariants:
            self._record(_Pending(
                f"loop in `{self.name}` has no `invariant`", None,
                "an invariant is what `continue` re-establishes and what holds at the loop head",
                _span_of(statement),
            ), entry_path)

        for spec in invariants:
            goal = self._translate(spec.value, state, env=head_env)
            if goal is not None:
                self._record(_Pending(
                    f"invariant `{render_proposition(spec.value)}` holds on entry "
                    f"to the loop in `{self.name}`",
                    goal, span=_span_of(spec),
                ), entry_path)

        variant_terms: list[tuple[object, z3.ExprRef]] = []
        for spec in variants:
            term = self._term(spec.value, state, env=head_env, what="variant")
            if term is None:
                continue
            variant_terms.append((spec, term))
            self._record(_Pending(
                f"variant `{render_proposition(spec.value)}` is non-negative "
                f"on entry to the loop in `{self.name}`",
                term >= 0, span=_span_of(spec),
            ), entry_path)

        invariant_head = _conjunction(
            goal for goal in (
                self._translate(spec.value, state, env=head_env) for spec in invariants
            ) if goal is not None
        )
        # At the loop head the invariant holds, and everything known on the way
        # in still holds -- including an enclosing loop's invariant, which is
        # what lets nested loops check each other.
        body = self._exec_block(
            statement.body,
            [_State(dict(head_env), _conjunction([state.path, invariant_head]))],
            loop=statement,
        )

        results = []
        for reached in body:
            if reached.exit is None:
                # Falling off the end runs another iteration with the state
                # unchanged, so a variant cannot strictly decrease.
                for spec, term in variant_terms:
                    self._record(_Pending(
                        f"loop in `{self.name}` may not terminate: the variant does not "
                        f"decrease when the body completes without `break` or `continue`",
                        term < term, span=_span_of(spec),
                    ), reached.path)
                continue
            if reached.exit.kind == "return":
                results.append(reached)
                continue
            if reached.exit.kind == "break":
                # The loop state is local to the loop, so only the path escapes.
                results.append(_State(dict(state.env), reached.path))
                continue
            self._check_continue(invariants, variant_terms, head_env, reached)
        return results

    def _check_continue(
        self,
        invariants: list,
        variant_terms: list,
        head_env: dict[str, z3.ExprRef],
        state: _State,
    ) -> None:
        updated = dict(head_env)
        updated.update(state.exit.bindings)
        for spec in invariants:
            goal = self._translate(spec.value, state, env=updated)
            if goal is not None:
                self._record(_Pending(
                    f"`continue` re-establishes the invariant "
                    f"`{render_proposition(spec.value)}` of `{self.name}`",
                    goal, span=_span_of(spec),
                ), state.path)
        for spec, term in variant_terms:
            after = self._term(spec.value, state, env=updated, what="variant")
            if after is None:
                continue
            rendered = render_proposition(spec.value)
            self._record(_Pending(
                f"`continue` strictly decreases the variant `{rendered}` of `{self.name}`",
                term > after, span=_span_of(spec),
            ), state.path)
            self._record(_Pending(
                f"`continue` keeps the variant `{rendered}` of `{self.name}` non-negative",
                after >= 0, span=_span_of(spec),
            ), state.path)

    # ------------------------------------------------------------- evaluation

    def _eval(
        self,
        expr: Expr | None,
        state: _State,
    ) -> tuple[z3.ExprRef | None, list[z3.BoolRef], list[_Pending]]:
        """Evaluate `expr` to a term under `state`.

        Also returns facts the calls inside the expression establish and the
        preconditions those calls need. Calls are visited wherever they appear,
        because the grammar allows one as an operand: `continue (i = f(x) + 1)`
        has to check `f`'s precondition like any other.
        """
        if expr is None or isinstance(expr, UnitLiteral):
            # `unit` carries no information, so it needs no encoding. Returning
            # `None` here is not a failure; callers treat a missing term as
            # "nothing to say about this path".
            return None, [], []
        gains: list[z3.BoolRef] = []
        pending: list[_Pending] = []
        translator = Z3Translator(
            bindings=state.env, calls=self._call_handler(state, gains, pending)
        )
        try:
            return translator.translate(expr), gains, pending
        except ProofError as error:
            pending.append(_Pending(
                f"{_describe(expr)} cannot be encoded", None, str(error), _span_of(expr)
            ))
            return None, gains, pending

    def _call_handler(
        self,
        state: _State,
        gains: list[z3.BoolRef],
        pending: list[_Pending],
    ) -> Callable[[Call], z3.ExprRef]:
        """Model a call: check the callee's `requires`, adopt its `ensures`.

        The result is a fresh variable, because nothing is known about what a
        function computes beyond its contract. What it *does* guarantee comes
        from `ensures`, which is exactly what makes a callee useful here.
        """

        def on_call(call: Call) -> z3.ExprRef:
            callee = call.function.value if isinstance(call.function, Name) else None
            contract = self.contracts.get(callee) if callee is not None else None
            if contract is None:
                raise ProofError(
                    f"only declared functions can be called: `{_callee_name(call)}`"
                )
            if callee not in self.usable:
                raise ProofError(
                    f"the contract of `{callee}` is not available here: it has not been proved"
                )
            outer = Z3Translator(bindings=state.env, calls=on_call)
            arguments = [outer.translate(argument) for argument in call.arguments]
            result = self._call_result(contract)
            # Inside the callee's specs its parameters stand for the arguments.
            inner_bindings = {
                name: term for (name, _), term in zip(contract.params, arguments)
            }
            if contract.has_post:
                inner_bindings["result"] = result
            inner = Z3Translator(bindings=inner_bindings, calls=on_call)
            for requirement in contract.requires:
                pending.append(_Pending(
                    f"precondition of `{contract.name}` holds at this call",
                    inner.translate(requirement),
                    span=_span_of(call),
                ))
            for guarantee in contract.ensures:
                fact = inner.translate(guarantee)
                if fact is not None:
                    gains.append(fact)
            return result

        return on_call

    def _call_result(self, contract: Contract) -> z3.ExprRef:
        """A fresh term standing for what a call returns.

        A `Unit`-returning callee gets an unconstrained `Int`. Nothing can
        observe the difference: `analyze` rejects a `Unit` value anywhere a term
        is required, so no obligation can ever mention it.
        """
        type_name = contract.return_type
        if type_name not in {"Int", "Bool", "String"}:
            type_name = "Int"
        return z3.Const(f"havoc_{contract.name}_{self._fresh()}", sort_of(type_name))

    def _translate(
        self,
        proposition: Proposition | Expr,
        state: _State,
        env: Mapping[str, z3.ExprRef] | None = None,
    ) -> z3.BoolRef | None:
        """Translate a proposition, recording the reason if it cannot be."""
        term = self._term(expression_of(proposition), state, env, "proposition")
        if term is None or z3.is_bool(term):
            return term
        self._record(_Pending(
            f"`{render_proposition(proposition)}` is not a proposition", None,
            "expected a Boolean", _span_of(proposition),
        ), state.path)
        return None

    def _term(
        self,
        expr: Expr,
        state: _State,
        env: Mapping[str, z3.ExprRef] | None = None,
        what: str = "expression",
    ) -> z3.ExprRef | None:
        """Translate any expression to a term, recording the reason if it cannot.

        Variants and pattern literals are Int expressions rather than
        propositions, so they go through here instead of `_translate`.
        """
        try:
            return Z3Translator(bindings=state.env if env is None else env).translate(expr)
        except ProofError as error:
            self._record(_Pending(
                f"{what} `{render_proposition(expr)}` cannot be encoded", None,
                str(error), _span_of(expr),
            ), state.path)
            return None

    def _postcondition(
        self,
        state: _State,
        path: z3.BoolRef,
        result: z3.ExprRef | None,
    ) -> z3.BoolRef | None:
        """The function's `ensures`, with `result` bound to the returned term."""
        if result is None and any(_mentions(spec, "result") for spec in self.post):
            self._record(_Pending(
                f"postcondition of `{self.name}` cannot be encoded", None,
                "`result` has no encoding when the function returns Unit",
            ))
            return None
        bindings = dict(state.env)
        if result is not None:
            bindings["result"] = result
        terms = []
        for spec in self.post:
            try:
                term = Z3Translator(bindings=bindings).translate(spec)
            except ProofError as error:
                self._record(_Pending(
                    f"postcondition of `{self.name}` cannot be encoded", None, str(error)
                ))
                return None
            if not z3.is_bool(term):
                self._record(_Pending(
                    f"postcondition of `{self.name}` is not a proposition", None,
                    "expected a Boolean",
                ))
                return None
            terms.append(term)
        return _conjunction(terms)

    # ------------------------------------------------------------- obligations

    def _record(self, pending: _Pending, context: z3.BoolRef | None = None) -> ProofResult:
        if pending.goal is None:
            result = ProofResult(ProofStatus.UNSUPPORTED, pending.reason)
        else:
            result = self.prover.check_implication(
                z3.BoolVal(True) if context is None else context,
                pending.goal,
                self.symbols,
            )
        self.obligations.append(Obligation(pending.description, result, pending.span))
        return result

    def _check_all(self, pending: Iterable[_Pending], context: z3.BoolRef) -> None:
        for item in pending:
            self._record(item, context)

    def _live(self, states: list[_State]) -> list[_State]:
        return [state for state in states if self._feasible(state.path)]

    def _fresh(self) -> int:
        self._counter += 1
        return self._counter

    def _feasible(self, path: z3.BoolRef) -> bool:
        """Can this path condition hold at all?

        An unsatisfiable path never runs, so it must not produce a failure. An
        undecided check keeps the path, since dropping it could hide a real bug.
        """
        key = path.sexpr()
        cached = self._feasibility.get(key)
        if cached is None:
            solver = z3.Solver()
            solver.set(timeout=self.prover.timeout_ms)
            solver.add(path)
            cached = solver.check() == z3.sat
            self._feasibility[key] = cached
        return cached


# --------------------------------------------------------------------- driver


def _walk_statement(statement) -> tuple[list, list[Expr]]:
    """Split a statement into the statements nested in it and the expressions it evaluates."""
    nested: list = []
    found: list[Expr] = []
    if isinstance(statement, LetStmt):
        found.append(statement.value)
    elif isinstance(statement, PrintStmt):
        found.append(statement.value)
    elif isinstance(statement, AssertStmt):
        found.append(expression_of(statement.proposition))
    elif isinstance(statement, (ReturnStmt, BreakStmt)):
        if statement.value is not None:
            found.append(statement.value)
    elif isinstance(statement, ContinueStmt):
        found.extend(statement.bindings.values())
    elif isinstance(statement, ExprStmt):
        found.append(statement.value)
    elif isinstance(statement, MatchStmt):
        found.append(statement.value)
        for arm in statement.arms:
            if arm.guard is not None:
                found.append(expression_of(arm.guard))
            nested.extend(arm.body)
    elif isinstance(statement, LoopStmt):
        found.extend(statement.bindings.values())
        nested.extend(statement.body)
    return nested, found


def call_graph(functions: Mapping[str, Function]) -> dict[str, set[str]]:
    """Which declared function each function calls, at any depth."""
    graph: dict[str, set[str]] = {}
    for name, function in functions.items():
        callees: set[str] = set()
        stack: list = list(function.body)
        while stack:
            nested, expressions = _walk_statement(stack.pop())
            stack.extend(nested)
            for expression in expressions:
                pending = [expression]
                while pending:
                    current = pending.pop()
                    if isinstance(current, Call) and isinstance(current.function, Name):
                        callees.add(current.function.value)
                    pending.extend(_children(current))
        graph[name] = callees & set(functions)
    return graph


def verify_module(
    module: Module,
    prover: Prover | None = None,
) -> list[FunctionReport]:
    """Verify every function contract in `module`, in dependency order.

    The module must already have passed `analyze`. Functions are grouped by call
    graph so a callee's contract is proved before a caller relies on it; a
    recursive group verifies against its own members' contracts, which the loop
    variant discharges inductively.
    """
    prover = prover or Prover()
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    contracts = {name: contract_of(function) for name, function in functions.items()}
    graph = call_graph(functions)

    reports: list[FunctionReport] = []
    proved: set[str] = set()
    for component in strongly_connected_components(graph):
        group = set(component)
        usable = frozenset(proved | group)
        for name in sorted(group):
            report = _FunctionVerifier(prover, contracts, usable).verify(functions[name])
            reports.append(report)
            if report.proved:
                proved.add(name)
    return reports


def verify_function(
    function: Function,
    module: Module | None = None,
    prover: Prover | None = None,
) -> FunctionReport:
    """Verify one function in isolation.

    With `module`, every sibling contract is assumed rather than proved, so this
    answers "is this function's own body sound given the others' contracts"
    rather than "does the whole module verify". Use `verify_module` for that.
    """
    prover = prover or Prover()
    contracts: dict[str, Contract] = {}
    if module is not None:
        contracts = {
            item.name: contract_of(item)
            for item in module.items if isinstance(item, Function)
        }
    contracts[function.name] = contract_of(function)
    return _FunctionVerifier(prover, contracts, frozenset(contracts)).verify(function)
