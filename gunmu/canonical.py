"""Content addressing for Gunmu definitions.

A definition's identity is its *structure*, not its name or its source text:

* spans are dropped, so whitespace and formatting are irrelevant;
* local binders are alpha-renamed to positional indices ``#0``, ``#1``, ...,
  so renaming a parameter or a ``let`` binding does not change the hash;
* references to other top-level definitions are replaced by *their* hashes, so
  editing a dependency changes every dependent's hash (Unison's key property);
* a definition's own name is never hashed, because names live in the namespace
  table and are pure metadata;
* self-recursion is encoded as ``(self)`` so a hash never depends on itself.

Mutually recursive definitions form a strongly connected component and are
hashed as a group: each member's canonical form refers to its peers as
``(peer N)``, the group is hashed as a whole, and each member then gets a
per-index hash derived from the group hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Iterable

from .ast import (
    AssertStmt, Binary, BindingPattern, BoolLiteral, BreakStmt, Call, ContinueStmt,
    Ensures, Expr, ExprStmt, FieldAccess, Function, IntLiteral, LetStmt,
    LiteralPattern, LoopStmt, MatchStmt, Module, Name, Pattern, PrintStmt,
    Proposition, Record, RecordPattern, RecordValue, Requires, ReturnStmt,
    StringLiteral, Theorem, TypeRef, Unary, UnitLiteral, WildcardPattern,
)
from .hashing import definition_hash, group_hash, group_member_hash

BUILTIN_TYPES = frozenset({"Int", "Bool", "String", "Unit", "Prop"})


class CanonicalError(ValueError):
    pass


class UnresolvedReference(CanonicalError):
    def __init__(self, name: str, owner: str):
        super().__init__(f"`{owner}` references undefined name `{name}`")
        self.name = name
        self.owner = owner


def node(*parts: str) -> str:
    """Build an S-expression, skipping empty parts.

    Deliberately avoids string `.replace` post-processing: canonical text may
    contain arbitrary string literals, so rewriting text after the fact would
    corrupt payloads that happen to contain the pattern being rewritten.
    """
    return "(" + " ".join(part for part in parts if part) + ")"


@dataclass(frozen=True)
class CanonicalDefinition:
    name: str
    local_name: str
    module: str
    kind: str
    canonical: str
    hash: str
    dependencies: tuple[str, ...] = ()
    group_hash: str | None = None
    group_index: int | None = None


@dataclass(frozen=True)
class CanonicalGroup:
    hash: str
    blob: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalModule:
    module: str
    definitions: tuple[CanonicalDefinition, ...]
    groups: tuple[CanonicalGroup, ...] = ()

    def by_local_name(self) -> dict[str, CanonicalDefinition]:
        return {definition.local_name: definition for definition in self.definitions}


def kind_of(item) -> str:
    if isinstance(item, Record):
        return "record"
    if isinstance(item, Function):
        return "fn"
    if isinstance(item, Theorem):
        return "theorem"
    raise CanonicalError(f"cannot store {type(item).__name__}")


# --------------------------------------------------------------------- references


def collect_refs(item) -> set[str]:
    """Top-level names referenced by a definition.

    Only syntactic positions that can hold a top-level name are considered:
    call targets, record/type names and `exact` theorem references. Plain `Name`
    nodes are always locals, so they can never be confused with definitions.
    """
    refs: set[str] = set()

    def type_ref(reference: TypeRef) -> None:
        if reference.name not in BUILTIN_TYPES:
            refs.add(reference.name)

    def expr(expression: Expr) -> None:
        if isinstance(expression, Call):
            if isinstance(expression.function, Name):
                refs.add(expression.function.value)
            for argument in expression.arguments:
                expr(argument)
        elif isinstance(expression, Unary):
            expr(expression.operand)
        elif isinstance(expression, Binary):
            expr(expression.left)
            expr(expression.right)
        elif isinstance(expression, FieldAccess):
            expr(expression.value)
        elif isinstance(expression, RecordValue):
            refs.add(expression.type_name)
            for value in expression.fields.values():
                expr(value)

    def proposition(value) -> None:
        expr(value.expression if isinstance(value, Proposition) else value)

    def pattern(value: Pattern) -> None:
        if isinstance(value, LiteralPattern):
            expr(value.value)
        elif isinstance(value, RecordPattern):
            refs.add(value.type_name)
            for nested in value.fields.values():
                pattern(nested)

    def block(body: Iterable) -> None:
        for statement in body:
            if isinstance(statement, LetStmt):
                if statement.type is not None:
                    type_ref(statement.type)
                expr(statement.value)
            elif isinstance(statement, PrintStmt):
                expr(statement.value)
            elif isinstance(statement, AssertStmt):
                proposition(statement.proposition)
            elif isinstance(statement, ReturnStmt):
                if statement.value is not None:
                    expr(statement.value)
            elif isinstance(statement, BreakStmt):
                if statement.value is not None:
                    expr(statement.value)
            elif isinstance(statement, ContinueStmt):
                for value in statement.bindings.values():
                    expr(value)
            elif isinstance(statement, ExprStmt):
                expr(statement.value)
            elif isinstance(statement, MatchStmt):
                expr(statement.value)
                for arm in statement.arms:
                    pattern(arm.pattern)
                    if arm.guard is not None:
                        proposition(arm.guard)
                    block(arm.body)
            elif isinstance(statement, LoopStmt):
                for value in statement.bindings.values():
                    expr(value)
                for spec in statement.specs:
                    if spec.kind == "invariant":
                        proposition(spec.value)
                    else:
                        expr(spec.value)
                block(statement.body)

    if isinstance(item, Record):
        for field in item.fields:
            type_ref(field.type)
    elif isinstance(item, Function):
        for parameter in item.params:
            type_ref(parameter.type)
        type_ref(item.return_type)
        for spec in item.specs:
            proposition(spec.proposition)
        block(item.body)
    elif isinstance(item, Theorem):
        for parameter in item.params:
            type_ref(parameter.type)
        for requirement in item.requires:
            proposition(requirement)
        proposition(item.proposition)
        for command in item.proof:
            if command.kind == "assert" and command.value is not None:
                proposition(command.value)
            elif command.kind == "exact" and isinstance(command.value, str):
                refs.add(command.value)
    return refs


# ------------------------------------------------------------------ canonicalizer


class _Canonicalizer:
    def __init__(self, resolve: Callable[[str], str], owner: str):
        self._resolve = resolve
        self._owner = owner
        self._names: dict[str, int] = {}
        self._counter = 0

    # -- scope ---------------------------------------------------------------

    def _bind(self, name: str) -> str:
        index = self._names.get(name)
        if index is None:
            index = self._counter
            self._counter += 1
            self._names[name] = index
        return f"#{index}"

    def _index(self, name: str) -> int:
        index = self._names.get(name)
        if index is None:
            raise CanonicalError(f"`{self._owner}` uses unbound name `{name}`")
        return index

    def _local(self, name: str) -> str:
        return f"#{self._index(name)}"

    # -- expressions ---------------------------------------------------------

    def type_ref(self, reference: TypeRef) -> str:
        if reference.name in BUILTIN_TYPES:
            return node("type", reference.name)
        return node("nom", self._resolve(reference.name))

    def expr(self, value: Expr) -> str:
        if isinstance(value, IntLiteral):
            return node("int", str(value.value))
        if isinstance(value, StringLiteral):
            return node("str", json.dumps(value.value, ensure_ascii=False))
        if isinstance(value, BoolLiteral):
            return node("bool", "true" if value.value else "false")
        if isinstance(value, UnitLiteral):
            return node("unit")
        if isinstance(value, Name):
            return self._local(value.value)
        if isinstance(value, Unary):
            return node("un", value.op, self.expr(value.operand))
        if isinstance(value, Binary):
            return node("bin", value.op, self.expr(value.left), self.expr(value.right))
        if isinstance(value, Call):
            name = value.function.value if isinstance(value.function, Name) else "?"
            return node("call", self._resolve(name), *[self.expr(a) for a in value.arguments])
        if isinstance(value, FieldAccess):
            return node("field", self.expr(value.value), value.field)
        if isinstance(value, RecordValue):
            fields = [node("init", name, self.expr(value.fields[name])) for name in sorted(value.fields)]
            return node("build", self._resolve(value.type_name), *fields)
        raise CanonicalError(f"`{self._owner}` has unsupported expression {type(value).__name__}")

    def proposition(self, value) -> str:
        return self.expr(value.expression if isinstance(value, Proposition) else value)

    def pattern(self, value: Pattern) -> str:
        if isinstance(value, WildcardPattern):
            return node("wild")
        if isinstance(value, BindingPattern):
            return node("bind", self._bind(value.name))
        if isinstance(value, LiteralPattern):
            return node("lit", self.expr(value.value))
        if isinstance(value, RecordPattern):
            fields = [node("field", name, self.pattern(value.fields[name])) for name in sorted(value.fields)]
            rest = node("rest") if value.rest else ""
            return node("rpat", self._resolve(value.type_name), rest, *fields)
        raise CanonicalError(f"`{self._owner}` has unsupported pattern {type(value).__name__}")

    # -- statements ----------------------------------------------------------

    def block(self, body: Iterable) -> str:
        return " ".join(self.stmt(statement) for statement in body)

    def stmt(self, value) -> str:
        if isinstance(value, LetStmt):
            evaluated = self.expr(value.value)
            annotation = self.type_ref(value.type) if value.type is not None else ""
            return node("let", self._bind(value.name), annotation, evaluated)
        if isinstance(value, PrintStmt):
            return node("print", self.expr(value.value))
        if isinstance(value, AssertStmt):
            return node("assert", self.proposition(value.proposition))
        if isinstance(value, ReturnStmt):
            return node("return") if value.value is None else node("return", self.expr(value.value))
        if isinstance(value, BreakStmt):
            return node("break") if value.value is None else node("break", self.expr(value.value))
        if isinstance(value, ContinueStmt):
            entries = sorted((self._index(name), self.expr(v)) for name, v in value.bindings.items())
            return node("continue", *[node("state", str(i), v) for i, v in entries])
        if isinstance(value, ExprStmt):
            return node("expr", self.expr(value.value))
        if isinstance(value, MatchStmt):
            arms = []
            for arm in value.arms:
                # The pattern must bind before the guard and body are visited:
                # guards refer to names introduced by the pattern.
                bound = self.pattern(arm.pattern)
                guard = node("guard", self.proposition(arm.guard)) if arm.guard is not None else ""
                arms.append(node("arm", bound, guard, node("body", self.block(arm.body))))
            return node("match", self.expr(value.value), *arms)
        if isinstance(value, LoopStmt):
            entries = sorted(value.bindings.items())
            inits = node(*[node("init", self.expr(v)) for _, v in entries])
            for name, _ in entries:
                self._bind(name)
            specs = []
            for spec in value.specs:
                if spec.kind == "invariant":
                    specs.append(node("invariant", self.proposition(spec.value)))
                else:
                    specs.append(node("variant", self.expr(spec.value)))
            return node("loop", inits, node(*specs), node("body", self.block(value.body)))
        raise CanonicalError(f"`{self._owner}` has unsupported statement {type(value).__name__}")

    # -- definitions ---------------------------------------------------------

    def record(self, item: Record) -> str:
        return node("record", *[node("field", f.name, self.type_ref(f.type)) for f in item.fields])

    def function(self, item: Function) -> str:
        parameters = node(
            "params",
            *[node("param", self._bind(p.name), self.type_ref(p.type)) for p in item.params],
        )
        returns = self.type_ref(item.return_type)
        if any(isinstance(spec, Ensures) for spec in item.specs):
            self._bind("result")
        specs = [
            node("requires" if isinstance(spec, Requires) else "ensures", self.proposition(spec.proposition))
            for spec in item.specs
        ]
        return node("fn", parameters, returns, *specs, node("body", self.block(item.body)))

    def theorem(self, item: Theorem) -> str:
        parameters = node(
            "params",
            *[node("param", self._bind(p.name), self.type_ref(p.type)) for p in item.params],
        )
        requires = [node("requires", self.proposition(r)) for r in item.requires]
        goal = node("goal", self.proposition(item.proposition))
        commands = []
        for command in item.proof:
            if command.kind == "assert" and command.value is not None:
                commands.append(node("assert", self.proposition(command.value)))
            elif command.kind == "exact" and isinstance(command.value, str):
                commands.append(node("exact", self._resolve(command.value)))
            else:
                commands.append(node(command.kind))
        return node("theorem", parameters, *requires, goal, node("proof", *commands))


def canonicalize_definition(item, resolve: Callable[[str], str]) -> str:
    canonicalizer = _Canonicalizer(resolve, getattr(item, "name", "?"))
    if isinstance(item, Record):
        return canonicalizer.record(item)
    if isinstance(item, Function):
        return canonicalizer.function(item)
    if isinstance(item, Theorem):
        return canonicalizer.theorem(item)
    raise CanonicalError(f"cannot canonicalize {type(item).__name__}")


# --------------------------------------------------------------------------- SCC


def strongly_connected_components(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan's algorithm, iterative, emitted dependencies-first.

    Reverse topological order is exactly what hashing needs: a component is
    emitted only after every component it references has been emitted.
    """
    counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    for root in sorted(graph):
        if root in indices:
            continue
        indices[root] = lowlink[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work: list[tuple[str, Iterable[str]]] = [(root, iter(sorted(graph[root])))]

        while work:
            current, neighbours = work[-1]
            descended = False
            for neighbour in neighbours:
                if neighbour not in indices:
                    indices[neighbour] = lowlink[neighbour] = counter
                    counter += 1
                    stack.append(neighbour)
                    on_stack.add(neighbour)
                    work.append((neighbour, iter(sorted(graph[neighbour]))))
                    descended = True
                    break
                if neighbour in on_stack:
                    lowlink[current] = min(lowlink[current], indices[neighbour])
            if descended:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[current])
            if lowlink[current] == indices[current]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == current:
                        break
                components.append(sorted(component))
    return components


# ----------------------------------------------------------------------- driver


def canonicalize_module(module: Module) -> CanonicalModule:
    by_local: dict[str, object] = {item.name: item for item in module.items}
    graph: dict[str, set[str]] = {}
    for name, item in by_local.items():
        references = collect_refs(item)
        missing = references - set(by_local)
        if missing:
            raise UnresolvedReference(sorted(missing)[0], name)
        graph[name] = references

    hashed: dict[str, str] = {}
    definitions: list[CanonicalDefinition] = []
    groups: list[CanonicalGroup] = []

    for component in strongly_connected_components(graph):
        is_group = len(component) > 1
        index_of = {name: position for position, name in enumerate(component)}
        canonical_forms: dict[str, str] = {}

        for name in component:
            item = by_local[name]

            def resolve(reference: str, name: str = name) -> str:
                if reference == name:
                    return node("peer", str(index_of[reference])) if is_group else node("self")
                if is_group and reference in index_of:
                    return node("peer", str(index_of[reference]))
                if reference not in hashed:
                    raise CanonicalError(f"`{reference}` was not hashed before `{name}`")
                return node("ref", hashed[reference])

            canonical_forms[name] = canonicalize_definition(item, resolve)

        if is_group:
            blob = "\n".join(
                node("member", str(position), "\n" + canonical_forms[name])
                for position, name in enumerate(component)
            )
            group = group_hash(blob)
            groups.append(CanonicalGroup(group, blob, tuple(component)))
            member_hashes = {
                name: group_member_hash(group, position)
                for position, name in enumerate(component)
            }
        else:
            group = None
            member_hashes = {
                component[0]: definition_hash(kind_of(by_local[component[0]]), canonical_forms[component[0]])
            }

        for name in component:
            hashed[name] = member_hashes[name]
            dependencies = tuple(
                sorted(
                    hashed[reference]
                    for reference in graph[name]
                    if reference != name and not (is_group and reference in index_of)
                )
            )
            definitions.append(
                CanonicalDefinition(
                    name=f"{module.name}.{name}" if module.name else name,
                    local_name=name,
                    module=module.name,
                    kind=kind_of(by_local[name]),
                    canonical=canonical_forms[name],
                    hash=member_hashes[name],
                    dependencies=dependencies,
                    group_hash=group,
                    group_index=index_of[name] if is_group else None,
                )
            )

    return CanonicalModule(module.name, tuple(definitions), tuple(groups))
