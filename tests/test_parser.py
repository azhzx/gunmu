from gunmu import ParseError, parse
from gunmu.ast import (
    BoolLiteral, Function, IntLiteral, LoopStmt, MatchStmt, Module, ProofCommand,
    Record, Theorem, TypeRef,
)


def test_module_record_function():
    module = parse('''
        module demo.core

        record Point {
            x: Int
            y: Int
        }

        fn origin(value: Int) -> Unit
        requires value >= 0
        ensures value >= 0 {
            print value
            return unit
        }
    ''')

    assert isinstance(module, Module)
    assert module.name == "demo.core"
    assert isinstance(module.items[0], Record)
    function = module.items[1]
    assert isinstance(function, Function)
    assert function.name == "origin"
    assert function.return_type == TypeRef("Unit")
    assert len(function.specs) == 2


def test_match_and_loop_are_structured():
    module = parse('''
        module demo.control

        fn run(value: Int) -> Unit {
            match value {
                0 => {
                    print "zero"
                }
                n if n > 0 => {
                    print "positive"
                }
                _ => {
                    print "negative"
                }
            }

            loop (i = 0, sum = 0)
            with {
                invariant i >= 0
                variant 10 - i
            } {
                match i {
                    10 => {
                        break sum
                    }
                    _ => {
                        continue (i = i + 1, sum = sum + i)
                    }
                }
            }

            return unit
        }
    ''')

    function = module.items[0]
    assert isinstance(function, Function)
    assert isinstance(function.body[0], MatchStmt)
    loop = function.body[1]
    assert isinstance(loop, LoopStmt)
    assert loop.bindings["i"] == IntLiteral(0)
    assert loop.bindings["sum"] == IntLiteral(0)
    assert [spec.kind for spec in loop.specs] == ["invariant", "variant"]


def test_theorem_proof_commands():
    module = parse('''
        module demo.proof

        theorem add_zero(x: Int): x + 0 == x
        proof {
            simp
            omega
            assert x == x
            exact reflexivity
        }
    ''')

    theorem = module.items[0]
    assert isinstance(theorem, Theorem)
    assert [command.kind for command in theorem.proof] == ["simp", "omega", "assert", "exact"]
    assert isinstance(theorem.proof[3], ProofCommand)


def test_module_is_required():
    try:
        parse("fn main() -> Unit { return unit }")
    except ParseError as error:
        assert error.line == 1
        assert error.column == 1
    else:
        raise AssertionError("parser accepted a source file without module")


def test_multi_argument_call_does_not_capture_commas():
    module = parse('''
        module demo.call
        fn add(a: Int, b: Int) -> Int {
            return a
        }
        fn main() -> Unit {
            let value = add(1, 2)
            print value
            return unit
        }
    ''')
    call = module.items[1].body[0].value
    assert [type(argument).__name__ for argument in call.arguments] == [
        "IntLiteral", "IntLiteral",
    ]
    assert [argument.value for argument in call.arguments] == [1, 2]


def test_three_argument_call():
    module = parse('''
        module demo.call3
        fn sum3(a: Int, b: Int, c: Int) -> Int {
            return a
        }
        fn main() -> Unit {
            let value = sum3(1, 2, 3)
            print value
            return unit
        }
    ''')
    call = module.items[1].body[0].value
    assert len(call.arguments) == 3


LET_PRELUDE = '''
    module demo.binding
    record Point {
        x: Int
        y: Int
    }
    fn f(a: Int, b: Int) -> Int {
        return a
    }
    fn g(a: Int) -> Int {
        return a
    }
    fn main(a: Int, b: Int, c: Int, p: Point) -> Unit {
        %s
        return unit
    }
'''


def _binding_value(source: str):
    from gunmu.ast import Function

    module = parse(source)
    main = next(item for item in module.items if isinstance(item, Function) and item.name == "main")
    return main.body[0].value


def test_binding_rule_accepts_single_operation():
    from gunmu.ast import Binary, Call, FieldAccess, Name, RecordValue, Unary

    accepted = {
        "let x = 1": "IntLiteral",
        "let x = a": "Name",
        "let x = a + b": "Binary",
        "let x = a + 1": "Binary",
        "let x = 1 + 1": "Binary",
        "let x = a > c": "Binary",
        "let x = a and c": "Binary",
        "let x = -a": "Unary",
        "let x = not a": "Unary",
        "let x = f(a)": "Call",
        "let x = f(a, b)": "Call",
        "let x = p.x": "FieldAccess",
        "let x = p.x.y": "FieldAccess",
        "let x = Point { x: a, y: 1 }": "RecordValue",
    }
    for statement, expected in accepted.items():
        value = _binding_value(LET_PRELUDE % statement)
        assert type(value).__name__ == expected, statement


def test_binding_rule_rejects_nested_operators():
    rejected = [
        "let x = 1 + 1 + 1",
        "let x = f(a) + 1",
        "let x = p.x + 1",
        "let x = g(f(a))",
        "let x = f(a + 1)",
        "let x = Point { x: a + 1 }",
        "let x = (a + b)",
    ]
    for statement in rejected:
        try:
            parse(LET_PRELUDE % statement)
        except ParseError:
            continue
        raise AssertionError(f"binding rule accepted: {statement}")


def test_record_literal_in_binding_allows_trailing_comma():
    module = parse('''
        module demo.trailing
        record Point {
            x: Int
            y: Int
        }
        fn main(a: Int) -> Unit {
            let p = Point {
                x: a,
                y: 1,
            }
            return unit
        }
    ''')
    value = module.items[1].body[0].value
    assert type(value).__name__ == "RecordValue"
    assert set(value.fields) == {"x", "y"}


def test_binding_rule_applies_to_print_return_break():
    for statement in ("print a + 1 + 1", "return f(a) + 1", "break a + 1 + 1"):
        try:
            parse(LET_PRELUDE % statement)
        except ParseError:
            continue
        raise AssertionError(f"binding rule accepted: {statement}")

    # A single operation stays legal in all three positions.
    for statement in ("print a + 1", "return a + 1", "break a + 1"):
        parse(LET_PRELUDE % statement)


def test_loop_initial_state_uses_binding_rule():
    source = '''
        module demo.loopbind
        fn main(a: Int) -> Unit {
            loop (i = a + 1) {
                break
            }
            return unit
        }
    '''
    from gunmu.ast import Binary, LoopStmt

    loop = parse(source).items[0].body[0]
    assert isinstance(loop, LoopStmt)
    assert isinstance(loop.bindings["i"], Binary)

    nested = '''
        module demo.loopbind
        fn main(a: Int) -> Unit {
            loop (i = a + 1 + 1) {
                break
            }
            return unit
        }
    '''
    try:
        parse(nested)
    except ParseError:
        pass
    else:
        raise AssertionError("loop initial state accepted a nested operator")
