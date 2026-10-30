from gunmu import ProofStatus, SemanticError, analyze, check_theorem, parse, prove
from gunmu.ast import Binary, IntLiteral


def test_semantic_analysis_accepts_typed_module():
    module = parse('''
        module sema.ok

        record Point {
            x: Int
            y: Int
        }

        fn distance(point: Point) -> Int
        ensures result >= 0 {
            let x = point.x
            return x
        }
    ''')
    assert analyze(module) is module


def test_semantic_analysis_rejects_unknown_name():
    module = parse('''
        module sema.bad
        fn main() -> Unit {
            print missing
            return unit
        }
    ''')
    try:
        analyze(module)
    except SemanticError as error:
        assert "undefined name" in str(error)
    else:
        raise AssertionError("unknown name was accepted")


def test_semantic_analysis_checks_loop_state():
    module = parse('''
        module sema.loop
        fn main() -> Unit {
            loop (i = 0) {
                continue (i = true)
            }
            return unit
        }
    ''')
    try:
        analyze(module)
    except SemanticError as error:
        assert "loop state" in str(error)
    else:
        raise AssertionError("invalid loop state was accepted")


def test_z3_proves_arithmetic():
    module = parse('''
        module proof.ok
        theorem add_zero(x: Int): x + 0 == x
        proof {
            omega
        }
    ''')
    result = check_theorem(module.items[0])
    assert result.status == ProofStatus.PROVED


def test_z3_uses_assumptions():
    module = parse('''
        module proof.assumption
        theorem positive_sum(a: Int, b: Int)
        requires a >= 0 and b >= 0
        : a + b >= 0
        proof {
            omega
        }
    ''')
    assert check_theorem(module.items[0]).status == ProofStatus.PROVED


def test_z3_reports_counterexample():
    module = parse('''
        module proof.bad
        theorem false_claim(x: Int): x > x
        proof {
            omega
        }
    ''')
    result = check_theorem(module.items[0])
    assert result.status == ProofStatus.DISPROVED


def test_proof_api_accepts_expression():
    module = parse('''
        module proof.api
        fn value() -> Int {
            return 1
        }
    ''')
    expression = Binary(IntLiteral(1), "==", IntLiteral(1))
    result = prove(expression)
    assert result.status == ProofStatus.PROVED
