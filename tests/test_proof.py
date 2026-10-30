"""The theorem prover: what it claims, and what it refuses to claim."""

import pytest

from gunmu import ProofStatus, SemanticError, analyze, check_theorem, parse, prove
from gunmu.ast import Binary, IntLiteral
from gunmu.proof import Prover, presburger_violation


def theorem(source: str):
    module = parse(source)
    analyze(module)
    table = {item.name: item for item in module.items if hasattr(item, "proof")}
    return module, table


def result_of(source: str):
    """The verdict on the theorem named `t` in `source`."""
    _, table = theorem(source)
    return check_theorem(table["t"], table)


def status_of(source: str) -> ProofStatus:
    return result_of(source).status


# ------------------------------------------------------------------ free names


def test_bool_parameter_is_a_boolean():
    module, table = theorem("""
        module proof.bool
        theorem t(b: Bool): b or not b
        proof { simp }
    """)
    assert check_theorem(table["t"], table).status == ProofStatus.PROVED


def test_bool_parameter_is_not_treated_as_an_integer():
    # Every free name used to become an Int, so a Bool goal was meaningless.
    module, table = theorem("""
        module proof.bool_sort
        theorem t(b: Bool): b and not b
        proof { simp }
    """)
    assert check_theorem(table["t"], table).status == ProofStatus.DISPROVED


def test_string_parameter_compares_against_a_literal():
    module, table = theorem("""
        module proof.string
        theorem t(s: String): s == "hi"
        proof { simp }
    """)
    assert check_theorem(table["t"], table).status == ProofStatus.DISPROVED


def test_string_equality_reflexivity_is_proved():
    module, table = theorem("""
        module proof.string_refl
        theorem t(s: String): s == s
        proof { simp }
    """)
    assert check_theorem(table["t"], table).status == ProofStatus.PROVED


def test_theorem_without_parameters_is_unaffected():
    assert status_of("module p\ntheorem t(): 1 == 1\nproof { simp }") == ProofStatus.PROVED


# ---------------------------------------------------------------------- omega


def test_omega_proves_a_presburger_goal():
    assert status_of(
        "module p\ntheorem t(x: Int): x + 1 > x\nproof { omega }"
    ) == ProofStatus.PROVED


def test_omega_accepts_multiplication_by_a_literal():
    # Linear: an omega test decides this, and it is false.
    assert status_of(
        "module p\ntheorem t(x: Int): 2 * x >= 0\nproof { omega }"
    ) == ProofStatus.DISPROVED


def test_omega_rejects_multiplying_two_variables():
    result = result_of("""
        module p
        theorem t(x: Int): x * x >= 0
        proof { omega }
    """)
    assert result.status == ProofStatus.UNSUPPORTED
    assert "not linear" in result.message


def test_omega_handles_a_bool_parameter_in_its_own_sort():
    # The parameter is a Boolean, so the goal is a Boolean combination of
    # integer propositions rather than an arithmetic one.
    result = result_of("""
        module p
        theorem t(b: Bool, x: Int): b implies x >= x
        proof { omega }
    """)
    assert result.status == ProofStatus.PROVED


def test_omega_rejects_strings():
    result = result_of("""
        module p
        theorem t(s: String): s == "hi"
        proof { omega }
    """)
    assert result.status == ProofStatus.UNSUPPORTED
    assert "not Int" in result.message


def test_omega_rejects_a_goal_the_assumptions_already_left_the_fragment():
    result = result_of("""
        module p
        theorem t(x: Int, y: Int)
        requires y * y >= 0
        : x >= 0
        proof { omega }
    """)
    assert result.status == ProofStatus.UNSUPPORTED


def test_simp_is_the_unrestricted_escape_hatch():
    # The same goal `omega` declines: `simp` asks the solver regardless.
    assert status_of(
        "module p\ntheorem t(x: Int): x * x >= 0\nproof { simp }"
    ) == ProofStatus.PROVED


def test_presburger_violation_is_none_for_a_linear_goal():
    module, table = theorem("module p\ntheorem t(x: Int, b: Bool): b implies x >= x\nproof { simp }")
    goal = table["t"].proposition
    assert presburger_violation([goal], {"x": "Int", "b": "Bool"}) is None


# ------------------------------------------------------- undefined arithmetic


def test_division_is_reported_rather_than_answered():
    # z3's integer division is total, so it would "refute" this at x = 0 and
    # imply a meaning for `/` that the language does not have.
    result = result_of("""
        module p
        theorem t(x: Int): 10 / x * x == 10
        proof { simp }
    """)
    assert result.status == ProofStatus.UNSUPPORTED
    assert "division" in result.message


def test_remainder_is_reported_rather_than_answered():
    result = result_of("""
        module p
        theorem t(x: Int): x % 2 >= 0
        proof { simp }
    """)
    assert result.status == ProofStatus.UNSUPPORTED
    assert "remainder" in result.message


# ---------------------------------------------------------------------- exact


def test_exact_accepts_the_same_statement():
    module, table = theorem("""
        module p
        theorem base(x: Int): x == x
        proof { simp }
        theorem t(x: Int): x == x
        proof { exact base }
    """)
    assert check_theorem(table["t"], table).status == ProofStatus.PROVED


def test_exact_rejects_a_different_statement():
    module, table = theorem("""
        module p
        theorem base(x: Int): x == x
        proof { simp }
        theorem t(x: Int): x >= 0
        proof { exact base }
    """)
    result = check_theorem(table["t"], table)
    assert result.status == ProofStatus.DISPROVED
    assert "not the statement" in result.message


def test_exact_needs_the_referenced_theorem_to_hold():
    module, table = theorem("""
        module p
        theorem base(x: Int): x > 0
        proof { simp }
        theorem t(x: Int): x > 0
        proof { exact base }
    """)
    result = check_theorem(table["t"], table)
    assert result.status == ProofStatus.DISPROVED


def test_exact_reports_an_unknown_theorem():
    # `analyze` rejects this first, so the prover is reached without it.
    module = parse("module p\ntheorem t(x: Int): x == x\nproof { exact nowhere }")
    result = check_theorem(module.items[0], {})
    assert result.status == ProofStatus.UNSUPPORTED
    assert "unknown theorem" in result.message


def test_exact_refuses_a_circular_reference():
    module, table = theorem("""
        module p
        theorem a(x: Int): x == x
        proof { exact b }
        theorem b(x: Int): x == x
        proof { exact a }
    """)
    assert check_theorem(table["a"], table).status == ProofStatus.UNSUPPORTED


def test_exact_parses_the_referenced_name():
    # The `EXACT` terminal used to be read as the name itself.
    module, table = theorem("""
        module p
        theorem base(x: Int): x == x
        proof { simp }
        theorem t(x: Int): x == x
        proof { exact base }
    """)
    assert [(c.kind, c.value) for c in table["t"].proof] == [("exact", "base")]


def test_an_exact_reference_survives_canonicalization():
    from gunmu import canonicalize_module

    module, _ = theorem("""
        module p
        theorem base(x: Int): x == x
        proof { simp }
        theorem t(x: Int): x == x
        proof { exact base }
    """)
    canonicalize_module(module)  # must not raise an unresolved reference


def test_semantics_rejects_exact_of_an_unknown_name():
    with pytest.raises(SemanticError) as error:
        analyze(parse("module p\ntheorem t(x: Int): x == x\nproof { exact nowhere }"))
    assert error.value.code == "E1025"


def test_semantics_rejects_exact_of_a_function():
    source = """
        module p
        fn helper(x: Int) -> Int { return x }
        theorem t(x: Int): x == x
        proof { exact helper }
    """
    with pytest.raises(SemanticError) as error:
        analyze(parse(source))
    assert error.value.code == "E1025"


# ----------------------------------------------------------------- proof block


def test_an_empty_block_still_has_to_close_the_goal():
    assert status_of("module p\ntheorem t(x: Int): x == x\nproof { }") == ProofStatus.PROVED
    assert status_of("module p\ntheorem t(x: Int): x > x\nproof { }") == ProofStatus.DISPROVED


def test_assert_adds_a_fact():
    assert status_of("""
        module p
        theorem t(x: Int)
        requires x >= 0
        : x + 1 >= 1
        proof { assert x >= 0 }
    """) == ProofStatus.PROVED


def test_assert_that_does_not_hold_is_reported_with_its_own_witness():
    module, table = theorem("""
        module p
        theorem t(x: Int): x == 1
        proof { assert x == 2 }
    """)
    result = check_theorem(table["t"], table)
    assert result.status == ProofStatus.DISPROVED
    assert result.model is not None and "x = " in result.model


def test_simp_discharges_the_goal_in_one_solver_call():
    calls = []
    original = Prover._refute

    class Counting(Prover):
        def _refute(self, goal, context, symbols):
            calls.append(goal)
            return original(self, goal, context, symbols)

    module, table = theorem("module p\ntheorem t(x: Int): x + 0 == x\nproof { simp }")
    Counting().check_theorem(table["t"], table)
    assert len(calls) == 1


def test_assert_then_close_costs_one_call_per_step():
    calls = []
    original = Prover._refute

    class Counting(Prover):
        def _refute(self, goal, context, symbols):
            calls.append(goal)
            return original(self, goal, context, symbols)

    module, table = theorem("""
        module p
        theorem t(x: Int)
        requires x >= 0
        : x + 1 >= 1
        proof { assert x >= 0 }
    """)
    Counting().check_theorem(table["t"], table)
    # The asserted fact, then the goal. Not a third call for the trailing check.
    assert len(calls) == 2


# -------------------------------------------------------------- counterexample


def test_a_counterexample_reports_the_parameters_that_constrain_it():
    module, table = theorem("""
        module p
        theorem t(x: Int): x > 0
        proof { simp }
    """)
    result = check_theorem(table["t"], table)
    assert result.status == ProofStatus.DISPROVED
    assert result.model is not None and "x = " in result.model


def test_an_unconstrained_name_is_not_given_an_invented_value():
    module, table = theorem("""
        module p
        theorem t(x: Int, y: Int)
        requires x > 0
        : x > 1
        proof { simp }
    """)
    result = check_theorem(table["t"], table)
    assert result.status == ProofStatus.DISPROVED
    # `y` plays no part in the goal, so the model must not pretend to pin it.
    assert result.model is not None and "y" not in result.model


def test_a_counterexample_with_no_symbols_has_no_model():
    assert status_of("module p\ntheorem t(): 1 == 2\nproof { simp }") == ProofStatus.DISPROVED


# ------------------------------------------------------------------- raw API


def test_proof_api_accepts_an_expression():
    assert prove(Binary(IntLiteral(1), "==", IntLiteral(1))).status == ProofStatus.PROVED


def test_proof_api_accepts_declared_types():
    from gunmu.ast import Name, Unary

    goal = Binary(Name("b"), "or", Unary("not", Name("b")))
    assert prove(goal, types={"b": "Bool"}).status == ProofStatus.PROVED
    # Without the declaration the same goal is meaningless rather than proved.
    assert prove(goal).status == ProofStatus.UNSUPPORTED
