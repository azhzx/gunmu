"""Function contracts: what `verify_module` proves, and what it declines to."""

import pytest

from gunmu import ProofStatus, analyze, parse, verify_function, verify_module
from gunmu.ast import Function
from gunmu.verify import call_graph, contract_of


def build(source: str):
    module = parse(source)
    analyze(module)
    return module


def reports(source: str) -> dict:
    return {report.name: report for report in verify_module(build(source))}


def report_of(source: str) -> object:
    """The report for the one function named `f`."""
    return reports(source)["f"]


def failed(report) -> list:
    return [obligation.description for obligation in report.failures]


# ----------------------------------------------------------------- postcondition


def test_a_postcondition_is_proved():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        ensures result > x
        {
            let next = x + 1
            return next
        }
    """)
    assert report.status == ProofStatus.PROVED
    assert report.obligations


def test_a_false_postcondition_is_disproved_with_a_witness():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        ensures result > x
        {
            return 0 - x
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("postcondition" in description for description in failed(report))
    assert any("x = " in obligation.result.model for obligation in report.failures)


def test_a_contract_is_checked_at_every_exit():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        ensures result > 3
        {
            match x {
                4 => {
                    return x
                }
                _ => {
                    return 1
                }
            }
            return 0
        }
    """)
    # The `4 =>` arm discharges the postcondition; the `_` arm does not, and
    # checking only the first arm would have proved this. The `return 0` after
    # the match is unreachable and must not be checked at all.
    assert report.status == ProofStatus.DISPROVED
    assert [o.result.status for o in report.obligations] == [
        ProofStatus.PROVED, ProofStatus.DISPROVED,
    ]


def test_a_bool_result_is_bound_to_the_returned_expression():
    report = report_of("""
        module m
        fn f(x: Int) -> Bool
        ensures result
        {
            return x == x
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_bool_result_that_is_not_always_true_is_disproved():
    report = report_of("""
        module m
        fn f(x: Int) -> Bool
        ensures result
        {
            return x > 0
        }
    """)
    assert report.status == ProofStatus.DISPROVED


def test_a_string_result_is_bound_to_the_returned_expression():
    report = report_of("""
        module m
        fn f(s: String) -> String
        ensures result == s
        {
            return s
        }
    """)
    assert report.status == ProofStatus.PROVED


# ------------------------------------------------------------------- precondition


def test_a_precondition_is_available_inside_the_body():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            return x + 1
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_precondition_is_not_available_after_an_unrelated_branch():
    # Nothing establishes `x >= 0` on the path that skips the branch.
    report = report_of("""
        module m
        fn f(b: Bool, x: Int) -> Int
        ensures result > 0
        {
            match b {
                true => {
                    assert x >= 0
                }
                false => {
                }
            }
            return x + 1
        }
    """)
    assert report.status == ProofStatus.DISPROVED


def test_a_call_site_must_satisfy_the_callee_precondition():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            g(0 - x)
            return unit
        }
        fn g(n: Int) -> Unit
        requires n >= 0
        {
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("precondition of `g`" in description for description in failed(report))


def test_a_call_site_satisfying_the_precondition_is_accepted():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        requires x >= 0
        {
            g(x)
            return unit
        }
        fn g(n: Int) -> Unit
        requires n >= 0
        {
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_an_unproved_callee_contract_is_not_adopted():
    # `g`'s own postcondition is false, so `f` may not lean on it.
    statuses = {
        name: report.status for name, report in reports("""
            module m
            fn g(x: Int) -> Int
            ensures result > x
            {
                return 0 - x
            }
            fn f(x: Int) -> Int
            ensures result > x
            {
                let y = g(x)
                return y
            }
        """).items()
    }
    assert statuses["g"] == ProofStatus.DISPROVED
    assert statuses["f"] == ProofStatus.UNSUPPORTED


# ------------------------------------------------------------------------ asserts


def test_an_assert_is_proved_where_it_stands():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        requires x >= 0
        {
            assert x >= 0
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_false_assert_is_reported_with_its_own_witness():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            assert x == 7
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("x = 7" in description or "assert" in description for description in failed(report))


def test_an_assert_is_assumed_by_later_statements():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        requires x > 5
        ensures result > 0
        {
            assert x > 5
            return x
        }
    """)
    assert report.status == ProofStatus.PROVED


# ------------------------------------------------------------------------ match
#
# A `match` scrutinee is a leaf, so arms are reached by excluding what earlier
# literal arms already took. Reaching an arm that no state can reach must not
# raise its obligations.


def test_a_literal_arm_excludes_its_value_from_the_rest():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            match x {
                0 => {
                    assert x == 0
                    return unit
                }
                _ => {
                    assert x != 0
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_binding_arm_narrows_the_scrutinee():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            match x {
                n => {
                    assert n == x
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_guarded_arm_does_not_exclude_anything():
    # `x > 5` still admits `x == 0`, so it proves nothing about the next arm.
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            match x {
                _ if x > 5 => {
                    return unit
                }
                _ => {
                    assert x != 0
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED


def test_an_unreachable_arm_is_not_reported():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        requires x <= 3
        {
            match x {
                _ if x > 5 => {
                    assert x > 1000
                    return unit
                }
                _ => {
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED
    assert not report.obligations


def test_an_unreachable_literal_arm_is_not_reported():
    report = report_of("""
        module m
        fn f(b: Bool, x: Int) -> Unit
        requires x <= 3
        {
            match b {
                false => {
                    return unit
                }
                _ => {
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_paths_are_split_rather_than_merged():
    report = report_of("""
        module m
        fn f(b: Bool, x: Int) -> Unit
        {
            match b {
                true => {
                    assert x != 0
                    return unit
                }
                false => {
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("asserted proposition" in description for description in failed(report))


def test_a_fact_from_one_arm_does_not_reach_another():
    report = report_of("""
        module m
        fn f(b: Bool, x: Int) -> Unit
        {
            match b {
                true => {
                    assert x != 0
                    return unit
                }
                false => {
                    assert x != 0
                    return unit
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
# ----------------------------------------------------------------------- loops
#
# The loop head is modelled as an *arbitrary* iteration state tied to the
# initialisers by `head_x == initial`, so a `continue` has to hold for every
# state the invariant permits -- not just the first one. Several tests below
# rely on that: counting to a bound needs a `match` that excludes the bound,
# because `i + 1 <= 10` does not follow from `i <= 10`.


def test_an_invariant_must_hold_on_entry():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with { invariant i >= 5 }
            {
                break
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("invariant" in description for description in failed(report))


def test_a_continue_must_re_establish_the_invariant():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_continue_is_checked_against_every_state_the_invariant_allows():
    # The first iteration is fine at `i = 0`; an arbitrary iteration with
    # `i == 10` is not, and nothing here excludes that.
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                continue (i = i + 1)
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("invariant" in description for description in failed(report))


def test_a_continue_that_breaks_the_invariant_is_reported():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                continue (i = i + 11)
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("invariant" in description for description in failed(report))


def test_a_variant_must_be_non_negative_on_entry():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 5)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 0 - i
            }
            {
                break
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("non-negative" in description for description in failed(report))


def test_a_variant_must_strictly_decrease():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                continue (i = i)
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("decreases" in description for description in failed(report))


def test_a_variant_that_goes_negative_is_reported():
    # Only `i >= 0` is assumed, so `10 - i` need not stay non-negative.
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                variant 10 - i
            }
            {
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("non-negative" in description for description in failed(report))


def test_a_loop_that_cannot_terminate_is_reported():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                print i
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED
    assert any("may not terminate" in description for description in failed(report))


def test_a_loop_without_an_invariant_is_unsupported():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0) {
                continue (i = i + 1)
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.UNSUPPORTED
    assert any("invariant" in description for description in failed(report))


def test_a_loop_break_exits_without_raising_an_obligation():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED
    assert not any("break" in description for description in failed(report))


def test_the_initial_state_is_tied_to_the_loop_head():
    # `invariant i == 0` holds on entry because the head is pinned to the
    # initialiser, not merely assumed.
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 3)
            with { invariant i == 3 }
            {
                break
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_loop_state_does_not_escape_the_loop():
    from gunmu import SemanticError

    with pytest.raises(SemanticError) as error:
        build("""
            module m
            fn f() -> Int
            {
                loop (i = 0)
                with { invariant i >= 0 }
                {
                    break
                }
                return i
            }
        """)
    assert error.value.code == "E1006"


def test_a_precondition_reaches_a_loop_body():
    # Regression: the enclosing path condition used to be dropped when the loop
    # body was walked, so this `assert` could not see `requires x >= 0`.
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        requires x >= 0
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                assert x >= 0
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_nested_loops_are_checked_separately():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                loop (j = 0)
                with {
                    invariant j >= 0
                    invariant j <= 10
                    variant 10 - j
                }
                {
                    match j {
                        10 => { break }
                        _ => { continue (j = j + 1) }
                    }
                }
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_an_inner_loop_sees_the_outer_invariant():
    # `assert i <= 10` in the inner body is only provable because the enclosing
    # loop's invariant is still in scope there.
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                loop (j = 0)
                with {
                    invariant j >= 0
                    invariant j <= 10
                    variant 10 - j
                }
                {
                    assert i <= 10
                    match j {
                        10 => { break }
                        _ => { continue (j = j + 1) }
                    }
                }
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_an_inner_loop_can_be_seeded_from_the_outer_one():
    report = report_of("""
        module m
        fn f() -> Unit
        {
            loop (i = 0)
            with {
                invariant i >= 0
                invariant i <= 10
                variant 10 - i
            }
            {
                loop (j = i)
                with {
                    invariant j >= 0
                    invariant j <= 10
                    variant j
                }
                {
                    match j {
                        0 => { break }
                        _ => { continue (j = j - 1) }
                    }
                }
                match i {
                    10 => { break }
                    _ => { continue (i = i + 1) }
                }
            }
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED
# --------------------------------------------------------------------- recursion
#
# A recursive group verifies against its own contracts, which the call graph's
# strongly connected components decide. Each recursive call is checked against
# the callee's precondition, and the postcondition is what makes the caller's
# goal provable.


def one(source: str, name: str):
    return reports(source)[name]


def test_recursion_is_discharged_inductively():
    report = one("""
        module m
        fn countdown(n: Int) -> Int
        requires n >= 0
        ensures result >= 0
        {
            match n {
                0 => {
                    return 0
                }
                _ => {
                    let smaller = n - 1
                    let rest = countdown(smaller)
                    return rest
                }
            }
            return 0
        }
    """, "countdown")
    assert report.status == ProofStatus.PROVED
    assert any("precondition of `countdown`" in o.description for o in report.obligations)


def test_a_recursive_call_that_breaks_the_precondition_is_reported():
    report = one("""
        module m
        fn countdown(n: Int) -> Unit
        requires n >= 0
        {
            match n {
                0 => {
                    return unit
                }
                _ => {
                    countdown(0 - n)
                    return unit
                }
            }
            return unit
        }
    """, "countdown")
    assert report.status == ProofStatus.DISPROVED
    assert any("precondition of `countdown`" in o.description for o in report.failures)


def test_a_recursive_call_cannot_lean_on_its_own_postcondition_unless_it_holds():
    report = one("""
        module m
        fn countdown(n: Int) -> Int
        requires n >= 0
        ensures result > 100
        {
            match n {
                0 => {
                    return 0
                }
                _ => {
                    let smaller = n - 1
                    let rest = countdown(smaller)
                    return rest
                }
            }
            return 0
        }
    """, "countdown")
    assert report.status == ProofStatus.DISPROVED


def test_mutual_recursion_is_discharged_inductively():
    statuses = {
        name: report.status
        for name, report in reports("""
            module m
            fn is_even(n: Int) -> Bool
            requires n >= 0
            ensures result == true or result == false
            {
                match n {
                    0 => {
                        return true
                    }
                    _ => {
                        let smaller = n - 1
                        return is_odd(smaller)
                    }
                }
                return false
            }
            fn is_odd(n: Int) -> Bool
            requires n >= 0
            ensures result == true or result == false
            {
                match n {
                    0 => {
                        return false
                    }
                    _ => {
                        let smaller = n - 1
                        return is_even(smaller)
                    }
                }
                return true
            }
        """).items()
    }
    assert statuses == {"is_even": ProofStatus.PROVED, "is_odd": ProofStatus.PROVED}


def test_a_function_may_call_one_declared_later():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            let y = g(x)
            return y
        }
        fn g(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            return x + 1
        }
    """)
    assert report.status == ProofStatus.PROVED


# ---------------------------------------------------------------------- reports


def test_a_function_without_a_contract_still_checks_its_asserts():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            assert x == 3
            return unit
        }
    """)
    assert report.status == ProofStatus.DISPROVED


def test_a_function_without_a_contract_is_proved_when_it_has_nothing_to_prove():
    report = report_of("""
        module m
        fn f(x: Int) -> Unit
        {
            print x
            return unit
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_reports_come_back_in_dependency_order():
    order = [
        report.name
        for report in verify_module(build("""
            module m
            fn a() -> Unit {
                b()
                return unit
            }
            fn b() -> Unit {
                c()
                return unit
            }
            fn c() -> Unit {
                return unit
            }
        """))
    ]
    assert order.index("c") < order.index("b") < order.index("a")


def test_contract_of_reads_the_signature():
    module = build("""
        module m
        fn f(x: Int, y: Bool) -> Int
        requires x >= 0
        ensures result > 0
        {
            return x
        }
    """)
    contract = contract_of(module.items[0])
    assert contract.params == (("x", "Int"), ("y", "Bool"))
    assert contract.return_type == "Int"
    assert len(contract.requires) == 1
    assert len(contract.ensures) == 1
    assert contract.has_post


def test_contract_of_an_empty_spec_is_empty():
    module = build("""
        module m
        fn f(x: Int) -> Int {
            return x
        }
    """)
    contract = contract_of(module.items[0])
    assert contract.requires == ()
    assert contract.ensures == ()
    assert not contract.has_post


def test_verify_function_in_isolation_assumes_sibling_contracts():
    module = build("""
        module m
        fn g(x: Int) -> Int
        ensures result > 0
        {
            return 0 - x
        }
        fn f(x: Int) -> Int
        ensures result > 0
        {
            let y = g(x)
            return y
        }
    """)
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    # `f`'s own body is fine given `g`'s contract, even though `g` is broken.
    assert verify_function(functions["f"], module).status == ProofStatus.PROVED
    # The module as a whole does not hold.
    statuses = {report.name: report.status for report in verify_module(module)}
    assert statuses["g"] == ProofStatus.DISPROVED


def test_verify_function_without_a_module_encodes_nothing_else():
    module = build("""
        module m
        fn g(x: Int) -> Int
        ensures result > 0
        {
            return x + 1
        }
        fn f(x: Int) -> Int
        ensures result > 0
        {
            let y = g(x)
            return y
        }
    """)
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    assert verify_function(functions["f"]).status == ProofStatus.UNSUPPORTED


def test_call_graph_finds_direct_calls():
    module = build("""
        module m
        fn a() -> Unit {
            b()
            return unit
        }
        fn b() -> Unit {
            return unit
        }
        fn c() -> Unit {
            return unit
        }
    """)
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    assert call_graph(functions) == {"a": {"b"}, "b": set(), "c": set()}


def test_call_graph_sees_through_nested_calls():
    module = build("""
        module m
        fn a() -> Unit {
            let x = b()
            print x
            return unit
        }
        fn b() -> Int {
            return c()
        }
        fn c() -> Int {
            return 1
        }
    """)
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    assert call_graph(functions)["b"] == {"c"}


def test_call_graph_reports_self_recursion():
    module = build("""
        module m
        fn a(n: Int) -> Unit {
            a(n)
            return unit
        }
    """)
    functions = {item.name: item for item in module.items if isinstance(item, Function)}
    assert call_graph(functions) == {"a": {"a"}}


# ------------------------------------------------------------- encoding limits


def test_a_record_postcondition_is_unsupported_rather_than_assumed():
    report = report_of("""
        module m
        record Point { x: Int }
        fn f(p: Point) -> Point
        requires p.x == 0
        ensures result.x == 0
        {
            return p
        }
    """)
    assert report.status == ProofStatus.UNSUPPORTED
    assert any("Point" in obligation.result.message for obligation in report.failures)


def test_the_encoding_limit_names_what_is_available():
    report = report_of("""
        module m
        record Point { x: Int }
        fn f(p: Point) -> Unit {
            return unit
        }
    """)
    reasons = [obligation.result.message for obligation in report.failures]
    assert any("Bool, Int, String" in reason for reason in reasons)


def test_division_is_unsupported_inside_a_body():
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        requires x != 0
        ensures result * x == x
        {
            let q = 10 / x
            return q
        }
    """)
    assert report.status == ProofStatus.UNSUPPORTED


def test_a_unit_return_keeps_result_unencodable_rather_than_inventing_it():
    report = report_of("""
        module m
        fn f() -> Unit
        ensures result == unit
        {
            return unit
        }
    """)
    assert report.status == ProofStatus.UNSUPPORTED
    assert any("Unit" in obligation.result.message for obligation in report.failures)


def test_string_parameters_are_reasoned_about_in_their_own_sort():
    report = report_of("""
        module m
        fn f(s: String) -> Bool
        ensures result
        {
            return s == s
        }
    """)
    assert report.status == ProofStatus.PROVED


def test_a_string_equality_that_does_not_hold_is_disproved():
    report = report_of("""
        module m
        fn f(s: String) -> Bool
        ensures result
        {
            return s == "hi"
        }
    """)
    assert report.status == ProofStatus.DISPROVED


def test_a_callee_postcondition_feeds_the_next_call():
    # `h`'s precondition is discharged by `g`'s postcondition, which is what
    # makes the chain work.
    report = report_of("""
        module m
        fn f(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            let y = g(x)
            let z = h(y)
            return z
        }
        fn g(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            return x + 1
        }
        fn h(x: Int) -> Int
        requires x >= 0
        ensures result > 0
        {
            return x + 1
        }
    """)
    assert report.status == ProofStatus.PROVED