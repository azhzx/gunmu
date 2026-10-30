"""Every shipped example must parse, analyse, and verify as documented.

`examples/contracts.gunmu` deliberately contains functions that do not verify,
so this file also pins *why* each one fails: a change that makes an example fail
for a new reason is a change worth looking at.
"""

from pathlib import Path

import pytest

from gunmu import ProofStatus, analyze, canonicalize_module, parse, verify_module

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name: str):
    module = parse((EXAMPLES / name).read_text())
    analyze(module)
    return module


def statuses(name: str) -> dict[str, ProofStatus]:
    return {report.name: report.status for report in verify_module(load(name))}


def proofs(name: str) -> dict[str, ProofStatus]:
    from gunmu import check_theorem
    from gunmu.ast import Theorem

    module = load(name)
    table = {item.name: item for item in module.items if isinstance(item, Theorem)}
    return {
        theorem_name: check_theorem(theorem, table).status
        for theorem_name, theorem in table.items()
    }


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.gunmu")), ids=lambda p: p.name)
def test_every_example_parses_and_analyses(path: Path):
    module = parse(path.read_text())
    analyze(module)


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.gunmu")), ids=lambda p: p.name)
def test_every_example_survives_a_canonicalization_round_trip(path: Path):
    from gunmu import pretty_module

    module = parse(path.read_text())
    analyze(module)
    canonicalize_module(module)
    again = parse(pretty_module(module))
    analyze(again)
    canonicalize_module(again)
    assert canonicalize_module(again) == canonicalize_module(module)


def test_hello_verifies():
    assert statuses("hello.gunmu") == {"main": ProofStatus.PROVED}


def test_theorem_examples():
    assert proofs("proofs.gunmu") == {
        "add_zero": ProofStatus.PROVED,
        "positive_successor": ProofStatus.PROVED,
        "sum_non_negative": ProofStatus.PROVED,
        "impossible_claim": ProofStatus.DISPROVED,
    }


def test_control_verifies():
    assert statuses("control.gunmu") == {
        "classify": ProofStatus.PROVED,
        "count_to_ten": ProofStatus.PROVED,
        "main": ProofStatus.PROVED,
    }


def test_records_are_unsupported_for_the_stated_reason():
    result = statuses("records.gunmu")
    assert set(result) == {"horizontal", "main"}
    assert set(result.values()) == {ProofStatus.UNSUPPORTED}


def test_contracts_example_splits_into_what_holds_and_what_does_not():
    assert statuses("contracts.gunmu") == {
        # `g(x + 1)` and friends: every exit, precondition and loop obligation
        # closes.
        "add_one": ProofStatus.PROVED,
        "add_twice": ProofStatus.PROVED,
        "report": ProofStatus.PROVED,
        "count_to_ten": ProofStatus.PROVED,
        "is_even": ProofStatus.PROVED,
        "is_odd": ProofStatus.PROVED,
        "countdown": ProofStatus.PROVED,
        # And these fail, each for the reason the file documents.
        "claim_positive": ProofStatus.DISPROVED,
        "missing_upper_bound": ProofStatus.DISPROVED,
        "spins_forever": ProofStatus.DISPROVED,
        "broken_recursion": ProofStatus.DISPROVED,
        "no_invariant": ProofStatus.UNSUPPORTED,
    }