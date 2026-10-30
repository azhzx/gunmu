import sqlite3

import pytest

from gunmu import parse
from gunmu.canonical import canonicalize_module
from gunmu.db import Codebase, CodebaseError
from gunmu.hashing import definition_hash, resolve_prefix, short_hash
from gunmu.pretty import pretty_module


def hashes_of(source: str) -> dict[str, str]:
    return {d.local_name: d.hash for d in canonicalize_module(parse(source)).definitions}


ADD = """
module demo
fn add(x: Int, y: Int) -> Int {
    let s = x + y
    return s
}
"""

RECURSIVE = """
module demo
fn count(n: Int) -> Int {
    match n {
        0 => {
            return 0
        }
        _ => {
            return count(n)
        }
    }
}
"""

MUTUAL = """
module demo
fn is_even(n: Int) -> Bool {
    match n {
        0 => {
            return true
        }
        _ => {
            return is_odd(n)
        }
    }
}
fn is_odd(n: Int) -> Bool {
    match n {
        0 => {
            return false
        }
        _ => {
            return is_even(n)
        }
    }
}
"""

DEPENDENT = """
module demo
fn helper(x: Int) -> Int {
    return x
}
fn main(x: Int) -> Int {
    return helper(x)
}
"""

DEPENDENT_EDITED = """
module demo
fn helper(x: Int) -> Int {
    return x + 1
}
fn main(x: Int) -> Int {
    return helper(x)
}
"""


# ------------------------------------------------------------- content addressing


def test_renaming_locals_does_not_change_hash():
    renamed = ADD.replace("x: Int, y: Int", "alpha: Int, beta: Int")
    renamed = renamed.replace("let s = x + y", "let total = alpha + beta")
    renamed = renamed.replace("return s", "return total")
    assert hashes_of(ADD)["add"] == hashes_of(renamed)["add"]


def test_whitespace_and_spans_do_not_change_hash():
    padded = "\n\nmodule demo\n\n\nfn add(x: Int, y: Int) -> Int {\n        let s = x + y\n\n        return s\n}\n\n"
    assert hashes_of(ADD)["add"] == hashes_of(padded)["add"]


def test_definition_name_is_metadata():
    renamed = ADD.replace("fn add(", "fn plus(")
    assert hashes_of(ADD)["add"] == hashes_of(renamed)["plus"]


def test_semantic_change_changes_hash():
    edited = ADD.replace("let s = x + y", "let s = x - y")
    assert hashes_of(ADD)["add"] != hashes_of(edited)["add"]


def test_dependency_change_propagates_upwards():
    before = hashes_of(DEPENDENT)
    after = hashes_of(DEPENDENT_EDITED)
    assert before["helper"] != after["helper"]
    # `main` is textually untouched but must change, because its dependency did.
    assert before["main"] != after["main"]


def test_consistent_parameter_swap_is_alpha_equivalence():
    """Renaming parameters and the references to them together changes nothing."""
    swapped = ADD.replace("fn add(x: Int, y: Int)", "fn add(y: Int, x: Int)")
    swapped = swapped.replace("let s = x + y", "let s = y + x")
    assert hashes_of(ADD)["add"] == hashes_of(swapped)["add"]


def test_parameter_types_bind_to_positions():
    """Types are positional, so reordering differently typed parameters differs."""
    first = "module d\nfn f(x: Int, y: String) -> Unit {\n    print x\n    return unit\n}\n"
    second = "module d\nfn f(a: String, b: Int) -> Unit {\n    print b\n    return unit\n}\n"
    assert hashes_of(first)["f"] != hashes_of(second)["f"]


def test_self_recursion_is_marked_self():
    module = canonicalize_module(parse(RECURSIVE))
    definition = module.definitions[0]
    assert "(self)" in definition.canonical
    assert definition.group_hash is None


def test_mutual_recursion_becomes_one_group():
    module = canonicalize_module(parse(MUTUAL))
    assert len(module.groups) == 1
    group = module.groups[0]
    assert set(group.members) == {"is_even", "is_odd"}
    for definition in module.definitions:
        assert definition.group_hash == group.hash
        assert "(peer " in definition.canonical
    assert len({d.hash for d in module.definitions}) == 2


def test_string_literals_survive_canonicalization():
    with_paren = 'module d\nfn s() -> String {\n    return " ) "\n}\n'
    without = 'module d\nfn s() -> String {\n    return " )"\n}\n'
    assert hashes_of(with_paren)["s"] != hashes_of(without)["s"]
    canonical = canonicalize_module(parse(with_paren)).definitions[0].canonical
    assert '" ) "' in canonical


def test_hash_has_unison_shape():
    value = hashes_of(ADD)["add"]
    assert value.startswith("#")
    assert len(value) > 40
    assert short_hash(value, 12) == value[:13]


def test_unresolved_reference_is_reported():
    from gunmu.canonical import UnresolvedReference

    source = "module d\nfn main(x: Int) -> Int {\n    return missing(x)\n}\n"
    with pytest.raises(UnresolvedReference):
        canonicalize_module(parse(source))


def test_prefix_resolution():
    value = hashes_of(ADD)["add"]
    assert resolve_prefix(value[:12], [value]) == [value]
    assert resolve_prefix("#zzz", [value]) == []


# ---------------------------------------------------------------- pretty printer


@pytest.mark.parametrize("source", [ADD, RECURSIVE, MUTUAL, DEPENDENT])
def test_pretty_roundtrip_preserves_ast_and_hash(source):
    original = parse(source)
    rendered = pretty_module(original)
    reparsed = parse(rendered)
    assert reparsed == original
    assert hashes_of(source) == {
        d.local_name: d.hash for d in canonicalize_module(reparsed).definitions
    }


def test_pretty_roundtrip_on_repository_examples():
    from pathlib import Path

    for path in sorted(Path("examples").glob("*.gunmu")):
        original = parse(path.read_text())
        reparsed = parse(pretty_module(original))
        assert reparsed == original, path


# -------------------------------------------------------------------- codebase


@pytest.fixture()
def codebase(tmp_path):
    base = Codebase(tmp_path / "home")
    base.init()
    return base


def test_import_is_idempotent(codebase):
    first = codebase.import_module(parse(ADD))
    assert len(first.added) == 1
    second = codebase.import_module(parse(ADD))
    assert second.added == []
    assert len(second.unchanged) == 1


def test_imported_definition_can_be_read_back(codebase):
    codebase.import_module(parse(ADD))
    record = codebase.get("demo.add")
    assert record.kind == "fn"
    assert record.hash == hashes_of(ADD)["add"]
    assert "fn add(" in record.source


def test_rename_keeps_hashes(codebase):
    codebase.import_module(parse(ADD))
    before = codebase.get("demo.add").hash
    codebase.rename("demo.add", "demo.plus")
    after = codebase.get("demo.plus").hash
    assert before == after
    # the definition row is untouched: same hash, same canonical text
    assert codebase.get("demo.plus").canonical == codebase.get(before).canonical
    with pytest.raises(CodebaseError):
        codebase.get("demo.add")


def test_rename_rejects_conflicting_target(codebase):
    codebase.import_module(parse(ADD))
    different = ADD.replace("fn add(", "fn other(").replace("let s = x + y", "let s = x - y")
    codebase.import_module(parse(different))
    assert codebase.get("demo.add").hash != codebase.get("demo.other").hash
    with pytest.raises(CodebaseError) as error:
        codebase.rename("demo.add", "demo.other")
    assert error.value.code == "E3006"
    # forcing overwrites the name but leaves both definitions stored
    add_hash = codebase.get("demo.add").hash
    codebase.rename("demo.add", "demo.other", force=True)
    assert codebase.get("demo.other").hash == add_hash


def test_rename_to_an_equivalent_name_is_not_a_conflict(codebase):
    """Same structure means same hash, so this is the same definition, not a clash."""
    codebase.import_module(parse(ADD))
    codebase.import_module(parse(ADD.replace("fn add(", "fn plus(")))
    assert codebase.get("demo.add").hash == codebase.get("demo.plus").hash
    codebase.rename("demo.add", "demo.plus")
    assert codebase.get("demo.plus").hash == hashes_of(ADD)["add"]


def test_dependencies_are_recorded(codebase):
    codebase.import_module(parse(DEPENDENT))
    main = codebase.get("demo.main")
    helper = codebase.get("demo.helper")
    assert codebase.dependencies("demo.main") == [helper.hash]
    assert codebase.dependencies("demo.helper", reverse=True) == [main.hash]
    assert codebase.dependencies("demo.main", transitive=True) == [helper.hash]


def test_structural_identity_shares_one_definition(codebase):
    """Two textually different definitions with the same structure share a hash."""
    codebase.import_module(parse(ADD))
    codebase.import_module(parse(ADD.replace("fn add(", "fn plus(")))
    assert codebase.get("demo.add").hash == codebase.get("demo.plus").hash
    assert len(codebase.entries()) == 2


def test_export_roundtrips_through_the_codebase(codebase):
    codebase.import_module(parse(DEPENDENT))
    text = codebase.export("demo.main")
    reparsed = parse(text)
    reparsed_hashes = {
        d.local_name: d.hash for d in canonicalize_module(reparsed).definitions
    }
    original_hashes = hashes_of(DEPENDENT)
    assert reparsed_hashes == original_hashes


def test_verify_passes_on_a_fresh_codebase(codebase):
    for source in (ADD, RECURSIVE, MUTUAL, DEPENDENT):
        codebase.import_module(parse(source))
    assert codebase.verify() == []


def test_verify_detects_tampering(codebase):
    codebase.import_module(parse(ADD))
    with sqlite3.connect(codebase.path) as connection:
        connection.execute("UPDATE def SET canonical = '(fn (body))'")
    problems = codebase.verify(deep=False)
    assert problems
    assert any("recomputed hash" in problem for problem in problems)


def test_verify_detects_dangling_namespace_entry(codebase):
    codebase.import_module(parse(ADD))
    with sqlite3.connect(codebase.path) as connection:
        connection.execute(
            "INSERT INTO namespace (name, hash, updated_at) VALUES ('demo.ghost', '#deadbeef', 'now')"
        )
    problems = codebase.verify()
    assert any("demo.ghost" in problem for problem in problems)


def test_unknown_reference_reports_a_code(codebase):
    with pytest.raises(CodebaseError) as error:
        codebase.get("demo.nope")
    assert error.value.code == "E3004"


def test_missing_codebase_reports_a_code(tmp_path):
    base = Codebase(tmp_path / "absent")
    with pytest.raises(CodebaseError) as error:
        base.entries()
    assert error.value.code == "E3001"


def test_verify_handles_multiple_versions_of_a_name(codebase):
    """Append-only storage keeps superseded versions, which must not break verify."""
    codebase.import_module(parse(DEPENDENT))
    codebase.import_module(parse(DEPENDENT_EDITED))
    assert codebase.verify() == []
    # both versions are still addressable even though one name moved on
    assert len({d.hash for d in canonicalize_module(parse(DEPENDENT)).definitions}) == 2
    old = hashes_of(DEPENDENT)["helper"]
    assert codebase.get(old).canonical
    assert codebase.get(old).hash != codebase.get("demo.helper").hash


def test_scc_members_are_stored_and_verified(codebase):
    codebase.import_module(parse(MUTUAL))
    assert codebase.verify() == []
    even = codebase.get("demo.is_even")
    odd = codebase.get("demo.is_odd")
    assert even.group_hash == odd.group_hash
    assert even.group_index != odd.group_index
