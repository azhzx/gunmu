from gunmu import (
    Diagnostic, DiagnosticRenderer, Label, ParseError, SemanticError, Severity,
    SourceFile, Span, analyze, parse,
)


def _render(source: str, path: str = "demo.gunmu") -> str:
    source_file = SourceFile(path, source)
    try:
        analyze(parse(source))
    except (ParseError, SemanticError) as error:
        return DiagnosticRenderer().render(error.diagnostic, source_file)
    raise AssertionError("expected a diagnostic")


def test_parse_error_has_code_and_span():
    source = "module demo\nfn main() -> Unit {\n    print 1\n"
    try:
        parse(source)
    except ParseError as error:
        assert error.code == "E0001"
        assert error.span is not None
        assert error.span.line == 3
        assert error.diagnostic.severity == Severity.ERROR
        assert error.diagnostic.labels[0].primary
    else:
        raise AssertionError("parse should have failed")


def test_semantic_error_has_code_and_span():
    source = "module demo\nfn main() -> Unit {\n    print missing\n    return unit\n}\n"
    try:
        analyze(parse(source))
    except SemanticError as error:
        assert error.code == "E1006"
        assert error.span is not None
        assert error.span.line == 3
        assert error.span.column == 11
        assert error.span.width == len("missing")
    else:
        raise AssertionError("semantic analysis should have failed")


def test_renderer_underlines_primary_label():
    rendered = _render("module demo\nfn main() -> Unit {\n    print missing\n    return unit\n}\n")
    assert rendered.splitlines() == [
        "error[E1006]: undefined name: missing",
        "  --> demo.gunmu:3:11",
        "   |",
        " 3 |     print missing",
        "   |",
        "   |           ^^^^^^^ not defined",
        "  = note: declare it with `let` before use",
    ]


def test_renderer_gutter_width_follows_line_number():
    source = "module demo\n" + "\n" * 20 + "fn main() -> Unit {\n    print missing\n    return unit\n}\n"
    rendered = _render(source)
    lines = rendered.splitlines()
    assert lines[1] == "  --> demo.gunmu:23:11"
    # Two-digit line numbers widen the gutter so every `|` still lines up.
    assert lines[2] == "    |"
    assert lines[3] == " 23 |     print missing"
    assert lines[4] == "    |"
    assert lines[5] == "    |           ^^^^^^^ not defined"


def test_renderer_without_labels_keeps_notes():
    from gunmu import Diagnostic

    diagnostic = Diagnostic.error("E9999", "standalone")
    rendered = DiagnosticRenderer().render(diagnostic)
    assert rendered == "error[E9999]: standalone"


def test_renderer_highlights_let_one_operator_rule():
    rendered = _render("module demo\nfn main(a: Int) -> Unit {\n    let x = a + 1 + 1\n    return unit\n}\n")
    assert "single operator application" in rendered
    assert "operators never nest" in rendered


def test_span_width_from_columns():
    assert Span.at(4, 7, width=3).width == 3
    assert Span.at(4, 7).width == 1
    assert Span(10, 14, 2, 5, 2, 9).width == 4


def test_ast_nodes_carry_spans():
    module = parse("module demo\nfn main(value: Int) -> Unit {\n    print value\n    return unit\n}\n")
    function = module.items[0]
    assert function.span is not None
    assert function.span.line == 2
    statement = function.body[0]
    assert statement.span is not None
    assert statement.span.line == 3
    assert statement.value.span is not None


def test_ignoring_position_keeps_ast_equality():
    from gunmu.ast import IntLiteral

    assert IntLiteral(1) == IntLiteral(1)


def test_diagnostic_bag_collects():
    from gunmu import DiagnosticBag

    bag = DiagnosticBag()
    bag.error("E0001", "first")
    bag.error("E0002", "second")
    assert len(bag) == 2
    assert bag.has_errors
    assert [diagnostic.code for diagnostic in bag] == ["E0001", "E0002"]


def test_renderer_without_source():
    diagnostic = Diagnostic.error("E9999", "standalone", Span.at(2, 3), "here")
    rendered = DiagnosticRenderer().render(diagnostic)
    assert "error[E9999]: standalone" in rendered
    assert "--> 2:3" in rendered
