#!/usr/bin/env python3
"""
Standalone test runner for semdrift.parser — no pytest needed.

Run from the project root:
    python3 scripts/run_parser_tests.py
"""

import json
import os
import sys
import tempfile
import textwrap
import traceback

# Ensure the project root is on sys.path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from semdrift.parser.ast_parser import ASTParser, FunctionInfo
from semdrift.parser.doc_extractor import DocExtractor, DocInfo
from semdrift.parser.formatter import ModelInputFormatter
from semdrift.parser import parse_codebase


# ======================================================================
# Test samples
# ======================================================================

SIMPLE_FUNCTION = textwrap.dedent('''\
    def greet(name: str) -> str:
        """Return a greeting message for the given name."""
        return f"Hello, {name}!"
''')

CLASS_WITH_METHODS = textwrap.dedent('''\
    class Calculator:
        """A simple calculator."""

        def add(self, a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: The first number.
                b: The second number.

            Returns:
                The sum of a and b.
            """
            return a + b

        def divide(self, a: float, b: float) -> float:
            """Divide a by b.

            Args:
                a: Numerator.
                b: Denominator.

            Returns:
                The result of a / b.

            Raises:
                ZeroDivisionError: If b is zero.
            """
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a / b
''')

NUMPY_DOCSTRING = textwrap.dedent("""\
    Apply a scaling transformation.

    Parameters
    ----------
    data : array-like
        The input data.
    factor : float
        Scaling factor.

    Returns
    -------
    array-like
        The scaled data.
""")

SPHINX_DOCSTRING = textwrap.dedent("""\
    Connect to the server.

    :param host: The hostname.
    :param port: The port number.
    :returns: A connection object.
    :raises ConnectionError: If connection fails.
""")

GOOGLE_DOCSTRING = textwrap.dedent("""\
    Add two numbers.

    Args:
        a: The first number.
        b: The second number.

    Returns:
        The sum of a and b.

    Raises:
        TypeError: If inputs are not numbers.
""")

NO_DOCSTRING_FUNC = textwrap.dedent('''\
    def helper(x, y):
        return x + y
''')


# ======================================================================
# Test harness
# ======================================================================

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator that runs a test function and tracks results."""
    def decorator(func):
        global passed, failed
        try:
            func()
            passed += 1
            print(f"  ✅ {name}")
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  ❌ {name}: {e}")
        return func
    return decorator


# ======================================================================
# AST Parser tests
# ======================================================================

print("\n🔍 AST Parser Tests")
print("=" * 60)

parser = ASTParser()


@test("Simple function extraction")
def _():
    funcs = parser._parse_source(SIMPLE_FUNCTION, "<test>")
    assert len(funcs) == 1, f"Expected 1 function, got {len(funcs)}"
    f = funcs[0]
    assert f.name == "greet", f"Expected 'greet', got '{f.name}'"
    assert f.has_docstring is True
    assert "name" in f.params
    assert f.return_annotation == "str"


@test("Docstring removed from code field")
def _():
    funcs = parser._parse_source(SIMPLE_FUNCTION, "<test>")
    f = funcs[0]
    assert "Return a greeting" not in f.source_code, \
        "Docstring should NOT be in source_code"
    assert "Return a greeting" in f.docstring, \
        "Docstring should be in .docstring field"


@test("Class methods extraction")
def _():
    funcs = parser._parse_source(CLASS_WITH_METHODS, "<test>")
    assert len(funcs) == 2, f"Expected 2 methods, got {len(funcs)}"
    add_func = next(f for f in funcs if f.name == "add")
    assert add_func.class_name == "Calculator"
    assert "a" in add_func.params
    assert "b" in add_func.params


@test("Raise extraction")
def _():
    funcs = parser._parse_source(CLASS_WITH_METHODS, "<test>")
    div_func = next(f for f in funcs if f.name == "divide")
    assert "ZeroDivisionError" in div_func.raises, \
        f"Expected ZeroDivisionError in raises, got {div_func.raises}"


@test("No docstring handling")
def _():
    funcs = parser._parse_source(NO_DOCSTRING_FUNC, "<test>")
    assert len(funcs) == 1
    assert funcs[0].has_docstring is False
    assert funcs[0].docstring == ""


@test("Skip dunder methods")
def _():
    source = textwrap.dedent('''\
        class Foo:
            def __init__(self):
                """Init."""
                pass
            def bar(self):
                """Bar method."""
                pass
    ''')
    p = ASTParser(skip_dunder=True)
    funcs = p._parse_source(source, "<test>")
    names = [f.name for f in funcs]
    assert "__init__" not in names, "Dunder should be skipped"
    assert "bar" in names


@test("Complex params (*args, **kwargs, keyword-only)")
def _():
    source = textwrap.dedent('''\
        def complex_func(a, b, *args, keyword_only=True, **kwargs):
            """Complex params."""
            pass
    ''')
    funcs = parser._parse_source(source, "<test>")
    f = funcs[0]
    assert "*args" in f.params, f"Expected *args in {f.params}"
    assert "**kwargs" in f.params
    assert "keyword_only" in f.params


@test("Async function extraction")
def _():
    source = textwrap.dedent('''\
        async def fetch_data(url: str) -> dict:
            """Fetch JSON data from a remote URL."""
            pass
    ''')
    funcs = parser._parse_source(source, "<test>")
    assert len(funcs) == 1
    assert funcs[0].name == "fetch_data"


@test("Parse real .py file from disk")
def _():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(SIMPLE_FUNCTION)
        f.flush()
        try:
            funcs = parser.parse_file(f.name)
            assert len(funcs) == 1
            assert funcs[0].name == "greet"
        finally:
            os.unlink(f.name)


@test("Parse directory with multiple files")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "mod_a.py"), "w") as f:
            f.write(SIMPLE_FUNCTION)
        with open(os.path.join(tmpdir, "mod_b.py"), "w") as f:
            f.write(CLASS_WITH_METHODS)

        funcs = parser.parse_directory(tmpdir)
        names = {f.name for f in funcs}
        assert "greet" in names
        assert "add" in names
        assert "divide" in names


@test("Multi-line docstring correctly stripped from code")
def _():
    source = textwrap.dedent('''\
        def example():
            """This is a multi-line
            docstring that spans several lines.
            """
            x = 1
            return x
    ''')
    funcs = parser._parse_source(source, "<test>")
    f = funcs[0]
    assert "multi-line" in f.docstring
    assert "multi-line" not in f.source_code
    assert "x = 1" in f.source_code
    assert "return x" in f.source_code


# ======================================================================
# Doc Extractor tests
# ======================================================================

print("\n🔍 Doc Extractor Tests")
print("=" * 60)

extractor = DocExtractor()


@test("Google-style docstring parsing")
def _():
    info = extractor.extract(GOOGLE_DOCSTRING)
    assert info.summary == "Add two numbers."
    assert "a" in info.params, f"'a' not in params: {info.params}"
    assert "b" in info.params
    assert "TypeError" in info.raises


@test("NumPy-style docstring parsing")
def _():
    info = extractor.extract(NUMPY_DOCSTRING)
    assert "scaling" in info.summary.lower()
    assert "data" in info.params
    assert "factor" in info.params


@test("Sphinx-style docstring parsing")
def _():
    info = extractor.extract(SPHINX_DOCSTRING)
    assert "host" in info.params
    assert "port" in info.params
    assert info.returns
    assert "ConnectionError" in info.raises


@test("Plain docstring")
def _():
    info = extractor.extract("Just a plain description.")
    assert info.summary == "Just a plain description."
    assert info.params == {}


@test("Empty docstring")
def _():
    info = extractor.extract("")
    assert info.summary == ""


@test("Normalise produces flat text")
def _():
    info = extractor.extract(GOOGLE_DOCSTRING)
    normalised = extractor.normalise(info)
    assert "Add two numbers" in normalised
    assert "Parameters:" in normalised
    assert "Returns:" in normalised


# ======================================================================
# Formatter tests
# ======================================================================

print("\n🔍 Formatter Tests")
print("=" * 60)


def make_func_info(**kwargs):
    defaults = dict(
        name="my_func",
        class_name=None,
        file_path="/project/src/utils.py",
        line_start=1,
        line_end=5,
        source_code="def my_func(x):\n    return x",
        docstring="Does something useful.",
        has_docstring=True,
        params=["x"],
        return_annotation=None,
        decorators=[],
        raises=[],
    )
    defaults.update(kwargs)
    return FunctionInfo(**defaults)


@test("Output schema matches {function_id, code, docstring}")
def _():
    func = make_func_info()
    formatter = ModelInputFormatter(base_path="/project")
    records = formatter.format([func])
    assert len(records) == 1
    assert set(records[0].keys()) == {"function_id", "code", "docstring"}


@test("function_id includes file, class, and function name")
def _():
    func = make_func_info(
        name="fetch_user",
        class_name="UserService",
        file_path="/project/app/services.py",
    )
    formatter = ModelInputFormatter(base_path="/project")
    records = formatter.format([func])
    fid = records[0]["function_id"]
    assert "app_services_py" in fid
    assert "UserService" in fid
    assert "fetch_user" in fid
    assert fid.endswith("001")


@test("Undocumented functions excluded by default")
def _():
    func = make_func_info(has_docstring=False, docstring="")
    formatter = ModelInputFormatter()
    records = formatter.format([func])
    assert len(records) == 0


@test("Undocumented functions included when enabled")
def _():
    func = make_func_info(has_docstring=False, docstring="")
    formatter = ModelInputFormatter(include_undocumented=True)
    records = formatter.format([func])
    assert len(records) == 1


@test("JSON file output")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        func = make_func_info()
        formatter = ModelInputFormatter(base_path="/project")
        out_file = os.path.join(tmpdir, "output.json")
        json_str = formatter.format_to_json([func], output_path=out_file)
        assert os.path.exists(out_file)
        data = json.loads(json_str)
        assert len(data) == 1


@test("Counter increments for multiple functions")
def _():
    funcs = [make_func_info(name="func_a"), make_func_info(name="func_b")]
    formatter = ModelInputFormatter(base_path="/project")
    records = formatter.format(funcs)
    assert records[0]["function_id"].endswith("001")
    assert records[1]["function_id"].endswith("002")


# ======================================================================
# Integration: parse_codebase
# ======================================================================

print("\n🔍 Integration Tests (parse_codebase)")
print("=" * 60)


@test("parse_codebase on a single file")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "module.py")
        with open(src, "w") as f:
            f.write(SIMPLE_FUNCTION)
        records = parse_codebase(src)
        assert len(records) == 1
        assert "greet" in records[0]["code"]


@test("parse_codebase on a directory")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "a.py"), "w") as f:
            f.write(SIMPLE_FUNCTION)
        with open(os.path.join(tmpdir, "b.py"), "w") as f:
            f.write(CLASS_WITH_METHODS)
        records = parse_codebase(tmpdir)
        all_code = " ".join(r["code"] for r in records)
        assert "greet" in all_code
        assert "add" in all_code


@test("parse_codebase writes JSON output file")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "module.py")
        with open(src, "w") as f:
            f.write(SIMPLE_FUNCTION)
        out = os.path.join(tmpdir, "result.json")
        records = parse_codebase(src, output_path=out)
        assert os.path.exists(out)
        with open(out) as fh:
            data = json.load(fh)
        assert len(data) == len(records)


@test("parse_codebase schema matches CodeBERT input format")
def _():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "module.py")
        with open(src, "w") as f:
            f.write(CLASS_WITH_METHODS)
        records = parse_codebase(src)
        for rec in records:
            assert set(rec.keys()) == {"function_id", "code", "docstring"}, \
                f"Schema mismatch: {set(rec.keys())}"
            assert isinstance(rec["function_id"], str) and rec["function_id"]
            assert isinstance(rec["code"], str) and rec["code"]
            assert isinstance(rec["docstring"], str) and rec["docstring"]


# ======================================================================
# Summary
# ======================================================================

print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 60)

if errors:
    print("\n❌ Failed tests:")
    for name, err in errors:
        print(f"\n  {name}:")
        traceback.print_exception(type(err), err, err.__traceback__)

sys.exit(0 if failed == 0 else 1)
