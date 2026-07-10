"""Tests for the semdrift.parser module.

Covers AST parsing, docstring extraction (Google, NumPy, Sphinx, plain),
model input formatting, and the top-level ``parse_codebase`` convenience
function.
"""

import json
import os
import textwrap

import pytest

from semdrift.parser.ast_parser import ASTParser, FunctionInfo
from semdrift.parser.doc_extractor import DocExtractor, DocInfo
from semdrift.parser.formatter import ModelInputFormatter
from semdrift.parser import parse_codebase


# ======================================================================
# Sample Python source strings used as test fixtures
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

NUMPY_DOCSTRING_FUNC = textwrap.dedent('''\
    def transform(data, factor=1.0):
        """Apply a scaling transformation to the data.

        Parameters
        ----------
        data : array-like
            The input data to transform.
        factor : float
            Scaling factor to apply.

        Returns
        -------
        array-like
            The scaled data.

        Raises
        ------
        ValueError
            If factor is negative.
        """
        if factor < 0:
            raise ValueError("factor must be non-negative")
        return [x * factor for x in data]
''')

SPHINX_DOCSTRING_FUNC = textwrap.dedent('''\
    def connect(host, port=8080):
        """Establish a connection to the remote server.

        :param host: The hostname or IP address.
        :param port: The port number (default 8080).
        :returns: A connection object.
        :raises ConnectionError: If the connection fails.
        """
        pass
''')

NO_DOCSTRING_FUNC = textwrap.dedent('''\
    def helper(x, y):
        return x + y
''')

DECORATED_FUNCTION = textwrap.dedent('''\
    class Service:
        @staticmethod
        def process(item: str) -> bool:
            """Process a single item."""
            return bool(item)

        @property
        def name(self) -> str:
            """The service name."""
            return "MyService"
''')

COMPLEX_PARAMS_FUNC = textwrap.dedent('''\
    def complex_func(a, b, *args, keyword_only=True, **kwargs):
        """A function with complex parameter types.

        Args:
            a: First argument.
            b: Second argument.
            *args: Variable positional args.
            keyword_only: A keyword-only argument.
            **kwargs: Variable keyword args.
        """
        pass
''')

ASYNC_FUNCTION = textwrap.dedent('''\
    async def fetch_data(url: str) -> dict:
        """Fetch JSON data from a remote URL."""
        pass
''')


# ======================================================================
# AST Parser Tests
# ======================================================================

class TestASTParser:
    """Tests for the ASTParser class."""

    def setup_method(self):
        self.parser = ASTParser()

    def _parse_source(self, source: str) -> list[FunctionInfo]:
        """Helper: parse a source string via the internal method."""
        return self.parser._parse_source(source, "<test>")

    # ---- Basic extraction ----

    def test_simple_function(self):
        """A standalone function with a one-line docstring."""
        funcs = self._parse_source(SIMPLE_FUNCTION)
        assert len(funcs) == 1

        f = funcs[0]
        assert f.name == "greet"
        assert f.class_name is None
        assert f.has_docstring is True
        assert "name" in f.params
        assert f.return_annotation == "str"
        # Docstring must NOT appear in the code field.
        assert "Return a greeting" not in f.source_code
        # But it must be captured in the docstring field.
        assert "Return a greeting" in f.docstring

    def test_class_methods(self):
        """Methods inside a class get their class_name set."""
        funcs = self._parse_source(CLASS_WITH_METHODS)
        assert len(funcs) == 2

        add_func = next(f for f in funcs if f.name == "add")
        assert add_func.class_name == "Calculator"
        assert "a" in add_func.params
        assert "b" in add_func.params
        assert add_func.return_annotation == "int"

        div_func = next(f for f in funcs if f.name == "divide")
        assert "ZeroDivisionError" in div_func.raises
        assert div_func.has_docstring is True

    def test_no_docstring(self):
        """Functions without docstrings should still be extracted."""
        funcs = self._parse_source(NO_DOCSTRING_FUNC)
        assert len(funcs) == 1
        assert funcs[0].has_docstring is False
        assert funcs[0].docstring == ""

    def test_decorators(self):
        """Decorators are captured in the decorators list."""
        funcs = self._parse_source(DECORATED_FUNCTION)

        process_func = next(f for f in funcs if f.name == "process")
        assert "staticmethod" in process_func.decorators

        name_func = next(f for f in funcs if f.name == "name")
        assert "property" in name_func.decorators

    def test_complex_params(self):
        """*args, keyword-only, and **kwargs are all captured."""
        funcs = self._parse_source(COMPLEX_PARAMS_FUNC)
        assert len(funcs) == 1

        f = funcs[0]
        assert "a" in f.params
        assert "b" in f.params
        assert "*args" in f.params
        assert "keyword_only" in f.params
        assert "**kwargs" in f.params

    def test_async_function(self):
        """Async functions are extracted just like sync ones."""
        funcs = self._parse_source(ASYNC_FUNCTION)
        assert len(funcs) == 1
        assert funcs[0].name == "fetch_data"
        assert funcs[0].has_docstring is True

    # ---- Skip filters ----

    def test_skip_dunder(self):
        """With skip_dunder=True, dunder methods are excluded."""
        source = textwrap.dedent('''\
            class Foo:
                def __init__(self):
                    """Init."""
                    pass

                def bar(self):
                    """Bar method."""
                    pass
        ''')
        parser = ASTParser(skip_dunder=True)
        funcs = parser._parse_source(source, "<test>")
        names = [f.name for f in funcs]
        assert "__init__" not in names
        assert "bar" in names

    def test_skip_private(self):
        """With skip_private=True, private methods are excluded."""
        source = textwrap.dedent('''\
            class Foo:
                def _internal(self):
                    """Private helper."""
                    pass

                def public(self):
                    """Public method."""
                    pass
        ''')
        parser = ASTParser(skip_private=True)
        funcs = parser._parse_source(source, "<test>")
        names = [f.name for f in funcs]
        assert "_internal" not in names
        assert "public" in names

    # ---- File and directory operations ----

    def test_parse_file(self, tmp_path):
        """parse_file reads a .py file and returns FunctionInfo."""
        test_file = tmp_path / "sample.py"
        test_file.write_text(SIMPLE_FUNCTION)

        funcs = self.parser.parse_file(str(test_file))
        assert len(funcs) == 1
        assert funcs[0].name == "greet"

    def test_parse_directory(self, tmp_path):
        """parse_directory aggregates functions from all .py files."""
        (tmp_path / "mod_a.py").write_text(SIMPLE_FUNCTION)
        (tmp_path / "mod_b.py").write_text(CLASS_WITH_METHODS)

        funcs = self.parser.parse_directory(str(tmp_path))
        names = {f.name for f in funcs}
        assert "greet" in names
        assert "add" in names
        assert "divide" in names

    def test_skip_test_files(self, tmp_path):
        """Test files are excluded by default."""
        (tmp_path / "test_foo.py").write_text(SIMPLE_FUNCTION)
        (tmp_path / "real.py").write_text(SIMPLE_FUNCTION)

        parser = ASTParser(skip_test_files=True)
        funcs = parser.parse_directory(str(tmp_path))
        files = {f.file_path for f in funcs}
        assert not any("test_foo" in p for p in files)
        assert any("real" in p for p in files)

    def test_large_file_skipped(self, tmp_path):
        """Files exceeding max_file_size_kb are skipped."""
        test_file = tmp_path / "big.py"
        test_file.write_text("x = 1\n" * 500)  # > 1 KB

        parser = ASTParser(max_file_size_kb=1)
        funcs = parser.parse_file(str(test_file))
        assert funcs == []

    def test_non_python_file_ignored(self, tmp_path):
        """Non-.py files return an empty list."""
        test_file = tmp_path / "notes.txt"
        test_file.write_text("hello world")

        funcs = self.parser.parse_file(str(test_file))
        assert funcs == []

    def test_syntax_error_returns_empty(self, tmp_path):
        """Files with syntax errors are silently skipped."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def foo(\n")  # Invalid syntax

        funcs = self.parser.parse_file(str(test_file))
        assert funcs == []

    def test_docstring_stripped_from_code(self):
        """The code field must not contain the docstring."""
        source = textwrap.dedent('''\
            def example():
                """This is a multi-line
                docstring that spans several lines.
                """
                x = 1
                return x
        ''')
        funcs = self._parse_source(source)
        assert len(funcs) == 1

        # The docstring text should be in .docstring but not in .source_code.
        assert "multi-line" in funcs[0].docstring
        assert "multi-line" not in funcs[0].source_code
        # The actual code should still be there.
        assert "x = 1" in funcs[0].source_code
        assert "return x" in funcs[0].source_code


# ======================================================================
# Doc Extractor Tests
# ======================================================================

class TestDocExtractor:
    """Tests for the DocExtractor class."""

    def setup_method(self):
        self.extractor = DocExtractor()

    def test_google_style(self):
        """Google-style docstrings are parsed correctly."""
        docstring = textwrap.dedent("""\
            Add two numbers.

            Args:
                a: The first number.
                b: The second number.

            Returns:
                The sum of a and b.

            Raises:
                TypeError: If inputs are not numbers.
        """)
        info = self.extractor.extract(docstring)
        assert info.summary == "Add two numbers."
        assert "a" in info.params
        assert "b" in info.params
        assert "sum" in info.returns.lower()
        assert "TypeError" in info.raises

    def test_numpy_style(self):
        """NumPy-style docstrings are parsed correctly."""
        docstring = textwrap.dedent("""\
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
        info = self.extractor.extract(docstring)
        assert "scaling" in info.summary.lower()
        assert "data" in info.params
        assert "factor" in info.params

    def test_sphinx_style(self):
        """Sphinx/reST-style docstrings are parsed correctly."""
        docstring = textwrap.dedent("""\
            Connect to the server.

            :param host: The hostname.
            :param port: The port number.
            :returns: A connection object.
            :raises ConnectionError: If connection fails.
        """)
        info = self.extractor.extract(docstring)
        assert "host" in info.params
        assert "port" in info.params
        assert info.returns
        assert "ConnectionError" in info.raises

    def test_plain_style(self):
        """Plain docstrings are treated as a single summary."""
        docstring = "This is just a plain description."
        info = self.extractor.extract(docstring)
        assert info.summary == "This is just a plain description."
        assert info.params == {}

    def test_empty_docstring(self):
        """Empty strings produce an empty DocInfo."""
        info = self.extractor.extract("")
        assert info.summary == ""
        assert info.params == {}
        assert info.raw == ""

    def test_none_docstring(self):
        """None is handled gracefully."""
        info = self.extractor.extract(None)
        assert info.summary == ""

    def test_normalise_produces_flat_text(self):
        """normalise() combines structured parts into a flat string."""
        docstring = textwrap.dedent("""\
            Fetch a user.

            Args:
                user_id: The user ID.

            Returns:
                The user object.
        """)
        info = self.extractor.extract(docstring)
        normalised = self.extractor.normalise(info)

        assert "Fetch a user" in normalised
        assert "user_id" in normalised
        assert "Returns:" in normalised

    def test_normalise_empty_docinfo(self):
        """Normalising an empty DocInfo returns an empty string."""
        info = DocInfo()
        result = self.extractor.normalise(info)
        assert result == ""


# ======================================================================
# Formatter Tests
# ======================================================================

class TestModelInputFormatter:
    """Tests for the ModelInputFormatter class."""

    def _make_func_info(self, **kwargs) -> FunctionInfo:
        """Helper to create FunctionInfo with sensible defaults."""
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

    def test_basic_format_matches_model_schema(self):
        """Output must have exactly function_id, code, docstring."""
        func = self._make_func_info()
        formatter = ModelInputFormatter(base_path="/project")
        records = formatter.format([func])

        assert len(records) == 1
        rec = records[0]
        assert set(rec.keys()) == {"function_id", "code", "docstring"}
        assert rec["code"] == "def my_func(x):\n    return x"
        assert "useful" in rec["docstring"]

    def test_function_id_includes_file_class_name(self):
        """function_id encodes file path, class, and function name."""
        func = self._make_func_info(
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

    def test_skip_undocumented_by_default(self):
        """Functions without docstrings are excluded by default."""
        func = self._make_func_info(has_docstring=False, docstring="")
        formatter = ModelInputFormatter()
        records = formatter.format([func])
        assert len(records) == 0

    def test_include_undocumented_when_enabled(self):
        """Undocumented functions included when flag is set."""
        func = self._make_func_info(has_docstring=False, docstring="")
        formatter = ModelInputFormatter(include_undocumented=True)
        records = formatter.format([func])
        assert len(records) == 1
        assert records[0]["docstring"] == ""

    def test_raw_docstring_mode(self):
        """With normalise_docstring=False, raw text is used."""
        func = self._make_func_info(docstring="Raw\n  docstring\n  text.")
        formatter = ModelInputFormatter(normalise_docstring=False)
        records = formatter.format([func])
        assert records[0]["docstring"] == "Raw\n  docstring\n  text."

    def test_format_to_json_writes_file(self, tmp_path):
        """format_to_json writes valid JSON to disk."""
        func = self._make_func_info()
        formatter = ModelInputFormatter(base_path="/project")
        out_file = str(tmp_path / "output.json")

        json_str = formatter.format_to_json([func], output_path=out_file)

        assert os.path.exists(out_file)
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["function_id"]

    def test_counter_increments_per_file(self):
        """Multiple functions in the same file get sequential counters."""
        funcs = [
            self._make_func_info(name="func_a"),
            self._make_func_info(name="func_b"),
        ]
        formatter = ModelInputFormatter(base_path="/project")
        records = formatter.format(funcs)

        assert records[0]["function_id"].endswith("001")
        assert records[1]["function_id"].endswith("002")


# ======================================================================
# Integration: parse_codebase convenience function
# ======================================================================

class TestParseCodebase:
    """Integration tests for the ``parse_codebase`` function."""

    def test_single_file(self, tmp_path):
        """Parsing a single .py file returns the expected records."""
        src = tmp_path / "module.py"
        src.write_text(SIMPLE_FUNCTION)

        records = parse_codebase(str(src))
        assert len(records) == 1
        assert records[0]["function_id"]
        assert "greet" in records[0]["code"]
        assert "greeting" in records[0]["docstring"].lower()

    def test_directory_aggregation(self, tmp_path):
        """Parsing a directory aggregates functions from all files."""
        (tmp_path / "a.py").write_text(SIMPLE_FUNCTION)
        (tmp_path / "b.py").write_text(CLASS_WITH_METHODS)

        records = parse_codebase(str(tmp_path))
        all_code = " ".join(r["code"] for r in records)
        assert "greet" in all_code
        assert "add" in all_code

    def test_output_file_created(self, tmp_path):
        """Passing output_path writes a JSON file."""
        src = tmp_path / "module.py"
        src.write_text(SIMPLE_FUNCTION)
        out = str(tmp_path / "result.json")

        records = parse_codebase(str(src), output_path=out)
        assert os.path.exists(out)

        with open(out) as f:
            data = json.load(f)
        assert len(data) == len(records)

    def test_undocumented_excluded_by_default(self, tmp_path):
        """Functions without docstrings are excluded by default."""
        src = tmp_path / "module.py"
        src.write_text(NO_DOCSTRING_FUNC)

        records = parse_codebase(str(src))
        assert len(records) == 0

    def test_undocumented_included_when_requested(self, tmp_path):
        """Undocumented functions included with flag."""
        src = tmp_path / "module.py"
        src.write_text(NO_DOCSTRING_FUNC)

        records = parse_codebase(str(src), include_undocumented=True)
        assert len(records) == 1

    def test_json_schema_matches_codebert_input(self, tmp_path):
        """Every record matches the exact schema CodeBERT expects."""
        src = tmp_path / "module.py"
        src.write_text(CLASS_WITH_METHODS)

        records = parse_codebase(str(src))
        for rec in records:
            # Must have exactly these three keys.
            assert set(rec.keys()) == {"function_id", "code", "docstring"}
            # All values must be non-empty strings.
            assert isinstance(rec["function_id"], str) and rec["function_id"]
            assert isinstance(rec["code"], str) and rec["code"]
            assert isinstance(rec["docstring"], str) and rec["docstring"]
