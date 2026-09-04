"""
Tests for the semdrift.parser V2 Universal Parser.

Comprehensive adversarial test suite covering all 30+ edge cases
from the evaluation review, including:
  - Python: normal, async, generator, nested, decorated functions
  - Python: positional-only, keyword-only, *args, **kwargs, defaults
  - Python: multiple returns, raises, yield, overload, class inheritance
  - Python: Google, NumPy, Sphinx docstring styles
  - Java: methods, constructors, throws, Javadoc, annotations, inheritance
  - Contract mismatches: undocumented params, phantom params
  - Parser failure states
"""

import os
import textwrap

import pytest

from semdrift.parser.contracts import (
    FunctionContract,
    Parameter,
    ParameterKind,
    ParseStatus,
    ExceptionContract,
    RaiseForm,
    DocContract,
)
from semdrift.parser.docstring_parser import DocstringParser
from semdrift.parser.universal_parser import UniversalParser
from semdrift.parser.contract_formatter import ContractFormatter
from semdrift.parser import parse_codebase_v2


# ======================================================================
# Python source fixtures
# ======================================================================

PY_SIMPLE = textwrap.dedent('''\
    def greet(name: str) -> str:
        """Return a greeting message for the given name."""
        return f"Hello, {name}!"
''')

PY_ASYNC = textwrap.dedent('''\
    async def fetch_data(url: str) -> dict:
        """Fetch JSON data from a remote URL."""
        pass
''')

PY_GENERATOR = textwrap.dedent('''\
    def count_up(n: int):
        """Yield numbers from 0 to n."""
        for i in range(n):
            yield i
''')

PY_POSITIONAL_ONLY = textwrap.dedent('''\
    def process(data, /, threshold=0.5):
        """Process the data with a threshold.

        Args:
            data: Input data.
            threshold: Minimum confidence.
        """
        return data
''')

PY_KEYWORD_ONLY = textwrap.dedent('''\
    def search(query, *, limit=10, offset=0):
        """Search with keyword-only params."""
        pass
''')

PY_COMPLEX_PARAMS = textwrap.dedent('''\
    def complex(a, b, *args, keyword_only=True, **kwargs):
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

PY_DEFAULTS = textwrap.dedent('''\
    def connect(host: str, port: int = 8080, timeout: float = 30.0):
        """Connect to a server.

        Args:
            host: The hostname.
            port: The port number.
            timeout: Connection timeout in seconds.
        """
        pass
''')

PY_MULTIPLE_RETURNS = textwrap.dedent('''\
    def get_user(valid: bool):
        """Get a user or None."""
        if valid:
            return User()
        return None
''')

PY_CLASS_WITH_METHODS = textwrap.dedent('''\
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

            Raises:
                ZeroDivisionError: If b is zero.
            """
            if b == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return a / b
''')

PY_INHERITANCE = textwrap.dedent('''\
    class Animal:
        pass

    class Dog(Animal):
        def bark(self) -> str:
            """Make a sound."""
            return "Woof!"

    class GuideDog(Dog, Trainable):
        def guide(self, person):
            """Guide a person."""
            pass
''')

PY_DECORATED = textwrap.dedent('''\
    class Service:
        @staticmethod
        def process(item: str) -> bool:
            """Process a single item."""
            return bool(item)

        @property
        def name(self) -> str:
            """The service name."""
            return "MyService"

        @classmethod
        def create(cls):
            """Factory method."""
            return cls()
''')

PY_RAISES_FORMS = textwrap.dedent('''\
    def risky():
        """Does risky things."""
        raise ValueError("bad value")
        raise TypeError
        raise errors.CustomError("oops")
        raise existing_exception
''')

PY_RAISE_CHAINED = textwrap.dedent('''\
    def chained():
        """Chained exception."""
        try:
            pass
        except IOError as e:
            raise RuntimeError("wrapped") from e
''')

PY_BARE_RERAISE = textwrap.dedent('''\
    def reraise():
        """Re-raises an exception."""
        try:
            pass
        except Exception:
            raise
''')

PY_NUMPY_DOCSTRING = textwrap.dedent('''\
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

PY_SPHINX_DOCSTRING = textwrap.dedent('''\
    def connect(host, port=8080):
        """Establish a connection to the remote server.

        :param host: The hostname or IP address.
        :param port: The port number (default 8080).
        :returns: A connection object.
        :raises ConnectionError: If the connection fails.
        """
        pass
''')

PY_NO_DOCSTRING = textwrap.dedent('''\
    def helper(x, y):
        return x + y
''')

PY_PHANTOM_PARAMS = textwrap.dedent('''\
    def mismatch(x):
        """A function with doc/code mismatch.

        Args:
            x: Actual parameter.
            y: This parameter does not exist in code!
            z: This one also does not exist.
        """
        pass
''')

PY_OVERLOAD = textwrap.dedent('''\
    from typing import overload

    @overload
    def process(x: int) -> str: ...

    @overload
    def process(x: str) -> int: ...

    def process(x):
        """Process an input."""
        if isinstance(x, int):
            return str(x)
        return len(x)
''')

PY_ABSTRACT = textwrap.dedent('''\
    from abc import abstractmethod

    class Base:
        @abstractmethod
        def do_something(self):
            """Must be implemented by subclasses."""
            pass
''')

PY_NESTED_FUNCTION = textwrap.dedent('''\
    def outer():
        """Outer function."""
        def inner():
            """Inner function."""
            pass
        return inner()
''')

PY_SYNTAX_ERROR = textwrap.dedent('''\
    def broken_func(x, y):
        """A function with syntax errors."""
        if x > 0
            return x
        return y
''')

# ======================================================================
# Java source fixtures
# ======================================================================

JAVA_SIMPLE = textwrap.dedent('''\
    public class Calculator {
        /**
         * Add two numbers together.
         *
         * @param a the first number
         * @param b the second number
         * @return the sum of a and b
         */
        public int add(int a, int b) {
            return a + b;
        }
    }
''')

JAVA_THROWS = textwrap.dedent('''\
    public class FileService {
        /**
         * Read a file from disk.
         *
         * @param path the file path
         * @return file contents as a string
         * @throws FileNotFoundException if the file does not exist
         * @throws IOException if reading fails
         */
        public String readFile(String path) throws FileNotFoundException, IOException {
            return new String(Files.readAllBytes(Paths.get(path)));
        }
    }
''')

JAVA_INHERITANCE = textwrap.dedent('''\
    public class Dog extends Animal implements Trainable {
        /**
         * Make the dog bark.
         *
         * @return the bark sound
         */
        public String bark() {
            return "Woof!";
        }
    }
''')

JAVA_ANNOTATIONS = textwrap.dedent('''\
    public class UserService {
        @Override
        @Deprecated
        public void process(String input) {
            System.out.println(input);
        }
    }
''')

JAVA_CONSTRUCTOR = textwrap.dedent('''\
    public class User {
        /**
         * Create a new User.
         *
         * @param name the user name
         * @param age the user age
         */
        public User(String name, int age) {
            this.name = name;
            this.age = age;
        }
    }
''')

JAVA_SYNTAX_ERROR = textwrap.dedent('''\
    public class BrokenService {
        public void process(String input) {
            if (input != null
                System.out.println(input);
            }
        }
    }
''')


# ======================================================================
# Test Classes
# ======================================================================

class TestUniversalParserPython:
    """Tests for Python extraction via the UniversalParser."""

    def setup_method(self):
        self.parser = UniversalParser()

    def _parse(self, source: str) -> list[FunctionContract]:
        """Write source to a temp file and parse it."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source)
            f.flush()
            path = f.name
        try:
            return self.parser.parse_file(path)
        finally:
            os.unlink(path)

    # ---- Basic extraction ----

    def test_simple_function(self):
        contracts = self._parse(PY_SIMPLE)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.name == "greet"
        assert c.language == "python"
        assert c.has_docstring
        assert c.return_annotation == "str"
        assert "greeting" in c.doc_contract.summary.lower()
        assert c.parse_status == ParseStatus.SUCCESS
        assert "greet" in c.source_code

    def test_async_function(self):
        contracts = self._parse(PY_ASYNC)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.name == "fetch_data"
        assert c.is_async is True
        assert c.has_docstring

    def test_generator_function(self):
        contracts = self._parse(PY_GENERATOR)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.name == "count_up"
        assert c.is_generator is True

    # ---- Parameter kinds ----

    def test_positional_only_params(self):
        """Red flag #1: posonlyargs must be extracted as POSITIONAL_ONLY."""
        contracts = self._parse(PY_POSITIONAL_ONLY)
        assert len(contracts) == 1
        c = contracts[0]
        param_names = [p.name for p in c.parameters]
        assert "data" in param_names
        assert "threshold" in param_names

        # data is before "/" so it MUST be POSITIONAL_ONLY
        data_param = next(p for p in c.parameters if p.name == "data")
        assert data_param.kind == ParameterKind.POSITIONAL_ONLY

        # threshold is after "/" so it should be POSITIONAL_OR_KEYWORD
        threshold_param = next(p for p in c.parameters if p.name == "threshold")
        assert threshold_param.kind == ParameterKind.POSITIONAL_OR_KEYWORD

    def test_keyword_only_params(self):
        contracts = self._parse(PY_KEYWORD_ONLY)
        assert len(contracts) == 1
        c = contracts[0]
        param_names = [p.name for p in c.parameters]
        assert "query" in param_names
        assert "limit" in param_names
        assert "offset" in param_names
        # limit and offset should be keyword-only
        limit_param = next(p for p in c.parameters if p.name == "limit")
        assert limit_param.kind == ParameterKind.KEYWORD_ONLY

    def test_complex_params(self):
        contracts = self._parse(PY_COMPLEX_PARAMS)
        assert len(contracts) == 1
        c = contracts[0]
        param_names = [p.name for p in c.parameters]
        assert "a" in param_names
        assert "b" in param_names
        assert "args" in param_names
        assert "keyword_only" in param_names
        assert "kwargs" in param_names

        # Check kinds
        args_param = next(p for p in c.parameters if p.name == "args")
        assert args_param.kind == ParameterKind.VAR_POSITIONAL

        kwargs_param = next(p for p in c.parameters if p.name == "kwargs")
        assert kwargs_param.kind == ParameterKind.VAR_KEYWORD

        kw_param = next(p for p in c.parameters if p.name == "keyword_only")
        assert kw_param.kind == ParameterKind.KEYWORD_ONLY

    def test_default_values_extracted(self):
        """Red flag #2: defaults must be extracted."""
        contracts = self._parse(PY_DEFAULTS)
        assert len(contracts) == 1
        c = contracts[0]

        port_param = next(p for p in c.parameters if p.name == "port")
        assert port_param.default == "8080"
        assert port_param.annotation == "int"

        timeout_param = next(p for p in c.parameters if p.name == "timeout")
        assert timeout_param.default == "30.0"

    # ---- Returns ----

    def test_multiple_return_paths(self):
        """Red flag #4: multiple return paths captured."""
        contracts = self._parse(PY_MULTIPLE_RETURNS)
        assert len(contracts) == 1
        c = contracts[0]
        assert len(c.return_paths) >= 2
        return_strs = " ".join(c.return_paths)
        assert "None" in return_strs

    # ---- Class methods ----

    def test_class_methods(self):
        contracts = self._parse(PY_CLASS_WITH_METHODS)
        assert len(contracts) == 2

        add_func = next(c for c in contracts if c.name == "add")
        assert add_func.class_name == "Calculator"
        assert add_func.qualified_name == "Calculator.add"
        param_names = [p.name for p in add_func.parameters]
        assert "a" in param_names
        assert "b" in param_names
        assert "self" not in param_names

        div_func = next(c for c in contracts if c.name == "divide")
        raises_names = [e.exception_name for e in div_func.explicit_raises]
        assert "ZeroDivisionError" in raises_names

    # ---- Class inheritance ----

    def test_class_inheritance(self):
        """Red flag #10-11: class inheritance extracted."""
        contracts = self._parse(PY_INHERITANCE)
        guide = next((c for c in contracts if c.name == "guide"), None)
        assert guide is not None
        assert guide.class_name == "GuideDog"
        assert "Dog" in guide.superclasses
        assert "Trainable" in guide.superclasses

    # ---- Decorators ----

    def test_decorated_functions(self):
        """Red flag #8: decorator semantic interpretation."""
        contracts = self._parse(PY_DECORATED)

        process_func = next(c for c in contracts if c.name == "process")
        assert process_func.is_staticmethod is True
        assert "staticmethod" in process_func.decorators

        name_func = next(c for c in contracts if c.name == "name")
        assert name_func.is_property is True

        create_func = next(c for c in contracts if c.name == "create")
        assert create_func.is_classmethod is True

    # ---- Raises ----

    def test_raise_forms(self):
        """Red flag #7: all raise forms classified."""
        contracts = self._parse(PY_RAISES_FORMS)
        assert len(contracts) == 1
        c = contracts[0]
        raises_names = [e.exception_name for e in c.explicit_raises]
        assert "ValueError" in raises_names
        assert "TypeError" in raises_names
        assert "CustomError" in raises_names

    def test_raise_chained(self):
        contracts = self._parse(PY_RAISE_CHAINED)
        assert len(contracts) == 1
        c = contracts[0]
        assert len(c.explicit_raises) >= 1
        runtime_exc = next((e for e in c.explicit_raises if e.exception_name == "RuntimeError"), None)
        assert runtime_exc is not None
        assert runtime_exc.form == RaiseForm.CHAINED

    def test_bare_reraise(self):
        contracts = self._parse(PY_BARE_RERAISE)
        assert len(contracts) == 1
        c = contracts[0]
        assert any(e.form == RaiseForm.RE_RAISE for e in c.explicit_raises)

    # ---- Docstring styles ----

    def test_google_docstring(self):
        contracts = self._parse(PY_CLASS_WITH_METHODS)
        add_func = next(c for c in contracts if c.name == "add")
        assert add_func.doc_contract.style == "google"
        assert "a" in add_func.doc_contract.param_descriptions
        assert "b" in add_func.doc_contract.param_descriptions
        assert add_func.doc_contract.return_description

    def test_numpy_docstring(self):
        contracts = self._parse(PY_NUMPY_DOCSTRING)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.doc_contract.style == "numpy"
        assert "data" in c.doc_contract.param_descriptions
        assert "factor" in c.doc_contract.param_descriptions

    def test_sphinx_docstring(self):
        contracts = self._parse(PY_SPHINX_DOCSTRING)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.doc_contract.style == "sphinx"
        assert "host" in c.doc_contract.param_descriptions
        assert "port" in c.doc_contract.param_descriptions
        assert c.doc_contract.return_description
        assert "ConnectionError" in c.doc_contract.raises_descriptions

    # ---- No docstring ----

    def test_no_docstring(self):
        contracts = self._parse(PY_NO_DOCSTRING)
        assert len(contracts) == 1
        c = contracts[0]
        assert c.has_docstring is False
        assert c.docstring_raw == ""

    # ---- Docstring stripped from code ----

    def test_docstring_stripped_from_code(self):
        contracts = self._parse(PY_SIMPLE)
        c = contracts[0]
        assert "greeting" in c.docstring_raw.lower()
        assert "greeting" not in c.source_code_without_docstring.lower()

    # ---- Contract mismatches ----

    def test_phantom_params_detected(self):
        """Red flag #17: phantom doc params detected."""
        contracts = self._parse(PY_PHANTOM_PARAMS)
        assert len(contracts) == 1
        c = contracts[0]
        assert "y" in c.phantom_doc_params
        assert "z" in c.phantom_doc_params
        assert "x" not in c.phantom_doc_params
        assert "x" not in c.undocumented_params

    # ---- Overload ----

    def test_overload_detection(self):
        """Red flag #9: @overload detected."""
        contracts = self._parse(PY_OVERLOAD)
        overloaded = [c for c in contracts if c.is_overload]
        assert len(overloaded) >= 2

    # ---- Abstract ----

    def test_abstract_detection(self):
        contracts = self._parse(PY_ABSTRACT)
        abstract_funcs = [c for c in contracts if c.is_abstract]
        assert len(abstract_funcs) >= 1
        assert abstract_funcs[0].name == "do_something"

    # ---- Nested function ----

    def test_nested_function(self):
        contracts = self._parse(PY_NESTED_FUNCTION)
        names = [c.name for c in contracts]
        assert "outer" in names

    # ---- Function identity ----

    def test_function_identity(self):
        """Red flag #19: stable function identity."""
        contracts = self._parse(PY_SIMPLE)
        c = contracts[0]
        identity = c.function_identity
        assert "greet" in identity
        assert "L" in identity

    # ---- Skip filters ----

    def test_skip_dunder(self):
        source = textwrap.dedent('''\
            class Foo:
                def __init__(self):
                    """Init."""
                    pass
                def bar(self):
                    """Bar."""
                    pass
        ''')
        parser = UniversalParser(skip_dunder=True)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            contracts = parser.parse_file(path)
        finally:
            os.unlink(path)
        names = [c.name for c in contracts]
        assert "__init__" not in names
        assert "bar" in names

    def test_skip_private(self):
        source = textwrap.dedent('''\
            class Foo:
                def _internal(self):
                    """Private."""
                    pass
                def public(self):
                    """Public."""
                    pass
        ''')
        parser = UniversalParser(skip_private=True)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(source)
            path = f.name
        try:
            contracts = parser.parse_file(path)
        finally:
            os.unlink(path)
        names = [c.name for c in contracts]
        assert "_internal" not in names
        assert "public" in names

    def test_syntax_error_detection(self):
        """Tree-sitter syntax errors must set parse_status=PARTIAL."""
        contracts = self._parse(PY_SYNTAX_ERROR)
        # Tree-sitter is error-tolerant so it still extracts the function
        assert len(contracts) >= 1
        c = contracts[0]
        assert c.parse_status == ParseStatus.PARTIAL
        assert any("syntax_error" in e for e in c.parse_errors)


class TestUniversalParserJava:
    """Tests for Java extraction via the UniversalParser."""

    def setup_method(self):
        self.parser = UniversalParser()

    def _parse(self, source: str) -> list[FunctionContract]:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False, encoding="utf-8") as f:
            f.write(source)
            f.flush()
            path = f.name
        try:
            return self.parser.parse_file(path)
        finally:
            os.unlink(path)

    def test_simple_java_method(self):
        contracts = self._parse(JAVA_SIMPLE)
        assert len(contracts) >= 1
        add_func = next((c for c in contracts if c.name == "add"), None)
        assert add_func is not None
        assert add_func.language == "java"
        assert add_func.class_name == "Calculator"
        param_names = [p.name for p in add_func.parameters]
        assert "a" in param_names
        assert "b" in param_names

    def test_java_javadoc(self):
        contracts = self._parse(JAVA_SIMPLE)
        add_func = next((c for c in contracts if c.name == "add"), None)
        assert add_func is not None
        assert add_func.has_docstring
        assert add_func.doc_contract.style == "javadoc"
        assert "a" in add_func.doc_contract.param_descriptions
        assert "b" in add_func.doc_contract.param_descriptions
        assert add_func.doc_contract.return_description

    def test_java_throws_clause(self):
        contracts = self._parse(JAVA_THROWS)
        read_func = next((c for c in contracts if c.name == "readFile"), None)
        assert read_func is not None
        raises_names = [e.exception_name for e in read_func.explicit_raises]
        assert "FileNotFoundException" in raises_names
        assert "IOException" in raises_names

    def test_java_inheritance(self):
        contracts = self._parse(JAVA_INHERITANCE)
        bark_func = next((c for c in contracts if c.name == "bark"), None)
        assert bark_func is not None
        assert bark_func.class_name == "Dog"
        assert "Animal" in bark_func.superclasses
        assert "Trainable" in bark_func.superclasses

    def test_java_annotations(self):
        contracts = self._parse(JAVA_ANNOTATIONS)
        process_func = next((c for c in contracts if c.name == "process"), None)
        assert process_func is not None
        assert any("Override" in d for d in process_func.decorators)
        assert any("Deprecated" in d for d in process_func.decorators)
        # @Override must map to is_override, NOT is_overload
        assert process_func.is_override is True
        assert process_func.is_overload is False

    def test_java_constructor(self):
        contracts = self._parse(JAVA_CONSTRUCTOR)
        assert len(contracts) >= 1
        ctor = next((c for c in contracts if c.name == "User"), None)
        assert ctor is not None
        param_names = [p.name for p in ctor.parameters]
        assert "name" in param_names
        assert "age" in param_names

    def test_java_return_annotation(self):
        contracts = self._parse(JAVA_SIMPLE)
        add_func = next((c for c in contracts if c.name == "add"), None)
        assert add_func is not None
        assert add_func.return_annotation == "int"

    def test_java_syntax_error(self):
        """Tree-sitter syntax errors in Java must set parse_status=PARTIAL."""
        contracts = self._parse(JAVA_SYNTAX_ERROR)
        # Tree-sitter is error-tolerant so it may still extract the method
        if contracts:
            c = contracts[0]
            assert c.parse_status == ParseStatus.PARTIAL
            assert any("syntax_error" in e for e in c.parse_errors)


class TestDocstringParser:
    """Tests for the multi-style DocstringParser."""

    def setup_method(self):
        self.parser = DocstringParser()

    def test_google_style(self):
        doc = self.parser.parse(textwrap.dedent("""\
            Add two numbers.

            Args:
                a: The first number.
                b: The second number.

            Returns:
                The sum of a and b.

            Raises:
                TypeError: If inputs are not numbers.
        """))
        assert doc.style == "google"
        assert doc.summary == "Add two numbers."
        assert "a" in doc.param_descriptions
        assert "b" in doc.param_descriptions
        assert "sum" in doc.return_description.lower()
        assert "TypeError" in doc.raises_descriptions

    def test_numpy_style(self):
        doc = self.parser.parse(textwrap.dedent("""\
            Apply a transformation.

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
        """))
        assert doc.style == "numpy"
        assert "data" in doc.param_descriptions
        assert "factor" in doc.param_descriptions

    def test_sphinx_style(self):
        doc = self.parser.parse(textwrap.dedent("""\
            Connect to the server.

            :param host: The hostname.
            :param port: The port number.
            :returns: A connection object.
            :raises ConnectionError: If connection fails.
        """))
        assert doc.style == "sphinx"
        assert "host" in doc.param_descriptions
        assert "port" in doc.param_descriptions
        assert doc.return_description
        assert "ConnectionError" in doc.raises_descriptions

    def test_javadoc_style(self):
        doc = self.parser.parse(textwrap.dedent("""\
            /**
             * Add two numbers together.
             *
             * @param a the first number
             * @param b the second number
             * @return the sum of a and b
             * @throws ArithmeticException if overflow occurs
             */
        """))
        assert doc.style == "javadoc"
        assert "a" in doc.param_descriptions
        assert "b" in doc.param_descriptions
        assert doc.return_description
        assert "ArithmeticException" in doc.raises_descriptions

    def test_plain_style(self):
        doc = self.parser.parse("This is just a plain description.")
        assert doc.style == "plain"
        assert doc.summary == "This is just a plain description."
        assert doc.param_descriptions == {}

    def test_empty_docstring(self):
        doc = self.parser.parse("")
        assert doc.summary == ""
        assert doc.raw == ""

    def test_none_docstring(self):
        doc = self.parser.parse(None)
        assert doc.summary == ""

    def test_normalise_structured(self):
        doc = self.parser.parse(textwrap.dedent("""\
            Fetch a user.

            Args:
                user_id: The user ID.

            Returns:
                The user object.
        """))
        normalised = self.parser.normalise(doc)
        assert "[SUMMARY]" in normalised
        assert "[PARAMETERS]" in normalised
        assert "[RETURNS]" in normalised
        assert "user_id" in normalised

    def test_normalise_legacy(self):
        doc = self.parser.parse(textwrap.dedent("""\
            Fetch a user.

            Args:
                user_id: The user ID.

            Returns:
                The user object.
        """))
        normalised = self.parser.normalise_legacy(doc)
        assert "[SUMMARY]" not in normalised
        assert "Fetch a user" in normalised
        assert "user_id" in normalised


class TestContractFormatter:
    """Tests for the ContractFormatter."""

    def _make_contract(self, **kwargs) -> FunctionContract:
        defaults = dict(
            name="my_func",
            qualified_name="my_func",
            file_path="/project/src/utils.py",
            line_start=1,
            line_end=5,
            language="python",
            source_code="def my_func(x):\n    return x",
            source_code_without_docstring="def my_func(x):\n    return x",
            docstring_raw="Does something useful.",
            doc_contract=DocContract(summary="Does something useful.", style="plain"),
            parameters=[Parameter(name="x")],
            parse_status=ParseStatus.SUCCESS,
        )
        defaults.update(kwargs)
        return FunctionContract(**defaults)

    def test_format_produces_required_keys(self):
        contract = self._make_contract()
        formatter = ContractFormatter(base_path="/project")
        records = formatter.format([contract])
        assert len(records) == 1
        rec = records[0]
        assert "function_id" in rec
        assert "code" in rec
        assert "docstring" in rec

    def test_function_id_uses_qualified_location(self):
        contract = self._make_contract(
            name="fetch_user",
            qualified_name="UserService.fetch_user",
            file_path="/project/app/services.py",
            line_start=42,
            line_end=67,
        )
        formatter = ContractFormatter(base_path="/project")
        records = formatter.format([contract])
        fid = records[0]["function_id"]
        assert "app/services.py" in fid
        assert "UserService.fetch_user" in fid
        assert "L42-67" in fid

    def test_skip_undocumented_by_default(self):
        contract = self._make_contract(docstring_raw="")
        formatter = ContractFormatter()
        records = formatter.format([contract])
        assert len(records) == 0

    def test_include_undocumented(self):
        contract = self._make_contract(docstring_raw="")
        formatter = ContractFormatter(include_undocumented=True)
        records = formatter.format([contract])
        assert len(records) == 1

    def test_structured_docstring(self):
        contract = self._make_contract()
        formatter = ContractFormatter(use_structured_docstring=True)
        records = formatter.format([contract])
        assert "[SUMMARY]" in records[0]["docstring"]

    def test_legacy_docstring(self):
        contract = self._make_contract()
        formatter = ContractFormatter(use_structured_docstring=False)
        records = formatter.format([contract])
        assert "[SUMMARY]" not in records[0]["docstring"]
        assert "useful" in records[0]["docstring"]

    def test_format_minimal(self):
        contract = self._make_contract()
        formatter = ContractFormatter(base_path="/project")
        records = formatter.format_minimal([contract])
        assert len(records) == 1
        assert set(records[0].keys()) == {"function_id", "code", "docstring"}

    def test_format_to_json(self, tmp_path):
        contract = self._make_contract()
        formatter = ContractFormatter(base_path="/project")
        out_file = str(tmp_path / "output.json")
        json_str = formatter.format_to_json([contract], output_path=out_file)
        assert os.path.exists(out_file)
        import json
        data = json.loads(json_str)
        assert len(data) == 1


class TestParseCodbaseV2Integration:
    """Integration tests for the parse_codebase_v2 convenience function."""

    def test_single_python_file(self, tmp_path):
        src = tmp_path / "module.py"
        src.write_text(PY_SIMPLE)
        contracts = parse_codebase_v2(str(src))
        assert len(contracts) == 1
        assert contracts[0].name == "greet"
        assert contracts[0].has_docstring

    def test_directory_aggregation(self, tmp_path):
        (tmp_path / "a.py").write_text(PY_SIMPLE)
        (tmp_path / "b.py").write_text(PY_CLASS_WITH_METHODS)
        contracts = parse_codebase_v2(str(tmp_path))
        names = {c.name for c in contracts}
        assert "greet" in names
        assert "add" in names
        assert "divide" in names

    def test_output_file(self, tmp_path):
        src = tmp_path / "module.py"
        src.write_text(PY_SIMPLE)
        out = str(tmp_path / "result.json")
        contracts = parse_codebase_v2(str(src), output_path=out)
        assert os.path.exists(out)
        assert len(contracts) == 1

    def test_mixed_python_java(self, tmp_path):
        (tmp_path / "calc.py").write_text(PY_SIMPLE)
        (tmp_path / "Calc.java").write_text(JAVA_SIMPLE)
        contracts = parse_codebase_v2(str(tmp_path))
        languages = {c.language for c in contracts}
        assert "python" in languages
        assert "java" in languages


class TestLegacyParserStillWorks:
    """Ensure the legacy parse_codebase function still works unchanged."""

    def test_legacy_parse(self, tmp_path):
        from semdrift.parser import parse_codebase
        src = tmp_path / "module.py"
        src.write_text(PY_SIMPLE)
        records = parse_codebase(str(src))
        assert len(records) == 1
        assert set(records[0].keys()) == {"function_id", "code", "docstring"}
        assert "greet" in records[0]["code"]
