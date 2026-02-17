"""Tests for language-specific symbol extractors."""

from agent_framework.indexing.extractors import get_extractor_for_language
from agent_framework.indexing.extractors.go_extractor import GoExtractor
from agent_framework.indexing.extractors.js_extractor import JSExtractor
from agent_framework.indexing.extractors.python_extractor import PythonExtractor
from agent_framework.indexing.extractors.ruby_extractor import RubyExtractor
from agent_framework.indexing.models import SymbolKind


# ---------------------------------------------------------------------------
# Python extractor
# ---------------------------------------------------------------------------
class TestPythonExtractor:
    def setup_method(self):
        self.ext = PythonExtractor()

    def test_class_with_methods_and_docstrings(self):
        source = '''\
class UserService(BaseService):
    """Handles user operations."""

    def __init__(self, db: Database, timeout: int = 30):
        self.db = db

    def get_user(self, user_id: str) -> User:
        """Fetch a user by ID."""
        return self.db.find(user_id)

    def _validate(self, data):
        pass

    def update_user(self, user_id: str, **kwargs) -> None:
        pass
'''
        symbols = self.ext.extract_symbols("service.py", source)
        names = [s.name for s in symbols]

        assert "UserService" in names
        assert "__init__" in names
        assert "get_user" in names
        assert "update_user" in names
        # Private method filtered out
        assert "_validate" not in names

        cls = next(s for s in symbols if s.name == "UserService")
        assert cls.kind == SymbolKind.CLASS
        assert cls.signature == "class UserService(BaseService)"
        assert cls.docstring == "Handles user operations."

        init = next(s for s in symbols if s.name == "__init__")
        assert init.kind == SymbolKind.METHOD
        assert init.parent == "UserService"
        assert "db: Database" in init.signature
        assert "timeout: int = 30" in init.signature
        # self should be stripped
        assert "self" not in init.signature

        get_user = next(s for s in symbols if s.name == "get_user")
        assert "-> User" in get_user.signature
        assert get_user.docstring == "Fetch a user by ID."

    def test_top_level_functions_with_async(self):
        source = '''\
def process_batch(items: list[str]) -> int:
    """Process items in batch."""
    return len(items)

async def fetch_data(url: str) -> bytes:
    pass

def _helper():
    pass
'''
        symbols = self.ext.extract_symbols("utils.py", source)
        names = [s.name for s in symbols]

        assert "process_batch" in names
        assert "fetch_data" in names
        assert "_helper" not in names

        fetch = next(s for s in symbols if s.name == "fetch_data")
        assert fetch.kind == SymbolKind.FUNCTION
        assert fetch.signature.startswith("async def")

    def test_nested_classes(self):
        source = '''\
class Outer:
    class Inner:
        def method(self):
            pass
'''
        symbols = self.ext.extract_symbols("nested.py", source)
        inner = next(s for s in symbols if s.name == "Inner")
        assert inner.parent == "Outer"
        assert inner.kind == SymbolKind.CLASS

        method = next(s for s in symbols if s.name == "method")
        assert method.parent == "Inner"

    def test_syntax_error_returns_empty(self):
        source = "def broken(:\n    pass"
        assert self.ext.extract_symbols("bad.py", source) == []

    def test_empty_source_returns_empty(self):
        assert self.ext.extract_symbols("empty.py", "") == []
        assert self.ext.extract_symbols("blank.py", "   \n  \n") == []

    def test_decorated_functions(self):
        source = '''\
@app.route("/users")
def list_users():
    pass

@staticmethod
def create():
    pass
'''
        symbols = self.ext.extract_symbols("routes.py", source)
        names = [s.name for s in symbols]
        assert "list_users" in names
        assert "create" in names


# ---------------------------------------------------------------------------
# Go extractor
# ---------------------------------------------------------------------------
class TestGoExtractor:
    def setup_method(self):
        self.ext = GoExtractor()

    def test_struct_with_methods_and_docstring(self):
        source = '''\
// Handler manages HTTP request handling.
// It delegates to services.
type Handler struct {
\tlogger *Logger
}

// Get returns a single resource by ID.
func (h *Handler) Get(ctx context.Context, id string) (*Resource, error) {
\treturn nil, nil
}

// unexportedHelper does something.
func (h *Handler) unexportedHelper() {
}
'''
        symbols = self.ext.extract_symbols("handler.go", source)
        names = [s.name for s in symbols]

        assert "Handler" in names
        assert "Get" in names
        assert "unexportedHelper" not in names

        handler = next(s for s in symbols if s.name == "Handler")
        assert handler.kind == SymbolKind.STRUCT
        assert handler.signature == "type Handler struct"
        assert "manages HTTP request handling" in handler.docstring

        get = next(s for s in symbols if s.name == "Get")
        assert get.kind == SymbolKind.METHOD
        assert get.parent == "Handler"
        assert "Get" in get.signature

    def test_value_receiver_methods(self):
        source = '''\
type Config struct {}

func (c Config) Validate() error {
\treturn nil
}
'''
        symbols = self.ext.extract_symbols("config.go", source)
        validate = next(s for s in symbols if s.name == "Validate")
        assert validate.parent == "Config"
        assert validate.kind == SymbolKind.METHOD

    def test_interface_extraction(self):
        source = '''\
type Repository interface {
\tFind(id string) (*Entity, error)
\tSave(entity *Entity) error
}
'''
        symbols = self.ext.extract_symbols("repo.go", source)
        repo = next(s for s in symbols if s.name == "Repository")
        assert repo.kind == SymbolKind.INTERFACE
        assert repo.signature == "type Repository interface"

    def test_multiline_function_signature(self):
        source = '''\
// New creates a new handler with all dependencies.
func New(
\tlogger *Logger,
\tdb *Database,
\tconfig *Config,
) *Handler {
\treturn &Handler{}
}
'''
        symbols = self.ext.extract_symbols("handler.go", source)
        new = next(s for s in symbols if s.name == "New")
        assert new.kind == SymbolKind.FUNCTION
        assert "New(" in new.signature
        assert "logger *Logger" in new.signature
        assert new.docstring == "New creates a new handler with all dependencies."

    def test_unexported_filtered(self):
        source = '''\
func Exported() {}
func unexported() {}
func AnotherExported(x int) string { return "" }
'''
        symbols = self.ext.extract_symbols("funcs.go", source)
        names = [s.name for s in symbols]
        assert "Exported" in names
        assert "AnotherExported" in names
        assert "unexported" not in names

    def test_multiple_return_values(self):
        source = '''\
func ParseHeaders(raw string) (map[string][]string, error) {
\treturn nil, nil
}
'''
        symbols = self.ext.extract_symbols("parse.go", source)
        parse = next(s for s in symbols if s.name == "ParseHeaders")
        assert "map[string][]string" in parse.signature
        assert "error" in parse.signature


# ---------------------------------------------------------------------------
# Ruby extractor
# ---------------------------------------------------------------------------
class TestRubyExtractor:
    def setup_method(self):
        self.ext = RubyExtractor()

    def test_class_with_inheritance_and_methods(self):
        source = '''\
# Service for managing users.
class UserService < BaseService
  def initialize(db)
    @db = db
  end

  def find_user(id)
    @db.find(id)
  end

  private

  def validate(data)
    true
  end
end
'''
        symbols = self.ext.extract_symbols("user_service.rb", source)
        names = [s.name for s in symbols]

        assert "UserService" in names
        assert "initialize" in names
        assert "find_user" in names
        # Private method filtered
        assert "validate" not in names

        cls = next(s for s in symbols if s.name == "UserService")
        assert cls.kind == SymbolKind.CLASS
        assert cls.signature == "class UserService < BaseService"
        assert cls.docstring == "Service for managing users."

    def test_nested_modules(self):
        source = '''\
module Api
  module V2
    class Handler
      def process
      end
    end
  end
end
'''
        symbols = self.ext.extract_symbols("handler.rb", source)

        v2 = next(s for s in symbols if s.name == "V2")
        assert v2.kind == SymbolKind.MODULE
        assert v2.parent == "Api"

        handler = next(s for s in symbols if s.name == "Handler")
        assert handler.kind == SymbolKind.CLASS
        assert handler.parent == "V2"

        process = next(s for s in symbols if s.name == "process")
        assert process.parent == "Handler"

    def test_class_self_methods(self):
        source = '''\
class Config
  class << self
    def load(path)
      new
    end

    def default
      load("config.yml")
    end
  end
end
'''
        symbols = self.ext.extract_symbols("config.rb", source)
        names = [s.name for s in symbols]
        assert "Config" in names
        assert "load" in names
        assert "default" in names

    def test_comment_extraction(self):
        source = '''\
# Performs the main computation.
# Returns the result hash.
def compute(input)
  {}
end
'''
        symbols = self.ext.extract_symbols("compute.rb", source)
        compute = next(s for s in symbols if s.name == "compute")
        assert compute.docstring == "Performs the main computation. Returns the result hash."

    def test_empty_source(self):
        assert self.ext.extract_symbols("empty.rb", "") == []
        assert self.ext.extract_symbols("blank.rb", "  \n\n  ") == []


# ---------------------------------------------------------------------------
# JS/TS extractor
# ---------------------------------------------------------------------------
class TestJSExtractor:
    def setup_method(self):
        self.ext = JSExtractor()

    def test_class_with_methods_and_jsdoc(self):
        source = '''\
/**
 * Manages API connections.
 * @param {string} baseUrl
 */
export class ApiClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }

  /** Fetch a resource by ID. */
  async get(id) {
    return fetch(`${this.baseUrl}/${id}`);
  }

  #privateMethod() {
    return null;
  }
}
'''
        symbols = self.ext.extract_symbols("client.ts", source)
        names = [s.name for s in symbols]

        assert "ApiClient" in names
        assert "constructor" in names
        assert "get" in names
        assert "#privateMethod" not in names

        cls = next(s for s in symbols if s.name == "ApiClient")
        assert cls.kind == SymbolKind.CLASS
        assert cls.docstring == "Manages API connections."

        get = next(s for s in symbols if s.name == "get")
        assert get.kind == SymbolKind.METHOD
        assert get.parent == "ApiClient"

    def test_functions_and_arrow_functions(self):
        source = '''\
export function processData(items) {
  return items.map(x => x * 2);
}

export const fetchUser = async (id) => {
  return db.find(id);
};

function helperFn(x, y) {
  return x + y;
}
'''
        symbols = self.ext.extract_symbols("utils.js", source)
        names = [s.name for s in symbols]

        assert "processData" in names
        assert "fetchUser" in names
        assert "helperFn" in names

        process = next(s for s in symbols if s.name == "processData")
        assert process.kind == SymbolKind.FUNCTION

        fetch_user = next(s for s in symbols if s.name == "fetchUser")
        assert fetch_user.kind == SymbolKind.FUNCTION

    def test_typescript_interface(self):
        source = '''\
/**
 * Represents a user in the system.
 */
export interface User extends BaseEntity {
  name: string;
  email: string;
}
'''
        symbols = self.ext.extract_symbols("types.ts", source)
        user = next(s for s in symbols if s.name == "User")
        assert user.kind == SymbolKind.INTERFACE
        assert "extends BaseEntity" in user.signature
        assert user.docstring == "Represents a user in the system."

    def test_typescript_type_alias(self):
        source = '''\
export type RequestConfig<T> = {
  url: string;
  body: T;
};
'''
        symbols = self.ext.extract_symbols("types.ts", source)
        config = next(s for s in symbols if s.name == "RequestConfig")
        assert config.kind == SymbolKind.CLASS
        assert config.signature == "type RequestConfig"

    def test_generic_async_function(self):
        source = '''\
export async function fetchJson<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  return resp.json();
}
'''
        symbols = self.ext.extract_symbols("http.ts", source)
        fetch_json = next(s for s in symbols if s.name == "fetchJson")
        assert fetch_json.kind == SymbolKind.FUNCTION

    def test_empty_source(self):
        assert self.ext.extract_symbols("empty.js", "") == []
        assert self.ext.extract_symbols("blank.ts", "  \n  \n") == []


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------
class TestExtractorRegistry:
    def test_all_languages_return_correct_type(self):
        assert isinstance(get_extractor_for_language("python"), PythonExtractor)
        assert isinstance(get_extractor_for_language("go"), GoExtractor)
        assert isinstance(get_extractor_for_language("ruby"), RubyExtractor)
        assert isinstance(get_extractor_for_language("javascript"), JSExtractor)
        assert isinstance(get_extractor_for_language("typescript"), JSExtractor)

    def test_case_insensitive(self):
        assert isinstance(get_extractor_for_language("Python"), PythonExtractor)
        assert isinstance(get_extractor_for_language("GO"), GoExtractor)
        assert isinstance(get_extractor_for_language("Ruby"), RubyExtractor)

    def test_unknown_language_returns_none(self):
        assert get_extractor_for_language("cobol") is None
        assert get_extractor_for_language("") is None
