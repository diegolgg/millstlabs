"""Static and runtime restrictions on LLM-written bot code (spec section 4; hardened in fix round 1).

The sandbox guards against bugs and against a model that stumbles on a way to read hidden state (its own cards). It is
not a security boundary against a determined adversary; the process isolation in evaluate/pool.py is the backstop.

Layers, all applied to every non-anchor bot:

1. **Static checks (AST)**: imports only from `ALLOWED_MODULES`; banned builtin names; no dunder names or dunder
   attributes (except a short allowlist); no frame/introspection attributes (`gi_frame`, `f_back`, `mro`, ...); no
   string or bytes constant containing `__` (docstrings and `"__main__"` excepted), which stops string-built dunder
   access; `str.format`/`format_map` only on a literal template without attribute fields; no stores into imported
   modules or classes (monkeypatching); no private names in `from x import _y` or in `case C(_attr=...)`.
2. **Token scan**: an artifact whose source mentions `sys`, `_os`, `_getframe`, `__subclasses__`, `__globals__` or
   `__builtins__` is invalid.
3. **Proxy modules**: `import random` yields a read-only proxy exposing only the public, non-module attributes of the
   real module (submodules of the same package are proxied, e.g. `collections.abc`). Known eval/introspection paths
   are removed or replaced (`typing.get_type_hints`, `functools.singledispatch`, `functools.update_wrapper`/`wraps`
   with custom attribute lists, `dataclasses.dataclass` with non-identifier field names, `random.SystemRandom`).
4. **Guarded builtins**: `getattr`/`hasattr`/`setattr` reject dunder, frame and format names, and private (`_x`)
   names unless the attribute belongs to a class the bot defined; `setattr` only writes to bot-defined objects.
5. **Private-attribute guard**: every `obj._x` in the source is rewritten to check at run time that `_x` resolves on a
   bot-defined class (so `self._cache` works, `typing.ForwardRef(...)._evaluate` does not).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import functools
import importlib
import keyword
import re
import string
import types
from typing import Any

from .interface import ALLOWED_IMPORTS

BOT_MODULE = "bot_module"  # `__name__` of the namespace bot code runs in; classes it defines carry this `__module__`
GUARD_NAME = "__sandbox_private__"  # runtime check inserted before private attribute access
ALLOWED_MODULES = frozenset(ALLOWED_IMPORTS) | {"collections.abc"}

BANNED_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "breakpoint", "dir",
    "memoryview", "__builtins__", "exit", "quit", "help", "delattr", "__loader__", "__spec__",
    "BaseException", "SystemExit", "KeyboardInterrupt", "GeneratorExit",
}
ALLOWED_DUNDER_ATTRS = {"__init__", "__name__", "__class__", "__eq__", "__hash__", "__lt__", "__repr__", "__str__",
                        "__len__"}
ALLOWED_DUNDER_NAMES = {"__name__", "__slots__"}
# frame and code introspection (reachable without dunders through generators and coroutines), plus `mro`
FRAME_ATTRS = {"gi_frame", "gi_code", "gi_yieldfrom", "cr_frame", "cr_code", "cr_await", "cr_origin", "ag_frame",
               "ag_code", "ag_await", "f_back", "f_builtins", "f_code", "f_globals", "f_locals", "f_trace",
               "tb_frame", "tb_next"}
BANNED_ATTRS = FRAME_ATTRS | {"mro"}
FORMAT_ATTRS = {"format", "format_map"}  # str.format walks attribute fields ("{0.gi_frame}") of its arguments
SUSPICIOUS_TOKENS = ("sys", "_os", "_getframe", "__subclasses__", "__globals__", "__builtins__")
_SUSPICIOUS_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(t) for t in SUSPICIOUS_TOKENS)
                            + r")(?![A-Za-z0-9_])")


class SourceRejected(ValueError):
    pass


class SandboxViolation(RuntimeError):
    """Raised inside bot code when it reaches for something the sandbox does not allow (deliberately not an
    AttributeError, so `getattr(x, name, default)` and `except AttributeError` cannot silently swallow it)."""


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


# ----------------------------------------------------------------------------------------------- static checks
def suspicious_tokens(code: str) -> list[str]:
    """Post-hoc scan of a produced artifact: tokens whose presence marks it invalid."""
    return sorted({m.group(1) for m in _SUSPICIOUS_RE.finditer(code)})


def _template_ok(s: str, depth: int = 0) -> bool:
    """A literal str.format template is allowed if no replacement field walks attributes ("{0.attr}")."""
    if depth > 2:
        return False
    try:
        parsed = list(string.Formatter().parse(s))
    except ValueError:
        return False
    for _lit, field, spec, _conv in parsed:
        if field is not None and ("." in field or "_" in field.split("[", 1)[0][:1]):
            return False
        if spec and "{" in spec and not _template_ok(spec, depth + 1):
            return False
    return True


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _imported_names(tree: ast.AST) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {a.asname or a.name for a in n.names}
    return out


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _static_problems(tree: ast.AST) -> list[str]:
    problems: list[str] = []
    docs = _docstring_nodes(tree)
    imported = _imported_names(tree)

    def bad(msg: str, node: ast.AST) -> None:
        problems.append(f"{msg} (line {getattr(node, 'lineno', '?')})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_MODULES:
                    bad(f"import of {a.name!r} not allowed", node)
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES:
                bad(f"import from {node.module!r} not allowed", node)
            for a in node.names:
                if a.name.startswith("_"):
                    bad(f"import of private name {a.name!r} not allowed", node)
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES:
                bad(f"name {node.id!r} not allowed", node)
            elif _is_dunder(node.id) and node.id not in ALLOWED_DUNDER_NAMES:
                bad(f"name {node.id!r} not allowed", node)
        elif isinstance(node, ast.Attribute):
            a = node.attr
            if a.startswith("__") and a not in ALLOWED_DUNDER_ATTRS:
                bad(f"attribute {a!r} not allowed", node)
            elif a in BANNED_ATTRS:
                bad(f"attribute {a!r} not allowed (introspection)", node)
            elif a in FORMAT_ATTRS and not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                                            and _template_ok(node.value.value)):
                bad(f"`.{a}` only on a literal template without attribute fields (use an f-string)", node)
            elif isinstance(node.ctx, (ast.Store, ast.Del)) and _root_name(node) in imported:
                bad(f"assignment into imported module or class {_root_name(node)!r} not allowed", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and id(node) not in docs:
            s = node.value if isinstance(node.value, str) else node.value.decode("latin-1")
            if "__" in s and s != "__main__":
                bad("string constants containing '__' are not allowed", node)
        elif isinstance(node, ast.MatchClass):
            for a in node.kwd_attrs:
                if a.startswith("_") or a in BANNED_ATTRS or a in FORMAT_ATTRS:
                    bad(f"class pattern attribute {a!r} not allowed", node)
    return problems


def check_source(code: str, max_lines: int = 400) -> list[str]:
    """Return a list of violations (empty means the source is admissible)."""
    problems: list[str] = []
    n_lines = len(code.splitlines())
    if n_lines > max_lines:
        problems.append(f"too long: {n_lines} lines > {max_lines}")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return problems + [f"syntax error: {e.msg} (line {e.lineno})"]
    problems += _static_problems(tree)
    toks = suspicious_tokens(code)
    if toks:
        problems.append(f"suspicious tokens {toks} (the artifact is invalid)")
    if not any(isinstance(n, ast.ClassDef) and n.name == "Bot" for n in tree.body):
        problems.append("no top-level `class Bot`")
    return problems


def complexity(code: str) -> int:
    """Logged complexity measure: 1 + number of branch points (if/for/while/try/boolop/comprehension/lambda)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return -1
    kinds = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.comprehension, ast.Lambda, ast.IfExp)
    return 1 + sum(isinstance(n, kinds) for n in ast.walk(tree))


# ----------------------------------------------------------------------------------------------- ownership
_tdict = type.__dict__["__dict__"].__get__  # a class's own namespace, bypassing any metaclass override
_tmro = type.__dict__["__mro__"].__get__
_PRIV_CACHE: dict[tuple, bool] = {}


def _bot_class(c: type) -> bool:
    return _tdict(c).get("__module__") == BOT_MODULE


def _bot_owned(obj: Any) -> bool:
    """True for classes defined by bot code and for instances of them (namedtuples and dataclasses the bot creates
    count: the standard library stamps them with the caller's module)."""
    cls = obj if issubclass(type(obj), type) else type(obj)
    return _bot_class(cls)


def _private_ok(obj: Any, name: str) -> bool:
    """May bot code read or write `obj.<name>` for a single-underscore name? Only if no class outside the bot's own
    code defines `name` anywhere in the lookup chain, and the object itself is bot-owned."""
    if name.startswith("__"):
        return False
    t = type(obj)
    if t is super:  # super()._helper(): resolve along the MRO of the class super() searches
        cls, is_cls = (obj.__self_class__ or obj.__thisclass__), False
    elif issubclass(t, type):
        cls, is_cls = obj, True
    else:
        cls, is_cls = t, False
    key = (cls, name, is_cls)
    hit = _PRIV_CACHE.get(key)
    if hit is not None:
        return hit
    chains = [_tmro(cls)] + ([_tmro(type(cls))] if is_cls else [])
    ok = _bot_class(cls) and not any(name in _tdict(c) for chain in chains for c in chain if not _bot_class(c))
    if len(_PRIV_CACHE) > 4096:
        _PRIV_CACHE.clear()
    _PRIV_CACHE[key] = ok
    return ok


def _guard_private(obj: Any, name: str) -> Any:
    if _private_ok(obj, name):
        return obj
    raise SandboxViolation(f"access to private attribute {name!r} of a {type(obj).__name__} object is not allowed")


def _check_attr_name(obj: Any, name: Any, verb: str) -> None:
    if type(name) is not str:
        return  # the real builtin raises TypeError
    if name.startswith("__") or name in BANNED_ATTRS or name in FORMAT_ATTRS:
        raise SandboxViolation(f"{verb}(..., {name!r}) is not allowed in bot code")
    if name.startswith("_") and not _private_ok(obj, name):
        raise SandboxViolation(f"{verb}(..., {name!r}) on a {type(obj).__name__} object is not allowed")


def _safe_getattr(obj, name, *default):
    _check_attr_name(obj, name, "getattr")
    return builtins.getattr(obj, name, *default)


def _safe_hasattr(obj, name):
    _check_attr_name(obj, name, "hasattr")
    return builtins.hasattr(obj, name)


def _safe_setattr(obj, name, value):
    _check_attr_name(obj, name, "setattr")
    if not _bot_owned(obj):
        raise SandboxViolation(f"setattr on a {type(obj).__name__} object is not allowed (only on your own objects)")
    return builtins.setattr(obj, name, value)


# ----------------------------------------------------------------------------------------------- proxy modules
class ModuleProxy:
    """Read-only stand-in for a whitelisted module: public, non-module attributes only."""

    __slots__ = ("_name", "_attrs")

    def __init__(self, name: str, attrs: dict[str, Any]):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_attrs", attrs)

    def __getattr__(self, key):
        attrs = object.__getattribute__(self, "_attrs")
        try:
            return attrs[key]
        except KeyError:
            name = object.__getattribute__(self, "_name")
            raise AttributeError(f"module {name!r} has no attribute {key!r} in the bot sandbox") from None

    def __setattr__(self, key, value):
        raise AttributeError("sandboxed modules are read-only")

    def __delattr__(self, key):
        raise AttributeError("sandboxed modules are read-only")

    def __dir__(self):
        return sorted(object.__getattribute__(self, "_attrs"))

    def __repr__(self):
        return f"<sandboxed module {object.__getattribute__(self, '_name')!r}>"


_DANGEROUS_VALUES = {id(v) for v in (builtins.getattr, builtins.setattr, builtins.delattr, builtins.eval, builtins.exec,
                                     builtins.compile, builtins.open, builtins.__import__, builtins.vars,
                                     builtins.globals, builtins.locals, builtins.breakpoint, builtins.input,
                                     types.FunctionType, types.CodeType, types.FrameType, types.ModuleType,
                                     types.TracebackType, types.CellType, types.MethodType, types.GeneratorType)}
EXCLUDED = {
    "typing": {"get_type_hints"},  # evaluates string annotations with the real builtins
    "functools": {"singledispatch", "singledispatchmethod"},  # register() calls get_type_hints on any object
    "random": {"SystemRandom"},  # os.urandom: not reproducible
}


def _field_names_ok(cls: type) -> None:
    """dataclasses generate __init__ source from field names; reject anything that is not a plain identifier."""
    names = list(_tdict(cls).get("__annotations__", {}))
    for c in _tmro(cls):
        fields = _tdict(c).get("__dataclass_fields__")
        if isinstance(fields, dict):
            names += list(fields) + [getattr(f, "name", "") for f in fields.values()]
    for n in names:
        if not (isinstance(n, str) and n.isidentifier() and not keyword.iskeyword(n) and not n.startswith("__")):
            raise SandboxViolation(f"dataclass field name {n!r} is not allowed")


def _safe_dataclass(cls=None, /, **kw):
    def wrap(c):
        if not (isinstance(c, type) and _bot_class(c)):
            raise SandboxViolation("dataclass() only applies to classes defined in bot code")
        _field_names_ok(c)
        return dataclasses.dataclass(c, **kw)

    return wrap if cls is None else wrap(cls)


def _safe_make_dataclass(cls_name, fields, *, bases=(), **kw):
    for b in bases:
        if isinstance(b, type):
            _field_names_ok(b)
    return dataclasses.make_dataclass(cls_name, fields, bases=bases, **kw)


def _safe_update_wrapper(wrapper, wrapped):
    if not (_bot_owned(wrapper) or getattr(wrapper, "__module__", None) == BOT_MODULE):
        raise SandboxViolation("update_wrapper only on your own functions or objects")
    return functools.update_wrapper(wrapper, wrapped)


def _safe_wraps(wrapped):
    return lambda wrapper: _safe_update_wrapper(wrapper, wrapped)


def _safe_total_ordering(cls):
    if not (isinstance(cls, type) and _bot_class(cls)):
        raise SandboxViolation("total_ordering only on classes defined in bot code")
    return functools.total_ordering(cls)


REPLACED = {
    ("functools", "update_wrapper"): _safe_update_wrapper,
    ("functools", "wraps"): _safe_wraps,
    ("functools", "total_ordering"): _safe_total_ordering,
    ("dataclasses", "dataclass"): _safe_dataclass,
    ("dataclasses", "make_dataclass"): _safe_make_dataclass,
}
_PROXIES: dict[str, ModuleProxy] = {}


def module_proxy(name: str) -> ModuleProxy:
    if name in _PROXIES:
        return _PROXIES[name]
    if name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in bot code")
    mod = importlib.import_module(name)
    attrs: dict[str, Any] = {}
    for k, v in sorted(vars(mod).items()):
        if k.startswith("_") or k in EXCLUDED.get(name, ()) or id(v) in _DANGEROUS_VALUES:
            continue
        if isinstance(v, types.ModuleType):
            sub = getattr(v, "__name__", "")
            if sub in ALLOWED_MODULES and sub.startswith(name + "."):
                attrs[k] = module_proxy(sub)
            continue
        attrs[k] = REPLACED.get((name, k), v)
    attrs["__name__"] = name
    attrs["__all__"] = sorted(k for k in attrs if not k.startswith("_"))
    _PROXIES[name] = ModuleProxy(name, attrs)
    return _PROXIES[name]


def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """`__import__` for bot code: whitelisted modules only, and the module objects handed out are proxies."""
    if level or name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in bot code")
    if fromlist:
        return module_proxy(name)
    module_proxy(name)
    return module_proxy(name.split(".")[0])


# ----------------------------------------------------------------------------------------------- compile
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset", "hash", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min", "next", "object", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "type", "property",
    "staticmethod", "classmethod", "super", "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
    "StopIteration", "AttributeError", "ZeroDivisionError", "RuntimeError", "AssertionError", "NotImplementedError",
    "ArithmeticError", "LookupError", "True", "False", "None", "callable", "chr", "ord", "format", "id", "bin", "hex",
    "NotImplemented", "Ellipsis",
]


def safe_builtins() -> dict[str, Any]:
    b = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(builtins, k)}
    b["getattr"], b["hasattr"], b["setattr"] = _safe_getattr, _safe_hasattr, _safe_setattr
    b["__import__"] = restricted_import
    b["__build_class__"] = builtins.__build_class__
    b["__name__"] = "bot"
    return b


class _PrivateAttrGuard(ast.NodeTransformer):
    """Rewrite `X._name` (any context) into `__sandbox_private__(X, '_name')._name`."""

    def visit_Attribute(self, node: ast.Attribute):
        self.generic_visit(node)
        if node.attr.startswith("_") and not node.attr.startswith("__"):
            node.value = ast.Call(func=ast.Name(id=GUARD_NAME, ctx=ast.Load()),
                                  args=[node.value, ast.Constant(node.attr)], keywords=[])
        return node


def compile_sandboxed(code: str) -> tuple[Any, dict[str, Any]]:
    """Code object and fresh namespace for (already checked) bot source."""
    tree = ast.fix_missing_locations(_PrivateAttrGuard().visit(ast.parse(code)))
    namespace: dict[str, Any] = {"__name__": BOT_MODULE, "__builtins__": safe_builtins(), GUARD_NAME: _guard_private}
    # dont_inherit: bot code must not pick up this module's `from __future__ import annotations`
    return compile(tree, "<bot.py>", "exec", dont_inherit=True), namespace
