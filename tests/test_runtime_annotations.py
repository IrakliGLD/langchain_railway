"""Guard against annotations that resolve locally but fail on the runtime Python.

The production image pins Python 3.11, which evaluates function annotations
eagerly at definition time. Local development runs 3.14, where PEP 649 defers
that evaluation, so a missing annotation import imports cleanly here and raises
``NameError`` on 3.11.

This test reproduces 3.11's rule without needing 3.11: for every module that
does *not* opt into ``from __future__ import annotations``, every unquoted name
used in an eagerly evaluated annotation must resolve in that module's namespace.
Quoted annotations and modules with the future import stay lazy on 3.11 too, so
they are correctly exempt.

"Eagerly evaluated" covers two moments, and the difference matters. A module- or
class-level annotation is evaluated while the module imports, so a bad name
there kills container start. A *nested* ``def`` is evaluated when its enclosing
function runs — later, but no less eagerly. That variant leaves the container
healthy and detonates on the first call instead, which is how an undefined
annotation name reached production and failed every report while the worker
started normally.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import os
import pkgutil
from pathlib import Path

os.environ.setdefault("SUPABASE_DB_URL", "postgresql://user:pass@localhost/db")
os.environ.setdefault("ENAI_GATEWAY_SECRET", "test-gateway-key")
os.environ.setdefault("ENAI_SESSION_SIGNING_SECRET", "test-session-key")
os.environ.setdefault("ENAI_EVALUATE_SECRET", "test-evaluate-key")
os.environ.setdefault("MODEL_TYPE", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

# Mirrors the Dockerfile's runtime allow-list: anything the image ships can
# break container start, so everything it ships is covered here.
_RUNTIME_PACKAGES = (
    "agent",
    "analysis",
    "config_metrics",
    "contracts",
    "core",
    "guardrails",
    "knowledge",
    "utils",
    "visualization",
)
_RUNTIME_MODULES = ("config", "context", "main", "models", "report_worker")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _defers_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _signature_annotations(node):
    """Yield the annotations a ``def`` statement evaluates when it executes."""

    arguments = node.args
    for arg in (
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ):
        if arg is not None and arg.annotation is not None:
            yield arg.annotation
    if node.returns is not None:
        yield node.returns


def _bound_in_function(node) -> frozenset[str]:
    """Names a function body makes resolvable to definitions nested inside it.

    A nested annotation may legitimately reference an enclosing parameter or
    local, which is not a module attribute. Collecting them scope-blind is
    deliberately permissive: the failure being guarded against is a name bound
    nowhere at all.
    """

    names: set[str] = set()
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.arg):
            names.add(descendant.arg)
        elif isinstance(
            descendant, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.add(descendant.name)
        elif isinstance(descendant, ast.Name) and isinstance(
            descendant.ctx, ast.Store
        ):
            names.add(descendant.id)
        elif isinstance(descendant, (ast.Import, ast.ImportFrom)):
            for alias in descendant.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def _blocks(node):
    """Yield the statement lists of a compound statement.

    ``def`` is not only found at module top level — it also appears under
    ``if``, ``try``, ``for``, ``while`` and ``with``.
    """

    if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
        # This branch never runs at runtime, so nothing in it is evaluated.
        return
    for field in ("body", "orelse", "finalbody"):
        block = getattr(node, field, None)
        if isinstance(block, list):
            yield block
    for handler in getattr(node, "handlers", []):
        yield handler.body


def _eagerly_evaluated_annotations(tree: ast.Module):
    """Yield ``(annotation, enclosing_names)`` pairs Python 3.11 evaluates."""

    def walk(body, enclosing: frozenset[str]):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for annotation in _signature_annotations(node):
                    yield annotation, enclosing
                # The body runs later, with this function's own names in scope.
                yield from walk(node.body, enclosing | _bound_in_function(node))
            elif isinstance(node, ast.ClassDef):
                # A class body executes at import, so its annotations do too.
                yield from walk(node.body, enclosing)
            else:
                for block in _blocks(node):
                    yield from walk(block, enclosing)

    yield from walk(tree.body, frozenset())


def _unresolved_annotation_names(module_name: str, source: str) -> list[str]:
    tree = ast.parse(source)
    if _defers_annotations(tree):
        return []
    module = importlib.import_module(module_name)
    unresolved: list[str] = []
    for annotation, enclosing in _eagerly_evaluated_annotations(tree):
        if isinstance(annotation, ast.Constant):
            # A quoted annotation is a plain string at runtime on 3.11.
            continue
        for node in ast.walk(annotation):
            if not isinstance(node, ast.Name):
                continue
            if node.id in enclosing:
                continue
            if hasattr(module, node.id) or hasattr(builtins, node.id):
                continue
            unresolved.append(f"{module_name}:{annotation.lineno} {node.id}")
    return sorted(set(unresolved))


def test_runtime_modules_have_no_annotations_that_break_on_python_311():
    failures: list[str] = []
    for package in _RUNTIME_PACKAGES:
        for module_info in pkgutil.iter_modules([str(_REPO_ROOT / package)]):
            module_name = f"{package}.{module_info.name}"
            source_path = _REPO_ROOT / package / f"{module_info.name}.py"
            if not source_path.exists():
                continue
            failures.extend(
                _unresolved_annotation_names(
                    module_name,
                    source_path.read_text(encoding="utf-8"),
                )
            )

    for module_name in _RUNTIME_MODULES:
        source_path = _REPO_ROOT / f"{module_name}.py"
        failures.extend(
            _unresolved_annotation_names(
                module_name,
                source_path.read_text(encoding="utf-8"),
            )
        )

    assert not failures, (
        "Unquoted annotation names that Python 3.11 evaluates but cannot "
        "resolve: " + ", ".join(failures)
    )


def test_guard_covers_annotations_on_nested_definitions():
    """A nested ``def`` must be reached, not just module- and class-level ones.

    The original guard walked only module and class bodies. A nested helper
    annotated with an unimported type therefore passed, and raised on 3.11 the
    first time its enclosing function ran. This pins the coverage so that gap
    cannot quietly reopen.
    """

    source = (
        "def outer(argument: int) -> int:\n"
        "    scale = 2\n"
        "    def inner(value: MissingType, factor: int = scale) -> int:\n"
        "        return value * factor * scale\n"
        "    return inner(argument)\n"
    )
    tree = ast.parse(source)
    found = {
        node.id
        for annotation, enclosing in _eagerly_evaluated_annotations(tree)
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and node.id not in enclosing
    }
    assert "MissingType" in found, "nested annotation was not inspected"


def test_guard_accepts_annotations_naming_an_enclosing_local():
    """An enclosing parameter or local resolves at runtime and is not a finding."""

    source = (
        "def outer(alias: object):\n"
        "    Local = dict\n"
        "    def inner(first: alias, second: Local) -> None:\n"
        "        return None\n"
        "    return inner\n"
    )
    tree = ast.parse(source)
    unresolved = {
        node.id
        for annotation, enclosing in _eagerly_evaluated_annotations(tree)
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name) and node.id not in enclosing
    }
    assert unresolved == {"object"}, unresolved
