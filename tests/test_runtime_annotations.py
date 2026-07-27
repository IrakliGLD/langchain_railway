"""Guard against annotations that resolve locally but fail on the runtime Python.

The production image pins Python 3.11, which evaluates function annotations
eagerly at definition time. Local development runs 3.14, where PEP 649 defers
that evaluation, so a missing annotation import imports cleanly here and raises
``NameError`` at container start.

This test reproduces 3.11's rule without needing 3.11: for every module that
does *not* opt into ``from __future__ import annotations``, every unquoted name
used in a module-level or class-level annotation must resolve in that module's
namespace. Quoted annotations and modules with the future import stay lazy on
3.11 too, so they are correctly exempt.
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


def _eagerly_evaluated_annotations(tree: ast.Module):
    """Yield annotation nodes Python 3.11 evaluates while importing the module."""

    def walk(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
            elif isinstance(node, ast.ClassDef):
                # A class body executes at import, so its annotations do too.
                yield from walk(node.body)

    yield from walk(tree.body)


def _unresolved_annotation_names(module_name: str, source: str) -> list[str]:
    tree = ast.parse(source)
    if _defers_annotations(tree):
        return []
    module = importlib.import_module(module_name)
    unresolved: list[str] = []
    for annotation in _eagerly_evaluated_annotations(tree):
        if isinstance(annotation, ast.Constant):
            # A quoted annotation is a plain string at runtime on 3.11.
            continue
        for node in ast.walk(annotation):
            if not isinstance(node, ast.Name):
                continue
            if hasattr(module, node.id) or hasattr(builtins, node.id):
                continue
            unresolved.append(f"{module_name}: {node.id}")
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
        "Unquoted annotation names that Python 3.11 evaluates at import but "
        "cannot resolve: " + ", ".join(failures)
    )
