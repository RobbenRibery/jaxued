"""Contract tests for public library documentation."""

import ast
from pathlib import Path
from typing import Iterable, Union

import pytest


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "jaxued"
PYTHON_MODULES = tuple(sorted(SOURCE_ROOT.rglob("*.py")))
Definition = Union[ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef]


def _public_definitions(module: ast.Module) -> Iterable[Definition]:
    """Yield module-level definitions and public class methods.

    Args:
        module: Parsed Python module.

    Yields:
        Public classes and functions directly owned by the module, followed by
        public methods directly owned by each public class.
    """
    for definition in module.body:
        if not isinstance(
            definition,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        if definition.name.startswith("_"):
            continue

        yield definition
        if isinstance(definition, ast.ClassDef):
            for method in definition.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not method.name.startswith("_"):
                        yield method


def _argument_names(
    function: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> list[str]:
    """Return documented positional and keyword-only argument names.

    Args:
        function: Parsed function definition.

    Returns:
        Argument names excluding conventional ``self`` and ``cls`` receivers.
    """
    arguments = (
        function.args.posonlyargs + function.args.args + function.args.kwonlyargs
    )
    return [
        argument.arg for argument in arguments if argument.arg not in {"self", "cls"}
    ]


def _returns_value(
    function: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> bool:
    """Determine whether an annotation declares a returned value.

    Args:
        function: Parsed function definition.

    Returns:
        True when the function has a return annotation other than ``None``.
    """
    annotation = function.returns
    return annotation is not None and not (
        isinstance(annotation, ast.Constant) and annotation.value is None
    )


@pytest.mark.parametrize(
    "module_path",
    PYTHON_MODULES,
    ids=lambda path: str(path.relative_to(SOURCE_ROOT)),
)
def test_public_library_uses_google_style_docstrings(module_path: Path) -> None:
    """Every public library contract should carry structured documentation."""
    module = ast.parse(module_path.read_text())
    assert ast.get_docstring(module), f"{module_path} has no module docstring"

    for definition in _public_definitions(module):
        docstring = ast.get_docstring(definition)
        location = f"{module_path}:{definition.lineno} ({definition.name})"
        assert docstring, f"{location} has no docstring"

        if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _argument_names(definition):
                assert "Args:" in docstring, f"{location} has no Args section"
            if _returns_value(definition):
                assert "Returns:" in docstring, f"{location} has no Returns section"
