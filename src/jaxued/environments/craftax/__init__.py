"""Optional Craftax integration for JaxUED.

Craftax is not imported at module import time. Install the optional extra with
``uv sync --extra craftax`` before constructing Craftax environments.
"""

from .env import (
    CraftaxSymbolicBundle,
    make_craftax_symbolic,
    make_craftax_symbolic_level_generator,
    make_craftax_symbolic_renderer,
)
from .mutators import make_craftax_symbolic_mutator
from .wrappers import CraftaxLogWrapper, CraftaxUEDWrapper

__all__ = [
    "CraftaxLogWrapper",
    "CraftaxSymbolicBundle",
    "CraftaxUEDWrapper",
    "make_craftax_symbolic",
    "make_craftax_symbolic_level_generator",
    "make_craftax_symbolic_mutator",
    "make_craftax_symbolic_renderer",
]
