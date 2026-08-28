"""Synthetic past performance library and QRA corpus generation.

The pipeline is deliberately one-directional: a seeded entity graph is built
first, deterministic scoring derives the golden answers from it, and prose is
only ever wrapped around facts the graph already fixed. Nothing in here asks a
language model what the right answer is.
"""

from .build import Multipliers, build_and_write, generate, write
from .graph import SCALES, World

__all__ = ["Multipliers", "SCALES", "World", "build_and_write", "generate", "write"]
