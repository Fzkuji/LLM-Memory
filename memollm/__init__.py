"""
MemOLLM: Memory-enhanced Language Learning Model
"""

from .llm import EbbinghausLLM
from .memory import EbbinghausMemoryManager as MemoryManager

__version__ = "0.1.0"
__all__ = ["EbbinghausLLM", "MemoryManager"]