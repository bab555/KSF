"""
S模块 (提示装配引擎)
负责收敛性的逻辑组织，将K模块的知识包装配成结构化提示词
"""

from .assembler import PromptAssembler
from .processors import *

__all__ = ['PromptAssembler'] 