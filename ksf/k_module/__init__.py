"""
K模块 (知识发现器)
负责发散性的知识探索，包括检索直接知识和发现隐藏关联
"""

from .discoverer import KnowledgeDiscoverer
from .data_structures import KnowledgePacket, KnowledgeItem, HiddenAssociation

__all__ = ['KnowledgeDiscoverer', 'KnowledgePacket', 'KnowledgeItem', 'HiddenAssociation'] 