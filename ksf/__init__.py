"""
Knowledge-Synthesized Framework (KSF) v3
基于模块化解耦的知识合成框架

核心架构：
- K模块 (知识发现器): 负责发散性的知识探索和关联发现
- S模块 (提示装配引擎): 负责收敛性的逻辑组织和提示词装配
- 核心编排器: 协调K模块和S模块的工作流程

主要特性：
- 模块化设计，K模块和S模块完全解耦
- 基于sentence-transformers的高效知识检索
- 智能关联发现，挖掘隐藏的语义关联
- 结构化提示词装配，生成高质量的LLM指令
- 灵活的模板系统，支持多种装配策略
"""

# 版本信息
__version__ = "3.0.0"
__author__ = "KSF Team"
__description__ = "Knowledge-Synthesized Framework v3 - 模块化知识合成框架"

# 核心组件导入
from .core import KSFOrchestrator
from .k_module import (
    KnowledgeDiscoverer, 
    RerankedItem,
    ResonancePacket, 
    EmergedConcept,
    RetrievalInstruction
)
from .s_module import PromptAssembler

# 主要接口
__all__ = [
    # 核心编排器 - 主要使用接口
    'KSFOrchestrator',
    
    # K模块组件
    'KnowledgeDiscoverer',
    'RerankedItem',
    'ResonancePacket',
    'EmergedConcept',
    'RetrievalInstruction',
    
    # S模块组件
    'PromptAssembler',
    
    # 版本信息
    '__version__',
    '__author__',
    '__description__'
] 