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
from .k_module import KnowledgeDiscoverer, KnowledgePacket, KnowledgeItem, HiddenAssociation
from .s_module import PromptAssembler

# 主要接口
__all__ = [
    # 核心编排器 - 主要使用接口
    'KSFOrchestrator',
    
    # K模块组件
    'KnowledgeDiscoverer',
    'KnowledgePacket',
    'KnowledgeItem', 
    'HiddenAssociation',
    
    # S模块组件
    'PromptAssembler',
    
    # 版本信息
    '__version__',
    '__author__',
    '__description__'
]

# 便捷函数
def create_ksf_system(knowledge_base_path: str, **kwargs):
    """
    快速创建KSF系统的便捷函数
    
    Args:
        knowledge_base_path: 知识库文件路径
        **kwargs: 传递给KSFOrchestrator的其他参数
        
    Returns:
        KSFOrchestrator实例
    """
    return KSFOrchestrator(knowledge_base_path=knowledge_base_path, **kwargs)


def quick_query(knowledge_base_path: str, query: str, **kwargs):
    """
    快速查询的便捷函数
    
    Args:
        knowledge_base_path: 知识库文件路径
        query: 用户查询
        **kwargs: 传递给process_query的其他参数
        
    Returns:
        查询结果
    """
    orchestrator = create_ksf_system(knowledge_base_path)
    return orchestrator.process_query(query, **kwargs) 