"""
KSF核心编排器
负责协调K模块和S模块的工作流程，提供端到端的服务
"""

import logging
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import json

from ..k_module.discoverer import KnowledgeDiscoverer
from ..k_module.data_structures import RerankedItem
from ..s_module.assembler import PromptAssembler
from ..utils.data_utils import load_knowledge_base_from_file
from ..utils import pseudo_api_wrapper
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KSFOrchestrator:
    """
    KSF核心编排器
    协调K模块(知识发现器)和S模块(提示装配引擎)的工作
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("🔧 Initializing KSF V3 Orchestrator...")
        
        # --- K-Module Initialization ---
        k_config = config['discoverer']
        self.logger.info(f"📚 Initializing K-Module with knowledge file: {k_config['knowledge_file']}")
        
        self.discoverer = KnowledgeDiscoverer(**k_config)
        self.discoverer.load_or_build_index(force_rebuild=config.get('force_rebuild_index', False))
        
        # --- S-Module Initialization ---
        s_config = config['assembler']
        self.logger.info("🔧 Initializing S-Module (Prompt Assembler)...")
        
        # Load the knowledge manifest
        manifest_path = s_config['manifest_path']
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            self.logger.info(f"✓ Knowledge Manifest loaded from {manifest_path}")
        except FileNotFoundError:
            self.logger.error(f"FATAL: Knowledge Manifest not found at {manifest_path}. Please run the build script.")
            raise
        except json.JSONDecodeError:
            self.logger.error(f"FATAL: Could not decode Knowledge Manifest at {manifest_path}.")
            raise

        self.assembler = PromptAssembler(
            templates_dir=s_config['templates_dir'],
            manifest=manifest
        )

        # Final check
        self.logger.info("✅ KSF V3 Orchestrator Initialized Successfully.")
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Processes a user query using the new "Dynamic K-S-Collaboration" model.
        K-Module provides an initial "instinctive" retrieval, while S-Module
        analyzes the query to provide "corrective" instructions if needed.
        """
        self.logger.info(f"🚀 Started processing query: {query_text}")

        # --- Stage 1a (Parallel): S-Module analyzes the query and generates a retrieval instruction ---
        s_instruction = self.assembler.generate_instruction(query_text)
        self.logger.info(f"✅ S-Module generated instruction: {s_instruction}")

        # --- Stage 2: Decision Point based on S-Module's analysis ---

        # Path 1: S-Module rejects the query outright.
        if s_instruction.mode == 'REJECT':
            self.logger.warning(f"🛑 Instruction rejected by S-Module: {s_instruction.filters.get('reason')}")
            return {"answer": s_instruction.filters.get('reason', "无法回答此问题。"), "knowledge_packet": {}}

        # Path 2: S-Module deems the query complex and requires a specific, guided retrieval.
        # This is the "S-Corrects" path where S's instruction overrides K's instinct.
        if s_instruction.mode != 'SEMANTIC':
            self.logger.info(f"S-Module issued a corrective instruction ({s_instruction.mode}). Overriding K-Module's instinct.")
            retrieved_items = self.discoverer.retrieve_direct_knowledge(s_instruction, top_k=top_k * 2)
            self.logger.info(f"🧠 K-Module performed a GUIDED retrieval, found {len(retrieved_items)} items.")
        
        # Path 3: S-Module finds the query straightforward (pass-through).
        # We trust K-Module's "instinctive" first pass.
        else:
            self.logger.info("S-Module passed the query. Using K-Module's instinctive retrieval.")
            retrieved_items = self.discoverer.retrieve_direct_knowledge(s_instruction, top_k=top_k * 2)
            self.logger.info(f"🧠 K-Module performed an INSTINCTIVE retrieval, found {len(retrieved_items)} items.")

        if not retrieved_items:
            self.logger.warning("K-Module returned no results.")
            return {"answer": "抱歉，我在知识库中找不到相关信息。", "knowledge_packet": {}}
            
        # --- Stage 3: S-Module assesses relevance of the chosen candidates (either from guided or instinctive path) ---
        self.logger.info("🔬 S-Module assessing relevance of candidates...")
        
        # Use a temporary list of tuples to hold items and their relevance scores
        scored_items = []
        for item in retrieved_items:
            relevance_score = self.assembler.analyzer.assess_relevance(query_text, item.content)
            if relevance_score > 0.3: # Relevance threshold
                scored_items.append((item, relevance_score))
                self.logger.info(f"  - Item '{item.id}' is RELEVANT (Score: {relevance_score:.2f})")
            else:
                self.logger.info(f"  - Item '{item.id}' is NOT RELEVANT (Score: {relevance_score:.2f})")

        # Sort by relevance score in descending order
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the sorted RerankedItem objects
        final_knowledge = [item for item, score in scored_items[:top_k]]
        
        self.logger.info(f"✅ Final filtered knowledge packet contains {len(final_knowledge)} items.")

        # --- Stage 4: S-Module assembles the final prompt/answer ---
        knowledge_packet = {
            "query": query_text,
            "direct_knowledge": final_knowledge,
            "associated_knowledge": [] # Placeholder for now
        }
        
        final_answer = self.assembler.assemble_prompt(knowledge_packet)
        self.logger.info(f"✅ Final answer assembled.")
        
        return {"answer": final_answer, "knowledge_packet": knowledge_packet}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Returns a dictionary with the current status of the system."""
        status = {
            "k_module_status": {
                "model_name": self.discoverer.model_name,
                "knowledge_base_size": len(self.discoverer.knowledge_base),
                "index_built": self.discoverer.index is not None,
                "relevance_threshold": self.discoverer.relevance_threshold
            },
            "s_module_status": {
                "templates": self.assembler.list_templates()
            },
            "system_ready": self.discoverer.index is not None
        }
        return status
    
    def rebuild_knowledge_index(self):
        """
        Rebuilds the knowledge index. Useful when the knowledge base is updated.
        """
        logger.info("🔄 Rebuilding knowledge index...")
        
        # The knowledge base is already loaded during init, so we just need to re-encode and build.
        # This assumes the file at self.knowledge_base_path has been updated.
        logger.info("Re-loading knowledge base from disk...")
        self.discoverer.load_knowledge_base(self.knowledge_base_path)
        
        logger.info("Building and saving new index...")
        self.discoverer.build_and_save_index()
        
        logger.info("✅ Knowledge index rebuilt successfully.")
    
    def add_custom_template(self, name: str, template_content: str):
        """
        添加自定义S模块模板
        
        Args:
            name: 模板名称
            template_content: 模板内容
        """
        self.assembler.add_custom_template(name, template_content)
        logger.info(f"✅ Added custom template: {name}")
    
    def test_system(self, test_query: str = "测试查询") -> Dict[str, Any]:
        """
        测试系统功能
        
        Args:
            test_query: 测试查询
            
        Returns:
            测试结果
        """
        logger.info("🧪 开始系统测试...")
        
        try:
            # 测试基本查询处理
            result = self.query(
                query_text=test_query,
                top_k=3
            )
            
            # 检查结果
            assert result is not None, "System test failed: query returned None"
            assert 'final_prompt' in result and result['final_prompt'], "System test failed: final_prompt is empty"
            
            test_results = {
                'basic_query_test': {
                    'success': True,
                    'direct_knowledge_count': len(result['analysis']['direct_knowledge']),
                    'hidden_associations_count': len(result['analysis']['hidden_associations']),
                    'final_prompt_length': len(result['final_prompt'])
                },
                'strategy_tests': {}
            }
            
            # 测试不同策略
            strategies = ['general', 'analysis', 'comparison', 'guide']
            strategy_results = {}
            
            for strategy in strategies:
                try:
                    strategy_result = self.query(
                        query_text=test_query,
                        top_k=3
                    )
                    strategy_results[strategy] = {
                        'success': True,
                        'prompt_length': len(strategy_result['final_prompt'])
                    }
                except Exception as e:
                    strategy_results[strategy] = {
                        'success': False,
                        'error': str(e)
                    }
            
            test_results['strategy_tests'] = strategy_results
            
            test_results['system_stats'] = self.get_system_status()
            
            logger.info("✅ 系统测试完成")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ 系统测试失败: {e}")
            return {
                'system_ready': False,
                'error': str(e),
                'system_stats': self.get_system_status()
            }


def main():
    """
    演示KSF编排器的使用
    """
    # 检查知识库文件是否存在
    knowledge_base_path = "./data/云和文旅知识库数据集.json"
    if not Path(knowledge_base_path).exists():
        print(f"❌ 知识库文件不存在: {knowledge_base_path}")
        print("请确保知识库文件存在后再运行")
        return
    
    # 初始化编排器
    orchestrator = KSFOrchestrator(
        knowledge_base_path=knowledge_base_path,
        k_module_model="Snowflake/snowflake-arctic-embed-m",
        k_module_index_dir="checkpoints/k_module_index",
        k_module_weights_path="data/knowledge_weights.json",
        auto_build_index=True
    )
    
    # 系统测试
    print("🧪 执行系统测试...")
    test_result = orchestrator.test_system("云和文旅有哪些主要景点？")
    print(f"测试结果: {test_result['system_ready']}")
    
    # 演示查询处理
    test_queries = [
        "云和文旅的主要景点有哪些？",
        "如何规划云和文旅的游览路线？",
        "云和文旅的住宿条件如何？"
    ]
    
    print("\n🚀 演示查询处理...")
    for query in test_queries:
        print(f"\n查询: {query}")
        result = orchestrator.query(query, top_k=3)
        print(f"提示词长度: {len(result['final_prompt'])} 字符")
        print("=" * 50)
        print(result['final_prompt'])
        print("=" * 50)


if __name__ == "__main__":
    main() 