"""
KSF核心编排器
负责协调K模块和S模块的工作流程，提供端到端的服务
"""

import logging
import json
from typing import Dict, Any

from ..k_module.discoverer import KnowledgeDiscoverer
from ..s_module.assembler import PromptAssembler

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KSFOrchestrator:
    """
    KSF核心编排器
    协调K模块(知识发现器)和S模块(提示装配引擎)的工作
    """
    
    def __init__(self, config: Dict[str, Any]):
        logger.info("🔧 正在初始化KSF V3编排器...")
        
        # --- K-Module 初始化 ---
        k_config = config['discoverer']
        logger.info(f"📚 正在初始化K模块，知识文件: {k_config['knowledge_file']}")
        
        self.discoverer = KnowledgeDiscoverer(**k_config)
        self.discoverer.load_or_build_index(force_rebuild=config.get('force_rebuild_index', False))
        
        # --- S-Module 初始化 ---
        s_config = config['assembler']
        logger.info("🔧 正在初始化S模块 (提示装配器)...")
        
        # 加载知识清单 (Manifest)
        manifest_path = s_config['manifest_path']
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            logger.info(f"✓ 知识清单加载成功: {manifest_path}")
        except FileNotFoundError:
            logger.error(f"致命错误: 在 {manifest_path} 未找到知识清单。请先运行构建脚本。")
            raise
        except json.JSONDecodeError:
            logger.error(f"致命错误: 无法解析 {manifest_path} 的知识清单。")
            raise

        self.assembler = PromptAssembler(
            templates_dir=s_config['templates_dir'],
            manifest=manifest
        )

        logger.info("✅ KSF V3 编排器初始化成功。")
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        使用"动态K-S协作"模型处理用户查询。
        K模块提供初步的"直觉"检索，S模块则分析查询以在需要时提供"校正"指令。
        """
        logger.info(f"🚀 开始处理查询: {query_text}")

        # --- 阶段1: S模块分析查询并生成检索指令 ---
        s_instruction = self.assembler.generate_instruction(query_text)
        logger.info(f"✅ S模块生成指令: {s_instruction}")

        # --- 阶段2: 基于S模块的分析进行决策 ---

        # 路径1: S模块直接拒绝查询
        if s_instruction.mode == 'REJECT':
            rejection_reason = s_instruction.filters.get('reason', "无法回答此问题。")
            logger.warning(f"🛑 S模块拒绝了该指令: {rejection_reason}")
            return {"answer": rejection_reason, "knowledge_packet": {}}

        # 路径2: S模块未拒绝，由K模块执行检索 (无论是指导性还是常规性)
        # 注意：无论S模块是提供校正指令(如'ENTITY_FOCUS')还是默认的'SEMANTIC'，
        # K模块的 `retrieve_direct_knowledge` 都使用该指令来指导其操作。
        # 这统一了"S校正"和"K直觉"两种路径。
        log_msg = (
            f"S模块发出了一个校正指令 ({s_instruction.mode})。"
            if s_instruction.mode != 'SEMANTIC' 
            else "S模块通过了查询，使用K模块的直觉检索。"
        )
        logger.info(log_msg)
        
        retrieved_items = self.discoverer.retrieve_direct_knowledge(s_instruction, top_k=top_k * 2) # 取2*top_k作为候选
        logger.info(f"🧠 K模块执行检索，找到 {len(retrieved_items)} 个候选项。")

        if not retrieved_items:
            logger.warning("K模块未返回任何结果。")
            return {"answer": "抱歉，我在知识库中找不到相关信息。", "knowledge_packet": {}}
            
        # --- 阶段3: S模块对候选集进行相关性评估 ---
        logger.info("🔬 S模块正在评估候选项的相关性...")
        
        # 使用一个临时元组列表来存储候选项及其相关性得分
        scored_items = []
        for item in retrieved_items:
            relevance_score = self.assembler.analyzer.assess_relevance(query_text, item.content)
            if relevance_score > 0.3: # 相关性阈值
                scored_items.append((item, relevance_score))
                logger.info(f"  - 候选项 '{item.id}' 相关 (得分: {relevance_score:.2f})")
            else:
                logger.info(f"  - 候选项 '{item.id}' 不相关 (得分: {relevance_score:.2f})")

        # 按相关性得分降序排序
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # 提取排序后的 RerankedItem 对象
        final_knowledge = [item for item, score in scored_items[:top_k]]
        
        logger.info(f"✅ 最终过滤后的知识包包含 {len(final_knowledge)} 个条目。")

        # --- 阶段4: S模块装配最终的答案 ---
        knowledge_packet = {
            "query": query_text,
            "direct_knowledge": final_knowledge,
            "associated_knowledge": [] # 暂时保留
        }
        
        final_answer = self.assembler.assemble_prompt(knowledge_packet)
        logger.info("✅ 最终答案已装配完成。")
        
        return {"answer": final_answer, "knowledge_packet": knowledge_packet}
    
    def get_system_status(self) -> Dict[str, Any]:
        """返回系统的当前状态字典。"""
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
    
    def rebuild_knowledge_index(self, force_rebuild: bool = True):
        """
        重建知识索引。当知识库更新时非常有用。
        """
        logger.info("🔄 正在重建知识索引...")
        
        # 知识库在初始化时已加载，我们只需重新编码和构建。
        # 此处假设 self.discoverer.knowledge_file_path 的文件已被更新。
        logger.info("正在从磁盘重新加载知识库...")
        self.discoverer.load_knowledge_base()
        
        logger.info("正在构建并保存新索引...")
        self.discoverer.load_or_build_index(force_rebuild=force_rebuild)
        
        logger.info("✅ 知识索引重建成功。")
    
    def add_custom_template(self, name: str, template_content: str):
        """
        添加自定义S模块模板
        
        Args:
            name: 模板名称
            template_content: 模板内容
        """
        self.assembler.add_custom_template(name, template_content)
        logger.info(f"✅ 已添加自定义模板: {name}")
    
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
            assert result is not None, "系统测试失败: 查询返回None"
            assert 'final_prompt' in result and result['final_prompt'], "系统测试失败: final_prompt is empty"
            
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