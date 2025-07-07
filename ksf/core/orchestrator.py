"""
KSF核心编排器
负责协调K模块和S模块的工作流程，提供端到端的服务
"""

import logging
import json
import os
from typing import Dict, Any

from sentence_transformers import SentenceTransformer
from ..k_module.discoverer import KnowledgeDiscoverer
from ..s_module.assembler import PromptAssembler
from ..utils.pseudo_api_wrapper import PseudoAPIWrapper
from ..k_module.data_structures import RetrievalInstruction
from ..connectors.faiss_connector import FAISSConnector

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KSFOrchestrator:
    """
    KSF框架的总编排器。
    负责初始化和协调K模块和S模块。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化编排器。

        Args:
            config (Dict[str, Any]): 包含所有必要路径和参数的配置字典。
        """
        logger.info("初始化KSF编排器...")
        self.config = config
        
        self.k_module_config = self.config.get('discoverer', {})
        self.s_module_config = self.config.get('assembler', {})
        # self.api_config = self.config.get('pseudo_api', {}) # Temporarily disable

        self._model = self._init_model()
        self._connector = self._init_connector()
        self.k_module = self._init_k_module()
        self.s_module = self._init_s_module()
        # self.api_wrapper = self._init_api_wrapper() # Temporarily disable
        
        logger.info("✓ KSF编排器初始化完成。")

    def _init_model(self) -> SentenceTransformer:
        """初始化句子转换器模型。"""
        model_name = self.k_module_config.get('model_name')
        adapter_path = self.k_module_config.get('adapter_path')
        logger.info(f"正在加载句子转换器模型: {model_name}")
        model = SentenceTransformer(model_name)
        if adapter_path and os.path.exists(adapter_path):
            try:
                model.load_adapter(adapter_path)
                logger.info(f"✓ 成功从 {adapter_path} 加载PEFT适配器。")
            except Exception as e:
                logger.error(f"加载PEFT适配器失败: {e}")
        return model
        
    def _init_connector(self) -> FAISSConnector:
        """初始化向量数据库连接器。"""
        index_dir = self.k_module_config.get('index_dir')
        logger.info(f"正在初始化FAISS连接器，指向目录: {index_dir}")
        return FAISSConnector(index_dir=index_dir)

    def _init_k_module(self) -> KnowledgeDiscoverer:
        """初始化知识发现器 (K-Module)。"""
        logger.info("正在初始化K模块...")
        graph_path = self.k_module_config.get('weights_file')
        
        # 将共振参数直接从k_module_config传递
        resonance_config = self.k_module_config
        
        return KnowledgeDiscoverer(
            model=self._model,
            connector=self._connector,
            graph_path=graph_path,
            config=resonance_config
        )

    def _init_s_module(self) -> PromptAssembler:
        """初始化提示装配器 (S-Module)。"""
        logger.info("正在初始化S模块 (提示装配器)...")
        
        # 加载知识清单 (Manifest)
        manifest_path = self.s_module_config['manifest_path']
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

        return PromptAssembler(
            templates_dir=self.s_module_config['templates_dir'],
            manifest=manifest
        )

    def query(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        执行一个完整的查询->检索->合成流程。
        """
        logger.info(f"收到查询: '{query_text}'")
        
        # 从kwargs或配置中获取top_k
        top_k = kwargs.get('top_k', self.k_module_config.get('top_k', 20))

        # --- 阶段1: S模块分析查询 (当前为直通模式) ---
        instruction = self.s_module.generate_instruction(query_text)

        # --- 阶段2: K模块执行共振检索 ---
        logger.info(f"K模块正在检索知识包，指令: {instruction.mode}")
        resonance_packet = self.k_module.discover(
            instruction=instruction,
            top_k=top_k
        )
        logger.info(f"🧠 K模块返回共振包: "
                    f"{len(resonance_packet.primary_atoms)} 个主知识, "
                    f"{len(resonance_packet.context_atoms)} 个上下文, "
                    f"{len(resonance_packet.emerged_concepts)} 个概念。")

        if not any([resonance_packet.primary_atoms, resonance_packet.context_atoms, resonance_packet.emerged_concepts]):
            logger.warning("K模块未返回任何结果。")
            # 使用S模块的回退机制生成答案
            final_answer = self.s_module.assemble_prompt(resonance_packet, query_text)
            return {"answer": final_answer, "knowledge_packet": resonance_packet.to_dict()}

        # --- 阶段3: S模块基于ResonancePacket装配最终答案 ---
        final_answer = self.s_module.assemble_prompt(resonance_packet, query_text)
        logger.info("✅ 最终答案已装配完成。")
        
        packet_dict = resonance_packet.to_dict()

        return {"answer": final_answer, "knowledge_packet": packet_dict}
    
    def get_system_status(self) -> Dict[str, Any]:
        """返回系统的当前状态字典。"""
        status = {
            "k_module_status": {
                "model_name": self.k_module.model_name,
                "index_built": self.k_module.index is not None,
                "index_size": self.k_module.index.ntotal if self.k_module.index else 0,
                "relevance_threshold": self.k_module.relevance_threshold,
                "resonance_params": {
                    "alpha": self.k_module.alpha,
                    "beta": self.k_module.beta,
                    "gamma": self.k_module.gamma,
                    "final_score_threshold": self.k_module.final_score_threshold,
                    "sc_weights": self.k_module.sc_weights
                }
            },
            "s_module_status": {
                "templates": self.s_module.list_templates()
            },
            "system_ready": self.k_module.index is not None
        }
        return status
    
    def rebuild_knowledge_index(self, force_rebuild: bool = True):
        # ... (此方法现在应提示用户运行外部脚本) ...
        logger.warning("警告: 'rebuild_knowledge_index' 已被弃用。")
        logger.warning("请在命令行运行 'scripts/build_extended_index.py' 脚本来重建索引。")
        pass
    
    def add_custom_template(self, name: str, template_content: str):
        """
        添加自定义S模块模板
        
        Args:
            name: 模板名称
            template_content: 模板内容
        """
        self.s_module.add_custom_template(name, template_content)
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