"""
K模块核心实现：知识发现器
KSF 4.x: 实现了共振分离模型
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import os
import logging

from .data_structures import RerankedItem, RetrievalInstruction, ResonancePacket, EmergedConcept
from ..utils.data_utils import load_knowledge_base_from_file, load_knowledge_weights
from ..connectors.base_connector import VectorDBConnector

# 日志设置
logger = logging.getLogger(__name__)

DEFAULT_SC_WEIGHTS = {
    "primary_atom": 1.0,
    "context_atom": 0.8,
    "concept": 0.0
}

class KnowledgeDiscoverer:
    """
    从一个统一的、多源的语义空间中发现和检索知识。
    实现了"三层知识共振模型"，能够并轨处理主知识、上下文知识和概念词汇。
    """
    def __init__(self, 
                 model: SentenceTransformer,
                 connector: VectorDBConnector,
                 graph_path: str,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化KnowledgeDiscoverer。

        Args:
            model (SentenceTransformer): 用于编码查询的句子转换器模型。
            connector (VectorDBConnector): 用于与向量数据库交互的连接器实例。
            graph_path (str): 知识图谱数据文件的路径。
            config (Optional[Dict[str, Any]], optional): 包含超参数的配置字典。
        """
        self.model = model
        self.connector = connector
        self.config = config or {}
        self.graph_path = graph_path
        
        self.node_centrality: Dict[str, float] = {}
        
        self._load_dependencies()

    def _load_dependencies(self):
        """
        加载所有必要的依赖项，如连接到数据库和加载图数据。
        """
        try:
            if not self.connector.is_connected:
                logger.info("连接到向量数据库...")
                self.connector.connect()
                logger.info("✓ 向量数据库已连接。")

            logger.info(f"加载知识权重: {self.graph_path}")
            self.node_centrality = load_knowledge_weights(self.graph_path)
            logger.info("✓ 知识权重已加载。")

        except Exception as e:
            logger.error(f"初始化KnowledgeDiscoverer依赖时出错: {e}")
            raise

    def discover(self, instruction: RetrievalInstruction, top_k: int = 10) -> ResonancePacket:
        """在统一语义空间中执行"三层知识共振"检索。"""
        logger.info(f"K模块收到指令: mode='{instruction.mode}', query='{instruction.query_text}'")
        if instruction.entities:
            logger.info(f"  - 指令中包含实体: {[e.get('text') for e in instruction.entities]}")

        if not self.connector.is_connected:
            logger.error("向量数据库未连接，无法执行检索。")
            return ResonancePacket()

        query_embedding = self.model.encode(instruction.query_text, convert_to_numpy=True, normalize_embeddings=True)
        
        # 1. 在统一索引中进行大规模初步搜索
        # We search for more candidates initially to have a richer pool for reranking.
        candidate_count = top_k * 5 
        distances, indices = self.connector.search(query_embedding, top_k=candidate_count)
        
        # 2. 对所有候选项进行初步处理和分拣
        initial_candidates = []
        all_metadata = self.connector.get_metadata()
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: continue

            # FAISS returns squared L2 distances. Convert to cosine similarity.
            # cos_sim = 1 - (dist^2 / 2)
            dist = distances[0][i]
            s_q = 1 - (dist**2 / 2) 

            # Discard irrelevant results early.
            if s_q < self.config.get('relevance_threshold', 0.15): continue
            
            initial_candidates.append({"idx": idx, "meta": all_metadata[idx], "s_q": s_q})

        # 3. 对所有候选项应用统一的"三层共振"算法
        scored_candidates = []

        alpha = self.config.get('alpha', 0.6)
        beta = self.config.get('beta', 0.4)
        gamma = self.config.get('gamma', 0.1)
        delta = self.config.get('delta', 0.2)
        sc_weights = self.config.get('sc_weights', DEFAULT_SC_WEIGHTS)
        
        entities_text = {e['text'] for e in instruction.entities} if instruction.entities else set()

        for cand in initial_candidates:
            meta = cand['meta']
            item_type = meta['type']
            item_id = meta['id']
            content = meta['content']
            
            # S_c: Source Confidence Score
            s_c = sc_weights.get(item_type, 0.0)
            
            # S_s: Structural Significance Score (PageRank)
            s_s = self.node_centrality.get(str(item_id), 0.0) # Ensure key is string

            # S_e: Entity Relevance Score
            s_e = 0.0
            if entities_text and any(entity in content for entity in entities_text):
                s_e = 1.0
            
            # Final Score Calculation
            final_score = (alpha * cand['s_q']) + (beta * s_s) + (gamma * s_c) + (delta * s_e)
            
            if final_score >= self.config.get('final_score_threshold', 0.35):
                scored_candidates.append({
                    "meta": meta, 
                    "final_score": final_score, 
                    "s_q": cand['s_q'], 
                    "s_s": s_s,
                    "s_e": s_e
                })
            
        # 4. 根据最终得分进行排序和分离
        scored_candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        packet = ResonancePacket()
        
        # Populate the packet, ensuring not to exceed top_k for each category
        for cand in scored_candidates:
            meta = cand['meta']
            item_type = meta['type']
            
            if item_type == 'primary_atom' and len(packet.primary_atoms) < top_k:
                packet.primary_atoms.append(RerankedItem(
                    id=meta['id'], content=meta['content'], final_score=cand['final_score'],
                    original_similarity=cand['s_q'], pagerank_weight=cand['s_s']
                ))
            elif item_type == 'context_atom' and len(packet.context_atoms) < top_k:
                 packet.context_atoms.append(RerankedItem(
                    id=meta['id'], content=meta['content'], final_score=cand['final_score'],
                    original_similarity=cand['s_q'], pagerank_weight=cand['s_s']
                ))
            elif item_type == 'concept' and len(packet.emerged_concepts) < top_k:
                packet.emerged_concepts.append(EmergedConcept(
                    concept=meta['content'], score=cand['final_score']
                ))

        logger.info(f"三层知识共振完成。发现 {len(packet.primary_atoms)} 个主知识, "
                    f"{len(packet.context_atoms)} 个上下文知识, {len(packet.emerged_concepts)} 个概念。")
        
        return packet

    def _get_embedding(self, texts: List[str]) -> np.ndarray:
        """为文本列表生成嵌入。"""
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def load_or_build_index(self, force_rebuild: bool = False):
        """加载FAISS索引。注意：构建过程已移至build_extended_index.py脚本。"""
        index_path_str = str(self.graph_path)
        
        if os.path.exists(index_path_str) and not force_rebuild:
            logger.info(f"正在从 {index_path_str} 加载现有的知识图谱")
            # This method is no longer used in the new implementation
            self.connector = None
            self.node_centrality = {}
            logger.info("知识图谱已加载。")
        else:
            logger.error(f"错误: 在 {index_path_str} 找不到知识图谱文件。")
            logger.error("请先运行 'scripts/build_extended_index.py' 脚本来构建知识图谱。")
            self.connector = None
            self.node_centrality = {} 