"""
K模块核心实现：知识发现器
负责从知识库中检索直接知识
"""
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import os
import logging

from .data_structures import KnowledgeItem, RerankedItem, RetrievalInstruction
from ..utils.data_utils import load_knowledge_base_from_file

# 日志设置
logger = logging.getLogger(__name__)

class KnowledgeDiscoverer:
    """
    从知识库中发现和检索知识。
    包括加载知识库、构建/加载FAISS索引、
    执行相似度搜索以及应用重排序算法。
    """
    def __init__(self, model_name: str, index_dir: str, knowledge_file: str, weights_file: str, adapter_path: Optional[str] = None, relevance_threshold: float = 0.25, rerank_alpha: float = 0.3):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.knowledge_file = knowledge_file
        self.weights_file = weights_file
        self.adapter_path = adapter_path
        self.relevance_threshold = relevance_threshold
        self.rerank_alpha = rerank_alpha
        
        self.model = self._load_model()
        self.knowledge_base, self.knowledge_base_dict, self.doc_ids = self._load_and_process_knowledge()
        self.knowledge_weights = self._load_knowledge_weights(weights_file)
        
        self.index: Optional[faiss.Index] = None # 稍后加载或构建
        
        logger.info("知识发现器初始化完成。")

    def _load_model(self) -> SentenceTransformer:
        logger.info(f"正在初始化模型: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        if self.adapter_path and os.path.exists(self.adapter_path):
            try:
                # PeftModel.from_pretrained(model.get_submodule("0"), self.adapter_path) -> This is complex
                # For SentenceTransformer, the PEFT library recommends using the `load_adapter` method if available,
                # or managing it manually if not. Let's assume `load_adapter` is the way.
                model.load_adapter(self.adapter_path)
                logger.info(f"✓ 成功从 {self.adapter_path} 加载PEFT适配器。")
            except Exception as e:
                logger.error(f"加载适配器失败: {e}")
        return model

    def _load_and_process_knowledge(self) -> tuple[List[Dict], Dict, List[str]]:
        logger.info(f"正在从 {self.knowledge_file} 加载和处理知识...")
        knowledge_base = load_knowledge_base_from_file(self.knowledge_file)
        knowledge_base_dict = {item['id']: item for item in knowledge_base}
        doc_ids = [item['id'] for item in knowledge_base]
        logger.info(f"✓ 知识处理完成。总条目: {len(doc_ids)}")
        return knowledge_base, knowledge_base_dict, doc_ids

    def _load_knowledge_weights(self, weights_file: str) -> Dict[str, float]:
        """从JSON文件加载知识权重。"""
        logger.info(f"正在从 {weights_file} 加载知识权重...")
        try:
            with open(weights_file, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
            
            # 确保权重是浮点数
            processed_weights = {str(k): float(v.get('weight', 0.0)) for k, v in weights_data.items()}
            logger.info(f"✓ 成功加载并处理了 {len(processed_weights)} 个知识权重。")
            return processed_weights
        except FileNotFoundError:
            logger.warning(f"⚠️ 在 {weights_file} 未找到知识权重文件。重排序功能将受影响。")
            return {}
        except Exception as e:
            logger.error(f"加载或处理权重文件 {weights_file} 时出错: {e}")
            return {}

    def retrieve_direct_knowledge(self, instruction: RetrievalInstruction, top_k: int = 5) -> List[RerankedItem]:
        """
        根据S模块的指令检索知识。
        """
        logger.info(f"K模块收到指令: {instruction.mode}")

        if instruction.mode == 'SEMANTIC':
            return self._retrieve_semantic(instruction.query_text, top_k)
        elif instruction.mode == 'FILTER_BASED':
            # FILTER_BASED 模式当前是一个占位符，实际需要更强的元数据过滤能力
            return self._retrieve_by_filter(instruction.filters, top_k)
        else:
            logger.warning(f"未知的检索模式: {instruction.mode}。将默认使用语义搜索。")
            return self._retrieve_semantic(instruction.query_text, top_k)

    def _retrieve_by_filter(self, filters: Dict[str, Any], top_k: int) -> List[RerankedItem]:
        """
        通过过滤元数据来检索知识，绕过向量搜索。
        这是一个占位符，需要更强大的元数据搜索实现。
        """
        logger.info(f"正在使用过滤器进行基于过滤的检索: {filters}")
        
        required_entities = set(filters.get("entities", []))
        if not required_entities:
            return []
            
        logger.info(f"需要用于过滤的实体: {required_entities}")

        results = []
        for doc in self.knowledge_base:
            # 基础实现：检查实体是否存在于内容中
            if all(entity in doc['content'] for entity in required_entities):
                # 因为没有语义分数，我们分配一个满分。PageRank权重仍然会在重排时起作用。
                results.append((doc['id'], 1.0))
        
        logger.info(f"在重排前，找到 {len(results)} 个符合过滤条件的条目。")
        
        reranked_results = self._rerank_results(results, top_k, is_semantic_search=False)
        return reranked_results

    def _retrieve_semantic(self, query_text: str, top_k: int) -> List[RerankedItem]:
        """
        使用向量相似性执行语义搜索。
        """
        logger.info(f"正在为查询执行语义检索: '{query_text}'")
        if not self.index or self.model is None:
            logger.error("FAISS索引或模型未加载。")
            return []

        query_embedding = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        # FAISS 使用 L2 距离，对于归一化嵌入，(dist^2 = 2 - 2 * cos_sim)
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k * 2) # 检索更多候选

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            # 将距离转换为余弦相似度 (0-1范围)
            similarity = 1 - (dist**2 / 2)

            if similarity < self.relevance_threshold:
                logger.debug(f"条目 {idx} 低于相关性阈值 ({similarity:.2f} < {self.relevance_threshold})。跳过。")
                continue

            doc_id = self.doc_ids[idx]
            results.append((doc_id, similarity))
        
        reranked_results = self._rerank_results(results, top_k)
        return reranked_results
        
    def _rerank_results(self, results: List[tuple[str, float]], top_k: int, is_semantic_search: bool = True) -> List[RerankedItem]:
        """使用PageRank权重对结果进行重排序。"""
        logger.info(f"正在以 alpha={self.rerank_alpha} 对 {len(results)} 个结果进行重排序...")
        reranked_results = []

        for doc_id, score in results:
            weight = self.knowledge_weights.get(doc_id, 0.0)
            
            # 在非语义搜索的情况下，原始分数为1.0
            similarity = score if is_semantic_search else 1.0

            # 最终分数是结构重要性（PageRank）和上下文相关性（相似度）的加权平均
            final_score = (self.rerank_alpha * weight) + ((1 - self.rerank_alpha) * similarity)
            
            content = self.knowledge_base_dict[doc_id]['content']

            reranked_results.append(RerankedItem(
                id=doc_id,
                content=content,
                final_score=float(final_score),
                original_similarity=float(similarity),
                pagerank_weight=float(weight)
            ))

        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return reranked_results[:top_k]

    def _get_embedding(self, texts: List[str]) -> np.ndarray:
        """为文本列表生成嵌入。"""
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def load_or_build_index(self, force_rebuild: bool = False):
        """构建或加载知识库的FAISS索引。"""
        index_path_str = str(self.index_dir / "faiss.index")
        
        if os.path.exists(index_path_str) and not force_rebuild:
            logger.info(f"正在从 {index_path_str} 加载现有的FAISS索引")
            self.index = faiss.read_index(index_path_str)
            logger.info(f"✓ FAISS索引已加载。条目数: {self.index.ntotal}")
        else:
            logger.info(f"开始编码知识库... ({len(self.knowledge_base)}个条目)")
            # 从知识库中提取内容进行编码
            contents = [item['content'] for item in self.knowledge_base]
            embeddings = self._get_embedding(contents)
            
            logger.info(f"✓ 知识库编码完成。总条目: {len(embeddings)}")
            
            d = embeddings.shape[1]
            logger.info(f"正在构建维度为 {d} 的FAISS索引...")
            # 使用IndexFlatIP，因为我们处理的是归一化的嵌入，内积等于余弦相似度
            self.index = faiss.IndexFlatIP(d)
            self.index.add(embeddings.astype('float32'))
            
            os.makedirs(self.index_dir, exist_ok=True)
            faiss.write_index(self.index, index_path_str)
            logger.info(f"✓ FAISS索引已构建并保存至 {index_path_str}") 