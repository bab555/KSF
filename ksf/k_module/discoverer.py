"""
K模块核心实现：知识发现器
负责从知识库中检索直接知识，并发现隐藏关联
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import os
from datetime import datetime
from sentence_transformers.util import cos_sim
import jieba
from peft import PeftModel
import torch
import logging

from .data_structures import KnowledgeItem, HiddenAssociation, KnowledgePacket, RerankedItem, RetrievalInstruction
from ..utils.data_utils import load_knowledge_base_from_file

# Constants
DEFAULT_MODEL = "Snowflake/snowflake-arctic-embed-m" 
# DEFAULT_MODEL = "all-mpnet-base-v2" # Old model
DEFAULT_INDEX_DIR = "checkpoints/k_module_index"
INDEX_FILENAME = "faiss.index"

# Get a logger
logger = logging.getLogger(__name__)

class KnowledgeDiscoverer:
    """
    Discovers and retrieves knowledge from the knowledge base.
    Includes loading the knowledge base, building/loading a FAISS index,
    performing similarity searches, and applying a reranking algorithm.
    """
    def __init__(self, model_name: str, index_dir: str, knowledge_file: str, weights_file: str, adapter_path: str = None, relevance_threshold: float = 0.25, rerank_alpha: float = 0.3):
        self.logger = logging.getLogger(self.__class__.__name__)
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
        
        self.index = None # Will be loaded or built later
        
        self.logger.info("KnowledgeDiscoverer initialized.")

    def _load_model(self):
        self.logger.info(f"Initializing model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        if self.adapter_path and os.path.exists(self.adapter_path):
            try:
                model.load_adapter(self.adapter_path)
                self.logger.info(f"✓ Successfully loaded PEFT adapter from {self.adapter_path}.")
            except Exception as e:
                self.logger.error(f"Failed to load adapter: {e}")
        return model

    def _load_and_process_knowledge(self):
        self.logger.info(f"Loading and processing knowledge from: {self.knowledge_file}")
        knowledge_base = load_knowledge_base_from_file(self.knowledge_file)
        knowledge_base_dict = {item['id']: item for item in knowledge_base}
        doc_ids = [item['id'] for item in knowledge_base]
        self.logger.info(f"✓ Knowledge processed. Total items: {len(doc_ids)}")
        return knowledge_base, knowledge_base_dict, doc_ids

    def _load_knowledge_weights(self, weights_file: str) -> Dict[str, float]:
        """Loads knowledge weights from a JSON file."""
        self.logger.info(f"Loading knowledge weights from {weights_file}...")
        try:
            with open(weights_file, "r", encoding="utf-8") as f:
                weights_data = json.load(f)
            
            # Ensure weights are floats
            processed_weights = {str(k): float(v.get('weight', 0.0)) for k, v in weights_data.items()}
            self.logger.info(f"✓ Successfully loaded and processed {len(processed_weights)} knowledge weights.")
            return processed_weights
        except FileNotFoundError:
            self.logger.warning(f"⚠️ Knowledge weights file not found at {weights_file}. Reranking will not be effective.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading or processing weights file {weights_file}: {e}")
            return {}

    def retrieve_direct_knowledge(self, instruction: RetrievalInstruction, top_k: int = 5) -> List[RerankedItem]:
        """
        Retrieves knowledge based on the instruction from the S-Module.
        """
        self.logger.info(f"K-Module received instruction: {instruction.mode}")

        if instruction.mode == 'SEMANTIC':
            return self._retrieve_semantic(instruction.query_text, top_k)
        elif instruction.mode == 'FILTER_BASED':
            return self._retrieve_by_filter(instruction.filters, top_k)
        else:
            self.logger.warning(f"Unknown retrieval mode: {instruction.mode}. Defaulting to semantic search.")
            return self._retrieve_semantic(instruction.query_text, top_k)

    def _retrieve_by_filter(self, filters: Dict[str, Any], top_k: int) -> List[RerankedItem]:
        """
        (New) Retrieves knowledge by filtering metadata, bypassing vector search.
        This is a placeholder for a more robust metadata search implementation.
        """
        self.logger.info(f"Performing filter-based retrieval with filters: {filters}")
        
        # This is a very basic implementation. A real one would need a proper search over metadata.
        # For now, we'll just check if any of the entity texts appear in the content.
        
        required_entities = set(filters.get("entities", []))
        if not required_entities:
            return []
            
        self.logger.info(f"Required entities for filtering: {required_entities}")

        results = []
        for doc_id, content in self.knowledge_base_dict.items():
            if all(entity in content for entity in required_entities):
                # Since there's no semantic score, we assign a perfect score.
                # The PageRank weight will still apply during reranking.
                item = KnowledgeItem(id=doc_id, content=content, metadata={}, score=1.0)
                results.append(item)
        
        self.logger.info(f"Found {len(results)} items matching filter criteria before reranking.")
        
        reranked_results = self._rerank_results(results, top_k, is_semantic_search=False)
        return reranked_results[:top_k]

    def _retrieve_semantic(self, query_text: str, top_k: int) -> List[RerankedItem]:
        """
        (Original logic) Performs semantic search using vector similarity.
        """
        self.logger.info(f"Performing semantic retrieval for query: '{query_text}'")
        if not self.index or self.model is None:
            self.logger.error("FAISS index or model not loaded.")
            return []

        query_embedding = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), top_k * 2)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            similarity = 1 - (dist / 2) # Assuming index is normalized (L2 distance to similarity)

            if similarity < self.relevance_threshold:
                self.logger.debug(f"Item {idx} below relevance threshold ({similarity:.2f} < {self.relevance_threshold}). Skipping.")
                continue

            doc_id = self.doc_ids[idx]
            content = self.knowledge_base_dict[doc_id]['content']
            if content:
                item = KnowledgeItem(id=doc_id, content=content, metadata={}, score=similarity)
                results.append(item)
        
        # Now, `results` is a list of KnowledgeItem objects.
        # We need to convert it to a list of (id, score) tuples for reranking.
        results_for_reranking = [(item.id, item.score) for item in results]
        
        reranked_results = self._rerank_results(results_for_reranking, top_k, is_semantic_search=True)
        return reranked_results
        
    def _rerank_results(self, results: List[tuple[str, float]], top_k: int, is_semantic_search: bool = True) -> List[RerankedItem]:
        """Reranks results using PageRank weights."""
        self.logger.info(f"Reranking {len(results)} results with alpha={self.rerank_alpha}...")
        reranked_results = []

        for doc_id, score in results:
            weight = self.knowledge_weights.get(doc_id, 0.0)
            
            similarity = (1 - score) if is_semantic_search else 1.0

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
        """Generates embeddings for a list of texts."""
        return self.model.encode(texts, normalize_embeddings=True)

    def load_or_build_index(self, force_rebuild: bool = False):
        """Builds or loads the FAISS index for the knowledge base."""
        index_path_str = str(self.index_dir / INDEX_FILENAME) # Convert Path to string
        
        if os.path.exists(index_path_str) and not force_rebuild:
            self.logger.info(f"Loading existing FAISS index from {index_path_str}")
            self.index = faiss.read_index(index_path_str)
            self.logger.info(f"✓ FAISS index loaded. Items: {self.index.ntotal}")
        else:
            self.logger.info(f"Start encoding knowledge base... ({len(self.knowledge_base)} items)")
            embeddings = self._get_embedding([item['content'] for item in self.knowledge_base])
            self.logger.info(f"✓ Knowledge base encoded. Total items: {len(embeddings)}")
            
            d = embeddings.shape[1]
            self.logger.info(f"Building FAISS index with dimension {d}...")
            self.index = faiss.IndexFlatIP(d)
            self.index.add(embeddings.astype('float32'))
            
            os.makedirs(self.index_dir, exist_ok=True)
            faiss.write_index(self.index, index_path_str)
            self.logger.info(f"✓ FAISS index built and saved to {index_path_str}")

    def find_hidden_associations(self, direct_knowledge: List[RerankedItem], top_k: int = 3) -> List[HiddenAssociation]:
        """Finds hidden associations based on the initial retrieved knowledge."""
        print(f"Finding hidden associations...")
        associations = []
        if not direct_knowledge:
            return associations

        # Use the embedding of the top direct knowledge item as a seed
        seed_embedding = self._get_embedding([direct_knowledge[0].content])[0]
        
        # Find items semantically close to the seed, excluding the direct results
        direct_ids = [item.id for item in direct_knowledge]
        
        # Perform a broader search to find potential associations
        distances, indices = self.index.search(np.array([seed_embedding]).astype('float32'), top_k + len(direct_ids))
        
        similarities = (2 - distances**2) / 2
        
        for i in range(len(indices[0])):
            assoc_id = indices[0][i]
            if assoc_id not in direct_ids:
                association = HiddenAssociation(
                    related_concept=f"Related to '{direct_knowledge[0].content[:30]}...'",
                    source_id=direct_knowledge[0].id,
                    associated_knowledge_id=assoc_id,
                    associated_knowledge_content=self.knowledge_base_dict[assoc_id]['content'],
                    score=similarities[0][i]
                )
                associations.append(association)
                if len(associations) >= top_k:
                    break
                    
        print(f"✓ Found {len(associations)} hidden associations.")
        return associations

    def discover_hidden_associations(self, direct_knowledge: List[KnowledgeItem], 
                                   query: str, max_associations: int = 3) -> List[HiddenAssociation]:
        """
        发现隐藏关联
        基于直接知识和语义地图，发现相关的概念和关联
        
        Args:
            direct_knowledge: 直接检索到的知识项
            query: 原始查询
            max_associations: 最大关联数量
            
        Returns:
            隐藏关联列表
        """
        # 这是一个简化的实现，实际应用中需要更复杂的关联发现算法
        associations = []
        
        # 从直接知识中提取关键概念
        key_concepts = self._extract_key_concepts(direct_knowledge)
        
        # 对每个关键概念，寻找语义相似的其他概念
        for concept in key_concepts[:max_associations]:
            # 使用概念作为查询，寻找相关知识
            related_items = self.retrieve_direct_knowledge(concept, top_k=3)
            
            # 过滤掉已经在直接知识中的项
            new_items = [item for item in related_items 
                        if not any(item.content == dk.content for dk in direct_knowledge)]
            
            if new_items:
                best_item = new_items[0]
                association = HiddenAssociation(
                    related_concept=concept,
                    association_type="semantic",
                    strength=best_item.score * 0.8,  # 降低权重
                    explanation=f"通过语义关联发现：{concept} 与 {best_item.content[:50]}... 相关",
                    metadata={'source_concept': concept, 'related_content': best_item.content}
                )
                associations.append(association)
        
        return associations
    
    def _extract_key_concepts(self, knowledge_items: List[KnowledgeItem]) -> List[str]:
        """
        从知识项中提取关键概念
        简化实现：提取常见的名词和专业术语
        """
        concepts = []
        
        for item in knowledge_items:
            content = item.content
            # 简单的关键词提取（实际应用中应使用更复杂的NLP技术）
            words = content.split()
            
            # 提取长度大于2的词作为潜在概念
            for word in words:
                if len(word) > 2 and word not in ['的', '是', '在', '有', '可以', '能够', '进行', '使用']:
                    concepts.append(word)
        
        # 去重并返回前几个概念
        return list(set(concepts))[:5]
    
    def process_query(self, query: str, top_k: int = 5, 
                     enable_associations: bool = True) -> KnowledgePacket:
        """
        处理用户查询，生成知识包
        
        Args:
            query: 用户查询
            top_k: 检索的知识项数量
            enable_associations: 是否启用关联发现
            
        Returns:
            结构化的知识包
        """
        print(f"🔍 处理查询: {query}")
        
        # 1. 检索直接知识
        direct_knowledge = self.retrieve_direct_knowledge(query, top_k)
        print(f"✓ 检索到 {len(direct_knowledge)} 条直接知识")
        
        # 2. 发现隐藏关联
        hidden_associations = []
        if enable_associations and direct_knowledge:
            hidden_associations = self.discover_hidden_associations(direct_knowledge, query)
            print(f"✓ 发现 {len(hidden_associations)} 个隐藏关联")
        
        # 3. 计算注意力权重
        attention_weights = {}
        for i, item in enumerate(direct_knowledge):
            attention_weights[f"direct_knowledge_{i}"] = item.score
        
        for i, assoc in enumerate(hidden_associations):
            attention_weights[f"hidden_association_{i}"] = assoc.strength
        
        # 4. 构建知识包
        knowledge_packet = KnowledgePacket(
            query=query,
            direct_knowledge=direct_knowledge,
            hidden_associations=hidden_associations,
            attention_weights=attention_weights,
            processing_metadata={
                'model_name': self.model_name,
                'knowledge_base_size': len(self.knowledge_base),
                'top_k': top_k,
                'associations_enabled': enable_associations,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        return knowledge_packet 

    def rerank_results(self, results: List[KnowledgeItem], alpha: float = 0.5) -> List[RerankedItem]:
        """
        Reranks the given list of results based on the knowledge weights.

        Args:
            results: List of KnowledgeItem objects to be reranked.
            alpha: Weight factor for the reranking algorithm.

        Returns:
            List of RerankedItem objects with reranked results.
        """
        reranked_results = []
        
        for item in results:
            # We use the item's id as the key, which should correspond to its index in the original knowledge base
            pagerank_weight = self.knowledge_weights.get(str(item.id), 0.0)
            
            # Calculate the new score
            new_score = alpha * item.score + (1 - alpha) * pagerank_weight
            
            reranked_results.append(RerankedItem(
                id=item.id,
                content=item.content,
                final_score=new_score,
                original_similarity=item.score,
                pagerank_weight=pagerank_weight
            ))
        
        # Sort by the new final score in descending order
        reranked_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return reranked_results

    def find_hidden_associations(self, direct_knowledge: List[RerankedItem], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        基于已发现的直接知识，寻找隐藏的关联知识。

        Args:
            direct_knowledge: List of RerankedItem objects representing directly retrieved knowledge.
            top_k: Number of hidden associations to find.

        Returns:
            List of dictionaries representing hidden associations.
        """
        # Implementation of the method
        # This is a placeholder and should be implemented based on the specific requirements
        return [] 