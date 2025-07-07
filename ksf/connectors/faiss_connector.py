from .base_connector import VectorDBConnector
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
import os
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FAISSConnector(VectorDBConnector):
    """
    使用本地FAISS文件作为向量存储的连接器。
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []

    def connect(self, **kwargs):
        """
        从指定目录加载FAISS索引和元数据文件。
        """
        index_path = str(self.index_dir / "faiss.index")
        meta_path = self.index_dir / "index_meta.json"

        if not os.path.exists(index_path):
            logger.error(f"FAISS索引文件未找到: {index_path}")
            raise FileNotFoundError(f"FAISS索引文件未找到: {index_path}")
        
        if not os.path.exists(meta_path):
            logger.error(f"元数据文件未找到: {meta_path}")
            raise FileNotFoundError(f"元数据文件未找到: {meta_path}")

        try:
            logger.info(f"正在加载FAISS索引: {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"✓ FAISS索引已加载。条目数: {self.index.ntotal}")
            
            logger.info(f"正在加载元数据: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
            logger.info(f"✓ 成功加载 {len(self._metadata)} 条元数据。")

        except Exception as e:
            logger.error(f"加载FAISS索引或元数据时出错: {e}")
            self.index = None
            self._metadata = []
            raise e

    def search(self, query_vector: np.ndarray, top_k: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_connected:
            raise ConnectionError("FAISS索引未加载，无法执行搜索。")
        
        # FAISS需要一个二维数组作为输入
        if query_vector.ndim == 1:
            query_vector = np.expand_dims(query_vector, axis=0)
            
        return self.index.search(query_vector.astype('float32'), top_k)

    def reconstruct(self, doc_id: Any) -> Optional[np.ndarray]:
        if not self.is_connected:
            raise ConnectionError("FAISS索引未加载，无法重建向量。")
        try:
            return self.index.reconstruct(int(doc_id))
        except Exception as e:
            logger.warning(f"使用FAISS重建向量 {doc_id} 时出错: {e}")
            return None
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    @property
    def is_connected(self) -> bool:
        return self.index is not None and len(self._metadata) > 0

    @property
    def count(self) -> int:
        return self.index.ntotal if self.index else 0 