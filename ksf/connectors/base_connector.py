from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class VectorDBConnector(ABC):
    """
    向量数据库连接器的抽象基类 (ABC)。
    定义了所有具体连接器（如FAISS, Milvus）必须实现的通用接口。
    """

    @abstractmethod
    def connect(self, **kwargs):
        """
        连接到数据库或加载索引文件。
        """
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        在向量数据库中执行相似性搜索。

        Args:
            query_vector (np.ndarray): 查询向量。
            top_k (int): 要返回的最相似结果的数量。

        Returns:
            tuple[np.ndarray, np.ndarray]: 一个包含距离和索引的元组。
        """
        pass

    @abstractmethod
    def reconstruct(self, doc_id: Any) -> Optional[np.ndarray]:
        """
        根据文档ID重建或获取其向量。
        对于不支持此操作的库，可以返回None。

        Args:
            doc_id (Any): 文档的唯一标识符。

        Returns:
            Optional[np.ndarray]: 重建的向量，如果不支持则返回None。
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        获取索引中所有条目的元数据。

        Returns:
            List[Dict[str, Any]]: 元数据列表。
        """
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """
        检查当前是否已连接到数据库或索引已加载。
        """
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """
        返回索引中的向量总数。
        """
        pass 