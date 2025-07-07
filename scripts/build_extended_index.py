import yaml
import json
import argparse
from tqdm import tqdm
import numpy as np
import faiss
from pathlib import Path
import os
import logging
from sentence_transformers import SentenceTransformer
import sys

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录添加到sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ksf.utils.data_utils import load_knowledge_base_from_file

class UnifiedIndexBuilder:
    """
    构建并保存统一的FAISS索引及其元数据。
    该索引可以并轨融合三种知识源：
    1. 主知识 (Primary Knowledge): 核心事实，标记为 'primary_atom'。
    2. 上下文知识 (Contextual Knowledge): 背景、原则、注解，标记为 'context_atom'。
    3. 概念词汇 (Concept Vocabulary): 用于意图发现的关键词，标记为 'concept'。
    """
    def __init__(self, config_path: str):
        logger.info(f"正在从 {config_path} 加载配置...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        discoverer_config = self.config['discoverer']
        self.model_name = discoverer_config['model_name']
        self.adapter_path = discoverer_config.get('adapter_path')
        self.index_dir = Path(discoverer_config['index_dir'])
        
        self.model = self._load_model()
        logger.info("统一索引构建器初始化完成。")

    def _load_model(self) -> SentenceTransformer:
        logger.info(f"正在加载嵌入模型: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        if self.adapter_path and os.path.exists(self.adapter_path):
            try:
                model.load_adapter(self.adapter_path)
                logger.info(f"✓ 成功从 {self.adapter_path} 加载PEFT适配器。")
            except Exception as e:
                logger.error(f"加载适配器失败: {e}")
        return model

    def build(self, primary_path, context_path=None, vocabulary_path=None):
        """执行完整的构建流程。"""
        logger.info("🚀 开始构建统一索引...")

        # 1. 加载所有数据源
        logger.info("--- 步骤 1/4: 加载所有数据源 ---")
        primary_data, context_data, vocab_data = self._load_all_sources(
            primary_path, context_path, vocabulary_path
        )

        # 2. 编码所有内容
        logger.info("--- 步骤 2/4: 编码所有文本为向量 ---")
        all_texts = ([item['content'] for item in primary_data] +
                     [item['content'] for item in context_data] +
                     vocab_data)
        
        if not all_texts:
            logger.error("错误: 没有加载任何数据，无法构建索引。")
            return
            
        embeddings = self.model.encode(
            all_texts, 
            normalize_embeddings=True, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        logger.info(f"✓ 所有 {len(all_texts)} 个条目编码完成。向量维度: {embeddings.shape[1]}")
        
        # 3. 构建并保存FAISS索引
        logger.info("--- 步骤 3/4: 构建并保存FAISS索引 ---")
        index = self._build_faiss_index(embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / "faiss.index"
        faiss.write_index(index, str(index_path))
        logger.info(f"✓ FAISS索引已构建并保存至 {index_path}")

        # 4. 构建并保存元数据
        logger.info("--- 步骤 4/4: 构建并保存索引元数据 ---")
        index_meta = self._build_index_meta(primary_data, context_data, vocab_data)
        meta_path = self.index_dir / "index_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(index_meta, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ 索引元数据已构建并保存至 {meta_path}")

        logger.info("✅ 统一索引构建流程全部完成！")

    def _load_all_sources(self, primary_path, context_path, vocabulary_path):
        """加载所有提供的知识源文件。"""
        primary_data = load_knowledge_base_from_file(primary_path)
        logger.info(f"已加载 {len(primary_data)} 条主知识。")

        context_data = []
        if context_path:
            context_data = load_knowledge_base_from_file(context_path)
            logger.info(f"已加载 {len(context_data)} 条上下文知识。")

        vocab_data = []
        if vocabulary_path:
            try:
                with open(vocabulary_path, 'r', encoding='utf-8') as f:
                    vocab_data = [line.strip() for line in f if line.strip()]
                logger.info(f"已加载 {len(vocab_data)} 个概念词汇。")
            except FileNotFoundError:
                logger.warning(f"警告: 未找到词汇表文件 {vocabulary_path}，将忽略。")

        return primary_data, context_data, vocab_data

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        dimension = embeddings.shape[1]
        # 使用IndexFlatIP，因为我们处理的是归一化的嵌入，内积等于余弦相似度
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype('float32'))
        return index

    def _build_index_meta(self, primary_data, context_data, vocab_data):
        """为所有源构建统一的元数据列表，并打上正确的类型标签。"""
        meta = []
        current_index = 0

        # 添加主知识元数据
        for item in primary_data:
            meta.append({
                "index_id": current_index,
                "type": "primary_atom",
                "id": item.get('id', f'primary_{current_index}'),
                "content": item['content']
            })
            current_index += 1
        
        # 添加上下文知识元数据
        for item in context_data:
            meta.append({
                "index_id": current_index,
                "type": "context_atom",
                "id": item.get('id', f'context_{current_index}'),
                "content": item['content']
            })
            current_index += 1

        # 添加概念词汇元数据
        for word in vocab_data:
            meta.append({
                "index_id": current_index,
                "type": "concept",
                "id": f"concept_{current_index}",
                "content": word
            })
            current_index += 1
            
        return meta

def main():
    parser = argparse.ArgumentParser(description="构建统一的、可并轨多种知识源的FAISS索引。")
    parser.add_argument(
        '--config', 
        type=str, 
        default="configs/ksf_config.yaml", 
        help="KSF配置文件路径。"
    )
    parser.add_argument(
        '--primary', 
        type=str, 
        required=True, 
        help="主知识库文件路径 (.jsonl格式)。"
    )
    parser.add_argument(
        '--context', 
        type=str, 
        default=None, 
        help="(可选) 上下文知识库文件路径 (.jsonl格式)。"
    )
    parser.add_argument(
        '--vocabulary', 
        type=str, 
        default=None, 
        help="(可选) 概念词汇文件路径 (.txt格式，每行一个词)。"
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"致命错误: 配置文件 '{args.config}' 不存在。")
        return
    
    builder = UnifiedIndexBuilder(config_path=args.config)
    builder.build(
        primary_path=args.primary,
        context_path=args.context,
        vocabulary_path=args.vocabulary
    )

if __name__ == "__main__":
    main() 