"""
K模块训练代码
整合对比学习预训练和领域适配微调
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveLearningDataset(Dataset):
    """
    对比学习数据集
    用于K模块的预训练，构建通用的语义地图
    """
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            max_samples: 最大样本数量（用于调试）
        """
        self.data = self._load_data(data_path)
        if max_samples:
            self.data = self.data[:max_samples]
    
    def _load_data(self, data_path: str) -> List[InputExample]:
        """
        加载训练数据
        支持多种数据格式
        """
        examples = []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # 处理问答对格式
                        if 'question' in item and 'answer' in item:
                            examples.append(InputExample(
                                texts=[item['question'], item['answer']],
                                label=1.0  # 正样本
                            ))
                        # 处理查询-知识对格式
                        elif 'query' in item and 'knowledge' in item:
                            examples.append(InputExample(
                                texts=[item['query'], item['knowledge']],
                                label=1.0
                            ))
                        # 处理内容对格式
                        elif 'text1' in item and 'text2' in item:
                            label = item.get('label', 1.0)
                            examples.append(InputExample(
                                texts=[item['text1'], item['text2']],
                                label=float(label)
                            ))
            
            logger.info(f"✓ 加载了 {len(examples)} 个训练样本")
            
        except Exception as e:
            logger.error(f"✗ 加载数据失败: {e}")
            examples = []
        
        return examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class KModuleTrainer:
    """
    K模块训练器
    负责预训练和微调
    """
    
    def __init__(self, 
                 model_name: str = "all-mpnet-base-v2",
                 output_dir: str = "./checkpoints/k_module",
                 max_seq_length: int = 512):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            max_seq_length: 最大序列长度
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_seq_length = max_seq_length
        
        # 初始化模型
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        
        logger.info(f"✓ 初始化K模块训练器，基础模型: {model_name}")
    
    def pretrain_contrastive(self, 
                           train_data_path: str,
                           eval_data_path: Optional[str] = None,
                           batch_size: int = 16,
                           epochs: int = 3,
                           learning_rate: float = 2e-5,
                           warmup_steps: int = 1000,
                           max_samples: Optional[int] = None):
        """
        对比学习预训练
        构建通用的语义地图
        
        Args:
            train_data_path: 训练数据路径
            eval_data_path: 评估数据路径
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
            warmup_steps: 预热步数
            max_samples: 最大样本数（用于调试）
        """
        logger.info("🚀 开始对比学习预训练...")
        
        # 加载数据
        train_dataset = ContrastiveLearningDataset(train_data_path, max_samples)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        
        # 设置损失函数
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # 设置评估器
        evaluator = None
        if eval_data_path and Path(eval_data_path).exists():
            eval_dataset = ContrastiveLearningDataset(eval_data_path, max_samples)
            if len(eval_dataset) > 0:
                # 构建评估样本
                eval_examples = []
                for example in eval_dataset.data[:100]:  # 限制评估样本数量
                    eval_examples.append(InputExample(
                        texts=example.texts,
                        label=example.label
                    ))
                
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    eval_examples, 
                    name='eval'
                )
        
        # 训练
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=str(self.output_dir / "pretrained"),
            save_best_model=True,
            optimizer_params={'lr': learning_rate}
        )
        
        logger.info("✓ 对比学习预训练完成")
    
    def finetune_domain_adaptation(self,
                                 knowledge_base_path: str,
                                 batch_size: int = 8,
                                 epochs: int = 2,
                                 learning_rate: float = 1e-5):
        """
        领域适配微调
        强化特定领域知识在语义地图中的连接权重
        
        Args:
            knowledge_base_path: 领域知识库路径
            batch_size: 批次大小
            epochs: 训练轮数
            learning_rate: 学习率
        """
        logger.info("🎯 开始领域适配微调...")
        
        # 加载预训练模型
        pretrained_path = self.output_dir / "pretrained"
        if pretrained_path.exists():
            self.model = SentenceTransformer(str(pretrained_path))
            logger.info(f"✓ 加载预训练模型: {pretrained_path}")
        else:
            logger.warning("⚠️ 预训练模型不存在，使用基础模型进行微调")
        
        # 从知识库构建训练数据
        train_examples = self._create_domain_training_data(knowledge_base_path)
        
        if not train_examples:
            logger.error("✗ 无法从知识库创建训练数据")
            return
        
        # 创建数据加载器
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # 设置损失函数
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # 微调
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=str(self.output_dir / "domain_adapted"),
            save_best_model=True,
            optimizer_params={'lr': learning_rate}
        )
        
        logger.info("✓ 领域适配微调完成")
    
    def _create_domain_training_data(self, knowledge_base_path: str) -> List[InputExample]:
        """
        从知识库创建领域训练数据
        生成知识项之间的相关性对
        """
        examples = []
        
        try:
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理不同格式的知识库
            knowledge_items = []
            if isinstance(data, list):
                knowledge_items = data
            elif isinstance(data, dict):
                if 'data' in data:
                    knowledge_items = data['data']
                elif 'knowledge' in data:
                    knowledge_items = data['knowledge']
            
            # 为每个知识项创建训练样本
            for item in knowledge_items:
                if isinstance(item, dict):
                    # 如果有问答对，创建正样本
                    if 'question' in item and 'answer' in item:
                        examples.append(InputExample(
                            texts=[item['question'], item['answer']],
                            label=1.0
                        ))
                    
                    # 如果有内容字段，创建自相关样本
                    content = item.get('content') or item.get('text', '')
                    if content:
                        # 将内容分割成句子，创建句子间的相关性
                        sentences = content.split('。')
                        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                        
                        # 创建相邻句子的正样本对
                        for i in range(len(sentences) - 1):
                            examples.append(InputExample(
                                texts=[sentences[i], sentences[i + 1]],
                                label=0.8  # 中等相关性
                            ))
            
            logger.info(f"✓ 从知识库创建了 {len(examples)} 个训练样本")
            
        except Exception as e:
            logger.error(f"✗ 创建领域训练数据失败: {e}")
        
        return examples
    
    def save_model(self, save_path: str):
        """保存训练好的模型"""
        self.model.save(save_path)
        logger.info(f"✓ 模型已保存到: {save_path}")
    
    def load_model(self, model_path: str):
        """加载训练好的模型"""
        self.model = SentenceTransformer(model_path)
        logger.info(f"✓ 模型已加载: {model_path}")


def main():
    """
    主训练函数
    演示如何使用K模块训练器
    """
    # 初始化训练器
    trainer = KModuleTrainer(
        model_name="all-mpnet-base-v2",
        output_dir="./checkpoints/k_module"
    )
    
    # 阶段1: 对比学习预训练
    # 注意：这里需要通用的对比学习数据
    # trainer.pretrain_contrastive(
    #     train_data_path="./data/contrastive_train.json",
    #     eval_data_path="./data/contrastive_eval.json",
    #     batch_size=16,
    #     epochs=3
    # )
    
    # 阶段2: 领域适配微调
    trainer.finetune_domain_adaptation(
        knowledge_base_path="./data/云和文旅知识库数据集.json",
        batch_size=8,
        epochs=2
    )
    
    # 保存最终模型
    trainer.save_model("./checkpoints/k_module/final")


if __name__ == "__main__":
    main() 