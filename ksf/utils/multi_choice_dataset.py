"""
多选题数据集类
支持4选1的逻辑推理题目
"""

import torch
from torch.utils.data import Dataset
import json
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MultiChoiceDataset(Dataset):
    """
    多选题数据集
    每个样本包含一个问题、4个选项和正确答案索引(0-3)
    """
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_length: int = 512,
                 mode: str = 'train'):
        """
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            mode: 模式 ('train' 或 'eval')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"加载{mode}数据集: {len(self.data)}条样本")
        
        # 统计数据
        self._analyze_data()
    
    def _analyze_data(self):
        """分析数据分布"""
        type_counts = {}
        answer_counts = [0, 0, 0, 0]
        
        for item in self.data:
            # 统计题目类型
            item_type = item.get('type', 'unknown')
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
            
            # 统计答案分布
            answer = item.get('answer', 0)
            if 0 <= answer < 4:
                answer_counts[answer] += 1
        
        logger.info(f"题目类型分布: {type_counts}")
        logger.info(f"答案分布: {answer_counts}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        
        # 构建输入文本
        # 格式: 问题: [问题] 选项: A. [选项1] B. [选项2] C. [选项3] D. [选项4]
        choice_labels = ['A', 'B', 'C', 'D']
        
        input_text = f"问题: {question}\n选项:\n"
        for i, choice in enumerate(choices):
            input_text += f"{choice_labels[i]}. {choice}\n"
        
        # 对输入文本进行编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 返回数据
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(answer, dtype=torch.long),  # 0-3的标签
            'text': input_text,
            'question': question,
            'choices': choices,
            'answer': answer,
            'type': item.get('type', 'unknown')
        }

class MultiChoiceDatasetWithReasoning(MultiChoiceDataset):
    """
    带推理过程的多选题数据集
    适合P-C-S架构，可以生成推理链
    """
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        question = item['question']
        choices = item['choices']
        answer = item['answer']
        item_type = item.get('type', 'unknown')
        
        # 根据题目类型生成不同的输入格式
        if item_type in ['number_pattern', 'arithmetic']:
            # 数字推理类题目，强调模式识别
            input_text = f"[数字推理任务]\n问题: {question}\n"
            input_text += "请分析数字间的规律，然后选择正确答案。\n选项:\n"
        elif item_type in ['syllogism', 'causal', 'conditional']:
            # 逻辑推理类题目，强调逻辑关系
            input_text = f"[逻辑推理任务]\n问题: {question}\n"
            input_text += "请仔细分析逻辑关系，然后选择正确答案。\n选项:\n"
        elif item_type == 'analogy':
            # 类比推理题目
            input_text = f"[类比推理任务]\n问题: {question}\n"
            input_text += "请找出相似的关系模式，然后选择正确答案。\n选项:\n"
        elif item_type == 'classification':
            # 分类题目
            input_text = f"[分类任务]\n问题: {question}\n"
            input_text += "请识别不同类的项目。\n选项:\n"
        else:
            # 默认格式
            input_text = f"问题: {question}\n选项:\n"
        
        # 添加选项
        choice_labels = ['A', 'B', 'C', 'D']
        for i, choice in enumerate(choices):
            input_text += f"{choice_labels[i]}. {choice}\n"
        
        # 编码
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(answer, dtype=torch.long),
            'text': input_text,
            'question': question,
            'choices': choices,
            'answer': answer,
            'type': item_type
        }

def create_multi_choice_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    mode: str = 'train',
    with_reasoning: bool = False
) -> Dataset:
    """
    创建多选题数据集
    
    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        mode: 模式 ('train' 或 'eval')
        with_reasoning: 是否使用带推理的数据集
    
    Returns:
        dataset: 多选题数据集
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    if with_reasoning:
        dataset = MultiChoiceDatasetWithReasoning(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mode=mode
        )
    else:
        dataset = MultiChoiceDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mode=mode
        )
    
    return dataset

def multi_choice_collate_fn(batch):
    """
    多选题数据批次整理函数
    
    Args:
        batch: 批次数据
    
    Returns:
        整理后的批次数据
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    # 额外的信息（用于分析）
    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'texts': [item['text'] for item in batch],
        'questions': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answers': [item['answer'] for item in batch],
        'types': [item['type'] for item in batch]
    }
    
    return batch_dict 