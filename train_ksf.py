#!/usr/bin/env python3
"""
KSF (Knowledge Synthesized Framework) 主训练脚本
使用K-S架构进行端到端训练
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from ksf.training.trainer import KsfTrainer


def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    
    # 创建日志目录
    log_dir = Path(log_config.get('log_dir', './logs/ksf'))
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / 'training.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("KSF训练日志系统已初始化")
    return logger


def validate_config(config: dict) -> bool:
    """验证配置文件的完整性"""
    required_sections = ['base_model', 'model', 'training', 'data', 'loss']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ 配置文件缺少必需的节: {section}")
            return False
    
    # 验证数据文件路径
    data_config = config['data']
    train_file = data_config.get('train_file')
    eval_file = data_config.get('eval_file')
    
    if not train_file or not Path(train_file).exists():
        print(f"❌ 训练数据文件不存在: {train_file}")
        return False
    
    if not eval_file or not Path(eval_file).exists():
        print(f"❌ 验证数据文件不存在: {eval_file}")
        return False
    
    # 验证基础模型路径
    base_model_path = config['base_model'].get('path')
    if not base_model_path or not Path(base_model_path).exists():
        print(f"❌ 基础模型路径不存在: {base_model_path}")
        return False
    
    print("✅ 配置文件验证通过")
    return True


def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({memory:.1f}GB)")
        print(f"✅ GPU数量: {gpu_count}")
    else:
        print("⚠️ 未检测到CUDA，将使用CPU训练")
    
    # 检查内存
    import psutil
    memory_gb = psutil.virtual_memory().total / 1024**3
    print(f"✅ 系统内存: {memory_gb:.1f}GB")
    
    # 检查Python环境
    print(f"✅ Python版本: {sys.version}")
    print(f"✅ PyTorch版本: {torch.__version__}")


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置文件加载成功: {config_path}")
    return config


def create_sample_data(data_dir: Path):
    """创建示例训练数据（如果不存在）"""
    train_file = data_dir / 'ksf_train.jsonl'
    eval_file = data_dir / 'ksf_eval.jsonl'
    
    if train_file.exists() and eval_file.exists():
        return
    
    print("📝 创建示例训练数据...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 示例数据
    import json
    
    sample_data = [
        {
            "query": "什么是人工智能？",
            "knowledge": "人工智能(AI)是指机器执行通常需要人类智能的任务的能力，包括学习、推理、感知、理解语言等。",
            "answer": "人工智能是让机器具备类似人类智能的技术，能够学习、推理和解决问题。"
        },
        {
            "query": "机器学习的基本类型有哪些？",
            "knowledge": "机器学习主要分为监督学习、无监督学习和强化学习三种类型。监督学习使用标记数据训练模型。",
            "answer": "机器学习主要包括监督学习、无监督学习和强化学习三种基本类型。"
        },
        {
            "query": "深度学习与传统机器学习的区别是什么？",
            "knowledge": "深度学习使用多层神经网络来学习数据的复杂模式，而传统机器学习通常使用较简单的算法和手工特征。",
            "answer": "深度学习使用多层神经网络自动学习特征，而传统机器学习依赖手工设计的特征。"
        }
    ]
    
    # 创建训练数据
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in sample_data * 100:  # 重复数据以增加样本量
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 创建验证数据
    with open(eval_file, 'w', encoding='utf-8') as f:
        for item in sample_data * 10:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 示例数据已创建: {train_file}, {eval_file}")


def main():
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Argument parser
    parser = argparse.ArgumentParser(description="Train the KSF model.")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/ksf_training_config.yaml',
        help='Path to the training configuration file.'
    )
    args = parser.parse_args()

    # Initialize and run the trainer
    try:
        trainer = KsfTrainer(config_path=args.config)
        trainer.train()
    except Exception as e:
        logging.error("An error occurred during the training process.", exc_info=True)


if __name__ == '__main__':
    # 设置多进程启动方法
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    main() 