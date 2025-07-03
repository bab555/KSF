# Knowledge Synthesized Framework (KSF)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)

**KSF (Knowledge Synthesized Framework)** 是一个创新的双专家架构AI框架，通过知识注入和智能合成实现高效的语言模型增强。

## 🌟 核心特性

### 🏗️ 双专家架构 (S→K→S)
- **SynthesizerConductor (S-Module)**: 中央"大脑"，处理用户查询并主动向K模块请求指导
- **KnowledgeBank (K-Module)**: 被动参数化记忆模块，通过交叉注意力提供知识检索

### 💉 知识注入技术
- **预加载知识**: K模块可预先注入外部知识而非从零学习
- **语义空间对齐**: 使用Qwen3-Embedding-4B确保查询向量与知识向量兼容
- **动态部署**: 支持RAG系统部署，将学习到的原则推广到新的动态检索知识

### 🧠 增强的S模块设计
- **三阶段处理**:
  1. 初始查询处理和K模块查询生成
  2. 总结（"思考"阶段）与专用SummarizerHead
  3. 指导融合回主推理路径
- **辅助损失**: 总结损失训练总结能力

### 🔧 技术优势
- **同质嵌入策略**: 使用Qwen3基础模型本身嵌入知识，确保完美兼容
- **灵活角色**: 可作为阅读理解专家（有外部知识库）或自包含专家（无外部知识库）
- **高效训练**: 支持知识注入模式，减少训练时间

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.7+
- CUDA 支持的GPU（推荐）
- 16GB+ GPU显存（用于Qwen3-4B模型）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 模型准备
确保您有以下模型文件：
- Qwen3基础模型（默认路径：`../qwen3`）
- Qwen3-Embedding-4B模型（默认路径：`../qwen3-embedding/Qwen/Qwen3-Embedding-4B`）

### 知识注入
在训练前，先注入知识到KnowledgeBank：

```bash
# 1. 准备知识文件
# 将您的知识内容放入 data/knowledge_base.txt
# 每行一个知识条目

# 2. 执行知识注入
python scripts/inject_knowledge.py
```

### 训练模型
```bash
# 使用默认配置训练
python train_ksf.py

# 自定义配置
python train_ksf.py --config configs/your_config.yaml
```

### 配置说明
主要配置文件：`configs/ksf_training_config.yaml`

关键配置项：
- `knowledge_injection`: 知识注入设置
- `base_model`: 基础模型配置
- `training`: 训练超参数
- `data`: 数据路径设置

## 📁 项目结构
```
Knowledge-Synthesized-Framework/
├── ksf/                          # 核心框架代码
│   ├── models/                   # 模型定义
│   │   ├── advanced_ksf_model.py      # 主模型
│   │   ├── advanced_knowledge_expert.py  # K模块
│   │   ├── advanced_synthesizer.py    # S模块
│   │   └── base_expert.py         # 专家基类
│   ├── training/                 # 训练相关
│   │   ├── trainer.py            # 训练器
│   │   └── losses.py             # 损失函数
│   └── utils/                    # 工具函数
├── configs/                      # 配置文件
├── data/                         # 数据目录
├── scripts/                      # 脚本
│   └── inject_knowledge.py       # 知识注入脚本
├── checkpoints/                  # 模型检查点
└── logs/                         # 训练日志
```

## 🎯 使用示例

### 基本推理
```python
from ksf.models.advanced_ksf_model import AdvancedKsfModel
import yaml

# 加载配置
with open('configs/ksf_training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化模型（自动加载注入的知识）
model = AdvancedKsfModel(config)

# 推理
query = "什么是机器学习？"
response = model.generate(query)
print(response)
```

### 知识注入流程
```python
# 知识注入已集成到训练流程中
# 1. 准备知识文件 data/knowledge_base.txt
# 2. 运行 python scripts/inject_knowledge.py
# 3. 训练时模型会自动加载注入的知识
```

## 📊 当前状态

✅ **已完成**:
- 双专家架构实现
- 知识注入系统
- 训练管道搭建
- 核心功能验证

🔄 **进行中**:
- 全面性能测试
- 基准评估
- 优化调参

🎯 **计划中**:
- 更多预训练模型支持
- 分布式训练
- Web API接口

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

- **开发者**: 易尘/bab555
- **邮箱**: bab55@163.com
- **GitHub**: [kfs](https://github.com/your-username/kfs) (即将创建)

## 📄 许可证

本项目采用 **CC BY-NC-SA 4.0** 许可证 - 详见 [LICENSE](LICENSE) 文件

⚠️ **重要说明**: 本项目仅供**非商业用途**使用。商业使用请联系开发者获取授权。

---

**开发团队**: 红点天枢 