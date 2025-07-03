# Knowledge Synthesized Framework (KSF)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)

**KSF (Knowledge Synthesized Framework)** 是一个创新的双专家架构AI框架，通过知识注入和智能合成实现高效的语言模型增强。

## 🌟 核心特性

### 🏗️ 双专家协同架构 (S→K→S)
- **SynthesizerConductor (S-Module)**: 作为框架的中央"大脑"，负责深度理解和处理用户查询，并主动向K模块查询以获取决策支持。
- **KnowledgeBank (K-Module)**: 一个参数化的被动记忆模块，通过可训练的`memory_matrix`存储知识，并通过交叉注意力机制响应S模块的查询。
- **S→K→S 循环流**: 这是KSF的核心交互模式。S模块将初步处理结果生成查询向量，K模块返回词汇偏置 (`vocab_bias`) 和增强嵌入 (`retrieved_memory_embedding`)，S模块再将这些指导信息融合回自身的推理路径中，实现一种"自我反思和修正"的闭环，持续优化输出。

### 🧠 内化与外联兼备的知识能力
- **参数化知识存储**: KSF最显著的特点之一是其K模块能将知识**学习并内化**到自身的模型参数中。这意味着即使**没有外部知识库**，KSF也能像一个自包含的专家一样，依赖其内部存储的知识进行回答。
- **动态知识处理 (RAG增强)**: 当与外部知识库（如RAG系统）结合使用时，KSF能将其学习到的推理和整合能力，泛化到**动态检索到的新知识**上。它不仅仅是检索，更是对新知识的深度理解和有机融合，极大地强化了知识库的处理能力。
- **知识注入机制 (Knowledge Injection)**: 框架提供了一套完整的知识注入流程。通过`scripts/inject_knowledge.py`脚本，可以使用指定的同源嵌入模型（如`Qwen3-Embedding`）将文本知识库向量化，并将其直接加载到K模块的`memory_matrix`中。这使得模型能以非训练的方式快速获取大量先验知识。

### ✨ 增强的S模块与流式注意力引导
- **三阶段"思考"过程**: S模块的推理过程被设计为三个阶段：**查询处理 -> 内部总结 -> 指导融合**。通过专用的`SummarizerHead`，模型在最终输出前会进行一个"思考总结"的中间步骤，提升了回答的深度和条理性。
- **流式注意力引导 (Flowing Attention Guidance)**: 从K模块获得的指导信息并非一次性的硬性修正，而是通过`guidance_fusion_attention`机制，像一股数据流一样**平滑地融入**S模块后续的每一个推理步骤中，持续、动态地引导和修正生成过程的轨迹。

## 🔧 技术优势

### 🧬 同源嵌入策略 (Homologous Embedding)
- **解决语义鸿沟**: 为了确保S模块的"提问"和K模块的"回答"在同一个语义频道上，KSF提倡采用**同源嵌入 (Homologous Embedding)** 策略。
- **保证高效兼容**: 这意味着应该选择与基础语言模型（S模块）属于**相同技术族系、共享语义空间**的嵌入模型来对知识进行编码。例如，当S模块使用Qwen3系列模型时，知识注入也应采用Qwen3的Embedding模型。这种方法从根源上保证了向量空间的对齐，避免了使用不同源模型时可能出现的语义偏差和需要额外适配器层的复杂性。

### 🎭 灵活的角色与部署
- **自包含专家**: 依赖内部参数化知识库，独立完成推理任务。
- **阅读理解与泛化专家**: 结合RAG，对外部动态知识进行深度加工和理解。

### ⚡ 高效训练与微调
- **知识注入加速**: 通过直接注入，极大缩短了模型在特定领域"冷启动"的学习成本。
- **伪API式冻结训练 (Pseudo-API Freezing)**: 在知识注入后，K模块的权重可以被"冻结"，使其成为一个固定的、仅供查询的"知识API"。S模块在训练中将专注于学习如何高效地查询这个"API"，并理解其返回的指导信息，而不是去改变知识本身。这种策略使得训练过程更稳定、目标更明确。
- **辅助损失监督**: 专为"总结能力"设计的辅助损失函数，帮助S模块更好地学习推理和归纳。

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
- **GitHub**: [ksf](https://github.com/bab555/ksf)

## 📄 许可证

本项目采用 **CC BY-NC-SA 4.0** 许可证 - 详见 [LICENSE](LICENSE) 文件

⚠️ **重要说明**: 本项目仅供**非商业用途**使用。商业使用请联系开发者获取授权。

---

**开发团队**: 红点天枢 