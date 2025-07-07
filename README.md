# 知识合成框架 (Knowledge-Synthesized Framework - KSF) v5.1

[![版本](https://img.shields.io/badge/Version-5.1-blue.svg)](https://github.com/your-repo/KSF)
[![状态](https://img.shields.io/badge/Status-功能增强迭代中-green.svg)](https://github.com/your-repo/KSF)
[![架构](https://img.shields.io/badge/Architecture-四层共振模型-orange.svg)](https://github.com/your-repo/KSF)

**知识合成框架 (KSF)** 是一套先进的本地知识处理系统，其设计目标是超越传统的检索增强生成 (RAG)，实现对复杂、多源知识的深度理解、结构化重组和精准合成，为大型语言模型（LLM）提供高度优化、低幻觉的上下文，同时最大化本地模型的分析能力。

v5.1版本标志着KSF在**架构自洽**和**功能激活**上的重大突破，正式从"三层共振模型"升级为**"四层知识共振模型"**，并全面激活了S模块的分析与内容增强能力，使框架的智能化水平和输出内容的价值得到了质的飞跃。

---

## 核心特性

-   **先进的四层知识共振模型**: 业界领先的混合式排序算法，在一次检索中综合评估四个维度的信息，确保结果既相关又重要。
    -   `α * S_q`: **语义相关度** - 知识与查询的直接相关性。
    -   `β * S_s`: **结构重要性** - 知识在全局知识图谱中的权威性 (PageRank)。
    -   `γ * S_c`: **来源置信度** - 知识的类型（核心事实、上下文、概念）。
    -   `δ * S_e`: **实体引导** - 来自S模块的"提醒"，对包含关键实体的知识进行定向加权。
-   **K-S解耦与动态协作**: K模块（直觉引擎）负责快速、广泛的召回；S模块（推理引擎）负责精准分析和编排。S模块通过可调节的`δ`权重，实现了对K模块的**可控影响**，完美平衡了"精准制导"与"意外发现"。
-   **全自动数据预处理工具链 (`scripts/`)**: 提供一整套强大的数据"脚手架"，将原始数据转化为高质量的知识库。
    -   **歧义自动发现**: 利用向量差异，自动化挖掘并报告潜在的同词不同义问题。
    -   **知识图谱构建**: 自动构建知识图谱并计算PageRank，量化每个知识点的重要性。
    -   **多轨统一索引**: 将"核心知识"、"上下文知识"和"概念词汇"三种异构数据源，统一编码进一个FAISS索引中。
-   **S模块的深度分析与内容增强**:
    -   **意图分析**: 使用零样本分类（Zero-Shot Classification）模型理解用户意图，为后续的决策提供依据。
    -   **自动摘要与标注**: 调用内置的`processors`工具包，自动从知识中提取**风险、行动项、优缺点**等信息，并在答案前端生成"综合洞察摘要"，极大提升输出内容的价值。
-   **灵活的"连接器"架构**: 将核心检索逻辑与底层向量存储（FAISS, Milvus等）完全解耦，可像"插拔U盘"一样轻松更换或组合数据库。

---

## 运行机制

<p align="center">
  <img src="https://i.imgur.com/your-diagram-image.png" alt="KSF Architecture" width="800"/>
  <br/>
  <em>KSF v5.1 完整工作流。详情请见 <code>docs/ARCHITECTURE.md</code>。</em>
</p>

1.  **查询分析 (S-Module)**: 用户查询首先由S模块的`IntentAnalyzer`进行意图识别和实体提取，生成一份结构化的`RetrievalInstruction`。
2.  **四层共振检索 (K-Module)**: K模块接收`Instruction`，并执行四层共振算法。`δ`权重在此刻生效，对S模块"提醒"的实体相关知识进行加分。最终产出一个包含主知识、上下文知识和涌现概念的`ResonancePacket`。
3.  **答案合成 (S-Module)**: S模块的`PromptAssembler`接收`ResonancePacket`。它首先调用`processors`工具包对知识进行分析和标注，然后在Jinja2模板中渲染出一个包含"综合洞察摘要"和详细知识支撑的、结构化的最终答案。

---

## 快速开始

1.  **环境设置**:
    ```bash
    # 建议在虚拟环境中操作
    python -m venv ksf_env
    source ksf_env/bin/activate  # on Linux/macOS
    # ksf_env\Scripts\activate   # on Windows

    # 安装依赖
    pip install -r requirements.txt
    ```

2.  **下载模型 (如果需要)**:
    S模块的`IntentAnalyzer`依赖一些外部模型。首次运行时，`transformers`库会自动下载它们。如果您的环境无法访问Hugging Face，请手动下载以下模型并修改`ksf/s_module/analyzer.py`中的加载路径：
    -   NER模型: `spacy/zh_core_web_md`
    -   分类模型: `MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33`

3.  **数据准备与索引构建**:
    请先准备好您的原始知识库文件（如`data/my_knowledge.jsonl`）。然后，依次运行以下脚本：
    ```bash
    # 步骤1: 识别原始数据中的潜在歧义词 (人工审核可选)
    # python scripts/identify_ambiguous_terms.py --input_file data/your_data.jsonl
    
    # 步骤2: (如果进行了人工审核) 根据审核结果生成消歧义词典
    # python scripts/build_disambiguation_dict.py
    
    # 步骤3: 构建知识图谱，计算PageRank权重
    python scripts/build_knowledge_graph.py --input_file data/your_data.jsonl --output_dir checkpoints/my_kg
    
    # 步骤4: 构建最终的多轨统一索引
    python scripts/build_extended_index.py --primary data/your_data.jsonl --output_dir checkpoints/my_index --config configs/ksf_config.yaml
    ```
    *注意: 请确保`ksf_config.yaml`中的路径指向您刚刚生成的索引和权重目录。*

4.  **运行交互式演示**:
    ```bash
    python run.py
    ```
    现在，您可以输入查询，体验KSF v5.1强大的知识合成能力了！

---

## 架构蓝图

想深入了解KSF的设计哲学、模块细节和交互协议吗？请参阅我们为您准备的详细技术文档：

**[KSF架构设计蓝图 (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md)**

---
*此README最后更新于 KSF v5.1 - 功能激活与四层共振模型版本。*