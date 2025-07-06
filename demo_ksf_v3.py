#!/usr/bin/env python3
"""
KSF v3 演示脚本
展示新架构的基本功能
"""

import sys
import os
import logging
from pathlib import Path

# 将项目根目录添加到Python路径，以便能够导入ksf模块
# This is a common practice for demonstration scripts
# to ensure they can find the project's modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入KSF v3
try:
    from ksf.core.orchestrator import KSFOrchestrator
    print("✅ KSF v3 导入成功")
except ImportError as e:
    print(f"❌ KSF v3 导入失败: {e}")
    # print full traceback for debugging
    import traceback
    traceback.print_exc()


def main():
    """主函数，用于演示KSF V3的功能"""
    
    # --- 配置 ---
    # 通过一个统一的config字典来管理所有配置
    config = {
        "discoverer": {
            "model_name": "./snowflake-arctic-embed-m",
            "adapter_path": "checkpoints/k_module_adapter",
            "index_dir": "checkpoints/k_module_index_v3_yunhe_adapted",
            "knowledge_file": "data/云和文旅知识库数据集.json",
            "weights_file": "data/knowledge_weights.json",
            "relevance_threshold": 0.25,
            "rerank_alpha": 0.3
        },
        "assembler": {
            "templates_dir": "ksf/s_module/templates",
            "manifest_path": "data/knowledge_manifest.json"
        },
        "force_rebuild_index": False # 如果为True，则强制重建索引；设为False以在后续运行中加快启动速度
    }

    print("==================================================")
    print("🚀 Initializing KSF V3 Orchestrator with RECALIBRATED Domain Adapter...")
    print("==================================================")

    try:
        # Pass the entire config dictionary to the orchestrator
        orchestrator = KSFOrchestrator(config)

        # The rest of your demo logic (asking questions, etc.) would go here.
        # For now, we just confirm it initializes.

    except Exception as e:
        print(f"❌ KSF v3 导入失败: {e}")
        # print full traceback for debugging
        import traceback
        traceback.print_exc()

    print("\n==================================================")
    
    # --- 运行查询 ---
    queries_to_run = [
        "云和的旅游特色是什么？",
        "苹果公司的核心竞争力体现在哪些方面？", # 这个应该会被判定为无关
        "介绍一下浙江省。", # 这个也可能被判定为无关
        "仙宫湖的门票多少钱？" # 这个应该是相关的
    ]

    for query in queries_to_run:
        run_and_print_query(orchestrator, query)

    print("\n==================================================")
    print("✅ Demo Finished")
    print("==================================================")


def run_and_print_query(orchestrator: KSFOrchestrator, query_text: str):
    """运行单个查询并打印结果"""
    print("\n" + "="*50)
    print(f"🤔 Processing Query: {query_text}")
    print("="*50)
    
    result = orchestrator.query(query_text)

    if result and result.get("answer"):
        print("✅ Final Assembled Prompt/Answer:")
        print("--------------------------------------------------")
        print(result["answer"])
        
        # Optionally, print details from the knowledge packet for analysis
        if "knowledge_packet" in result and result["knowledge_packet"].get("direct_knowledge"):
            print("\n🔬 Top Direct Knowledge Items:")
            print("--------------------------------------------------")
            for i, item in enumerate(result['knowledge_packet']['direct_knowledge'][:3]):
                content_preview = ' '.join(item.content.splitlines())
                print(f"  [{i+1}] ID: {item.id} | Score: {item.final_score:.4f}")
                print(f"      Content: {content_preview[:100]}...")

    else:
        print("❌ An unexpected error or empty result occurred.")
        print(result)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc() 