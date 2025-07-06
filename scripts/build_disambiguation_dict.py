import pandas as pd
import json
import os

# --- 配置 ---
ANNOTATED_CSV_PATH = "reports/ambiguous_terms_for_review.csv"
OUTPUT_JSON_PATH = "data/disambiguation_dict.json"

def main():
    """
    读取经过人工标注的CSV文件, 并生成一个结构化的JSON消歧义词典。
    """
    print(f"Reading annotated data from: {ANNOTATED_CSV_PATH}")
    
    try:
        df = pd.read_csv(ANNOTATED_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Annotated file not found at {ANNOTATED_CSV_PATH}")
        print("Please run identify_ambiguous_terms.py first, and then manually annotate the CSV.")
        return

    # 筛选出已经标注过的行
    df_annotated = df.dropna(subset=['clarification_needed(human_input)'])

    if df_annotated.empty:
        print("No annotated terms found in the CSV file. Nothing to do.")
        return

    disambiguation_dict = {}

    print("Building disambiguation dictionary...")
    for index, row in df_annotated.iterrows():
        term = row['term']
        context = row['context_sentence']
        clarification = row['clarification_needed(human_input)']
        
        # 构建增强后的词条, 例如 "苹果(水果)"
        new_term = f"{term}{clarification}"
        
        # 如果词条是第一次出现, 初始化一个list
        if term not in disambiguation_dict:
            disambiguation_dict[term] = []
        
        # 添加包含上下文的完整消歧义信息
        disambiguation_dict[term].append({
            "context_sentence": context,
            "clarification": clarification,
            "enhanced_term": new_term
        })
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    print(f"Saving disambiguation dictionary to: {OUTPUT_JSON_PATH}")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(disambiguation_dict, f, ensure_ascii=False, indent=4)

    print("\nDictionary built successfully:")
    print(json.dumps(disambiguation_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main() 