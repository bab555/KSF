import torch
import jieba
import jieba.posseg as pseg
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import re
import os

# --- 配置 ---
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m"
KNOWLEDGE_BASE_PATH = "data/knowledge_base.txt"
OUTPUT_CSV_PATH = "reports/ambiguous_terms_for_review.csv"
SIMILARITY_THRESHOLD = 0.8  # 相似度阈值
# 只考虑这些词性的词
ALLOWED_POS = ['n', 'nr', 'ns', 'nt', 'nz', 'vn'] 

# --- 初始化 ---
def initialize():
    """初始化模型和分词器, 并创建输出目录"""
    print("Initializing model and tokenizer...")
    # 确保有可用的GPU, 否则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    
    # 初始化jieba, 使用精确模式
    # jieba.set_dictionary(jieba.DEFAULT_DICT)
    
    print("Initialization complete.")
    return model

def get_contextual_embedding(model, sentence, term):
    """
    获取一个词在特定句子中的上下文感知向量。
    这是一个高级技巧, 我们需要访问模型的内部状态来获取特定token的embedding。
    """
    # 获取模型的底层transformers模型和tokenizer
    transformer_model = model[0].auto_model
    tokenizer = model.tokenizer

    # 编码句子以获取token id
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True).to(model.device)
    
    # 获取所有token的embedding
    with torch.no_grad():
        outputs = transformer_model(**inputs, output_hidden_states=True)
        # 我们使用最后一层的hidden state
        hidden_states = outputs.hidden_states[-1].squeeze(0)

    # 找到目标词对应的token(s)
    term_tokens = tokenizer.tokenize(term)
    input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    term_indices = []
    for i in range(len(input_tokens) - len(term_tokens) + 1):
        if input_tokens[i:i+len(term_tokens)] == term_tokens:
            term_indices = list(range(i, i + len(term_tokens)))
            break
            
    if not term_indices:
        return None # 如果找不到, 返回None

    # 对目标词的所有token的embedding取平均值, 作为其上下文向量
    term_embedding = hidden_states[term_indices].mean(dim=0)
    return term_embedding.unsqueeze(0) # 返回一个 [1, dim] 的tensor

def get_generic_embedding(model, term):
    """获取一个词的通用向量"""
    return torch.tensor(model.encode(term, convert_to_tensor=True)).unsqueeze(0)

def main():
    model = initialize()
    
    print(f"Loading knowledge base from: {KNOWLEDGE_BASE_PATH}")
    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []
    print("Analyzing terms for ambiguity...")
    
    for sentence in tqdm(lines, desc="Processing sentences"):
        # 使用jieba进行带词性的分词
        words = pseg.cut(sentence)
        
        # 提取所有符合条件的关键词
        key_terms = list(set([word for word, flag in words if flag in ALLOWED_POS and len(word) > 1]))
        
        for term in key_terms:
            # 1. 获取上下文感知向量
            context_emb = get_contextual_embedding(model, sentence, term)
            if context_emb is None:
                continue

            # 2. 获取通用向量
            generic_emb = get_generic_embedding(model, term)

            # 3. 计算余弦相似度
            sim = cosine_similarity(context_emb, generic_emb).item()

            # 4. 如果低于阈值, 记录下来
            if sim < SIMILARITY_THRESHOLD:
                print(f"Found potential ambiguity: '{term}' in '{sentence[:30]}...' (Similarity: {sim:.4f})")
                results.append({
                    "term": term,
                    "context_sentence": sentence,
                    "similarity_score": sim,
                    "clarification_needed(human_input)": "" # 留空待填写
                })

    if results:
        df = pd.DataFrame(results)
        print(f"\nAnalysis complete. Found {len(df)} potentially ambiguous terms.")
        print(f"Saving results to {OUTPUT_CSV_PATH}")
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    else:
        print("\nAnalysis complete. No ambiguous terms found with the current threshold.")

if __name__ == "__main__":
    main() 