import json
import os
import jieba
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

# --- Configuration ---
KNOWLEDGE_BASE_PATH = "data/云和文旅知识库数据集.json"
DISAMBIGUATION_DICT_PATH = "data/disambiguation_dict.json"
OUTPUT_GRAPH_PATH = "data/knowledge_graph.gml"
OUTPUT_WEIGHTS_PATH = "data/knowledge_weights.json"
# 使用我们之前统一的模型
MODEL_NAME = "./snowflake-arctic-embed-m" 
# 相似度阈值，用于决定是否在两个节点间创建边
SIMILARITY_THRESHOLD = 0.5  
# 确定运行设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data():
    """加载知识库和消歧字典"""
    print(f"Loading knowledge base from {KNOWLEDGE_BASE_PATH}...")
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    knowledge_chunks = []
    # 从"qa_pairs"中提取问答对作为知识块
    for item in data.get("qa_pairs", []):
        question = item.get('question', '')
        answer = item.get('answer', '')
        # 将问题和答案合并成一个知识块
        knowledge_chunks.append(f"问题：{question}\n回答：{answer}")

    print(f"Loaded {len(knowledge_chunks)} knowledge chunks from JSON.")

    print(f"Loading disambiguation dictionary from {DISAMBIGUATION_DICT_PATH}...")
    with open(DISAMBIGUATION_DICT_PATH, "r", encoding="utf-8") as f:
        disambiguation_dict = json.load(f)
        
    return knowledge_chunks, disambiguation_dict

def precompute_disambiguation_embeddings(disambiguation_dict, model):
    """预计算消歧选项中的上下文句子embedding"""
    print("Pre-computing embeddings for disambiguation contexts...")
    precomputed_dict = {}
    for term, options in tqdm(disambiguation_dict.items(), desc="Pre-computing"):
        precomputed_dict[term] = []
        for option in options:
            context_embedding = model.encode(option["context_sentence"], convert_to_tensor=True, device=DEVICE)
            option_copy = option.copy()
            option_copy["embedding"] = context_embedding
            precomputed_dict[term].append(option_copy)
    return precomputed_dict

def disambiguate_chunk(chunk, model, precomputed_disamb_dict):
    """对单个知识块进行消歧处理"""
    chunk_embedding = model.encode(chunk, convert_to_tensor=True, device=DEVICE)
    modified_chunk = chunk
    
    # 使用jieba分词来查找可能存在的歧义词
    words = jieba.lcut(chunk)
    for word in words:
        if word in precomputed_disamb_dict:
            options = precomputed_disamb_dict[word]
            if not options:
                continue

            # 比较当前chunk与哪个消歧上下文最相似
            context_embeddings = torch.stack([opt["embedding"].squeeze(0) for opt in options])
            similarities = util.cos_sim(chunk_embedding, context_embeddings)
            
            best_option_idx = torch.argmax(similarities).item()
            best_option = options[best_option_idx]
            
            # 执行替换，确保只替换独立的词
            # 为避免错误替换（如"苹果"替换"苹果电脑"），使用更安全的替换逻辑
            # 这里简单实现，实际可能需要更复杂的基于词边界的替换
            modified_chunk = modified_chunk.replace(word, best_option["enhanced_term"])
            
    return modified_chunk

def main():
    """主函数，执行整个图谱构建流程"""
    knowledge_chunks, disambiguation_dict = load_data()
    
    print(f"Loading embedding model: {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # 1. 消歧处理
    precomputed_disamb_dict = precompute_disambiguation_embeddings(disambiguation_dict, model)
    
    print("Applying disambiguation to all knowledge chunks...")
    processed_chunks = [disambiguate_chunk(chunk, model, precomputed_disamb_dict) for chunk in tqdm(knowledge_chunks, desc="Disambiguating")]

    # 2. 为处理后的知识块生成Embedding
    print("Embedding processed knowledge chunks...")
    chunk_embeddings = model.encode(processed_chunks, convert_to_tensor=True, show_progress_bar=True, device=DEVICE)

    # 3. 构建知识图谱
    print("Building knowledge graph...")
    G = nx.Graph()
    for i, chunk in enumerate(knowledge_chunks):
        # 节点中同时保存原始内容和处理后内容
        G.add_node(i, original_content=chunk, processed_content=processed_chunks[i])

    # 计算所有chunk对之间的相似度
    cosine_scores = util.cos_sim(chunk_embeddings, chunk_embeddings)

    print("Adding edges based on similarity threshold...")
    for i in tqdm(range(len(processed_chunks)), desc="Adding graph edges"):
        for j in range(i + 1, len(processed_chunks)):
            similarity = cosine_scores[i][j].item()
            if similarity > SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=similarity)

    print(f"Graph created successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # 4. 计算PageRank
    if G.number_of_edges() > 0:
        print("Calculating PageRank for nodes...")
        # 使用边的权重来计算PageRank
        pagerank_scores = nx.pagerank(G, weight='weight')

        # 将权重归一化到0-1范围，便于后续使用
        max_score = max(pagerank_scores.values()) if pagerank_scores else 1.0
        min_score = min(pagerank_scores.values()) if pagerank_scores else 0.0
        
        final_weights = {}
        for node_id, score in pagerank_scores.items():
            # 使用节点ID作为key，保存权重和原始内容
            normalized_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 0
            final_weights[node_id] = {
                "content": knowledge_chunks[node_id], 
                "weight": normalized_score
            }
    else:
        print("No edges were created, skipping PageRank. All nodes will have a default weight.")
        final_weights = {i: {"content": chunk, "weight": 0.0} for i, chunk in enumerate(knowledge_chunks)}


    # 5. 保存结果
    output_dir = os.path.dirname(OUTPUT_GRAPH_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving graph to {OUTPUT_GRAPH_PATH}...")
    nx.write_gml(G, OUTPUT_GRAPH_PATH)

    print(f"Saving weights to {OUTPUT_WEIGHTS_PATH}...")
    with open(OUTPUT_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_weights, f, ensure_ascii=False, indent=4)

    print("\nKnowledge graph construction complete!")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Graph saved to: {OUTPUT_GRAPH_PATH}")
    print(f"Node weights saved to: {OUTPUT_WEIGHTS_PATH}")

if __name__ == "__main__":
    main() 