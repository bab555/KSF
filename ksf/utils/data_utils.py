import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, default_data_collator
from typing import List, Dict, Any, Union
import logging
import networkx as nx

# Get a logger
logger = logging.getLogger(__name__)

# Define the special token for keyword/query generation mode
KSF_GENERATE_QUERY_TOKEN = "[KSF_GENERATE_QUERY]"

def load_knowledge_base_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads a knowledge base from a file, supporting .txt and .json formats.
    Returns a list of dictionaries, each with an 'id' and 'content' key.
    """
    print(f"Loading knowledge base from: {file_path}")
    knowledge_items = []
    
    _, file_extension = os.path.splitext(file_path)

    if file_extension == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "qa_pairs" in data and isinstance(data["qa_pairs"], list):
            for i, item in enumerate(data["qa_pairs"]):
                question = item.get('question', '')
                answer = item.get('answer', '')
                content = f"问题：{question}\n回答：{answer}"
                knowledge_items.append({"id": i, "content": content})
        
        elif isinstance(data, list):
            for i, item_content in enumerate(data):
                 knowledge_items.append({"id": i, "content": str(item_content)})
            
        else:
            raise ValueError("Unsupported JSON structure for knowledge base.")

    elif file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    knowledge_items.append({"id": i, "content": line.strip()})
    
    elif file_extension == ".jsonl":
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    item = json.loads(line)
                    # 确保'id'和'content'键存在
                    if 'id' in item and 'content' in item:
                        knowledge_items.append(item)
                    else:
                        logger.warning(f"跳过第 {i+1} 行，因为它缺少'id'或'content'键。")

    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please use .json, .txt, or .jsonl.")
        
    print(f"✓ Successfully loaded knowledge base with {len(knowledge_items)} items.")
    return knowledge_items

def load_knowledge_weights(weights_file: str) -> Dict[str, float]:
    """
    从JSON文件加载预先计算好的知识权重（如PageRank）。
    """
    logger.info(f"正在从 {weights_file} 加载知识权重...")
    try:
        with open(weights_file, "r", encoding="utf-8") as f:
            weights_data = json.load(f)
        # 确保键是字符串，值是浮点数
        processed_weights = {str(k): float(v.get('weight', 0.0)) for k, v in weights_data.items()}
        logger.info(f"✓ 成功加载并处理了 {len(processed_weights)} 个知识权重。")
        return processed_weights
    except FileNotFoundError:
        logger.warning(f"⚠️ 在 {weights_file} 未找到知识权重文件。Ss权重将始终为0。")
        return {}
    except Exception as e:
        logger.error(f"加载或处理权重文件 {weights_file} 时出错: {e}")
        return {}

def calculate_pagerank(graph_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    根据图数据计算PageRank。

    Args:
        graph_data (List[Dict[str, Any]]): 包含'nodes'和'links'的图数据。

    Returns:
        Dict[str, float]: 包含每个节点ID及其PageRank中心性的字典。
    """
    G = nx.DiGraph()
    if not graph_data or 'nodes' not in graph_data or 'links' not in graph_data:
        logger.warning("图数据格式不正确或为空，无法计算PageRank。")
        return {}
        
    for node in graph_data.get('nodes', []):
        G.add_node(node['id'])
    
    for link in graph_data.get('links', []):
        G.add_edge(link['source'], link['target'])

    if not G.nodes:
        logger.warning("图中没有节点，无法计算PageRank。")
        return {}
        
    pagerank = nx.pagerank(G)
    return pagerank

def save_data_to_json(data: Union[List, Dict], file_path: str):
    """
    Saves data (list or dict) to a JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def add_special_tokens_to_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """
    Adds custom special tokens required for the KSF framework to the tokenizer.

    Args:
        tokenizer: The original Hugging Face tokenizer.

    Returns:
        The tokenizer with added special tokens.
    """
    special_tokens_dict = {
        'additional_special_tokens': [
            '[KSF_GENERATE_QUERY]' # Used to trigger the bypass valve mechanism
        ]
    }
    
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_toks > 0:
        logging.info(f"Added {num_added_toks} special token(s) to tokenizer: {special_tokens_dict['additional_special_tokens']}")
    
    return tokenizer

class KsfDataset(Dataset):
    """
    Dataset for KSF. Reads a JSONL file where each line is a dictionary.
    Expected keys: 'query', 'answer'.
    Optional key: 'knowledge'.
    """
    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class KsfCollator:
    """
    Data collator for the new K -> S -> LLM architecture.

    It prepares batches by creating two sets of tokenized inputs:
    1.  `query_input_ids`: For the K-Module to analyze (query only).
    2.  `input_ids`: For the full model flow (query + answer).

    It also handles label masking, so loss is only computed on the answer part.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._add_special_tokens_if_needed()

    def _add_special_tokens_if_needed(self):
        """Adds the special KSF token to the tokenizer if it doesn't exist."""
        if KSF_GENERATE_QUERY_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [KSF_GENERATE_QUERY_TOKEN]}
            )
            logger.info(f"Added special token to tokenizer: {KSF_GENERATE_QUERY_TOKEN}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        answers = []
        is_keyword_generation_batch = 'keywords' in batch[0]

        for item in batch:
            query = item['query']
            if is_keyword_generation_batch:
                # This is for the keyword generation pre-training task
                prompt = f"Extract keywords for: {query}"
                answer = ", ".join(item['keywords'])
            else:
                # For supervised fine-tuning, the label is the answer itself.
                # The model should learn to predict the answer tokens after the query tokens.
                prompt = query
                answer = item['pos']
            
            prompts.append(prompt)
            answers.append(answer)

        # 1. Tokenize the PROMPT/QUERY part separately for the K-Module
        query_tokenized = self.tokenizer(
            prompts,
            padding="longest", # Pad to the longest query in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 2. Tokenize the FULL sequence (prompt + answer) for the main model flow
        full_sequences = [p + a + self.tokenizer.eos_token for p, a in zip(prompts, answers)]
        full_tokenized = self.tokenizer(
            full_sequences,
            padding="longest", # Pad to the longest full sequence in the batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 3. Create labels and mask out the prompt part
        labels = full_tokenized.input_ids.clone()
        
        # We need to find the length of the prompt in the tokenized *full sequence*
        # to know how much to mask. We can't use query_tokenized['input_ids'] directly
        # because tokenization can differ. We re-tokenize prompts without padding.
        prompt_only_tokenized = self.tokenizer(prompts, padding=False, truncation=True)
        prompt_lengths = [len(p) for p in prompt_only_tokenized['input_ids']]
        
        for i in range(len(batch)):
            labels[i, :prompt_lengths[i]] = -100 # Mask prompt
        
        # Also mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        model_batch = {
            "input_ids": full_tokenized.input_ids,
            "attention_mask": full_tokenized.attention_mask,
            "query_input_ids": query_tokenized.input_ids,
            "query_attention_mask": query_tokenized.attention_mask,
            "labels": labels,
        }
        
        return model_batch


def create_ksf_dataloaders(
    train_path: str, 
    eval_path: str, 
    tokenizer: PreTrainedTokenizer, 
    batch_size: int, 
    max_length: int,
    **kwargs # Absorb unused arguments for compatibility
) -> tuple[DataLoader, DataLoader]:
    
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset = KsfDataset(train_path)
    eval_dataset = KsfDataset(eval_path)
    
    # Initialize the collator once
    collator = KsfCollator(tokenizer=tokenizer, max_length=max_length)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False
    )
    return train_dataloader, eval_dataloader


# --- New SFT Data Handling ---

class SftDataset(Dataset):
    """
    Dataset for the new SFT task. Reads a JSONL file where each line is a
    dictionary with 'knowledge_context' and 'golden_prompt'.
    """
    def __init__(self, data_path: str):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # We only need the context and the target prompt for SFT
        item = self.data[idx]
        return {
            "knowledge_context": item.get("knowledge_context", ""),
            "golden_prompt": item.get("golden_prompt", "")
        }

class SftCollator:
    """
    Data collator for the new SFT task.
    It tokenizes the context and the golden prompt, and prepares the labels.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # For SFT, the input to the model is the knowledge context.
        # The model's task is to generate the golden_prompt.
        inputs = self.tokenizer(
            [item["knowledge_context"] for item in batch],
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # The labels are the tokenized golden prompts.
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                [item["golden_prompt"] for item in batch],
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids

        # Replace padding token id in labels with -100 to ignore in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs

def create_sft_dataloaders(
    train_path: str,
    eval_path: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int,
    **kwargs
) -> tuple[DataLoader, DataLoader]:
    """
    Creates dataloaders for the SFT task.
    """
    if tokenizer.pad_token is None:
        logger.info("Tokenizer has no pad_token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = SftDataset(train_path)
    eval_dataset = SftDataset(eval_path)
    
    collator = SftCollator(tokenizer=tokenizer, max_length=max_length)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False
    )
    return train_dataloader, eval_dataloader
