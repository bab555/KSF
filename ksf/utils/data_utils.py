import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any

class KsfDataset(Dataset):
    """
    Dataset for KSF. Reads a JSONL file where each line is a dictionary
    with 'query', 'knowledge', and 'answer'.
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

def ksf_collate_fn(
    batch: List[Dict[str, Any]], 
    tokenizer: PreTrainedTokenizer, 
    max_length: int = 512, 
    generate_target_prompts: bool = False,
    max_prompt_length: int = 20
) -> Dict[str, torch.Tensor]:
    """
    Data collator for KSF. Tokenizes batches of data and prepares them for the model.
    If generate_target_prompts is True, it uses the 'query' as the target for prompt generation.
    """
    queries = [item['query'] for item in batch]
    knowledges = [item['knowledge'] for item in batch]
    answers = [item['answer'] for item in batch]

    # Create complete training sequences (query + answer)
    # This ensures input_ids and labels have the same length
    complete_sequences = [f"{query} {answer}" for query, answer in zip(queries, answers)]
    
    # Tokenize the complete sequences
    complete_inputs = tokenizer(
        complete_sequences, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    
    # For KSF, we still want to process the query separately for the S-module
    query_inputs = tokenizer(
        queries, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    
    # Tokenize knowledge for external input
    knowledge_inputs = tokenizer(
        knowledges, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    # Labels are the same as complete input_ids, but with padding masked
    labels = complete_inputs.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100 # Standard ignore_index for LM loss

    model_batch = {
        "input_ids": complete_inputs.input_ids,  # Complete sequence for base model processing
        "attention_mask": complete_inputs.attention_mask,
        "knowledge_input_ids": knowledge_inputs.input_ids,
        "knowledge_attention_mask": knowledge_inputs.attention_mask,
        "labels": labels,  # Complete sequence labels for loss calculation
        "query_input_ids": query_inputs.input_ids,  # Query for S-module processing
        "query_attention_mask": query_inputs.attention_mask,
    }

    # Handle optional prompt generation targets
    if generate_target_prompts:
        # Use the query as the target for the prompt generation head.
        # This teaches the synthesizer to generate a good, concise query.
        prompt_inputs = tokenizer(
            queries, padding='max_length', truncation=True, max_length=max_prompt_length, return_tensors="pt"
        )
        
        target_prompt_ids = prompt_inputs.input_ids.clone()
        # We don't want to ignore padding here, because the loss function will do it.
        # target_prompt_ids[target_prompt_ids == tokenizer.pad_token_id] = -100 # Let loss handle this
        model_batch["target_prompt_ids"] = target_prompt_ids

    return model_batch


def create_ksf_dataloaders(
    train_path: str, 
    eval_path: str, 
    tokenizer: PreTrainedTokenizer, 
    batch_size: int, 
    max_length: int, 
    generate_target_prompts: bool,
    max_prompt_length: int
) -> tuple[DataLoader, DataLoader]:
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    train_dataset = KsfDataset(train_path)
    eval_dataset = KsfDataset(eval_path)
    
    collator = lambda batch: ksf_collate_fn(
        batch, 
        tokenizer=tokenizer, 
        max_length=max_length,
        generate_target_prompts=generate_target_prompts,
        max_prompt_length=max_prompt_length
    )
    
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
