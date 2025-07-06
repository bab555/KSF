"""
Prepares the Yunhe tourism dataset for KSF V2 training.

This script performs two main functions:
1.  Processes the "Virtual Text Map" (`云和文旅虚拟文字地图.json`):
    - It treats each entry in the map as a core piece of knowledge.
    - It converts these structured entries into simple, declarative sentences.
    - These sentences are then written to `data/knowledge_base.txt`, which will be
      used to inject knowledge into the K-Module's memory platforms.

2.  Processes the "Q&A Knowledge Base" (`云和文旅知识库数据集.json`):
    - It assumes this file contains structured Q&A pairs.
    - It splits the data into a training set (90%) and an evaluation set (10%).
    - It reformats these sets into the `.jsonl` format expected by the trainer,
      creating `data/train.jsonl` and `data/eval.jsonl`.
"""
import json
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(input_file, train_output_file, eval_output_file, knowledge_base_file, test_size=0.1, random_state=42):
    """
    Reads an augmented Q&A JSONL file, splits it into training and evaluation sets,
    and creates a combined knowledge base file.
    """
    
    try:
        # Ensure the output directories exist
        os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(eval_output_file), exist_ok=True)
        os.makedirs(os.path.dirname(knowledge_base_file), exist_ok=True)

        # Read the consolidated dataset
        all_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_data.append(json.loads(line.strip()))

        logging.info(f"Total records read from {input_file}: {len(all_data)}")

        # Split the data into training and evaluation sets
        train_data, eval_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
        
        # Write the training data
        with open(train_output_file, 'w', encoding='utf-8') as f:
            for item in train_data:
                record = {"query": item["query"], "pos": item["answer"], "neg": ""}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Write the evaluation data
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            for item in eval_data:
                record = {"query": item["query"], "pos": item["answer"], "neg": ""}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logging.info(f"Training data written to {train_output_file}: {len(train_data)} records")
        logging.info(f"Evaluation data written to {eval_output_file}: {len(eval_data)} records")
        
        # Create the knowledge base from all Q&A pairs
        knowledge_base_content = []
        for item in all_data:
            knowledge_base_content.append(f"问：{item['query']} 答：{item['answer']}")

        with open(knowledge_base_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(knowledge_base_content))
            
        logging.info(f"Knowledge base created at {knowledge_base_file} with {len(knowledge_base_content)} entries.")
        logging.info("✅ Data preparation script finished successfully.")

    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {input_file}")
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {input_file}. Please check the file format.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Define file paths
    input_dataset_path = 'data/augmented_qa_dataset.jsonl'
    train_file = 'data/staged_training_test/train/ksf_train.jsonl'
    eval_file = 'data/staged_training_test/eval/ksf_eval.jsonl'
    knowledge_base = 'data/knowledge_base.txt'
    
    # Run the data preparation
    prepare_data(input_dataset_path, train_file, eval_file, knowledge_base) 