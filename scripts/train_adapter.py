from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
import torch

# --- Disable Weights & Biases Logging ---
# This prevents the interactive login prompt during training
os.environ["WANDB_DISABLED"] = "true"

# --- Configuration ---
BASE_MODEL_NAME = "./snowflake-arctic-embed-m"
TRIPLET_DATA_PATH = "data/triplets_for_training.jsonl"
# This is where we will save our lightweight adapter
OUTPUT_ADAPTER_PATH = "checkpoints/k_module_adapter"
# Training parameters
TRAIN_BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
# Margin for the triplet loss
TRIPLET_MARGIN = 5.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_triplet_data(file_path):
    """Loads the triplet data from the JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            examples.append(InputExample(texts=[data['anchor'], data['positive'], data['negative']]))
    print(f"Loaded {len(examples)} training examples from {file_path}")
    return examples

def main():
    """
    Main function to train a LoRA adapter on top of the base model
    using triplet data.
    """
    print("--- Starting Lightweight Adapter Training ---")
    
    # 1. Load the base model
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = SentenceTransformer(BASE_MODEL_NAME, device=DEVICE)

    # 2. Configure and apply LoRA adapter using PEFT
    # We need to manually target the underlying transformer model within the SentenceTransformer
    transformer_model = model[0].auto_model 

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION, # Use FEATURE_EXTRACTION for sentence embeddings
        target_modules=["query", "key", "value"],
    )

    # Apply the adapter to the transformer model itself, not the whole SentenceTransformer object
    peft_model = get_peft_model(transformer_model, lora_config)
    
    # Place the PEFT-enhanced model back into the SentenceTransformer pipeline
    model[0].auto_model = peft_model
    
    print("Applied LoRA adapter to the underlying transformer model.")
    peft_model.print_trainable_parameters()

    # 3. Load the training data
    train_examples = load_triplet_data(TRIPLET_DATA_PATH)
    # The dataloader handles batching and shuffling
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
    
    # 4. Define the loss function
    # This loss function will try to push the anchor-positive pair closer and the
    # anchor-negative pair further apart.
    train_loss = losses.TripletLoss(model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=TRIPLET_MARGIN)

    # 5. Train the model
    print("Starting model training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=10,
        optimizer_params={'lr': LEARNING_RATE},
        output_path=OUTPUT_ADAPTER_PATH,
        show_progress_bar=True
    )
    
    print("--- Adapter Training Complete ---")
    print(f"Trained adapter saved to: {OUTPUT_ADAPTER_PATH}")


if __name__ == "__main__":
    main() 