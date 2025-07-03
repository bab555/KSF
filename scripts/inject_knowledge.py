"""
Knowledge Injection Script for KSF V2

This script is a one-time utility to pre-load the KnowledgeBank module
with knowledge embeddings derived from an external text file.

It performs the following steps:
1. Loads the main KSF training configuration.
2. Initializes the full AdvancedKsfModel.
3. Retrieves the KnowledgeBank sub-module.
4. Calls the `inject_knowledge_from_file` method on the KnowledgeBank.
5. Saves the state dictionary of the modified KnowledgeBank to a file.

This pre-computed knowledge bank can then be loaded into the model before
starting the main training loop.
"""
import os
import yaml
import torch
import sys
import logging

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from ksf.models.advanced_ksf_model import AdvancedKsfModel

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the knowledge injection process."""
    
    # --- 1. Load Configuration ---
    config_path = os.path.join(project_root, 'configs', 'ksf_training_config.yaml')
    logger.info(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}. Please ensure it exists.")
        return
    except Exception as e:
        logger.error(f"Error loading YAML configuration: {e}")
        return

    # --- 2. Initialize the Full KSF Model ---
    # We initialize the full model to ensure all components are correctly built.
    # The base model will be loaded, which might take time and memory.
    logger.info("Initializing the AdvancedKsfModel...")
    # Note: This will download and load the base model from Hugging Face.
    model = AdvancedKsfModel(config)
    
    # --- 3. Perform Knowledge Injection ---
    knowledge_bank = model.knowledge_bank
    logger.info("Calling inject_knowledge_from_file on the KnowledgeBank module...")
    knowledge_bank.inject_knowledge_from_file(config)

    # --- 4. Save the KnowledgeBank State Dictionary ---
    output_dir = config.get('training', {}).get('output_dir', 'checkpoints/ksf_v2')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
        
    output_path = os.path.join(output_dir, "injected_knowledge_bank.pt")
    logger.info(f"Saving the state dictionary of the knowledge-injected bank to: {output_path}")
    
    # We only save the state dict of the knowledge bank, not the whole model.
    torch.save(knowledge_bank.state_dict(), output_path)
    
    logger.info("✅ Knowledge injection process complete. The state dictionary has been saved.")


if __name__ == "__main__":
    main() 