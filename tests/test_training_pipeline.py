import unittest
import torch
import yaml
import json
import tempfile
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ksf.training.trainer import KsfTrainer

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory with dummy configs, data, and a lightweight model."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # 1. Create dummy model config and files
        # We will use a very small model for testing to avoid heavy downloads.
        # Let's use 'sshleifer/tiny-gpt2' as it's small and has a standard architecture.
        self.model_name = 'sshleifer/tiny-gpt2'
        
        # Create configs directory
        self.configs_dir = self.temp_path / "configs"
        self.configs_dir.mkdir()

        # 2. Create dummy training config
        self.config = {
            'base_model': {'path': self.model_name, 'torch_dtype': 'float32'},
            'model': {
                'hidden_size': 64, # tiny-gpt2 has a hidden size of 64
                'intermediate_size': 256,
                'dropout_rate': 0.1,
                'knowledge_heads': 2,
                'synthesizer_heads': 2,
                'reasoning_dim': 64,
                'enable_prompt_generation': True,
                'max_prompt_length': 10,
                'fusion': {'enable_dynamic_gating': True, 'top_k': 5},
            },
            'training': {
                'batch_size': 2,
                'epochs': 1,
                'warmup_steps': 1,
                'lr_scheduler_type': 'linear',
                'loss_weights': {'main': 1.0, 'guidance': 0.1, 'quality': 0.05, 'prompt': 0.5},
            },
            'data': {
                'train_path': str(self.temp_path / 'train.jsonl'),
                'eval_path': str(self.temp_path / 'eval.jsonl'),
                'max_length': 32,
            },
            'logging': {'level': 'ERROR'},
        }
        self.config_path = self.configs_dir / 'test_config.yaml'
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

        # 3. Create dummy data files
        self.dummy_data = {
            "query": "What is AI?",
            "knowledge": "AI is the simulation of human intelligence.",
            "answer": "AI simulates human intelligence.",
            "prompt": "artificial intelligence definition" # Target prompt
        }
        with open(self.config['data']['train_path'], 'w') as f:
            for _ in range(4): # A few samples
                f.write(json.dumps(self.dummy_data) + '\n')
        with open(self.config['data']['eval_path'], 'w') as f:
            for _ in range(4):
                f.write(json.dumps(self.dummy_data) + '\n')

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test to avoid heavy CPU load")
    def test_single_training_step(self):
        """
        Tests a single training step to ensure the pipeline runs end-to-end
        and computes all expected loss components.
        """
        # 1. Initialize trainer
        # This will load the model, data, optimizer etc.
        trainer = KsfTrainer(config_path=str(self.config_path))
        
        # We need to manually adjust the hidden_size from the downloaded model config
        # as tiny-gpt2's config might differ slightly.
        actual_hidden_size = trainer.model.model_config.hidden_size
        if self.config['model']['hidden_size'] != actual_hidden_size:
            self.fail(f"Config hidden_size ({self.config['model']['hidden_size']}) does not match model's actual hidden_size ({actual_hidden_size}). Please update the test config.")


        # 2. Get one batch from the dataloader
        train_dataloader = trainer.train_dataloader
        try:
            batch = next(iter(train_dataloader))
        except StopIteration:
            self.fail("Train dataloader is empty.")

        # 3. Perform a single training step
        loss_dict = trainer._training_step(batch)

        # 4. Assertions
        self.assertIn('total_loss', loss_dict)
        self.assertIn('main_loss', loss_dict)
        self.assertIn('guidance_loss', loss_dict)
        self.assertIn('quality_loss', loss_dict)
        self.assertIn('prompt_loss', loss_dict)

        # Check that losses are valid numbers
        self.assertGreater(loss_dict['total_loss'], 0)
        self.assertTrue(torch.isfinite(torch.tensor(loss_dict['total_loss'])))
        
        # Check that prompt loss was actually computed
        self.assertGreater(loss_dict['prompt_loss'], 0)
        
        # Check that the batch contains the target prompt ids
        self.assertIn('target_prompt_ids', batch)
        self.assertEqual(batch['target_prompt_ids'].shape[0], self.config['training']['batch_size'])
        self.assertEqual(batch['target_prompt_ids'].shape[1], self.config['model']['max_prompt_length'])


if __name__ == '__main__':
    unittest.main() 