"""
Main KSF Trainer

This module contains the KsfTrainer class, which orchestrates the
end-to-end training and evaluation process for the KSF model.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import logging
import yaml
from tqdm import tqdm
from typing import Dict, Any

from ..models.advanced_ksf_model import AdvancedKsfModel
from ..training.losses import KsfLoss
from ..utils.data_utils import create_ksf_dataloaders
from ..utils.trainer_utils import EarlyStopper, Checkpointer

class KsfTrainer:
    """
    Orchestrates the training process for the AdvancedKsfModel.
    """
    def __init__(self, config_path: str):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        self.logger.info("Initializing KSF Trainer...")
        # Use the new AdvancedKsfModel
        self.model = AdvancedKsfModel(self.config)
        self.device = self.model.device
        
        # Setup tokenizer pad_token_id for loss
        pad_token_id = self.model.pseudo_api.tokenizer.pad_token_id
        if pad_token_id is None:
            # Using a standard ignore_index if pad token is not set
            pad_token_id = -100
            self.logger.warning("Tokenizer does not have a pad_token_id. Using -100 for ignore_index in prompt loss.")

        # Use the new KsfLoss
        self.loss_fn = KsfLoss(self.config)
        self.optimizer = self._setup_optimizer()
        
        self.train_dataloader, self.eval_dataloader = self._create_dataloaders()
        
        self.lr_scheduler = self._setup_scheduler()
        
        # New: Setup early stopping and checkpointing
        training_config = self.config['training']
        if training_config.get('early_stopping', {}).get('enabled', False):
            es_config = training_config['early_stopping']
            self.early_stopper = EarlyStopper(
                patience=es_config.get('patience', 3),
                min_delta=es_config.get('min_delta', 0.0),
                monitor=es_config.get('monitor', 'eval_loss')
            )
        else:
            self.early_stopper = None
            
        self.checkpointer = Checkpointer(self.model, self.optimizer, self.config)
        self.grad_clipping = float(training_config.get('gradient_clipping', 1.0))
        self.logger.info("✅ KSF Trainer initialized successfully.")

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        optimizer_config = self.config.get('training', {}).get('optimizer', {})
        optimizer_type = optimizer_config.get("type", "adamw").lower()
        
        trainable_params = self.model.get_trainable_parameters()
        
        if not trainable_params:
            self.logger.warning("No trainable parameters found. Training will not proceed.")
            return None

        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=float(self.config['training'].get('learning_rate', 5e-5)),
                betas=tuple(optimizer_config.get('betas', [0.9, 0.999])),
                eps=float(optimizer_config.get('eps', 1e-8)),
                weight_decay=float(self.config['training'].get('weight_decay', 0.01))
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    def _create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        data_config = self.config.get('data', {})
        model_config = self.config.get('model', {})
        
        # The tokenizer is part of the base model wrapper
        tokenizer = self.model.pseudo_api.tokenizer

        return create_ksf_dataloaders(
            train_path=data_config.get('train_path'),
            eval_path=data_config.get('eval_path'),
            tokenizer=tokenizer,
            batch_size=self.config['training'].get('batch_size', 4),
            max_length=data_config.get('max_length', 512),
            # Pass the prompt generation flag and max length to the data loader
            generate_target_prompts=model_config.get('enable_prompt_generation', False),
            max_prompt_length=model_config.get('max_prompt_length', 20)
        )

    def _setup_scheduler(self):
        training_config = self.config.get('training', {})
        scheduler_config = training_config.get('lr_scheduler', {})
        scheduler_type = scheduler_config.get('type', 'linear')

        if not self.optimizer:
            return None
            
        num_training_steps = len(self.train_dataloader) * training_config.get('num_epochs', 3)
        num_warmup_steps = int(num_training_steps * scheduler_config.get('warmup_ratio', 0.1)) \
            if 'warmup_ratio' in scheduler_config else training_config.get('warmup_steps', 0)

        return get_scheduler(
            name=scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        
        batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}
        
        # The model's forward pass now handles all logic, including robustness training
        outputs = self.model(
            query_input_ids=batch['input_ids'],
            query_attention_mask=batch['attention_mask'],
            labels=batch.get('labels'),
            summary_labels=batch.get('summary_labels'),
            summary_attention_mask=batch.get('summary_attention_mask')
        )
        
        # The model already calculates the loss internally
        total_loss = outputs.loss
        
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), self.grad_clipping)

        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        return {'total_loss': total_loss.item()}

    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.eval()
        batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}
        
        with torch.no_grad():
            outputs = self.model(
                query_input_ids=batch['input_ids'],
                query_attention_mask=batch['attention_mask'],
                labels=batch.get('labels'),
                summary_labels=batch.get('summary_labels'),
                summary_attention_mask=batch.get('summary_attention_mask')
            )
            total_loss = outputs.loss
        
        return {'total_loss': total_loss.item()}

    def train(self):
        num_epochs = self.config['training'].get('num_epochs', 3)
        if self.optimizer is None:
            self.logger.error("Optimizer not initialized. Aborting training.")
            return

        self.logger.info("🚀 Starting KSF model training...")
        global_step = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
            
            # Training loop
            self.model.train()
            train_loss_log = {}
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1} Training")
            for batch in progress_bar:
                loss_items = self._training_step(batch)
                global_step += 1
                for k, v in loss_items.items():
                    train_loss_log.setdefault(k, []).append(v)
                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_items.items()})

            avg_train_losses = {f"train_{k}": sum(v) / len(v) for k, v in train_loss_log.items()}
            self.logger.info(f"Avg Training Losses: {avg_train_losses}")

            # Evaluation loop
            self.model.eval()
            eval_loss_log = {}
            eval_progress_bar = tqdm(self.eval_dataloader, desc=f"Epoch {epoch+1} Evaluation")
            for batch in eval_progress_bar:
                loss_items = self._evaluation_step(batch)
                for k, v in loss_items.items():
                    eval_loss_log.setdefault(k, []).append(v)
                eval_progress_bar.set_postfix({k: f"{v:.4f}" for k, v in loss_items.items()})
            
            avg_eval_losses = {f"eval_{k}": sum(v) / len(v) for k, v in eval_loss_log.items()}
            self.logger.info(f"Avg Evaluation Losses: {avg_eval_losses}")

            # Checkpoint and Early Stopping
            eval_metric_val = avg_eval_losses.get(f"eval_{self.early_stopper.monitor}" if self.early_stopper else "eval_loss")
            
            if self.checkpointer:
                self.checkpointer.save(epoch, global_step, eval_metric_val)

            if self.early_stopper:
                if self.early_stopper(eval_metric_val):
                    self.logger.info(f"Early stopping triggered after epoch {epoch+1}. Best {self.early_stopper.monitor}: {self.early_stopper.best_metric:.4f}")
                    break
        
        self.logger.info("🏁 KSF model training finished.") 