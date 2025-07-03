import torch
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class EarlyStopper:
    """
    Handles early stopping logic.
    Stops training when a monitored metric has stopped improving.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0, monitor: str = "eval_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        """
        Args:
            metric_value (float): The metric value from the current epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_metric is None:
            self.best_metric = metric_value
        elif (self.monitor == "eval_loss" and metric_value < self.best_metric - self.min_delta) or \
             (self.monitor != "eval_loss" and metric_value > self.best_metric + self.min_delta):
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            logger.warning(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class Checkpointer:
    """
    Handles model checkpointing.
    Saves the model state based on configuration (best, last, etc.).
    """
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        checkpoint_config = config.get('checkpoints', {})
        self.dir = Path(checkpoint_config.get('dir', './checkpoints/ksf'))
        self.save_best_only = checkpoint_config.get('save_best_only', True)
        self.save_last = checkpoint_config.get('save_last', True)
        
        early_stopping_config = config.get('training', {}).get('early_stopping', {})
        self.monitor = early_stopping_config.get('monitor', 'eval_loss')
        self.best_metric = None
        
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, epoch: int, step: int, metric_value: float):
        """
        Saves a checkpoint based on the provided metric.

        Args:
            epoch (int): The current epoch.
            step (int): The current global step.
            metric_value (float): The metric value to check for improvement.
        """
        is_best = False
        if self.best_metric is None or \
           (self.monitor == 'eval_loss' and metric_value < self.best_metric) or \
           (self.monitor != 'eval_loss' and metric_value > self.best_metric):
            self.best_metric = metric_value
            is_best = True

        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }

        if is_best:
            self._save_checkpoint(state, "best_model.pth")
        
        if self.save_last:
            self._save_checkpoint(state, "last_model.pth")

        if not self.save_best_only:
             self._save_checkpoint(state, f"checkpoint_epoch_{epoch}_step_{step}.pth")


    def _save_checkpoint(self, state: dict, filename: str):
        """Saves the state dictionary to a file."""
        save_path = self.dir / filename
        logger.info(f"Saving checkpoint to {save_path}...")
        torch.save(state, save_path) 