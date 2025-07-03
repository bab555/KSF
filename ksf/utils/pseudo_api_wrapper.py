import torch
import torch.nn as nn
import warnings
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer

class PseudoAPIWrapper(nn.Module):
    def __init__(self, base_model: nn.Module, enable_monitoring: bool = True):
        super().__init__()
        self.base_model = base_model
        self.enable_monitoring = enable_monitoring
        self.original_param_checksums = {}
        
        # Load tokenizer for the base model
        self._load_tokenizer()
        
        self._freeze_base_model()
        if enable_monitoring:
            self._setup_gradient_monitoring()
    
    def _load_tokenizer(self):
        """Load the tokenizer for the base model."""
        try:
            # Try to get tokenizer from model config first
            if hasattr(self.base_model.config, 'name_or_path'):
                model_path = self.base_model.config.name_or_path
            elif hasattr(self.base_model.config, '_name_or_path'):
                model_path = self.base_model.config._name_or_path
            else:
                # Fallback to a reasonable default
                model_path = "hf-internal-testing/tiny-random-LlamaForCausalLM"
                warnings.warn(f"Could not determine model path, using default: {model_path}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception as e:
            warnings.warn(f"Failed to load tokenizer: {e}. Using a dummy tokenizer.")
            # Create a minimal dummy tokenizer for testing
            class DummyTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.eos_token_id = 1
                    self.pad_token = "<pad>"
                    self.eos_token = "<eos>"
            self._tokenizer = DummyTokenizer()

    @property
    def tokenizer(self):
        """Access to the tokenizer."""
        return self._tokenizer

    def get_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Get embeddings from input token IDs.
        
        Args:
            input_ids: Token IDs tensor [batch_size, seq_len]
            
        Returns:
            Embeddings tensor [batch_size, seq_len, hidden_size]
        """
        # Get the embedding layer from the base model
        if hasattr(self.base_model, 'get_input_embeddings'):
            embeddings = self.base_model.get_input_embeddings()(input_ids)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embed_tokens'):
            embeddings = self.base_model.model.embed_tokens(input_ids)
        elif hasattr(self.base_model, 'transformer') and hasattr(self.base_model.transformer, 'wte'):
            embeddings = self.base_model.transformer.wte(input_ids)
        else:
            # Fallback: run full forward pass and extract hidden states
            with torch.no_grad():
                outputs = self.base_model(input_ids=input_ids, output_hidden_states=True)
                embeddings = outputs.hidden_states[0]  # First layer hidden states
        
        return embeddings

    def forward_from_embeddings(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        Run the model forward pass starting from embeddings.
        
        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with logits
        """
        return self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def _freeze_base_model(self):
        frozen_count = 0
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
            self.original_param_checksums[name] = param.data.sum().item()
        print(f" PseudoAPI: Frozen {frozen_count} base model parameters")
    
    def _setup_gradient_monitoring(self):
        self.gradient_norms = []
        def gradient_hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                norm = grad_output[0].norm().item()
                self.gradient_norms.append(norm)
                if len(self.gradient_norms) > 100:
                    self.gradient_norms.pop(0)
        if hasattr(self.base_model, 'lm_head'):
            self.base_model.lm_head.register_backward_hook(gradient_hook)
    
    def forward(self, inputs_embeds: torch.Tensor = None, input_ids: torch.Tensor = None, **kwargs):
        if inputs_embeds is None and input_ids is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided")
        if inputs_embeds is not None and self.training and not inputs_embeds.requires_grad:
            warnings.warn("PseudoAPI: inputs_embeds doesn't require gradients!")
        if inputs_embeds is not None:
            outputs = self.base_model(inputs_embeds=inputs_embeds, **kwargs)
        else:
            outputs = self.base_model(input_ids=input_ids, **kwargs)
        return outputs
    
    def verify_parameters_unchanged(self) -> bool:
        for name, param in self.base_model.named_parameters():
            current_checksum = param.data.sum().item()
            original_checksum = self.original_param_checksums.get(name)
            if original_checksum is not None and abs(current_checksum - original_checksum) > 1e-10:
                warnings.warn(f"Parameter {name} may have changed!")
                return False
        return True
    
    def verify_weights_unchanged(self) -> bool:
        return self.verify_parameters_unchanged()
    
    def get_gradient_stats(self) -> Dict[str, float]:
        if not self.gradient_norms:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0, 'count': 0}
        return {
            'mean': sum(self.gradient_norms) / len(self.gradient_norms),
            'max': max(self.gradient_norms),
            'min': min(self.gradient_norms),
            'count': len(self.gradient_norms)
        }
    
    @property
    def model(self):
        return self.base_model

class GradientValidator:
    def __init__(self):
        self.validation_history = []
    
    def validate_expert_gradients(self, model, stage_name: str) -> Dict[str, Any]:
        expert_modules = ['knowledge_expert', 'synthesizer']
        results = {}
        for expert_name in expert_modules:
            if hasattr(model, expert_name):
                expert_module = getattr(model, expert_name)
                has_grad = any(p.grad is not None for p in expert_module.parameters() if p.requires_grad)
                grad_norm = sum(p.grad.norm().item() for p in expert_module.parameters() 
                               if p.grad is not None and p.requires_grad)
                results[expert_name] = {
                    'has_gradients': has_grad,
                    'gradient_norm': grad_norm
                }
        validation_record = {
            'stage': stage_name,
            'results': results
        }
        self.validation_history.append(validation_record)
        return results
    
    def validate_base_model_frozen(self, base_model) -> bool:
        return all(not p.requires_grad for p in base_model.parameters())
