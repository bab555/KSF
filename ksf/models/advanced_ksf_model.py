"""
Refactored KSF Reasoning Model with Transfer Learning Adapters

This is the top-level model that integrates the new KSF architecture and
includes optional adapter layers for cross-model transfer learning.
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.pseudo_api_wrapper import PseudoAPIWrapper
# Import the new, refactored modules
from .advanced_knowledge_expert import KnowledgeBank
from .advanced_synthesizer import SynthesizerConductor
# Import the new loss function
from ..training.losses import KsfLoss

class AdvancedKsfModel(nn.Module):
    """
    The complete, refactored KSF model, orchestrating the interaction between
    the Synthesizer Conductor and the Knowledge Bank. Includes optional adapters
    for transfer learning from a small to a large base model.
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        self._load_base_model()

        model_config = config.get('model', {})
        self.hidden_size = model_config.get('hidden_size')
        
        # Dynamic gating (for legacy compatibility)
        self.enable_dynamic_gating = False  # Set to False for now since we removed gating
        
        # --- Input Bottleneck Simulator for Robustness Training ---
        bottleneck_config = config.get('input_bottleneck_training', {})
        self.bottleneck_training_enabled = bottleneck_config.get('enabled', False)
        self.bottleneck_simulator = None
        if self.bottleneck_training_enabled:
            bottleneck_dim = bottleneck_config.get('bottleneck_dim', self.hidden_size // 2)
            self.bottleneck_probability = bottleneck_config.get('probability', 0.5)
            self.bottleneck_simulator = nn.Sequential(
                nn.Linear(self.hidden_size, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, self.hidden_size)
            ).to(self.device)
            self.logger.info(f"✅ Input Bottleneck Training enabled with prob={self.bottleneck_probability} and dim={bottleneck_dim}.")

        # --- Transfer Mode Adapters ---
        self.transfer_config = config.get('transfer_mode', {})
        self.transfer_mode_enabled = self.transfer_config.get('enabled', False)
        
        self.downsampler_adapter = None
        self.upsampler_adapter = None

        if self.transfer_mode_enabled:
            large_hidden_size = self.transfer_config.get('large_hidden_size')
            if not large_hidden_size:
                raise ValueError("large_hidden_size must be specified in transfer_mode config.")
            
            self.downsampler_adapter = nn.Linear(large_hidden_size, self.hidden_size).to(self.device)
            self.upsampler_adapter = nn.Linear(self.hidden_size, large_hidden_size).to(self.device)
            self.logger.info(f"✅ Transfer Mode enabled. Adapters created: {large_hidden_size} <-> {self.hidden_size}")

        # The trainable K and S expert modules
        self.knowledge_bank = KnowledgeBank(model_config)
        self.synthesizer = SynthesizerConductor(model_config)
        
        # Auto-load injected knowledge if available
        self._load_injected_knowledge_if_available()
        
        # Loss function
        self.loss_fn = KsfLoss(config)

        # Ensure all modules use the same dtype as the base model
        self._ensure_dtype_consistency()
        
        self.logger.info("✅ KSF V2 Model initialized with SynthesizerConductor and KnowledgeBank.")

    def _load_injected_knowledge_if_available(self):
        """
        Auto-load injected knowledge bank if the checkpoint exists.
        """
        import os
        from pathlib import Path
        
        # Check if injected knowledge checkpoint exists
        checkpoint_path = Path("checkpoints/ksf_v2/injected_knowledge_bank.pt")
        if checkpoint_path.exists():
            self.logger.info(f"🔄 Loading injected knowledge from: {checkpoint_path}")
            try:
                # Load the state dict
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Load only the knowledge bank part
                knowledge_bank_state = {}
                for key, value in checkpoint.items():
                    if key.startswith('knowledge_bank.'):
                        # Remove the 'knowledge_bank.' prefix
                        new_key = key[len('knowledge_bank.'):]
                        knowledge_bank_state[new_key] = value
                
                # Load the state dict into the knowledge bank
                self.knowledge_bank.load_state_dict(knowledge_bank_state, strict=False)
                self.logger.info("✅ Injected knowledge bank loaded successfully.")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load injected knowledge: {e}")
                self.logger.info("🔄 Continuing with randomly initialized knowledge bank.")
        else:
            self.logger.info("ℹ️ No injected knowledge checkpoint found. Using randomly initialized knowledge bank.")

    def _ensure_dtype_consistency(self):
        """
        Ensures all custom modules use the same dtype and device as the base model.
        """
        target_dtype = self.pseudo_api.base_model.dtype
        target_device = self.device
        self.logger.info(f"🔧 Ensuring all modules use dtype: {target_dtype} and device: {target_device}")
        
        # Convert K and S modules to correct device and dtype
        self.knowledge_bank = self.knowledge_bank.to(device=target_device, dtype=target_dtype)
        self.synthesizer = self.synthesizer.to(device=target_device, dtype=target_dtype)
        self.loss_fn = self.loss_fn.to(device=target_device, dtype=target_dtype)
        
        # Convert optional modules if they exist
        if self.bottleneck_simulator is not None:
            self.bottleneck_simulator = self.bottleneck_simulator.to(device=target_device, dtype=target_dtype)
        if self.downsampler_adapter is not None:
            self.downsampler_adapter = self.downsampler_adapter.to(device=target_device, dtype=target_dtype)
        if self.upsampler_adapter is not None:
            self.upsampler_adapter = self.upsampler_adapter.to(device=target_device, dtype=target_dtype)
            
        self.logger.info("✅ All modules converted to consistent dtype and device.")

    def _load_base_model(self):
        try:
            model_config = self.config.get('base_model', {})
            path = model_config.get('path')
            
            base_model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=getattr(torch, model_config.get('torch_dtype', 'bfloat16')),
                device_map="auto",
                trust_remote_code=True
            )
            self.pseudo_api = PseudoAPIWrapper(base_model, enable_monitoring=True)
            self.model_config = self.pseudo_api.base_model.config
            self.device = next(self.pseudo_api.base_model.parameters()).device
            self.logger.info(f"✅ Base model '{path}' loaded and wrapped with PseudoAPI.")
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}", exc_info=True)
            raise e

    def get_trainable_parameters(self):
        """
        Returns trainable parameters. In transfer mode, this can be configured
        to return only the adapter parameters.
        """
        if self.transfer_mode_enabled and self.transfer_config.get('freeze_experts', True):
            self.logger.info("Transfer mode: Returning only adapter and gate generator parameters for training.")
            params = list(self.downsampler_adapter.parameters()) + \
                     list(self.upsampler_adapter.parameters())
        else:
            self.logger.info("Native mode: Returning expert module parameters for training.")
            params = list(self.knowledge_bank.parameters()) + \
                     list(self.synthesizer.parameters())

        if self.enable_dynamic_gating:
            params += list(self.gate_generator.parameters())
            
        return params

    def forward(
        self,
        query_input_ids: torch.LongTensor,
        query_attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        summary_labels: Optional[torch.LongTensor] = None,
        summary_attention_mask: Optional[torch.LongTensor] = None
    ) -> CausalLMOutputWithPast:
        
        # --- Step 1: Get initial embeddings from the query ---
        query_embeddings = self.pseudo_api.get_embeddings(query_input_ids)

        # --- Optional: Input Bottleneck Training Logic ---
        if self.training and self.bottleneck_training_enabled and torch.rand(1).item() < self.bottleneck_probability:
            query_embeddings = self.bottleneck_simulator(query_embeddings)

        # --- Optional: Adapter Logic: Downsampling ---
        if self.transfer_mode_enabled:
            processed_query_embeddings = self.downsampler_adapter(query_embeddings)
        else:
            processed_query_embeddings = query_embeddings

        # --- Step 2: Query the Knowledge Bank (S->K part of the S->K->S loop) ---
        # The Synthesizer first processes the query to decide "what to ask".
        s_output_pre_k = self.synthesizer(
            query_hidden_states=processed_query_embeddings, 
            query_attention_mask=query_attention_mask
        )
        internal_query_vector = s_output_pre_k["internal_query_vector"]
        
        # The Knowledge Bank retrieves relevant knowledge based on the query vector.
        k_output = self.knowledge_bank(internal_query_vector)
        retrieved_memory_embedding = k_output["retrieved_memory_embedding"]
        vocab_bias_internal = k_output["vocab_bias_internal"]

        # --- Step 3: Final Synthesis with retrieved knowledge (K->S part of the loop) ---
        # The Synthesizer now takes the original query and the retrieved memory
        # to "think" (summarize) and produce the final guided hidden states.
        s_output_post_k = self.synthesizer(
            query_hidden_states=processed_query_embeddings,
            query_attention_mask=query_attention_mask,
            retrieved_memory_embedding=retrieved_memory_embedding
        )
        final_hidden_states = s_output_post_k["final_hidden_states"]
        summary_vector = s_output_post_k["summary_vector"]

        # --- Optional: Adapter Logic: Upsampling ---
        if self.transfer_mode_enabled:
            # The final hidden states are upsampled back to the base model's dimension
            final_hidden_states = self.upsampler_adapter(final_hidden_states)
        
        # --- Step 4: Generate base logits ---
        # We pass the final, synthesized hidden states to the base model's LM head.
        base_logits = self.pseudo_api.base_model.lm_head(final_hidden_states)

        # --- Step 5: Calculate Loss ---
        loss = None
        if labels is not None:
            # The loss function now handles both the main LM loss and the auxiliary summary loss.
            
            # For summary loss, we need to get the embeddings of the ground truth summary
            summary_label_embeddings = None
            if summary_labels is not None:
                with torch.no_grad(): # Don't track gradients for label embedding
                    summary_label_embeddings = self.pseudo_api.get_embeddings(summary_labels)

            loss = self.loss_fn(
                base_logits=base_logits, 
                labels=labels, 
                vocab_bias=vocab_bias_internal,
                predicted_summary_vector=summary_vector,
                target_summary_embeddings=summary_label_embeddings,
                summary_attention_mask=summary_attention_mask
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=base_logits,
            past_key_values=None, # Not handling past_key_values in this simplified forward pass
            hidden_states=final_hidden_states,
            attentions=None,
        ) 