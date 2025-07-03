"""
Synthesizer Conductor (S) Module for KSF V2

This module is the master conductor of the KSF framework. It embodies the
"thinking" and "summarizing" capabilities. It processes the user's query,
generates a query for the KnowledgeBank, receives retrieved knowledge,
summarizes the combined information into a "summary_vector", and finally
fuses this summary back into the main reasoning path to guide the final
text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from .base_expert import ExpertModule


class SynthesizerConductor(ExpertModule):
    """
    The advanced S-Module for KSF V2, acting as the 'Conductor'.
    It orchestrates the S->K->S information flow with a dedicated summarization stage.
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, role_name="synthesizer_conductor")
        
        # --- 1. Global Representation Helper ---
        # This layer helps in creating a single vector representation from token sequences.
        # It uses a learned weighted average instead of simple averaging.
        self.global_pooling_attention = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # --- 2. Summarizer Head ---
        # This is the core of the S-Module's "thinking" process.
        # It takes the query's global representation and the retrieved memory
        # to produce a condensed "summary_vector".
        # Input: hidden_size (query) + hidden_size (memory)
        # Output: hidden_size (summary)
        summarizer_config = config.get('summarizer_head', {})
        self.summarizer_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.intermediate_size),
            nn.GELU(),
            nn.LayerNorm(self.intermediate_size),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.Tanh() # Tanh to stabilize the summary vector's values
        )

        # --- 3. Guidance Fusion Mechanism ---
        # Fuses the original query's hidden states with the summary_vector.
        # This uses cross-attention where the query is the sequence and the summary is the context.
        self.guidance_fusion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.get('synthesizer_heads', 8),
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.guidance_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        self.logger.info("✅ Synthesizer Conductor (v2 - with Summarizer) initialized.")

    def _get_global_representation(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Creates a global vector from token embeddings using a learned weighted average.
        """
        # Get attention scores for each token
        scores = self.global_pooling_attention(hidden_states).squeeze(-1) # [batch, seq_len]

        if attention_mask is not None:
            # Apply mask before softmax
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # Get weights
        weights = F.softmax(scores, dim=1).unsqueeze(-1) # [batch, seq_len, 1]
        
        # Compute weighted average
        global_repr = torch.sum(hidden_states * weights, dim=1) # [batch, hidden_size]
        return global_repr

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        retrieved_memory_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        
        # --- Stage 1: Initial Query Processing & K-Module Query Generation ---
        # Create a global representation of the query to be sent to the Knowledge Bank.
        global_query_repr = self._get_global_representation(query_hidden_states, query_attention_mask)
        internal_query_vector = global_query_repr

        # --- Stage 2: Summarization (S-Module's "Thinking" phase) ---
        # This stage occurs after the K-Module returns the retrieved memory.
        if retrieved_memory_embedding is not None:
            # Fuse the query's essence with the retrieved knowledge.
            summarizer_input = torch.cat([global_query_repr, retrieved_memory_embedding], dim=1)
            summary_vector = self.summarizer_head(summarizer_input)
        else:
            # If no memory is retrieved, the summary is based solely on the query.
            # A zero tensor is used as a placeholder for the memory part.
            dummy_memory = torch.zeros_like(global_query_repr)
            summarizer_input = torch.cat([global_query_repr, dummy_memory], dim=1)
            summary_vector = self.summarizer_head(summarizer_input)

        # --- Stage 3: Guidance Fusion ---
        # The summary_vector acts as a condensed guide. We fuse it back into the
        # original query's token-level hidden states using cross-attention.
        summary_context = summary_vector.unsqueeze(1) # [batch, 1, hidden_size]
        
        # The query_hidden_states (Q) attend to the summary_vector (K, V).
        guided_hidden_states, _ = self.guidance_fusion_attention(
            query=query_hidden_states, 
            key=summary_context, 
            value=summary_context
        )
        
        # Residual connection and feed-forward network.
        fused_hidden_states = self.guidance_ffn(guided_hidden_states + query_hidden_states)

        # --- Assemble final output ---
        output = {
            # Primary output for final logit calculation.
            "final_hidden_states": fused_hidden_states,
            # Query vector sent to the K-Module.
            "internal_query_vector": internal_query_vector,
            # Output of the summarizer head, for auxiliary loss calculation.
            "summary_vector": summary_vector,
        }
            
        return output
    
    def compute_synthesis_metrics(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Computes metrics related to the synthesis process.
        
        Args:
            output_dict: The output dictionary from the forward method.
            
        Returns:
            A dictionary of computed metrics.
        """
        with torch.no_grad():
            metrics = {}
            
            # Stability of the final hidden states
            final_hidden_states = output_dict['final_hidden_states']
            stability = 1.0 / (1.0 + final_hidden_states.var().item())
            metrics['synthesis_stability'] = stability
            
            # Norm of the summary vector as an indicator of its magnitude
            summary_vector_norm = output_dict['summary_vector'].norm(p=2, dim=1).mean().item()
            metrics['summary_vector_norm'] = summary_vector_norm
            
        return metrics 