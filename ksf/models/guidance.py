"""
Cross-Context Relevance Guidance Module for KSF

This module implements the core guidance mechanism for the KSF framework,
enabling bi-directional, dynamic guidance between the S and K expert modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class CrossContextGuider(nn.Module):
    """
    Implements the "Cross-Context Relevance Guidance" mechanism.

    This module computes relevance vectors that guide the attention of the
    S (Strategist) and K (Knowledge) experts based on the context of each other.
    """
    def __init__(self, hidden_size: int, embedding_dim: Optional[int] = None):
        """
        Args:
            hidden_size (int): The hidden size of the token embeddings.
            embedding_dim (int, optional): The dimension for the global context vectors. 
                                           Defaults to hidden_size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim or hidden_size

        # Projection layers to create the global context vectors.
        # This allows the model to learn the best way to "summarize" the contexts.
        self.query_proj = nn.Linear(hidden_size, self.embedding_dim)
        self.knowledge_proj = nn.Linear(hidden_size, self.embedding_dim)

    def _create_global_context_vector(self, hidden_states: torch.Tensor, 
                                      attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Creates a global context vector from a sequence of hidden states.

        Args:
            hidden_states (torch.Tensor): The token embeddings [batch, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): The attention mask [batch, seq_len].

        Returns:
            torch.Tensor: The global context vector [batch, embedding_dim].
        """
        if attention_mask is None:
            # If no mask, use simple mean pooling.
            return torch.mean(hidden_states, dim=1)
        
        # Weighted average pooling, ignoring padding tokens.
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(
        self,
        query_hidden_states: torch.Tensor,
        knowledge_hidden_states: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        knowledge_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates the bi-directional guidance vectors.

        Args:
            query_hidden_states (torch.Tensor): Token embeddings from S-module [batch, query_seq_len, hidden_size].
            knowledge_hidden_states (torch.Tensor): Token embeddings from K-module [batch, knowledge_seq_len, hidden_size].
            query_attention_mask (Optional[torch.Tensor]): Mask for the query.
            knowledge_attention_mask (Optional[torch.Tensor]): Mask for the knowledge.

        Returns:
            A dictionary containing:
            - 'relevance_for_query': Guidance vector for the S-module [batch, query_seq_len].
            - 'relevance_for_knowledge': Guidance vector for the K-module [batch, knowledge_seq_len].
        """
        # Phase 1: Create global context vectors for both query and knowledge.
        global_query_vector = self._create_global_context_vector(query_hidden_states, query_attention_mask)
        global_knowledge_vector = self._create_global_context_vector(knowledge_hidden_states, knowledge_attention_mask)

        # Project the global vectors to the target embedding dimension.
        global_query_vector_proj = self.query_proj(global_query_vector)
        global_knowledge_vector_proj = self.knowledge_proj(global_knowledge_vector)
        
        # Normalize vectors for cosine similarity calculation.
        global_query_vec_norm = F.normalize(global_query_vector_proj, p=2, dim=1)
        global_knowledge_vec_norm = F.normalize(global_knowledge_vector_proj, p=2, dim=1)
        
        query_tokens_norm = F.normalize(self.query_proj(query_hidden_states), p=2, dim=2)
        knowledge_tokens_norm = F.normalize(self.knowledge_proj(knowledge_hidden_states), p=2, dim=2)

        # Phase 2: Cross-calculate relevance (the "划重点" mechanism).
        # S guides K: Use the global query vector to find relevant tokens in the knowledge content.
        relevance_for_knowledge = torch.einsum('bd,bsd->bs', global_query_vec_norm, knowledge_tokens_norm)

        # K guides S: Use the global knowledge vector to find relevant tokens in the query.
        relevance_for_query = torch.einsum('bd,bsd->bs', global_knowledge_vec_norm, query_tokens_norm)

        # Apply masks to the relevance vectors to ensure padding tokens have zero relevance.
        if query_attention_mask is not None:
            relevance_for_query = relevance_for_query.masked_fill(query_attention_mask == 0, 0)
        
        if knowledge_attention_mask is not None:
            relevance_for_knowledge = relevance_for_knowledge.masked_fill(knowledge_attention_mask == 0, 0)

        return {
            "relevance_for_query": relevance_for_query,       # To be used by S-Module
            "relevance_for_knowledge": relevance_for_knowledge, # To be used by K-Module
            "global_query_vector": global_query_vector,
            "global_knowledge_vector": global_knowledge_vector,
        }

def compute_guidance_loss(relevance_vector: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes a loss to encourage the guidance to be discriminative (not flat).

    A good guidance signal should have high variance, indicating that some tokens
    are clearly more important than others. We encourage this by penalizing low variance.

    Args:
        relevance_vector (torch.Tensor): The relevance scores [batch, seq_len].
        attention_mask (Optional[torch.Tensor]): The attention mask to ignore padding.

    Returns:
        torch.Tensor: The guidance loss value.
    """
    if attention_mask is None:
        # If no mask, variance is calculated over the whole sequence.
        variance = torch.var(relevance_vector, dim=1)
    else:
        # Calculate variance only for non-padded tokens.
        batch_size, seq_len = relevance_vector.shape
        variances = []
        for i in range(batch_size):
            masked_relevance = relevance_vector[i, attention_mask[i] == 1]
            if len(masked_relevance) > 1:
                variances.append(torch.var(masked_relevance))
            else:
                variances.append(torch.tensor(0.0, device=relevance_vector.device))
        variance = torch.stack(variances)

    # We want to maximize variance, so we return the negative variance.
    # The 1.0 is added for numerical stability and scaling.
    loss = -torch.mean(variance) + 1.0
    return F.relu(loss) # Loss should not be negative 