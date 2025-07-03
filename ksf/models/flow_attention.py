"""
Flow Attention Module for KSF

A placeholder implementation of a streaming-capable, linear-complexity
attention mechanism, designed to be guided by an external relevance vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FlowAttention(nn.Module):
    """
    A simplified linear attention mechanism (Flow Attention).

    This attention has a complexity of O(N * d^2), making it suitable for
    long sequences. It is designed to accept an external guidance bias,
    which allows the `CrossContextGuider` to influence its behavior.
    """
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            hidden_size (int): The hidden size of the model.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        guidance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Flow Attention.

        Args:
            hidden_states (torch.Tensor): Input tensor [batch, seq_len, hidden_size].
            attention_mask (Optional[torch.Tensor]): Mask to prevent attention to padding tokens 
                                                      [batch, seq_len] or [batch, seq_len, seq_len].
            guidance_bias (Optional[torch.Tensor]): External guidance from the CrossContextGuider
                                                    [batch, seq_len]. It will be added to the
                                                    attention scores before softmax.

        Returns:
            torch.Tensor: The output of the attention mechanism [batch, seq_len, hidden_size].
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project and reshape for multi-head attention
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Linear attention computation: QK^T V -> φ(Q) (φ(K)^T V)
        # Using elu as the feature map φ for numerical stability.
        q_mapped = F.elu(q) + 1.0
        k_mapped = F.elu(k) + 1.0

        # Apply guidance bias to the keys. This scales the keys based on their relevance.
        if guidance_bias is not None:
            # Reshape bias from [batch, seq_len] to [batch, 1, seq_len, 1] for broadcasting.
            # Using exp() to convert the bias scores into a positive, multiplicative scaling factor.
            # This makes "more relevant" tokens have a stronger voice.
            bias = guidance_bias.unsqueeze(1).unsqueeze(-1)
            k_mapped = k_mapped * torch.exp(bias)

        # Create attention mask if it's not in the right format
        if attention_mask is not None:
            if attention_mask.dim() == 2: # [batch, seq_len]
                # Expand mask for heads and head_dim dimensions to match k_mapped shape
                # k_mapped shape: [batch, num_heads, seq_len, head_dim]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(-1) # [batch, 1, seq_len, 1]
            # Apply mask to keys
            k_mapped = k_mapped.masked_fill(attention_mask == 0, 0)
        
        # This is the key step of linear attention
        kv_context = torch.einsum('bhsd,bhsv->bhdv', k_mapped, v)
        
        # The scaling factor can be learned or fixed. Here we use a simple sum.
        # This prevents the values from growing too large.
        scaling_factor = 1.0 / torch.einsum('bhsd,bhd->bhs', q_mapped, torch.sum(k_mapped, dim=2))
        scaling_factor = scaling_factor.nan_to_num(0.0).unsqueeze(-1)
        
        # Apply scaling to query
        q_scaled = q_mapped * scaling_factor
        
        # Compute final output
        attn_output = torch.einsum('bhsd,bhdv->bhsv', q_scaled, kv_context)
        
        # Reshape back to the original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final output projection
        output = self.out_proj(attn_output)

        return self.dropout(output) 