"""
Knowledge Bank (K) Module for KSF

This module acts as a parameterized, trainable knowledge base.
It retrieves relevant information from its internal memory based on a query 
from the S-module and provides both a direct vocabulary bias and a rich 
memory embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging
from .base_expert import ExpertModule

# We'll use the sentence-transformers library for easy access to embedding models.
# This needs to be added to requirements.txt
from sentence_transformers import SentenceTransformer


class VocabBiasGenerator(nn.Module):
    """
    Generates a vocabulary bias vector from a knowledge representation.
    This is the core mechanism for injecting parameterized knowledge.
    """
    def __init__(self, hidden_size: int, vocab_size: int, bias_strength: float = 1.0):
        super().__init__()
        self.bias_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size)
        )
        self.bias_strength = bias_strength

    def forward(self, knowledge_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            knowledge_repr (torch.Tensor): Global knowledge vector [batch, hidden_size].
        
        Returns:
            torch.Tensor: Vocabulary bias vector [batch, vocab_size].
        """
        raw_bias = self.bias_network(knowledge_repr)
        # Use tanh to keep the bias values in a stable range [-1, 1]
        # and scale by a strength factor.
        return torch.tanh(raw_bias) * self.bias_strength


class KnowledgeBank(ExpertModule):
    """
    The K-Module, reimagined as a trainable Knowledge Bank.
    It uses cross-attention to retrieve from an internal memory matrix, which
    can be pre-loaded with knowledge embeddings (knowledge injection).
    """
    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, role_name="knowledge_bank")
        
        k_module_config = config.get('knowledge_bank', {})
        self.memory_matrix_size = k_module_config.get('memory_matrix_size', 4096)
        
        injection_config = k_module_config.get('knowledge_injection', {})
        self.injection_enabled = injection_config.get('embedding_model_id') is not None
        
        # The core of the knowledge bank: a memory matrix.
        # If knowledge injection is enabled, it's a non-trainable buffer.
        # Otherwise, it's a trainable parameter.
        if self.injection_enabled:
            # When injecting, the memory is fixed. We use a buffer.
            self.register_buffer(
                "memory_matrix", 
                torch.zeros(self.memory_matrix_size, self.hidden_size)
            )
            self.logger.info("KnowledgeBank initialized in 'Injection Mode'. Memory matrix is a non-trainable buffer.")
        else:
            # When learning from scratch, the memory is a trainable parameter.
            self.memory_matrix = nn.Parameter(
                torch.randn(self.memory_matrix_size, self.hidden_size)
            )
            nn.init.xavier_uniform_(self.memory_matrix)
            self.logger.info("KnowledgeBank initialized in 'Trainable Mode'. Memory matrix is a trainable parameter.")

        # A cross-attention mechanism to retrieve from memory.
        self.memory_retriever = nn.MultiheadAttention(
            embed_dim=self.hidden_size, # Query dimension
            kdim=self.hidden_size,      # Key dimension (matches hidden_size now)
            vdim=self.hidden_size,      # Value dimension (matches hidden_size now)
            num_heads=k_module_config.get('knowledge_heads', 8),
            dropout=self.dropout_rate,
            batch_first=True
        )

        # A feed-forward network to process the retrieved memory.
        self.memory_processor = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size)
        )

        # The module that generates the final vocabulary bias.
        self.vocab_bias_generator = VocabBiasGenerator(
            hidden_size=self.hidden_size,
            vocab_size=config.get('vocab_size', 151669), # Get vocab_size from config with corrected default
            bias_strength=k_module_config.get('bias_strength', 1.0)
        )
        
        self.logger.info(f"✅ Knowledge Bank initialized with {self.memory_matrix_size} memories.")

    def inject_knowledge_from_file(self, config: Dict[str, Any]):
        """
        Injects knowledge into the memory_matrix from a text file using a
        specified sentence embedding model.
        """
        if not self.injection_enabled:
            self.logger.warning("Knowledge injection called, but not enabled in config. Skipping.")
            return

        k_module_config = config.get('model', {}).get('knowledge_bank', {})
        injection_config = k_module_config.get('knowledge_injection', {})
        embedding_model_id = injection_config.get('embedding_model_id')
        
        data_config = config.get('data', {})
        knowledge_file_path = data_config.get('knowledge_path') # New config entry

        if not knowledge_file_path:
            raise ValueError("`data.knowledge_path` must be specified in the config for injection.")

        self.logger.info(f"🧠 Starting knowledge injection from '{knowledge_file_path}'...")
        self.logger.info(f"   Using embedding model: '{embedding_model_id}'")
        
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"   Using device: {device}")
            
            # Load from local path instead of downloading
            embedder = SentenceTransformer(embedding_model_id, device=device, local_files_only=True)
            
            with open(knowledge_file_path, 'r', encoding='utf-8') as f:
                knowledge_lines = [line.strip() for line in f if line.strip()]

            if not knowledge_lines:
                self.logger.warning("Knowledge file is empty. No knowledge injected.")
                return

            self.logger.info(f"   Found {len(knowledge_lines)} knowledge entries.")
            
            # Generate embeddings
            knowledge_embeddings = embedder.encode(
                knowledge_lines, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                device=device
            )

            # Clear the embedding model from memory to save resources
            del embedder
            torch.cuda.empty_cache()

            num_to_inject = min(len(knowledge_lines), self.memory_matrix_size)
            if len(knowledge_lines) > self.memory_matrix_size:
                self.logger.warning(
                    f"Found {len(knowledge_lines)} knowledge entries, but memory matrix "
                    f"only has space for {self.memory_matrix_size}. Truncating."
                )
            
            # Fill the memory matrix
            self.memory_matrix.data[:num_to_inject] = knowledge_embeddings[:num_to_inject].to(self.memory_matrix.device)
            
            self.logger.info(f"✅ Successfully injected {num_to_inject} knowledge vectors into the KnowledgeBank.")

        except Exception as e:
            self.logger.error(f"❌ Knowledge injection failed: {e}", exc_info=True)
            raise

    def forward(
        self,
        internal_query_vector: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for the Knowledge Bank.

        Args:
            internal_query_vector (torch.Tensor): The query vector from the S-module,
                                                  Shape: [batch, hidden_size].

        Returns:
            A dictionary containing:
            - 'vocab_bias_internal': The vocabulary bias from internal memory [batch, vocab_size].
            - 'retrieved_memory_embedding': The rich embedding of the retrieved memory 
                                             [batch, hidden_size].
        """
        batch_size = internal_query_vector.shape[0]

        # Reshape query for MultiheadAttention: [batch, 1, hidden_size]
        query = internal_query_vector.unsqueeze(1)
        
        # Expand memory matrix for batch processing: [batch, num_memories, memory_dim]
        memory_bank = self.memory_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Step 1: Retrieve from memory using cross-attention.
        # Query: internal_query_vector, Key & Value: memory_matrix
        retrieved_memory_embedding, _ = self.memory_retriever(
            query=query,
            key=memory_bank,
            value=memory_bank
        )
        # Squeeze out the sequence length dimension: [batch, hidden_size]
        retrieved_memory_embedding = retrieved_memory_embedding.squeeze(1)

        # Step 2: Process the retrieved memory through a feed-forward network.
        processed_memory = self.memory_processor(retrieved_memory_embedding)
        
        # Step 3: Generate the final vocabulary bias from the processed memory.
        vocab_bias_internal = self.vocab_bias_generator(processed_memory)

        return {
            "vocab_bias_internal": vocab_bias_internal,
            "retrieved_memory_embedding": processed_memory
        }
    
    def get_memory_utilization(self) -> float:
        """
        Calculates a simple metric for memory utilization based on variance.
        """
        with torch.no_grad():
            # High variance across memories suggests diverse storage.
            return torch.mean(torch.var(self.memory_matrix, dim=0)).item() 