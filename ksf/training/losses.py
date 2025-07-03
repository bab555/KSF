"""
Custom Loss Functions for the KSF V2

This module defines specialized loss functions for the KSF V2 architecture,
including the main task loss and an auxiliary loss for the summarizer head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import logging


class SummaryLoss(nn.Module):
    """
    Computes an auxiliary loss to train the SummarizerHead.
    This loss encourages the predicted summary_vector to be semantically
    similar to the embedding of the ground-truth summary. We use CosineEmbeddingLoss
    as it is effective for comparing high-dimensional vectors.
    """
    def __init__(self, weight: float = 1.0, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if not self.enabled:
            return
            
        self.weight = weight
        # CosineEmbeddingLoss expects a target of 1 (for similarity) or -1 (for dissimilarity).
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.1)

    def forward(self, 
                predicted_summary_vector: torch.Tensor, 
                target_summary_embeddings: torch.Tensor,
                summary_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled or target_summary_embeddings is None:
            return torch.tensor(0.0, device=predicted_summary_vector.device)

        # To get a single target vector, we average the embeddings of the target summary, respecting the mask.
        if summary_attention_mask is not None:
            mask = summary_attention_mask.unsqueeze(-1).expand_as(target_summary_embeddings)
            masked_embeddings = target_summary_embeddings * mask
            target_vector = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            target_vector = target_summary_embeddings.mean(dim=1)

        # We want the vectors to be similar, so the target for CosineEmbeddingLoss is a tensor of 1s.
        target_similarity = torch.ones(predicted_summary_vector.size(0), device=predicted_summary_vector.device)
        
        loss = self.loss_fn(predicted_summary_vector, target_vector, target_similarity)
        return self.weight * loss


class KsfLoss(nn.Module):
    """
    The main loss function for KSF V2, combining the primary language modeling 
    task loss with an auxiliary loss for the summarization module.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        loss_config = config.get('loss', {})
        self.main_loss_weight = loss_config.get('main_weight', 1.0)
        
        # The primary task loss (Cross Entropy)
        self.main_loss_fn = nn.CrossEntropyLoss()

        # Auxiliary loss for the summarizer head
        summary_loss_config = loss_config.get('summary_loss', {})
        self.summary_loss_enabled = summary_loss_config.get('enabled', False)
        if self.summary_loss_enabled:
            self.summary_loss_fn = SummaryLoss(
                weight=summary_loss_config.get('weight', 0.1),
                enabled=True
            )
            self.logger.info("✅ Auxiliary Summary Loss enabled.")

        self.logger.info("✅ KSF V2 Loss module initialized.")

    def forward(
        self, 
        base_logits: torch.Tensor, 
        labels: torch.Tensor,
        vocab_bias: Optional[torch.Tensor] = None,
        predicted_summary_vector: Optional[torch.Tensor] = None,
        target_summary_embeddings: Optional[torch.Tensor] = None,
        summary_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the total loss for the KSF V2 model.
        """
        # --- 1. Main Task Loss ---
        # Add the vocabulary bias from the K-Module before computing the loss.
        if vocab_bias is not None:
            # Expand vocab_bias to match the sequence length of logits
            final_logits = base_logits + vocab_bias.unsqueeze(1)
        else:
            final_logits = base_logits

        # Standard cross-entropy loss calculation
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        main_loss = self.main_loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # --- 2. Auxiliary Summary Loss ---
        summary_loss = torch.tensor(0.0, device=main_loss.device)
        if self.summary_loss_enabled and predicted_summary_vector is not None:
            summary_loss = self.summary_loss_fn(
                predicted_summary_vector=predicted_summary_vector,
                target_summary_embeddings=target_summary_embeddings,
                summary_attention_mask=summary_attention_mask
            )
            
        # --- 3. Total Weighted Loss ---
        total_loss = (self.main_loss_weight * main_loss) + summary_loss
        
        # For logging purposes, we can return a dict, but the primary output is the total loss.
        # This will be simplified later if a more complex logging mechanism is needed.
        return total_loss


class FocalLoss(nn.Module):
    """
    Focal Loss - 用于处理类别不平衡
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ContrastiveLoss(nn.Module):
    """
    对比学习损失 - 用于知识表征学习
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            anchor: 锚点表征 [batch_size, dim]
            positive: 正样本表征 [batch_size, dim] 
            negative: 负样本表征 [batch_size, dim]
        """
        # 标准化
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)
        
        # 计算相似度
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=-1) / self.temperature
        
        # 对比损失
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)


def compute_knowledge_quality_metrics(knowledge_hidden: torch.Tensor, 
                                     synthesis_hidden: torch.Tensor) -> Dict[str, float]:
    """
    计算知识质量相关指标
    
    Args:
        knowledge_hidden: 知识专家的隐状态
        synthesis_hidden: 合成器的隐状态
        
    Returns:
        质量指标字典
    """
    metrics = {}
    
    with torch.no_grad():
        # 知识利用率：合成器对知识的依赖程度
        knowledge_flat = knowledge_hidden.view(knowledge_hidden.size(0), -1)
        synthesis_flat = synthesis_hidden.view(synthesis_hidden.size(0), -1)
        
        knowledge_influence = F.cosine_similarity(knowledge_flat, synthesis_flat, dim=1).mean()
        metrics['knowledge_influence'] = knowledge_influence.item()
        
        # 知识多样性
        if knowledge_hidden.size(0) > 1:
            pairwise_sim = F.cosine_similarity(
                knowledge_flat.unsqueeze(1), 
                knowledge_flat.unsqueeze(0), 
                dim=2
            )
            # 排除对角线
            mask = ~torch.eye(pairwise_sim.size(0), dtype=torch.bool, device=pairwise_sim.device)
            diversity = 1.0 - pairwise_sim[mask].mean()
            metrics['knowledge_diversity'] = diversity.item()
        
        # 知识稳定性
        knowledge_std = knowledge_hidden.std(dim=1).mean()
        metrics['knowledge_stability'] = 1.0 / (1.0 + knowledge_std.item())
    
    return metrics 