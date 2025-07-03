"""
Base Expert Module - KSF专家模块基类
为KSF的K-S架构提供基础接口规范
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class ExpertModule(nn.Module, ABC):
    """
    KSF专家模块基类
    为Knowledge Expert和Synthesizer提供统一接口
    """
    
    def __init__(self, config: Dict[str, Any], role_name: str = "expert"):
        super().__init__()
        self.config = config
        self.role_name = role_name
        self.logger = logging.getLogger(f"{__name__}.{role_name}")
        
        # 基础配置
        self.hidden_size = config.get('hidden_size', 2560)
        self.intermediate_size = config.get('intermediate_size', 10240)
        self.dropout_rate = config.get('dropout', 0.1)
        
        # 角色嵌入向量 - 用于专家识别
        self.role_embedding_dim = config.get('role_embedding_dim', 128)
        self.register_buffer(
            'role_embedding',
            torch.randn(self.role_embedding_dim) * 0.02
        )
        self.role_embedding = nn.Parameter(self.role_embedding.clone())
        
        # dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.logger.info(f"✅ {role_name} Expert initialized with hidden_size={self.hidden_size}")
        
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            hidden_states: 输入隐状态 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            output_hidden_states: 专家输出隐状态 [batch_size, seq_len, hidden_size]
        """
        raise NotImplementedError("子类必须实现forward方法")
    
    def get_role_embedding(self) -> torch.Tensor:
        """
        返回角色特征向量
        
        Returns:
            role_embedding: 角色嵌入向量 [role_embedding_dim]
        """
        return self.role_embedding
    
    def get_expert_statistics(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        计算专家统计信息
        
        Args:
            hidden_states: 专家输出的隐状态 [batch_size, seq_len, hidden_size]
            
        Returns:
            统计信息字典
        """
        with torch.no_grad():
            stats = {
                f'{self.role_name}_mean_activation': hidden_states.mean().item(),
                f'{self.role_name}_std_activation': hidden_states.std().item(),
                f'{self.role_name}_max_activation': hidden_states.max().item(),
                f'{self.role_name}_min_activation': hidden_states.min().item(),
                f'{self.role_name}_sparsity': (hidden_states.abs() < 1e-6).float().mean().item()
            }
        return stats
    
    def check_gradient_flow(self) -> Dict[str, float]:
        """
        检查梯度流情况
        
        Returns:
            梯度统计信息字典
        """
        grad_stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats[f'{self.role_name}_{name}_grad_norm'] = grad_norm
            else:
                grad_stats[f'{self.role_name}_{name}_grad_norm'] = 0.0
        return grad_stats
    
    def compute_output_quality(self, hidden_states: torch.Tensor) -> Dict[str, float]:
        """
        计算输出质量指标
        
        Args:
            hidden_states: 专家输出隐状态
            
        Returns:
            质量指标字典
        """
        with torch.no_grad():
            quality_metrics = {}
            
            # 输出稳定性
            if hidden_states.size(1) > 1:
                temporal_var = hidden_states.var(dim=1).mean().item()
                quality_metrics[f'{self.role_name}_temporal_stability'] = 1.0 / (1.0 + temporal_var)
            
            # 输出多样性
            if hidden_states.size(0) > 1:
                batch_flat = hidden_states.view(hidden_states.size(0), -1)
                pairwise_sim = F.cosine_similarity(
                    batch_flat.unsqueeze(1), 
                    batch_flat.unsqueeze(0), 
                    dim=2
                )
                mask = ~torch.eye(pairwise_sim.size(0), dtype=torch.bool, device=pairwise_sim.device)
                diversity = 1.0 - pairwise_sim[mask].mean().item()
                quality_metrics[f'{self.role_name}_output_diversity'] = diversity
            
            # 激活健康度
            activation_health = 1.0 - torch.isnan(hidden_states).float().mean().item()
            quality_metrics[f'{self.role_name}_activation_health'] = activation_health
            
        return quality_metrics


def compute_knowledge_synthesis_alignment(knowledge_hidden: torch.Tensor, 
                                        synthesis_hidden: torch.Tensor) -> Dict[str, float]:
    """
    计算知识专家和合成器的对齐度
    
    Args:
        knowledge_hidden: 知识专家隐状态 [batch_size, seq_len, hidden_size]
        synthesis_hidden: 合成器隐状态 [batch_size, seq_len, hidden_size]
        
    Returns:
        对齐度指标字典
    """
    with torch.no_grad():
        metrics = {}
        
        # 扁平化处理
        k_flat = knowledge_hidden.view(knowledge_hidden.size(0), -1)
        s_flat = synthesis_hidden.view(synthesis_hidden.size(0), -1)
        
        # 余弦相似度
        cosine_sim = F.cosine_similarity(k_flat, s_flat, dim=1).mean().item()
        metrics['knowledge_synthesis_alignment'] = cosine_sim
        
        # 信息保留度
        k_norm = k_flat.norm(dim=1).mean().item()
        s_norm = s_flat.norm(dim=1).mean().item()
        if k_norm > 0:
            info_retention = min(s_norm / k_norm, 1.0)
            metrics['knowledge_info_retention'] = info_retention
        
        # 合成质量（基于方差）
        synthesis_quality = 1.0 / (1.0 + synthesis_hidden.var().item())
        metrics['synthesis_quality'] = synthesis_quality
        
    return metrics


class KnowledgeQualityValidator:
    """
    知识质量验证器
    用于监控K-S架构的训练质量
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.KnowledgeQualityValidator")
        
        # 质量阈值
        self.thresholds = config.get('thresholds', {
            'knowledge_influence_min': 0.1,
            'synthesis_smoothness_max': 0.5,
            'knowledge_diversity_min': 0.3
        })
        
        self.validation_history = []
    
    def validate_training_step(self, 
                             knowledge_output: torch.Tensor,
                             synthesis_output: torch.Tensor,
                             step: int) -> Dict[str, Any]:
        """
        验证单步训练质量
        
        Args:
            knowledge_output: 知识专家输出
            synthesis_output: 合成器输出
            step: 训练步数
            
        Returns:
            验证结果字典
        """
        validation_result = {
            'step': step,
            'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None,
            'quality_passed': True,
            'warnings': [],
            'metrics': {}
        }
        
        # 计算对齐度指标
        alignment_metrics = compute_knowledge_synthesis_alignment(knowledge_output, synthesis_output)
        validation_result['metrics'].update(alignment_metrics)
        
        # 检查质量阈值
        knowledge_influence = alignment_metrics.get('knowledge_synthesis_alignment', 0.0)
        if knowledge_influence < self.thresholds['knowledge_influence_min']:
            validation_result['warnings'].append(
                f"Knowledge influence ({knowledge_influence:.3f}) below threshold "
                f"({self.thresholds['knowledge_influence_min']})"
            )
            validation_result['quality_passed'] = False
        
        # 检查合成质量
        synthesis_quality = alignment_metrics.get('synthesis_quality', 0.0)
        if synthesis_quality < 0.5:  # 基础质量阈值
            validation_result['warnings'].append(
                f"Synthesis quality ({synthesis_quality:.3f}) below threshold (0.5)"
            )
        
        # 记录验证历史
        self.validation_history.append(validation_result)
        
        # 保持历史记录在合理范围内
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return validation_result
    
    def get_training_health_report(self) -> Dict[str, Any]:
        """
        获取训练健康度报告
        
        Returns:
            健康度报告字典
        """
        if not self.validation_history:
            return {'status': 'no_data', 'message': 'No validation history available'}
        
        recent_validations = self.validation_history[-100:]  # 最近100步
        
        health_report = {
            'total_validations': len(self.validation_history),
            'recent_validations': len(recent_validations),
            'quality_pass_rate': sum(1 for v in recent_validations if v['quality_passed']) / len(recent_validations),
            'common_warnings': {},
            'trend_analysis': {}
        }
        
        # 统计常见警告
        all_warnings = []
        for validation in recent_validations:
            all_warnings.extend(validation.get('warnings', []))
        
        from collections import Counter
        warning_counts = Counter(all_warnings)
        health_report['common_warnings'] = dict(warning_counts.most_common(5))
        
        return health_report 