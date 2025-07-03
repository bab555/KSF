"""
CoD V6 Metrics - 评估指标
专门为V6架构设计的评估指标，包括Flow质量、注意力多样性、思维质量等
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import numpy as np
from collections import defaultdict


class CoDMetrics:
    """
    CoD V6架构的评估指标类
    
    包含以下核心指标：
    1. Flow质量指标：评估信息在P-C-S之间的流动质量
    2. 注意力多样性：评估32个注意力头的多样性
    3. 角色分化指标：确保P、C、S保持不同的角色
    4. 思维质量评估：评估S生成的thinking质量
    5. 隐式传递效果：评估roundtable attention的效果
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有指标累积器"""
        self.metrics_accumulator = defaultdict(list)
        self.batch_count = 0
    
    def compute_flow_quality(self, 
                           p_hidden: torch.Tensor,
                           c_hidden: torch.Tensor,
                           s_hidden: torch.Tensor) -> Dict[str, float]:
        """
        计算Flow质量指标
        
        Args:
            p_hidden: Proposer的隐状态
            c_hidden: Challenger的隐状态  
            s_hidden: Synthesizer的隐状态
            
        Returns:
            flow质量指标字典
        """
        with torch.no_grad():
            # 信息保留度：从P到C
            p_to_c_retention = F.cosine_similarity(
                p_hidden.mean(dim=1), 
                c_hidden.mean(dim=1), 
                dim=-1
            ).mean().item()
            
            # 信息增强度：C相对于P的新信息
            c_novelty = 1.0 - p_to_c_retention
            
            # 综合整合度：S整合P和C的程度
            s_integration = (
                F.cosine_similarity(s_hidden.mean(dim=1), p_hidden.mean(dim=1), dim=-1).mean() +
                F.cosine_similarity(s_hidden.mean(dim=1), c_hidden.mean(dim=1), dim=-1).mean()
            ).item() / 2.0
            
            # Flow连贯性：整体信息流的平滑度
            flow_coherence = min(p_to_c_retention, s_integration)
            
            return {
                'p_to_c_retention': p_to_c_retention,
                'c_novelty': c_novelty,
                's_integration': s_integration,
                'flow_coherence': flow_coherence
            }
    
    def compute_attention_diversity(self,
                                  p_attention: Optional[torch.Tensor],
                                  c_attention: Optional[torch.Tensor],
                                  s_attention: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        计算注意力多样性指标
        
        Args:
            p_attention: P的注意力模式 [batch, 10, seq, seq]
            c_attention: C的注意力模式 [batch, 10, seq, seq]
            s_attention: S的注意力模式 [batch, 12, seq, seq]
            
        Returns:
            注意力多样性指标
        """
        diversity_scores = {}
        
        def compute_head_diversity(attention_weights):
            """计算单个模块内注意力头的多样性"""
            if attention_weights is None:
                return 0.0
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # 将注意力权重展平
            attn_flat = attention_weights.view(batch_size, num_heads, -1)
            
            # 计算不同头之间的相似度
            similarities = []
            for i in range(num_heads):
                for j in range(i+1, num_heads):
                    sim = F.cosine_similarity(
                        attn_flat[:, i], 
                        attn_flat[:, j], 
                        dim=-1
                    ).mean()
                    similarities.append(sim)
            
            # 多样性 = 1 - 平均相似度
            avg_similarity = torch.stack(similarities).mean().item()
            return 1.0 - avg_similarity
        
        # 计算各模块的注意力多样性
        if p_attention is not None:
            diversity_scores['p_attention_diversity'] = compute_head_diversity(p_attention)
        
        if c_attention is not None:
            diversity_scores['c_attention_diversity'] = compute_head_diversity(c_attention)
            
        if s_attention is not None:
            diversity_scores['s_attention_diversity'] = compute_head_diversity(s_attention)
        
        # 计算整体32头的多样性
        if all(x is not None for x in [p_attention, c_attention, s_attention]):
            # 拼接所有注意力头
            all_attention = torch.cat([p_attention, c_attention, s_attention], dim=1)
            diversity_scores['total_32_head_diversity'] = compute_head_diversity(all_attention)
        
        return diversity_scores
    
    def compute_role_differentiation(self,
                                   p_output: Dict[str, Any],
                                   c_output: Dict[str, Any],
                                   s_output: Dict[str, Any]) -> Dict[str, float]:
        """
        计算角色分化指标，确保P、C、S保持不同的角色
        
        Returns:
            角色分化指标
        """
        metrics = {}
        
        # 获取隐状态
        p_hidden = p_output.get('hidden_states')
        c_hidden = c_output.get('hidden_states')
        s_hidden = s_output.get('hidden_states')
        
        if all(x is not None for x in [p_hidden, c_hidden, s_hidden]):
            # 计算角色相似度（越低越好）
            with torch.no_grad():
                # 池化隐状态
                p_pooled = p_hidden.mean(dim=1)
                c_pooled = c_hidden.mean(dim=1)
                s_pooled = s_hidden.mean(dim=1)
                
                # 计算两两相似度
                sim_pc = F.cosine_similarity(p_pooled, c_pooled, dim=-1).mean().item()
                sim_ps = F.cosine_similarity(p_pooled, s_pooled, dim=-1).mean().item()
                sim_cs = F.cosine_similarity(c_pooled, s_pooled, dim=-1).mean().item()
                
                metrics['proposer_challenger_similarity'] = sim_pc
                metrics['proposer_synthesizer_similarity'] = sim_ps
                metrics['challenger_synthesizer_similarity'] = sim_cs
                
                # 角色分化度（越高越好）
                avg_similarity = (sim_pc + sim_ps + sim_cs) / 3
                metrics['role_differentiation'] = 1.0 - avg_similarity
                
                # 角色坍塌风险
                max_similarity = max(sim_pc, sim_ps, sim_cs)
                metrics['role_collapse_risk'] = max_similarity
        
        return metrics
    
    def compute_thinking_quality(self, s_output: Dict[str, Any]) -> Dict[str, float]:
        """
        评估Synthesizer生成的thinking质量
        
        Returns:
            thinking质量指标
        """
        metrics = {}
        
        thinking = s_output.get('thinking')
        if thinking is not None:
            with torch.no_grad():
                # thinking的信息熵（多样性）
                thinking_flat = thinking.view(thinking.size(0), -1)
                thinking_probs = F.softmax(thinking_flat, dim=-1)
                entropy = -torch.sum(thinking_probs * torch.log(thinking_probs + 1e-8), dim=-1)
                metrics['thinking_entropy'] = entropy.mean().item()
                
                # thinking的稀疏度
                sparsity = (thinking.abs() < 1e-6).float().mean().item()
                metrics['thinking_sparsity'] = sparsity
                
                # thinking的激活强度
                metrics['thinking_activation_mean'] = thinking.abs().mean().item()
                metrics['thinking_activation_std'] = thinking.std().item()
        
        # 从s_output中获取质量评分
        if 'thinking_quality' in s_output:
            metrics['thinking_quality_score'] = s_output['thinking_quality'].mean().item()
        
        if 'integration_scores' in s_output:
            metrics['integration_score'] = s_output['integration_scores'].mean().item()
        
        return metrics
    
    def compute_all_metrics(self,
                          outputs: Dict[str, Any],
                          targets: Optional[torch.Tensor] = None,
                          computation_time: Optional[float] = None) -> Dict[str, float]:
        """
        计算所有V6架构相关指标
        
        Args:
            outputs: 模型输出字典
            targets: 目标标签（可选）
            computation_time: 计算时间（可选）
            
        Returns:
            所有指标的字典
        """
        all_metrics = {}
        
        # 获取CoD输出
        cod_outputs = outputs.get('cod_outputs', {})
        p_output = cod_outputs.get('proposer', {})
        c_output = cod_outputs.get('challenger', {})
        s_output = cod_outputs.get('synthesizer', {})
        
        # 1. Flow质量指标
        if all('hidden_states' in x for x in [p_output, c_output, s_output]):
            flow_metrics = self.compute_flow_quality(
                p_output['hidden_states'],
                c_output['hidden_states'],
                s_output['hidden_states']
            )
            all_metrics.update(flow_metrics)
        
        # 2. 注意力多样性指标
        attention_metrics = self.compute_attention_diversity(
            p_output.get('attention_patterns'),
            c_output.get('attention_patterns'),
            s_output.get('attention_patterns')
        )
        all_metrics.update(attention_metrics)
        
        # 3. 角色分化指标
        if cod_outputs:
            role_metrics = self.compute_role_differentiation(p_output, c_output, s_output)
            all_metrics.update(role_metrics)
        
        # 4. 思维质量指标
        thinking_metrics = self.compute_thinking_quality(s_output)
        all_metrics.update(thinking_metrics)
        
        # 5. 性能指标
        if computation_time is not None:
            all_metrics['computation_time'] = computation_time
        
        # 6. 任务相关指标（如果有targets）
        if targets is not None and 'logits' in outputs:
            logits = outputs['logits']
            if logits.dim() == 3:  # 生成任务
                # 计算困惑度
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    reduction='mean'
                )
                all_metrics['perplexity'] = torch.exp(loss).item()
            else:  # 分类任务
                # 计算准确率
                predictions = torch.argmax(logits, dim=-1)
                accuracy = (predictions == targets).float().mean().item()
                all_metrics['accuracy'] = accuracy
        
        # 7. V6特定指标
        all_metrics['num_attention_heads'] = 32  # P(10) + C(10) + S(12)
        all_metrics['uses_flow_attention'] = True
        all_metrics['uses_implicit_thinking'] = self._check_implicit_thinking(outputs)
        
        # 累积指标
        self._accumulate_metrics(all_metrics)
        
        return all_metrics
    
    def _check_implicit_thinking(self, outputs: Dict[str, Any]) -> bool:
        """检查是否使用了隐式thinking传递"""
        # 可以通过检查是否有roundtable attention的调制来判断
        return outputs.get('uses_roundtable', False)
    
    def _accumulate_metrics(self, metrics: Dict[str, float]):
        """累积批次指标用于计算平均值"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_accumulator[key].append(value)
        self.batch_count += 1
    
    def get_average_metrics(self) -> Dict[str, float]:
        """获取所有批次的平均指标"""
        avg_metrics = {}
        for key, values in self.metrics_accumulator.items():
            if values:
                avg_metrics[key] = np.mean(values)
        avg_metrics['total_batches'] = self.batch_count
        return avg_metrics
    
    def print_metrics_summary(self, metrics: Dict[str, float]):
        """打印指标摘要"""
        print("\n" + "="*60)
        print("CoD V6 评估指标摘要")
        print("="*60)
        
        # Flow质量
        print("\n📊 Flow质量指标:")
        for key in ['flow_coherence', 'p_to_c_retention', 'c_novelty', 's_integration']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # 注意力多样性
        print("\n🎯 注意力多样性:")
        for key in ['total_32_head_diversity', 'p_attention_diversity', 
                    'c_attention_diversity', 's_attention_diversity']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # 角色分化
        print("\n👥 角色分化:")
        for key in ['role_differentiation', 'role_collapse_risk']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # 思维质量
        print("\n🧠 思维质量:")
        for key in ['thinking_quality_score', 'thinking_entropy', 'integration_score']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        
        # 性能指标
        print("\n⚡ 性能指标:")
        if 'computation_time' in metrics:
            print(f"  计算时间: {metrics['computation_time']:.3f}s")
        if 'perplexity' in metrics:
            print(f"  困惑度: {metrics['perplexity']:.2f}")
        if 'accuracy' in metrics:
            print(f"  准确率: {metrics['accuracy']:.4f}")
        
        print("="*60) 