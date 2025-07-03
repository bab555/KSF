"""
CoD Metrics - 评估指标计算
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_type: str = 'classification'
) -> Dict[str, float]:
    """
    计算基本评估指标
    
    Args:
        predictions: 模型预测结果
        targets: 真实标签
        task_type: 任务类型 ('classification', 'regression')
    
    Returns:
        指标字典
    """
    metrics = {}
    
    if task_type == 'classification':
        # 转换为numpy数组
        if isinstance(predictions, torch.Tensor):
            pred_numpy = predictions.detach().cpu().numpy()
        else:
            pred_numpy = np.array(predictions)
            
        if isinstance(targets, torch.Tensor):
            target_numpy = targets.detach().cpu().numpy()
        else:
            target_numpy = np.array(targets)
        
        # 如果predictions是概率分布，取argmax
        if len(pred_numpy.shape) > 1 and pred_numpy.shape[-1] > 1:
            pred_numpy = np.argmax(pred_numpy, axis=-1)
        
        # 计算分类指标
        metrics['accuracy'] = accuracy_score(target_numpy, pred_numpy)
        metrics['f1_macro'] = f1_score(target_numpy, pred_numpy, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(target_numpy, pred_numpy, average='micro', zero_division=0)
        metrics['precision'] = precision_score(target_numpy, pred_numpy, average='macro', zero_division=0)
        metrics['recall'] = recall_score(target_numpy, pred_numpy, average='macro', zero_division=0)
        
    elif task_type == 'regression':
        # 回归指标
        if isinstance(predictions, torch.Tensor):
            pred_tensor = predictions.detach().cpu()
        else:
            pred_tensor = torch.tensor(predictions)
            
        if isinstance(targets, torch.Tensor):
            target_tensor = targets.detach().cpu()
        else:
            target_tensor = torch.tensor(targets)
        
        mse = F.mse_loss(pred_tensor, target_tensor)
        mae = F.l1_loss(pred_tensor, target_tensor)
        
        metrics['mse'] = mse.item()
        metrics['mae'] = mae.item()
        metrics['rmse'] = torch.sqrt(mse).item()
    
    return metrics

def compute_cod_specific_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    计算CoD特定的评估指标
    
    Args:
        outputs: 模型输出字典，包含P、C、S的输出
        targets: 真实标签（可选）
    
    Returns:
        CoD特定指标字典
    """
    metrics = {}
    
    # 提取各专家输出
    proposer_output = outputs.get('proposer_output')
    challenger_output = outputs.get('challenger_output')
    synthesizer_output = outputs.get('synthesizer_output')
    final_output = outputs.get('final_output')
    
    # 1. 角色相似度分析
    if proposer_output is not None and challenger_output is not None:
        # 计算P和C的相似度（应该较低，表示角色分化）
        p_flat = proposer_output.view(proposer_output.size(0), -1)
        c_flat = challenger_output.view(challenger_output.size(0), -1)
        
        # 余弦相似度
        pc_similarity = F.cosine_similarity(p_flat, c_flat, dim=1).mean().item()
        metrics['proposer_challenger_similarity'] = pc_similarity
        
        # 角色坍塌检测（相似度过高表示角色坍塌）
        metrics['role_collapse_risk'] = 1.0 if pc_similarity > 0.9 else 0.0
    
    if proposer_output is not None and synthesizer_output is not None:
        p_flat = proposer_output.view(proposer_output.size(0), -1)
        s_flat = synthesizer_output.view(synthesizer_output.size(0), -1)
        ps_similarity = F.cosine_similarity(p_flat, s_flat, dim=1).mean().item()
        metrics['proposer_synthesizer_similarity'] = ps_similarity
    
    if challenger_output is not None and synthesizer_output is not None:
        c_flat = challenger_output.view(challenger_output.size(0), -1)
        s_flat = synthesizer_output.view(synthesizer_output.size(0), -1)
        cs_similarity = F.cosine_similarity(c_flat, s_flat, dim=1).mean().item()
        metrics['challenger_synthesizer_similarity'] = cs_similarity
    
    # 2. 置信度分析
    confidence_scores = outputs.get('confidence_scores', {})
    for role, confidence in confidence_scores.items():
        if isinstance(confidence, torch.Tensor):
            metrics[f'{role}_confidence_mean'] = confidence.mean().item()
            metrics[f'{role}_confidence_std'] = confidence.std().item()
    
    # 3. 辩论轮次分析
    debate_rounds = outputs.get('debate_rounds', 0)
    metrics['debate_rounds'] = debate_rounds
    
    # 4. 挑战者影响分析
    if 'challenger_impact' in outputs:
        metrics['challenger_impact'] = outputs['challenger_impact'].item()
    
    # 5. 多样性指标
    if proposer_output is not None:
        # 提议多样性（批次内方差）
        p_flat = proposer_output.view(proposer_output.size(0), -1)
        if p_flat.size(0) > 1:
            pairwise_distances = torch.cdist(p_flat, p_flat, p=2)
            # 取上三角矩阵的平均值
            mask = torch.triu(torch.ones_like(pairwise_distances, dtype=torch.bool), diagonal=1)
            avg_distance = pairwise_distances[mask].mean().item()
            metrics['proposer_diversity'] = avg_distance
    
    # 6. 任务性能指标
    if targets is not None and final_output is not None:
        task_metrics = compute_metrics(final_output, targets)
        metrics.update({f'task_{k}': v for k, v in task_metrics.items()})
    
    return metrics

def compute_debate_analysis(
    debate_history: List[Dict[str, torch.Tensor]]
) -> Dict[str, float]:
    """
    分析多轮辩论过程
    
    Args:
        debate_history: 辩论历史，每个元素包含一轮的P、C、S输出
    
    Returns:
        辩论分析指标
    """
    if not debate_history:
        return {}
    
    metrics = {}
    num_rounds = len(debate_history)
    metrics['total_rounds'] = num_rounds
    
    # 跟踪意见变化
    synthesizer_changes = []
    
    for i in range(1, num_rounds):
        prev_s = debate_history[i-1].get('synthesizer_output')
        curr_s = debate_history[i].get('synthesizer_output')
        
        if prev_s is not None and curr_s is not None:
            # 计算综合者输出的变化程度
            prev_flat = prev_s.view(prev_s.size(0), -1)
            curr_flat = curr_s.view(curr_s.size(0), -1)
            
            change = F.mse_loss(prev_flat, curr_flat).item()
            synthesizer_changes.append(change)
    
    if synthesizer_changes:
        metrics['synthesizer_change_mean'] = np.mean(synthesizer_changes)
        metrics['synthesizer_change_std'] = np.std(synthesizer_changes)
        
        # 收敛检测（变化趋势）
        if len(synthesizer_changes) >= 3:
            # 检查最后几轮的变化是否在减小
            recent_changes = synthesizer_changes[-3:]
            is_converging = all(recent_changes[i] >= recent_changes[i+1] 
                              for i in range(len(recent_changes)-1))
            metrics['is_converging'] = 1.0 if is_converging else 0.0
    
    return metrics

def compute_efficiency_metrics(
    model_outputs: Dict[str, torch.Tensor],
    computation_time: float,
    memory_usage: Optional[float] = None
) -> Dict[str, float]:
    """
    计算效率相关指标
    
    Args:
        model_outputs: 模型输出
        computation_time: 计算时间（秒）
        memory_usage: 内存使用量（MB，可选）
    
    Returns:
        效率指标字典
    """
    metrics = {}
    
    # 时间效率
    metrics['computation_time'] = computation_time
    
    # 内存效率
    if memory_usage is not None:
        metrics['memory_usage_mb'] = memory_usage
    
    # 参数效率（需要模型信息）
    if 'final_output' in model_outputs:
        batch_size = model_outputs['final_output'].size(0)
        metrics['time_per_sample'] = computation_time / batch_size
    
    return metrics

def compute_robustness_metrics(
    clean_outputs: Dict[str, torch.Tensor],
    noisy_outputs: Dict[str, torch.Tensor],
    noise_level: float
) -> Dict[str, float]:
    """
    计算鲁棒性指标
    
    Args:
        clean_outputs: 清洁输入的输出
        noisy_outputs: 噪声输入的输出
        noise_level: 噪声水平
    
    Returns:
        鲁棒性指标字典
    """
    metrics = {}
    
    # 输出稳定性
    for key in ['proposer_output', 'challenger_output', 'synthesizer_output', 'final_output']:
        if key in clean_outputs and key in noisy_outputs:
            clean = clean_outputs[key].view(clean_outputs[key].size(0), -1)
            noisy = noisy_outputs[key].view(noisy_outputs[key].size(0), -1)
            
            # 计算输出变化
            stability = F.cosine_similarity(clean, noisy, dim=1).mean().item()
            metrics[f'{key}_stability'] = stability
    
    # 预测一致性
    if 'final_output' in clean_outputs and 'final_output' in noisy_outputs:
        clean_pred = torch.argmax(clean_outputs['final_output'], dim=-1)
        noisy_pred = torch.argmax(noisy_outputs['final_output'], dim=-1)
        
        consistency = (clean_pred == noisy_pred).float().mean().item()
        metrics['prediction_consistency'] = consistency
    
    metrics['noise_level'] = noise_level
    
    return metrics

def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    聚合多个指标字典
    
    Args:
        metrics_list: 指标字典列表
    
    Returns:
        聚合后的指标字典
    """
    if not metrics_list:
        return {}
    
    # 收集所有键
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    aggregated = {}
    
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_min'] = np.min(values)
            aggregated[f'{key}_max'] = np.max(values)
    
    return aggregated

def print_metrics_report(metrics: Dict[str, float], title: str = "Metrics Report"):
    """
    打印格式化的指标报告
    
    Args:
        metrics: 指标字典
        title: 报告标题
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # 按类别组织指标
    categories = {
        'Task Performance': ['accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall'],
        'Role Analysis': ['proposer_challenger_similarity', 'role_collapse_risk', 'challenger_impact'],
        'Debate Process': ['debate_rounds', 'is_converging', 'synthesizer_change_mean'],
        'Efficiency': ['computation_time', 'time_per_sample', 'memory_usage_mb'],
        'Robustness': ['prediction_consistency', 'final_output_stability']
    }
    
    for category, metric_keys in categories.items():
        category_metrics = {k: v for k, v in metrics.items() 
                          if any(key in k for key in metric_keys)}
        
        if category_metrics:
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in sorted(category_metrics.items()):
                if isinstance(value, float):
                    print(f"  {key:<25}: {value:.4f}")
                else:
                    print(f"  {key:<25}: {value}")
    
    # 其他指标
    other_metrics = {k: v for k, v in metrics.items() 
                    if not any(any(key in k for key in cat_keys) 
                             for cat_keys in categories.values())}
    
    if other_metrics:
        print(f"\nOther Metrics:")
        print("-" * 30)
        for key, value in sorted(other_metrics.items()):
            if isinstance(value, float):
                print(f"  {key:<25}: {value:.4f}")
            else:
                print(f"  {key:<25}: {value}")
    
    print(f"{'='*60}\n") 