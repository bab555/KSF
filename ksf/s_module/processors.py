"""
S模块的语义标注处理器
包含各种语义标注函数，用于给知识点打标签
"""

from typing import Dict, List, Any, Tuple
import re


def tag_pros_cons(knowledge_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    标注优缺点
    将知识项中的优势和劣势提取并标注
    """
    pros = []
    cons = []
    
    for item in knowledge_items:
        content = item.content.lower()
        
        # 简单的关键词匹配来识别优缺点
        if any(keyword in content for keyword in ['优势', '优点', '好处', '便利', '高效', '简单']):
            pros.append(f"[Pro: {item.content}]")
        elif any(keyword in content for keyword in ['缺点', '劣势', '问题', '困难', '复杂', '缺陷']):
            cons.append(f"[Con: {item.content}]")
    
    return {'pros': pros, 'cons': cons}


def extract_strategic_points(knowledge_items: List[Dict[str, Any]]) -> List[str]:
    """
    提取战略要点
    识别并标注具有战略意义的知识点
    """
    strategic_points = []
    
    strategic_keywords = ['战略', '策略', '规划', '方向', '目标', '核心', '关键', '重要']
    
    for item in knowledge_items:
        content = item.content
        if any(keyword in content for keyword in strategic_keywords):
            strategic_points.append(f"[Strategic_Consideration: {content}]")
    
    return strategic_points


def tag_comparison_aspects(knowledge_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    标注对比维度
    识别可用于对比的不同维度
    """
    aspects = {
        'performance': [],
        'usability': [],
        'cost': [],
        'scalability': [],
        'maintenance': []
    }
    
    aspect_keywords = {
        'performance': ['性能', '速度', '效率', '响应时间', '吞吐量'],
        'usability': ['易用性', '用户体验', '界面', '操作', '学习曲线'],
        'cost': ['成本', '价格', '费用', '投入', '预算'],
        'scalability': ['扩展性', '可扩展', '伸缩性', '规模'],
        'maintenance': ['维护', '运维', '管理', '更新', '升级']
    }
    
    for item in knowledge_items:
        content = item.content.lower()
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in content for keyword in keywords):
                aspects[aspect].append(f"[{aspect.title()}: {item.content}]")
    
    return aspects


def tag_temporal_aspects(knowledge_items: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    标注时间维度
    识别短期、中期、长期的考虑因素
    """
    temporal = {
        'short_term': [],
        'medium_term': [],
        'long_term': []
    }
    
    temporal_keywords = {
        'short_term': ['立即', '马上', '当前', '短期', '临时'],
        'medium_term': ['中期', '阶段性', '逐步', '过渡'],
        'long_term': ['长期', '未来', '持续', '战略', '发展']
    }
    
    for item in knowledge_items:
        content = item.content.lower()
        for period, keywords in temporal_keywords.items():
            if any(keyword in content for keyword in keywords):
                temporal[period].append(f"[{period.replace('_', ' ').title()}: {item.content}]")
    
    return temporal


def extract_action_items(knowledge_items: List[Dict[str, Any]]) -> List[str]:
    """
    提取行动项
    识别需要采取的具体行动
    """
    action_items = []
    
    action_keywords = ['需要', '应该', '必须', '建议', '推荐', '执行', '实施', '采取']
    
    for item in knowledge_items:
        content = item.content
        if any(keyword in content for keyword in action_keywords):
            action_items.append(f"[Action_Item: {content}]")
    
    return action_items


def tag_risk_factors(knowledge_items: List[Dict[str, Any]]) -> List[str]:
    """
    标注风险因素
    识别潜在的风险和注意事项
    """
    risk_factors = []
    
    risk_keywords = ['风险', '注意', '警告', '问题', '隐患', '挑战', '困难', '限制']
    
    for item in knowledge_items:
        content = item.content
        if any(keyword in content for keyword in risk_keywords):
            risk_factors.append(f"[Risk_Factor: {content}]")
    
    return risk_factors 