"""
S模块核心实现：提示装配引擎
负责将K模块的知识包装配成结构化提示词
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
import logging

from .processors import (
    tag_pros_cons, extract_strategic_points, tag_comparison_aspects,
    tag_temporal_aspects, extract_action_items, tag_risk_factors
)
from .analyzer import IntentAnalyzer
from ..k_module.data_structures import RetrievalInstruction

logger = logging.getLogger(__name__)


class PromptAssembler:
    """
    提示装配引擎
    非生成式的、基于规则的结构化数据转换引擎
    """
    
    def __init__(self, templates_dir: str, manifest: Dict[str, Any]):
        """
        初始化提示装配引擎
        
        Args:
            templates_dir: 模板目录路径
            manifest: 知识库的元数据
        """
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # 初始化Jinja2环境
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Initialize the intelligent toolkit with the knowledge manifest
        self.analyzer = IntentAnalyzer(manifest=manifest)
        self.manifest = manifest
        
        # 策略映射：根据查询类型选择处理策略
        self.strategy_mapping = {
            'comparison': self._handle_comparison_strategy,
            'analysis': self._handle_analysis_strategy,
            'guide': self._handle_guide_strategy,
            'decision': self._handle_decision_strategy,
            'general': self._handle_general_strategy
        }
        
        # 初始化默认模板
        self._create_default_templates()
        
        logger.info(f"✓ 提示装配引擎初始化完成，模板目录: {self.templates_dir}")
        logger.info("✓ S-Module 'Intelligent Toolkit' initialized with knowledge manifest.")
    
    def _create_default_templates(self):
        """创建默认的模板文件"""
        templates = {
            'comparison.jinja2': '''**Analysis Topic:**
{{ topic }}

**Comparison Framework:**
{% if comparison_aspects %}
{% for aspect, items in comparison_aspects.items() %}
{% if items %}
**{{ aspect.title() }} Dimension:**
{% for item in items %}
• {{ item }}
{% endfor %}
{% endif %}
{% endfor %}
{% endif %}

**Direct Knowledge Points:**
{% for item in direct_knowledge %}
• {{ item.content }} (Score: {{ "%.2f"|format(item.final_score) }})
{% endfor %}

{% if hidden_associations %}
**Related Considerations:**
{% for assoc in hidden_associations %}
• {{ assoc.explanation }} (Strength: {{ "%.2f"|format(assoc.strength) }})
{% endfor %}
{% endif %}

{% if pros_cons.pros or pros_cons.cons %}
**Pros & Cons Analysis:**
{% if pros_cons.pros %}
**Advantages:**
{% for pro in pros_cons.pros %}
• {{ pro }}
{% endfor %}
{% endif %}
{% if pros_cons.cons %}
**Disadvantages:**
{% for con in pros_cons.cons %}
• {{ con }}
{% endfor %}
{% endif %}
{% endif %}''',
            
            'analysis.jinja2': '''**Analysis Subject:**
{{ topic }}

**Strategic Overview:**
{% if strategic_points %}
{% for point in strategic_points %}
• {{ point }}
{% endfor %}
{% endif %}

**Key Information:**
{% for item in direct_knowledge %}
• {{ item.content }}
  - Source: {{ item.id }}
  - Score: {{ "%.2f"|format(item.final_score) }}
  - (Similarity: {{ "%.2f"|format(item.original_similarity) }}, PageRank: {{ "%.2f"|format(item.pagerank_weight) }})
{% endfor %}

{% if hidden_associations %}
**Extended Context:**
{% for assoc in hidden_associations %}
• {{ assoc.related_concept }}: {{ assoc.explanation }}
{% endfor %}
{% endif %}

{% if temporal_aspects %}
**Temporal Considerations:**
{% for period, items in temporal_aspects.items() %}
{% if items %}
**{{ period.replace('_', ' ').title() }}:**
{% for item in items %}
• {{ item }}
{% endfor %}
{% endif %}
{% endfor %}
{% endif %}

{% if risk_factors %}
**Risk Assessment:**
{% for risk in risk_factors %}
• {{ risk }}
{% endfor %}
{% endif %}''',
            
            'guide.jinja2': '''**Guide Topic:**
{{ topic }}

**Step-by-Step Framework:**

{% if action_items %}
**Action Items:**
{% for action in action_items %}
**Step {{ loop.index }}:** {{ action }}
{% endfor %}
{% endif %}

**Supporting Knowledge:**
{% for item in direct_knowledge %}
• {{ item.content }} (Score: {{ "%.2f"|format(item.final_score) }})
{% endfor %}

{% if hidden_associations %}
**Additional Considerations:**
{% for assoc in hidden_associations %}
• {{ assoc.explanation }}
{% endfor %}
{% endif %}

{% if risk_factors %}
**Important Warnings:**
{% for risk in risk_factors %}
• {{ risk }}
{% endfor %}
{% endif %}''',
            
            'general.jinja2': '''**Query:**
{{ query }}

**Knowledge Base Response:**

**Primary Information:**
{% for item in direct_knowledge %}
• {{ item.content }}
  (Score: {{ "%.2f"|format(item.final_score) }}, Similarity: {{ "%.2f"|format(item.original_similarity) }})
{% endfor %}

{% if hidden_associations %}
**Related Insights:**
{% for assoc in hidden_associations %}
• {{ assoc.explanation }}
{% endfor %}
{% endif %}

**Confidence Mapping:**
{% for key, weight in attention_weights.items() %}
• {{ key }}: {{ "%.2f"|format(weight) }}
{% endfor %}'''
        }
        
        # 创建模板文件
        for filename, content in templates.items():
            template_path = self.templates_dir / filename
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"✓ 创建默认模板: {filename}")
    
    def assemble_prompt(self, knowledge_packet: Dict[str, Any], 
                       strategy: Optional[str] = None) -> str:
        """
        装配提示词
        主要入口函数，执行完整的装配工作流
        
        Args:
            knowledge_packet: K模块输出的知识包
            strategy: 装配策略 ('comparison', 'analysis', 'guide', 'decision', 'general')
            
        Returns:
            装配好的结构化提示词
        """
        logger.info(f"🔧 开始装配提示词，策略: {strategy or 'auto'}")
        
        # 1. 策略选择
        if not strategy:
            strategy = self._select_strategy(knowledge_packet)
        
        # 2. 语义标注
        tagged_data = self._semantic_tagging(knowledge_packet)
        
        # 3. 构建提示数据对象
        prompt_data = self._build_prompt_data_object(knowledge_packet, tagged_data)
        
        # 4. 模板渲染
        final_prompt = self._render_template(strategy, prompt_data)
        
        logger.info(f"✓ 提示词装配完成，长度: {len(final_prompt)} 字符")
        
        return final_prompt
    
    def _select_strategy(self, knowledge_packet: Dict[str, Any]) -> str:
        """
        自动选择装配策略
        基于查询内容和知识特征选择最合适的策略
        """
        query = knowledge_packet.get('query', '').lower()
        
        # 关键词匹配策略选择
        if any(keyword in query for keyword in ['对比', '比较', 'vs', '选择', '差异']):
            return 'comparison'
        elif any(keyword in query for keyword in ['分析', '评估', '研究', '解读']):
            return 'analysis'
        elif any(keyword in query for keyword in ['如何', '怎样', '步骤', '指南', '教程']):
            return 'guide'
        elif any(keyword in query for keyword in ['决策', '选择', '建议', '推荐']):
            return 'decision'
        else:
            return 'general'
    
    def _semantic_tagging(self, knowledge_packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        语义标注
        给知识点打标签，这是S模块的核心智能所在
        """
        direct_knowledge = knowledge_packet.get('direct_knowledge', [])
        
        # 应用各种标注器
        tagged_data = {
            'pros_cons': tag_pros_cons(direct_knowledge),
            'strategic_points': extract_strategic_points(direct_knowledge),
            'comparison_aspects': tag_comparison_aspects(direct_knowledge),
            'temporal_aspects': tag_temporal_aspects(direct_knowledge),
            'action_items': extract_action_items(direct_knowledge),
            'risk_factors': tag_risk_factors(direct_knowledge)
        }
        
        return tagged_data
    
    def _build_prompt_data_object(self, knowledge_packet: Dict[str, Any], 
                                 tagged_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建提示数据对象
        将所有信息整合成一个用于模板渲染的数据对象
        """
        prompt_data = {
            'query': knowledge_packet.get('query', ''),
            'topic': self._extract_topic(knowledge_packet.get('query', '')),
            'direct_knowledge': knowledge_packet.get('direct_knowledge', []),
            'hidden_associations': knowledge_packet.get('hidden_associations', []),
            'attention_weights': knowledge_packet.get('attention_weights', {}),
            'processing_metadata': knowledge_packet.get('processing_metadata', {}),
            'tagged_data': tagged_data,
            'knowledge_packet': knowledge_packet
        }
        
        return prompt_data
    
    def _extract_topic(self, query: str) -> str:
        """
        从查询中提取主题
        简化实现：直接使用查询作为主题
        """
        # 简单的主题提取逻辑
        if '如何' in query:
            return query.replace('如何', '').strip()
        elif '什么是' in query:
            return query.replace('什么是', '').strip()
        elif '?' in query:
            return query.replace('?', '').strip()
        else:
            return query.strip()
    
    def _render_template(self, strategy: str, prompt_data: Dict[str, Any]) -> str:
        """
        模板渲染
        使用Jinja2模板引擎渲染最终提示词
        """
        template_name = f"{strategy}.jinja2"
        
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**prompt_data)
            return rendered.strip()
        except Exception as e:
            logger.error(f"✗ 模板渲染失败: {e}")
            # 回退到通用模板
            try:
                template = self.env.get_template("general.jinja2")
                return template.render(**prompt_data).strip()
            except Exception as e2:
                logger.error(f"✗ 通用模板渲染也失败: {e2}")
                return self._fallback_prompt(prompt_data)
    
    def _fallback_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """A simple fallback prompt generator if all else fails."""
        lines = ["--- Fallback Prompt ---"]
        lines.append(f"Query: {prompt_data.get('query', 'N/A')}")
        lines.append("\nRelevant Information:")
        
        knowledge = prompt_data.get('direct_knowledge', [])
        if not knowledge:
            lines.append("  - No relevant information found.")
        else:
            for item in knowledge:
                lines.append(f"  • {item.content}")
                
        return "\\n".join(lines)
    
    def _handle_comparison_strategy(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对比策略的特殊逻辑"""
        # 这里可以添加对比策略的特殊处理逻辑
        return prompt_data
    
    def _handle_analysis_strategy(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析策略的特殊逻辑"""
        # 这里可以添加分析策略的特殊处理逻辑
        return prompt_data
    
    def _handle_guide_strategy(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理指南策略的特殊逻辑"""
        # 这里可以添加指南策略的特殊处理逻辑
        return prompt_data
    
    def _handle_decision_strategy(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理决策策略的特殊逻辑"""
        # 这里可以添加决策策略的特殊处理逻辑
        return prompt_data
    
    def _handle_general_strategy(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理通用策略的特殊逻辑"""
        # 这里可以添加通用策略的特殊处理逻辑
        return prompt_data
    
    def add_custom_template(self, name: str, template_content: str):
        """
        添加自定义模板
        
        Args:
            name: 模板名称（不包含.jinja2后缀）
            template_content: 模板内容
        """
        template_path = self.templates_dir / f"{name}.jinja2"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        logger.info(f"✓ 添加自定义模板: {name}")
    
    def list_available_templates(self) -> List[str]:
        """列出可用的模板"""
        templates = []
        for template_file in self.templates_dir.glob("*.jinja2"):
            templates.append(template_file.stem)
        return templates

    def list_templates(self) -> List[str]:
        """Lists the available template names."""
        return self.env.list_templates()

    def generate_instruction(self, query_text: str) -> RetrievalInstruction:
        """
        Analyzes a query and generates a structured instruction for the K-Module,
        validated against the knowledge manifest.
        """
        analysis_packet = self.analyzer.analyze(query_text)
        logger.info(f"✅ S-Module Analysis: {json.dumps(analysis_packet, ensure_ascii=False, indent=2)}")

        intent = analysis_packet.get("intent")
        
        # New logic: Use manifest to decide if a filter-based search is possible
        # if intent == "规划行程":
        #     # This requires entity extraction for preferences
        #     entities = analysis_packet.get("entities", [])
        #     activity_preferences = [e['text'] for e in entities if e['label'] == 'ACTIVITY']
        #     travel_style = [e['text'] for e in entities if e['label'] == 'STYLE']

        #     # Check if the manifest supports filtering by these
        #     # This is a simplified check; a real implementation would be more robust
        #     if self.manifest.get('supports_activity_filter') and activity_preferences:
        #         return RetrievalInstruction(
        #             mode='FILTER_BASED',
        #             query_text=query_text,
        #             filters={'activity': activity_preferences[0]} # Simplified
        #         )

        # Default to semantic search if no specific strategy applies
        return RetrievalInstruction(query_text=query_text, entities=analysis_packet.get("entities"))

    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Uses the internal toolkit to analyze the user query for intent and entities.
        This is the first step in the "S-Led Query Dispatch" model.
        """
        logger.info(f"🔬 S-Module analyzing query: '{query_text}'")
        return self.analyzer.analyze(query_text)