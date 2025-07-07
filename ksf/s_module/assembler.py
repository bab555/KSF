"""
S模块核心实现：提示装配引擎
KSF 4.x: 负责将K模块的ResonancePacket包装成结构化输出
"""
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
import logging

from .analyzer import IntentAnalyzer
from ..k_module.data_structures import RetrievalInstruction, ResonancePacket
from . import processors

logger = logging.getLogger(__name__)

# --- 自定义 Jinja2 过滤器 ---
def _first_line_filter(text: str) -> str:
    """返回文本的第一行。"""
    if not text:
        return ""
    return text.splitlines()[0]

def _other_lines_filter(text: str) -> str:
    """返回除第一行外的所有行。"""
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[1:])

class PromptAssembler:
    """
    提示装配引擎
    KSF 4.x: 将K模块的输出（ResonancePacket）进行结构化呈现。
    不再进行复杂的语义标注，而是专注于展示K模块的发现。
    """
    
    def __init__(self, templates_dir: str, manifest: Dict[str, Any]):
        self.templates_dir = Path(templates_dir) if templates_dir else Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
        # 注册自定义过滤器
        self.env.filters['first_line'] = _first_line_filter
        self.env.filters['other_lines'] = _other_lines_filter
        
        # 注册 processors 模块中的所有函数为全局函数，以便在模板中直接调用
        self.env.globals['processors'] = processors

        self.analyzer = IntentAnalyzer(manifest=manifest)
        self.manifest = manifest
        
        logger.info(f"✓ S模块 (提示装配器) 初始化完成，模板目录: {self.templates_dir}")

    def generate_instruction(self, query_text: str) -> RetrievalInstruction:
        """
        根据用户查询生成K模块的检索指令。
        通过分析用户意图，决定检索模式和参数。
        """
        analysis = self.analyzer.analyze(query_text)
        
        # 默认使用标准的语义检索
        return RetrievalInstruction(
            mode='SEMANTIC',
            query_text=query_text,
            filters={},
            entities=analysis.get('entities', [])
        )

    def assemble_prompt(self, resonance_packet: ResonancePacket, query: str) -> str:
        """
        装配提示词
        主要入口函数，根据分析的意图选择合适的模板来渲染K模块的共振包。
        """
        logger.info("🔧 S模块开始装配最终答案...")
        
        # 预分析，未来可用于模板选择
        analysis = self.analyzer.analyze(query)
        intent = analysis.get("intent", "general")
        logger.info(f"识别到用户意图: '{intent}' (置信度: {analysis.get('intent_score', 0):.2f})")

        prompt_data = {
            "query": query,
            "packet": resonance_packet,
            "analysis": analysis
        }
        
        # 当前固定使用 resonance 模板，未来可以根据 intent 选择不同模板
        template_name = "resonance.jinja2"
        logger.info(f"选择模板: '{template_name}'")

        final_prompt = self._render_template(template_name, prompt_data)
        
        logger.info(f"✓ 最终答案装配完成，长度: {len(final_prompt)} 字符")
        
        return final_prompt

    def _render_template(self, template_name: str, prompt_data: Dict[str, Any]) -> str:
        """
        模板渲染
        使用Jinja2模板引擎渲染最终提示词
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**prompt_data)
            return rendered.strip()
        except Exception as e:
            logger.error(f"✗ 模板 '{template_name}' 渲染失败: {e}")
            return self._fallback_prompt(prompt_data)

    def _fallback_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """一个健壮的备用提示，以防主模板渲染失败。"""
        packet = prompt_data.get('packet')
        if not packet:
            return "发生未知错误，无法生成内容。"
            
        try:
            template_string = """
            **查询:**
            {{ query }}

            **核心知识:**
            {% if packet.primary_atoms %}
                {% for item in packet.primary_atoms %}
                - {{ item.content }} (ID: {{ item.id }}, Score: {{ "%.2f"|format(item.final_score) }})
                {% endfor %}
            {% else %}
                未找到直接相关的信息。
            {% endif %}
            
            **相关上下文:**
            {% if packet.context_atoms %}
                {% for item in packet.context_atoms %}
                - {{ item.content }} (ID: {{ item.id }}, Score: {{ "%.2f"|format(item.final_score) }})
                {% endfor %}
            {% else %}
                未找到相关的上下文信息。
            {% endif %}

            **相关概念:**
            {% if packet.emerged_concepts %}
                {% for concept in packet.emerged_concepts %}
                - {{ concept.concept }} (Score: {{ "%.2f"|format(concept.score) }})
                {% endfor %}
            {% else %}
                未发现相关的扩展概念。
            {% endif %}
            """
            template = self.env.from_string(template_string)
            return template.render(**prompt_data)
        except Exception as e:
            logger.error(f"✗ 备用模板渲染也失败: {e}")
            return f"查询: {prompt_data.get('query')}\n\n抱歉，系统在生成回答时遇到严重错误。"

    def add_custom_template(self, name: str, template_content: str):
        """
        添加自定义S模块模板
        
        Args:
            name: 模板名称
            template_content: 模板内容
        """
        template_path = self.templates_dir / f"{name}.jinja2"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        # 重新加载环境以识别新模板
        self.env.loader = FileSystemLoader(str(self.templates_dir))
        logger.info(f"✅ 已添加并加载自定义模板: {name}")

    def list_templates(self) -> List[str]:
        """列出所有可用的模板"""
        return self.env.list_templates()