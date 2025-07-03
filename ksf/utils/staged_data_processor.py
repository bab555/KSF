"""
四阶段训练数据处理器
将统一的CoD数据分解为P-C-S三个专家的专门训练数据
"""

import json
import re
import random
from typing import Dict, List, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StagedDataProcessor:
    """四阶段训练数据处理器"""
    
    def __init__(self):
        # CoD格式模式匹配
        self.cod_patterns = {
            'proposer': [
                r'(?:提议者|初步观点|初始分析)[:：]\s*(.*?)(?=(?:挑战者|质疑|批评)|$)',
                r'(?:首先|第一|初步).*?[:：]\s*(.*?)(?=(?:然而|但是|不过)|$)',
                r'(?:我认为|可以说|初步判断).*?(.*?)(?=(?:但|然而|不过)|$)'
            ],
            'challenger': [
                r'(?:挑战者|质疑|批评|反对)[:：]\s*(.*?)(?=(?:综合者|总结|最终)|$)',
                r'(?:然而|但是|不过|存在问题).*?[:：]\s*(.*?)(?=(?:综合|总结|最终)|$)',
                r'(?:问题在于|缺陷是|不足之处).*?(.*?)(?=(?:综合|因此|所以)|$)'
            ],
            'synthesizer': [
                r'(?:综合者|总结|最终|结论)[:：]\s*(.*?)$',
                r'(?:综合|考虑|权衡).*?[:：]\s*(.*?)$',
                r'(?:因此|所以|综上).*?(.*?)$'
            ]
        }
        
        # 角色特定的指令模板
        self.instruction_templates = {
            'proposer': [
                "请对以下问题给出直接、完整的回答：",
                "基于你的知识，回答以下问题：",
                "请提供对以下问题的详细解答："
            ],
            'challenger': [
                "请分析以下回答中可能存在的问题和不足：",
                "对以下答案进行批判性分析，指出其局限性：",
                "请质疑以下回答，找出其中的漏洞和问题："
            ],
            'synthesizer': [
                "请综合以下不同观点，给出平衡的结论：",
                "考虑以下多个角度，形成综合性判断：",
                "权衡以下观点，提供最终的平衡分析："
            ]
        }
    
    def extract_cod_components(self, answer: str) -> Dict[str, str]:
        """从CoD格式的答案中提取P-C-S三个组件"""
        components = {'proposer': '', 'challenger': '', 'synthesizer': ''}
        
        # 改进的模式匹配
        enhanced_patterns = {
            'proposer': [
                r'(?:提议者|初步观点|初始分析|首先|第一步?)[:：]\s*(.*?)(?=(?:挑战者|质疑|批评|然而|但是|不过)|$)',
                r'(?:我认为|可以说|初步判断|基本上|总的来说)[:：]?\s*(.*?)(?=(?:但|然而|不过|质疑|挑战|问题)|$)',
                r'^(.*?)(?=(?:然而|但是|不过|问题在于|挑战者|质疑))',
                r'(?:^|\n)(.*?)(?=(?:\n.*?(?:然而|但是|问题|质疑))|\n.*?(?:挑战者|综合))'
            ],
            'challenger': [
                r'(?:挑战者|质疑|批评|反对|问题)[:：]\s*(.*?)(?=(?:综合者|总结|最终|因此|所以)|$)',
                r'(?:然而|但是|不过|存在问题|局限性|缺陷|不足)[:：]?\s*(.*?)(?=(?:综合|总结|最终|因此|所以)|$)',
                r'(?:问题在于|缺陷是|不足之处|需要注意|值得商榷)[:：]?\s*(.*?)(?=(?:综合|因此|所以|最终)|$)',
                r'(?:然而|但是|不过).*?(.*?)(?=(?:综合|总结|因此|所以|最终)|$)'
            ],
            'synthesizer': [
                r'(?:综合者|总结|最终|结论|综上所述)[:：]\s*(.*?)$',
                r'(?:综合|考虑|权衡|平衡).*?[:：]\s*(.*?)$',
                r'(?:因此|所以|综上|总之|最终|总的来说)\s*[:：]?\s*(.*?)$',
                r'(?:最终|总结|结论).*?(.*?)$'
            ]
        }
        
        # 尝试匹配每个角色的内容
        for role, patterns in enhanced_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, answer, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                if matches:
                    # 取最长的匹配作为该角色的内容，并清理
                    content = max(matches, key=len).strip()
                    # 清理多余的空白和换行
                    content = re.sub(r'\s+', ' ', content).strip()
                    if len(content) > 10:  # 确保内容足够长
                        components[role] = content
                        break
        
        # 改进的启发式方法
        if not components['proposer']:
            # 提取前半部分作为提议者内容
            sentences = [s.strip() for s in answer.split('。') if s.strip()]
            if sentences:
                half_point = max(1, len(sentences) // 2)
                components['proposer'] = '。'.join(sentences[:half_point]).strip()
        
        if not components['challenger']:
            # 查找包含质疑词汇的部分，改进匹配
            sentences = [s.strip() for s in answer.split('。') if s.strip()]
            challenger_indicators = ['然而', '但是', '不过', '问题', '局限', '不足', '缺陷', '质疑', '挑战', '存在', '需要注意']
            best_challenger = ""
            for sent in sentences:
                if any(indicator in sent for indicator in challenger_indicators):
                    if len(sent) > len(best_challenger):
                        best_challenger = sent
            components['challenger'] = best_challenger.strip()
        
        if not components['synthesizer']:
            # 提取最后部分作为综合者内容，改进匹配
            sentences = [s.strip() for s in answer.split('。') if s.strip()]
            synthesis_indicators = ['因此', '所以', '综上', '总之', '最终', '总结', '综合']
            
            # 先尝试找包含综合词汇的句子
            for sent in reversed(sentences):
                if any(indicator in sent for indicator in synthesis_indicators):
                    components['synthesizer'] = sent.strip()
                    break
            
            # 如果没找到，使用最后几句
            if not components['synthesizer'] and sentences:
                last_count = min(2, len(sentences))
                components['synthesizer'] = '。'.join(sentences[-last_count:]).strip()
        
        # 质量检查：确保每个组件都有合理的长度
        for role, content in components.items():
            if content and len(content.strip()) < 5:
                components[role] = f"{content.strip()}。" if content.strip() else ""
        
        return components
    
    def create_proposer_data(self, records: List[Dict]) -> List[Dict]:
        """创建提议者专门训练数据"""
        proposer_data = []
        
        for record in records:
            question = record['question']
            full_answer = record['answer']
            components = self.extract_cod_components(full_answer)
            
            # 使用原始问题和提议者部分创建训练数据
            proposer_answer = components['proposer'] or full_answer.split('。')[0]
            
            instruction = random.choice(self.instruction_templates['proposer'])
            
            proposer_record = {
                'id': f"{record['id']}_proposer",
                'question': f"{instruction}\n{question}",
                'answer': proposer_answer,
                'task_type': record['task_type'],
                'stage': 'proposer',
                'original_id': record['id']
            }
            
            proposer_data.append(proposer_record)
        
        logger.info(f"创建提议者训练数据: {len(proposer_data)} 条")
        return proposer_data
    
    def create_challenger_data(self, records: List[Dict]) -> List[Dict]:
        """创建挑战者专门训练数据"""
        challenger_data = []
        
        for record in records:
            question = record['question']
            full_answer = record['answer']
            components = self.extract_cod_components(full_answer)
            
            # 使用提议者答案作为输入，挑战者分析作为目标
            proposer_answer = components['proposer'] or full_answer.split('。')[0]
            challenger_analysis = components['challenger']
            
            if not challenger_analysis:
                # 如果没有挑战者内容，生成一个基本的分析模板
                challenger_analysis = f"需要进一步分析该回答的完整性和准确性，考虑是否有遗漏的重要信息。"
            
            instruction = random.choice(self.instruction_templates['challenger'])
            
            # 构建挑战者输入：原问题 + 提议者答案
            challenger_input = f"原问题：{question}\n提议者回答：{proposer_answer}"
            
            challenger_record = {
                'id': f"{record['id']}_challenger",
                'question': f"{instruction}\n{challenger_input}",
                'answer': challenger_analysis,
                'task_type': record['task_type'],
                'stage': 'challenger',
                'original_id': record['id']
            }
            
            challenger_data.append(challenger_record)
        
        logger.info(f"创建挑战者训练数据: {len(challenger_data)} 条")
        return challenger_data
    
    def create_synthesizer_data(self, records: List[Dict]) -> List[Dict]:
        """创建综合者专门训练数据"""
        synthesizer_data = []
        
        for record in records:
            question = record['question']
            full_answer = record['answer']
            components = self.extract_cod_components(full_answer)
            
            proposer_answer = components['proposer'] or full_answer.split('。')[0]
            challenger_analysis = components['challenger'] or "需要进一步考虑该回答的完整性。"
            synthesizer_conclusion = components['synthesizer'] or full_answer.split('。')[-1]
            
            instruction = random.choice(self.instruction_templates['synthesizer'])
            
            # 构建综合者输入：原问题 + 提议者观点 + 挑战者分析
            synthesizer_input = f"原问题：{question}\n提议者观点：{proposer_answer}\n挑战者分析：{challenger_analysis}"
            
            synthesizer_record = {
                'id': f"{record['id']}_synthesizer",
                'question': f"{instruction}\n{synthesizer_input}",
                'answer': synthesizer_conclusion,
                'task_type': record['task_type'],
                'stage': 'synthesizer',
                'original_id': record['id']
            }
            
            synthesizer_data.append(synthesizer_record)
        
        logger.info(f"创建综合者训练数据: {len(synthesizer_data)} 条")
        return synthesizer_data
    
    def create_joint_training_data(self, records: List[Dict]) -> List[Dict]:
        """创建联合训练数据（阶段4）"""
        # 阶段4直接使用原始的完整CoD数据
        joint_data = []
        
        for record in records:
            joint_record = {
                'id': f"{record['id']}_joint",
                'question': record['question'],
                'answer': record['answer'],
                'task_type': record['task_type'],
                'stage': 'joint',
                'original_id': record['id']
            }
            joint_data.append(joint_record)
        
        logger.info(f"创建联合训练数据: {len(joint_data)} 条")
        return joint_data
    
    def process_unified_data(self, 
                           input_path: str, 
                           output_dir: str = "data/staged_training") -> Dict[str, str]:
        """处理统一数据，生成四阶段训练数据"""
        logger.info(f"开始处理统一数据: {input_path}")
        
        # 读取统一数据
        with open(input_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        logger.info(f"读取到 {len(records)} 条记录")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 生成四个阶段的数据
        stage_data = {
            'proposer': self.create_proposer_data(records),
            'challenger': self.create_challenger_data(records),
            'synthesizer': self.create_synthesizer_data(records),
            'joint': self.create_joint_training_data(records)
        }
        
        # 保存各阶段数据
        output_files = {}
        for stage, data in stage_data.items():
            output_file = output_path / f"stage_{stage}_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            output_files[stage] = str(output_file)
            logger.info(f"阶段 {stage} 数据已保存到: {output_file}")
        
        # 生成处理报告
        report = {
            'input_file': input_path,
            'total_records': len(records),
            'stages': {
                stage: {
                    'output_file': output_files[stage],
                    'record_count': len(data),
                    'sample': data[0] if data else None
                }
                for stage, data in stage_data.items()
            }
        }
        
        report_file = output_path / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"处理报告已保存到: {report_file}")
        return output_files

def main():
    """主函数 - 演示数据处理"""
    processor = StagedDataProcessor()
    
    # 处理训练数据
    train_files = processor.process_unified_data(
        'data/processed_unified/unified_train_dataset.json',
        'data/staged_training/train'
    )
    
    # 处理验证数据
    eval_files = processor.process_unified_data(
        'data/processed_unified/unified_eval_dataset.json',
        'data/staged_training/eval'
    )
    
    print("四阶段训练数据生成完成！")
    print("训练数据文件:", train_files)
    print("验证数据文件:", eval_files)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 