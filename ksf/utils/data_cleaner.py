"""
CoD 数据清理和格式统一工具
处理编码、unicode等问题，统一数据格式用于生成式训练
"""

import json
import re
import os
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class CoDDataCleaner:
    """CoD数据清理器 - 处理各种数据格式问题并统一格式"""
    
    def __init__(self):
        # 文本清理的正则表达式模式
        self.cleaning_patterns = [
            # 修复unicode转义
            (r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16))),
            # 修复换行符
            (r'\\n', '\n'),
            (r'\\r', '\r'),
            (r'\\t', '\t'),
            # 清理多余空格
            (r'\s+', ' '),
            # 清理行首行尾空格
            (r'^\s+|\s+$', ''),
            # 修复引号
            (r'\\"', '"'),
            (r"\\'", "'"),
            # 清理控制字符（保留基本的换行符和制表符）
            (r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ''),
        ]
        
        # 统一的字段映射
        self.field_mappings = {
            # ultra_clean_dataset格式映射
            'input': 'question',
            'output': 'answer',
            'task_type': 'task_type',
            
            # distilled_1500格式映射  
            'prompt': 'question',
            'cod_prompt': 'question',  # 如果有cod_prompt，优先使用
            'response': 'answer',
            'id': 'id',
            'model_used': 'model_used',
            'timestamp': 'timestamp'
        }
        
    def clean_text(self, text: str) -> str:
        """清理文本中的编码和格式问题"""
        if not isinstance(text, str):
            return str(text)
        
        cleaned = text
        
        # 应用所有清理模式
        for pattern, replacement in self.cleaning_patterns:
            if callable(replacement):
                cleaned = re.sub(pattern, replacement, cleaned)
            else:
                cleaned = re.sub(pattern, replacement, cleaned)
        
        # 清理连续的换行符（保留段落结构）
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # 最终去除首尾空白
        cleaned = cleaned.strip()
        
        return cleaned
    
    def detect_data_format(self, data: List[Dict]) -> str:
        """检测数据格式类型"""
        if not data:
            return 'unknown'
        
        first_record = data[0]
        keys = set(first_record.keys())
        
        if 'input' in keys and 'output' in keys:
            return 'ultra_clean'
        elif 'prompt' in keys and 'response' in keys:
            return 'distilled_1500'
        elif 'cod_prompt' in keys and 'response' in keys:
            return 'distilled_1500_enhanced'
        else:
            return 'unknown'
    
    def standardize_record(self, record: Dict, data_format: str) -> Dict:
        """将记录转换为标准格式"""
        standard_record = {}
        
        # 生成唯一ID（如果没有）
        if 'id' in record:
            standard_record['id'] = record['id']
        else:
            import hashlib
            # 基于内容生成ID
            content = str(record.get('input', record.get('prompt', record.get('cod_prompt', ''))))[:100]
            standard_record['id'] = f"auto_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        # 映射问题字段
        if data_format == 'ultra_clean':
            question = record.get('input', '')
        elif data_format in ['distilled_1500', 'distilled_1500_enhanced']:
            # 优先使用cod_prompt，因为它包含了CoD格式的完整提示
            question = record.get('cod_prompt', record.get('prompt', ''))
        else:
            question = ''
        
        # 映射答案字段
        if data_format == 'ultra_clean':
            answer = record.get('output', '')
        elif data_format in ['distilled_1500', 'distilled_1500_enhanced']:
            answer = record.get('response', '')
        else:
            answer = ''
        
        # 清理文本
        standard_record['question'] = self.clean_text(question)
        standard_record['answer'] = self.clean_text(answer)
        
        # 保留任务类型
        standard_record['task_type'] = record.get('task_type', 'unknown')
        
        # 移除评分字段，改为生成式任务
        # 不再保存quality_score，因为我们用输出方式训练
        
        # 保留元数据（可选）
        metadata = {}
        if 'model_used' in record:
            metadata['model_used'] = record['model_used']
        if 'timestamp' in record:
            metadata['timestamp'] = record['timestamp']
        
        if metadata:
            standard_record['metadata'] = metadata
        
        return standard_record
    
    def validate_record(self, record: Dict) -> Tuple[bool, List[str]]:
        """验证记录是否符合要求"""
        errors = []
        
        # 检查必需字段
        required_fields = ['id', 'question', 'answer', 'task_type']
        for field in required_fields:
            if field not in record:
                errors.append(f"Missing required field: {field}")
            elif not record[field] or (isinstance(record[field], str) and not record[field].strip()):
                errors.append(f"Empty required field: {field}")
        
        # 检查文本长度
        if 'question' in record and len(record['question']) < 10:
            errors.append("Question too short")
        
        if 'answer' in record and len(record['answer']) < 20:
            errors.append("Answer too short")
        
        # 检查是否包含CoD格式
        if 'answer' in record:
            answer = record['answer']
            cod_indicators = ['提议者', '挑战者', '综合者', 'Chain of Debate']
            if not any(indicator in answer for indicator in cod_indicators):
                errors.append("Answer doesn't contain CoD format indicators")
        
        return len(errors) == 0, errors
    
    def process_dataset(self, input_path: str, output_path: str = None, validate: bool = True) -> Dict:
        """处理整个数据集"""
        logger.info(f"开始处理数据集: {input_path}")
        
        # 读取原始数据
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        logger.info(f"原始数据记录数: {len(raw_data)}")
        
        # 检测数据格式
        data_format = self.detect_data_format(raw_data)
        logger.info(f"检测到数据格式: {data_format}")
        
        # 处理记录
        processed_data = []
        validation_errors = []
        
        for i, record in enumerate(raw_data):
            try:
                # 标准化记录
                standard_record = self.standardize_record(record, data_format)
                
                # 验证记录（如果启用）
                if validate:
                    is_valid, errors = self.validate_record(standard_record)
                    if not is_valid:
                        validation_errors.append({
                            'record_index': i,
                            'errors': errors,
                            'record_id': standard_record.get('id', 'unknown')
                        })
                        logger.warning(f"记录 {i} 验证失败: {errors}")
                        continue  # 跳过无效记录
                
                processed_data.append(standard_record)
                
            except Exception as e:
                logger.error(f"处理记录 {i} 时出错: {e}")
                validation_errors.append({
                    'record_index': i,
                    'errors': [f"Processing error: {str(e)}"],
                    'record_id': record.get('id', 'unknown')
                })
        
        logger.info(f"处理完成，有效记录数: {len(processed_data)}")
        if validation_errors:
            logger.warning(f"验证错误数: {len(validation_errors)}")
        
        # 保存处理后的数据
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"处理后的数据已保存到: {output_path}")
        
        # 返回处理统计
        return {
            'original_count': len(raw_data),
            'processed_count': len(processed_data),
            'validation_errors': validation_errors,
            'data_format': data_format,
            'success_rate': len(processed_data) / len(raw_data) if raw_data else 0
        }
    
    def merge_datasets(self, dataset_paths: List[str], output_path: str) -> Dict:
        """合并多个数据集"""
        logger.info(f"开始合并 {len(dataset_paths)} 个数据集")
        
        all_data = []
        merge_stats = {}
        
        for dataset_path in dataset_paths:
            logger.info(f"处理数据集: {dataset_path}")
            
            # 临时处理单个数据集
            temp_output = dataset_path.replace('.json', '_temp_processed.json')
            stats = self.process_dataset(dataset_path, temp_output)
            
            # 读取处理后的数据
            with open(temp_output, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            all_data.extend(processed_data)
            merge_stats[dataset_path] = stats
            
            # 清理临时文件
            os.remove(temp_output)
        
        # 去重（基于ID）
        seen_ids = set()
        unique_data = []
        for record in all_data:
            if record['id'] not in seen_ids:
                unique_data.append(record)
                seen_ids.add(record['id'])
        
        # 保存合并后的数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合并完成，总记录数: {len(unique_data)}")
        logger.info(f"合并后数据已保存到: {output_path}")
        
        return {
            'total_records': len(unique_data),
            'original_total': len(all_data),
            'duplicates_removed': len(all_data) - len(unique_data),
            'individual_stats': merge_stats
        }

def main():
    """主函数 - 演示数据清理功能"""
    cleaner = CoDDataCleaner()
    
    # 设置路径
    datasets = [
        'data/ultra_clean_dataset/train_dataset.json',
        'data/distilled_1500/train_dataset.json'
    ]
    
    output_dir = Path('data/processed_unified')
    output_dir.mkdir(exist_ok=True)
    
    # 合并和清理所有数据集
    merged_output = output_dir / 'unified_train_dataset.json'
    stats = cleaner.merge_datasets(datasets, str(merged_output))
    
    print("数据清理和合并完成！")
    print(f"统计信息: {stats}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 