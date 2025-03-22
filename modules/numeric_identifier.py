import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import re

class NumericIdentifier:
    def __init__(self):
        """初始化数值识别器"""
        # 常见的数值模式
        self.numeric_patterns = [
            # 整数
            r'^\s*[-+]?\d+\s*$',
            # 小数
            r'^\s*[-+]?\d*\.\d+\s*$',
            # 科学计数法
            r'^\s*[-+]?\d*\.?\d+[eE][-+]?\d+\s*$',
            # 带千位分隔符的数字
            r'^\s*[-+]?\d{1,3}(,\d{3})*(\.\d+)?\s*$',
            # 带货币符号的数字
            r'^\s*[$€¥£]\s*[-+]?\d+(\.\d+)?\s*$',
            r'^\s*[-+]?\d+(\.\d+)?\s*[$€¥£]\s*$',
            # 百分比
            r'^\s*[-+]?\d+(\.\d+)?\s*%\s*$'
        ]
        
        # 编译正则表达式以提高性能
        self.numeric_regex = [re.compile(pattern) for pattern in self.numeric_patterns]
    
    def is_numeric_string(self, value: str) -> bool:
        """
        检查字符串是否可以转换为数值
        
        Args:
            value: 要检查的字符串
            
        Returns:
            bool: 是否为数值字符串
        """
        if not isinstance(value, str):
            return False
            
        # 检查是否匹配任何数值模式
        if any(regex.match(value) for regex in self.numeric_regex):
            return True
            
        # 尝试转换为数值
        try:
            float(value.replace(',', '').strip('$€¥£%'))
            return True
        except (ValueError, TypeError):
            return False
            
        return False
    
    def identify_numeric_columns(self, df: pd.DataFrame, 
                               min_numeric_ratio: float = 0.8,
                               ignore_na: bool = True) -> Dict[str, dict]:
        """
        识别DataFrame中的数值列
        
        Args:
            df: 输入的DataFrame
            min_numeric_ratio: 最小数值比例，超过这个比例的非空值是数值时，认为是数值列
            ignore_na: 是否忽略空值，如果为True，则在计算数值比例时不考虑空值
            
        Returns:
            Dict: 包含每列数值信息的字典，格式如下：
                {
                    'column_name': {
                        'is_numeric': bool,  # 是否是数值列
                        'numeric_ratio': float,  # 数值比例
                        'na_ratio': float,  # 空值比例
                        'total_rows': int,  # 总行数
                        'numeric_rows': int,  # 数值行数
                        'na_rows': int,  # 空值行数
                        'sample_values': List[str],  # 样本值
                        'detected_format': str  # 检测到的格式（如'integer', 'decimal', 'percentage'等）
                    }
                }
        """
        results = {}
        
        for column in df.columns:
            column_info = {
                'is_numeric': False,
                'numeric_ratio': 0.0,
                'na_ratio': 0.0,
                'total_rows': len(df),
                'numeric_rows': 0,
                'na_rows': 0,
                'sample_values': [],
                'detected_format': 'unknown'
            }
            
            # 获取非空值
            non_na_values = df[column].dropna()
            na_count = df[column].isna().sum()
            column_info['na_rows'] = na_count
            column_info['na_ratio'] = na_count / len(df)
            
            # 如果列是数值类型
            if pd.api.types.is_numeric_dtype(df[column]):
                column_info['is_numeric'] = True
                column_info['numeric_ratio'] = 1.0
                column_info['numeric_rows'] = len(non_na_values)
                column_info['detected_format'] = 'native_numeric'
                column_info['sample_values'] = df[column].head().tolist()
                
            # 如果列是对象类型，检查是否可以转换为数值
            elif pd.api.types.is_object_dtype(df[column]):
                numeric_count = 0
                sample_values = []
                format_counts = {
                    'integer': 0,
                    'decimal': 0,
                    'percentage': 0,
                    'currency': 0,
                    'scientific': 0
                }
                
                # 检查每个非空值
                for value in non_na_values:
                    if not isinstance(value, str):
                        value = str(value)
                    
                    sample_values.append(value)
                    
                    # 检查数值格式
                    if re.match(r'^\s*[-+]?\d+\s*$', value):
                        numeric_count += 1
                        format_counts['integer'] += 1
                    elif re.match(r'^\s*[-+]?\d*\.\d+\s*$', value):
                        numeric_count += 1
                        format_counts['decimal'] += 1
                    elif re.match(r'^\s*[-+]?\d+(\.\d+)?\s*%\s*$', value):
                        numeric_count += 1
                        format_counts['percentage'] += 1
                    elif re.match(r'^\s*[$€¥£]\s*[-+]?\d+(\.\d+)?\s*$', value) or \
                         re.match(r'^\s*[-+]?\d+(\.\d+)?\s*[$€¥£]\s*$', value):
                        numeric_count += 1
                        format_counts['currency'] += 1
                    elif re.match(r'^\s*[-+]?\d*\.?\d+[eE][-+]?\d+\s*$', value):
                        numeric_count += 1
                        format_counts['scientific'] += 1
                    elif self.is_numeric_string(value):
                        numeric_count += 1
                        format_counts['decimal'] += 1
                
                # 计算数值比例
                total_values = len(non_na_values) if ignore_na else len(df)
                numeric_ratio = numeric_count / total_values if total_values > 0 else 0
                
                # 确定主要格式
                if numeric_count > 0:
                    main_format = max(format_counts.items(), key=lambda x: x[1])[0]
                else:
                    main_format = 'non_numeric'
                
                column_info['is_numeric'] = numeric_ratio >= min_numeric_ratio
                column_info['numeric_ratio'] = numeric_ratio
                column_info['numeric_rows'] = numeric_count
                column_info['sample_values'] = sample_values[:5]
                column_info['detected_format'] = main_format
            
            results[column] = column_info
        
        return results
    
    def generate_report(self, numeric_info: Dict[str, dict]) -> str:
        """
        生成数值列分析报告
        
        Args:
            numeric_info: identify_numeric_columns返回的结果
            
        Returns:
            str: 格式化的报告文本
        """
        report = []
        report.append("数值列分析报告")
        report.append("=" * 50)
        
        # 数值列
        numeric_columns = [col for col, info in numeric_info.items() if info['is_numeric']]
        report.append(f"\n发现的数值列 ({len(numeric_columns)}):")
        
        for col in numeric_columns:
            info = numeric_info[col]
            report.append(f"\n列名: {col}")
            report.append(f"  格式: {info['detected_format']}")
            report.append(f"  数值比例: {info['numeric_ratio']:.2%}")
            report.append(f"  空值比例: {info['na_ratio']:.2%}")
            report.append(f"  样本值: {info['sample_values'][:5]}")
        
        # 非数值列
        non_numeric_columns = [col for col, info in numeric_info.items() if not info['is_numeric']]
        if non_numeric_columns:
            report.append(f"\n非数值列 ({len(non_numeric_columns)}):")
            for col in non_numeric_columns:
                info = numeric_info[col]
                if info['numeric_ratio'] > 0:
                    report.append(f"\n列名: {col}")
                    report.append(f"  部分数值比例: {info['numeric_ratio']:.2%}")
                    report.append(f"  样本值: {info['sample_values'][:5]}")
        
        return "\n".join(report)

def identify_numeric_columns(df: pd.DataFrame, 
                           min_numeric_ratio: float = 0.8,
                           ignore_na: bool = True) -> Tuple[List[str], str]:
    """
    便捷函数：识别DataFrame中的数值列并生成报告
    
    Args:
        df: 输入的DataFrame
        min_numeric_ratio: 最小数值比例
        ignore_na: 是否忽略空值
        
    Returns:
        Tuple[List[str], str]: (数值列列表, 分析报告)
    """
    identifier = NumericIdentifier()
    numeric_info = identifier.identify_numeric_columns(df, min_numeric_ratio, ignore_na)
    report = identifier.generate_report(numeric_info)
    numeric_columns = [col for col, info in numeric_info.items() if info['is_numeric']]
    return numeric_columns, report
