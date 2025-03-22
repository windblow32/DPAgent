import pandas as pd
import numpy as np
import re
from typing import Dict, List, Union, Tuple

class StructureNormalizer:
    def __init__(self):
        """初始化结构规范化器"""
        # 常见的列名映射（可以根据需要扩展）
        self.column_name_mappings = {
            # ID类
            r'(?i)(^id$|^index$|^key$)': 'id',
            
            # 数值类
            r'(?i)(^number$|^num$|^count$)': 'number',
            r'(?i)(price|cost|fee|amount)': 'price',
            r'(?i)(percentage|ratio|rate)': 'ratio',
            
            # 日期时间类
            r'(?i)(date|time|datetime)': 'date',
            r'(?i)(year|yr)': 'year',
            r'(?i)(month|mon)': 'month',
            r'(?i)(day|dy)': 'day',
            
            # 分类类
            r'(?i)(category|type|class|group)': 'category',
            r'(?i)(status|state|condition)': 'status',
            
            # 文本类
            r'(?i)(name|title)': 'name',
            r'(?i)(description|desc|comment)': 'description',
            r'(?i)(address|addr|location)': 'address'
        }
        
        # 数据类型映射
        self.dtype_mappings = {
            'id': 'int64',
            'number': 'float64',
            'price': 'float64',
            'ratio': 'float64',
            'date': 'datetime64[ns]',
            'year': 'int64',
            'month': 'int64',
            'day': 'int64',
            'category': 'category',
            'status': 'category',
            'name': 'string',
            'description': 'string',
            'address': 'string'
        }
    
    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """规范化列名"""
        # 创建一个新的DataFrame以避免修改原始数据
        df = df.copy()
        
        # 规范化列名
        new_columns = []
        for col in df.columns:
            # 转换为小写并替换特殊字符
            normalized = col.lower().strip()
            normalized = re.sub(r'[\s\-]+', '_', normalized)
            normalized = re.sub(r'[^a-z0-9_]', '', normalized)
            
            # 检查是否匹配预定义的模式
            matched = False
            for pattern, replacement in self.column_name_mappings.items():
                if re.search(pattern, col):
                    normalized = replacement
                    matched = True
                    break
            
            # 如果存在重复的列名，添加数字后缀
            base_name = normalized
            counter = 1
            while normalized in new_columns:
                normalized = f"{base_name}_{counter}"
                counter += 1
            
            new_columns.append(normalized)
        
        df.columns = new_columns
        return df
    
    def infer_and_convert_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """推断并转换数据类型"""
        df = df.copy()
        type_conversion_info = {}
        
        for col in df.columns:
            # 检查是否有预定义的数据类型
            dtype = None
            for pattern, type_name in self.dtype_mappings.items():
                if re.search(pattern, col):
                    dtype = type_name
                    break
            
            if dtype is None:
                # 自动推断数据类型
                sample = df[col].dropna().head(100)
                if len(sample) == 0:
                    continue
                
                try:
                    # 尝试转换为数值类型
                    pd.to_numeric(sample)
                    # 检查是否都是整数
                    if all(float(x).is_integer() for x in sample):
                        dtype = 'int64'
                    else:
                        dtype = 'float64'
                except:
                    # 检查是否是日期
                    try:
                        pd.to_datetime(sample)
                        dtype = 'datetime64[ns]'
                    except:
                        # 如果唯一值较少，可能是分类
                        if len(sample.unique()) < len(sample) * 0.5:
                            dtype = 'category'
                        else:
                            dtype = 'string'
            
            # 尝试转换数据类型
            try:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif dtype in ['int64', 'float64']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif dtype == 'category':
                    df[col] = df[col].astype('category')
                else:
                    df[col] = df[col].astype('string')
                
                type_conversion_info[col] = dtype
                
            except Exception as e:
                print(f"警告：列 '{col}' 转换为 {dtype} 类型失败：{str(e)}")
                type_conversion_info[col] = str(df[col].dtype)
        
        return df, type_conversion_info
    
    def normalize_structure(self, df: pd.DataFrame, 
                          normalize_names: bool = True,
                          convert_types: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        规范化表格结构
        
        Args:
            df: 输入的DataFrame
            normalize_names: 是否规范化列名
            convert_types: 是否转换数据类型
            
        Returns:
            规范化后的DataFrame和规范化信息字典
        """
        info = {
            'original_columns': list(df.columns),
            'original_types': df.dtypes.to_dict(),
            'changes': {}
        }
        
        # 规范化列名
        if normalize_names:
            df = self.normalize_column_names(df)
            info['changes']['column_names'] = dict(zip(info['original_columns'], df.columns))
        
        # 转换数据类型
        if convert_types:
            df, type_info = self.infer_and_convert_types(df)
            info['changes']['data_types'] = type_info
        
        return df, info
    
    def generate_report(self, info: Dict) -> str:
        """生成规范化报告"""
        report = []
        report.append("表格结构规范化报告")
        report.append("=" * 30)
        
        # 列名变更
        if 'column_names' in info['changes']:
            report.append("\n列名规范化:")
            for old, new in info['changes']['column_names'].items():
                if old != new:
                    report.append(f"- {old} -> {new}")
        
        # 数据类型转换
        if 'data_types' in info['changes']:
            report.append("\n数据类型转换:")
            for col, dtype in info['changes']['data_types'].items():
                original_type = info['original_types'][info['original_columns'][list(info['changes']['column_names'].values()).index(col)]]
                if str(original_type) != str(dtype):
                    report.append(f"- {col}: {original_type} -> {dtype}")
        
        return "\n".join(report)
