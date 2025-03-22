"""数据标准化模块"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Union, List, Optional, Tuple

class DataNormalizer:
    """数据标准化器，支持多种标准化策略
    
    支持的策略:
    - 'zscore': Z-score标准化 ((x - mean) / std)
    - 'minmax': MinMax缩放到[0,1]范围
    - 'decimal': 小数缩放
    - 'log10': Log10变换
    """
    
    VALID_STRATEGIES = ['zscore', 'minmax', 'decimal', 'log10']
    
    def __init__(self):
        self._scaler = None
        self._stats = {}
        self.numeric_columns = None
        
    def normalize(self, df: pd.DataFrame, method: str = 'zscore', 
                 columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        标准化数据
        
        参数:
        -----
        df : pd.DataFrame
            需要标准化的数据
        method : str, default='zscore'
            标准化方法，可选: 'zscore', 'minmax', 'decimal', 'log10'
        columns : List[str], optional
            需要标准化的列。如果为None，则处理所有数值列
            
        返回:
        -----
        Tuple[pd.DataFrame, List[str], List[str]]
            - 标准化后的数据
            - 处理的列名列表
            - 可视化图表路径列表
        """
        if method not in self.VALID_STRATEGIES:
            raise ValueError(f"不支持的标准化方法。请使用以下方法之一: {self.VALID_STRATEGIES}")
            
        # 如果没有指定列，使用所有数值列
        if columns is None:
            self.numeric_columns = []
            print("开始识别数值列...")
            for col in df.columns:
                # 检查列的内容
                sample = df[col].dropna().head(100)  # 取前100个非空值作为样本
                if len(sample) == 0:
                    continue
                
                try:
                    # 尝试将样本数据转换为数值
                    numeric_values = pd.to_numeric(sample, errors='raise')
                    # 如果成功转换，将整列数据转换为数值
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    self.numeric_columns = self.numeric_columns.append(col)
                    print(f"列 '{col}' 被识别为数值列")
                except (ValueError, TypeError):
                    print(f"列 '{col}' 不是数值列")
                    continue
        else:
            # 验证指定的列是否都存在且为数值类型
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"列 '{col}' 不存在")
                try:
                    # 尝试将列转换为数值类型
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    raise ValueError(f"列 '{col}' 不能转换为数值类型")
            self.numeric_columns = columns
            
        if not self.numeric_columns:
            raise ValueError("没有找到可以标准化的数值列")
            
        result = df.copy()
        plots = []  # 用于存储可视化图表的路径
        
        # 确保所有要处理的列都是数值类型
        for col in self.numeric_columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
        
        if method == 'minmax':
            self._scaler = MinMaxScaler()
            result[self.numeric_columns] = self._scaler.fit_transform(result[self.numeric_columns])
            
        elif method == 'zscore':
            # 存储统计信息用于后续可能的逆变换
            self._stats['mean'] = result[self.numeric_columns].mean()
            self._stats['std'] = result[self.numeric_columns].std()
            
            # 使用 z-score 方法标准化
            result[self.numeric_columns] = (result[self.numeric_columns] - self._stats['mean']) / self._stats['std']
            print("normalized successfully")

        elif method == 'decimal':
            for col in self.numeric_columns:
                max_abs = result[col].abs().max()
                max_digits = int(np.log10(max_abs)) + 1 if max_abs > 0 else 1
                result[col] = result[col] / (10 ** max_digits)
                
        elif method == 'log10':
            for col in self.numeric_columns:
                min_val = result[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    result[col] = np.log10(result[col] + shift)
                else:
                    result[col] = np.log10(result[col])
                    
        return result, self.numeric_columns, plots
