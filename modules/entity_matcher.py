"""实体匹配模块"""
import pandas as pd
import numpy as np

class EntityMatcher:
    """实体匹配器，用于在两个数据集之间找到相似的记录"""
    
    def match(self, df1, df2, method='exact', columns=None):
        """
        在两个数据集之间进行实体匹配
        
        参数:
            df1 (pd.DataFrame): 第一个数据集
            df2 (pd.DataFrame): 第二个数据集
            method (str): 匹配方法，目前仅支持 'exact'
            columns (list): 用于匹配的列名列表，如果为None则使用所有共同列
            
        返回:
            pd.DataFrame: 匹配结果
        """
        # 如果没有指定列，使用所有共同列
        if not columns:
            columns = list(set(df1.columns) & set(df2.columns))
            
        if not columns:
            raise ValueError("没有找到可以用于匹配的共同列")
            
        # 验证所有指定的列都存在于两个数据集中
        for col in columns:
            if col not in df1.columns or col not in df2.columns:
                raise ValueError(f"列 '{col}' 不存在于两个数据集中")
                
        # 进行精确匹配
        matches = self._exact_match(df1, df2, columns)
        
        return matches
        
    def _exact_match(self, df1, df2, columns):
        """
        在指定列上进行精确匹配
        
        参数:
            df1 (pd.DataFrame): 第一个数据集
            df2 (pd.DataFrame): 第二个数据集
            columns (list): 用于匹配的列名列表
            
        返回:
            pd.DataFrame: 匹配结果
        """
        # 在指定列上进行精确匹配
        # 为非匹配列添加后缀，匹配列保持原名
        non_match_cols = list(set(df1.columns) | set(df2.columns) - set(columns))
        merged = pd.merge(df1, df2, on=columns, how='inner', 
                         suffixes=('_1', '_2'))
        print("entity matching成功")
        print(df1.shape)
        print(df2.shape)
        if len(merged) == 0:
            print("没有找到完全匹配的记录")
            return pd.DataFrame()  # 返回空DataFrame
            
        return merged
