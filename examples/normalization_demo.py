"""数据标准化模块使用示例"""
import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.normalizer import DataNormalizer

def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    data = {
        'id': range(1, 101),
        'age': np.random.normal(35, 10, 100),  # 正态分布的年龄数据
        'salary': np.random.lognormal(10, 0.5, 100),  # 对数正态分布的薪资数据
        'score': np.random.uniform(0, 100, 100),  # 均匀分布的分数数据
        'name': [f'Person_{i}' for i in range(1, 101)]  # 非数值列
    }
    return pd.DataFrame(data)

def demonstrate_normalization():
    """演示各种标准化方法"""
    # 创建示例数据
    df = create_sample_data()
    print("原始数据示例：")
    print(df.head())
    print("\n数值列的统计信息：")
    print(df.describe())
    
    # 创建标准化器
    normalizer = DataNormalizer()
    
    # 演示不同的标准化方法
    methods = ['zscore', 'minmax', 'decimal', 'log10']
    numeric_columns = ['age', 'salary', 'score']
    
    for method in methods:
        print(f"\n\n使用 {method} 方法进行标准化:")
        normalized_df, processed_cols, _ = normalizer.normalize(
            df, 
            method=method,
            columns=numeric_columns
        )
        
        print(f"\n处理的列: {processed_cols}")
        print("\n标准化后的数据示例：")
        print(normalized_df[processed_cols].head())
        print("\n标准化后的统计信息：")
        print(normalized_df[processed_cols].describe())

if __name__ == "__main__":
    demonstrate_normalization()
