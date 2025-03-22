import pandas as pd
import numpy as np
from modules.numeric_identifier import identify_numeric_columns

def main():
    # 创建一个示例DataFrame，包含各种类型的数据
    data = {
        'ID': [1, 2, 3, 4, 5],
        'Name': ['John', 'Alice', 'Bob', 'Carol', 'David'],
        'Age': ['25', '30', '35', None, '40'],
        'Score': ['98.5%', '87.3%', '92.8%', '76.5%', None],
        'Income': ['$50,000', '$75,000.00', '$100,000', None, '$80,000'],
        'Temperature': ['37.2', '36.8', None, '37.5', '36.9'],
        'Scientific': ['1.23e4', '4.56e3', '7.89e2', None, '1.01e3'],
        'Mixed': ['abc', '123', 'def', '456', 'ghi'],
        'Empty': [None, None, None, None, None],
        'Constant': ['100', '100', '100', '100', '100']
    }
    
    df = pd.DataFrame(data)
    
    print("原始数据：")
    print("=" * 50)
    print(df)
    print("\n")
    
    # 识别数值列
    numeric_columns, report = identify_numeric_columns(
        df,
        min_numeric_ratio=0.6,  # 允许40%的非数值或空值
        ignore_na=True  # 忽略空值
    )
    
    # 打印报告
    print(report)
    
    # 提取数值列数据
    if numeric_columns:
        print("\n提取的数值列数据：")
        print("=" * 50)
        print(df[numeric_columns])

if __name__ == "__main__":
    main()
