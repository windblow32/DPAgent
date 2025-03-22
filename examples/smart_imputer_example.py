import pandas as pd
import numpy as np
from modules.smart_imputer import SmartImputer
import argparse

def create_sample_data():
    """创建包含各种特征的示例数据"""
    np.random.seed(42)
    n_samples = 100
    
    # 正态分布的数值数据
    normal_data = np.random.normal(loc=50, scale=10, size=n_samples)
    
    # 偏斜的数值数据
    skewed_data = np.random.exponential(scale=10, size=n_samples)
    
    # 带异常值的数值数据
    outlier_data = np.random.normal(loc=100, scale=15, size=n_samples)
    outlier_data[np.random.choice(n_samples, 5)] *= 5  # 添加异常值
    
    # 分类数据
    categories = ['A', 'B', 'C', 'D']
    categorical_data = np.random.choice(categories, size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    
    # 创建DataFrame
    df = pd.DataFrame({
        'normal_column': normal_data,
        'skewed_column': skewed_data,
        'outlier_column': outlier_data,
        'category_column': categorical_data
    })
    
    # 随机添加缺失值
    for col in df.columns:
        mask = np.random.choice([True, False], size=n_samples, p=[0.2, 0.8])
        df.loc[mask, col] = np.nan
        
    return df

def process_data(df):
    """处理数据并执行智能填充"""
    # 保存原始列顺序
    original_columns = df.columns.tolist()
    
    print("\n原始数据概览:")
    print(df.head())
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    # 初始化SmartImputer
    imputer = SmartImputer()
    
    # 对每列进行分析并选择合适的填充方法
    print("\n开始智能填充分析...")
    for column in df.columns:
        print(f"\n分析列: {column}")
        method, description = imputer.suggest_method(df, column)
        print(f"推荐方法: {method}")
        print(f"分析结果: {description}")
        if pd.api.types.is_numeric_dtype(df[column]):
            print(f"数据类型: 数值型")
            clean_data = df[column].dropna()
            if len(clean_data) > 0:
                print(f"偏度: {clean_data.skew():.2f}")
                print(f"是否有异常值: {np.abs(clean_data - clean_data.mean()).max() > 3 * clean_data.std()}")
        else:
            print(f"数据类型: 非数值型")
            print(f"唯一值数量: {df[column].nunique()}")
    
    # 执行智能填充
    print("\n执行智能填充...")
    imputed_df, methods, _ = imputer.impute_smart(df)
    
    # 恢复原始列顺序
    imputed_df = imputed_df[original_columns]
    
    print("\n填充后的数据概览:")
    print(imputed_df.head())
    print("\n填充后的缺失值统计:")
    print(imputed_df.isnull().sum())
    
    # 输出每列使用的填充方法
    print("\n各列使用的填充方法:")
    for col, info in methods.items():
        print(f"{col}: {info['method']} - {info['description']}")
        
    return imputed_df

def main():
    parser = argparse.ArgumentParser(description='智能数据填充工具')
    parser.add_argument('--input', type=str, help='输入数据文件路径（支持csv、excel）')
    parser.add_argument('--output', type=str, help='输出文件路径（可选）')
    args = parser.parse_args()
    
    if args.input:
        # 读取用户提供的数据文件
        file_ext = args.input.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(args.input)
        elif file_ext in ['xls', 'xlsx']:
            df = pd.read_excel(args.input)
        else:
            raise ValueError("不支持的文件格式。请使用csv或excel文件。")
    else:
        # 使用示例数据
        print("未提供输入文件，使用示例数据...")
        df = create_sample_data()
    
    # 处理数据
    imputed_df = process_data(df)
    
    # 保存结果
    if args.output:
        file_ext = args.output.split('.')[-1].lower()
        if file_ext == 'csv':
            imputed_df.to_csv(args.output, index=False)
        elif file_ext in ['xls', 'xlsx']:
            imputed_df.to_excel(args.output, index=False)
        print(f"\n结果已保存到: {args.output}")
    else:
        print("\n\n填充效果验证：")
        print("=" * 50)
        print("缺失值检查：")
        print(imputed_df.isnull().sum())

if __name__ == "__main__":
    main()
