import pandas as pd
import sys

def analyze_csv(file_path):
    """
    分析CSV文件中的数值列
    
    Args:
        file_path: CSV文件路径
    """
    try:
        # 读取CSV文件
        print(f"\n正在读取文件: {file_path}")
        df = pd.read_csv(file_path)
        
        print("\n数据基本信息:")
        print(f"行数: {df.shape[0]}")
        print(f"列数: {df.shape[1]}")
        
        # 获取数值列
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        print("\n发现的数值列:")
        for col in numeric_cols:
            print(f"- {col} (类型: {df[col].dtype})")
            print(f"  样本值: {df[col].head().tolist()}")
        
        # 检查潜在的数值列（字符串格式）
        string_cols = df.select_dtypes(include=['object']).columns
        potential_numeric = []
        
        print("\n潜在的数值列（字符串格式）:")
        for col in string_cols:
            # 获取非空值
            non_null_values = df[col].dropna()
            if len(non_null_values) == 0:
                continue
            
            # 检查前100个非空值是否都可以转换为数值
            sample_values = non_null_values.head(100)
            try:
                pd.to_numeric(sample_values)
                potential_numeric.append(col)
                print(f"- {col}")
                print(f"  样本值: {df[col].head().tolist()}")
            except:
                continue
        
        print("\n建议:")
        if potential_numeric:
            print("以下列可以转换为数值类型:")
            for col in potential_numeric:
                print(f"- {col}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件 {file_path} 是空的")
    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python analyze_csv.py <csv文件路径>")
        sys.exit(1)
    
    file_path = 'data/tableB.csv'
    analyze_csv(file_path)
