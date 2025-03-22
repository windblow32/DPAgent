import pandas as pd
from modules.table_normalizer import TableNormalizer
from modules.llm_processor import LLMProcessor

def main():
    # 创建一个示例DataFrame
    data = {
        'Player Name': ['John Smith (USA)', 'Maria Garcia (ESP)', 'Liu Wei (CHN)'],
        'Score/Points': ['1,234.5 pts', '987.0pts', '2,345.8 pts'],
        'Tournament Date': ['2023/01/15', '2023-02-20', '23/03/2023'],
        'Ranking Range': ['2020-2023', '2019/20', '2021-22'],
        'Prize Money': ['$50,000', '$75,000.00', '$100,000'],
        'Performance': ['98.5%', '87.3%', '92.8%']
    }
    df = pd.DataFrame(data)
    
    # 初始化规范化器
    llm_processor = LLMProcessor()
    normalizer = TableNormalizer(llm_processor)
    
    print("原始表格：")
    print("=" * 50)
    print(df)
    print("\n")
    
    try:
        # 规范化表格
        normalized_df, info = normalizer.normalize_table(df, title="Tournament Results")
        
        print("规范化后的表格：")
        print("=" * 50)
        print(normalized_df)
        print("\n")
        
        # 生成报告
        report = normalizer.generate_report(info)
        print(report)
        
    except Exception as e:
        print(f"规范化过程中出现错误：{str(e)}")

if __name__ == "__main__":
    main()
