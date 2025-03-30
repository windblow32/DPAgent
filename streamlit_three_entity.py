import streamlit as st
import pandas as pd
import numpy as np
from modules.smart_imputer import SmartImputer
import random
from datetime import datetime, timedelta

def generate_sample_data(n_rows=100):
    """生成示例数据"""
    companies = ['Tech Corp', 'Data Inc', 'AI Solutions', 'Smart Systems', 'Digital Ltd']
    domains = ['tech', 'data', 'ai', 'digital', 'smart']
    
    data = {
        'company_name': [random.choice(companies) + ' ' + str(random.randint(1, 100)) for _ in range(n_rows)],
        'revenue': np.random.uniform(1000000, 10000000, n_rows),
        'employees': np.random.randint(50, 1000, n_rows),
        'founded_date': [(datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d') for _ in range(n_rows)],
        'domain': [random.choice(domains) for _ in range(n_rows)]
    }
    return pd.DataFrame(data)

def main():
    st.set_page_config(layout="wide")
    st.title("数据预处理与实体识别系统")

    # 初始数据处理部分
    st.header("数据预处理")
    uploaded_file = st.file_uploader("上传CSV文件", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("原始数据预览：")
        st.dataframe(df.head())

        # 数据类型识别
        st.subheader("第一步：数据类型识别")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        st.write("**数值型列：**", numeric_cols)
        st.write("**类别型列：**", categorical_cols)

        # 缺失值处理
        st.subheader("第二步：缺失值处理")
        missing_stats = df.isnull().sum()
        st.write("**缺失值统计：**")
        st.write(missing_stats)

        if missing_stats.sum() > 0:
            imputer = SmartImputer()
            imputed_data = imputer.fit_transform(df)
            st.write("**填充后数据缺失值统计：**")
            st.write(imputed_data.isnull().sum())
        else:
            imputed_data = df

        # 实体识别部分 - 三列布局
        st.header("实体识别与匹配")
        
        # 生成三个示例数据集
        df1 = generate_sample_data()
        df2 = generate_sample_data()
        df3 = generate_sample_data()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("数据集 1")
            st.dataframe(df1.head())
            st.write("**数据统计：**")
            st.write(f"记录数: {len(df1)}")
            st.write("列统计信息：")
            st.write(df1.describe())
            
            # 数据处理选项
            st.subheader("处理选项")
            selected_cols1 = st.multiselect(
                "选择要处理的列（数据集1）",
                df1.columns.tolist(),
                key="cols1"
            )

        with col2:
            st.subheader("数据集 2")
            st.dataframe(df2.head())
            st.write("**数据统计：**")
            st.write(f"记录数: {len(df2)}")
            st.write("列统计信息：")
            st.write(df2.describe())
            
            # 数据处理选项
            st.subheader("处理选项")
            selected_cols2 = st.multiselect(
                "选择要处理的列（数据集2）",
                df2.columns.tolist(),
                key="cols2"
            )

        with col3:
            st.subheader("数据集 3")
            st.dataframe(df3.head())
            st.write("**数据统计：**")
            st.write(f"记录数: {len(df3)}")
            st.write("列统计信息：")
            st.write(df3.describe())
            
            # 数据处理选项
            st.subheader("处理选项")
            selected_cols3 = st.multiselect(
                "选择要处理的列（数据集3）",
                df3.columns.tolist(),
                key="cols3"
            )

        # 实体匹配设置
        st.subheader("实体匹配配置")
        col1, col2 = st.columns(2)
        
        with col1:
            similarity_threshold = st.slider(
                "设置相似度阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1
            )

        with col2:
            matching_method = st.selectbox(
                "选择匹配方法",
                ["基于规则的匹配", "模糊匹配", "混合匹配"]
            )

        if st.button("开始实体匹配"):
            st.info("正在进行实体匹配分析...")
            # 这里可以添加实际的实体匹配逻辑
            st.success("实体匹配完成！")
            
            # 显示匹配结果示例
            st.subheader("匹配结果示例")
            match_results = pd.DataFrame({
                "数据集1_实体": df1['company_name'].head(),
                "数据集2_实体": df2['company_name'].head(),
                "数据集3_实体": df3['company_name'].head(),
                "相似度得分": np.random.uniform(0.7, 1.0, 5)
            })
            st.dataframe(match_results)

if __name__ == "__main__":
    main()
