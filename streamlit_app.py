import streamlit as st
import pandas as pd
import numpy as np
from modules.normalizer import DataNormalizer
from modules.llm_processor import LLMProcessor
from modules.imputer import DataImputer
from modules.table_normalizer import TableNormalizer
from modules.numeric_identifier import identify_numeric_columns
from modules.smart_imputer import SmartImputer  # 导入SmartImputer

# 设置页面标题和布局
st.set_page_config(
    page_title="智能数据处理助手",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化处理器
normalizer = DataNormalizer()
llm_processor = LLMProcessor()
imputer = DataImputer()

def main():
    st.title("智能数据处理助手")
    
    # 文件上传部分
    with st.container():
        st.subheader("数据上传", divider='rainbow')
        uploaded_file = st.file_uploader("请选择数据文件", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                # 读取数据
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                try:
                    if file_extension == 'csv':
                        df = pd.read_csv(uploaded_file)
                    elif file_extension == 'xlsx':
                        df = pd.read_excel(uploaded_file)
                    else:
                        st.error(f"不支持的文件格式：{file_extension}")
                        return
                except Exception as e:
                    st.error(f"读取文件时出错：{str(e)}")
                    return
                
                st.success(f"文件上传成功！数据大小: {df.shape}")
                
                # 显示原始数据预览
                with st.expander("查看原始数据"):
                    st.dataframe(df.head())
                
                # 用户输入处理需求
                st.subheader("数据处理", divider='rainbow')
                user_input = st.text_area(
                    "请输入您的数据处理需求",
                    placeholder="例如：\n- 标准化处理数据\n- 对数据进行z-score标准化\n- 分析输入数据",
                    height=100
                )
                
                # 处理按钮
                if st.button("开始处理", type="primary"):
                    with st.spinner("正在处理数据..."):
                        try:
                            # 解析用户需求
                            parsed_request = llm_processor.parse_user_request(user_input)
                            
                            if parsed_request is None:
                                st.error("无法理解您的需求，请尝试重新描述")
                                return
                            
                            if parsed_request['operation'] == 'all':
                                
                                # 子集划分
                                # 输出划分结果，划分规则


                                st.subheader("第一步：表格规范化")
                                
                                try:
                                    # 初始化表格规范化器
                                    table_normalizer = TableNormalizer(llm_processor)
                                    
                                    # 执行表格规范化
                                    normalized_df, norm_info = table_normalizer.normalize_table(df)
                                    
                                    # 显示规范化报告
                                    st.text(table_normalizer.generate_report(norm_info))
                                    
                                    # 显示规范化后的数据
                                    st.write("规范化后的数据：")
                                    st.dataframe(normalized_df)
                                    
                                    # 更新数据框，用于后续处理
                                    df = normalized_df
                                    
                                    st.success("表格规范化完成")
                                except Exception as e:
                                    st.error(f"表格规范化失败：{str(e)}")
                                    return
                                
                                # 2. 进行缺失值填充
                                st.subheader("第二步：缺失值填充")
                                
                                # 使用NumericIdentifier识别数值列
                                numeric_cols, report = identify_numeric_columns(
                                    df,
                                    min_numeric_ratio=0.8,  # 允许20%的非数值或空值
                                    ignore_na=True  # 忽略空值
                                )
                                
                                if not numeric_cols:
                                    st.warning("未发现数值类型的列，请检查数据")
                                    with st.expander("查看数值列分析报告"):
                                        st.text(report)
                                    return
                                
                                # 显示数值列分析报告
                                with st.expander("查看数值列分析报告"):
                                    st.text(report)
                                
                                try:
                                    # 初始化SmartImputer
                                    smart_imputer = SmartImputer()
                                    
                                    # 显示智能填充分析
                                    st.subheader("智能填充分析")
                                    # 显示原始数据概览
                                    st.write("**原始数据概览:**")
                                    st.dataframe(df.head())
                                    st.write("**缺失值统计:**")
                                    st.write(df.isnull().sum())
                                    
                                    # 保存原始列顺序
                                    original_columns = df.columns.tolist()
                                    
                                    # 分析所有列
                                    for column in df.columns:
                                        with st.expander(f"列: {column} 的分析"):
                                            method, description = smart_imputer.suggest_method(df, column)
                                            st.write(f"**列类型:** {'数值型' if pd.api.types.is_numeric_dtype(df[column]) else '非数值型'}")
                                            st.write(f"**推荐填充方法:** {method}")
                                            st.text(description)
                                            
                                            # 显示更多列统计信息
                                            if pd.api.types.is_numeric_dtype(df[column]):
                                                clean_data = df[column].dropna()
                                                if len(clean_data) > 0:
                                                    st.write(f"**偏度:** {clean_data.skew():.2f}")
                                                    has_outliers = np.abs(clean_data - clean_data.mean()).max() > 3 * clean_data.std()
                                                    st.write(f"**是否有异常值:** {has_outliers}")
                                            else:
                                                st.write(f"**唯一值数量:** {df[column].nunique()}")
                                            
                                            # 显示缺失值比例
                                            missing_ratio = df[column].isnull().mean() * 100
                                            st.write(f"**缺失值比例:** {missing_ratio:.2f}%")
                                    
                                    # 执行智能填充
                                    st.subheader("执行智能填充")
                                    with st.spinner('正在执行智能填充...'):
                                        # 预处理：将空字符串转换为 NaN
                                        df_processed = df.replace('', np.nan)
                                        imputed_data, methods, _ = smart_imputer.impute_smart(df_processed)
                                    
                                    # 显示填充结果，和original_columns顺序一致
                                    st.write("**填充后的数据概览:**")
                                    imputed_data = imputed_data[original_columns]
                                    st.dataframe(imputed_data.head())
                                    
                                    # 显示填充方法总结
                                    st.write("**各列使用的填充方法:**")
                                    for col, info in methods.items():
                                        st.write(f"- {col}: {info['method']}")
                                    
                                    # 显示缺失值统计对比
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**原始数据缺失值统计:**")
                                        st.write(df_processed.isnull().sum())
                                    with col2:
                                        st.write("**填充后数据缺失值统计:**")
                                        st.write(imputed_data.isnull().sum())
                                    
                                    # 实体识别输入数据，haoxuan 实现
                                    st.subheader("第三步：实体识别")
                                    st.write("构建相似度函数推荐规则，开始分析列统计信息")
                                    st.write("length variance")

                                    st.write("提示：subset2 length variance 过低，need rule-augmented data generation")
                                    # 提供按钮，执行数据生成
                                    st.subheader("数据生成")
                                    # explanation of generator(里面写生成的依据)
                                    # 输出生成结果
                                    st.write("**生成后的数据概览:**")
                                    # 点击show generated data按钮，显示生成结果(只展示生成的数据)
                                

                                    st.subheader("continue with entity matching...")
                                    # apply augmented data（展示合并后的数据）
                                    # 显示：mining similarity recommmendation rules,
                                    # 输出子集划分的规则
                                    # 
                                

                                    

                                    # 弹出窗口
                                    


                                    # # 进行标准化
                                    # st.subheader("第四步：数据标准化")
                                    # processed_data, used_columns, plots = normalizer.normalize(
                                    #     imputed_data,
                                    #     method='zscore',
                                    #     columns=numeric_cols
                                    # )
                                    
                                    # # 显示标准化结果
                                    # col1, col2, col3 = st.columns([1, 1, 1])
                                    
                                    # with col1:
                                    #     st.subheader("处理信息")
                                    #     st.info(f"""
                                    #     缺失值填充:
                                    #     - 使用智能填充
                                    #     - 处理的列: {', '.join(numeric_cols)}
                                        
                                    #     标准化处理:
                                    #     - 使用方法: zscore
                                    #     - 处理的列: {', '.join(used_columns)}
                                    #     """)
                                    
                                    # with col2:
                                    #     st.subheader("原始数据")
                                    #     st.dataframe(df)
                                    
                                    # with col3:
                                    #     st.subheader("处理后的数据")
                                    #     st.dataframe(processed_data)
                                    
                                    # # 提供下载处理后的数据
                                    # st.download_button(
                                    #     "下载处理后的数据",
                                    #     processed_data.to_csv(index=False).encode('utf-8'),
                                    #     "processed_data.csv",
                                    #     "text/csv",
                                    #     key='download-csv'
                                    # )
                                    
                                except Exception as e:
                                    st.error(f"数据处理时出错：{str(e)}")
                                    return
                                    
                            elif parsed_request['operation'] == 'normalizer':
                                try:
                                    # 执行标准化
                                    method = parsed_request.get('method', 'zscore')
                                    columns = parsed_request.get('columns', [])
                                    
                                    processed_data, used_columns, _ = normalizer.normalize(df, method=method, columns=columns)
                                    
                                    # 显示结果
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    
                                    with col1:
                                        st.subheader("处理信息")
                                        st.info(f"""
                                        - 使用的标准化方法: {method}
                                        - 处理的列: {', '.join(used_columns)}
                                        """)
                                    
                                    with col2:
                                        st.subheader("原始数据")
                                        st.dataframe(df)
                                    
                                    with col3:
                                        st.subheader("处理后的数据")
                                        st.dataframe(processed_data)
                                    
                                    # 提供下载处理后的数据
                                    st.download_button(
                                        "下载处理后的数据",
                                        processed_data.to_csv(index=False).encode('utf-8'),
                                        "processed_data.csv",
                                        "text/csv",
                                        key='download-csv'
                                    )
                                    
                                except Exception as e:
                                    st.error(f"数据处理时出错：{str(e)}")
                                    return
                                    
                        except Exception as e:
                            st.error(f"处理请求时出错：{str(e)}")
                            return
                            
            except Exception as e:
                st.error(f"处理文件时出错：{str(e)}")
                return

if __name__ == "__main__":
    main()
