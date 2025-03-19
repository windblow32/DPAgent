import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from matplotlib.figure import Figure
import networkx as nx
from fuzzywuzzy import fuzz

class DataVisualizer:
    def __init__(self):
        self.style = 'seaborn-v0_8'
        plt.style.use(self.style)
        
    def plot_to_base64(self, fig):
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    def compare_distributions(self, original_data, processed_data, columns=None, title_suffix=""):
        """
        Create distribution plots comparing original and processed data
        
        Args:
            original_data: Original DataFrame or array
            processed_data: Processed DataFrame or array
            columns: List of column names (optional)
            title_suffix: Additional text to add to plot title
            
        Returns:
            Base64 encoded PNG image
        """
        # Convert to DataFrames if necessary
        if not isinstance(original_data, pd.DataFrame):
            original_data = pd.DataFrame(original_data, columns=columns)
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = pd.DataFrame(processed_data, columns=columns)
            
        n_features = original_data.shape[1]
        fig, axes = plt.subplots(2, n_features, figsize=(5*n_features, 8))
        
        if n_features == 1:
            axes = axes.reshape(2, 1)
            
        # Plot original data distributions
        for i, col in enumerate(original_data.columns):
            sns.histplot(data=original_data, x=col, kde=True, ax=axes[0, i])
            axes[0, i].set_title(f'Original {col} Distribution')
            
        # Plot processed data distributions
        for i, col in enumerate(processed_data.columns):
            sns.histplot(data=processed_data, x=col, kde=True, ax=axes[1, i])
            axes[1, i].set_title(f'Processed {col} Distribution {title_suffix}')
            
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')
        
    def create_boxplots(self, original_data, processed_data, columns=None, title_suffix=""):
        """
        Create box plots comparing original and processed data
        
        Args:
            original_data: Original DataFrame or array
            processed_data: Processed DataFrame or array
            columns: List of column names (optional)
            title_suffix: Additional text to add to plot title
            
        Returns:
            Base64 encoded PNG image
        """
        # Convert to DataFrames if necessary
        if not isinstance(original_data, pd.DataFrame):
            original_data = pd.DataFrame(original_data, columns=columns)
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = pd.DataFrame(processed_data, columns=columns)
            
        n_features = original_data.shape[1]
        fig, axes = plt.subplots(2, 1, figsize=(5*n_features, 8))
        
        # Prepare data for plotting
        original_melted = original_data.melt()
        processed_melted = processed_data.melt()
        
        # Plot original data boxplots
        sns.boxplot(data=original_melted, x='variable', y='value', ax=axes[0])
        axes[0].set_title('Original Data Distribution')
        
        # Plot processed data boxplots
        sns.boxplot(data=processed_melted, x='variable', y='value', ax=axes[1])
        axes[1].set_title(f'Processed Data Distribution {title_suffix}')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        graphic = base64.b64encode(image_png)
        return graphic.decode('utf-8')

    def visualize_matches(self, matches_df, columns=None):
        """
        Visualize entity matching results
        
        Args:
            matches_df: DataFrame containing match results
            columns: List of columns used for matching
            
        Returns:
            List of base64 encoded plot images
        """
        plots = []
        
        # 检查是否有匹配结果
        if len(matches_df) == 0:
            # 创建一个显示"没有找到匹配"的图
            fig = plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, '没有找到匹配的记录', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    fontsize=14)
            plt.axis('off')
            plots.append(self.plot_to_base64(fig))
            plt.close(fig)
            return plots
            
        # 1. 相似度分布图
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data=matches_df, x='similarity', bins=20)
        plt.title('Distribution of Match Similarities')
        plt.xlabel('Similarity Score')
        plt.ylabel('Count')
        plots.append(self.plot_to_base64(fig))
        plt.close(fig)
        
        # 2. 如果有指定列，创建热力图
        if columns and len(columns) > 1:
            try:
                # 计算每列的相似度
                similarity_matrix = np.zeros((len(columns), len(columns)))
                for i, col1 in enumerate(columns):
                    for j, col2 in enumerate(columns):
                        # 计算两列之间的平均相似度
                        col1_values = matches_df[f"{col1}_1"].astype(str)
                        col2_values = matches_df[f"{col2}_2"].astype(str)
                        similarities = [fuzz.ratio(str(a), str(b))/100 
                                     for a, b in zip(col1_values, col2_values)]
                        similarity_matrix[i, j] = np.mean(similarities)
                
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(similarity_matrix, annot=True, cmap='YlOrRd',
                           xticklabels=columns, yticklabels=columns)
                plt.title('Column-wise Similarity Heatmap')
                plots.append(self.plot_to_base64(fig))
                plt.close(fig)
            except Exception as e:
                print(f"Error generating heatmap: {str(e)}")
        
        # 3. 匹配网络图（只展示前10个最相似的匹配）
        try:
            G = nx.Graph()
            top_matches = matches_df.nlargest(min(10, len(matches_df)), 'similarity')
            
            # 添加节点和边
            for _, row in top_matches.iterrows():
                # 使用第一列作为节点标签
                first_cols = [col for col in row.index if col.endswith('_1')]
                if not first_cols:  # 如果找不到带_1后缀的列
                    continue
                    
                first_col = first_cols[0]
                second_col = first_col.replace('_1', '_2')
                source = str(row[first_col])
                target = str(row[second_col])
                similarity = row['similarity']
                
                G.add_edge(source, target, weight=similarity)
            
            if len(G.edges()) > 0:  # 只在有边时绘制图
                fig = plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(G)
                
                # 绘制节点
                nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                     node_size=500)
                
                # 绘制边，边的粗细表示相似度
                edge_weights = [G[u][v]['weight']*2 for u, v in G.edges()]
                nx.draw_networkx_edges(G, pos, width=edge_weights)
                
                # 添加节点标签
                nx.draw_networkx_labels(G, pos)
                
                plt.title('Entity Matching Network (Top 10 Matches)')
                plt.axis('off')
                plots.append(self.plot_to_base64(fig))
                plt.close(fig)
        except Exception as e:
            print(f"Error generating network graph: {str(e)}")
        
        return plots

    def visualize_data(self, data):
        """生成单个数据表的可视化图表"""
        plots = []
        
        try:
            # 选择数值型列
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # 生成数值分布图
                plt.figure(figsize=(12, 6))
                for i, col in enumerate(numeric_cols[:5]):  # 最多显示前5个数值列
                    plt.subplot(1, len(numeric_cols[:5]), i+1)
                    sns.histplot(data=data, x=col, bins=20)
                    plt.title(f'{col}分布')
                    plt.xticks(rotation=45)
                
                # 保存图表
                img_path = 'static/numeric_dist.png'
                plt.savefig(img_path, bbox_inches='tight')
                plt.close()
                plots.append(img_path)
                
                # 生成相关性热图
                if len(numeric_cols) > 1:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
                    plt.title('数值特征相关性')
                    
                    # 保存图表
                    img_path = 'static/correlation.png'
                    plt.savefig(img_path, bbox_inches='tight')
                    plt.close()
                    plots.append(img_path)
            
            # 处理类别型列
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) > 0:
                # 生成类别分布图
                plt.figure(figsize=(12, 6))
                for i, col in enumerate(categorical_cols[:5]):  # 最多显示前5个类别列
                    plt.subplot(1, len(categorical_cols[:5]), i+1)
                    value_counts = data[col].value_counts()
                    if len(value_counts) > 10:  # 如果类别太多，只显示前10个
                        value_counts = value_counts.head(10)
                    sns.barplot(x=value_counts.values, y=value_counts.index)
                    plt.title(f'{col}分布')
                    plt.xlabel('频次')
                
                # 保存图表
                img_path = 'static/categorical_dist.png'
                plt.savefig(img_path, bbox_inches='tight')
                plt.close()
                plots.append(img_path)
                
        except Exception as e:
            print(f"生成可视化时出错: {str(e)}")
            
        return plots
