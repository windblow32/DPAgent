import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from modules.imputer import DataImputer

class SmartImputer:
    def __init__(self):
        """初始化智能填充选择器"""
        self.imputer = DataImputer()
        self.method_descriptions = {
            'mean': '均值填充 - 适用于数值型且分布对称的数据',
            'median': '中位数填充 - 适用于有异常值或分布偏斜的数据',
            'most_frequent': '众数填充 - 适用于分类数据或离散型数据',
            'constant': '常数填充 - 适用于需要特定值填充的场景',
            'knn': 'KNN填充 - 适用于数据相似性强的场景',
            'iterative': '迭代填充 - 适用于特征间存在复杂关系的数据'
        }
        
    def _analyze_distribution(self, series):
        """深入分析数据分布特征"""
        clean_data = series.dropna()
        if len(clean_data) < 2:
            return None
            
        # 基础统计特征
        stats_info = {
            'mean': np.mean(clean_data),
            'median': np.median(clean_data),
            'std': np.std(clean_data),
            'skewness': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data),
            'unique_ratio': len(clean_data.unique()) / len(clean_data),
            'missing_ratio': series.isna().mean(),
            'value_range': clean_data.max() - clean_data.min() if pd.api.types.is_numeric_dtype(series) else None
        }
        
        # 分布特征分析
        stats_info.update({
            'is_normal': self._test_normality(clean_data),
            'has_outliers': self._detect_outliers(clean_data),
            'distribution_type': self._identify_distribution(clean_data),
            'spatial_correlation': self._analyze_spatial_correlation(clean_data),
            'pattern_strength': self._analyze_pattern_strength(clean_data),
            'cluster_tendency': self._analyze_cluster_tendency(clean_data) if pd.api.types.is_numeric_dtype(series) else None
        })
        
        return stats_info
        
    def _test_normality(self, data):
        """检验数据是否符合正态分布"""
        if not pd.api.types.is_numeric_dtype(data):
            return False
            
        try:
            _, p_value = stats.normaltest(data)
            return p_value > 0.05
        except:
            return False
            
    def _detect_outliers(self, data):
        """使用多种方法检测异常值"""
        if not pd.api.types.is_numeric_dtype(data):
            return False
            
        try:
            # Z-score方法
            z_scores = np.abs(stats.zscore(data))
            z_score_outliers = np.any(z_scores > 3)
            
            # IQR方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            iqr_outliers = np.any((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
            
            # Isolation Forest方法
            if len(data) >= 10:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                yhat = iso_forest.fit_predict(data.values.reshape(-1, 1))
                iso_outliers = np.any(yhat == -1)
            else:
                iso_outliers = False
                
            # 综合判断
            return z_score_outliers or iqr_outliers or iso_outliers
        except:
            return False
            
    def _identify_distribution(self, data):
        """识别数据可能的分布类型"""
        if not pd.api.types.is_numeric_dtype(data):
            return 'categorical'
            
        try:
            # 标准化数据
            data_std = (data - np.mean(data)) / np.std(data)
            
            # 计算各种分布的拟合优度
            distributions = {
                'normal': stats.norm,
                'uniform': stats.uniform,
                'exponential': stats.expon,
                'lognormal': stats.lognorm,
                'gamma': stats.gamma
            }
            
            best_dist = None
            best_ks_stat = float('inf')
            
            for dist_name, dist in distributions.items():
                try:
                    # 拟合分布
                    params = dist.fit(data_std)
                    # 进行KS检验
                    ks_stat = stats.kstest(data_std, dist_name, params)[0]
                    if ks_stat < best_ks_stat:
                        best_ks_stat = ks_stat
                        best_dist = dist_name
                except:
                    continue
            
            return best_dist if best_dist else 'unknown'
        except:
            return 'unknown'
            
    def _analyze_spatial_correlation(self, data):
        """分析数据的空间相关性"""
        if not pd.api.types.is_numeric_dtype(data) or len(data) < 3:
            return 0
            
        try:
            # 使用最近邻分析空间相关性
            nbrs = NearestNeighbors(n_neighbors=min(3, len(data)-1))
            nbrs.fit(data.values.reshape(-1, 1))
            distances, _ = nbrs.kneighbors()
            
            # 计算平均最近邻距离
            mean_dist = np.mean(distances)
            # 计算总体标准差
            total_std = np.std(data)
            
            # 归一化空间相关性指标
            if total_std == 0:
                return 0
            return 1 - (mean_dist / total_std)
        except:
            return 0
            
    def _analyze_pattern_strength(self, data):
        """分析数据的模式强度"""
        if not pd.api.types.is_numeric_dtype(data):
            # 对于分类数据，使用类别分布的熵
            try:
                value_counts = data.value_counts(normalize=True)
                entropy = stats.entropy(value_counts)
                max_entropy = np.log(len(value_counts))
                return 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            except:
                return 0
        else:
            try:
                # 对于数值数据，使用自相关性
                autocorr = pd.Series(data).autocorr()
                return abs(autocorr) if not np.isnan(autocorr) else 0
            except:
                return 0
                
    def _analyze_cluster_tendency(self, data):
        """分析数据的聚类倾向"""
        if len(data) < 10:
            return 0
            
        try:
            # 使用K-means评估聚类趋势
            X = data.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=min(3, len(data)), random_state=42)
            kmeans.fit(X)
            
            # 计算轮廓系数
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette = np.mean([
                    stats.silhouette_score(X, kmeans.labels_)
                    if len(np.unique(kmeans.labels_)) > 1 else 0
                ])
                return max(0, silhouette)  # 确保返回非负值
            return 0
        except:
            return 0
            
    def _analyze_column(self, series):
        """分析列数据特征"""
        info = {
            'dtype': str(series.dtype),
            'missing_ratio': series.isna().mean(),
            'is_numeric': pd.api.types.is_numeric_dtype(series),
            'unique_count': series.nunique(),
            'distribution': None
        }
        
        if info['is_numeric']:
            info['distribution'] = self._analyze_distribution(series)
            
        return info
    
    def _get_smart_suggestion(self, column_info):
        """基于深度数据分析智能推荐填充方法"""
        if not column_info['is_numeric']:
            if column_info['unique_count'] / (1 - column_info['missing_ratio']) < 10:
                return 'most_frequent', '非数值型数据且类别较少，使用众数填充最为合适'
            return 'knn', '非数值型数据但类别较多，使用KNN填充以保持数据分布特征'
        
        dist_info = column_info['distribution']
        if not dist_info:
            return 'most_frequent', '数据分析不充分，保守使用众数填充'
        
        # 基于分布特征的决策树
        if dist_info['missing_ratio'] > 0.5:
            if dist_info['spatial_correlation'] > 0.7:
                return 'knn', '虽然缺失值较多，但数据空间相关性强，使用KNN填充'
            if dist_info['pattern_strength'] > 0.6:
                return 'iterative', '数据存在明显模式，使用迭代填充以保持特征关系'
            return 'median', '缺失值比例高且无明显模式，使用中位数填充更稳健'
        
        if dist_info['is_normal'] and not dist_info['has_outliers']:
            if dist_info['cluster_tendency'] > 0.5:
                return 'knn', '数据呈正态分布且有明显的聚类趋势，使用KNN填充'
            return 'mean', '数据呈正态分布且无异常值，使用均值填充最优'
        
        if dist_info['has_outliers'] or abs(dist_info['skewness']) > 1:
            if dist_info['pattern_strength'] > 0.7:
                return 'iterative', '数据虽有异常值但存在强模式，使用迭代填充'
            return 'median', '数据存在异常值或严重偏斜，使用中位数填充更稳健'
        
        if dist_info['distribution_type'] in ['exponential', 'lognormal', 'gamma']:
            return 'median', '数据呈偏态分布，使用中位数填充'
        
        if dist_info['spatial_correlation'] > 0.6:
            return 'knn', '数据空间相关性强，使用KNN填充可保持局部特征'
        
        if dist_info['cluster_tendency'] > 0.4:
            return 'knn', '数据存在聚类趋势，使用KNN填充以保持群体特征'
        
        if dist_info['pattern_strength'] > 0.5:
            return 'iterative', '数据存在中等强度的模式，使用迭代填充'
        
        # 默认策略
        return 'iterative', '无明显特征，使用迭代填充以平衡各种情况'
    
    def suggest_method(self, data, column):
        """为指定列推荐填充方法"""
        column_info = self._analyze_column(data[column])
        method, reason = self._get_smart_suggestion(column_info)
        
        # 生成详细的分析报告
        if column_info['is_numeric'] and column_info['distribution']:
            dist_info = column_info['distribution']
            analysis = f"\n数据分析报告：\n" + \
                      f"- 分布类型: {dist_info['distribution_type']}\n" + \
                      f"- 是否正态: {'是' if dist_info['is_normal'] else '否'}\n" + \
                      f"- 偏度: {dist_info['skewness']:.2f}\n" + \
                      f"- 峰度: {dist_info['kurtosis']:.2f}\n" + \
                      f"- 存在异常值: {'是' if dist_info['has_outliers'] else '否'}\n" + \
                      f"- 空间相关性: {dist_info['spatial_correlation']:.2f}\n" + \
                      f"- 模式强度: {dist_info['pattern_strength']:.2f}\n" + \
                      f"- 聚类趋势: {dist_info['cluster_tendency']:.2f}\n" + \
                      f"- 缺失比例: {dist_info['missing_ratio']:.2%}"
        else:
            analysis = f"\n数据分析报告：\n" + \
                      f"- 数据类型: {column_info['dtype']}\n" + \
                      f"- 唯一值数量: {column_info['unique_count']}\n" + \
                      f"- 缺失比例: {column_info['missing_ratio']:.2%}"
        
        return method, f"{self.method_descriptions[method]}\n原因：{reason}{analysis}"
    
    def impute_smart(self, data, columns=None):
        """智能选择填充方法并执行填充"""
        if columns is None:
            columns = data.columns[data.isnull().any()].tolist()
            
        if not columns:
            return data, {}, None
            
        methods = {}
        processed_data = data.copy()
        
        for col in columns:
            # 获取推荐的填充方法
            method, description = self.suggest_method(data, col)
            methods[col] = {'method': method, 'description': description}
            
            # 对该列执行填充
            imputed, _, _ = self.imputer.impute(
                processed_data,
                method=method,
                columns=[col],
                visualize=False
            )
            processed_data = imputed
                
        return processed_data, methods, None
