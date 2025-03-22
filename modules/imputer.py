import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from modules.visualizer import DataVisualizer

class DataImputer:
    def __init__(self):
        """
        Initialize DataImputer with various imputation strategies
        """
        self.imputers = {
            'mean': SimpleImputer(strategy='mean'),
            'median': SimpleImputer(strategy='median'),
            'most_frequent': SimpleImputer(strategy='most_frequent'),
            'constant': SimpleImputer(strategy='constant', fill_value=0),
            'knn': KNNImputer(n_neighbors=5),
            'iterative': IterativeImputer(max_iter=10, random_state=42)
        }
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.fitted_imputer = None
        self.fitted_scaler = None
        self.visualizer = DataVisualizer()
        
    def _preprocess_data(self, data, columns=None):
        """
        预处理数据，将空字符串转换为 NaN
        
        Args:
            data: DataFrame 或类数组数据
            columns: 要处理的列名列表
            
        Returns:
            处理后的数据
        """
        if isinstance(data, pd.DataFrame):
            processed_data = data.copy()
            if columns is None:
                columns = data.columns
                
            for col in columns:
                # 将空字符串转换为 NaN
                mask = processed_data[col].astype(str).str.strip() == ''
                processed_data.loc[mask, col] = np.nan
                
            return processed_data
        else:
            # 对于numpy数组，将空字符串转换为NaN
            processed_data = np.array(data, dtype=object)
            mask = (processed_data.astype(str) == '') | (processed_data.astype(str).str.strip() == '')
            processed_data[mask] = np.nan
            return processed_data
            
    def impute(self, data, method='mean', columns=None, visualize=False, scale_method=None):
        """
        使用指定方法填补缺失值。对于非数值型数据，自动使用众数填充。
        
        Args:
            data: DataFrame 或类数组数据
            method: 'mean', 'median', 'most_frequent', 'constant', 'knn', 或 'iterative'
            columns: 如果data是DataFrame，则为列名列表
            visualize: 是否生成缺失值可视化（默认关闭）
            scale_method: 可选的缩放方法（'standard' 或 'minmax'）
            
        Returns:
            填补后的数据、使用的列和可视化（如果请求）
        """
        if method not in self.imputers:
            raise ValueError(f"Method {method} not supported. Use one of {list(self.imputers.keys())}")
            
        if isinstance(data, list):
            data = np.array(data)
            
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be DataFrame, numpy array, or list")
            
        # 预处理数据，处理空字符串
        data = self._preprocess_data(data, columns)
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                # 选择包含缺失值的列
                columns = data.columns[data.isnull().any()].tolist()
                if not columns:
                    return data, [], None
                    
            # 存储原始数据用于可视化
            original_data = data[columns].copy()
            
            # 存储未选择的列
            other_columns = data.drop(columns=columns).copy()
            
            # 分离数值列和非数值列
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
            non_numeric_cols = [col for col in columns if col not in numeric_cols]
            
            imputed_data = data.copy()
            
            # 处理数值列
            if numeric_cols:
                # 如果请求，则缩放数据
                if scale_method:
                    if scale_method not in self.scalers:
                        raise ValueError(f"Scale method {scale_method} not supported. Use one of {list(self.scalers.keys())}")
                    scaler = self.scalers[scale_method]
                    scaled_data = scaler.fit_transform(data[numeric_cols])
                    self.fitted_scaler = scaler
                    imputed_values = self.imputers[method].fit_transform(scaled_data)
                    # 逆变换
                    imputed_values = scaler.inverse_transform(imputed_values)
                else:
                    # 不缩放直接填补
                    imputed_values = self.imputers[method].fit_transform(data[numeric_cols])
                
                # 更新数值列
                imputed_data[numeric_cols] = imputed_values
            
            # 处理非数值列
            if non_numeric_cols:
                # 对非数值列使用众数填充
                non_numeric_imputer = SimpleImputer(strategy='most_frequent')
                non_numeric_values = non_numeric_imputer.fit_transform(data[non_numeric_cols])
                imputed_data[non_numeric_cols] = non_numeric_values
            
            # 合并其他列
            if not other_columns.empty:
                imputed_data = pd.concat([imputed_data[columns], other_columns], axis=1)
                
            self.fitted_imputer = self.imputers[method]
            
            return imputed_data, columns, None
            
        else:  # numpy array
            if scale_method:
                if scale_method not in self.scalers:
                    raise ValueError(f"Scale method {scale_method} not supported. Use one of {list(self.scalers.keys())}")
                scaler = self.scalers[scale_method]
                scaled_data = scaler.fit_transform(data)
                self.fitted_scaler = scaler
                imputed_values = self.imputers[method].fit_transform(scaled_data)
                # 逆变换
                imputed_values = scaler.inverse_transform(imputed_values)
            else:
                imputed_values = self.imputers[method].fit_transform(data)
                
            self.fitted_imputer = self.imputers[method]
            return imputed_values, None, None
            
    def get_statistics(self, data, columns=None):
        """
        获取数据缺失值统计
        
        Args:
            data: DataFrame 或类数组数据
            columns: 如果data是DataFrame，则为列名列表
            
        Returns:
            包含缺失值统计的字典
        """
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.columns
                
            stats = {
                'missing_counts': data[columns].isnull().sum().to_dict(),
                'missing_percentages': (data[columns].isnull().mean() * 100).to_dict(),
                'total_missing': data[columns].isnull().sum().sum(),
                'total_missing_percentage': (data[columns].isnull().sum().sum() / 
                                          (data[columns].shape[0] * len(columns)) * 100)
            }
            
        else:
            data_array = np.array(data)
            stats = {
                'missing_counts': {f"Feature_{i}": np.isnan(data_array[:, i]).sum()
                                 for i in range(data_array.shape[1])},
                'missing_percentages': {f"Feature_{i}": np.isnan(data_array[:, i]).mean() * 100
                                      for i in range(data_array.shape[1])},
                'total_missing': np.isnan(data_array).sum(),
                'total_missing_percentage': (np.isnan(data_array).sum() / 
                                          (data_array.shape[0] * data_array.shape[1]) * 100)
            }
            
        return stats