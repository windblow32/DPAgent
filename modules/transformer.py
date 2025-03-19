import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataTransformer:
    def __init__(self):
        self.transformers = {
            'yeo-johnson': PowerTransformer(method='yeo-johnson'),
            'box-cox': PowerTransformer(method='box-cox'),
            'quantile_normal': QuantileTransformer(output_distribution='normal'),
            'quantile_uniform': QuantileTransformer(output_distribution='uniform')
        }
        self.fitted_transformer = None
        
    def transform(self, data, method='yeo-johnson', columns=None):
        """
        Transform data using specified method
        
        Args:
            data: DataFrame or array-like data
            method: 'yeo-johnson', 'box-cox', 'quantile_normal', or 'quantile_uniform'
            columns: list of column names if data is DataFrame
            
        Returns:
            transformed data and columns used
        """
        if method not in self.transformers:
            raise ValueError(f"Method {method} not supported. Use one of {list(self.transformers.keys())}")
            
        if isinstance(data, list):
            data = np.array(data)
            
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be DataFrame, numpy array, or list")
            
        transformer = self.transformers[method]
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
                
            # Store non-numeric columns
            non_numeric = data.select_dtypes(exclude=[np.number]).copy()
            
            # Transform numeric columns
            transformed_data = data.copy()
            transformed_data[columns] = transformer.fit_transform(data[columns])
            
            # Combine with non-numeric columns
            if not non_numeric.empty:
                transformed_data = pd.concat([transformed_data, non_numeric], axis=1)
                
            self.fitted_transformer = transformer
            return transformed_data, columns
            
        else:
            transformed_data = transformer.fit_transform(data)
            self.fitted_transformer = transformer
            return transformed_data, None
            
    def inverse_transform(self, data):
        """
        Inverse transform the transformed data
        """
        if self.fitted_transformer is None:
            raise ValueError("No transformer has been fitted yet")
            
        return self.fitted_transformer.inverse_transform(data)
