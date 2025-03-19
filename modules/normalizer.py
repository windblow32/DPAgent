import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from .visualizer import DataVisualizer

class DataNormalizer:
    def __init__(self):
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        self.fitted_scaler = None
        self.visualizer = DataVisualizer()
        
    def normalize(self, data, method='standard', columns=None, visualize=True):
        """
        Normalize data using specified method
        
        Args:
            data: DataFrame or array-like data
            method: 'standard', 'minmax', or 'robust'
            columns: list of column names if data is DataFrame
            visualize: whether to generate visualization plots
            
        Returns:
            normalized data, columns used, and visualization plots if requested
        """
        if method not in self.scalers:
            raise ValueError(f"Method {method} not supported. Use one of {list(self.scalers.keys())}")
            
        if isinstance(data, list):
            data = np.array(data)
        
        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be DataFrame, numpy array, or list")
            
        scaler = self.scalers[method]
        
        if isinstance(data, pd.DataFrame):
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns
            
            # Store original data for visualization
            original_data = data[columns].copy()
            
            # Store non-numeric columns
            non_numeric = data.select_dtypes(exclude=[np.number]).copy()
            
            # Normalize numeric columns
            normalized_data = data.copy()
            normalized_data[columns] = scaler.fit_transform(data[columns])
            
            # Combine with non-numeric columns
            if not non_numeric.empty:
                normalized_data = pd.concat([normalized_data, non_numeric], axis=1)
                
            self.fitted_scaler = scaler
            
            # Generate visualizations if requested
            plots = None
            if visualize:
                dist_plot = self.visualizer.compare_distributions(
                    original_data, 
                    normalized_data[columns], 
                    columns=columns,
                    title_suffix=f"({method} normalization)"
                )
                box_plot = self.visualizer.create_boxplots(
                    original_data, 
                    normalized_data[columns],
                    columns=columns,
                    title_suffix=f"({method} normalization)"
                )
                plots = {
                    'distribution_plot': dist_plot,
                    'box_plot': box_plot
                }
                
            return normalized_data, columns, plots
            
        else:
            # For numpy arrays
            original_data = data.copy()
            normalized_data = scaler.fit_transform(data)
            self.fitted_scaler = scaler
            
            # Generate visualizations if requested
            plots = None
            if visualize:
                dist_plot = self.visualizer.compare_distributions(
                    original_data, 
                    normalized_data,
                    columns=[f"Feature_{i}" for i in range(data.shape[1])],
                    title_suffix=f"({method} normalization)"
                )
                box_plot = self.visualizer.create_boxplots(
                    original_data, 
                    normalized_data,
                    columns=[f"Feature_{i}" for i in range(data.shape[1])],
                    title_suffix=f"({method} normalization)"
                )
                plots = {
                    'distribution_plot': dist_plot,
                    'box_plot': box_plot
                }
                
            return normalized_data, None, plots
            
    def inverse_transform(self, data):
        """
        Inverse transform normalized data
        """
        if self.fitted_scaler is None:
            raise ValueError("No scaler has been fitted yet")
            
        return self.fitted_scaler.inverse_transform(data)
