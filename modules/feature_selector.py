import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

class FeatureSelector:
    def __init__(self):
        self.methods = {
            'kbest': self._select_k_best,
            'mutual_info': self._mutual_info,
            'random_forest': self._random_forest
        }
        self.selected_features = None
        self.selector = None
        
    def select_features(self, X, y, method='kbest', n_features=10, **kwargs):
        """
        Select features using specified method
        
        Args:
            X: Feature DataFrame or array
            y: Target variable
            method: 'kbest', 'mutual_info', or 'random_forest'
            n_features: number of features to select
            **kwargs: additional parameters for specific methods
            
        Returns:
            Selected features and importance scores
        """
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported. Use one of {list(self.methods.keys())}")
            
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Encode target if it's categorical
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        if y.dtype == object or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
        selected_features, scores = self.methods[method](X, y, n_features, **kwargs)
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': scores
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        self.selected_features = feature_names[selected_features]
        return self.selected_features, importance_df
        
    def _select_k_best(self, X, y, n_features, **kwargs):
        """Select features using ANOVA F-value"""
        selector = SelectKBest(score_func=f_classif, k=n_features)
        selector.fit(X, y)
        self.selector = selector
        return selector.get_support(), selector.scores_
        
    def _mutual_info(self, X, y, n_features, **kwargs):
        """Select features using mutual information"""
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        selector.fit(X, y)
        self.selector = selector
        return selector.get_support(), selector.scores_
        
    def _random_forest(self, X, y, n_features, **kwargs):
        """Select features using Random Forest importance"""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = SelectFromModel(rf, max_features=n_features, prefit=False)
        selector.fit(X, y)
        self.selector = selector
        
        # Get feature importance scores
        importance_scores = selector.estimator_.feature_importances_
        return selector.get_support(), importance_scores
        
    def transform(self, X):
        """
        Transform data using selected features
        """
        if self.selector is None:
            raise ValueError("No feature selector has been fitted yet")
            
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(self.selector.transform(X), columns=self.selected_features)
        return self.selector.transform(X)
