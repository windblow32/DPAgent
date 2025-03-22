import unittest
import pandas as pd
import numpy as np
from modules.imputer import DataImputer

class TestDataImputer(unittest.TestCase):
    def setUp(self):
        """
        创建测试数据
        """
        # 创建一个包含缺失值的数据框
        self.data = pd.DataFrame({
            'numeric1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'numeric2': [10.0, np.nan, 30.0, np.nan, 50.0],
            'category': ['A', 'B', np.nan, 'B', 'C'],
            'all_missing': [np.nan] * 5
        })
        self.imputer = DataImputer()

    def test_mean_imputation(self):
        """测试均值填充"""
        imputed_data, cols, _ = self.imputer.impute(
            self.data, 
            method='mean',
            columns=['numeric1', 'numeric2']
        )
        
        # 验证数值是否被正确填充
        self.assertFalse(imputed_data['numeric1'].isnull().any())
        self.assertFalse(imputed_data['numeric2'].isnull().any())
        
        # 验证填充值是否正确（使用均值）
        expected_mean1 = self.data['numeric1'].mean()
        expected_mean2 = self.data['numeric2'].mean()
        
        # 检查原本是 NaN 的位置是否被填充为均值
        self.assertAlmostEqual(imputed_data.loc[2, 'numeric1'], expected_mean1)
        self.assertAlmostEqual(imputed_data.loc[1, 'numeric2'], expected_mean2)

    def test_median_imputation(self):
        """测试中位数填充"""
        imputed_data, cols, _ = self.imputer.impute(
            self.data,
            method='median',
            columns=['numeric1', 'numeric2']
        )
        
        # 验证数值是否被正确填充
        self.assertFalse(imputed_data['numeric1'].isnull().any())
        self.assertFalse(imputed_data['numeric2'].isnull().any())
        
        # 验证填充值是否正确（使用中位数）
        expected_median1 = self.data['numeric1'].median()
        expected_median2 = self.data['numeric2'].median()
        
        self.assertAlmostEqual(imputed_data.loc[2, 'numeric1'], expected_median1)
        self.assertAlmostEqual(imputed_data.loc[1, 'numeric2'], expected_median2)

    def test_most_frequent_imputation(self):
        """测试众数填充"""
        imputed_data, cols, _ = self.imputer.impute(
            self.data,
            method='most_frequent',
            columns=['category']
        )
        
        # 验证分类值是否被正确填充
        self.assertFalse(imputed_data['category'].isnull().any())
        
        # 验证填充值是否为众数
        mode_value = self.data['category'].mode()[0]  # 'B'
        self.assertEqual(imputed_data.loc[2, 'category'], mode_value)

    def test_knn_imputation(self):
        """测试KNN填充"""
        imputed_data, cols, _ = self.imputer.impute(
            self.data,
            method='knn',
            columns=['numeric1', 'numeric2']
        )
        
        # 验证数值是否被填充
        self.assertFalse(imputed_data['numeric1'].isnull().any())
        self.assertFalse(imputed_data['numeric2'].isnull().any())

    def test_invalid_method(self):
        """测试无效的填充方法"""
        with self.assertRaises(ValueError):
            self.imputer.impute(self.data, method='invalid_method')

    def test_scaled_imputation(self):
        """测试带缩放的填充"""
        imputed_data, cols, _ = self.imputer.impute(
            self.data,
            method='mean',
            columns=['numeric1', 'numeric2'],
            scale_method='standard'
        )
        
        # 验证数值是否被填充
        self.assertFalse(imputed_data['numeric1'].isnull().any())
        self.assertFalse(imputed_data['numeric2'].isnull().any())
        
        # 验证数据是否被标准化（均值接近0，标准差接近1）
        self.assertAlmostEqual(imputed_data['numeric1'].mean(), 0, places=1)
        self.assertAlmostEqual(imputed_data['numeric1'].std(), 1, places=1)

if __name__ == '__main__':
    unittest.main()
