import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 路径修复逻辑：确保能找到 src 目录
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.data_cleaner import StockDataCleaner

class TestStockDataCleaner(unittest.TestCase):
    
    def setUp(self):
        """初始化 Cleaner 和 模拟脏数据"""
        self.cleaner = StockDataCleaner()
        
        # 模拟一个极其糟糕的原始数据 DataFrame
        self.dirty_df = pd.DataFrame({
            '日期': ['2023-01-05', '2023-01-01', '2023-01-02', '2023-01-02'], # 乱序、重复
            '收盘价': [100.0, 98.0, np.nan, 99.0], # 包含缺失值和重复日期
            '成交量': [1000, 800, 900, 900]
        })

    def test_column_standardization(self):
        """测试中文列名是否能正确映射为英文"""
        # 注意：clean_data 会保存文件，测试时我们主要检查其返回的 DataFrame
        cleaned = self.cleaner.clean_data(self.dirty_df, 'TEST_TICKER', 'TestStock')
        
        self.assertIn('close', cleaned.columns)
        self.assertIn('volume', cleaned.columns)
        self.assertNotIn('收盘价', cleaned.columns)

    def test_date_sorting_and_indexing(self):
        """测试日期是否排序并正确设置为索引"""
        cleaned = self.cleaner.clean_data(self.dirty_df, 'TEST_TICKER', 'TestStock')
        
        # 检查索引是否为 DatetimeIndex
        self.assertIsInstance(cleaned.index, pd.DatetimeIndex)
        # 检查是否升序排列：第一行应该是 1月1日
        self.assertEqual(cleaned.index[0], pd.Timestamp('2023-01-01'))

    def test_missing_value_handling(self):
        """测试 ffill/bfill 逻辑是否处理了 NaN"""
        cleaned = self.cleaner.clean_data(self.dirty_df, 'TEST_TICKER', 'TestStock')
        
        # 原数据 1月2日 有个 NaN，检查处理后是否消失
        self.assertFalse(cleaned['close'].isnull().any())
        # 验证填充值是否合理（应该是前后的均值或邻近值）
        self.assertTrue(cleaned.loc['2023-01-02', 'close'].iloc[0] > 0)

    def test_indicators_calculation(self):
        """测试收益率和均线是否生成"""
        # 为了计算 MA20，我们需要更多数据
        large_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=25),
            'close': np.linspace(100, 125, 25)
        })
        cleaned = self.cleaner.clean_data(large_df, 'MA_TEST', 'MATest')
        
        self.assertIn('daily_return', cleaned.columns)
        self.assertIn('ma_5', cleaned.columns)
        self.assertIn('ma_20', cleaned.columns)
        # 第20天应该有 MA20 的值
        self.assertFalse(np.isnan(cleaned['ma_20'].iloc[19]))

    def test_empty_df_handling(self):
        """测试空数据的防御性"""
        result = self.cleaner.clean_data(pd.DataFrame(), 'EMPTY', 'None')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()