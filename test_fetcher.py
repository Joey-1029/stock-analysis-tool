import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
from pathlib import Path
import json

# 路径修复
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.data_fetcher_akshare import AKShareDataFetcher

class TestAKShareDataFetcher(unittest.TestCase):

    def setUp(self):
        """每个测试前初始化"""
        self.fetcher = AKShareDataFetcher()
        # 模拟一份合规的股票数据
        self.mock_df = pd.DataFrame({
            'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000, 1100]
        })

    @patch('akshare.stock_hk_daily')
    def test_retry_mechanism(self, mock_ak):
        """测试重试机制：前两次失败，第三次成功"""
        # 设置模拟行为：两次异常，一次返回正常数据
        mock_ak.side_effect = [
            Exception("Network Timeout 1"),
            Exception("Network Timeout 2"),
            self.mock_df
        ]

        # 执行获取
        result = self.fetcher.get_hk_stock("00700")

        # 断言：虽然报错了两次，但最终应该拿到数据
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        # 断言：内部确实调用了 3 次 akshare
        self.assertEqual(mock_ak.call_count, 3)

    def test_validate_data_logic(self):
        """测试数据验证逻辑"""
        # 1. 正常数据应返回 True
        self.assertTrue(self.fetcher._validate_data(self.mock_df, "TEST"))

        # 2. 含有 NaN 的数据应能识别（根据你的代码，它会报 warning 但返回 True）
        bad_df = self.mock_df.copy()
        bad_df.loc[0, 'close'] = None
        # 验证其是否能检测到 NaN（观察日志或检查逻辑）
        self.assertTrue(self.fetcher._validate_data(bad_df, "TEST"))

        # 3. 缺少必要列的数据应返回 False
        empty_df = pd.DataFrame({'wrong_col': [1, 2]})
        self.assertFalse(self.fetcher._validate_data(empty_df, "TEST"))

    @patch('src.data_fetcher_akshare.AKShareDataFetcher._is_already_downloaded')
    def test_cache_skip_logic(self, mock_cache_check):
        """测试缓存跳过逻辑"""
        # 模拟缓存检查返回 True（表示今日已下载）
        mock_cache_check.return_value = True
        
        # 即使调用下载，也应该直接返回 None 并不执行后续逻辑
        result = self.fetcher.get_a_stock("600519")
        self.assertIsNone(result)

    def test_save_data_integrity(self):
        """测试数据保存是否生成了文件"""
        filename = self.fetcher._save_data(self.mock_df, "UNIT_TEST", "TestMarket")
        filepath = self.fetcher.raw_dir / filename
        
        # 检查文件物理存在
        self.assertTrue(filepath.exists())
        
        # 清理测试产生的文件
        if filepath.exists():
            filepath.unlink()

if __name__ == '__main__':
    unittest.main()