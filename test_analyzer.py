import sys
import os
from pathlib import Path

# 获取当前文件所在目录的父目录（即项目根目录）
root_path = str(Path(__file__).parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import unittest
import pandas as pd
import numpy as np
from src.analyzer import EnhancedStockAnalyzer

class TestEnhancedStockAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """
        构造一组可预测的静态测试数据。
        用 1, 2, 3... 这种数据可以人工算出预期结果，用来验证算法是否正确。
        """
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        self.data = pd.DataFrame({
            'open': np.linspace(100, 130, 30),
            'high': np.linspace(105, 135, 30),
            'low': np.linspace(95, 125, 30),
            'close': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] * 3, # 循环序列
            'volume': [1000] * 30
        }, index=dates)
        self.data.index.name = 'date'
        self.analyzer = EnhancedStockAnalyzer()

    def test_calculate_moving_averages(self):
        """测试移动平均线向量化计算"""
        period = 5
        result = self.analyzer.calculate_moving_averages(self.data, periods=[period])
        
        # 验证列名是否正确生成
        self.assertIn(f'MA_{period}', result.columns)
        
        # 验证第五个点的均线值：(10+11+12+13+14)/5 = 12.0
        # 注意：iloc[4] 是第五个元素
        self.assertAlmostEqual(result[f'MA_{period}'].iloc[4], 12.0)

    def test_max_drawdown(self):
        """测试最大回撤算法的核心逻辑"""
        # 构造一个先升后跌的序列：100 -> 200 -> 150
        prices = pd.Series([100, 120, 200, 180, 150], index=pd.date_range('2023-01-01', periods=5))
        mdd, peak, trough = self.analyzer.calculate_max_drawdown(prices)
        
        # 预期最大回撤 = (150 - 200) / 200 = -0.25
        self.assertEqual(mdd, -0.25)
        # 验证峰值点是否在 200 的位置
        self.assertEqual(prices.loc[peak], 200)

    def test_sharpe_ratio_with_zero_returns(self):
        """测试特殊情况：如果收益率为0，夏普比率不应报错"""
        returns = pd.Series([0.001, 0.001, 0.001], index=pd.date_range('2023-01-01', periods=3))
        # 这种情况标准差为 0，代码应能捕获并返回 0.0 或处理异常
        sharpe = self.analyzer.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

    def test_var_calculation(self):
        """测试风险价值（VaR）计算"""
        # 构造正态分布收益率
        returns = pd.Series(np.random.normal(0, 0.01, 100))
        var_results = self.analyzer.calculate_var(returns, confidence_level=0.95)
        
        self.assertIn('historical_var', var_results)
        self.assertIn('cvar', var_results)
        # VaR 在这种情况下应该是负数（代表亏损）
        self.assertLess(var_results['historical_var'], 0)

    def test_data_insufficient(self):
        """测试防御性逻辑：当数据行数极少时"""
        short_data = self.data.iloc[:2] # 只有两行
        prices = short_data['close']
        returns = prices.pct_change().dropna()
        
        report = self.analyzer.generate_risk_report(prices, returns)
        # 根据你的代码逻辑，数据不足应返回空字典或记录日志
        self.assertEqual(report, {})

if __name__ == '__main__':
    unittest.main()