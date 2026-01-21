# notebooks/utils.py
"""
Jupyter Notebook工具函数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def setup_notebook():
    """设置Notebook环境"""
    # 添加项目路径
    project_root = Path.cwd().parent
    sys.path.append(str(project_root))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100
    
    print(f"✅ Notebook环境设置完成")
    print(f"项目根目录: {project_root}")
    
    return project_root

def load_project_modules():
    """加载项目模块"""
    try:
        from src.analyzer import StockAnalyzer
        from src.visualizer import StockVisualizer
        from src.data_reporter import DataQualityReporter
        
        analyzer = StockAnalyzer()
        visualizer = StockVisualizer()
        reporter = DataQualityReporter()
        
        print("✅ 项目模块加载成功")
        return analyzer, visualizer, reporter
        
    except ImportError as e:
        print(f"❌ 模块加载失败: {e}")
        return None, None, None

def create_sample_data(days=100):
    """创建示例数据用于演示"""
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    np.random.seed(42)
    
    # 生成有趋势和波动性的价格数据
    trend = np.linspace(100, 120, days)
    noise = np.cumsum(np.random.randn(days) * 0.5)
    
    data = pd.DataFrame({
        'date': dates,
        'open': trend + noise - 0.5,
        'high': trend + noise + np.random.uniform(0.5, 2.0, days),
        'low': trend + noise - np.random.uniform(0.5, 2.0, days),
        'close': trend + noise,
        'volume': np.random.randint(1000000, 5000000, days),
    })
    
    data.set_index('date', inplace=True)
    return data

def format_percentage(value):
    """格式化百分比显示"""
    return f"{value:.2%}"

def format_currency(value):
    """格式化货币显示"""
    return f"¥{value:,.2f}" if value >= 0 else f"-¥{abs(value):,.2f}"

def create_summary_table(df, ticker):
    """创建数据摘要表格"""
    if 'close' not in df.columns:
        return None
    
    summary = {
        '股票代码': ticker,
        '数据天数': len(df),
        '起始日期': df.index.min().strftime('%Y-%m-%d'),
        '结束日期': df.index.max().strftime('%Y-%m-%d'),
        '起始价格': format_currency(df['close'].iloc[0]),
        '结束价格': format_currency(df['close'].iloc[-1]),
        '最高价格': format_currency(df['close'].max()),
        '最低价格': format_currency(df['close'].min()),
        '平均价格': format_currency(df['close'].mean()),
        '总收益率': format_percentage(df['close'].iloc[-1] / df['close'].iloc[0] - 1),
        '日收益率均值': format_percentage(df['close'].pct_change().mean()),
        '日收益率标准差': format_percentage(df['close'].pct_change().std()),
        '夏普比率': f"{(df['close'].pct_change().mean() / df['close'].pct_change().std() * np.sqrt(252)):.3f}" 
                   if df['close'].pct_change().std() > 0 else "N/A"
    }
    
    return pd.DataFrame(list(summary.items()), columns=['指标', '值'])