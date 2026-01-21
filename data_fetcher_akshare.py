# src/data_fetcher_akshare.py
import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import os

# ========== 添加SSL修复代码（解决你的网络问题）==========
import ssl
import urllib3

# 禁用SSL验证警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 创建不验证SSL的上下文（解决SSL错误）
ssl._create_default_https_context = ssl._create_unverified_context
# =====================================================

class AKShareDataFetcher:
    """使用AKShare获取股票数据"""
    
    def __init__(self):
        """初始化数据获取器"""
        # 设置项目路径
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.raw_dir = self.data_dir / 'raw'  # 原始数据目录
        self.cleaned_dir = self.data_dir / 'cleaned'  # 清洗数据目录
        
        # 确保目录存在
        self.data_dir.mkdir(exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.cleaned_dir.mkdir(exist_ok=True)
        
        # 设置请求延迟（避免被封IP）
        self.request_delay = 1  # 秒
        
        print("=" * 60)
        print("AKShare 数据获取器已初始化")
        print(f"原始数据目录: {self.raw_dir}")
        print(f"清洗数据目录: {self.cleaned_dir}")
        print("=" * 60)
    
    def get_a_stock(self, symbol, start_date='20230101', end_date=None):
        """
        获取A股数据
        symbol: 股票代码，格式如 '000001'（平安银行）
                深市股票: 000001（平安银行），300059（东方财富）
                沪市股票: 600519（贵州茅台），601318（中国平安）
        """
        print(f"\n获取A股: {symbol}")
        
        # 确定交易所前缀
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"  # 上海交易所
        elif symbol.startswith('0') or symbol.startswith('3'):
            ts_code = f"{symbol}.SZ"  # 深圳交易所
        else:
            print(f"无法识别股票代码格式: {symbol}")
            return None
        
        try:
            # 获取日线数据
            print(f"正在下载 {ts_code} ({start_date} 到 {end_date or '最新'})...")
            
            stock_zh_a_daily_df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if stock_zh_a_daily_df.empty:
                print(f"警告: {ts_code} 数据为空")
                return None
            
            print(f"✓ 成功获取 {len(stock_zh_a_daily_df)} 条记录")
            
            # 保存数据到raw目录
            self._save_data(stock_zh_a_daily_df, ts_code, 'A股')
            
            # 添加延迟
            time.sleep(self.request_delay)
            
            return stock_zh_a_daily_df
            
        except Exception as e:
            print(f"✗ 获取 {ts_code} 失败: {e}")
            return None
    
    def get_us_stock(self, symbol, period='daily'):
        """
        获取美股数据
        symbol: 股票代码，如 'AAPL', 'MSFT'
        period: 周期，'daily'(日线), 'weekly'(周线), 'monthly'(月线)
        """
        print(f"\n获取美股: {symbol}")
        
        # AKShare的美股代码映射（需要转换）
        symbol_mapping = {
            'AAPL': '105.AAPL',
            'MSFT': '106.MSFT', 
            'GOOGL': '107.GOOGL',
            'AMZN': '108.AMZN',
            'TSLA': '109.TSLA',
            'NVDA': '110.NVDA',
            'META': '128.META',
            'BABA': '116.BABA',
            'PDD': '113.PDD',
            'JD': '115.JD',
        }
        
        ak_code = symbol_mapping.get(symbol)
        if not ak_code:
            print(f"暂不支持的美股代码: {symbol}")
            # 尝试直接使用
            ak_code = symbol
        
        try:
            print(f"正在下载 {symbol} ({ak_code})...")
            
            if period == 'daily':
                stock_us_daily_df = ak.stock_us_daily(symbol=ak_code, adjust="qfq")
            elif period == 'weekly':
                stock_us_daily_df = ak.stock_us_weekly(symbol=ak_code, adjust="qfq")
            elif period == 'monthly':
                stock_us_daily_df = ak.stock_us_monthly(symbol=ak_code, adjust="qfq")
            else:
                print(f"不支持的周期: {period}")
                return None
            
            if stock_us_daily_df.empty:
                print(f"警告: {symbol} 数据为空")
                return None
            
            print(f"✓ 成功获取 {len(stock_us_daily_df)} 条记录")
            
            # 保存数据到raw目录
            self._save_data(stock_us_daily_df, symbol, '美股')
            
            # 添加延迟
            time.sleep(self.request_delay)
            
            return stock_us_daily_df
            
        except Exception as e:
            print(f"✗ 获取 {symbol} 失败: {e}")
            return None
    
    def get_hk_stock(self, symbol):
        """
        获取港股数据
        symbol: 港股代码，如 '00700'（腾讯）, '09988'（阿里）
        """
        print(f"\n获取港股: {symbol}")
        
        try:
            print(f"正在下载 {symbol}...")
            
            stock_hk_daily_df = ak.stock_hk_daily(symbol=symbol, adjust="qfq")
            
            if stock_hk_daily_df.empty:
                print(f"警告: {symbol} 数据为空")
                return None
            
            print(f"✓ 成功获取 {len(stock_hk_daily_df)} 条记录")
            
            # 保存数据到raw目录
            self._save_data(stock_hk_daily_df, f"HK_{symbol}", '港股')
            
            # 添加延迟
            time.sleep(self.request_delay)
            
            return stock_hk_daily_df
            
        except Exception as e:
            print(f"✗ 获取 {symbol} 失败: {e}")
            return None
    
    def _save_data(self, data, symbol, market_type, data_type='raw'):
        """
        保存数据到CSV文件
        data_type: 'raw'（原始数据）或 'cleaned'（清洗后数据）
        """
        # 根据数据类型选择目录
        if data_type == 'raw':
            save_dir = self.raw_dir
            prefix = 'RAW'
        else:
            save_dir = self.cleaned_dir
            prefix = 'CLEANED'
        
        # 生成文件名（更简洁的格式）
        filename = f"{symbol.replace('.', '_')}_{market_type}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = save_dir / filename
        
        # 确保数据列名标准化
        if 'date' in data.columns:
            data = data.sort_values('date')
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
        
        data.to_csv(filepath)
        print(f"  数据保存到: {filepath}")
        
        # 显示数据预览
        print("  前5行数据:")
        print(data.head())
        print()
        
        return filepath
    
    def get_index_data(self, symbol='sh000001'):
        """
        获取指数数据
        symbol: 指数代码
            sh000001: 上证指数
            sz399001: 深证成指
            sz399006: 创业板指
            hkhsi: 恒生指数
        """
        print(f"\n获取指数: {symbol}")
        
        try:
            if symbol.startswith('sh') or symbol.startswith('sz'):
                # A股指数
                index_zh_a_df = ak.index_zh_a_hist(symbol=symbol, period="daily")
            elif symbol == 'hkhsi':
                # 恒生指数
                index_zh_a_df = ak.index_hk_hist(symbol="HSI")
            else:
                print(f"不支持的指数代码: {symbol}")
                return None
            
            if index_zh_a_df.empty:
                print(f"警告: {symbol} 指数数据为空")
                return None
            
            print(f"✓ 成功获取 {len(index_zh_a_df)} 条记录")
            
            # 保存数据到raw目录
            self._save_data(index_zh_a_df, symbol, '指数')
            
            return index_zh_a_df
            
        except Exception as e:
            print(f"✗ 获取 {symbol} 失败: {e}")
            return None
    
    def download_hk_stocks_only(self):
        """只下载港股数据（避免网络问题）"""
        print("\n" + "=" * 60)
        print("开始下载港股数据...")
        print("=" * 60)
        
        # 港股示例
        hk_stocks = ['00700', '09988']
        
        for stock in hk_stocks:
            self.get_hk_stock(stock)
        
        print("=" * 60)
        print("港股数据下载完成!")
        print(f"数据保存在: {self.raw_dir}")
        print("=" * 60)
    
    def download_multiple_stocks(self):
        """下载多只股票数据（示例）"""
        print("\n" + "=" * 60)
        print("开始下载多只股票数据...")
        print("=" * 60)
        
        # 先下载港股（最需要的）
        print("\n第一步：下载港股数据")
        hk_stocks = ['00700', '09988']
        for stock in hk_stocks:
            self.get_hk_stock(stock)
        
        # 询问是否下载其他数据
        choice = input("\n是否下载A股和美股数据? (y/n): ").lower()
        
        if choice == 'y':
            print("\n第二步：下载A股数据")
            a_stocks = ['000001', '600519']
            for stock in a_stocks:
                self.get_a_stock(stock, start_date='20240101')
            
            print("\n第三步：下载美股数据")
            us_stocks = ['AAPL', 'MSFT']
            for stock in us_stocks:
                self.get_us_stock(stock)
            
            print("\n第四步：下载指数数据")
            self.get_index_data('sh000001')
        
        print("=" * 60)
        print("数据下载完成!")
        print(f"所有原始数据保存在: {self.raw_dir}")
        print(f"清洗数据目录: {self.cleaned_dir}")
        print("=" * 60)
    
    def check_existing_data(self):
        """检查已有数据"""
        print("\n" + "=" * 60)
        print("检查已有数据文件")
        print("=" * 60)
        
        print(f"\n原始数据目录 ({self.raw_dir}):")
        raw_files = list(self.raw_dir.glob("*.csv"))
        if raw_files:
            for file in raw_files:
                print(f"  - {file.name}")
        else:
            print("  暂无原始数据")
        
        print(f"\n清洗数据目录 ({self.cleaned_dir}):")
        cleaned_files = list(self.cleaned_dir.glob("*.csv"))
        if cleaned_files:
            for file in cleaned_files:
                print(f"  - {file.name}")
        else:
            print("  暂无清洗数据")

# 主程序入口
def main():
    """主函数 - 用于从main.py调用"""
    fetcher = AKShareDataFetcher()
    
    # 先检查已有数据
    fetcher.check_existing_data()
    
    print("\n选择下载模式:")
    print("1. 只下载港股（腾讯、阿里）")
    print("2. 下载全部数据（交互式）")
    print("3. 检查数据状态")
    
    choice = input("\n请输入选择 (1-3): ").strip()
    
    if choice == '1':
        fetcher.download_hk_stocks_only()
    elif choice == '2':
        fetcher.download_multiple_stocks()
    elif choice == '3':
        fetcher.check_existing_data()
    else:
        print("无效选择，使用默认模式（只下载港股）")
        fetcher.download_hk_stocks_only()
    
    print("\n操作完成！")

# 直接运行此文件时的入口
if __name__ == "__main__":
    main()