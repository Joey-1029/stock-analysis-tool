# src/data_cleaner.py
import pandas as pd
import numpy as np
from pathlib import Path
import os

class StockDataCleaner:
    def __init__(self):
        # 根据你的实际结构调整路径
        self.project_root = Path(__file__).parent.parent
        self.raw_data_dir = self.project_root / 'data' / 'raw'
        self.cleaned_data_dir = self.project_root / 'data' / 'cleaned'
        
        # 确保目录存在
        self.cleaned_data_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("STOCK DATA CLEANER")
        print(f"Raw data: {self.raw_data_dir}")
        print(f"Cleaned data: {self.cleaned_data_dir}")
        print("=" * 60)
    
    def load_raw_data(self, filename):
        """加载原始数据文件"""
        filepath = self.raw_data_dir / filename
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded: {filename} ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def clean_data(self, df, ticker, stock_name):
        """清洗单个股票数据"""
        if df is None or df.empty:
            return None
        
        print(f"\nCleaning {stock_name} ({ticker})...")
        
        # 复制数据，避免修改原始
        cleaned = df.copy()
        
        # 1. 检查列名
        print(f"  Columns: {list(cleaned.columns)}")
        
        # 2. 处理日期列
        if 'date' in cleaned.columns:
            cleaned['date'] = pd.to_datetime(cleaned['date'])
            cleaned = cleaned.sort_values('date')
        elif 'Date' in cleaned.columns:
            cleaned['date'] = pd.to_datetime(cleaned['Date'])
            cleaned = cleaned.sort_values('date')
        else:
            # 尝试使用第一列作为日期
            cleaned.columns = ['date'] + list(cleaned.columns[1:])
            cleaned['date'] = pd.to_datetime(cleaned['date'])
            cleaned = cleaned.sort_values('date')
        
        # 3. 设置日期为索引
        cleaned.set_index('date', inplace=True)
        
        # 4. 重命名列（标准化）
        column_mapping = {
            '开盘': 'open', '开盘价': 'open',
            '最高': 'high', '最高价': 'high',
            '最低': 'low', '最低价': 'low',
            '收盘': 'close', '收盘价': 'close',
            '成交量': 'volume', '成交额': 'amount'
        }
        cleaned.rename(columns=column_mapping, inplace=True)
        
        # 5. 处理缺失值
        cleaned = cleaned.ffill().bfill()  # 向前填充，然后向后填充
        
        # 6. 添加计算列
        if 'close' in cleaned.columns:
            cleaned['daily_return'] = cleaned['close'].pct_change()
            cleaned['log_return'] = np.log(cleaned['close'] / cleaned['close'].shift(1))
            
            # 移动平均线
            cleaned['ma_5'] = cleaned['close'].rolling(window=5).mean()
            cleaned['ma_20'] = cleaned['close'].rolling(window=20).mean()
            cleaned['ma_60'] = cleaned['close'].rolling(window=60).mean()
        
        # 7. 保存清洗后的数据
        output_file = self.cleaned_data_dir / f"{ticker}_cleaned.csv"
        cleaned.to_csv(output_file)
        
        print(f"  ✓ Cleaned {len(cleaned)} rows")
        print(f"  ✓ Saved to: {output_file}")
        
        return cleaned
    
    def clean_all_stocks(self):
        """清洗所有股票数据"""
        print("\n" + "="*60)
        print("CLEANING ALL STOCK DATA")
        print("="*60)
        
        # 查找所有CSV文件
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        
        if not csv_files:
            print("No CSV files found in raw data directory")
            return {}
        
        cleaned_data = {}
        
        for filepath in csv_files:
            filename = filepath.name
            
            # 根据文件名确定股票信息
            if '00700' in filename:
                ticker = '00700.HK'
                name = 'Tencent'
            elif '09988' in filename:
                ticker = '09988.HK'
                name = 'Alibaba'
            else:
                ticker = filename.replace('.csv', '')
                name = 'Unknown'
            
            # 加载和清洗
            raw_df = self.load_raw_data(filename)
            if raw_df is not None:
                cleaned_df = self.clean_data(raw_df, ticker, name)
                if cleaned_df is not None:
                    cleaned_data[ticker] = cleaned_df
        
        # 创建合并的数据集
        if len(cleaned_data) >= 2:
            self.create_merged_dataset(cleaned_data)
        
        return cleaned_data
    
    def create_merged_dataset(self, cleaned_data):
        """创建合并的数据集用于分析"""
        print("\n" + "="*60)
        print("CREATING MERGED DATASET")
        print("="*60)
        
        # 合并所有股票的收盘价
        close_prices = pd.DataFrame()
        
        for ticker, df in cleaned_data.items():
            if 'close' in df.columns:
                close_prices[ticker] = df['close']
        
        if not close_prices.empty:
            # 保存合并的收盘价数据
            merged_file = self.cleaned_data_dir / 'merged_close_prices.csv'
            close_prices.to_csv(merged_file)
            
            print(f"✓ Merged dataset created")
            print(f"  Stocks: {list(close_prices.columns)}")
            print(f"  Trading days: {len(close_prices)}")
            print(f"  Saved to: {merged_file}")
        
        return close_prices

# 使用示例
if __name__ == "__main__":
    cleaner = StockDataCleaner()
    
    # 方法1: 清洗所有股票
    cleaned_data = cleaner.clean_all_stocks()
    
    # 方法2: 清洗单个股票
    # cleaned_df = cleaner.clean_data(df, '00700.HK', 'Tencent')
    
    if cleaned_data:
        print("\n" + "="*60)
        print("CLEANING COMPLETE!")
        print("="*60)
        
        for ticker, df in cleaned_data.items():
            print(f"\n{ticker}:")
            print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"  Latest price: {df['close'].iloc[-1] if 'close' in df.columns else 'N/A'}")