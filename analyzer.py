# src/analyzer.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class StockAnalyzer:
    """股票技术分析器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.cleaned_dir = self.data_dir / 'cleaned'
        
        # 设置图表样式
        plt.rcParams.update({
            'figure.autolayout': True,
            'figure.titlesize': 'large',
            'axes.titlesize': 'medium',
            'axes.labelsize': 'small',
        })
        
        print("=" * 60)
        print("STOCK TECHNICAL ANALYZER")
        print(f"Data directory: {self.cleaned_dir}")
        print("=" * 60)
    
    def load_cleaned_data(self, ticker):
        """加载清洗后的数据"""
        # 支持多种文件名格式
        possible_files = [
            f"{ticker}_cleaned.csv",
            f"{ticker.replace('.', '_')}_cleaned.csv",
            f"RAW_{ticker.replace('.', '_')}_*_cleaned.csv"
        ]
        
        for pattern in possible_files:
            files = list(self.cleaned_dir.glob(pattern))
            if files:
                filepath = files[0]
                print(f"Loading: {filepath.name}")
                df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
                return df
        
        print(f"No cleaned data found for {ticker}")
        return None
    
    def calculate_moving_averages(self, df, periods=[5, 10, 20, 50, 200]):
        """计算移动平均线"""
        if 'close' not in df.columns:
            print("Error: 'close' column not found")
            return df
        
        for period in periods:
            if len(df) >= period:
                df[f'MA{period}'] = df['close'].rolling(window=period).mean()
                print(f"  Added MA{period}")
            else:
                print(f"  Skipped MA{period} (insufficient data: {len(df)} < {period})")
        
        return df
    
    def calculate_rsi(self, df, period=14):
        """计算相对强弱指数RSI"""
        if 'close' not in df.columns:
            return df
        
        delta = df['close'].diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均增益和平均损失
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算RS
        rs = avg_gain / avg_loss
        
        # 计算RSI
        df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # 添加超买超卖线
        df['RSI_Overbought'] = 70
        df['RSI_Oversold'] = 30
        
        print(f"  Added RSI_{period}")
        return df
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        if 'close' not in df.columns:
            return df
        
        # 计算EMA
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD线
        df['MACD'] = ema_fast - ema_slow
        
        # 信号线
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # 柱状图
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        print(f"  Added MACD (fast={fast}, slow={slow}, signal={signal})")
        return df
    
    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """计算布林带"""
        if 'close' not in df.columns:
            return df
        
        # 中轨（移动平均）
        df['BB_Middle'] = df['close'].rolling(window=period).mean()
        
        # 计算标准差
        rolling_std = df['close'].rolling(window=period).std()
        
        # 上轨和下轨
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
        
        # 带宽和%b
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_%B'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        print(f"  Added Bollinger Bands (period={period}, std={std_dev})")
        return df
    
    def calculate_volume_indicators(self, df, period=20):
        """计算成交量指标"""
        if 'volume' not in df.columns:
            return df
        
        # 成交量移动平均
        df[f'Volume_MA{period}'] = df['volume'].rolling(window=period).mean()
        
        # 成交量比率
        df['Volume_Ratio'] = df['volume'] / df[f'Volume_MA{period}']
        
        # 成交量加权平均价
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        print(f"  Added volume indicators (period={period})")
        return df
    
    def calculate_support_resistance(self, df, window=20):
        """计算支撑位和阻力位"""
        if 'close' not in df.columns:
            return df
        
        # 滚动最高价和最低价
        df['Resistance'] = df['high'].rolling(window=window, center=True).max()
        df['Support'] = df['low'].rolling(window=window, center=True).min()
        
        print(f"  Added support/resistance levels (window={window})")
        return df
    
    def calculate_all_indicators(self, df):
        """计算所有技术指标"""
        print("\nCalculating technical indicators...")
        
        # 基础指标
        df = self.calculate_moving_averages(df, periods=[5, 10, 20, 50, 100])
        
        # 动量指标
        df = self.calculate_rsi(df, period=14)
        df = self.calculate_macd(df)
        
        # 波动率指标
        df = self.calculate_bollinger_bands(df)
        
        # 成交量指标
        df = self.calculate_volume_indicators(df)
        
        # 支撑阻力
        df = self.calculate_support_resistance(df)
        
        print("✓ All technical indicators calculated")
        return df
    
    def generate_analysis_report(self, df, ticker):
        """生成分析报告"""
        report = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': f"{df.index.min().date()} to {df.index.max().date()}",
            'trading_days': len(df),
            'basic_stats': {},
            'technical_signals': {},
            'recommendations': []
        }
        
        # 基础统计
        if 'close' in df.columns:
            report['basic_stats'] = {
                'first_price': df['close'].iloc[0],
                'last_price': df['close'].iloc[-1],
                'total_return': f"{((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100):.2f}%",
                'highest_price': df['close'].max(),
                'lowest_price': df['close'].min(),
                'average_price': df['close'].mean(),
                'price_std': df['close'].std(),
                'avg_daily_return': f"{(df['daily_return'].mean() * 100):.3f}%" if 'daily_return' in df.columns else 'N/A',
                'daily_volatility': f"{(df['daily_return'].std() * 100):.3f}%" if 'daily_return' in df.columns else 'N/A',
            }
        
        # 技术信号分析
        technical_signals = {}
        
        # RSI信号
        if 'RSI_14' in df.columns:
            current_rsi = df['RSI_14'].iloc[-1]
            if current_rsi > 70:
                technical_signals['rsi'] = 'Overbought'
                report['recommendations'].append('RSI indicates overbought conditions - consider taking profits')
            elif current_rsi < 30:
                technical_signals['rsi'] = 'Oversold'
                report['recommendations'].append('RSI indicates oversold conditions - potential buying opportunity')
            else:
                technical_signals['rsi'] = 'Neutral'
        
        # MACD信号
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_Signal'].iloc[-1]
            prev_macd = df['MACD'].iloc[-2]
            prev_signal = df['MACD_Signal'].iloc[-2]
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                technical_signals['macd'] = 'Bullish Crossover'
                report['recommendations'].append('MACD bullish crossover detected')
            elif current_macd < current_signal and prev_macd >= prev_signal:
                technical_signals['macd'] = 'Bearish Crossover'
                report['recommendations'].append('MACD bearish crossover detected')
            else:
                technical_signals['macd'] = 'Neutral'
        
        # 移动平均线信号
        ma_columns = [col for col in df.columns if col.startswith('MA')]
        if len(ma_columns) >= 2:
            ma_columns.sort(key=lambda x: int(x[2:]) if x[2:].isdigit() else 0)
            short_ma = ma_columns[0]
            long_ma = ma_columns[1] if len(ma_columns) > 1 else None
            
            if long_ma:
                current_price = df['close'].iloc[-1]
                short_ma_value = df[short_ma].iloc[-1]
                long_ma_value = df[long_ma].iloc[-1]
                
                if current_price > short_ma_value > long_ma_value:
                    technical_signals['trend'] = 'Strong Uptrend'
                elif current_price < short_ma_value < long_ma_value:
                    technical_signals['trend'] = 'Strong Downtrend'
                else:
                    technical_signals['trend'] = 'Consolidation'
        
        report['technical_signals'] = technical_signals
        
        # 如果没有建议，添加默认建议
        if not report['recommendations']:
            report['recommendations'].append('No strong technical signals detected. Monitor key support/resistance levels.')
        
        return report
    
    def plot_technical_analysis(self, df, ticker):
        """绘制技术分析图表"""
        if df is None or df.empty:
            print("No data to plot")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 价格和移动平均线
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=2, color='black')
        
        # 添加移动平均线
        ma_colors = ['blue', 'green', 'orange', 'red', 'purple']
        ma_columns = [col for col in df.columns if col.startswith('MA')]
        for i, ma_col in enumerate(ma_columns[:5]):  # 最多显示5条MA
            ax1.plot(df.index, df[ma_col], label=ma_col, linewidth=1, 
                    alpha=0.7, color=ma_colors[i % len(ma_colors)])
        
        ax1.set_title(f'{ticker} - Price and Moving Averages')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = plt.subplot(4, 1, 2)
        if 'RSI_14' in df.columns:
            ax2.plot(df.index, df['RSI_14'], label='RSI(14)', linewidth=2, color='blue')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax2.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
            ax2.set_title('Relative Strength Index (RSI)')
            ax2.set_ylabel('RSI')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = plt.subplot(4, 1, 3)
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            ax3.plot(df.index, df['MACD'], label='MACD', linewidth=2, color='blue')
            ax3.plot(df.index, df['MACD_Signal'], label='Signal Line', linewidth=1.5, color='red')
            
            # 柱状图
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
            ax3.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.5, label='Histogram')
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('MACD')
            ax3.set_ylabel('MACD')
            ax3.legend(loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # 4. 成交量
        ax4 = plt.subplot(4, 1, 4)
        if 'volume' in df.columns:
            # 上涨下跌的颜色
            colors = ['green' if close >= open_ else 'red' 
                     for close, open_ in zip(df['close'], df['open'])]
            ax4.bar(df.index, df['volume'], color=colors, alpha=0.7, label='Volume')
            
            if 'Volume_MA20' in df.columns:
                ax4.plot(df.index, df['Volume_MA20'], label='20-day MA', 
                        color='blue', linewidth=2, alpha=0.8)
            
            ax4.set_title('Trading Volume')
            ax4.set_ylabel('Volume')
            ax4.set_xlabel('Date')
            ax4.legend(loc='upper left')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_stock(self, ticker):
        """分析单个股票的完整流程"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*60}")
        
        # 1. 加载数据
        df = self.load_cleaned_data(ticker)
        if df is None:
            print(f"No data found for {ticker}")
            return None
        
        # 2. 计算技术指标
        df_with_indicators = self.calculate_all_indicators(df.copy())
        
        # 3. 生成报告
        report = self.generate_analysis_report(df_with_indicators, ticker)
        
        # 4. 打印报告
        self.print_analysis_report(report)
        
        # 5. 绘制图表
        self.plot_technical_analysis(df_with_indicators, ticker)
        
        # 6. 保存分析结果
        self.save_analysis_results(df_with_indicators, ticker, report)
        
        return df_with_indicators, report
    
    def print_analysis_report(self, report):
        """打印分析报告"""
        print("\n" + "="*60)
        print("TECHNICAL ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nStock: {report['ticker']}")
        print(f"Analysis Date: {report['analysis_date']}")
        print(f"Data Period: {report['data_period']}")
        print(f"Trading Days: {report['trading_days']}")
        
        print("\nBASIC STATISTICS:")
        print("-" * 40)
        for key, value in report['basic_stats'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\nTECHNICAL SIGNALS:")
        print("-" * 40)
        for key, value in report['technical_signals'].items():
            print(f"  {key.upper()}: {value}")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*60)
    
    def save_analysis_results(self, df, ticker, report):
        """保存分析结果"""
        # 保存带指标的数据
        output_dir = self.data_dir / 'analysis'
        output_dir.mkdir(exist_ok=True)
        
        # 保存数据
        data_filename = f"{ticker.replace('.', '_')}_with_indicators_{datetime.now().strftime('%Y%m%d')}.csv"
        data_path = output_dir / data_filename
        df.to_csv(data_path)
        print(f"✓ Data with indicators saved to: {data_path}")
        
        # 保存报告
        report_filename = f"{ticker.replace('.', '_')}_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt"
        report_path = output_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("STOCK TECHNICAL ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Stock: {report['ticker']}\n")
            f.write(f"Analysis Date: {report['analysis_date']}\n")
            f.write(f"Data Period: {report['data_period']}\n")
            f.write(f"Trading Days: {report['trading_days']}\n\n")
            
            f.write("BASIC STATISTICS:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['basic_stats'].items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nTECHNICAL SIGNALS:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['technical_signals'].items():
                f.write(f"  {key.upper()}: {value}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        print(f"✓ Analysis report saved to: {report_path}")

# 主函数
def main():
    """主函数 - 用于测试"""
    analyzer = StockAnalyzer()
    
    # 检查有哪些股票数据
    cleaned_files = list(analyzer.cleaned_dir.glob("*_cleaned.csv"))
    
    if not cleaned_files:
        print("No cleaned data found. Please run data cleaner first.")
        return
    
    print("\nAvailable stocks:")
    for i, file in enumerate(cleaned_files, 1):
        ticker = file.stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
        print(f"  {i}. {ticker}")
    
    choice = input("\nEnter stock number to analyze (or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        for file in cleaned_files:
            ticker = file.stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
            analyzer.analyze_stock(ticker)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cleaned_files):
                ticker = cleaned_files[idx].stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
                analyzer.analyze_stock(ticker)
            else:
                print("Invalid selection")
        except:
            print("Invalid input")

if __name__ == "__main__":
    main()