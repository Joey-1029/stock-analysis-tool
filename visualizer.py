# src/visualizer.py (English Version)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

class StockVisualizer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.raw_dir = self.data_dir / 'raw'      # 新增
        self.cleaned_dir = self.data_dir / 'cleaned'
        
        # Clean styling for English
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.autolayout': True,
            'figure.titlesize': 'large',
            'axes.titlesize': 'medium',
            'axes.labelsize': 'small',
            'xtick.labelsize': 'small',
            'ytick.labelsize': 'small',
        })
        
        print("=" * 60)
        print("Stock Visualizer Initialized")
        print(f"Data directory: {self.data_dir}")
        print("=" * 60)
    
    def load_stock_data(self, filename):
        """加载股票数据 - 支持子目录"""
        # 如果文件名包含路径分隔符
        if '/' in filename or '\\' in filename:
            filepath = self.data_dir / filename
        else:
            # 先尝试cleaned目录，再尝试raw目录
            cleaned_path = self.cleaned_dir / filename
            raw_path = self.raw_dir / filename
            
            if cleaned_path.exists():
                filepath = cleaned_path
            elif raw_path.exists():
                filepath = raw_path
            else:
                # 直接尝试data目录
                filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"File not found: {filepath}")
            # 列出可用文件帮助用户
            print("\nAvailable files:")
            print("Raw data:")
            for f in self.raw_dir.glob("*.csv"):
                print(f"  raw/{f.name}")
            print("Cleaned data:")
            for f in self.cleaned_dir.glob("*.csv"):
                print(f"  cleaned/{f.name}")
            return None
        
        try:
            df = pd.read_csv(filepath, index_col='date', parse_dates=True)
            return df
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def plot_price_trend(self, df, stock_name, ticker):
        """Plot price trend and technical indicators"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'{stock_name} ({ticker}) - Stock Analysis', fontsize=16, y=1.02)
        
        # 1. Price Trend
        axes[0, 0].plot(df.index, df['close'], linewidth=2, color='blue', alpha=0.7)
        axes[0, 0].set_title('Price Trend (Close)')
        axes[0, 0].set_ylabel('Price (HKD)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add moving averages
        if len(df) > 50:
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            axes[0, 0].plot(df.index, df['MA20'], 'orange', linewidth=1.5, alpha=0.8, label='MA20')
            axes[0, 0].plot(df.index, df['MA50'], 'red', linewidth=1.5, alpha=0.8, label='MA50')
            axes[0, 0].legend(loc='upper left')
        
        # 2. Volume
        axes[0, 1].bar(df.index, df['volume'], color='gray', alpha=0.6)
        axes[0, 1].set_title('Trading Volume')
        axes[0, 1].set_ylabel('Volume')
        
        # 3. Daily Returns Distribution
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {returns.mean():.3%}')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Return')
            axes[1, 0].legend()
        
        # 4. Cumulative Returns
        if 'close' in df.columns:
            cum_returns = (1 + returns).cumprod()
            axes[1, 1].plot(cum_returns.index, cum_returns, linewidth=2, color='green')
            axes[1, 1].set_title('Cumulative Returns')
            axes[1, 1].set_ylabel('Cumulative Return')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Price-Volume Relationship
        axes[2, 0].scatter(df['volume'], df['close'], alpha=0.5, s=10, color='purple')
        axes[2, 0].set_title('Price vs Volume')
        axes[2, 0].set_xlabel('Volume')
        axes[2, 0].set_ylabel('Price')
        
        # 6. Rolling Volatility (30-day)
        if 'close' in df.columns:
            rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
            axes[2, 1].plot(rolling_vol.index, rolling_vol, color='darkred')
            axes[2, 1].set_title('30-Day Rolling Annualized Volatility')
            axes[2, 1].set_ylabel('Volatility')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n{stock_name} ({ticker}) Statistics:")
        print("-" * 40)
        if 'close' in df.columns:
            print(f"Total days: {len(df)}")
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"Highest price: HKD {df['close'].max():.2f}")
            print(f"Lowest price: HKD {df['close'].min():.2f}")
            print(f"Average price: HKD {df['close'].mean():.2f}")
            print(f"Latest price: HKD {df['close'].iloc[-1]:.2f}")
            print(f"Average daily return: {returns.mean():.3%}")
            print(f"Daily volatility: {returns.std():.3%}")
            print(f"Annualized volatility: {returns.std() * np.sqrt(252):.3%}")
        
        return df
    
    def compare_two_stocks(self, file1, file2, name1, name2, ticker1, ticker2):
        """Compare two stocks"""
        df1 = self.load_stock_data(file1)
        df2 = self.load_stock_data(file2)
        
        if df1 is None or df2 is None:
            print("Error loading data")
            return
        
        # Align date indices
        common_index = df1.index.intersection(df2.index)
        df1_aligned = df1.loc[common_index]
        df2_aligned = df2.loc[common_index]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{name1} ({ticker1}) vs {name2} ({ticker2}) - Comparison', fontsize=16, y=1.02)
        
        # 1. Normalized Price Comparison
        norm_price1 = df1_aligned['close'] / df1_aligned['close'].iloc[0] * 100
        norm_price2 = df2_aligned['close'] / df2_aligned['close'].iloc[0] * 100
        
        axes[0, 0].plot(df1_aligned.index, norm_price1, label=f'{name1} ({ticker1})', linewidth=2)
        axes[0, 0].plot(df2_aligned.index, norm_price2, label=f'{name2} ({ticker2})', linewidth=2)
        axes[0, 0].set_title('Normalized Price Comparison (Base=100)')
        axes[0, 0].set_ylabel('Normalized Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Returns Correlation
        returns1 = df1_aligned['close'].pct_change().dropna()
        returns2 = df2_aligned['close'].pct_change().dropna()
        
        correlation = returns1.corr(returns2)
        axes[0, 1].scatter(returns1, returns2, alpha=0.5, s=20)
        axes[0, 1].set_xlabel(f'{name1} Daily Return')
        axes[0, 1].set_ylabel(f'{name2} Daily Return')
        axes[0, 1].set_title(f'Returns Correlation: {correlation:.3f}')
        axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        # 3. Cumulative Returns Comparison
        cum_returns1 = (1 + returns1).cumprod()
        cum_returns2 = (1 + returns2).cumprod()
        
        axes[1, 0].plot(cum_returns1.index, cum_returns1, label=f'{name1}')
        axes[1, 0].plot(cum_returns2.index, cum_returns2, label=f'{name2}')
        axes[1, 0].set_title('Cumulative Returns Comparison')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Correlation (30-day window)
        rolling_corr = returns1.rolling(window=30).corr(returns2)
        axes[1, 1].plot(rolling_corr.index, rolling_corr, linewidth=2)
        axes[1, 1].axhline(correlation, color='red', linestyle='--', 
                          label=f'Overall: {correlation:.3f}')
        axes[1, 1].set_title('30-Day Rolling Correlation')
        axes[1, 1].set_ylabel('Correlation Coefficient')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison statistics
        print("=" * 60)
        print(f"{name1} ({ticker1}) vs {name2} ({ticker2}) - Statistics")
        print("=" * 60)
        print(f"Common trading days: {len(common_index)}")
        print(f"Price correlation: {correlation:.3f}")
        print(f"\n{name1} ({ticker1}):")
        print(f"  Avg daily return: {returns1.mean():.3%}")
        print(f"  Daily volatility: {returns1.std():.3%}")
        print(f"  Total return: {(df1_aligned['close'].iloc[-1]/df1_aligned['close'].iloc[0]-1):.2%}")
        print(f"\n{name2} ({ticker2}):")
        print(f"  Avg daily return: {returns2.mean():.3%}")
        print(f"  Daily volatility: {returns2.std():.3%}")
        print(f"  Total return: {(df2_aligned['close'].iloc[-1]/df2_aligned['close'].iloc[0]-1):.2%}")
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe1 = returns1.mean() / returns1.std() * np.sqrt(252) if returns1.std() > 0 else 0
        sharpe2 = returns2.mean() / returns2.std() * np.sqrt(252) if returns2.std() > 0 else 0
        print(f"\nSharpe Ratio (annualized, rf=0):")
        print(f"  {name1}: {sharpe1:.3f}")
        print(f"  {name2}: {sharpe2:.3f}")
    
    def analyze_portfolio(self, files, names, tickers, weights=None):
        """Analyze a portfolio of stocks"""
        print("\n" + "=" * 60)
        print("PORTFOLIO ANALYSIS")
        print("=" * 60)
        
        # Load all data
        data_frames = []
        for file in files:
            df = self.load_stock_data(file)
            if df is not None and 'close' in df.columns:
                data_frames.append(df['close'])
        
        if len(data_frames) < 2:
            print("Need at least 2 stocks for portfolio analysis")
            return
        
        # Create price DataFrame
        prices_df = pd.concat(data_frames, axis=1, join='inner')
        prices_df.columns = tickers[:len(prices_df.columns)]
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Default equal weights if not provided
        if weights is None:
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Plot portfolio analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Analysis', fontsize=16, y=1.02)
        
        # 1. Portfolio Cumulative Returns
        cum_portfolio = (1 + portfolio_returns).cumprod()
        axes[0, 0].plot(cum_portfolio.index, cum_portfolio, linewidth=2, color='darkblue')
        axes[0, 0].set_title('Portfolio Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Individual Stock Cumulative Returns
        for i, ticker in enumerate(returns_df.columns):
            cum_stock = (1 + returns_df[ticker]).cumprod()
            axes[0, 1].plot(cum_stock.index, cum_stock, label=ticker, alpha=0.7)
        axes[0, 1].set_title('Individual Stock Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Correlation Matrix
        corr_matrix = returns_df.corr()
        im = axes[1, 0].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 0].set_title('Returns Correlation Matrix')
        axes[1, 0].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 0].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 0].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 0].set_yticklabels(corr_matrix.columns)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Portfolio Returns Distribution
        axes[1, 1].hist(portfolio_returns, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[1, 1].axvline(portfolio_returns.mean(), color='red', linestyle='--',
                          label=f'Mean: {portfolio_returns.mean():.3%}')
        axes[1, 1].set_title('Portfolio Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print portfolio statistics
        print(f"\nPortfolio Statistics:")
        print("-" * 40)
        print(f"Number of assets: {len(returns_df.columns)}")
        print(f"Portfolio average daily return: {portfolio_returns.mean():.3%}")
        print(f"Portfolio daily volatility: {portfolio_returns.std():.3%}")
        print(f"Portfolio annualized Sharpe (rf=0): {portfolio_returns.mean()/portfolio_returns.std()*np.sqrt(252):.3f}")
        print(f"Maximum daily loss: {portfolio_returns.min():.3%}")
        print(f"Maximum daily gain: {portfolio_returns.max():.3%}")
        
        # Correlation summary
        print(f"\nCorrelation Summary:")
        print(f"Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean():.3f}")
        print(f"Minimum correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].min():.3f}")
        print(f"Maximum correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].max():.3f}")

# Example usage
if __name__ == "__main__":
    visualizer = StockVisualizer()
    
    # Example 1: Analyze single stock
    tencent_df = visualizer.load_stock_data('HK_00700_港股_20260119.csv')
    if tencent_df is not None:
        visualizer.plot_price_trend(tencent_df, "Tencent Holdings", "00700.HK")
    
    # Example 2: Compare two stocks
    visualizer.compare_two_stocks(
        file1='HK_00700_港股_20260119.csv',
        file2='HK_09988_港股_20260119.csv',
        name1="Tencent",
        name2="Alibaba",
        ticker1="00700.HK",
        ticker2="09988.HK"
    )
    
    # Example 3: Portfolio analysis (if you have more stocks)
    # visualizer.analyze_portfolio(
    #     files=['HK_00700_港股_20250119.csv', 'HK_09988_港股_20250119.csv'],
    #     names=["Tencent", "Alibaba"],
    #     tickers=["00700", "09988"],
    #     weights=[0.6, 0.4]  # 60% Tencent, 40% Alibaba
    # )