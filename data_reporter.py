# src/data_reporter.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityReporter:
    """数据质量报告生成器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.reports_dir = self.data_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        plt.rcParams.update({
            'figure.autolayout': True,
            'figure.titlesize': 'large',
        })
    
    def analyze_data_quality(self, df, ticker):
        """分析数据质量"""
        print(f"\nAnalyzing data quality for {ticker}...")
        
        report = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_info': {},
            'missing_data': {},
            'data_issues': [],
            'statistics': {},
            'recommendations': []
        }
        
        # 基础信息
        report['basic_info'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_range': f"{df.index.min()} to {df.index.max()}" if not df.empty else 'N/A',
            'columns': list(df.columns)
        }
        
        # 缺失值分析
        missing_stats = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            if missing_count > 0:
                missing_stats[col] = {
                    'missing_count': missing_count,
                    'missing_percentage': f"{missing_pct:.2f}%"
                }
        
        report['missing_data'] = missing_stats
        
        # 数据问题检测
        issues = []
        
        # 检查重复日期
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            issues.append(f"Duplicate dates found: {duplicate_dates}")
            report['recommendations'].append("Remove duplicate dates")
        
        # 检查异常值（价格）
        if 'close' in df.columns:
            q1 = df['close'].quantile(0.25)
            q3 = df['close'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df['close'] < lower_bound) | (df['close'] > upper_bound)]
            if len(outliers) > 0:
                issues.append(f"Price outliers detected: {len(outliers)} rows")
                report['recommendations'].append("Review price outliers")
            
            # 价格统计
            report['statistics']['price'] = {
                'mean': df['close'].mean(),
                'std': df['close'].std(),
                'min': df['close'].min(),
                'max': df['close'].max(),
                'median': df['close'].median()
            }
        
        # 检查成交量异常
        if 'volume' in df.columns:
            zero_volume = (df['volume'] == 0).sum()
            if zero_volume > 0:
                issues.append(f"Zero volume days: {zero_volume}")
                report['recommendations'].append("Check zero volume days")
            
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                issues.append(f"Negative volume: {negative_volume}")
                report['recommendations'].append("Fix negative volume values")
        
        # 检查价格合理性
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high_low = (df['high'] < df['low']).sum()
            if invalid_high_low > 0:
                issues.append(f"High < Low cases: {invalid_high_low}")
                report['recommendations'].append("Fix invalid high/low prices")
            
            invalid_open = ((df['open'] > df['high']) | (df['open'] < df['low'])).sum()
            if invalid_open > 0:
                issues.append(f"Open price outside daily range: {invalid_open}")
                report['recommendations'].append("Fix open prices")
        
        report['data_issues'] = issues
        
        # 总体评估
        total_issues = len(issues) + len(missing_stats)
        if total_issues == 0:
            report['quality_rating'] = 'Excellent'
            report['recommendations'].append("Data quality is excellent. Ready for analysis.")
        elif total_issues <= 3:
            report['quality_rating'] = 'Good'
        elif total_issues <= 10:
            report['quality_rating'] = 'Fair'
        else:
            report['quality_rating'] = 'Poor'
            report['recommendations'].append("Data requires significant cleaning before analysis.")
        
        return report
    
    def generate_quality_report(self, ticker):
        """生成数据质量报告"""
        print(f"\n{'='*60}")
        print(f"DATA QUALITY REPORT: {ticker}")
        print(f"{'='*60}")
        
        # 加载数据
        data_path = self.data_dir / 'cleaned' / f"{ticker.replace('.', '.')}_cleaned.csv"
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return
        
        df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
        
        # 分析数据质量
        report = self.analyze_data_quality(df, ticker)
        
        # 打印报告
        self.print_quality_report(report)
        
        # 生成可视化
        self.create_quality_visualizations(df, ticker)
        
        # 保存报告
        self.save_quality_report(report, ticker)
        
        return report
    
    def print_quality_report(self, report):
        """打印质量报告"""
        print("\nDATA QUALITY REPORT")
        print("-" * 40)
        
        print(f"\nBasic Information:")
        for key, value in report['basic_info'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nQuality Rating: {report['quality_rating']}")
        
        print(f"\nMissing Data Analysis:")
        if report['missing_data']:
            for col, stats in report['missing_data'].items():
                print(f"  {col}: {stats['missing_count']} missing ({stats['missing_percentage']})")
        else:
            print("  No missing data found")
        
        print(f"\nData Issues Found:")
        if report['data_issues']:
            for i, issue in enumerate(report['data_issues'], 1):
                print(f"  {i}. {issue}")
        else:
            print("  No data issues found")
        
        print(f"\nStatistics:")
        for category, stats in report['statistics'].items():
            print(f"  {category.title()}:")
            for key, value in stats.items():
                print(f"    {key}: {value:.2f}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\nReport generated on: {report['analysis_date']}")
    
    def create_quality_visualizations(self, df, ticker):
        """创建数据质量可视化"""
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Data Quality Analysis - {ticker}', fontsize=16)
        
        # 1. 缺失值热图
        if len(df.columns) > 0:
            missing_data = df.isnull()
            axes[0, 0].imshow(missing_data.T, cmap='binary', aspect='auto')
            axes[0, 0].set_title('Missing Data Heatmap')
            axes[0, 0].set_xlabel('Row Index')
            axes[0, 0].set_ylabel('Columns')
            
            # 设置y轴标签
            axes[0, 0].set_yticks(range(len(df.columns)))
            axes[0, 0].set_yticklabels(df.columns)
        
        # 2. 价格分布
        if 'close' in df.columns:
            axes[0, 1].hist(df['close'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(df['close'].mean(), color='red', linestyle='--', label=f'Mean: {df["close"].mean():.2f}')
            axes[0, 1].axvline(df['close'].median(), color='green', linestyle='--', label=f'Median: {df["close"].median():.2f}')
            axes[0, 1].set_title('Closing Price Distribution')
            axes[0, 1].set_xlabel('Price')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
        
        # 3. 时间序列完整性
        if not df.empty:
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            trading_days = df.index.normalize().unique()
            missing_days = date_range.difference(trading_days)
            
            axes[1, 0].plot(df.index, range(len(df)), 'b-', linewidth=2)
            axes[1, 0].set_title('Data Completeness Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Cumulative Trading Days')
            axes[1, 0].grid(True, alpha=0.3)
            
            if len(missing_days) > 0:
                axes[1, 0].text(0.05, 0.95, f'Missing Days: {len(missing_days)}', 
                               transform=axes[1, 0].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        # 4. 相关性矩阵
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_title('Feature Correlation Matrix')
            axes[1, 1].set_xticks(range(len(numeric_cols)))
            axes[1, 1].set_xticklabels(numeric_cols, rotation=45, ha='right')
            axes[1, 1].set_yticks(range(len(numeric_cols)))
            axes[1, 1].set_yticklabels(numeric_cols)
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()
    
    def save_quality_report(self, report, ticker):
        """保存质量报告"""
        report_filename = f"{ticker.replace('.', '_')}_quality_report_{datetime.now().strftime('%Y%m%d')}.txt"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("DATA QUALITY ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Stock: {report['ticker']}\n")
            f.write(f"Analysis Date: {report['analysis_date']}\n")
            f.write(f"Quality Rating: {report['quality_rating']}\n\n")
            
            f.write("BASIC INFORMATION:\n")
            f.write("-" * 40 + "\n")
            for key, value in report['basic_info'].items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\nMISSING DATA ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            if report['missing_data']:
                for col, stats in report['missing_data'].items():
                    f.write(f"  {col}: {stats['missing_count']} missing ({stats['missing_percentage']})\n")
            else:
                f.write("  No missing data found\n")
            
            f.write("\nDATA ISSUES FOUND:\n")
            f.write("-" * 40 + "\n")
            if report['data_issues']:
                for i, issue in enumerate(report['data_issues'], 1):
                    f.write(f"  {i}. {issue}\n")
            else:
                f.write("  No data issues found\n")
            
            f.write("\nSTATISTICS:\n")
            f.write("-" * 40 + "\n")
            for category, stats in report['statistics'].items():
                f.write(f"  {category.title()}:\n")
                for key, value in stats.items():
                    f.write(f"    {key}: {value:.2f}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        print(f"\n✓ Quality report saved to: {report_path}")

# 主函数
def main():
    """主函数 - 用于测试"""
    reporter = DataQualityReporter()
    
    # 检查cleaned目录中的文件
    cleaned_dir = reporter.data_dir / 'cleaned'
    cleaned_files = list(cleaned_dir.glob("*_cleaned.csv"))
    
    if not cleaned_files:
        print("No cleaned data found. Please run data cleaner first.")
        return
    
    print("\nAvailable stocks for quality analysis:")
    for i, file in enumerate(cleaned_files, 1):
        ticker = file.stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
        print(f"  {i}. {ticker}")
    
    choice = input("\nEnter stock number to analyze (or 'all' for all): ").strip()
    
    if choice.lower() == 'all':
        for file in cleaned_files:
            ticker = file.stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
            reporter.generate_quality_report(ticker)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(cleaned_files):
                ticker = cleaned_files[idx].stem.replace('_cleaned', '').replace('RAW_', '').split('_')[0]
                reporter.generate_quality_report(ticker)
            else:
                print("Invalid selection")
        except:
            print("Invalid input")

if __name__ == "__main__":
    main()