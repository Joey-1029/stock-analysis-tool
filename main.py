# main.py
import sys
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

def check_data_directories():
    """检查并创建数据目录结构"""
    base_dir = Path(__file__).parent
    
    # 只创建必要的目录，不创建config下的目录
    directories = [
        base_dir / 'data',
        base_dir / 'data' / 'raw',
        base_dir / 'data' / 'cleaned',
        base_dir / 'data' / 'analysis',
        base_dir / 'data' / 'reports'
    ]
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    return directories

def get_data_status():
    """获取数据文件状态"""
    data_dir = Path(__file__).parent / 'data'
    raw_dir = data_dir / 'raw'
    cleaned_dir = data_dir / 'cleaned'
    analysis_dir = data_dir / 'analysis'
    
    raw_files = list(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    cleaned_files = list(cleaned_dir.glob("*.csv")) if cleaned_dir.exists() else []
    analysis_files = list(analysis_dir.glob("*.csv")) if analysis_dir.exists() else []
    
    return {
        'raw_count': len(raw_files),
        'cleaned_count': len(cleaned_files),
        'analysis_count': len(analysis_files),
        'raw_files': [f.name for f in raw_files],
        'cleaned_files': [f.name for f in cleaned_files],
        'analysis_files': [f.name for f in analysis_files]
    }

def run_data_fetcher():
    """运行数据获取模块"""
    print("\n" + "="*60)
    print("DATA FETCHER MODULE")
    print("="*60)
    print("Fetching stock data using AKShare...")
    
    try:
        # 直接运行Python文件
        result = subprocess.run(
            [sys.executable, "src/data_fetcher_akshare.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0:
            print("✓ Data fetcher completed successfully")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print("✗ Data fetcher encountered errors")
            if result.stderr:
                print("\nErrors:")
                print(result.stderr)
            
    except Exception as e:
        print(f"Error running data fetcher: {e}")
        print("\nYou can also run directly: python src/data_fetcher_akshare.py")

def run_data_cleaner():
    """运行数据清洗模块"""
    print("\n" + "="*60)
    print("DATA CLEANER MODULE")
    print("="*60)
    
    try:
        from src.data_cleaner import StockDataCleaner
        
        # 创建清理器实例
        cleaner = StockDataCleaner()
        
        # 检查原始数据
        data_status = get_data_status()
        if data_status['raw_count'] == 0:
            print("No raw data files found in data/raw/")
            print("Please run 'Fetch data' first to download stock data.")
            return
        
        print(f"Found {data_status['raw_count']} raw data files")
        
        # 显示菜单
        print("\nAvailable options:")
        print("1. Clean all raw data files")
        print("2. Clean specific stock")
        print("3. Create merged dataset only")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nCleaning all raw data files...")
            cleaned_data = cleaner.clean_all_stocks()
            
            if cleaned_data:
                print(f"\n✓ Successfully cleaned {len(cleaned_data)} stocks:")
                for ticker in cleaned_data.keys():
                    print(f"  - {ticker}")
            else:
                print("✗ No data was cleaned")
                
        elif choice == '2':
            print("\nAvailable raw data files:")
            for i, filename in enumerate(data_status['raw_files'], 1):
                print(f"  {i}. {filename}")
            
            try:
                file_choice = int(input("\nSelect file number: ").strip())
                if 1 <= file_choice <= len(data_status['raw_files']):
                    filename = data_status['raw_files'][file_choice - 1]
                    
                    # 确定股票代码和名称
                    if '00700' in filename:
                        ticker = '00700.HK'
                        name = 'Tencent'
                    elif '09988' in filename:
                        ticker = '09988.HK'
                        name = 'Alibaba'
                    else:
                        # 尝试从文件名提取
                        ticker = filename.replace('.csv', '')
                        name = 'Unknown Stock'
                    
                    print(f"\nCleaning {name} ({ticker})...")
                    raw_df = cleaner.load_raw_data(filename)
                    
                    if raw_df is not None:
                        cleaned_df = cleaner.clean_data(raw_df, ticker, name)
                        if cleaned_df is not None:
                            print(f"✓ Successfully cleaned {len(cleaned_df)} rows")
                    else:
                        print("✗ Failed to load raw data")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            print("\nCreating merged dataset...")
            # 先尝试清理所有数据，然后创建合并数据集
            cleaned_data = cleaner.clean_all_stocks()
            if cleaned_data:
                merged_df = cleaner.create_merged_dataset(cleaned_data)
                if merged_df is not None:
                    print(f"✓ Created merged dataset with {len(merged_df.columns)} stocks")
            else:
                print("✗ No data available to merge")
        else:
            print("Invalid choice. Cleaning all data...")
            cleaner.clean_all_stocks()
            
    except ImportError:
        print("Error: data_cleaner.py not found in src/ directory")
    except Exception as e:
        print(f"Error running data cleaner: {e}")
        import traceback
        traceback.print_exc()

def run_visualizer():
    """运行可视化模块"""
    print("\n" + "="*60)
    print("VISUALIZER MODULE")
    print("="*60)
    
    try:
        from src.visualizer import StockVisualizer
        
        # 创建可视化器实例
        visualizer = StockVisualizer()
        
        # 检查数据
        data_status = get_data_status()
        
        if data_status['cleaned_count'] == 0 and data_status['raw_count'] == 0:
            print("No data files found.")
            print("Please run 'Fetch data' and 'Clean data' first.")
            return
        
        # 显示菜单
        print("\nAvailable visualization options:")
        print("1. Analyze single stock")
        print("2. Compare two stocks")
        print("3. Portfolio analysis")
        print("4. Back to main menu")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # 单只股票分析
            if data_status['cleaned_count'] > 0:
                print("\nUsing cleaned data (recommended)...")
                files = data_status['cleaned_files']
                prefix = 'cleaned/'
            else:
                print("\nUsing raw data (cleaned data not available)...")
                files = data_status['raw_files']
                prefix = 'raw/'
            
            if len(files) == 0:
                print("No data files available.")
                return
            
            print(f"\nAvailable stocks:")
            for i, filename in enumerate(files, 1):
                display_name = filename.replace('_cleaned.csv', '').replace('.csv', '')
                print(f"  {i}. {display_name}")
            
            try:
                stock_choice = int(input("\nSelect stock number: ").strip())
                if 1 <= stock_choice <= len(files):
                    filename = files[stock_choice - 1]
                    filepath = f"{prefix}{filename}"
                    
                    # 确定股票名称
                    display_name = filename.replace('_cleaned.csv', '').replace('.csv', '')
                    
                    if '00700' in filename:
                        name = "Tencent Holdings"
                        ticker = "00700.HK"
                    elif '09988' in filename:
                        name = "Alibaba Group"
                        ticker = "09988.HK"
                    else:
                        name = display_name
                        ticker = display_name
                    
                    print(f"\nLoading {name} ({ticker})...")
                    df = visualizer.load_stock_data(filepath)
                    
                    if df is not None:
                        print(f"✓ Loaded {len(df)} rows of data")
                        visualizer.plot_price_trend(df, name, ticker)
                    else:
                        print("✗ Failed to load data")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            # 比较两只股票
            if data_status['cleaned_count'] >= 2:
                print("\nUsing cleaned data...")
                files = data_status['cleaned_files']
                prefix = 'cleaned/'
            elif data_status['raw_count'] >= 2:
                print("\nUsing raw data (cleaned data insufficient)...")
                files = data_status['raw_files']
                prefix = 'raw/'
            else:
                print("Need at least 2 data files for comparison.")
                return
            
            print(f"\nSelect first stock:")
            for i, filename in enumerate(files, 1):
                display_name = filename.replace('_cleaned.csv', '').replace('.csv', '')
                print(f"  {i}. {display_name}")
            
            try:
                stock1_choice = int(input("\nSelect first stock number: ").strip())
                if not (1 <= stock1_choice <= len(files)):
                    print("Invalid selection")
                    return
                
                print(f"\nSelect second stock:")
                for i, filename in enumerate(files, 1):
                    display_name = filename.replace('_cleaned.csv', '').replace('.csv', '')
                    print(f"  {i}. {display_name}")
                
                stock2_choice = int(input("\nSelect second stock number: ").strip())
                if not (1 <= stock2_choice <= len(files)):
                    print("Invalid selection")
                    return
                
                # 获取文件信息
                file1 = files[stock1_choice - 1]
                file2 = files[stock2_choice - 1]
                
                # 确定股票名称
                name1 = file1.replace('_cleaned.csv', '').replace('.csv', '')
                name2 = file2.replace('_cleaned.csv', '').replace('.csv', '')
                
                if '00700' in name1:
                    display_name1 = "Tencent"
                    ticker1 = "00700.HK"
                elif '09988' in name1:
                    display_name1 = "Alibaba"
                    ticker1 = "09988.HK"
                else:
                    display_name1 = name1
                    ticker1 = name1
                
                if '00700' in name2:
                    display_name2 = "Tencent"
                    ticker2 = "00700.HK"
                elif '09988' in name2:
                    display_name2 = "Alibaba"
                    ticker2 = "09988.HK"
                else:
                    display_name2 = name2
                    ticker2 = name2
                
                print(f"\nComparing {display_name1} vs {display_name2}...")
                visualizer.compare_two_stocks(
                    file1=f"{prefix}{file1}",
                    file2=f"{prefix}{file2}",
                    name1=display_name1,
                    name2=display_name2,
                    ticker1=ticker1,
                    ticker2=ticker2
                )
                
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            # 投资组合分析
            if data_status['cleaned_count'] < 2:
                print("Need at least 2 cleaned stocks for portfolio analysis.")
                print("Please run 'Clean data' first.")
                return
            
            print("\nAvailable cleaned stocks:")
            files = data_status['cleaned_files']
            for i, filename in enumerate(files, 1):
                ticker = filename.replace('_cleaned.csv', '')
                print(f"  {i}. {ticker}")
            
            try:
                choices = input("\nEnter stock numbers (comma-separated, e.g., 1,2): ").strip()
                selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
                
                # 验证选择
                valid_indices = [idx for idx in selected_indices if 0 <= idx < len(files)]
                if len(valid_indices) < 2:
                    print("Need at least 2 valid stocks selected.")
                    return
                
                selected_files = [files[idx] for idx in valid_indices]
                tickers = [f.replace('_cleaned.csv', '') for f in selected_files]
                
                # 映射名称
                name_map = {
                    '00700.HK': 'Tencent Holdings',
                    '09988.HK': 'Alibaba Group'
                }
                names = [name_map.get(t, t) for t in tickers]
                
                print(f"\nPortfolio with {len(selected_files)} stocks:")
                for name, ticker in zip(names, tickers):
                    print(f"  - {name} ({ticker})")
                
                # 询问权重
                weight_choice = input("\nUse equal weights? (y/n): ").lower().strip()
                
                if weight_choice == 'y' or weight_choice == '':
                    weights = None
                    print("Using equal weights.")
                else:
                    weights = []
                    total = 0
                    for name, ticker in zip(names, tickers):
                        while True:
                            try:
                                weight = float(input(f"  Weight for {name} ({ticker}) (0-1): "))
                                if 0 <= weight <= 1:
                                    weights.append(weight)
                                    total += weight
                                    break
                                else:
                                    print("Weight must be between 0 and 1.")
                            except ValueError:
                                print("Invalid input. Please enter a number.")
                    
                    if abs(total - 1.0) > 0.001:
                        print(f"Weights sum to {total:.3f}, not 1. Using equal weights instead.")
                        weights = None
                
                # 运行投资组合分析
                file_paths = [f"cleaned/{f}" for f in selected_files]
                visualizer.analyze_portfolio(
                    files=file_paths,
                    names=names,
                    tickers=tickers,
                    weights=weights
                )
                
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '4':
            return
        else:
            print("Invalid choice")
            
    except ImportError:
        print("Error: visualizer.py not found in src/ directory")
    except Exception as e:
        print(f"Error running visualizer: {e}")
        import traceback
        traceback.print_exc()

def run_analyzer():
    """运行分析模块"""
    print("\n" + "="*60)
    print("TECHNICAL ANALYZER MODULE")
    print("="*60)
    
    try:
        from src.analyzer import StockAnalyzer
        
        # 创建分析器实例
        analyzer = StockAnalyzer()
        
        # 检查数据
        data_status = get_data_status()
        if data_status['cleaned_count'] == 0:
            print("No cleaned data found.")
            print("Please run 'Clean data' first.")
            return
        
        print(f"Found {data_status['cleaned_count']} cleaned data files")
        
        # 显示菜单
        print("\nAvailable analysis options:")
        print("1. Complete technical analysis (full report + charts)")
        print("2. Quick analysis (summary only)")
        print("3. Analyze all stocks")
        print("4. Compare stock returns")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # 完整技术分析
            print("\nAvailable stocks:")
            files = data_status['cleaned_files']
            for i, filename in enumerate(files, 1):
                ticker = filename.replace('_cleaned.csv', '')
                print(f"  {i}. {ticker}")
            
            try:
                stock_choice = int(input("\nSelect stock number: ").strip())
                if 1 <= stock_choice <= len(files):
                    filename = files[stock_choice - 1]
                    ticker = filename.replace('_cleaned.csv', '')
                    
                    print(f"\nRunning complete technical analysis for {ticker}...")
                    analyzer.analyze_stock(ticker)
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            # 快速分析
            print("\nAvailable stocks:")
            files = data_status['cleaned_files']
            for i, filename in enumerate(files, 1):
                ticker = filename.replace('_cleaned.csv', '')
                print(f"  {i}. {ticker}")
            
            try:
                stock_choice = int(input("\nSelect stock number: ").strip())
                if 1 <= stock_choice <= len(files):
                    filename = files[stock_choice - 1]
                    ticker = filename.replace('_cleaned.csv', '')
                    
                    print(f"\nRunning quick analysis for {ticker}...")
                    # 加载数据
                    df = analyzer.load_cleaned_data(ticker)
                    
                    if df is not None:
                        # 计算指标
                        df_with_indicators = analyzer.calculate_all_indicators(df.copy())
                        # 生成报告
                        report = analyzer.generate_analysis_report(df_with_indicators, ticker)
                        # 打印报告
                        analyzer.print_analysis_report(report)
                    else:
                        print(f"✗ Could not load data for {ticker}")
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '3':
            # 分析所有股票
            print("\nAnalyzing all stocks...")
            files = data_status['cleaned_files']
            
            for filename in files:
                ticker = filename.replace('_cleaned.csv', '')
                print(f"\n{'='*40}")
                print(f"Analyzing {ticker}...")
                print(f"{'='*40}")
                
                try:
                    analyzer.analyze_stock(ticker)
                except Exception as e:
                    print(f"✗ Error analyzing {ticker}: {e}")
            
            print("\n✓ Analysis of all stocks completed")
            
        elif choice == '4':
            # 比较股票收益
            if data_status['cleaned_count'] < 2:
                print("Need at least 2 cleaned stocks for comparison.")
                return
            
            print("\nSelect stocks to compare (comma-separated numbers):")
            files = data_status['cleaned_files']
            for i, filename in enumerate(files, 1):
                ticker = filename.replace('_cleaned.csv', '')
                print(f"  {i}. {ticker}")
            
            try:
                choices = input("\nEnter stock numbers (e.g., 1,2,3): ").strip()
                selected_indices = [int(x.strip()) - 1 for x in choices.split(',')]
                
                # 加载选中的股票数据
                selected_stocks = []
                for idx in selected_indices:
                    if 0 <= idx < len(files):
                        filename = files[idx]
                        ticker = filename.replace('_cleaned.csv', '')
                        df = analyzer.load_cleaned_data(ticker)
                        
                        if df is not None and 'close' in df.columns:
                            selected_stocks.append({
                                'ticker': ticker,
                                'data': df
                            })
                
                if len(selected_stocks) >= 2:
                    # 比较分析
                    print(f"\n{'='*60}")
                    print(f"COMPARING {len(selected_stocks)} STOCKS")
                    print(f"{'='*60}")
                    
                    print(f"\n{'Ticker':<15} {'Start Price':>12} {'End Price':>12} {'Return %':>12} {'Volatility':>12}")
                    print("-" * 65)
                    
                    for stock in selected_stocks:
                        ticker = stock['ticker']
                        df = stock['data']
                        
                        start_price = df['close'].iloc[0]
                        end_price = df['close'].iloc[-1]
                        total_return = ((end_price / start_price) - 1) * 100
                        volatility = df['close'].std() if len(df) > 1 else 0
                        
                        print(f"{ticker:<15} {start_price:>12.2f} {end_price:>12.2f} {total_return:>12.2f}% {volatility:>12.2f}")
                    
                    print("\n✓ Comparison completed")
                else:
                    print("Need at least 2 valid stocks for comparison.")
                    
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Invalid choice")
            
    except ImportError:
        print("Error: analyzer.py not found in src/ directory")
    except Exception as e:
        print(f"Error running analyzer: {e}")
        import traceback
        traceback.print_exc()

def run_reporter():
    """运行报告生成模块"""
    print("\n" + "="*60)
    print("DATA QUALITY REPORTER MODULE")
    print("="*60)
    
    try:
        from src.data_reporter import DataQualityReporter
        
        # 创建报告器实例
        reporter = DataQualityReporter()
        
        # 检查数据
        data_status = get_data_status()
        if data_status['cleaned_count'] == 0:
            print("No cleaned data found.")
            print("Please run 'Clean data' first.")
            return
        
        print(f"Found {data_status['cleaned_count']} cleaned data files")
        
        # 显示菜单
        print("\nAvailable reporting options:")
        print("1. Generate quality report for single stock")
        print("2. Generate quality reports for all stocks")
        print("3. Generate summary quality report")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            # 单只股票质量报告
            print("\nAvailable stocks:")
            files = data_status['cleaned_files']
            for i, filename in enumerate(files, 1):
                ticker = filename.replace('_cleaned.csv', '')
                print(f"  {i}. {ticker}")
            
            try:
                stock_choice = int(input("\nSelect stock number: ").strip())
                if 1 <= stock_choice <= len(files):
                    filename = files[stock_choice - 1]
                    ticker = filename.replace('_cleaned.csv', '')
                    
                    print(f"\nGenerating quality report for {ticker}...")
                    reporter.generate_quality_report(ticker)
                else:
                    print("Invalid selection")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            # 所有股票质量报告
            print("\nGenerating quality reports for all stocks...")
            files = data_status['cleaned_files']
            
            for filename in files:
                ticker = filename.replace('_cleaned.csv', '')
                print(f"\n{'='*40}")
                print(f"Generating report for {ticker}...")
                print(f"{'='*40}")
                
                try:
                    reporter.generate_quality_report(ticker)
                except Exception as e:
                    print(f"✗ Error generating report for {ticker}: {e}")
            
            print("\n✓ Quality reports for all stocks completed")
            
        elif choice == '3':
            # 总结质量报告
            print("\nGenerating summary quality report...")
            files = data_status['cleaned_files']
            
            if not files:
                print("No cleaned data files found.")
                return
            
            all_reports = []
            for filename in files:
                ticker = filename.replace('_cleaned.csv', '')
                try:
                    # 加载数据
                    data_path = Path(__file__).parent / 'data' / 'cleaned' / filename
                    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
                    
                    # 分析数据质量
                    report = reporter.analyze_data_quality(df, ticker)
                    all_reports.append(report)
                    print(f"  ✓ Analyzed {ticker}")
                    
                except Exception as e:
                    print(f"  ✗ Error analyzing {ticker}: {e}")
            
            # 显示总结
            if all_reports:
                print(f"\n{'='*60}")
                print("DATA QUALITY SUMMARY REPORT")
                print(f"{'='*60}")
                
                print(f"\n{'Ticker':<15} {'Rows':>8} {'Missing':>10} {'Issues':>10} {'Rating':>12}")
                print("-" * 60)
                
                for report in all_reports:
                    ticker = report['ticker']
                    rows = report['basic_info']['total_rows']
                    missing = len(report['missing_data'])
                    issues = len(report['data_issues'])
                    rating = report['quality_rating']
                    
                    print(f"{ticker:<15} {rows:>8} {missing:>10} {issues:>10} {rating:>12}")
                
                # 总体统计
                total_stocks = len(all_reports)
                good_count = sum(1 for r in all_reports if r['quality_rating'] in ['Excellent', 'Good'])
                avg_quality = (good_count / total_stocks) * 100 if total_stocks > 0 else 0
                
                print("\nSUMMARY STATISTICS:")
                print(f"  Total stocks analyzed: {total_stocks}")
                print(f"  Good or Excellent quality: {good_count} ({avg_quality:.1f}%)")
                print(f"  Total data issues found: {sum(len(r['data_issues']) for r in all_reports)}")
                print(f"  Total missing values: {sum(len(r['missing_data']) for r in all_reports)}")
            else:
                print("✗ No reports were generated")
                
        else:
            print("Invalid choice")
            
    except ImportError:
        print("Error: data_reporter.py not found in src/ directory")
    except Exception as e:
        print(f"Error running reporter: {e}")
        import traceback
        traceback.print_exc()

def run_full_pipeline():
    """运行完整流程"""
    print("\n" + "="*60)
    print("FULL PIPELINE EXECUTION")
    print("="*60)
    print("This will run all modules in sequence:")
    print("1. Fetch data")
    print("2. Clean data")
    print("3. Generate quality reports")
    print("4. Technical analysis")
    print("5. Data visualization")
    
    confirm = input("\nDo you want to continue? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Pipeline cancelled.")
        return
    
    modules = [
        ("1. DATA FETCHING", run_data_fetcher),
        ("2. DATA CLEANING", run_data_cleaner),
        ("3. QUALITY REPORTING", run_reporter),
        ("4. TECHNICAL ANALYSIS", run_analyzer),
        ("5. DATA VISUALIZATION", run_visualizer)
    ]
    
    for module_name, module_function in modules:
        print(f"\n{'='*40}")
        print(module_name)
        print(f"{'='*40}")
        
        try:
            module_function()
        except Exception as e:
            print(f"✗ Error in {module_name}: {e}")
        
        # 询问是否继续
        if module_name != modules[-1][0]:
            cont = input("\nContinue to next module? (y/n): ").lower().strip()
            if cont != 'y':
                print("Pipeline stopped by user.")
                break
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("="*60)

def show_data_status():
    """显示数据状态"""
    print("\n" + "="*60)
    print("DATA STATUS")
    print("="*60)
    
    data_status = get_data_status()
    
    print(f"\nRaw Data Files: {data_status['raw_count']}")
    if data_status['raw_files']:
        for filename in data_status['raw_files']:
            print(f"  - {filename}")
    else:
        print("  No raw data files")
    
    print(f"\nCleaned Data Files: {data_status['cleaned_count']}")
    if data_status['cleaned_files']:
        for filename in data_status['cleaned_files']:
            print(f"  - {filename}")
    else:
        print("  No cleaned data files")
    
    print(f"\nAnalysis Files: {data_status['analysis_count']}")
    if data_status['analysis_files']:
        for filename in data_status['analysis_files']:
            print(f"  - {filename}")
    else:
        print("  No analysis files")
    
    print(f"\nDirectories:")
    data_dir = Path(__file__).parent / 'data'
    for subdir in ['raw', 'cleaned', 'analysis', 'reports']:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.glob("*")))
            print(f"  - data/{subdir}: {file_count} files")
        else:
            print(f"  - data/{subdir}: does not exist")

def main():
    """主菜单"""
    # 初始化目录结构
    print("Initializing Stock Analysis Tool...")
    check_data_directories()
    
    while True:
        # 获取当前数据状态
        data_status = get_data_status()
        
        print("\n" + "=" * 60)
        print("STOCK ANALYSIS TOOL - MAIN MENU")
        print("=" * 60)
        print(f"Data Status: {data_status['raw_count']} raw, {data_status['cleaned_count']} cleaned, {data_status['analysis_count']} analysis")
        
        print("\nAvailable modules:")
        print("1. 📥 Fetch data (download new stock data)")
        print("2. 🧹 Clean data (process raw data)")
        print("3. 📊 Analyze data (technical analysis)")
        print("4. 📈 Visualize data (charts and graphs)")
        print("5. 📋 Generate reports (data quality)")
        print("6. 🔄 Run full pipeline (all modules)")
        print("7. 📊 Show data status")
        print("8. ❌ Exit")
        
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                run_data_fetcher()
            elif choice == '2':
                run_data_cleaner()
            elif choice == '3':
                run_analyzer()
            elif choice == '4':
                run_visualizer()
            elif choice == '5':
                run_reporter()
            elif choice == '6':
                run_full_pipeline()
            elif choice == '7':
                show_data_status()
            elif choice == '8':
                print("\nThank you for using Stock Analysis Tool!")
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-8.")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()