"""
专业版主程序 - 港股专用分析流程
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, Any

# 添加src到Python路径
sys.path.append(str(Path(__file__).parent / 'src'))

def generate_hk_report(analysis_results: Dict, config, logger: logging.Logger):
    """
    生成港股专用HTML报告
    Args:
        analysis_results: 分析结果字典
        config: 配置对象
        logger: 日志记录器
    """
    report_dir = Path(config.paths.reports_dir)
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'hk_stock_analysis_report_{timestamp}.html'
    
    # 港股名称映射 - 支持多种格式的代码
    hk_stock_names = {
        # 原始格式
        '00700': '腾讯控股',
        '09988': '阿里巴巴',
        '00941': '中国移动',
        '01810': '小米集团',
        '03690': '美团点评',
        '02020': '安踏体育',
        '09618': '京东集团',
        '09868': '小鹏汽车',
        '09999': '网易',
        '09626': '哔哩哔哩',
        # 整数格式
        700: '腾讯控股',
        9988: '阿里巴巴',
        941: '中国移动',
        1810: '小米集团',
        3690: '美团点评',
        2020: '安踏体育',
        9618: '京东集团',
        9868: '小鹏汽车',
        9999: '网易',
        9626: '哔哩哔哩'
    }
    
    # 创建HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>港股分析报告 - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .stock-card {{ display: inline-block; width: 300px; margin: 15px; padding: 15px; border: 1px solid #eee; }}
            .metric {{ margin: 10px 0; }}
            .good {{ color: green; font-weight: bold; }}
            .warning {{ color: orange; font-weight: bold; }}
            .bad {{ color: red; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
            .summary {{ background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>港股分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>分析股票数: {len(analysis_results)}</p>
        </div>
    """
    
    # 添加港股分析结果
    for ticker, result in analysis_results.items():
        risk_report = result.get('risk_report', {})
        summary = risk_report.get('risk_summary', {})
        
        # 获取港股中文名（支持多种代码格式）
        ticker_str = str(ticker) if not isinstance(ticker, str) else ticker
        stock_name = hk_stock_names.get(ticker, hk_stock_names.get(ticker_str, ticker_str))
        
        # 如果代码是整数且小于5位，前面补0到5位
        if isinstance(ticker, int) and ticker < 10000:
            ticker_display = f"{ticker:05d}"
        else:
            ticker_display = ticker_str
        
        # 风险评级颜色
        risk_color = "good"
        risk_rating = summary.get('risk_rating', '')
        if '中高风险' in risk_rating or '高风险' in risk_rating:
            risk_color = "bad"
        elif '中等风险' in risk_rating:
            risk_color = "warning"
        
        html_content += f"""
        <div class="section">
            <h2>{stock_name} ({ticker_display})</h2>
            <div class="metric">
                <strong>风险评级:</strong> <span class="{risk_color}">{risk_rating}</span>
            </div>
            <div class="metric">
                <strong>年化收益率:</strong> {summary.get('annual_return', 0):.2%}
            </div>
            <div class="metric">
                <strong>最大回撤:</strong> {summary.get('max_drawdown', 0):.2%}
            </div>
            <div class="metric">
                <strong>夏普比率:</strong> {summary.get('sharpe_ratio', 0):.3f}
            </div>
            <div class="metric">
                <strong>胜率:</strong> {summary.get('win_rate', 0):.2%}
            </div>
            <div class="metric">
                <strong>年化波动率:</strong> {summary.get('annual_volatility', 0):.2%}
            </div>
        </div>
        """
    
    # 添加比较表格
    html_content += """
    <div class="section">
        <h2>港股表现对比</h2>
        <table>
            <thead>
                <tr>
                    <th>股票代码</th>
                    <th>股票名称</th>
                    <th>年化收益率</th>
                    <th>最大回撤</th>
                    <th>夏普比率</th>
                    <th>风险评级</th>
                    <th>建议</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for ticker, result in analysis_results.items():
        risk_report = result.get('risk_report', {})
        summary = risk_report.get('risk_summary', {})
        
        # 获取港股中文名
        ticker_str = str(ticker) if not isinstance(ticker, str) else ticker
        stock_name = hk_stock_names.get(ticker, hk_stock_names.get(ticker_str, ticker_str))
        
        # 格式化显示代码
        if isinstance(ticker, int) and ticker < 10000:
            ticker_display = f"{ticker:05d}"
        else:
            ticker_display = ticker_str
        
        # 风险评级颜色
        risk_color = "good"
        risk_rating = summary.get('risk_rating', '')
        if '中高风险' in risk_rating or '高风险' in risk_rating:
            risk_color = "bad"
        elif '中等风险' in risk_rating:
            risk_color = "warning"
        
        # 生成建议
        sharpe = summary.get('sharpe_ratio', 0)
        max_dd = abs(summary.get('max_drawdown', 0))
        
        if sharpe > 1.0 and max_dd < 0.15:
            recommendation = "强烈推荐"
            rec_class = "good"
        elif sharpe > 0.5 and max_dd < 0.25:
            recommendation = "推荐"
            rec_class = "good"
        elif sharpe > 0:
            recommendation = "谨慎持有"
            rec_class = "warning"
        else:
            recommendation = "建议回避"
            rec_class = "bad"
        
        html_content += f"""
                <tr>
                    <td><strong>{ticker_display}</strong></td>
                    <td>{stock_name}</td>
                    <td>{summary.get('annual_return', 0):.2%}</td>
                    <td>{summary.get('max_drawdown', 0):.2%}</td>
                    <td>{summary.get('sharpe_ratio', 0):.3f}</td>
                    <td class="{risk_color}">{risk_rating}</td>
                    <td class="{rec_class}">{recommendation}</td>
                </tr>
        """
    
    html_content += """
            </tbody>
        </table>
    </div>
    
    <div class="summary">
        <h3>港股市场总结</h3>
        <p>1. 港股市场特点：国际化程度高，受全球市场影响较大</p>
        <p>2. 交易时间：09:30-12:00, 13:00-16:00 (香港时间)</p>
        <p>3. 交易单位：通常以手为单位，不同股票每手股数不同</p>
        <p>4. 结算周期：T+2交收制度</p>
    </div>
    
    <div class="section">
        <h3>报告说明</h3>
        <ul>
            <li><strong>夏普比率</strong>: 衡量风险调整后收益，越高越好</li>
            <li><strong>最大回撤</strong>: 历史上最大亏损幅度，越低越好</li>
            <li><strong>风险评级</strong>: A-低风险, B-中低风险, C-中等风险, D-中高风险, E-高风险</li>
            <li>港股数据基于前复权价格计算</li>
            <li>本报告仅供参考，不构成投资建议</li>
        </ul>
    </div>
    
    <footer style="margin-top: 50px; text-align: center; color: #666;">
        <p>Generated by HK Stock Analysis Tool | {datetime.now().strftime('%Y')}</p>
    </footer>
    </body>
    </html>
    """
    
    try:
        # 保存HTML文件
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 保存JSON格式的详细数据
        json_file = report_dir / f'hk_analysis_data_{timestamp}.json'
        
        # 创建可序列化的结果（确保所有键都是字符串）
        serializable_results = {}
        
        for ticker, result in analysis_results.items():
            # 确保键是字符串
            ticker_key = str(ticker)
            
            # 获取股票名称
            ticker_for_name = ticker if isinstance(ticker, (str, int)) else str(ticker)
            stock_name = hk_stock_names.get(ticker_for_name, 
                                          hk_stock_names.get(str(ticker_for_name), str(ticker_for_name)))
            
            serializable_results[ticker_key] = {
                'name': stock_name,
                'risk_summary': result.get('risk_report', {}).get('risk_summary', {}),
                'data_points': len(result.get('data', pd.DataFrame())),
                'analysis_date': datetime.now().isoformat(),
                'original_ticker': ticker_key
            }
        
        # 保存JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("HTML报告已保存: %s", report_file)
        logger.info("JSON数据已保存: %s", json_file)
        
        return {
            'html_report': str(report_file),
            'json_data': str(json_file),
            'timestamp': timestamp
        }
        
    except Exception as e:
        logger.error("生成报告失败: %s", str(e), exc_info=True)
        return None

def setup_project_environment():
    """设置项目环境"""
    # 创建必要的目录
    directories = [
        'data/raw',
        'data/cleaned', 
        'data/analysis',
        'data/reports',
        'logs',
        'config',
        'models'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print("创建目录: %s" % dir_path)
    
    # 检查配置文件
    config_file = Path('config/settings.yaml')
    if not config_file.exists():
        print("警告: 配置文件不存在，创建默认配置...")
        try:
            from config.config import Config
            default_config = Config()
            default_config.save_to_yaml('config/settings.yaml')
            print("默认配置文件已创建: config/settings.yaml")
        except Exception as e:
            print("创建配置文件失败: %s" % e)

def check_dependencies():
    """检查依赖"""
    print("\n检查项目依赖...")
    try:
        import akshare
        import pandas
        import numpy
        import matplotlib
        import seaborn
        
        print("核心依赖检查通过")
        return True
    except ImportError as e:
        print("缺少依赖: %s" % e)
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    """主函数 - 港股专用分析流程"""
    print("=" * 60)
    print("港股分析专业版 v2.0")
    print("=" * 60)
    
    # 1. 设置项目环境
    setup_project_environment()
    
    # 2. 检查依赖
    if not check_dependencies():
        print("依赖检查失败，请先安装依赖")
        return
    
    # 3. 加载配置
    try:
        from config.config import get_config, setup_logging
        config = get_config()
        logger = setup_logging(config.logging)
    except Exception as e:
        print("配置加载失败: %s" % e)
        print("请检查 config/config.py 和 config/settings.yaml")
        return
    
    logger.info("配置和日志系统初始化完成")
    logger.info("项目根目录: %s", Path(__file__).parent)
    
    try:
        # 4. 港股数据获取
        logger.info("步骤1: 港股数据获取")
        print("\n正在获取港股数据...")
        
        # 港股代码列表（简化版）
        hk_stocks_to_download = [
            ('00700', '腾讯控股'),
            ('09988', '阿里巴巴'),
            ('00941', '中国移动'),
            ('01810', '小米集团')
        ]
        
        import akshare as ak
        
        raw_dir = Path('data/raw')
        raw_dir.mkdir(exist_ok=True)
        
        downloaded_files = []
        
        for code, name in hk_stocks_to_download:
            try:
                logger.info("下载港股: %s (%s)", code, name)
                
                # 下载港股数据
                df = ak.stock_hk_daily(symbol=code, adjust="qfq")
                
                if df is not None and not df.empty:
                    # 保存数据
                    timestamp = datetime.now().strftime('%Y%m%d')
                    filename = f"HK_{code}_{name}_{timestamp}.csv"
                    filepath = raw_dir / filename
                    
                    # 标准化数据格式
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    df['symbol'] = code
                    df['name'] = name
                    
                    df.to_csv(filepath, encoding='utf-8')
                    downloaded_files.append(filepath)
                    
                    logger.info("成功下载 %s: %s 条数据 -> %s", code, len(df), filename)
                    print(f"  ✓ {name} ({code}): {len(df)} 条数据")
                else:
                    logger.warning("下载 %s 失败或数据为空", code)
                    print(f"  ✗ {name} ({code}): 下载失败")
                
                # 延迟避免被封IP
                time.sleep(2.0)
                
            except Exception as e:
                logger.error("下载 %s 出错: %s", code, str(e))
                print(f"  ✗ {name} ({code}): 错误 - {str(e)[:50]}")
        
        logger.info("港股下载完成: 成功 %s/%s", len(downloaded_files), len(hk_stocks_to_download))
        
        # 如果下载失败，询问是否继续
        if len(downloaded_files) == 0:
            print("\n警告: 没有成功下载任何港股数据")
            user_choice = input("是否使用现有数据进行后续分析? (y/n): ").strip().lower()
            if user_choice != 'y':
                logger.info("用户选择退出")
                return
        
        # 5. 数据清洗
        logger.info("步骤2: 数据清洗")
        print("\n正在清洗数据...")
        
        # 简单的数据清洗
        cleaned_dir = Path('data/cleaned')
        cleaned_dir.mkdir(exist_ok=True)
        
        cleaned_data = {}
        
        # 使用下载的文件或已有文件
        files_to_clean = downloaded_files if downloaded_files else list(raw_dir.glob("HK_*.csv"))
        
        for filepath in files_to_clean[:10]:  # 最多处理10个文件
            try:
                df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
                symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else filepath.stem.split('_')[1]
                name = df['name'].iloc[0] if 'name' in df.columns else 'Unknown'
                
                # 简单清洗：移除空值，确保必要列存在
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        logger.warning("文件 %s 缺少列: %s", filepath.name, col)
                
                # 只保留必要列
                df_clean = df[required_cols].copy() if all(col in df.columns for col in required_cols) else df.copy()
                
                # 保存清洗后的数据
                cleaned_filename = f"{symbol}_cleaned.csv"
                cleaned_filepath = cleaned_dir / cleaned_filename
                df_clean.to_csv(cleaned_filepath, encoding='utf-8')
                
                cleaned_data[symbol] = df_clean
                logger.info("清洗完成: %s -> %s", filepath.name, cleaned_filename)
                print(f"  ✓ {name} ({symbol}): 清洗完成")
                
            except Exception as e:
                logger.error("清洗文件 %s 失败: %s", filepath.name, str(e))
                print(f"  ✗ {filepath.name}: 清洗失败")
        
        if not cleaned_data:
            logger.error("没有清洗后的数据可用")
            print("错误: 没有可用的清洗数据")
            return
        
        # 6. 技术分析
        logger.info("步骤3: 技术分析")
        print("\n正在进行技术分析...")
        
        try:
            from src.analyzer import StockAnalyzer
            analyzer = StockAnalyzer(config)
            
            analysis_results = {}
            
            for symbol, data in cleaned_data.items():
                if len(data) < 10:
                    logger.warning("%s 数据不足 (%s行)，跳过分析", symbol, len(data))
                    continue
                    
                logger.info("分析 %s...", symbol)
                print(f"  分析 {symbol}...")
                
                try:
                    # 计算技术指标
                    data_with_indicators = analyzer.calculate_all_indicators(data)
                    
                    # 计算收益率
                    if 'close' in data_with_indicators.columns:
                        prices = data_with_indicators['close']
                        returns = prices.pct_change().dropna()
                        
                        if len(returns) < 5:
                            logger.warning("%s 收益率数据不足", symbol)
                            continue
                        
                        # 生成风险报告
                        risk_report = analyzer.generate_risk_report(prices, returns)
                        analysis_results[symbol] = {
                            'data': data_with_indicators,
                            'risk_report': risk_report
                        }
                        
                        # 打印报告摘要
                        if 'risk_summary' in risk_report:
                            summary = risk_report['risk_summary']
                            print(f"    - 年化收益: {summary.get('annual_return', 0):.2%}")
                            print(f"    - 最大回撤: {summary.get('max_drawdown', 0):.2%}")
                            print(f"    - 夏普比率: {summary.get('sharpe_ratio', 0):.3f}")
                    else:
                        logger.warning("%s 缺少close列", symbol)
                        
                except Exception as e:
                    logger.error("分析 %s 失败: %s", symbol, str(e), exc_info=True)
                    continue
            
            if not analysis_results:
                logger.warning("没有生成任何分析结果")
                print("警告: 没有生成分析结果，使用示例数据")
                
                # 创建示例数据
                dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
                sample_data = pd.DataFrame({
                    'open': np.random.normal(100, 10, len(dates)),
                    'high': np.random.normal(105, 10, len(dates)),
                    'low': np.random.normal(95, 10, len(dates)),
                    'close': np.random.normal(100, 10, len(dates)),
                    'volume': np.random.randint(100000, 1000000, len(dates))
                }, index=dates)
                
                analysis_results = {
                    'DEMO': {
                        'data': sample_data,
                        'risk_report': {
                            'risk_summary': {
                                'annual_return': 0.12,
                                'max_drawdown': -0.08,
                                'sharpe_ratio': 1.05,
                                'risk_rating': 'B-中低风险'
                            }
                        }
                    }
                }
                
        except Exception as e:
            logger.error("技术分析失败: %s", str(e), exc_info=True)
            print(f"技术分析失败: {str(e)}")
            return
        
        # 7. 可视化
        logger.info("步骤4: 数据可视化")
        print("\n正在生成图表...")
        try:
            from src.visualizer import StockVisualizer
            visualizer = StockVisualizer()
            
            # 生成关键图表
            for symbol, result in analysis_results.items():
                if len(result.get('data', pd.DataFrame())) > 10:
                    logger.info("为 %s 生成图表...", symbol)
                    try:
                        # 获取股票名称
                        hk_stock_names = {
                            '00700': '腾讯控股',
                            '09988': '阿里巴巴',
                            '00941': '中国移动',
                            '01810': '小米集团'
                        }
                        stock_name = hk_stock_names.get(symbol, symbol)
                        
                        visualizer.plot_price_trend(
                            result['data'], 
                            f"港股分析 - {stock_name}", 
                            symbol
                        )
                        print(f"  ✓ {symbol}: 图表生成成功")
                    except Exception as e:
                        logger.warning("生成图表失败 %s: %s", symbol, e)
                        print(f"  ✗ {symbol}: 图表生成失败")
        except Exception as e:
            logger.warning("可视化步骤失败: %s", e)
            print(f"可视化失败: {str(e)}")
            print("继续执行报告生成...")
        
        # 8. 生成报告
        logger.info("步骤5: 生成港股分析报告")
        print("\n正在生成港股分析报告...")
        try:
            report_info = generate_hk_report(analysis_results, config, logger)
            
            if report_info:
                logger.info("港股分析流程完成！")
                print("\n" + "=" * 60)
                print("港股分析完成！")
                print("=" * 60)
                print(f"HTML报告: {report_info['html_report']}")
                print(f"JSON数据: {report_info['json_data']}")
                print(f"生成时间: {report_info['timestamp']}")
                print("=" * 60)
                
                # 在浏览器中打开报告
                try:
                    import webbrowser
                    html_path = Path(report_info['html_report'])
                    if html_path.exists():
                        webbrowser.open(f"file://{html_path.absolute()}")
                        print("已在浏览器中打开报告")
                except:
                    pass
                
                # 显示分析总结
                print("\n分析总结:")
                for symbol, result in analysis_results.items():
                    if symbol == 'DEMO':
                        continue
                        
                    summary = result.get('risk_report', {}).get('risk_summary', {})
                    stock_names = {
                        '00700': '腾讯控股',
                        '09988': '阿里巴巴',
                        '00941': '中国移动',
                        '01810': '小米集团'
                    }
                    stock_name = stock_names.get(symbol, symbol)
                    
                    print(f"  {stock_name} ({symbol}):")
                    print(f"    年化收益: {summary.get('annual_return', 0):.2%}")
                    print(f"    最大回撤: {summary.get('max_drawdown', 0):.2%}")
                    print(f"    夏普比率: {summary.get('sharpe_ratio', 0):.3f}")
                    print(f"    风险评级: {summary.get('risk_rating', '未评级')}")
                    print()
            else:
                logger.warning("报告生成失败")
                print("报告生成失败")
                
        except Exception as e:
            logger.error("报告生成失败: %s", str(e), exc_info=True)
            print(f"报告生成失败: {str(e)}")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        print("\n用户中断程序")
    except Exception as e:
        logger.error("程序运行出错: %s", str(e), exc_info=True)
        print(f"程序运行出错: {str(e)}")

if __name__ == "__main__":
    import time
    time.sleep(1)  # 延迟1秒，确保输出显示
    main()