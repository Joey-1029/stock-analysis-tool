"""
港股分析专业版 v2.2 - 核心资产对比与自动化报告版
功能：实时抓取、技术分析、个股诊断、多股收益率 PK、HTML报告整合
"""
import sys
import time
import logging
import webbrowser
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict

# 1. 路径与配置初始化
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / 'src'))

try:
    from config.config import get_config, setup_logging
    from src.analyzer import StockAnalyzer
    from src.visualizer import StockVisualizer
except ImportError as e:
    print(f"导入模块失败，请检查目录结构: {e}")
    sys.exit(1)

def generate_hk_report(analysis_results: Dict, config, logger: logging.Logger, has_comparison=False):
    """生成包含多股 PK 图和个股图表的增强型报告"""
    report_dir = Path(config.paths.reports_dir)
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = report_dir / f'hk_analysis_report_{timestamp}.html'
    
    HK_NAMES = {'00700': '腾讯控股', '09988': '阿里巴巴', '00941': '中国移动', 
                '01810': '小米集团', '03690': '美团', '09618': '京东集团'}

    # 1. HTML Header & CSS
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>港股核心资产对比报告 - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f0f2f5; color: #1a1a1a; }}
            .container {{ width: 95%; max-width: 1200px; margin: 20px auto; }}
            .header {{ background: linear-gradient(135deg, #1e3a5f 0%, #2c3e50 100%); color: white; padding: 40px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .section-title {{ border-left: 5px solid #3498db; padding-left: 15px; margin: 30px 0 20px 0; font-size: 1.5em; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 25px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; }}
            .metric-box {{ display: flex; justify-content: space-between; background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }}
            .chart-img {{ width: 100%; border-radius: 8px; margin-top: 15px; border: 1px solid #eee; }}
            .status-good {{ color: #27ae60; font-weight: bold; }}
            .status-danger {{ color: #e74c3c; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>港股核心资产自动化分析报告</h1>
                <p>数据更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 数据来源：AKShare</p>
            </div>
    """

    # 2. Add Multi-Stock Comparison Section (Top)
    if has_comparison:
        html_content += f"""
            <div class="card">
                <h3 style="margin-top:0;">🚀 资产收益率 PK (基准: 100)</h3>
                <img src="comparison_trend.png" class="chart-img" style="max-height: 500px; object-fit: contain;">
                <p style="color: #666; font-size: 0.9em; margin-top: 10px;">* 图表展示了统计周期内各资产的累计收益走势，忽略绝对价格差异。</p>
            </div>
        """

    # 3. Add Individual Analysis Grid
    html_content += '<div class="section-title">个股详细诊断</div><div class="grid">'
    
    for ticker, result in analysis_results.items():
        summary = result.get('risk_report', {}).get('risk_summary', {})
        name = HK_NAMES.get(ticker, f"港股 {ticker}")
        risk_val = summary.get('risk_rating', 'C')
        status_class = "status-good" if '低风险' in risk_val else "status-danger"
        img_path = f"plot_{ticker}.png" 
        
        html_content += f"""
            <div class="card">
                <h3 style="margin-top:0; color:#2c3e50;">{name} ({ticker})</h3>
                <div class="metric-box">
                    <div>风险评级: <span class="{status_class}">{risk_val}</span></div>
                    <div>年化收益: {summary.get('annual_return', 0):.2%}</div>
                    <div>夏普比率: {summary.get('sharpe_ratio', 0):.3f}</div>
                    <div>最大回撤: <span class="status-danger">{summary.get('max_drawdown', 0):.2%}</span></div>
                </div>
                <img src="{img_path}" class="chart-img" alt="分析图加载中...">
            </div>
        """

    html_content += "</div></div></body></html>"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return report_file

def main():
    print("🚀 港股分析系统 v2.2 启动中...")
    config = get_config()
    logger = setup_logging(config.logging)
    import akshare as ak
    
    analyzer = StockAnalyzer(config)
    visualizer = StockVisualizer()
    
    hk_pool = ['00700', '09988', '00941', '01810']
    final_results = {}
    HK_NAMES = {'00700': 'Tencent', '09988': 'Alibaba', '00941': 'China Mobile', '01810': 'Xiaomi'}

    # --- 阶段 1: 逐个资产分析 ---
    for code in hk_pool:
        try:
            print(f"正在分析 {HK_NAMES.get(code)} ({code})...")
            df = ak.stock_hk_daily(symbol=code, adjust="qfq")
            if df.empty: continue
            
            df.index = pd.to_datetime(df['date'])
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # 计算技术指标与风险
            df_analyzed = analyzer.calculate_all_indicators(df)
            returns = df_analyzed['close'].pct_change().dropna()
            risk_report = analyzer.generate_risk_report(df_analyzed['close'], returns)
            
            final_results[code] = {'data': df_analyzed, 'risk_report': risk_report}
            
            # 生成个股分析图
            img_path = Path(config.paths.reports_dir) / f"plot_{code}.png"
            visualizer.plot_price_trend(df_analyzed, HK_NAMES.get(code, "HK Stock"), code, save_path=str(img_path))
            
            time.sleep(1) 
        except Exception as e:
            logger.error(f"{code} 分析失败: {e}")

    # --- 阶段 2: 生成多股 PK 对比图 ---
    has_comp = False
    if len(final_results) > 1:
        try:
            print("正在绘制多股收益率 PK 图...")
            # 合并所有收盘价
            price_series = {HK_NAMES.get(k): v['data']['close'] for k, v in final_results.items()}
            comp_df = pd.DataFrame(price_series).ffill()
            
            # 归一化处理 (Base 100)
            normalized_df = comp_df / comp_df.iloc[0] * 100
            
            plt.figure(figsize=(12, 6), dpi=120)
            for col in normalized_df.columns:
                plt.plot(normalized_df.index, normalized_df[col], label=col, linewidth=2)
            
            plt.title("Portfolio Performance Comparison (Normalized)", fontsize=14, pad=20)
            plt.ylabel("Relative Value (Base 100)")
            plt.legend(loc='upper left', frameon=True)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            comp_path = Path(config.paths.reports_dir) / "comparison_trend.png"
            plt.savefig(comp_path, bbox_inches='tight')
            plt.close()
            has_comp = True
        except Exception as e:
            logger.error(f"PK图生成失败: {e}")

    # --- 阶段 3: 汇总报告 ---
    if final_results:
        report_path = generate_hk_report(final_results, config, logger, has_comparison=has_comp)
        print(f"\n✨ 报告生成成功！\n路径: {report_path.absolute()}")
        webbrowser.open(f"file://{report_path.absolute()}")
    else:
        print("❌ 错误：未获取到有效数据。")

if __name__ == "__main__":
    main()