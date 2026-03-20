"""
股票分析工具 - 主入口

用法：
  python main.py                          # 分析配置文件中所有股票
  python main.py --symbol 600519          # 分析单只A股
  python main.py --symbol 00700 --market HK   # 分析单只港股
  python main.py --fetch-only             # 只拉数据，不分析
  python main.py --list                   # 查看数据库中已有股票

流程：
  取数（AkShare → SQLite）→ 清洗 → 技术指标 → 风险报告 → 可视化 → HTML报告
"""

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config.config import get_config, setup_logging
from src.db_manager import list_symbols, query_stock
from src.data_fetcher_akshare import StockDataFetcher
from src.data_cleaner import StockDataCleaner
from src.analyzer import StockAnalyzer
from src.visualizer import StockVisualizer
from src.data_reporter import StockReporter


def parse_args():
    parser = argparse.ArgumentParser(description="股票数据分析工具")
    parser.add_argument("--symbol", type=str, help="股票代码，如 600519 或 00700")
    parser.add_argument("--market", type=str, choices=["A", "HK"],
                        default="HK", help="市场类型（默认 HK）")
    parser.add_argument("--start", type=str, help="开始日期，格式 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="结束日期，格式 YYYY-MM-DD")
    parser.add_argument("--fetch-only", action="store_true", help="只拉取数据，不做分析")
    parser.add_argument("--no-fetch", action="store_true", help="跳过数据拉取，直接分析库中数据")
    parser.add_argument("--list", action="store_true", help="列出数据库中所有股票")
    parser.add_argument("--open-report", action="store_true",
                        default=True, help="分析完成后自动打开 HTML 报告（默认开启）")
    return parser.parse_args()


def fetch_step(config, symbol: str = None, market: str = None):
    """数据获取步骤"""
    fetcher = StockDataFetcher(config)
    if symbol:
        if market == "A":
            fetcher.fetch_a_stock(symbol)
        else:
            fetcher.fetch_hk_stock(symbol)
    else:
        fetcher.fetch_all()


def analyze_symbol(symbol: str,
                   start_date: str,
                   end_date: str,
                   config) -> str:
    """
    对单只股票执行完整分析流程

    Returns:
        生成的 HTML 报告路径，失败返回空字符串
    """
    # 1. 清洗
    cleaner = StockDataCleaner(config.paths.cleaned_dir)
    df = cleaner.clean(symbol, start_date, end_date)
    if df.empty:
        logging.warning("%s 清洗后数据为空，跳过分析", symbol)
        return ""

    # 2. 技术指标
    analyzer = StockAnalyzer(config)
    df_ind = analyzer.calculate_all_indicators(df)

    # 3. 风险报告
    prices = df_ind["close"]
    returns = prices.pct_change().dropna()
    if len(returns) < 20:
        logging.warning("%s 数据量不足（%d 行），跳过风险分析", symbol, len(returns))
        return ""

    risk_report = analyzer.generate_risk_report(prices, returns)
    analyzer.print_risk_report(risk_report, symbol)

    # 4. 可视化
    visualizer = StockVisualizer(config.paths.reports_dir)
    chart_path = visualizer.plot_analysis(df_ind, symbol)

    # 5. HTML 报告
    reporter = StockReporter(config.paths.reports_dir)
    report_path = reporter.generate(symbol, risk_report, chart_path)

    return report_path


def main():
    args = parse_args()
    config = get_config()
    logger = setup_logging(config.logging)

    # ── 列出已有股票 ──────────────────────
    if args.list:
        symbols = list_symbols()
        if not symbols:
            print("数据库为空，请先运行数据获取。")
        else:
            print(f"\n数据库中共 {len(symbols)} 只股票：")
            print(f"{'代码':<12}{'市场':<6}{'起始日':<12}{'最新日':<12}{'交易天数':>8}")
            print("-" * 52)
            for s in symbols:
                print(f"{s['symbol']:<12}{s['market']:<6}"
                      f"{s['start_date']:<12}{s['end_date']:<12}{s['trading_days']:>8}")
        return

    # ── 确定分析目标 ──────────────────────
    if args.symbol:
        target_symbols = [(args.symbol, args.market)]
    else:
        # 来自配置文件
        target_symbols = (
            [(s, "HK") for s in config.data.hk_stocks] +
            [(s, "A") for s in config.data.a_stocks]
        )

    # ── 数据获取 ──────────────────────────
    if not args.no_fetch:
        logger.info("=== 数据获取 ===")
        fetcher = StockDataFetcher(config)
        for symbol, market in target_symbols:
            try:
                if market == "A":
                    fetcher.fetch_a_stock(symbol)
                else:
                    fetcher.fetch_hk_stock(symbol)
            except Exception as e:
                logger.error("%s 获取失败: %s", symbol, e)

    if args.fetch_only:
        logger.info("--fetch-only 模式，数据获取完成，退出。")
        return

    # ── 分析 ──────────────────────────────
    logger.info("=== 开始分析 ===")
    report_paths = []

    for symbol, market in target_symbols:
        logger.info("分析 %s [%s]", symbol, market)
        try:
            report_path = analyze_symbol(
                symbol,
                start_date=args.start,
                end_date=args.end,
                config=config,
            )
            if report_path:
                report_paths.append(report_path)
                logger.info("✓ %s 报告: %s", symbol, report_path)
        except Exception as e:
            logger.error("✗ %s 分析失败: %s", symbol, e, exc_info=True)

    # ── 汇总 ──────────────────────────────
    print(f"\n{'='*50}")
    print(f"分析完成：{len(report_paths)}/{len(target_symbols)} 只股票")
    for p in report_paths:
        print(f"  报告: {p}")

    # 打开最后一份报告
    if args.open_report and report_paths:
        webbrowser.open(f"file://{Path(report_paths[-1]).absolute()}")


if __name__ == "__main__":
    main()