"""
数据清洗模块

职责：从 SQLite 读取原始数据，执行清洗（缺失值、异常值处理），
      将清洗后的数据保存到 data/cleaned/，供分析模块使用。
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db_manager import query_stock, list_symbols

logger = logging.getLogger(__name__)


class StockDataCleaner:
    """股票数据清洗器"""

    def __init__(self, cleaned_dir: str = "data/cleaned"):
        self.cleaned_dir = Path(cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, symbol: str,
              start_date: str = None,
              end_date: str = None) -> pd.DataFrame:
        """
        清洗单只股票数据

        流程：
          1. 从数据库读取
          2. 处理缺失值（前向填充）
          3. 过滤价格为零的异常行
          4. 添加衍生指标（日收益率、对数收益率）
          5. 保存 cleaned CSV

        Returns:
            清洗后的 DataFrame（以 date 为索引）
        """
        df = query_stock(symbol, start_date, end_date)
        if df.empty:
            logger.warning("%s 数据库中无数据，请先运行数据获取", symbol)
            return df

        raw_len = len(df)

        # 1. 过滤收盘价为零或负的行
        df = df[df["close"] > 0]

        # 2. 缺失值：前向填充，再后向填充边界
        df = df.ffill().bfill()

        # 3. 衍生列
        df["daily_return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        cleaned_len = len(df)
        dropped = raw_len - cleaned_len
        if dropped:
            logger.info("%s 清洗完成：过滤 %d 行异常数据", symbol, dropped)
        else:
            logger.info("%s 清洗完成：%d 行，无异常", symbol, cleaned_len)

        # 4. 保存
        out_path = self.cleaned_dir / f"{symbol}_cleaned.csv"
        df.to_csv(out_path)
        logger.info("已保存: %s", out_path)

        return df

    def clean_all(self, start_date: str = None, end_date: str = None) -> dict:
        """清洗数据库中所有股票"""
        symbols = [s["symbol"] for s in list_symbols()]
        if not symbols:
            logger.warning("数据库为空，请先运行数据获取")
            return {}

        results = {}
        for symbol in symbols:
            try:
                df = self.clean(symbol, start_date, end_date)
                results[symbol] = df
            except Exception as e:
                logger.error("%s 清洗失败: %s", symbol, e)

        logger.info("批量清洗完成，共 %d 只股票", len(results))
        return results