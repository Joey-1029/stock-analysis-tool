"""
数据获取模块 - 基于 AkShare

职责：从 AkShare 拉取股票数据，写入 SQLite（通过 db_manager），
      同时保存原始 CSV 到 data/raw/ 作为备份。

支持市场：A股、港股
"""

import sys
import ssl
import time
import logging
import urllib3
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Optional

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import get_config
from src.db_manager import init_db, write_stock_data, query_latest_date

# 屏蔽 SSL 警告（AkShare 部分接口需要）
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# 重试装饰器
# ──────────────────────────────────────────

def retry(max_retries: int = 3, delay: float = 2.0):
    """网络请求失败时指数退避重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error("%s 已达最大重试次数，放弃: %s", func.__name__, e)
                        raise
                    wait = delay * (2 ** attempt)
                    logger.warning("%s 第%d次失败，%.1fs后重试: %s",
                                   func.__name__, attempt + 1, wait, e)
                    time.sleep(wait)
        return wrapper
    return decorator


# ──────────────────────────────────────────
# 数据获取器
# ──────────────────────────────────────────

class StockDataFetcher:
    """AkShare 数据获取器，拉取后直接写入 SQLite"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.raw_dir = Path(self.config.paths.raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        init_db()
        logger.info("StockDataFetcher 初始化完成，raw_dir=%s", self.raw_dir)

    # ── A股 ──────────────────────────────

    @retry(max_retries=3, delay=2.0)
    def fetch_a_stock(self, symbol: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        获取A股日线数据并写入数据库

        Args:
            symbol:     6位股票代码，如 '600519'
            start_date: 'YYYYMMDD'，默认取配置中的 start_date
            end_date:   'YYYYMMDD'，默认取今天

        Returns:
            DataFrame 或 None（失败时）
        """
        start_date = start_date or self.config.data.start_date

        # 增量更新：如果库里已有数据，从最新日期续拉
        latest = query_latest_date(symbol)
        if latest:
            # 把 YYYY-MM-DD 转成 YYYYMMDD
            incremental_start = latest.replace("-", "")
            if incremental_start >= (end_date or datetime.now().strftime("%Y%m%d")):
                logger.info("%s 数据已是最新，跳过", symbol)
                return None
            start_date = incremental_start
            logger.info("%s 增量更新，从 %s 开始", symbol, start_date)

        logger.info("拉取A股 %s，%s ~ %s", symbol, start_date, end_date or "最新")
        df = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=self.config.data.adjust_type,
        )

        if df is None or df.empty:
            logger.warning("%s 返回数据为空", symbol)
            return None

        self._save_raw(df, symbol, "A")
        write_stock_data(df, symbol, "A")
        time.sleep(self.config.data.request_delay)
        return df

    # ── 港股 ──────────────────────────────

    @retry(max_retries=3, delay=2.0)
    def fetch_hk_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        获取港股日线数据并写入数据库

        Args:
            symbol: 5位港股代码，如 '00700'

        Returns:
            DataFrame 或 None（失败时）
        """
        latest = query_latest_date(symbol)
        if latest:
            today = datetime.now().strftime("%Y-%m-%d")
            if latest >= today:
                logger.info("%s 数据已是最新，跳过", symbol)
                return None

        logger.info("拉取港股 %s", symbol)
        df = ak.stock_hk_daily(
            symbol=symbol,
            adjust=self.config.data.adjust_type,
        )

        if df is None or df.empty:
            logger.warning("%s 返回数据为空", symbol)
            return None

        self._save_raw(df, symbol, "HK")
        write_stock_data(df, symbol, "HK")
        time.sleep(self.config.data.request_delay)
        return df

    # ── 批量下载 ──────────────────────────

    def fetch_all(self) -> dict:
        """批量下载配置文件中所有股票"""
        results = {"success": [], "failed": [], "skipped": []}

        for symbol in self.config.data.hk_stocks:
            try:
                df = self.fetch_hk_stock(symbol)
                (results["skipped"] if df is None else results["success"]).append(f"HK_{symbol}")
            except Exception as e:
                logger.error("港股 %s 失败: %s", symbol, e)
                results["failed"].append(f"HK_{symbol}")

        for symbol in self.config.data.a_stocks:
            try:
                df = self.fetch_a_stock(symbol)
                (results["skipped"] if df is None else results["success"]).append(f"A_{symbol}")
            except Exception as e:
                logger.error("A股 %s 失败: %s", symbol, e)
                results["failed"].append(f"A_{symbol}")

        logger.info("批量下载完成 — 成功:%d 跳过:%d 失败:%d",
                    len(results["success"]), len(results["skipped"]), len(results["failed"]))
        return results

    # ── 内部工具 ──────────────────────────

    def _save_raw(self, df: pd.DataFrame, symbol: str, market: str):
        """保存原始数据到 CSV（作为备份，不参与后续分析流程）"""
        filename = f"{market}_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = self.raw_dir / filename
        df.to_csv(filepath, index=True)
        logger.debug("原始数据已保存: %s", filepath)