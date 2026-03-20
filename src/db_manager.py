"""
数据库管理模块 - SQLite 取数层

职责：
  1. 将 AkShare 拉取的原始数据写入 SQLite
  2. 提供标准化查询接口供分析模块使用（替代直接读 CSV）
  3. 支持增量更新，避免重复写入
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

DB_FILE = Path(__file__).parent.parent / "data" / "stocks.db"


def get_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表结构"""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol    TEXT    NOT NULL,
                market    TEXT    NOT NULL,
                date      TEXT    NOT NULL,
                open      REAL,
                high      REAL,
                low       REAL,
                close     REAL    NOT NULL,
                volume    REAL,
                UNIQUE(symbol, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_date
            ON stock_daily(symbol, date)
        """)
        conn.commit()
        logger.info("数据库初始化完成: %s", DB_FILE)
    finally:
        conn.close()


def write_stock_data(df: pd.DataFrame, symbol: str, market: str) -> int:
    """
    将股票数据写入数据库（增量，重复日期自动跳过）

    Args:
        df:     包含 date/open/high/low/close/volume 列的 DataFrame
        symbol: 股票代码，如 '600519'、'00700'
        market: 市场标识，如 'A', 'HK'

    Returns:
        实际写入的行数
    """
    if df is None or df.empty:
        logger.warning("%s 数据为空，跳过写入", symbol)
        return 0

    # 统一列名
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})
    df.columns = [c.lower() for c in df.columns]

    required = {"date", "close"}
    missing = required - set(df.columns)
    if missing:
        logger.error("%s 缺少必要列: %s", symbol, missing)
        return 0

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["symbol"] = symbol
    df["market"] = market

    keep_cols = ["symbol", "market", "date", "open", "high", "low", "close", "volume"]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing]

    conn = get_connection()
    try:
        cursor = conn.executemany(
            f"""
            INSERT OR IGNORE INTO stock_daily
                ({', '.join(existing)})
            VALUES
                ({', '.join(['?'] * len(existing))})
            """,
            df.itertuples(index=False, name=None),
        )
        conn.commit()
        written = cursor.rowcount
        logger.info("写入 %s (%s): %d 行", symbol, market, written)
        return written
    finally:
        conn.close()


# ──────────────────────────────────────────
# 查询接口
# ──────────────────────────────────────────

def query_stock(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    查询单只股票历史数据

    Args:
        symbol:     股票代码
        start_date: 开始日期，格式 'YYYY-MM-DD'，不传则取全部
        end_date:   结束日期，格式 'YYYY-MM-DD'，不传则取最新

    Returns:
        以 date 为索引的 DataFrame，含 open/high/low/close/volume
    """
    conditions = ["symbol = ?"]
    params: list = [symbol]

    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)

    sql = f"""
        SELECT date, open, high, low, close, volume
        FROM   stock_daily
        WHERE  {' AND '.join(conditions)}
        ORDER  BY date ASC
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params, parse_dates=["date"])
        df.set_index("date", inplace=True)
        logger.info("查询 %s: %d 行", symbol, len(df))
        return df
    finally:
        conn.close()


def query_latest_date(symbol: str) -> Optional[str]:
    """查询某只股票在库中的最新日期，用于增量更新判断"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()


def list_symbols() -> List[dict]:
    """列出库中所有股票及基本信息"""
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT
                symbol,
                market,
                MIN(date) AS start_date,
                MAX(date) AS end_date,
                COUNT(*)  AS trading_days
            FROM stock_daily
            GROUP BY symbol, market
            ORDER BY market, symbol
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_close_prices(symbols: List[str]) -> pd.DataFrame:
    """
    批量查询多只股票的收盘价，返回宽表（列为股票代码）

    用于多股对比分析，不属于投资组合功能。
    """
    placeholders = ",".join(["?"] * len(symbols))
    sql = f"""
        SELECT date, symbol, close
        FROM   stock_daily
        WHERE  symbol IN ({placeholders})
        ORDER  BY date ASC
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=symbols, parse_dates=["date"])
        wide = df.pivot(index="date", columns="symbol", values="close")
        return wide
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    init_db()

    symbols = list_symbols()
    if symbols:
        print(f"\n数据库中共 {len(symbols)} 只股票：")
        for s in symbols:
            print(f"  {s['symbol']:10s} [{s['market']}]  "
                  f"{s['start_date']} ~ {s['end_date']}  "
                  f"({s['trading_days']} 日)")
    else:
        print("数据库为空，请先运行 main.py 拉取数据。")