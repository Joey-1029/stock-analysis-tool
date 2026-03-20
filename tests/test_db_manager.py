"""
测试 db_manager 模块

使用临时数据库，测试完自动清理，不污染正式数据。
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import date


# ── 测试用临时数据库 ──────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """每个测试用独立的临时数据库文件"""
    db_file = tmp_path / "test_stocks.db"
    with patch("src.db_manager.DB_FILE", db_file):
        from src.db_manager import init_db
        init_db()
        yield db_file


def make_df(dates, closes):
    """构造最小测试 DataFrame"""
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open":   closes,
        "high":   closes,
        "low":    closes,
        "close":  closes,
        "volume": [1000] * len(closes),
    })


# ── 测试 write_stock_data ─────────────────────

def test_write_basic(tmp_db):
    """正常写入后行数应与输入一致"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data, query_stock
        df = make_df(["2024-01-02", "2024-01-03", "2024-01-04"], [100, 101, 102])
        written = write_stock_data(df, "00700", "HK")
        assert written == 3

        result = query_stock("00700")
        assert len(result) == 3


def test_write_dedup(tmp_db):
    """重复写入同一日期应被忽略（UNIQUE 约束）"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data, query_stock
        df = make_df(["2024-01-02", "2024-01-03"], [100, 101])
        write_stock_data(df, "00700", "HK")
        write_stock_data(df, "00700", "HK")  # 重复写

        result = query_stock("00700")
        assert len(result) == 2  # 不应变成 4


def test_write_empty(tmp_db):
    """写入空 DataFrame 应返回 0"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data
        written = write_stock_data(pd.DataFrame(), "00700", "HK")
        assert written == 0


# ── 测试 query_stock ──────────────────────────

def test_query_date_filter(tmp_db):
    """按日期范围查询应只返回区间内数据"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data, query_stock
        df = make_df(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            [100, 101, 102, 103]
        )
        write_stock_data(df, "00700", "HK")

        result = query_stock("00700", start_date="2024-01-03", end_date="2024-01-04")
        assert len(result) == 2
        assert result["close"].tolist() == [101, 102]


def test_query_no_data(tmp_db):
    """查询不存在的股票应返回空 DataFrame"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import query_stock
        result = query_stock("99999")
        assert result.empty


# ── 测试 query_latest_date ────────────────────

def test_query_latest_date(tmp_db):
    """最新日期应返回写入数据中的最大日期"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data, query_latest_date
        df = make_df(["2024-01-02", "2024-01-05", "2024-01-03"], [100, 103, 101])
        write_stock_data(df, "00700", "HK")

        latest = query_latest_date("00700")
        assert latest == "2024-01-05"


def test_query_latest_date_empty(tmp_db):
    """库中没有该股票时应返回 None"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import query_latest_date
        assert query_latest_date("00700") is None


# ── 测试 list_symbols ─────────────────────────

def test_list_symbols(tmp_db):
    """写入两只股票后 list_symbols 应返回两条记录"""
    with patch("src.db_manager.DB_FILE", tmp_db):
        from src.db_manager import write_stock_data, list_symbols
        write_stock_data(make_df(["2024-01-02"], [100]), "00700", "HK")
        write_stock_data(make_df(["2024-01-02"], [80]),  "09988", "HK")

        symbols = list_symbols()
        codes = [s["symbol"] for s in symbols]
        assert "00700" in codes
        assert "09988" in codes