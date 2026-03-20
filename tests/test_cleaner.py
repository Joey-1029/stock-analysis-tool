"""
测试 data_cleaner 模块

mock 掉数据库查询，直接测试清洗逻辑本身。
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch


def make_raw_df(dates, closes, inject_nan=False, inject_zero=False):
    """构造模拟从数据库读出的原始 DataFrame"""
    df = pd.DataFrame({
        "open":   closes,
        "high":   closes,
        "low":    closes,
        "close":  list(closes),
        "volume": [1000] * len(closes),
    }, index=pd.to_datetime(dates))
    df.index.name = "date"

    if inject_nan:
        df.loc[df.index[1], "close"] = np.nan
    if inject_zero:
        df.loc[df.index[2], "close"] = 0.0

    return df


# ── 辅助：跳过真实文件写入 ───────────────────

@pytest.fixture
def cleaner(tmp_path):
    from src.data_cleaner import StockDataCleaner
    return StockDataCleaner(cleaned_dir=str(tmp_path))


# ── 测试正常清洗 ──────────────────────────────

def test_clean_basic(cleaner):
    """正常数据清洗后行数不变，且包含衍生列"""
    raw = make_raw_df(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        [100, 101, 102, 103]
    )
    with patch("src.data_cleaner.query_stock", return_value=raw):
        df = cleaner.clean("00700")

    assert not df.empty
    assert "daily_return" in df.columns
    assert "log_return" in df.columns
    assert len(df) == 4


def test_clean_removes_zero_price(cleaner):
    """收盘价为 0 的行应被过滤掉"""
    raw = make_raw_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100, 0, 102],
        inject_zero=False   # 手动构造
    )
    raw.iloc[1, raw.columns.get_loc("close")] = 0.0

    with patch("src.data_cleaner.query_stock", return_value=raw):
        df = cleaner.clean("00700")

    assert len(df) == 2
    assert (df["close"] > 0).all()


def test_clean_fills_nan(cleaner):
    """NaN 收盘价应被前向填充"""
    raw = make_raw_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 101.0, 102.0],
        inject_nan=True   # index[1] 的 close 变成 NaN
    )
    with patch("src.data_cleaner.query_stock", return_value=raw):
        df = cleaner.clean("00700")

    assert df["close"].isna().sum() == 0
    # 前向填充后 index[1] 应等于 index[0] 的值
    assert df["close"].iloc[1] == pytest.approx(100.0)


def test_clean_saves_csv(cleaner, tmp_path):
    """清洗后应在 cleaned_dir 下生成 CSV 文件"""
    raw = make_raw_df(
        ["2024-01-02", "2024-01-03"],
        [100, 101]
    )
    with patch("src.data_cleaner.query_stock", return_value=raw):
        cleaner.clean("00700")

    csv_file = tmp_path / "00700_cleaned.csv"
    assert csv_file.exists()


def test_clean_empty_db(cleaner):
    """数据库无数据时应返回空 DataFrame"""
    with patch("src.data_cleaner.query_stock", return_value=pd.DataFrame()):
        df = cleaner.clean("00700")
    assert df.empty


# ── 测试 daily_return 计算正确性 ─────────────

def test_daily_return_values(cleaner):
    """daily_return 第二行应等于 (101-100)/100"""
    raw = make_raw_df(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        [100.0, 101.0, 99.0]
    )
    with patch("src.data_cleaner.query_stock", return_value=raw):
        df = cleaner.clean("00700")

    assert df["daily_return"].iloc[1] == pytest.approx(0.01)
    assert df["daily_return"].iloc[2] == pytest.approx(-2 / 101)