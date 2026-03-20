"""
测试 analyzer 模块

覆盖技术指标计算和风险评估核心逻辑。
"""

import pytest
import numpy as np
import pandas as pd
from src.analyzer import StockAnalyzer


# ── 测试数据 ──────────────────────────────────

@pytest.fixture
def analyzer():
    return StockAnalyzer(config=None)


@pytest.fixture
def sample_df():
    """200 个交易日的模拟价格数据，足够计算所有指标"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=dates)


# ── 移动均线 ──────────────────────────────────

def test_ma_columns_exist(analyzer, sample_df):
    """calculate_moving_averages 应生成对应的 MA_N 列"""
    result = analyzer.calculate_moving_averages(sample_df, periods=[5, 20])
    assert "MA_5" in result.columns
    assert "MA_20" in result.columns


def test_ma_values(analyzer, sample_df):
    """MA_5 第5行的值应等于前5日收盘价均值"""
    result = analyzer.calculate_moving_averages(sample_df, periods=[5])
    expected = sample_df["close"].iloc[:5].mean()
    assert result["MA_5"].iloc[4] == pytest.approx(expected, rel=1e-6)


# ── RSI ───────────────────────────────────────

def test_rsi_range(analyzer, sample_df):
    """RSI 值应始终在 0-100 之间"""
    result = analyzer.calculate_rsi(sample_df, period=14)
    rsi_col = "RSI_14"
    assert rsi_col in result.columns
    valid = result[rsi_col].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


# ── MACD ──────────────────────────────────────

def test_macd_columns(analyzer, sample_df):
    """MACD 应生成 MACD / MACD_Signal / MACD_Hist 三列"""
    result = analyzer.calculate_macd(sample_df)
    for col in ["MACD", "MACD_Signal", "MACD_Hist"]:
        assert col in result.columns


def test_macd_hist_equals_diff(analyzer, sample_df):
    """MACD_Hist 应等于 MACD - MACD_Signal"""
    result = analyzer.calculate_macd(sample_df)
    diff = result["MACD"] - result["MACD_Signal"]
    pd.testing.assert_series_equal(
        result["MACD_Hist"].dropna(),
        diff.dropna(),
        check_names=False,
        rtol=1e-6,
    )


# ── 布林带 ────────────────────────────────────

def test_bollinger_bands(analyzer, sample_df):
    """上轨应始终 >= 中轨 >= 下轨"""
    result = analyzer.calculate_bollinger_bands(sample_df)
    valid = result.dropna(subset=["BB_Upper", "BB_Middle", "BB_Lower"])
    assert (valid["BB_Upper"] >= valid["BB_Middle"]).all()
    assert (valid["BB_Middle"] >= valid["BB_Lower"]).all()


# ── 最大回撤 ──────────────────────────────────

def test_max_drawdown_flat(analyzer):
    """价格不变时最大回撤应为 0"""
    prices = pd.Series([100.0] * 50,
                       index=pd.date_range("2024-01-01", periods=50))
    mdd, _, _ = analyzer.calculate_max_drawdown(prices)
    assert mdd == pytest.approx(0.0)


def test_max_drawdown_known(analyzer):
    """价格从 100 跌到 50 再涨回来，最大回撤应为 -50%"""
    prices = pd.Series(
        [100, 90, 80, 70, 60, 50, 60, 70, 80],
        index=pd.date_range("2024-01-01", periods=9)
    )
    mdd, _, _ = analyzer.calculate_max_drawdown(prices)
    assert mdd == pytest.approx(-0.5, rel=1e-6)


# ── 夏普比率 ──────────────────────────────────

def test_sharpe_positive_returns(analyzer):
    """稳定正收益的序列夏普比率应为正"""
    returns = pd.Series([0.001] * 252)
    sharpe = analyzer.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0


def test_sharpe_zero_std(analyzer):
    """收益率标准差为 0 时夏普比率应返回 0，不报错"""
    returns = pd.Series([0.0] * 100)
    sharpe = analyzer.calculate_sharpe_ratio(returns)
    assert sharpe == 0.0


# ── 风险报告 ──────────────────────────────────

def test_risk_report_keys(analyzer, sample_df):
    """generate_risk_report 应包含所有必要的顶层 key"""
    prices = sample_df["close"]
    returns = prices.pct_change().dropna()
    report = analyzer.generate_risk_report(prices, returns)

    for key in ["returns_analysis", "max_drawdown_analysis",
                "performance_ratios", "risk_summary"]:
        assert key in report


def test_risk_summary_fields(analyzer, sample_df):
    """risk_summary 应包含面试中会提到的核心指标"""
    prices = sample_df["close"]
    returns = prices.pct_change().dropna()
    report = analyzer.generate_risk_report(prices, returns)
    summary = report["risk_summary"]

    for field in ["annual_return", "annual_volatility",
                  "max_drawdown", "sharpe_ratio", "win_rate", "risk_rating"]:
        assert field in summary


def test_risk_report_insufficient_data(analyzer):
    """数据不足 20 行时应返回空字典，不报错"""
    prices = pd.Series([100.0, 101.0, 99.0],
                       index=pd.date_range("2024-01-01", periods=3))
    returns = prices.pct_change().dropna()
    report = analyzer.generate_risk_report(prices, returns)
    assert report == {}