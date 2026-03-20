"""
可视化模块

职责：生成单只股票的分析图表，保存为 PNG 文件。
支持：价格趋势 + 技术指标（MA/RSI/MACD）、成交量、收益率分布。
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互模式，避免服务器环境弹窗
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 英文标签，避免 matplotlib 中文字体问题
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


class StockVisualizer:
    """股票可视化器"""

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_analysis(self, df: pd.DataFrame,
                      symbol: str,
                      save_path: str = None) -> str:
        """
        生成综合分析图（2×3 布局）：
          - 价格趋势 + MA20/MA60
          - 成交量
          - RSI
          - MACD
          - 日收益率分布
          - 滚动波动率（30日年化）

        Args:
            df:         含技术指标的 DataFrame（来自 analyzer）
            symbol:     股票代码，用于标题和文件名
            save_path:  指定保存路径；不传则自动保存到 output_dir

        Returns:
            保存的文件路径（字符串）
        """
        if df is None or df.empty:
            logger.warning("%s 数据为空，跳过绘图", symbol)
            return ""

        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f"{symbol} — Stock Analysis", fontsize=13, y=1.01)

        returns = df["close"].pct_change().dropna()

        # ① 价格 + 移动均线
        ax = axes[0, 0]
        ax.plot(df.index, df["close"], linewidth=1.5, color="#2563eb", label="Close")
        for col, color in [("MA_20", "#f59e0b"), ("MA_60", "#ef4444")]:
            if col in df.columns:
                ax.plot(df.index, df[col], linewidth=1, color=color,
                        alpha=0.85, label=col)
        ax.set_title("Price & Moving Averages")
        ax.set_ylabel("Price")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # ② 成交量
        ax = axes[0, 1]
        ax.bar(df.index, df["volume"], color="#94a3b8", alpha=0.7, width=1)
        ax.set_title("Volume")
        ax.set_ylabel("Volume")
        ax.grid(True, alpha=0.3)

        # ③ RSI
        ax = axes[1, 0]
        rsi_col = next((c for c in df.columns if c.startswith("RSI")), None)
        if rsi_col:
            ax.plot(df.index, df[rsi_col], color="#7c3aed", linewidth=1)
            ax.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.axhline(30, color="#16a34a", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.fill_between(df.index, 30, 70, alpha=0.05, color="gray")
            ax.set_ylim(0, 100)
        ax.set_title("RSI (14)")
        ax.set_ylabel("RSI")
        ax.grid(True, alpha=0.3)

        # ④ MACD
        ax = axes[1, 1]
        if "MACD" in df.columns and "MACD_Signal" in df.columns:
            ax.plot(df.index, df["MACD"], color="#2563eb", linewidth=1, label="MACD")
            ax.plot(df.index, df["MACD_Signal"], color="#f59e0b",
                    linewidth=1, label="Signal")
            if "MACD_Hist" in df.columns:
                colors = ["#16a34a" if v >= 0 else "#ef4444"
                          for v in df["MACD_Hist"]]
                ax.bar(df.index, df["MACD_Hist"], color=colors,
                       alpha=0.5, width=1)
            ax.legend(fontsize=7)
        ax.set_title("MACD (12, 26, 9)")
        ax.grid(True, alpha=0.3)

        # ⑤ 日收益率分布
        ax = axes[2, 0]
        ax.hist(returns, bins=60, color="#3b82f6", alpha=0.7, edgecolor="white")
        ax.axvline(returns.mean(), color="#ef4444", linestyle="--",
                   linewidth=1, label=f"Mean {returns.mean():.3%}")
        ax.set_title("Daily Returns Distribution")
        ax.set_xlabel("Daily Return")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ⑥ 30日滚动年化波动率
        ax = axes[2, 1]
        vol = returns.rolling(30).std() * np.sqrt(252)
        ax.plot(vol.index, vol, color="#dc2626", linewidth=1)
        ax.set_title("30-Day Rolling Volatility (Annualized)")
        ax.set_ylabel("Volatility")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = str(self.output_dir / f"plot_{symbol}.png")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("图表已保存: %s", save_path)
        return save_path