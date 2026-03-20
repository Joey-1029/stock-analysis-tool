"""
报告生成模块

职责：接收分析结果，生成 HTML 格式的股票分析报告。
      main.py 只做流程编排，报告渲染逻辑全部在此。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class StockReporter:
    """HTML 分析报告生成器"""

    def __init__(self, reports_dir: str = "data/reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate(self,
                 symbol: str,
                 risk_report: Dict[str, Any],
                 chart_path: Optional[str] = None) -> str:
        """
        生成单只股票的 HTML 分析报告

        Args:
            symbol:      股票代码
            risk_report: analyzer.generate_risk_report() 的返回值
            chart_path:  图表 PNG 文件路径（相对或绝对均可）

        Returns:
            生成的 HTML 文件路径（字符串）
        """
        summary = risk_report.get("risk_summary", {})
        mdd_info = risk_report.get("max_drawdown_analysis", {})
        var_info = risk_report.get("var_analysis", {})
        ret_info = risk_report.get("returns_analysis", {})

        # 图表路径转为相对路径（HTML 中引用）
        img_tag = ""
        if chart_path:
            img_name = Path(chart_path).name
            img_tag = f'<img src="{img_name}" class="chart" alt="Analysis Chart">'

        risk_rating = summary.get("risk_rating", "—")
        rating_class = (
            "good" if "低风险" in risk_rating else
            "warn" if "中" in risk_rating else
            "bad"
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{symbol} 分析报告</title>
<style>
  body {{font-family: 'Segoe UI', sans-serif; margin: 0; background: #f1f5f9; color: #1e293b;}}
  .wrap {{max-width: 960px; margin: 32px auto; padding: 0 16px;}}
  h1 {{font-size: 1.6rem; margin-bottom: 4px;}}
  .meta {{color: #64748b; font-size: 0.85rem; margin-bottom: 24px;}}
  .cards {{display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 28px;}}
  .card {{background: #fff; border-radius: 10px; padding: 18px 20px; box-shadow: 0 1px 4px rgba(0,0,0,.07);}}
  .card .label {{font-size: 0.78rem; color: #64748b; margin-bottom: 6px;}}
  .card .value {{font-size: 1.35rem; font-weight: 600;}}
  .good  {{color: #16a34a;}}
  .warn  {{color: #d97706;}}
  .bad   {{color: #dc2626;}}
  .section {{background: #fff; border-radius: 10px; padding: 22px 24px;
             box-shadow: 0 1px 4px rgba(0,0,0,.07); margin-bottom: 24px;}}
  .section h2 {{font-size: 1rem; margin: 0 0 14px; border-left: 3px solid #3b82f6;
                padding-left: 10px; color: #1e293b;}}
  table {{width: 100%; border-collapse: collapse; font-size: 0.88rem;}}
  td, th {{padding: 8px 12px; text-align: left; border-bottom: 1px solid #f1f5f9;}}
  th {{color: #64748b; font-weight: 500;}}
  .chart {{width: 100%; border-radius: 8px; margin-top: 4px;}}
</style>
</head>
<body>
<div class="wrap">
  <h1>{symbol} 股票分析报告</h1>
  <p class="meta">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;|&nbsp; 数据来源：AkShare</p>

  <div class="cards">
    <div class="card">
      <div class="label">风险评级</div>
      <div class="value {rating_class}">{risk_rating}</div>
    </div>
    <div class="card">
      <div class="label">年化收益率</div>
      <div class="value">{summary.get('annual_return', 0):.2%}</div>
    </div>
    <div class="card">
      <div class="label">年化波动率</div>
      <div class="value">{summary.get('annual_volatility', 0):.2%}</div>
    </div>
    <div class="card">
      <div class="label">夏普比率</div>
      <div class="value">{summary.get('sharpe_ratio', 0):.3f}</div>
    </div>
    <div class="card">
      <div class="label">最大回撤</div>
      <div class="value bad">{summary.get('max_drawdown', 0):.2%}</div>
    </div>
    <div class="card">
      <div class="label">胜率</div>
      <div class="value">{summary.get('win_rate', 0):.2%}</div>
    </div>
  </div>

  <div class="section">
    <h2>详细风险指标</h2>
    <table>
      <tr><th>指标</th><th>数值</th><th>说明</th></tr>
      <tr><td>索提诺比率</td>
          <td>{summary.get('sortino_ratio', 0):.3f}</td>
          <td>仅考虑下行风险的收益率调整指标</td></tr>
      <tr><td>VaR（95%）</td>
          <td>{summary.get('var_95', 0):.3%}</td>
          <td>95% 置信水平下单日最大损失</td></tr>
      <tr><td>CVaR</td>
          <td>{var_info.get('cvar', 0):.3%}</td>
          <td>超过 VaR 时的平均损失（条件风险价值）</td></tr>
      <tr><td>峰值日期</td>
          <td>{mdd_info.get('peak_date', '—')}</td>
          <td>最大回撤起始高点</td></tr>
      <tr><td>谷底日期</td>
          <td>{mdd_info.get('trough_date', '—')}</td>
          <td>最大回撤终点</td></tr>
      <tr><td>日均收益率</td>
          <td>{ret_info.get('mean', 0):.4%}</td><td></td></tr>
      <tr><td>收益率偏度</td>
          <td>{ret_info.get('skewness', 0):.3f}</td>
          <td>正偏：右尾厚（偶有大涨）；负偏：左尾厚（偶有暴跌）</td></tr>
      <tr><td>收益率峰度</td>
          <td>{ret_info.get('kurtosis', 0):.3f}</td>
          <td>&gt;3 说明存在厚尾，极端行情概率高于正态分布</td></tr>
    </table>
  </div>

  {'<div class="section"><h2>技术指标图</h2>' + img_tag + '</div>' if img_tag else ''}

</div>
</body>
</html>"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.reports_dir / f"report_{symbol}_{timestamp}.html"
        out_path.write_text(html, encoding="utf-8")
        logger.info("报告已生成: %s", out_path)
        return str(out_path)