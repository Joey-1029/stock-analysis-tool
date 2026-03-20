# 📈 Stock Analysis Tool

基于 Python 的股票数据分析项目，实现从**数据获取 → SQLite 存储 → 清洗 → 技术指标分析 → 可视化 → HTML 报告**的完整 pipeline。

> **技术栈**：Python · Pandas · AkShare · SQLite · Matplotlib · pytest

---

## 项目结构

```
stock-analysis-tool/
├── config/
│   ├── config.py          # 配置管理（dataclass + YAML）
│   └── settings.yaml      # 参数配置（股票池、指标参数、路径等）
├── data/
│   ├── raw/               # AkShare 原始数据备份（CSV）
│   ├── cleaned/           # 清洗后数据（CSV）
│   ├── analysis/          # 分析结果
│   ├── reports/           # HTML 报告 + 图表 PNG
│   └── stocks.db          # SQLite 数据库（取数层核心）
├── logs/                  # 运行日志
├── notebooks/
│   ├── 01_pipeline_demo.ipynb   # 完整流程演示
│   └── 02_technical_analysis.ipynb  # 技术指标专题
├── src/
│   ├── db_manager.py      # SQLite 取数层（写入 / 查询接口）
│   ├── data_fetcher_akshare.py  # AkShare 数据获取（A股 / 港股）
│   ├── data_cleaner.py    # 数据清洗（从数据库读取）
│   ├── analyzer.py        # 技术指标 + 风险评估
│   ├── visualizer.py      # 图表生成
│   └── data_reporter.py   # HTML 报告生成
├── tests/
│   ├── test_db_manager.py
│   ├── test_cleaner.py
│   └── test_analyzer.py
├── main.py                # 主入口（支持命令行参数）
└── requirements.txt
```

---

## 快速开始

```bash
git clone https://github.com/Joey-1029/stock-analysis-tool.git
cd stock-analysis-tool
pip install -r requirements.txt
```

### 运行方式

```bash
# 分析配置文件中所有股票（拉数据 + 分析 + 生成报告）
python main.py

# 分析单只港股
python main.py --symbol 00700 --market HK

# 分析单只A股
python main.py --symbol 600519 --market A

# 只拉数据，不分析
python main.py --fetch-only

# 跳过拉数据，直接分析库中已有数据
python main.py --no-fetch --symbol 00700

# 查看数据库中已有的股票
python main.py --list
```

### 运行测试

```bash
pytest tests/
```

---

## 数据流架构

```
AkShare API
    ↓  data_fetcher_akshare.py
SQLite (stocks.db)          ← 取数层核心
    ↓  db_manager.query_stock()
data_cleaner.py（清洗）
    ↓
analyzer.py（MA / RSI / MACD / 风险评估）
    ↓
visualizer.py  →  PNG 图表
data_reporter.py  →  HTML 报告
```

**为什么用 SQLite 做取数层？**

直接操作 CSV 文件存在以下问题：数据散落多个文件、无法增量更新、多股查询需要手动合并。
引入 SQLite 后，数据统一存入 `stocks.db`，通过 `db_manager.query_stock()` 接口读取，支持按时间范围过滤、增量更新（不重复写入已有日期），分析模块无需感知文件路径。

---

## 主要功能

| 模块 | 功能 |
|------|------|
| `db_manager` | SQLite 写入 / 查询 / 增量更新 / 列出库存 |
| `data_fetcher_akshare` | A股 / 港股日线数据获取，含重试机制 |
| `data_cleaner` | 缺失值处理、异常价格过滤、衍生列计算 |
| `analyzer` | MA / RSI / MACD / 布林带 / 夏普比率 / 最大回撤 / VaR |
| `visualizer` | 6 panel 综合图（价格、成交量、RSI、MACD、收益率分布、波动率）|
| `data_reporter` | HTML 风险报告（指标卡片 + 详细表格 + 图表嵌入）|

---

## 配置说明

编辑 `config/settings.yaml` 修改股票池和分析参数：

```yaml
data:
  a_stocks: ['600519', '000858']   # A股
  hk_stocks: ['00700', '09988']    # 港股
  start_date: '20230101'

analysis:
  ma_periods: [5, 10, 20, 60]
  rsi_period: 14
  risk_free_rate: 0.02
```