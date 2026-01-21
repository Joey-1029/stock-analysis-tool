股票数据分析工具 (Stock Analysis Tool)
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/github/stars/yourusername/stock-analysis-tool?style=social

一个功能完整的Python股票数据分析工具，支持技术分析、数据可视化和自动化报告生成。

✨ 功能特性
📊 数据处理
多源数据获取: 支持A股、港股、美股数据获取

智能数据清洗: 自动处理缺失值、异常值、日期标准化

数据合并: 创建统一数据集用于投资组合分析

📈 技术分析
移动平均线: MA5, MA10, MA20, MA50, MA200

动量指标: RSI相对强弱指标（含超买超卖信号）

趋势指标: MACD指数平滑异同移动平均线

波动率指标: 布林带（宽度和%B值计算）

成交量分析: 成交量移动平均、VWAP成交量加权平均价

📉 可视化与报告
专业图表: 价格趋势、成交量分析、技术指标叠加

对比分析: 多股票对比（标准化价格）

投资组合分析: 相关性矩阵、风险收益分析

质量报告: 数据质量评估与建议

技术报告: 完整的分析报告与操作建议

🚀 快速开始
环境要求
Python 3.8 或更高版本

Git

安装步骤
bash
# 1. 克隆仓库
git clone https://github.com/yourusername/stock-analysis-tool.git
cd stock-analysis-tool

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行程序
python main.py
基本使用
运行主程序后，您将看到以下菜单：

text
============================================================
STOCK ANALYSIS TOOL - MAIN MENU
============================================================
Data Status: 0 raw, 0 cleaned, 0 analysis

Available modules:
1. 📥 Fetch data (download new stock data)
2. 🧹 Clean data (process raw data)
3. 📊 Analyze data (technical analysis)
4. 📈 Visualize data (charts and graphs)
5. 📋 Generate reports (data quality)
6. 🔄 Run full pipeline (all modules)
7. 📊 Show data status
8. ❌ Exit
使用示例
获取数据: 选择选项1下载股票数据

清洗数据: 选择选项2处理原始数据

技术分析: 选择选项3进行技术指标分析

可视化: 选择选项4生成专业图表

生成报告: 选择选项5创建数据质量报告

📁 项目结构
text
stock-analysis-tool/
├── data/                      # 数据存储目录
│   ├── raw/                  # 原始下载数据
│   ├── cleaned/              # 清洗后数据
│   ├── analysis/             # 分析结果
│   └── reports/              # 生成报告
├── src/                      # 源代码目录
│   ├── analyzer.py           # 技术分析引擎
│   ├── visualizer.py         # 可视化模块
│   ├── data_cleaner.py       # 数据清洗模块
│   ├── data_fetcher_akshare.py # 数据获取模块
│   ├── data_reporter.py      # 报告生成模块
│   └── data_loader.py        # 数据加载工具
├── config/                   # 配置文件
├── main.py                   # 主程序入口
├── requirements.txt          # 项目依赖
├── README.md                 # 说明文档
└── LICENSE                   # MIT许可证
📊 支持的技术指标
类别	指标	说明
趋势指标	MA5, MA10, MA20, MA50, MA200	移动平均线，识别价格趋势
动量指标	RSI14, MACD(12,26,9)	相对强弱指标，判断超买超卖
波动率指标	布林带(20,2)	价格波动通道分析
成交量指标	成交量MA20, VWAP	成交量分析和确认
支撑阻力	滚动高/低点	动态支撑阻力位识别
收益率	日收益率, 累计收益率	投资绩效衡量
🛠 依赖安装
基础依赖
txt
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
数据获取依赖
txt
akshare>=1.10.0
可选依赖
txt
scikit-learn>=1.3.0      # 机器学习功能
jupyter>=1.0.0           # 交互式分析
tabulate>=0.9.0          # 表格美化输出
📝 代码示例
单股票分析
python
from src.analyzer import StockAnalyzer

analyzer = StockAnalyzer()
df, report = analyzer.analyze_stock('00700.HK')  # 分析腾讯控股
数据可视化
python
from src.visualizer import StockVisualizer

visualizer = StockVisualizer()
df = visualizer.load_stock_data('cleaned/00700.HK_cleaned.csv')
visualizer.plot_price_trend(df, "腾讯控股", "00700.HK")
生成质量报告
python
from src.data_reporter import DataQualityReporter

reporter = DataQualityReporter()
reporter.generate_quality_report('00700.HK')
🔧 配置说明
数据源配置
项目默认支持以下股票：

港股: 00700.HK (腾讯控股), 09988.HK (阿里巴巴)

A股: 可通过修改代码添加

美股: 可通过修改代码添加

技术指标参数
所有技术指标参数可在 src/analyzer.py 中修改：

RSI周期: 默认14天

MACD参数: 快线12天, 慢线26天, 信号线9天

布林带: 周期20天, 标准差2倍

🤝 贡献指南
欢迎贡献代码！请按以下步骤操作：

Fork 本仓库

创建功能分支 (git checkout -b feature/新功能)

提交更改 (git commit -m '添加新功能')

推送到分支 (git push origin feature/新功能)

开启 Pull Request

代码规范
遵循 PEP 8 代码规范

为所有函数和类添加文档字符串

为新功能添加单元测试

更新相关文档

📈 项目路线图
已完成功能 ✅
核心数据获取和清洗流水线

基础技术指标实现

单股票可视化分析

数据质量报告系统

开发中功能 🔄
Web界面（Streamlit集成）

实时数据更新

回测模块

机器学习预测

计划功能 📅
REST API接口

Docker容器化

云部署选项

移动端界面

❓ 常见问题
Q: 数据获取失败怎么办？
A: 检查网络连接，或尝试单独运行 python src/data_fetcher_akshare.py

Q: 如何添加新的股票？
A: 修改 src/data_fetcher_akshare.py 中的股票代码列表

Q: 如何修改技术指标参数？
A: 在 src/analyzer.py 中修改相应的参数设置

Q: 生成的图表保存在哪里？
A: 图表会直接显示，如需保存可修改 visualizer.py 中的代码

📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情。

📞 联系与支持
提交 Issue: GitHub Issues

功能请求: 通过 Issue 提交

问题反馈: 描述详细的问题现象和复现步骤

🙏 致谢
AKShare - 数据源提供

Matplotlib - 可视化库

Pandas - 数据处理库
