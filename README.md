📊 股票分析工具 (Stock Analysis Tool)
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/License-MIT-yellow.svg
https://img.shields.io/github/stars/yourusername/stock-analysis-tool?style=social

一个功能完整的Python股票数据分析工具，支持技术分析、投资组合优化、数据质量评估和自动化报告生成。

✨ 功能特性
📊 数据处理
多源数据获取: 支持A股、港股、美股数据获取（AKShare/Yahoo Finance）

智能数据清洗: 自动处理缺失值、异常值、日期标准化

数据质量评估: 6维度数据质量评分系统

投资组合管理: 多股票数据合并和对比分析

📈 技术分析
移动平均线: MA5, MA10, MA20, MA50, MA100, MA200

动量指标: RSI相对强弱指标（含超买超卖信号）

趋势指标: MACD指数平滑异同移动平均线

波动率指标: 布林带（宽度和%B值计算）

成交量分析: 成交量移动平均、VWAP成交量加权平均价

支撑阻力: 动态支撑阻力位识别

📉 可视化与报告
专业图表: 价格趋势、成交量分析、技术指标叠加

对比分析: 多股票对比（标准化价格）

投资组合分析: 相关性矩阵、风险收益分析

数据质量报告: 自动化质量评估与建议

技术分析报告: 完整的分析报告与操作建议

HTML报告: 生成交互式HTML分析报告

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

# 3. 获取数据（可选）
python src/data_fetcher_akshare.py

# 4. 运行主程序
python main.py
基础使用
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

运行完整流程: 选择选项6自动执行所有步骤

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
├── notebooks/                # Jupyter Notebook演示
│   ├── 01_项目演示.ipynb     # 完整项目演示
│   ├── 02_技术分析.ipynb     # 技术分析演示
│   ├── 03_投资组合分析.ipynb # 投资组合分析
│   └── 04_数据质量报告.ipynb # 数据质量评估
├── config/                   # 配置文件
├── main.py                   # 主程序入口
├── requirements.txt          # 项目依赖
├── README.md                 # 说明文档
└── LICENSE                   # MIT许可证
📊 支持的技术指标
类别	指标	说明
趋势指标	MA5, MA10, MA20, MA50, MA100, MA200	移动平均线，识别价格趋势
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
Jupyter相关
txt
jupyter>=1.0.0
notebook>=6.5.0
ipython>=8.0.0
报告生成
txt
nbconvert>=7.0.0
nbformat>=5.0.0
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
使用Jupyter Notebook
python
# 启动Jupyter
jupyter notebook

# 打开 notebooks/01_项目演示.ipynb
# 按顺序运行所有单元格查看完整演示
📊 Jupyter Notebook演示
项目包含4个完整的Jupyter Notebook演示：

1. 01_项目演示.ipynb
完整项目功能演示

环境设置和模块导入

数据加载和基本分析

技术指标计算和可视化

2. 02_技术分析.ipynb
深度技术指标分析

多图表技术分析仪表板

信号检测和交易建议

3. 03_投资组合分析.ipynb
多股票投资组合构建

相关性分析和风险评估

投资组合优化建议

压力测试和风险管理

4. 04_数据质量报告.ipynb
数据完整性检查

异常值检测和处理

自动化质量评分系统

清洗建议和报告生成

📄 生成HTML报告
从Notebook生成报告
bash
# 生成单个Notebook的HTML报告
jupyter nbconvert --to html notebooks/01_项目演示.ipynb --output-dir reports/

# 生成所有Notebook的报告
jupyter nbconvert --to html notebooks/*.ipynb --output-dir reports/

# 生成无代码版本（适合演示）
jupyter nbconvert --to html notebooks/01_项目演示.ipynb --no-input --output-dir reports/
从主程序生成报告
在主菜单中选择选项5（Generate reports）可以生成：

数据质量报告

技术分析报告

投资组合报告

🔧 配置说明
股票配置
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

投资组合分析模块

Jupyter Notebook演示

HTML报告生成

开发中功能 🔄
Web界面（Streamlit集成）

实时数据更新

机器学习预测模块

回测系统

计划功能 📅
REST API接口

Docker容器化

移动端界面

云部署选项

❓ 常见问题
Q: 数据获取失败怎么办？
A: 检查网络连接，或尝试单独运行 python src/data_fetcher_akshare.py

Q: 如何添加新的股票？
A: 修改 src/data_fetcher_akshare.py 中的股票代码列表

Q: 如何修改技术指标参数？
A: 在 src/analyzer.py 中修改相应的参数设置

Q: 生成的图表保存在哪里？
A: 图表会直接显示，如需保存可修改 visualizer.py 中的代码

Q: 如何在VS Code中使用Jupyter？
A: 安装Python和Jupyter扩展，然后直接打开.ipynb文件

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

Jupyter - 交互式计算环境

⭐ 如果这个项目对您有帮助，请给它一个Star！

📧 有任何问题或建议，欢迎提交Issue或Pull Request！

🎯 适用场景
对于个人投资者
分析股票趋势和识别买卖点

比较多只股票表现

生成专业的分析图表

监控投资组合风险

对于学生和研究者
学习金融数据分析方法

研究技术指标的有效性

进行量化分析实验

创建学术研究报告

对于开发者
学习Python数据分析项目架构

作为金融数据分析的基础框架

扩展和定制特定功能

学习模块化设计和代码组织

🚀 性能优化建议
大数据量处理: 对于大量股票数据，建议使用Dask或Modin替代Pandas

并行计算: 多股票分析可以使用多进程加速

缓存机制: 频繁访问的数据可以加入缓存

增量更新: 只下载和处理最新数据

🔧 开发环境设置
bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black flake8  # 开发工具

# 运行测试
python -m pytest tests/

# 代码格式化
black src/
项目持续更新中，欢迎关注和贡献！ 🚀
