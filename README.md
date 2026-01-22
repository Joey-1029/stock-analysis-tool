# 股票数据分析工具 (港股专业版)

一个专业的港股数据分析工具，提供完整的数据获取、清洗、技术分析和报告生成流程。

## 🌟 项目特点

- **港股专用**：专门为港股市场优化的分析工具
- **完整流程**：从数据获取 → 清洗 → 分析 → 可视化 → 报告
- **专业分析**：技术指标计算 + 风险评估 + 投资组合优化
- **自动化报告**：自动生成HTML和JSON格式的详细报告
- **多格式支持**：支持CSV、HTML、JSON等多种数据格式

## 📁 项目结构
stock-analysis-tool/
├── data/ # 数据存储目录
│ ├── raw/ # 原始下载数据
│ ├── cleaned/ # 清洗后数据
│ ├── analysis/ # 分析结果
│ └── reports/ # 生成报告
├── src/ # 源代码目录
│ ├── analyzer.py # 技术分析引擎（增强版）
│ ├── visualizer.py # 可视化模块（英文版）
│ ├── data_cleaner.py # 数据清洗模块
│ ├── data_fetcher_akshare.py # 数据获取模块（带重试机制）
│ ├── data_reporter.py # 数据质量报告模块
│ └── compat.py # 兼容层模块（可选）
├── notebooks/ # Jupyter Notebook演示
│ ├── 01_项目演示.ipynb # 完整项目演示
│ ├── 02_技术分析.ipynb # 技术分析演示
│ ├── 03_投资组合分析.ipynb # 投资组合分析
│ └── 04_数据质量报告.ipynb # 数据质量评估（可选）
├── config/ # 配置文件目录
│ ├── config.py # Python配置类
│ └── settings.yaml # YAML配置文件
├── output/ # 输出目录（自动生成）
├── logs/ # 日志目录
├── models/ # 机器学习模型目录
├── main.py # 主程序入口（港股专业版）
├── requirements.txt # 项目依赖
└── README.md # 说明文档

text

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/Joey-1029/stock-analysis-tool.git
cd stock-analysis-tool

# 安装依赖
pip install -r requirements.txt
2. 运行主程序
bash
python main.py
主程序将执行以下步骤：

✅ 创建项目目录结构

✅ 检查并安装依赖

✅ 下载港股数据（腾讯、阿里、移动、小米）

✅ 自动清洗数据

✅ 进行技术分析

✅ 生成可视化图表

✅ 创建HTML报告

✅ 在浏览器中打开报告

3. 使用Jupyter Notebook
bash
# 启动Jupyter
jupyter notebook

# 然后打开 notebooks/01_项目演示.ipynb
# 按顺序运行所有单元格
📊 数据源
港股数据：通过AKShare获取（前复权）

支持股票：

00700 - 腾讯控股

09988 - 阿里巴巴

00941 - 中国移动

01810 - 小米集团

更多港股可自行添加

🔧 核心功能
1. 数据获取模块 (src/data_fetcher_akshare.py)
支持A股、港股、美股、指数

重试机制和错误处理

缓存避免重复下载

延迟避免被封IP

2. 数据清洗模块 (src/data_cleaner.py)
自动处理缺失值

标准化列名（中英文支持）

日期格式统一化

异常值检测

3. 技术分析模块 (src/analyzer.py)
技术指标：MA、RSI、MACD、布林带

风险评估：最大回撤、夏普比率、索提诺比率

风险价值：VaR、CVaR计算

专业报告：自动生成风险评估报告

4. 可视化模块 (src/visualizer.py)
价格趋势图表

技术指标叠加

收益率分布

相关性热图

投资组合分析

5. 报告生成模块 (src/data_reporter.py)
数据质量报告

HTML可视化报告

JSON详细数据导出

港股专用报告模板

📈 技术指标
移动平均线：MA5、MA10、MA20、MA50、MA60

相对强弱指数：RSI(14)

MACD：快线12、慢线26、信号线9

布林带：20日均线 ± 2倍标准差

波动率：20日、60日滚动波动率

📋 风险评估
最大回撤：历史最大亏损幅度

夏普比率：风险调整后收益

索提诺比率：下行风险调整后收益

风险价值：95%、99%置信水平VaR

风险评级：A-低风险 到 E-高风险

💡 使用示例
示例1：快速分析单只股票
python
from src.analyzer import StockAnalyzer

# 初始化分析器
analyzer = StockAnalyzer()

# 加载数据
df = analyzer.load_cleaned_data('00700')

# 计算技术指标
df_with_indicators = analyzer.calculate_all_indicators(df)

# 生成风险报告
prices = df['close']
returns = prices.pct_change()
risk_report = analyzer.generate_risk_report(prices, returns)

# 打印报告
analyzer.print_risk_report(risk_report, '腾讯控股')
示例2：投资组合分析
python
from src.visualizer import StockVisualizer

visualizer = StockVisualizer()

# 比较两只股票
visualizer.compare_two_stocks(
    file1='00700_cleaned.csv',
    file2='09988_cleaned.csv',
    name1="腾讯",
    name2="阿里",
    ticker1="00700.HK",
    ticker2="09988.HK"
)

# 投资组合分析
visualizer.analyze_portfolio(
    files=['00700_cleaned.csv', '09988_cleaned.csv', '00941_cleaned.csv'],
    names=["腾讯", "阿里", "移动"],
    tickers=["00700", "09988", "00941"],
    weights=[0.4, 0.3, 0.3]  # 自定义权重
)
🔍 数据目录说明
data/raw/
原始下载数据

文件名格式：HK_00700_腾讯控股_YYYYMMDD.csv

包含原始OHLCV数据

data/cleaned/
清洗后的数据

文件名格式：00700_cleaned.csv

标准化列名，无缺失值

data/analysis/
分析结果

技术指标CSV文件

风险评估JSON报告

data/reports/
HTML格式报告

JSON详细数据

时间戳命名，避免覆盖

🛠️ 故障排除
问题1：AKShare下载失败
bash
# 解决方案：更新AKShare
pip install akshare --upgrade

# 或使用代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
问题2：中文显示乱码
python
# 在代码中添加
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
问题3：依赖冲突
bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
📚 扩展功能
添加新股票
在 main.py 的 hk_stocks_to_download 列表中添加新股票

在 generate_hk_report 函数的 hk_stock_names 字典中添加中文名

重新运行主程序

自定义分析参数
编辑 config/settings.yaml：

yaml
analysis:
  ma_periods: [5, 10, 20, 30, 60]
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2
  risk_free_rate: 0.02
  trading_days_per_year: 252
添加新数据源
修改 src/data_fetcher_akshare.py：

支持更多AKShare接口

添加其他数据源（Yahoo Finance、Tushare等）

实现自定义数据获取逻辑

📈 性能优化
向量化计算：技术指标使用向量化运算

缓存机制：避免重复下载和计算

内存优化：分批处理大数据集

并发下载：多线程获取多只股票数据

🤝 贡献指南
Fork 本仓库

创建功能分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启 Pull Request

📄 许可证
本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

📞 联系方式
Joey - GitHub

项目链接：https://github.com/Joey-1029/stock-analysis-tool

🙏 致谢
AKShare - 免费开源股票数据接口

Pandas - 数据分析库

Matplotlib - 数据可视化库

所有开源贡献者

免责声明：本工具仅供学习和研究使用，不构成任何投资建议。股市有风险，投资需谨慎