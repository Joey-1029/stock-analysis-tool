# 📈 Stock Analysis Tool

一个面向**金融数据分析 / 数据分析实习**的 Python 股票数据分析项目，适合具备基础编程能力（如 C++ / Python 入门）的计算机专业学生，用于展示**数据获取 → 清洗 → 分析 → 可视化 → 报告生成 → 测试**的完整数据分析流程。

> 🎯 项目定位：
>
> * **实习导向**：面向国内中厂 / 大厂【数据分析 / 数据开发 / 金融数据方向】
> * **技术栈清晰**：Python + Pandas + AkShare + Matplotlib
> * **工程化意识**：模块化代码结构 + 日志 + 测试 + Notebook 演示

---

## ✨ 项目功能概览

* 📥 **股票数据获取**（基于 AkShare）
* 🧹 **数据清洗与质量评估**
* 📊 **技术指标分析**（MA / RSI / MACD 等）
* 💼 **投资组合分析**（收益、波动率、相关性）
* 📈 **数据可视化**（趋势图、指标图）
* 📝 **分析报告自动生成**（CSV / HTML）
* 🧪 **单元测试支持**（pytest）

---

## 📂 项目结构说明

```text
stock-analysis-tool/
├── config/                 # 配置文件（预留，如参数/路径配置）
├── data/                   # 数据目录
│   ├── raw/               # 原始股票数据
│   ├── cleaned/           # 清洗后数据
│   ├── analysis/          # 分析结果数据
│   └── reports/           # 生成的分析报告
├── logs/                   # 日志文件
├── models/                 # 分析模型/策略（预留扩展）
├── notebooks/              # Jupyter Notebook 演示
│   ├── 01_项目演示.ipynb
│   ├── 02_技术分析.ipynb
│   ├── 03_投资组合分析.ipynb
│   └── 04_数据质量报告.ipynb
│   └── utils.py
├── output/                 # Notebook / 程序输出结果
├── src/                    # 核心源码
│   ├── analyzer.py         # 技术指标与分析逻辑
│   ├── data_cleaner.py     # 数据清洗模块
│   ├── data_fetcher_akshare.py  # 数据获取模块
│   ├── data_reporter.py    # 报告生成模块
│   └── visualizer.py       # 可视化模块
├── tests/                  # 单元测试
│   ├── test_analyzer.py
│   ├── test_cleaner.py
│   └── test_fetcher.py
├── main.py                 # 项目主入口
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
└── LICENSE                 # MIT License
```

---

## 🔧 技术栈

* **语言**：Python 3.9+
* **数据处理**：Pandas / NumPy
* **金融数据**：AkShare
* **可视化**：Matplotlib
* **测试**：pytest
* **开发工具**：Jupyter Notebook / VS Code

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Joey-1029/stock-analysis-tool.git
cd stock-analysis-tool
```

### 2️⃣ 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

## ▶️ 运行方式

### 方式一：直接运行主程序

```bash
python main.py
```

主程序将自动完成：

* 股票数据获取
* 数据清洗
* 技术分析
* 图表生成
* 报告输出

---

### 方式二：Notebook 演示（推荐）

```bash
jupyter notebook
```

按顺序运行：

* `01_项目演示.ipynb`
* `02_技术分析.ipynb`
* `03_投资组合分析.ipynb`
* `04_数据质量报告.ipynb`

> 💡 **Notebook 非常适合面试展示与讲解项目思路**

---

## 🧪 单元测试

本项目已加入基础单元测试，覆盖核心模块。

运行测试：

```bash
pytest tests/
```

测试内容包括：

* 数据获取正确性
* 数据清洗逻辑
* 技术指标计算结果

---

## 📊 示例分析内容

* 📉 股价走势与均线对比
* 📈 RSI / MACD 技术指标分析
* 💰 投资组合收益率与风险
* 🧹 数据缺失率与异常值报告

---

## 🎯 项目亮点（面试可说）

* ✅ 完整数据分析 Pipeline
* ✅ 模块化、工程化设计
* ✅ 金融 + 数据分析结合
* ✅ Notebook + Script 双模式
* ✅ 含测试，体现工程素养

> 非“课程作业式项目”，而是**实习友好型项目**

---

## 🔮 可扩展方向（进阶）

* 多股票批量分析
* 回测策略模块
* 简单机器学习预测（如线性回归）
* Streamlit 可视化 Dashboard
* 参数配置文件（config）统一管理

---

## 👨‍🎓 适用人群

* 985 / 211 计算机、信息、金融工程学生
* 想转 **数据分析 / 金融科技** 的 CS 学生
* 暑期实习 / 秋招前项目准备

---

## 📄 License

本项目采用 **MIT License**，可自由使用与修改。

---

## 🙌 作者

* GitHub: [Joey-1029](https://github.com/Joey-1029)
* 项目方向：金融数据分析 / 数据分析实习

---

