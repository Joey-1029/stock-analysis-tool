

import yfinance as yf

# 定义股票代码（美股直接输入代码，如 AAPL；港股加后缀如 0700.HK）
ticker = 'AAPL' 

# 获取数据（参数：代码，开始时间，结束时间）
# 建议大二学生练习时取近 5 年的数据，样本量足够做分析
data = yf.download(ticker, start='2020-01-01', end='2025-01-01')

# 查看前 5 行
print(data.head())

# 将数据保存到本地 CSV，这样下次就不需要重复下载了
data.to_csv(f'../data/{ticker}_data.csv')
