"""
增强版数据获取模块 - 加入重试机制、日志记录、异常处理
"""
import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import os
import ssl
import urllib3
import logging
from typing import Optional, List, Dict, Any
from functools import wraps
import hashlib
import json

# ========== 添加项目根目录到sys.path ==========
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以导入config模块了
try:
    from config.config import Config, get_config
except ImportError:
    # 如果直接运行此文件，尝试相对导入
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.config import Config, get_config

# ========== 配置日志系统 ==========
def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """设置日志记录器"""
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler
        log_file = log_dir_path / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台handler（修复Windows编码问题）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger('data_fetcher')

# ========== SSL修复 ==========
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ========== 重试装饰器 ==========
def retry_on_failure(max_retries: int = 3, delay: float = 2.0, backoff_factor: float = 2.0):
    """
    重试装饰器 - 用于网络请求
    Args:
        max_retries: 最大重试次数
        delay: 基础延迟时间（秒）
        backoff_factor: 延迟时间倍增因子
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):  # +1 包括第一次尝试
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error("函数 %s 失败，已达最大重试次数 %s", func.__name__, max_retries)
                        raise last_exception
                    
                    # 计算延迟时间（指数退避）
                    sleep_time = current_delay * (backoff_factor ** attempt)
                    logger.warning(
                        "函数 %s 第 %s 次尝试失败: %s... 将在 %.1f 秒后重试",
                        func.__name__, attempt + 1, str(e)[:100], sleep_time
                    )
                    
                    time.sleep(sleep_time)
            
            raise last_exception
        return wrapper
    return decorator

class AKShareDataFetcher:
    """增强版AKShare数据获取器"""
    
    def __init__(self, config: Optional[Config] = None):
        """初始化数据获取器"""
        # 获取配置
        try:
            if config is None:
                self.config = get_config()
            else:
                self.config = config
        except Exception as e:
            logger.error("获取配置失败: %s，使用默认配置", e)
            # 创建默认配置
            from dataclasses import dataclass, field
            from typing import List
            
            @dataclass
            class SimpleConfig:
                data = type('Data', (), {
                    'a_stocks': ['000001', '600519'],
                    'us_stocks': ['AAPL', 'MSFT'],
                    'hk_stocks': ['00700', '09988'],
                    'indices': ['sh000001'],
                    'start_date': '20230101',
                    'end_date': None,
                    'adjust_type': 'qfq',
                    'request_delay': 1.0,
                    'max_retries': 3,
                    'retry_delay': 2.0,
                    'backoff_factor': 1.5
                })()
            
            self.config = SimpleConfig()
        
        # 设置项目路径
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.cleaned_dir = self.data_dir / 'cleaned'
        
        # 确保目录存在
        for directory in [self.data_dir, self.raw_dir, self.cleaned_dir]:
            directory.mkdir(exist_ok=True)
        
        # 缓存已下载的股票，避免重复下载
        self._downloaded_cache_file = self.data_dir / 'download_cache.json'
        self._load_download_cache()
        
        logger.info("=" * 60)
        logger.info("AKShare 数据获取器已初始化")
        logger.info("原始数据目录: %s", self.raw_dir)
        logger.info("清洗数据目录: %s", self.cleaned_dir)
        logger.info("=" * 60)
    
    def _load_download_cache(self):
        """加载下载缓存"""
        if self._downloaded_cache_file.exists():
            try:
                with open(self._downloaded_cache_file, 'r', encoding='utf-8') as f:
                    self._downloaded_cache = json.load(f)
            except:
                self._downloaded_cache = {}
        else:
            self._downloaded_cache = {}
    
    def _save_download_cache(self):
        """保存下载缓存"""
        try:
            with open(self._downloaded_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._downloaded_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error("保存下载缓存失败: %s", e)
    
    def _is_already_downloaded(self, symbol: str, market_type: str) -> bool:
        """检查是否已经下载过最新数据"""
        cache_key = f"{market_type}_{symbol}"
        
        if cache_key not in self._downloaded_cache:
            return False
        
        last_download = self._downloaded_cache[cache_key]
        last_date = datetime.strptime(last_download['date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        
        # 如果今天已经下载过，跳过
        if last_date == today:
            return True
        
        return False
    
    def _update_download_cache(self, symbol: str, market_type: str, filename: str):
        """更新下载缓存"""
        cache_key = f"{market_type}_{symbol}"
        self._downloaded_cache[cache_key] = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'filename': filename
        }
        self._save_download_cache()
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_a_stock(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        获取A股数据 - 增强版
        加入重试机制、缓存检查、详细日志
        """
        logger.info("开始获取A股: %s", symbol)
        
        # 检查缓存
        if self._is_already_downloaded(symbol, 'A股'):
            logger.info("%s 今日已下载，跳过", symbol)
            return None
        
        # 确定交易所前缀
        if symbol.startswith('6'):
            ts_code = f"{symbol}.SH"
            exchange = "SH"
        elif symbol.startswith('0') or symbol.startswith('3'):
            ts_code = f"{symbol}.SZ"
            exchange = "SZ"
        else:
            logger.warning("无法识别股票代码格式: %s", symbol)
            return None
        
        try:
            start_date = start_date or self.config.data.start_date
            logger.info("正在下载 %s (%s 到 %s)...", ts_code, start_date, end_date or '最新')
            
            # 使用重试机制获取数据
            stock_zh_a_daily_df = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust=self.config.data.adjust_type  # 前复权
            )
            
            if stock_zh_a_daily_df.empty:
                logger.warning("%s 数据为空，可能停牌或数据源问题", ts_code)
                return None
            
            logger.info("成功获取 %s 条记录", len(stock_zh_a_daily_df))
            
            # 数据验证
            if not self._validate_data(stock_zh_a_daily_df, symbol):
                logger.error("%s 数据验证失败", symbol)
                return None
            
            # 保存数据
            filename = self._save_data(stock_zh_a_daily_df, ts_code, 'A股')
            
            # 更新缓存
            self._update_download_cache(symbol, 'A股', filename)
            
            # 添加延迟避免被封IP
            time.sleep(self.config.data.request_delay)
            
            return stock_zh_a_daily_df
            
        except Exception as e:
            logger.error("获取 %s 失败: %s", ts_code, str(e), exc_info=True)
            raise  # 让重试装饰器处理
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_us_stock(self, symbol: str, period: str = 'daily') -> Optional[pd.DataFrame]:
        """获取美股数据 - 增强版"""
        logger.info("开始获取美股: %s", symbol)
        
        # 检查缓存
        if self._is_already_downloaded(symbol, '美股'):
            logger.info("%s 今日已下载，跳过", symbol)
            return None
        
        # AKShare的美股代码映射
        symbol_mapping = {
            'AAPL': '105.AAPL', 'MSFT': '106.MSFT', 'GOOGL': '107.GOOGL',
            'AMZN': '108.AMZN', 'TSLA': '109.TSLA', 'NVDA': '110.NVDA',
            'META': '128.META', 'BABA': '116.BABA', 'PDD': '113.PDD',
            'JD': '115.JD', 'NFLX': '111.NFLX', 'BRK.B': '112.BRK.B'
        }
        
        ak_code = symbol_mapping.get(symbol, symbol)
        
        try:
            logger.info("正在下载 %s (%s) 周期: %s...", symbol, ak_code, period)
            
            # 根据周期选择函数
            if period == 'daily':
                df = ak.stock_us_daily(symbol=ak_code, adjust=self.config.data.adjust_type)
            elif period == 'weekly':
                df = ak.stock_us_weekly(symbol=ak_code, adjust=self.config.data.adjust_type)
            elif period == 'monthly':
                df = ak.stock_us_monthly(symbol=ak_code, adjust=self.config.data.adjust_type)
            else:
                logger.error("不支持的周期: %s", period)
                return None
            
            if df.empty:
                logger.warning("%s 数据为空", symbol)
                return None
            
            logger.info("成功获取 %s 条记录", len(df))
            
            # 数据验证
            if not self._validate_data(df, symbol):
                logger.error("%s 数据验证失败", symbol)
                return None
            
            # 保存数据
            filename = self._save_data(df, symbol, '美股')
            
            # 更新缓存
            self._update_download_cache(symbol, '美股', filename)
            
            time.sleep(self.config.data.request_delay)
            
            return df
            
        except Exception as e:
            logger.error("获取 %s 失败: %s", symbol, str(e), exc_info=True)
            raise
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def get_hk_stock(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取港股数据 - 增强版"""
        logger.info("开始获取港股: %s", symbol)
        
        # 检查缓存
        if self._is_already_downloaded(symbol, '港股'):
            logger.info("%s 今日已下载，跳过", symbol)
            return None
        
        try:
            logger.info("正在下载 %s...", symbol)
            
            df = ak.stock_hk_daily(symbol=symbol, adjust=self.config.data.adjust_type)
            
            if df.empty:
                logger.warning("%s 数据为空", symbol)
                return None
            
            logger.info("成功获取 %s 条记录", len(df))
            
            # 数据验证
            if not self._validate_data(df, symbol):
                logger.error("%s 数据验证失败", symbol)
                return None
            
            # 保存数据
            filename = self._save_data(df, f"HK_{symbol}", '港股')
            
            # 更新缓存
            self._update_download_cache(symbol, '港股', filename)
            
            time.sleep(self.config.data.request_delay)
            
            return df
            
        except Exception as e:
            logger.error("获取 %s 失败: %s", symbol, str(e), exc_info=True)
            raise
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """验证数据质量"""
        if df.empty:
            logger.error("%s 数据为空", symbol)
            return False
        
        # 检查必要列
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error("%s 缺少必要列: %s", symbol, missing_cols)
            return False
        
        # 检查NaN值
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning("%s 有 %s 个NaN值", symbol, nan_counts.sum())
        
        # 检查价格合理性
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                logger.warning("%s %s 列包含非正数值", symbol, col)
        
        return True
    
    def _save_data(self, data: pd.DataFrame, symbol: str, market_type: str, data_type: str = 'raw') -> str:
        """
        保存数据到CSV文件 - 增强版
        返回文件名用于缓存
        """
        # 根据数据类型选择目录
        save_dir = self.raw_dir if data_type == 'raw' else self.cleaned_dir
        prefix = 'RAW' if data_type == 'raw' else 'CLEANED'
        
        # 生成文件名（包含时间戳和哈希值，避免重复）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_hash = hashlib.md5(data.to_string().encode()).hexdigest()[:8]
        filename = f"{prefix}_{symbol.replace('.', '_')}_{market_type}_{timestamp}_{data_hash}.csv"
        filepath = save_dir / filename
        
        try:
            # 标准化数据格式
            if 'date' in data.columns:
                data = data.copy()
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            # 保存到CSV
            data.to_csv(filepath, encoding='utf-8')
            
            # 记录文件信息
            file_info = {
                'symbol': symbol,
                'market': market_type,
                'rows': len(data),
                'columns': list(data.columns),
                'date_range': f"{data.index.min()} to {data.index.max()}" if hasattr(data.index, 'min') else 'N/A'
            }
            
            logger.info("数据保存到: %s", filepath)
            logger.info("文件信息: %s", file_info)
            
            # 数据预览
            logger.debug("数据预览:\n%s", data.head(3))
            
            return filename
            
        except Exception as e:
            logger.error("保存数据失败: %s", str(e))
            raise
    
    def download_all_stocks(self) -> Dict[str, Any]:
        """
        下载所有配置的股票数据
        返回下载结果的统计信息
        """
        logger.info("=" * 60)
        logger.info("开始批量下载股票数据")
        logger.info("=" * 60)
        
        results = {
            'success': [],
            'failed': [],
            'skipped': [],
            'total': 0
        }
        
        # 下载港股
        logger.info("下载港股数据:")
        for stock in self.config.data.hk_stocks:
            try:
                df = self.get_hk_stock(stock)
                if df is not None:
                    results['success'].append(f"HK_{stock}")
                else:
                    results['skipped'].append(f"HK_{stock}")
            except Exception as e:
                logger.error("下载港股 %s 失败: %s", stock, e)
                results['failed'].append(f"HK_{stock}")
        
        # 下载A股
        logger.info("下载A股数据:")
        for stock in self.config.data.a_stocks:
            try:
                df = self.get_a_stock(stock)
                if df is not None:
                    results['success'].append(f"A_{stock}")
                else:
                    results['skipped'].append(f"A_{stock}")
            except Exception as e:
                logger.error("下载A股 %s 失败: %s", stock, e)
                results['failed'].append(f"A_{stock}")
        
        # 下载美股
        logger.info("下载美股数据:")
        for stock in self.config.data.us_stocks:
            try:
                df = self.get_us_stock(stock)
                if df is not None:
                    results['success'].append(f"US_{stock}")
                else:
                    results['skipped'].append(f"US_{stock}")
            except Exception as e:
                logger.error("下载美股 %s 失败: %s", stock, e)
                results['failed'].append(f"US_{stock}")
        
        # 下载指数
        logger.info("下载指数数据:")
        for idx in self.config.data.indices:
            try:
                df = self.get_index_data(idx)
                if df is not None:
                    results['success'].append(f"IDX_{idx}")
                else:
                    results['skipped'].append(f"IDX_{idx}")
            except Exception as e:
                logger.error("下载指数 %s 失败: %s", idx, e)
                results['failed'].append(f"IDX_{idx}")
        
        # 统计结果
        results['total'] = len(results['success']) + len(results['failed']) + len(results['skipped'])
        
        # 打印总结
        logger.info("=" * 60)
        logger.info("下载结果总结:")
        logger.info("成功: %s", len(results['success']))
        logger.info("跳过: %s", len(results['skipped']))
        logger.info("失败: %s", len(results['failed']))
        logger.info("总计: %s", results['total'])
        
        if results['failed']:
            logger.warning("失败的股票: %s", results['failed'])
        
        logger.info("数据保存在: %s", self.raw_dir)
        logger.info("=" * 60)
        
        return results
    
    def get_index_data(self, symbol: str = 'sh000001') -> Optional[pd.DataFrame]:
        """获取指数数据 - 增强版"""
        logger.info("获取指数: %s", symbol)
        
        try:
            if symbol.startswith('sh') or symbol.startswith('sz'):
                df = ak.index_zh_a_hist(symbol=symbol, period="daily")
            elif symbol == 'hkhsi':
                df = ak.index_hk_hist(symbol="HSI")
            else:
                logger.error("不支持的指数代码: %s", symbol)
                return None
            
            if df.empty:
                logger.warning("%s 指数数据为空", symbol)
                return None
            
            logger.info("成功获取 %s 条记录", len(df))
            
            # 保存数据
            self._save_data(df, symbol, '指数')
            
            return df
            
        except Exception as e:
            logger.error("获取 %s 失败: %s", symbol, str(e), exc_info=True)
            return None

# 主程序入口
def main():
    """主函数"""
    logger.info("启动 AKShare 数据获取器")
    
    # 从配置文件加载配置
    config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
    if config_path.exists():
        try:
            config = get_config()
            logger.info("从配置文件加载配置")
        except Exception as e:
            logger.warning("配置文件加载失败，使用默认配置: %s", e)
            config = None
    else:
        logger.info("配置文件不存在，使用默认配置")
        config = None
    
    fetcher = AKShareDataFetcher(config)
    
    # 显示菜单
    logger.info("选择下载模式:")
    logger.info("1. 批量下载所有股票")
    logger.info("2. 下载港股")
    logger.info("3. 下载A股")
    logger.info("4. 下载美股")
    logger.info("5. 查看数据状态")
    logger.info("6. 测试单只股票")
    
    try:
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == '1':
            results = fetcher.download_all_stocks()
            
            # 保存下载报告
            report_file = fetcher.data_dir / 'download_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            logger.info("下载报告已保存到: %s", report_file)
            
        elif choice == '2':
            for stock in fetcher.config.data.hk_stocks:
                fetcher.get_hk_stock(stock)
                
        elif choice == '3':
            for stock in fetcher.config.data.a_stocks:
                fetcher.get_a_stock(stock)
                
        elif choice == '4':
            for stock in fetcher.config.data.us_stocks:
                fetcher.get_us_stock(stock)
                
        elif choice == '5':
            # 检查数据状态
            raw_files = list(fetcher.raw_dir.glob("*.csv"))
            logger.info("原始数据文件: %s", len(raw_files))
            for f in raw_files[:5]:  # 只显示前5个
                logger.info("  - %s", f.name)
                
        elif choice == '6':
            test_symbol = input("输入测试股票代码 (如 00700): ").strip()
            if test_symbol.startswith('6') or test_symbol.startswith('0') or test_symbol.startswith('3'):
                fetcher.get_a_stock(test_symbol)
            elif test_symbol in ['AAPL', 'MSFT', 'GOOGL', 'TSLA']:
                fetcher.get_us_stock(test_symbol)
            else:
                fetcher.get_hk_stock(test_symbol)
                
        else:
            logger.warning("无效选择，使用默认模式（批量下载）")
            fetcher.download_all_stocks()
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error("程序运行出错: %s", e, exc_info=True)
    
    logger.info("程序执行完成")

if __name__ == "__main__":
    main()