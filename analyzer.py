"""
增强版股票分析引擎 - 包含专业风险评估和向量化计算
"""
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from config.config import get_config
except ImportError:
    # 如果直接运行此文件，尝试相对导入
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config.config import get_config


import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedStockAnalyzer:
    """增强版股票分析引擎 - 面试级专业分析"""
    
    def __init__(self, config=None):
        self.config = config
        self.risk_free_rate = config.analysis.risk_free_rate if config else 0.02
        self.trading_days = config.analysis.trading_days_per_year if config else 252
        
        # 性能优化：缓存计算结果
        self._cache = {}
        
        logger.info("增强版股票分析引擎已初始化")
        logger.info("无风险利率: %.2f%%, 年交易日数: %s", self.risk_free_rate*100, self.trading_days)
    
    # ========== 技术指标（向量化计算）==========
    
    def calculate_moving_averages(self, data: pd.DataFrame, 
                                  periods: List[int] = None) -> pd.DataFrame:
        """
        计算移动平均线 - 向量化计算
        """
        if periods is None:
            periods = self.config.analysis.ma_periods if self.config else [5, 10, 20, 30, 60]
        
        result = data.copy()
        
        # 向量化计算所有移动平均线
        for period in periods:
            col_name = f'MA_{period}'
            result[col_name] = result['close'].rolling(
                window=period, min_periods=1
            ).mean()
        
        logger.debug("已计算 %s 条移动平均线", len(periods))
        return result
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """计算RSI - 向量化优化版本"""
        period = period or (self.config.analysis.rsi_period if self.config else 14)
        result = data.copy()
        
        # 计算价格变化
        delta = result['close'].diff()
        
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 计算RSI
        rs = gain / loss
        result[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        return result
    
    def calculate_macd(self, data: pd.DataFrame, 
                       fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """计算MACD指标"""
        fast = fast or (self.config.analysis.macd_fast if self.config else 12)
        slow = slow or (self.config.analysis.macd_slow if self.config else 26)
        signal = signal or (self.config.analysis.macd_signal if self.config else 9)
        
        result = data.copy()
        
        # 计算EMA
        ema_fast = result['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD线
        result['MACD'] = ema_fast - ema_slow
        
        # 信号线
        result['MACD_Signal'] = result['MACD'].ewm(span=signal, adjust=False).mean()
        
        # 柱状图
        result['MACD_Hist'] = result['MACD'] - result['MACD_Signal']
        
        return result
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, 
                                  period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """计算布林带"""
        result = data.copy()
        
        # 中间线（移动平均）
        result['BB_Middle'] = result['close'].rolling(window=period).mean()
        
        # 标准差
        rolling_std = result['close'].rolling(window=period).std()
        
        # 上轨和下轨
        result['BB_Upper'] = result['BB_Middle'] + (rolling_std * std_dev)
        result['BB_Lower'] = result['BB_Middle'] - (rolling_std * std_dev)
        
        return result
    
    # ========== 专业风险评估模块 ==========
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """计算最大回撤"""
        if len(prices) < 2:
            return 0, None, None
        
        # 计算累积最大值
        cumulative_max = prices.cummax()
        
        # 计算回撤
        drawdown = (prices - cumulative_max) / cumulative_max
        
        # 找到最大回撤
        max_drawdown = drawdown.min()
        max_drawdown_date = drawdown.idxmin()
        
        # 找到对应的高点
        peak_before_drawdown = cumulative_max.loc[:max_drawdown_date].idxmax()
        
        return max_drawdown, peak_before_drawdown, max_drawdown_date
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = None) -> float:
        """计算夏普比率"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) < 2:
            return 0.0
        
        # 日无风险利率
        daily_rf = risk_free_rate / self.trading_days
        
        # 超额收益
        excess_returns = returns - daily_rf
        
        # 计算夏普比率
        if excess_returns.std() > 0:
            sharpe = np.sqrt(self.trading_days) * excess_returns.mean() / excess_returns.std()
            return sharpe
        else:
            return 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series,
                               risk_free_rate: float = None,
                               target_return: float = 0.0) -> float:
        """计算索提诺比率"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        if len(returns) < 2:
            return 0.0
        
        # 日无风险利率
        daily_rf = risk_free_rate / self.trading_days
        
        # 超额收益率
        excess_returns = returns - daily_rf
        
        # 只考虑低于目标收益率的回报（下行偏差）
        downside_returns = excess_returns[excess_returns < target_return]
        
        if len(downside_returns) > 0:
            # 下行偏差
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
            
            if downside_deviation > 0:
                sortino = np.sqrt(self.trading_days) * excess_returns.mean() / downside_deviation
                return sortino
        
        return 0.0
    
    def calculate_var(self, returns: pd.Series, 
                     confidence_level: float = 0.95) -> Dict[str, float]:
        """计算风险价值（VaR）"""
        if len(returns) < 10:
            return {'historical_var': 0}
        
        results = {}
        
        # 历史模拟法
        historical_var = np.percentile(returns, (1 - confidence_level) * 100)
        results['historical_var'] = historical_var
        
        # 条件VaR
        cvar = returns[returns <= historical_var].mean()
        results['cvar'] = cvar
        
        return results
    
    def analyze_returns_distribution(self, returns: pd.Series) -> Dict[str, Any]:
        """分析收益率分布特征"""
        if len(returns) < 10:
            return {}
        
        results = {
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'median': returns.median(),
            'min': returns.min(),
            'max': returns.max(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'total_days': len(returns),
            'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        }
        
        return results
    
    # ========== 风险报告生成 ==========
    
    def generate_risk_report(self, prices: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """生成风险评估报告"""
        if len(returns) < 20:
            logger.warning("数据不足，无法生成完整的风险报告")
            return {}
        
        report = {}
        
        # 1. 基本收益特征
        report['returns_analysis'] = self.analyze_returns_distribution(returns)
        
        # 2. 最大回撤分析
        max_drawdown, peak_date, trough_date = self.calculate_max_drawdown(prices)
        report['max_drawdown_analysis'] = {
            'max_drawdown': max_drawdown,
            'peak_date': peak_date,
            'trough_date': trough_date
        }
        
        # 3. 风险价值
        report['var_analysis'] = self.calculate_var(returns)
        
        # 4. 绩效比率
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        
        report['performance_ratios'] = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino
        }
        
        # 5. 风险评级
        report['risk_rating'] = self._calculate_risk_rating(report)
        
        # 6. 风险摘要
        report['risk_summary'] = self._generate_risk_summary(report)
        
        logger.info("综合风险评估报告生成完成")
        return report
    
    def _calculate_risk_rating(self, report: Dict[str, Any]) -> str:
        """计算风险评级"""
        try:
            max_dd = abs(report['max_drawdown_analysis']['max_drawdown'])
            volatility = report['returns_analysis'].get('std', 0) * np.sqrt(self.trading_days)
            sharpe = report['performance_ratios']['sharpe_ratio']
            
            # 综合评分
            score = 0
            
            # 最大回撤评分
            if max_dd < 0.1:  # <10%
                score += 3
            elif max_dd < 0.2:  # <20%
                score += 2
            elif max_dd < 0.3:  # <30%
                score += 1
            
            # 波动率评分
            if volatility < 0.2:  # <20%
                score += 3
            elif volatility < 0.3:  # <30%
                score += 2
            elif volatility < 0.4:  # <40%
                score += 1
            
            # 夏普比率评分
            if sharpe > 1.0:
                score += 3
            elif sharpe > 0.5:
                score += 2
            elif sharpe > 0:
                score += 1
            
            # 风险评级
            if score >= 8:
                return 'A-低风险'
            elif score >= 6:
                return 'B-中低风险'
            elif score >= 4:
                return 'C-中等风险'
            elif score >= 2:
                return 'D-中高风险'
            else:
                return 'E-高风险'
                
        except:
            return '未评级'
    
    def _generate_risk_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险摘要"""
        try:
            max_dd_info = report['max_drawdown_analysis']
            perf_ratios = report['performance_ratios']
            returns_info = report['returns_analysis']
            
            return {
                'annual_return': returns_info.get('mean', 0) * self.trading_days,
                'annual_volatility': returns_info.get('std', 0) * np.sqrt(self.trading_days),
                'max_drawdown': max_dd_info.get('max_drawdown', 0),
                'sharpe_ratio': perf_ratios.get('sharpe_ratio', 0),
                'sortino_ratio': perf_ratios.get('sortino_ratio', 0),
                'win_rate': returns_info.get('win_rate', 0),
                'var_95': report.get('var_analysis', {}).get('historical_var', 0),
                'risk_rating': report.get('risk_rating', '未评级')
            }
        except:
            return {}
    
    # ========== 技术指标计算 ==========
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        logger.info("开始计算技术指标...")
        
        # 复制数据避免修改原始数据
        result = data.copy()
        
        # 确保有收益率数据
        if 'returns' not in result.columns:
            result['returns'] = result['close'].pct_change()
        
        # 计算技术指标
        result = self.calculate_moving_averages(result)
        result = self.calculate_rsi(result)
        result = self.calculate_macd(result)
        
        # 计算布林带（如果数据足够）
        if len(result) >= 20:
            try:
                result = self.calculate_bollinger_bands(result)
            except Exception as e:
                logger.warning("布林带计算失败: %s", str(e))
        
        # 计算波动率
        if 'returns' in result.columns and len(result) >= 20:
            result['volatility_20d'] = result['returns'].rolling(window=20).std() * np.sqrt(self.trading_days)
            if len(result) >= 60:
                result['volatility_60d'] = result['returns'].rolling(window=60).std() * np.sqrt(self.trading_days)
        
        logger.info("技术指标计算完成，新增 %s 个指标列", len(result.columns) - len(data.columns))
        
        return result
    
    def print_risk_report(self, report: Dict[str, Any], stock_name: str = "股票"):
        """格式化打印风险报告"""
        print(f"\n{'='*60}")
        print(f"{stock_name} 风险评估报告")
        print(f"{'='*60}")
        
        # 打印摘要
        if 'risk_summary' in report:
            summary = report['risk_summary']
            print("\n风险摘要:")
            print(f"   年化收益率: {summary.get('annual_return', 0):.2%}")
            print(f"   年化波动率: {summary.get('annual_volatility', 0):.2%}")
            print(f"   最大回撤: {summary.get('max_drawdown', 0):.2%}")
            print(f"   夏普比率: {summary.get('sharpe_ratio', 0):.3f}")
            print(f"   索提诺比率: {summary.get('sortino_ratio', 0):.3f}")
            print(f"   胜率: {summary.get('win_rate', 0):.2%}")
            print(f"   风险评级: {summary.get('risk_rating', '未评级')}")
        
        # 打印最大回撤详情
        if 'max_drawdown_analysis' in report:
            mdd = report['max_drawdown_analysis']
            print(f"\n最大回撤分析:")
            print(f"   最大回撤: {mdd.get('max_drawdown', 0):.2%}")
            if mdd.get('peak_date'):
                print(f"   峰值日期: {mdd.get('peak_date')}")
            if mdd.get('trough_date'):
                print(f"   谷底日期: {mdd.get('trough_date')}")
        
        # 打印绩效比率
        if 'performance_ratios' in report:
            ratios = report['performance_ratios']
            print(f"\n绩效比率:")
            print(f"   夏普比率: {ratios.get('sharpe_ratio', 0):.3f}")
            print(f"   索提诺比率: {ratios.get('sortino_ratio', 0):.3f}")
            
            # 夏普比率解释
            sharpe = ratios.get('sharpe_ratio', 0)
            if sharpe > 1.0:
                print("   夏普比率 > 1.0: 优秀，每承担一单位风险获得超额回报")
            elif sharpe > 0.5:
                print("   夏普比率 > 0.5: 良好，风险调整后收益为正")
            elif sharpe > 0:
                print("   夏普比率 > 0: 尚可，但风险调整后收益较低")
            else:
                print("   夏普比率 ≤ 0: 不如持有无风险资产")
        
        print(f"{'='*60}\n")

StockAnalyzer = EnhancedStockAnalyzer

# ========== 可以直接运行的主函数 ==========
def analyze_single_stock():
    """分析单只股票的主函数"""
    import logging
    from pathlib import Path
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("股票分析工具 - 单只股票分析")
    print("=" * 60)
    
    try:
        # 尝试加载配置
        from config.config import get_config, setup_logging
        config = get_config()
        logger = setup_logging(config.logging)
        logger.info("配置加载成功")
    except:
        logger.info("使用默认配置")
        config = None
    
    # 创建分析器
    analyzer = EnhancedStockAnalyzer(config)
    
    # 检查数据目录
    data_dir = Path('data')
    cleaned_dir = data_dir / 'cleaned'
    
    if not cleaned_dir.exists():
        print(f"错误: 清理数据目录不存在: {cleaned_dir}")
        print("请先运行 data_cleaner.py")
        return
    
    # 获取可用的股票文件
    cleaned_files = list(cleaned_dir.glob("*_cleaned.csv"))
    if not cleaned_files:
        print(f"错误: 清理数据目录中没有找到数据文件")
        print(f"目录: {cleaned_dir}")
        print("请先运行 data_cleaner.py")
        return
    
    print("\n可用的股票数据文件:")
    for i, file in enumerate(cleaned_files, 1):
        print(f"  {i}. {file.name}")
    
    try:
        choice = int(input("\n请选择要分析的股票 (输入编号): ").strip())
        if 1 <= choice <= len(cleaned_files):
            selected_file = cleaned_files[choice - 1]
            ticker = selected_file.stem.replace('_cleaned', '')
            
            print(f"\n分析股票: {ticker}")
            print(f"数据文件: {selected_file.name}")
            
            # 加载数据
            try:
                df = pd.read_csv(selected_file, parse_dates=['date'], index_col='date')
                print(f"数据加载成功: {len(df)} 行")
                
                if df.empty:
                    print("错误: 数据为空")
                    return
                
                # 检查必要列
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    print(f"警告: 缺少列: {missing_cols}")
                
                # 计算技术指标
                print("\n计算技术指标...")
                df_with_indicators = analyzer.calculate_all_indicators(df)
                
                # 计算收益率和风险报告
                if 'close' in df_with_indicators.columns:
                    prices = df_with_indicators['close']
                    returns = prices.pct_change().dropna()
                    
                    if len(returns) >= 5:
                        # 生成风险报告
                        print("生成风险报告...")
                        risk_report = analyzer.generate_risk_report(prices, returns)
                        
                        # 打印报告
                        analyzer.print_risk_report(risk_report, ticker)
                        
                        # 保存分析结果
                        analysis_dir = Path('data/analysis')
                        analysis_dir.mkdir(exist_ok=True)
                        
                        # 保存带指标的数据
                        analysis_file = analysis_dir / f"{ticker}_analysis.csv"
                        df_with_indicators.to_csv(analysis_file, encoding='utf-8')
                        print(f"\n分析结果已保存到: {analysis_file}")
                        
                        # 保存风险报告为JSON
                        import json
                        report_file = analysis_dir / f"{ticker}_risk_report.json"
                        with open(report_file, 'w', encoding='utf-8') as f:
                            json.dump(risk_report, f, indent=2, ensure_ascii=False, default=str)
                        print(f"风险报告已保存到: {report_file}")
                        
                        # 显示基本统计信息
                        print(f"\n{ticker} 基本统计信息:")
                        print(f"  数据期间: {df.index.min().strftime('%Y-%m-%d')} 到 {df.index.max().strftime('%Y-%m-%d')}")
                        print(f"  交易天数: {len(df)}")
                        print(f"  收盘价范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
                        print(f"  平均成交量: {df['volume'].mean():,.0f}")
                        
                        # 显示技术指标
                        print(f"\n技术指标:")
                        indicator_cols = [col for col in df_with_indicators.columns if col not in required_cols]
                        for col in indicator_cols[:10]:  # 只显示前10个
                            if col in df_with_indicators.columns:
                                latest_value = df_with_indicators[col].iloc[-1]
                                print(f"  {col}: {latest_value:.4f}")
                        
                    else:
                        print("错误: 收益率数据不足，无法生成风险报告")
                else:
                    print("错误: 数据中缺少 'close' 列")
                
            except Exception as e:
                print(f"分析过程中出错: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("错误: 无效的选择")
    except ValueError:
        print("错误: 请输入有效的数字")
    except Exception as e:
        print(f"运行出错: {e}")

def analyze_all_stocks():
    """分析所有可用股票"""
    import logging
    from pathlib import Path
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("股票分析工具 - 批量分析所有股票")
    print("=" * 60)
    
    try:
        # 尝试加载配置
        from config.config import get_config, setup_logging
        config = get_config()
        logger = setup_logging(config.logging)
        logger.info("配置加载成功")
    except:
        logger.info("使用默认配置")
        config = None
    
    # 创建分析器
    analyzer = EnhancedStockAnalyzer(config)
    
    # 检查数据目录
    data_dir = Path('data')
    cleaned_dir = data_dir / 'cleaned'
    
    if not cleaned_dir.exists():
        print(f"错误: 清理数据目录不存在: {cleaned_dir}")
        print("请先运行 data_cleaner.py")
        return
    
    # 获取所有股票文件
    cleaned_files = list(cleaned_dir.glob("*_cleaned.csv"))
    if not cleaned_files:
        print(f"错误: 清理数据目录中没有找到数据文件")
        print("请先运行 data_cleaner.py")
        return
    
    print(f"找到 {len(cleaned_files)} 个股票数据文件")
    
    # 创建分析目录
    analysis_dir = Path('data/analysis')
    analysis_dir.mkdir(exist_ok=True)
    
    results = []
    
    for file in cleaned_files:
        ticker = file.stem.replace('_cleaned', '')
        print(f"\n分析 {ticker}...")
        
        try:
            # 加载数据
            df = pd.read_csv(file, parse_dates=['date'], index_col='date')
            
            if df.empty or len(df) < 10:
                print(f"  ✗ 跳过: 数据不足 ({len(df)} 行)")
                continue
            
            # 计算技术指标
            df_with_indicators = analyzer.calculate_all_indicators(df)
            
            # 计算收益率和风险报告
            if 'close' in df_with_indicators.columns:
                prices = df_with_indicators['close']
                returns = prices.pct_change().dropna()
                
                if len(returns) >= 5:
                    # 生成风险报告
                    risk_report = analyzer.generate_risk_report(prices, returns)
                    
                    # 保存分析结果
                    analysis_file = analysis_dir / f"{ticker}_analysis.csv"
                    df_with_indicators.to_csv(analysis_file, encoding='utf-8')
                    
                    # 保存风险报告
                    import json
                    report_file = analysis_dir / f"{ticker}_risk_report.json"
                    
                    with open(report_file, 'w', encoding='utf-8') as f:
                        json.dump(risk_report, f, indent=2, ensure_ascii=False, default=str)
                    
                    # 收集结果
                    if 'risk_summary' in risk_report:
                        summary = risk_report['risk_summary']
                        results.append({
                            'ticker': ticker,
                            'annual_return': summary.get('annual_return', 0),
                            'max_drawdown': summary.get('max_drawdown', 0),
                            'sharpe_ratio': summary.get('sharpe_ratio', 0),
                            'risk_rating': summary.get('risk_rating', '未评级')
                        })
                    
                    print(f"  ✓ 完成: {len(df)} 行数据")
                else:
                    print(f"  ✗ 跳过: 收益率数据不足")
            else:
                print(f"  ✗ 跳过: 缺少 'close' 列")
                
        except Exception as e:
            print(f"  ✗ 错误: {str(e)[:50]}")
    
    # 显示总结报告
    if results:
        print("\n" + "=" * 60)
        print("批量分析总结")
        print("=" * 60)
        print(f"成功分析: {len(results)}/{len(cleaned_files)} 只股票")
        
        print(f"\n{'股票':<10} {'年化收益':>12} {'最大回撤':>12} {'夏普比率':>12} {'风险评级':>12}")
        print("-" * 60)
        
        for result in results:
            print(f"{result['ticker']:<10} {result['annual_return']:>12.2%} {result['max_drawdown']:>12.2%} {result['sharpe_ratio']:>12.3f} {result['risk_rating']:>12}")
        
        # 保存总结报告
        summary_file = analysis_dir / 'analysis_summary.csv'
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(summary_file, index=False, encoding='utf-8')
        print(f"\n总结报告已保存到: {summary_file}")
        
        # 找出最佳和最差的股票
        if len(results) > 1:
            best_return = max(results, key=lambda x: x['annual_return'])
            best_sharpe = max(results, key=lambda x: x['sharpe_ratio'])
            worst_drawdown = min(results, key=lambda x: x['max_drawdown'])
            
            print(f"\n最佳年化收益: {best_return['ticker']} ({best_return['annual_return']:.2%})")
            print(f"最佳夏普比率: {best_sharpe['ticker']} ({best_sharpe['sharpe_ratio']:.3f})")
            print(f"最低最大回撤: {worst_drawdown['ticker']} ({worst_drawdown['max_drawdown']:.2%})")
    
    print(f"\n所有分析结果保存在: {analysis_dir}")

def main():
    """主函数 - 股票分析器"""
    print("=" * 60)
    print("股票分析器主菜单")
    print("=" * 60)
    print("1. 分析单只股票")
    print("2. 批量分析所有股票")
    print("3. 查看数据目录")
    print("4. 退出")
    
    try:
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            analyze_single_stock()
        elif choice == '2':
            analyze_all_stocks()
        elif choice == '3':
            from pathlib import Path
            
            data_dir = Path('data')
            cleaned_dir = data_dir / 'cleaned'
            analysis_dir = data_dir / 'analysis'
            
            print(f"\n数据目录结构:")
            print(f"  原始数据: {data_dir / 'raw'}")
            print(f"  清理数据: {cleaned_dir}")
            print(f"  分析结果: {analysis_dir}")
            
            if cleaned_dir.exists():
                cleaned_files = list(cleaned_dir.glob("*.csv"))
                print(f"\n清理数据文件 ({len(cleaned_files)} 个):")
                for file in cleaned_files[:10]:  # 只显示前10个
                    print(f"  - {file.name}")
                if len(cleaned_files) > 10:
                    print(f"  ... 还有 {len(cleaned_files) - 10} 个文件")
            
            if analysis_dir.exists():
                analysis_files = list(analysis_dir.glob("*.csv"))
                json_files = list(analysis_dir.glob("*.json"))
                print(f"\n分析结果文件:")
                print(f"  CSV文件: {len(analysis_files)} 个")
                print(f"  JSON报告: {len(json_files)} 个")
        elif choice == '4':
            print("退出程序")
        else:
            print("无效选择，请重新运行")
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"运行出错: {e}")

if __name__ == "__main__":
    main()