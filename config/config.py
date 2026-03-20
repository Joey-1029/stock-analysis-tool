"""
配置管理模块
"""
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

@dataclass
class DataConfig:
    """数据配置"""
    a_stocks: List[str] = field(default_factory=lambda: ['000001', '600519'])
    us_stocks: List[str] = field(default_factory=lambda: ['AAPL', 'MSFT'])
    hk_stocks: List[str] = field(default_factory=lambda: ['00700', '09988'])
    indices: List[str] = field(default_factory=lambda: ['sh000001', 'sz399001'])
    
    start_date: str = '20230101'
    end_date: Optional[str] = None
    
    adjust_type: str = 'qfq'  # 前复权
    request_delay: float = 1.0
    
    max_retries: int = 3
    retry_delay: float = 2.0
    backoff_factor: float = 1.5

@dataclass
class AnalysisConfig:
    """分析配置"""
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 60])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    
    # 风险参数
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252
    var_confidence: float = 0.95
    
    # 回测参数
    initial_capital: float = 100000
    commission_rate: float = 0.0003

@dataclass
class LoggingConfig:
    """日志配置"""
    log_level: str = 'INFO'
    log_dir: str = 'logs'
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_retention_days: int = 7

@dataclass
class PathConfig:
    """路径配置"""
    data_dir: str = 'data'
    raw_dir: str = 'data/raw'
    cleaned_dir: str = 'data/cleaned'
    analysis_dir: str = 'data/analysis'
    reports_dir: str = 'data/reports'

@dataclass
class ReportConfig:
    """报告配置"""
    format: str = 'html'  # html, pdf, markdown
    include_charts: bool = True
    include_statistics: bool = True
    include_recommendations: bool = True

@dataclass
class Config:
    """主配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str = "config/settings.yaml"):
        """从YAML文件加载配置"""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            # 如果配置文件不存在，创建默认配置
            default_config = cls()
            default_config.save_to_yaml(yaml_path)
            logging.info(f"配置文件不存在，已创建默认配置: {yaml_path}")
            return default_config
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # 递归创建配置对象
            return cls(
                data=DataConfig(**yaml_data.get('data', {})),
                analysis=AnalysisConfig(**yaml_data.get('analysis', {})),
                logging=LoggingConfig(**yaml_data.get('logging', {})),
                paths=PathConfig(**yaml_data.get('paths', {})),
                report=ReportConfig(**yaml_data.get('report', {}))
            )
            
        except Exception as e:
            logging.error(f"加载配置文件失败: {e}")
            return cls()
    
    def save_to_yaml(self, yaml_path: str = "config/settings.yaml"):
        """保存配置到YAML文件"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(exist_ok=True)
        
        config_dict = {
            'data': asdict(self.data),
            'analysis': asdict(self.analysis),
            'logging': asdict(self.logging),
            'paths': asdict(self.paths),
            'report': asdict(self.report)
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def setup_directories(self):
        """创建所有必要的目录"""
        paths = [
            self.paths.data_dir,
            self.paths.raw_dir,
            self.paths.cleaned_dir,
            self.paths.analysis_dir,
            self.paths.reports_dir,
            self.logging.log_dir,
            'config'
        ]
        
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)

def setup_logging(config: LoggingConfig):
    """设置日志系统 - 修复Windows编码问题"""
    # 创建日志目录
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.log_level))
    
    # 清除现有的handlers（避免重复）
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件handler（使用UTF-8编码）
    log_file = log_dir / f'app_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, config.log_level))
    
    # 控制台handler（修复Windows编码问题）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.log_level))
    
    # 创建formatter
    formatter = logging.Formatter(config.log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logger
# 全局配置实例
_config_instance = None

def get_config(config_path: str = "config/settings.yaml") -> Config:
    """获取配置实例（单例模式）"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config.from_yaml(config_path)
        _config_instance.setup_directories()
    
    return _config_instance