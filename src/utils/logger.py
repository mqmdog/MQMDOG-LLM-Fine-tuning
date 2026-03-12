"""
日志管理器
=========
核心功能: 统一的日志配置, 支持控制台和文件输出

特性:
- 彩色控制台输出 (区分不同级别)
- 文件日志轮转 (避免日志文件过大)
- 训练指标专用日志 (分离训练日志和系统日志)
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "llm_finetune",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    配置并返回logger实例

    Args:
        name: logger名称
        log_dir: 日志文件目录
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 日志文件名 (默认自动按时间生成)

    Returns:
        配置好的logger
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler (如果指定了日志目录)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"train_{timestamp}.log"

        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "llm_finetune") -> logging.Logger:
    """获取已配置的logger (如果未配置则使用默认设置)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
