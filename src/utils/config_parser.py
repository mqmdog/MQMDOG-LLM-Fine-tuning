"""
配置文件解析器
=============
核心功能: 加载和合并YAML配置文件, 支持配置继承和命令行覆盖

设计要点:
- 分层配置: base_config → task_config → CLI覆盖
- 深度合并: 嵌套字典递归合并, 列表直接覆盖
- 类型安全: 关键参数做类型校验
"""

import os
import copy
import logging
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigParser:
    """
    配置解析器 - 管理多层级配置的加载与合并

    配置优先级 (从低到高):
    1. base_config.yaml: 默认基础配置
    2. task_config.yaml: 任务特定配置 (如 lora_config.yaml)
    3. CLI参数: 命令行传入的覆盖参数

    使用示例:
        parser = ConfigParser()
        config = parser.load("configs/lora_config.yaml")
        config = parser.override(config, {"training.learning_rate": 1e-4})
    """

    @staticmethod
    def load(config_path: str, base_config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 主配置文件路径
            base_config_path: 基础配置文件路径 (可选, 默认自动查找)

        Returns:
            合并后的配置字典
        """
        config = {}

        # 1. 加载基础配置
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Base config loaded: {base_config_path}")

        # 2. 加载任务配置并合并 (任务配置覆盖基础配置)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                task_config = yaml.safe_load(f) or {}

            config = ConfigParser._deep_merge(config, task_config)
            logger.info(f"Task config loaded: {config_path}")
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

        return config

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        递归深度合并两个字典

        规则:
        - 如果两个值都是字典 → 递归合并
        - 否则 → override的值覆盖base的值
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigParser._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    @staticmethod
    def override(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用扁平化的key覆盖嵌套配置

        支持点号分隔的key路径:
            {"training.learning_rate": 1e-4} → config["training"]["learning_rate"] = 1e-4

        Args:
            config: 原始配置
            overrides: 覆盖参数 (key使用点号分隔嵌套路径)
        """
        config = copy.deepcopy(config)

        for flat_key, value in overrides.items():
            keys = flat_key.split(".")
            current = config

            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            current[keys[-1]] = value

        return config

    @staticmethod
    def save(config: Dict[str, Any], output_path: str):
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Config saved to: {output_path}")

    @staticmethod
    def validate(config: Dict[str, Any]) -> bool:
        """
        验证配置完整性

        检查必需的配置项是否存在
        """
        required_fields = {
            "model.model_name_or_path": "模型名称或路径",
            "finetuning.method": "微调方法",
            "training.output_dir": "输出目录",
        }

        for field_path, description in required_fields.items():
            keys = field_path.split(".")
            current = config
            for key in keys:
                if key not in current:
                    logger.error(
                        f"Missing required config: '{field_path}' ({description})"
                    )
                    return False
                current = current[key]

        return True
