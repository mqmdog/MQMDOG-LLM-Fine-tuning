"""
工具模块初始化
使用延迟导入避免在 torch 未安装时阻塞非 torch 依赖的模块
"""


def __getattr__(name):
    if name == "ConfigParser":
        from src.utils.config_parser import ConfigParser
        return ConfigParser
    elif name == "setup_logger":
        from src.utils.logger import setup_logger
        return setup_logger
    elif name == "set_seed":
        from src.utils.common import set_seed
        return set_seed
    elif name == "print_trainable_parameters":
        from src.utils.common import print_trainable_parameters
        return print_trainable_parameters
    elif name == "get_device_info":
        from src.utils.common import get_device_info
        return get_device_info
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")


__all__ = [
    "ConfigParser",
    "setup_logger",
    "set_seed",
    "print_trainable_parameters",
    "get_device_info",
]
