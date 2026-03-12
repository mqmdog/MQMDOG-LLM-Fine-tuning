"""
数据处理模块初始化
使用延迟导入避免在 torch 未安装时阻塞非 torch 依赖的模块
"""


def __getattr__(name):
    if name == "DataManager":
        from src.data.data_loader import DataManager
        return DataManager
    elif name in ("TemplateManager", "TEMPLATES"):
        from src.data.data_template import TemplateManager, TEMPLATES
        return TemplateManager if name == "TemplateManager" else TEMPLATES
    elif name == "DataCollatorForSFT":
        from src.data.data_collator import DataCollatorForSFT
        return DataCollatorForSFT
    elif name == "DataCollatorForDPO":
        from src.data.data_collator import DataCollatorForDPO
        return DataCollatorForDPO
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")


__all__ = [
    "DataManager",
    "TemplateManager",
    "TEMPLATES",
    "DataCollatorForSFT",
    "DataCollatorForDPO",
]
