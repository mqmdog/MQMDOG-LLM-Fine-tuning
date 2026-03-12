"""
模型模块初始化
使用延迟导入避免在 torch 未安装时阻塞
"""


def __getattr__(name):
    if name == "ModelLoader":
        from src.models.model_loader import ModelLoader
        return ModelLoader
    elif name == "PeftConfigFactory":
        from src.models.peft_config import PeftConfigFactory
        return PeftConfigFactory
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")


__all__ = [
    "ModelLoader",
    "PeftConfigFactory",
]
