"""
训练引擎模块初始化
使用延迟导入避免在 torch 未安装时阻塞
"""


def __getattr__(name):
    if name == "FineTuneTrainer":
        from src.training.trainer import FineTuneTrainer
        return FineTuneTrainer
    elif name == "DPOFineTuneTrainer":
        from src.training.dpo_trainer import DPOFineTuneTrainer
        return DPOFineTuneTrainer
    raise AttributeError(f"module 'src.training' has no attribute {name!r}")


__all__ = ["FineTuneTrainer", "DPOFineTuneTrainer"]
