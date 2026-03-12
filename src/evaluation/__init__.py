"""
评估模块初始化
使用延迟导入避免在 torch 未安装时阻塞
"""


def __getattr__(name):
    if name == "Evaluator":
        from src.evaluation.evaluator import Evaluator
        return Evaluator
    elif name == "MetricsCalculator":
        from src.evaluation.metrics import MetricsCalculator
        return MetricsCalculator
    raise AttributeError(f"module 'src.evaluation' has no attribute {name!r}")


__all__ = ["Evaluator", "MetricsCalculator"]
