"""
推理模块初始化
使用延迟导入避免在 torch 未安装时阻塞
"""


def __getattr__(name):
    if name == "TextGenerator":
        from src.inference.generator import TextGenerator
        return TextGenerator
    raise AttributeError(f"module 'src.inference' has no attribute {name!r}")


__all__ = ["TextGenerator"]
