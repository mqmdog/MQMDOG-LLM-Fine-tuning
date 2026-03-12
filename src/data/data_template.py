"""
数据模板管理器
============
核心功能: 将不同格式的原始数据统一转换为模型可接受的输入格式

支持的数据模板:
1. Alpaca格式: instruction + input + output (Stanford Alpaca风格)
2. ShareGPT格式: multi-turn conversations (多轮对话)
3. DPO格式: prompt + chosen + rejected (偏好数据)
4. Custom格式: 用户自定义字段映射


- 数据模板的设计直接影响模型学习效果
- SFT通常只对response部分计算loss (通过label masking实现)
- 不同Chat模型有不同的特殊token格式 (如ChatML, Llama-chat等)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Template:
    """
    数据模板基类
    定义了从原始数据到模型输入的映射规则

    属性说明:
    - name: 模板名称
    - system_prompt: 系统提示 (可选, 用于设定模型角色和行为)
    - instruction_key: 原始数据中指令字段的键名
    - input_key: 原始数据中补充输入字段的键名
    - output_key: 原始数据中期望输出字段的键名
    """
    name: str
    system_prompt: str = ""
    instruction_key: str = "instruction"
    input_key: str = "input"
    output_key: str = "output"
    history_key: str = "history"

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        将单个样本格式化为 prompt + response 的形式

        这是模板的核心方法: 将原始数据中的各个字段组合成模型可理解的文本

        Args:
            example: 原始数据样本 (字典格式)

        Returns:
            包含 "prompt" 和 "response" 的字典
        """
        raise NotImplementedError


@dataclass
class AlpacaTemplate(Template):
    """
    Alpaca数据格式模板

    数据格式示例:
    {
        "instruction": "给出以下问题的答案",
        "input": "什么是机器学习?",
        "output": "机器学习是人工智能的一个分支..."
    }

    当存在input时, 将instruction和input拼接作为prompt;
    当不存在input时, 仅使用instruction作为prompt.
    """
    name: str = "alpaca"
    system_prompt: str = "You are a helpful assistant."

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        instruction = example.get(self.instruction_key, "")
        input_text = example.get(self.input_key, "")
        output_text = example.get(self.output_key, "")

        # 构建prompt: 根据是否有补充输入决定格式
        if input_text:
            prompt = (
                f"Below is an instruction that describes a task, "
                f"paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        return {"prompt": prompt, "response": output_text}


@dataclass
class ShareGPTTemplate(Template):
    """
    ShareGPT多轮对话格式模板

    数据格式示例:
    {
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好! 有什么可以帮你的吗?"},
            {"from": "human", "value": "请介绍一下深度学习"},
            {"from": "gpt", "value": "深度学习是..."}
        ]
    }


    - 多轮对话训练时, 通常只对assistant的回复计算loss
    - 需要正确处理对话历史的截断和拼接
    """
    name: str = "sharegpt"
    conversations_key: str = "conversations"
    role_key: str = "from"
    content_key: str = "value"
    user_role: str = "human"
    assistant_role: str = "gpt"

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        conversations = example.get(self.conversations_key, [])
        if not conversations:
            return {"prompt": "", "response": ""}

        # 提取最后一轮的assistant回复作为response
        # 其余部分作为prompt (包含历史对话)
        prompt_parts = []
        response = ""

        for i, conv in enumerate(conversations):
            role = conv.get(self.role_key, "")
            content = conv.get(self.content_key, "")

            if i == len(conversations) - 1 and role == self.assistant_role:
                response = content
            else:
                role_label = "User" if role == self.user_role else "Assistant"
                prompt_parts.append(f"{role_label}: {content}")

        prompt = "\n".join(prompt_parts)
        if prompt_parts:
            prompt += "\nAssistant: "

        return {"prompt": prompt, "response": response}


@dataclass
class DPOTemplate(Template):
    """
    DPO偏好数据格式模板

    数据格式示例:
    {
        "prompt": "请解释量子计算",
        "chosen": "量子计算利用量子力学原理...(高质量回答)",
        "rejected": "量子就是很小的东西...(低质量回答)"
    }


    - DPO通过对比chosen和rejected学习人类偏好
    - 不需要显式的奖励模型, 隐式地将策略优化和奖励建模统一
    """
    name: str = "dpo"
    prompt_key: str = "prompt"
    chosen_key: str = "chosen"
    rejected_key: str = "rejected"

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prompt": example.get(self.prompt_key, ""),
            "chosen": example.get(self.chosen_key, ""),
            "rejected": example.get(self.rejected_key, ""),
        }


@dataclass
class CustomTemplate(Template):
    """
    自定义数据格式模板
    允许用户通过配置文件指定字段映射
    """
    name: str = "custom"
    prompt_template: str = "{instruction}"
    field_mapping: Dict[str, str] = field(default_factory=dict)

    def format_example(self, example: Dict[str, Any]) -> Dict[str, str]:
        # 通过字段映射将原始数据映射到标准格式
        mapped = {}
        for target_key, source_key in self.field_mapping.items():
            mapped[target_key] = example.get(source_key, "")

        prompt = self.prompt_template.format(**{**example, **mapped})
        response = mapped.get("output", example.get(self.output_key, ""))

        return {"prompt": prompt, "response": response}


# ============================================================================
# 预定义模板注册表
# ============================================================================
TEMPLATES: Dict[str, Template] = {
    "alpaca": AlpacaTemplate(),
    "sharegpt": ShareGPTTemplate(),
    "dpo": DPOTemplate(),
    "custom": CustomTemplate(),
}


class TemplateManager:
    """
    模板管理器 - 负责模板的注册、获取和应用

    使用示例:
        manager = TemplateManager()
        template = manager.get_template("alpaca")
        formatted = template.format_example(raw_data)
    """

    def __init__(self):
        self.templates = dict(TEMPLATES)

    def get_template(self, name: str) -> Template:
        """获取指定名称的模板"""
        if name not in self.templates:
            available = ", ".join(self.templates.keys())
            raise ValueError(
                f"Unknown template '{name}'. Available: {available}"
            )
        return self.templates[name]

    def register_template(self, name: str, template: Template):
        """注册自定义模板"""
        self.templates[name] = template

    def list_templates(self) -> List[str]:
        """列出所有可用模板"""
        return list(self.templates.keys())
