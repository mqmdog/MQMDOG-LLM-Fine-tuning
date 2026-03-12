"""
Data Collator - 动态批处理与填充
================================
核心功能: 在DataLoader中将不同长度的样本填充为统一长度, 组成batch

技术要点:
1. 动态Padding: 每个batch只填充到该batch的最大长度, 而非全局最大长度
   - 优势: 减少无效计算, 提升训练效率 (特别是序列长度差异大时)
   - 对比: 静态Padding填充到固定max_length, 简单但浪费计算
2. Labels处理: 填充位置的label设为-100, 确保不参与loss计算

- 为什么选择动态padding? -> 减少padding token的计算开销
- 为什么labels的padding是-100? -> PyTorch CrossEntropyLoss的ignore_index默认值
- Batch大小对训练的影响? -> 影响梯度估计的方差和训练稳定性
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForSFT:
    """
    SFT训练的动态Data Collator

    功能:
    - 将batch中的样本动态填充到相同长度
    - 正确处理input_ids, attention_mask, labels的padding
    - 支持左填充 (left padding) 和右填充 (right padding)

    左填充 vs 右填充:
    - 训练时: 通常使用右填充 (padding在序列末尾)
    - 生成时: 通常使用左填充 (padding在序列开头, 便于batch生成)
    """
    tokenizer: PreTrainedTokenizer
    padding: str = "longest"  # "longest" 或 "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None  # 填充到指定倍数 (对Tensor Core友好)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将一个batch的样本处理为模型输入张量

        Args:
            features: 样本列表, 每个样本是包含input_ids等的字典

        Returns:
            包含 input_ids, attention_mask, labels 的张量字典
        """
        # 确定填充长度
        if self.padding == "max_length" and self.max_length:
            max_len = self.max_length
        else:
            max_len = max(len(f["input_ids"]) for f in features)

        # 如果需要填充到指定倍数 (如8的倍数, 对GPU计算更友好)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_token_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature.get(
                "attention_mask", [1] * len(input_ids))
            labels = feature.get("labels", input_ids.copy())

            # 计算需要填充的长度
            padding_length = max_len - len(input_ids)

            # 右填充 (训练模式)
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # padding位置不计算loss

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


@dataclass
class DataCollatorForDPO:
    """
    DPO训练的Data Collator

    DPO需要同时处理chosen和rejected两组数据:
    - chosen_input_ids / rejected_input_ids: 分别填充和对齐
    - 两组数据的attention_mask需要独立生成

    
    - DPO的loss函数: L = -log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))
    - β控制偏好强度, 较大的β使模型更严格地遵循偏好
    """
    tokenizer: PreTrainedTokenizer
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 分别处理chosen和rejected
        chosen_max = max(len(f["chosen_input_ids"]) for f in features)
        rejected_max = max(len(f["rejected_input_ids"]) for f in features)

        if self.max_length:
            chosen_max = min(chosen_max, self.max_length)
            rejected_max = min(rejected_max, self.max_length)

        pad_token_id = self.tokenizer.pad_token_id
        result = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
        }

        for feature in features:
            for prefix, max_len in [("chosen", chosen_max), ("rejected", rejected_max)]:
                ids = feature[f"{prefix}_input_ids"][:max_len]
                padding_length = max_len - len(ids)

                result[f"{prefix}_input_ids"].append(
                    ids + [pad_token_id] * padding_length
                )
                result[f"{prefix}_attention_mask"].append(
                    [1] * len(ids) + [0] * padding_length
                )

        return {
            k: torch.tensor(v, dtype=torch.long) for k, v in result.items()
        }
